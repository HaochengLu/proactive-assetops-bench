"""Ten AssetOpsBench built-in tools across 4 domains + WorldState + custom_tools + approval_policy."""

from __future__ import annotations

import bisect
import math
from datetime import datetime
from typing import Any

from framework import VALID_MODALITIES, get_asset_map, parse_iso_utc, recursive_subset_match

BUILTIN_TOOL_DECISIONS = {
    "query_iot_metadata": "query",
    "query_iot_history": "query",
    "query_workorder_history": "query",
    "check_tsfm_anomaly": "investigate",
    "forecast_tsfm_metric": "investigate",
    "lookup_fmea_manual": "investigate",
    "submit_approval_request": "request_approval",
    "update_workorder": "execute",
    "create_maintenance_workorder": "execute",
    "create_urgent_shutdown_workorder": "execute",
}
BUILTIN_TOOLS = set(BUILTIN_TOOL_DECISIONS.keys())
FMEA_TEXT_FIELDS = (
    "fault_type", "failure_mode", "symptoms", "cause", "effects",
    "remediation", "description", "symptom_metric", "recommended_tool",
)


def _result(executed: bool, authorized: bool, status: str, output: Any) -> dict[str, Any]:
    return {"executed": executed, "authorized": authorized, "status": status, "output": output}


def _is_num(val: Any) -> bool:
    return isinstance(val, (int, float)) and not isinstance(val, bool)


class ToolExecutor:
    """Executes built-in AssetOpsBench tools and declarative custom_tools with WorldState."""

    def __init__(self, scenario: dict[str, Any]) -> None:
        self.scenario = scenario
        self.asset_map = get_asset_map(scenario)
        perms = scenario.get("permissions", {})
        self.allowed_decisions = set(perms.get("allowed_decisions", []))
        self.approval_required_tools = set(perms.get("approval_required_tools", []))
        self.forbidden_tools = set(perms.get("forbidden_tools", []))
        self.approval_policy: dict[str, dict[str, Any]] = perms.get("approval_policy") or {}
        self.approved_from_turn: dict[tuple[str, str], int] = {}

        self.custom_tools: dict[str, dict[str, Any]] = {
            t["name"]: t for t in (scenario.get("custom_tools") or []) if isinstance(t, dict) and t.get("name")
        }
        self.world_state: dict[str, Any] = {
            "assets": {
                aid: {"status": meta.get("initial_status", "running"), **(meta.get("initial_state") or {})}
                for aid, meta in self.asset_map.items()
            },
            "workorders": [],
            "approvals": {},
        }

        indexed = sorted(
            enumerate(scenario.get("observations", [])),
            key=lambda item: (parse_iso_utc(item[1]["release_time"]), item[0]),
        )
        self._obs_sorted = [item[1] for item in indexed]
        self._obs_dts: list[datetime] = [parse_iso_utc(o["release_time"]) for o in self._obs_sorted]

    def _released_slice(self, replay_time: str) -> list[dict[str, Any]]:
        return self._obs_sorted[: bisect.bisect_right(self._obs_dts, parse_iso_utc(replay_time))]

    def is_terminal_tool(self, name: str) -> bool:
        if name in self.custom_tools:
            return bool(self.custom_tools[name].get("terminal", True))
        return True

    def execute(
        self, action: dict[str, Any], replay_time: str, turn_idx: int, step_idx: int = 0
    ) -> dict[str, Any]:
        """Validate permissions, check preconditions, and run at most one synchronous tool call."""
        decision = action.get("decision")
        if decision not in self.allowed_decisions:
            return _result(False, False, "unauthorized", {"reason": f"Decision '{decision}' not allowed"})

        tool_call = action.get("tool_call")
        if tool_call is None:
            return _result(False, True, "skipped", None)
        if not isinstance(tool_call, dict) or not isinstance(tool_call.get("name"), str):
            return _result(False, True, "tool_error", {"reason": "Malformed tool_call object"})

        name, args = tool_call["name"], tool_call.get("args") or {}
        if not isinstance(args, dict):
            return _result(False, True, "tool_error", {"reason": "tool_call.args must be an object"})
        if name in self.forbidden_tools:
            return _result(False, False, "unauthorized", {"reason": f"Tool '{name}' is forbidden"})

        asset_id = str(args.get("asset_id") or self.scenario["asset_info"]["asset_id"])
        if name in self.approval_required_tools:
            eff_turn = self.approved_from_turn.get((asset_id, name))
            if eff_turn is None or turn_idx < eff_turn:
                return _result(False, False, "unauthorized", {"reason": f"Tool '{name}' not approved for '{asset_id}' at turn {turn_idx}"})

        if name in self.custom_tools:
            req_decision = str(self.custom_tools[name]["decision"])
        elif name in BUILTIN_TOOLS:
            req_decision = BUILTIN_TOOL_DECISIONS[name]
        else:
            return _result(False, True, "tool_error", {"reason": f"Unknown tool '{name}'"})

        if req_decision not in self.allowed_decisions:
            return _result(False, False, "unauthorized", {"reason": f"Decision '{req_decision}' required by '{name}' is not allowed"})
        if decision != req_decision:
            return _result(False, True, "tool_error", {"reason": f"Tool '{name}' requires decision '{req_decision}', got '{decision}'"})

        try:
            output = (
                self._exec_custom(name, args)
                if name in self.custom_tools
                else self._dispatch_builtin(name, args, replay_time, turn_idx, step_idx)
            )
        except Exception as exc:
            return _result(False, True, "tool_error", {"reason": str(exc)})
        if name == "submit_approval_request" and not output.get("approved", False):
            return _result(False, False, "unauthorized", {"reason": output.get("reason", "Approval denied")})
        return _result(True, True, "ok", output)

    def _exec_custom(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        spec = self.custom_tools[name]
        for req in spec.get("required_args") or []:
            if req not in args or args[req] is None:
                raise ValueError(f"Missing required argument '{req}' for custom tool '{name}'")
        target_asset = str(args.get("asset_id") or self.scenario["asset_info"]["asset_id"])
        effects = spec.get("state_effects") or {}
        applied: dict[str, Any] = {}
        for k, v in effects.items():
            if isinstance(v, str) and v.startswith("$args."):
                arg_key = v[len("$args.") :]
                applied[k] = args.get(arg_key, v)
            else:
                applied[k] = v
        if applied:
            self.world_state["assets"].setdefault(target_asset, {}).update(applied)
        return {"tool": name, "asset_id": target_asset, "applied_effects": applied, **(spec.get("static_output") or {})}

    def _numeric_series(self, asset_id: str, metric: str, replay_time: str, limit: int) -> list[tuple[datetime, float]]:
        pts: list[tuple[datetime, float]] = []
        for o in self._released_slice(replay_time):
            if o.get("asset_id") == asset_id and o.get("modality") == "telemetry":
                val = (o.get("payload") or {}).get(metric)
                if _is_num(val):
                    pts.append((parse_iso_utc(o["release_time"]), float(val)))
        return pts[-limit:]

    def _merged_workorders(self, replay_time: str) -> list[dict[str, Any]]:
        cutoff = parse_iso_utc(replay_time)
        by_id: dict[str, dict[str, Any]] = {}
        for w in self.scenario.get("tool_catalog_data", {}).get("historical_workorders", []):
            if not w.get("created_at") or parse_iso_utc(w["created_at"]) <= cutoff:
                by_id[w["workorder_id"]] = dict(w)
        for w in self.world_state["workorders"]:
            by_id[w["workorder_id"]] = dict(w)
        return sorted(by_id.values(), key=lambda w: w.get("created_at", ""))

    def _dispatch_builtin(
        self, name: str, args: dict[str, Any], replay_time: str, turn_idx: int, step_idx: int
    ) -> dict[str, Any]:
        default_asset = self.scenario["asset_info"]["asset_id"]

        if name == "query_iot_metadata":
            req_aid = args.get("asset_id")
            if req_aid is not None and req_aid not in self.asset_map:
                raise ValueError(f"Unknown asset_id '{req_aid}'")
            aids = [req_aid] if req_aid is not None else list(self.asset_map.keys())
            out_assets = []
            for aid in aids:
                meta = self.asset_map[aid]
                ranges = meta.get("normal_ranges") or {}
                out_assets.append({
                    "site_id": meta.get("site_id", self.scenario["asset_info"].get("site_id", "")),
                    "asset_id": aid,
                    "asset_type": meta.get("asset_type", ""),
                    "sensors": meta.get("sensors") or sorted(ranges.keys()),
                    "sensor_units": meta.get("sensor_units") or {},
                    "normal_ranges": ranges,
                    "world_state": dict(self.world_state["assets"].get(aid, {})),
                })
            return {"site_id": self.scenario["asset_info"].get("site_id", ""), "count": len(out_assets), "assets": out_assets}

        if name == "query_iot_history":
            asset_id = str(args.get("asset_id") or default_asset)
            modality = str(args.get("modality", "all"))
            aggregation = str(args.get("aggregation", "raw"))
            metrics_filter = args.get("metrics")
            start_time = args.get("start_time")
            if modality != "all" and modality not in VALID_MODALITIES:
                raise ValueError(f"Invalid modality '{modality}'")
            if aggregation not in {"raw", "mean", "min", "max", "latest"}:
                raise ValueError(f"Invalid aggregation '{aggregation}'")

            matched = [
                o for o in self._released_slice(replay_time)
                if o.get("asset_id") == asset_id and (modality == "all" or o.get("modality") == modality)
            ]
            if start_time:
                start_dt = parse_iso_utc(str(start_time))
                matched = [o for o in matched if parse_iso_utc(o["release_time"]) >= start_dt]

            def _project(obs: dict[str, Any]) -> dict[str, Any]:
                if not metrics_filter:
                    return obs
                m_set = set(metrics_filter)
                return {**obs, "payload": {k: v for k, v in (obs.get("payload") or {}).items() if k in m_set}}

            if aggregation == "raw":
                limit = max(int(args.get("limit", 5)), 0)
                records = [_project(o) for o in (matched[-limit:] if limit else [])]
                return {"asset_id": asset_id, "modality": modality, "aggregation": "raw", "count": len(records), "records": records}

            if start_time is not None and "limit" not in args:
                window_obs = matched
            else:
                limit = max(int(args.get("limit", 5)), 0)
                window_obs = matched[-limit:] if limit else []

            series_by_metric: dict[str, list[float]] = {}
            for o in window_obs:
                for k, v in (o.get("payload") or {}).items():
                    if metrics_filter and k not in metrics_filter:
                        continue
                    if _is_num(v):
                        series_by_metric.setdefault(k, []).append(float(v))
            agg_out: dict[str, float] = {}
            for k, vals in series_by_metric.items():
                if aggregation == "mean":
                    agg_out[k] = round(sum(vals) / len(vals), 4)
                elif aggregation == "min":
                    agg_out[k] = round(min(vals), 4)
                elif aggregation == "max":
                    agg_out[k] = round(max(vals), 4)
                else:
                    agg_out[k] = round(vals[-1], 4)
            return {"asset_id": asset_id, "modality": modality, "aggregation": aggregation, "sample_count": len(window_obs), "metrics": agg_out}

        if name == "check_tsfm_anomaly":
            asset_id, metric = str(args.get("asset_id") or default_asset), str(args["metric"])
            w = max(int(args.get("window", 8)), 1)
            bounds = (self.asset_map.get(asset_id, self.scenario["asset_info"]).get("normal_ranges") or {}).get(metric)
            pts = self._numeric_series(asset_id, metric, replay_time, w)
            m = len(pts)
            if m == 0:
                return {
                    "asset_id": asset_id, "metric": metric, "is_anomaly": False, "out_of_range": False,
                    "latest": None, "range": bounds, "window_count": 0, "mean": None, "std": None,
                    "z_score": None, "slope_per_min": 0.0, "consecutive_breaches": 0, "pattern": "nominal",
                }
            vals = [v for _, v in pts]
            latest, mean = vals[-1], sum(vals) / m
            std = math.sqrt(sum((v - mean) ** 2 for v in vals) / m)
            z_score = (latest - mean) / std if std > 1e-9 else 0.0
            ts_mins = [(dt - pts[0][0]).total_seconds() / 60.0 for dt, _ in pts]
            t_mean = sum(ts_mins) / m
            denom = sum((t - t_mean) ** 2 for t in ts_mins)
            slope = (sum((t - t_mean) * (v - mean) for t, v in zip(ts_mins, vals)) / denom) if (m >= 2 and denom > 1e-12) else 0.0

            out_of_range, consec = False, 0
            if bounds and len(bounds) == 2:
                low, high = float(bounds[0]), float(bounds[1])
                out_of_range = latest < low or latest > high
                if out_of_range:
                    for v in reversed(vals):
                        if v < low or v > high:
                            consec += 1
                        else:
                            break
            pattern = "nominal" if not out_of_range else ("transient_spike" if consec == 1 else "sustained_anomaly")
            return {
                "asset_id": asset_id, "metric": metric, "is_anomaly": pattern == "sustained_anomaly",
                "out_of_range": out_of_range, "latest": latest, "range": bounds, "window_count": m,
                "mean": round(mean, 4), "std": round(std, 4), "z_score": round(z_score, 4),
                "slope_per_min": round(slope, 4), "consecutive_breaches": consec, "pattern": pattern,
            }

        if name == "forecast_tsfm_metric":
            asset_id, metric = str(args.get("asset_id") or default_asset), str(args["metric"])
            horizon = max(float(args.get("horizon_minutes", 30.0)), 0.0)
            w = max(int(args.get("window", 8)), 2)
            bounds = (self.asset_map.get(asset_id, self.scenario["asset_info"]).get("normal_ranges") or {}).get(metric)
            pts = self._numeric_series(asset_id, metric, replay_time, w)
            m = len(pts)
            if m == 0:
                return {
                    "asset_id": asset_id, "metric": metric, "latest": None, "slope_per_min": 0.0,
                    "predicted_value": None, "residual_std": 0.0, "prediction_band": [None, None],
                    "range": bounds, "predicted_breach": False, "estimated_minutes_to_breach": None,
                    "window_count": 0, "horizon_minutes": horizon,
                }
            vals = [v for _, v in pts]
            ts_mins = [(dt - pts[0][0]).total_seconds() / 60.0 for dt, _ in pts]
            t_last, latest = ts_mins[-1], vals[-1]
            v_mean, t_mean = sum(vals) / m, sum(ts_mins) / m
            denom = sum((t - t_mean) ** 2 for t in ts_mins)
            if m >= 2 and denom > 1e-12:
                slope = sum((t - t_mean) * (v - v_mean) for t, v in zip(ts_mins, vals)) / denom
                intercept = v_mean - slope * t_mean
                resid_ss = sum((v - (intercept + slope * t)) ** 2 for t, v in zip(ts_mins, vals))
                resid_std = math.sqrt(resid_ss / max(m - 2, 1))
            else:
                slope, intercept, resid_std = 0.0, latest, 0.0
            pred = intercept + slope * (t_last + horizon)
            band = [round(pred - 1.96 * resid_std, 4), round(pred + 1.96 * resid_std, 4)]

            pred_breach, est_mins = False, None
            if bounds and len(bounds) == 2:
                low, high = float(bounds[0]), float(bounds[1])
                already = latest < low or latest > high
                pred_breach = already or (pred < low or pred > high)
                if already:
                    est_mins = 0.0
                elif slope > 1e-9:
                    est_mins = round(max(0.0, (high - latest) / slope), 4)
                elif slope < -1e-9:
                    est_mins = round(max(0.0, (low - latest) / (-slope)), 4)

            return {
                "asset_id": asset_id, "metric": metric, "latest": round(latest, 4),
                "slope_per_min": round(slope, 4), "predicted_value": round(pred, 4),
                "residual_std": round(resid_std, 4), "prediction_band": band, "range": bounds,
                "predicted_breach": pred_breach, "estimated_minutes_to_breach": est_mins,
                "window_count": m, "horizon_minutes": round(horizon, 4),
            }

        if name == "lookup_fmea_manual":
            asset_type = args.get("asset_type")
            keyword = args.get("keyword")
            fault_type = args.get("fault_type")
            if asset_type is None and keyword is None and fault_type is None:
                asset_type = self.scenario["asset_info"]["asset_type"]
            manuals = list(self.scenario.get("tool_catalog_data", {}).get("fmea_manuals", []))
            if asset_type is not None:
                manuals = [m for m in manuals if m.get("asset_type") == asset_type]
            if fault_type is not None:
                ft_low = str(fault_type).strip().lower()
                manuals = [m for m in manuals if str(m.get("fault_type", "")).lower() == ft_low]
            if keyword is not None:
                tokens = str(keyword).lower().split()
                filtered = []
                for m in manuals:
                    blob_parts = []
                    for f in FMEA_TEXT_FIELDS:
                        val = m.get(f)
                        if isinstance(val, list):
                            blob_parts.extend(str(x) for x in val)
                        elif val is not None:
                            blob_parts.append(str(val))
                    blob = " ".join(blob_parts).lower()
                    if all(tok in blob for tok in tokens):
                        filtered.append(m)
                manuals = filtered
            return {"asset_type": asset_type, "keyword": keyword, "fault_type": fault_type, "count": len(manuals), "manuals": manuals}

        if name == "query_workorder_history":
            req_aid = args.get("asset_id")
            status = str(args.get("status", "all"))
            limit = max(int(args.get("limit", 10)), 0)
            if status not in {"all", "open", "in_progress", "closed"}:
                raise ValueError(f"Invalid workorder status '{status}'")
            wos = self._merged_workorders(replay_time)
            if req_aid is not None:
                wos = [w for w in wos if w.get("asset_id") == req_aid]
            if status != "all":
                wos = [w for w in wos if w.get("status") == status]
            wos = wos[-limit:] if limit else []
            return {"asset_id": req_aid, "status": status, "count": len(wos), "workorders": wos}

        if name == "update_workorder":
            wo_id = str(args["workorder_id"])
            status = args.get("status")
            note = str(args.get("note", ""))
            priority = args.get("priority")
            if status is not None and status not in {"open", "in_progress", "closed"}:
                raise ValueError(f"Invalid workorder status '{status}'")

            merged = {w["workorder_id"]: w for w in self._merged_workorders(replay_time)}
            if wo_id not in merged:
                raise ValueError(f"Workorder '{wo_id}' not found or not released")
            orig = merged[wo_id]
            if orig.get("status") == "closed":
                raise ValueError(f"Cannot update closed workorder '{wo_id}'")

            target = next((w for w in self.world_state["workorders"] if w["workorder_id"] == wo_id), None)
            if target is None:
                target = dict(orig)
                self.world_state["workorders"].append(target)
            if status is not None:
                target["status"] = status
            if priority is not None:
                target["priority"] = str(priority)
            if note:
                target.setdefault("notes", []).append(note)
            target["updated_at"] = replay_time

            wo_asset = str(target.get("asset_id") or default_asset)
            self.world_state["assets"].setdefault(wo_asset, {}).update({
                "last_workorder_id": wo_id,
                "last_workorder_action": "updated",
                "workorder_status": target["status"],
            })
            return {
                "workorder_id": wo_id, "asset_id": wo_asset, "status": target["status"],
                "priority": target.get("priority"), "updated_at": replay_time, "notes": list(target.get("notes", [])),
            }

        if name == "submit_approval_request":
            asset_id, target_tool = str(args.get("asset_id") or default_asset), str(args["target_tool"])
            if target_tool in self.forbidden_tools:
                return {"approved": False, "reason": f"Target tool '{target_tool}' is forbidden"}
            known_all = BUILTIN_TOOLS | set(self.custom_tools) | self.forbidden_tools | self.approval_required_tools
            if target_tool not in known_all:
                raise ValueError(f"Unknown target_tool '{target_tool}'")
            policy = self.approval_policy.get(target_tool) or self.approval_policy.get("*") or {}
            if policy.get("auto_approve", True) is False:
                return {"approved": False, "reason": f"Approval denied by policy for '{target_tool}'"}
            req_ws = policy.get("required_world_state")
            if req_ws and not recursive_subset_match(req_ws, self.world_state):
                return {"approved": False, "reason": "required_world_state preconditions not satisfied"}

            delay_turns = max(int(policy.get("delay_turns", 1)), 0)
            eff = turn_idx + delay_turns
            if target_tool in self.approval_required_tools:
                prev_eff = self.approved_from_turn.get((asset_id, target_tool))
                eff = min(prev_eff, eff) if prev_eff is not None else eff
                self.approved_from_turn[(asset_id, target_tool)] = eff
            self.world_state["approvals"][f"{asset_id}:{target_tool}"] = {
                "asset_id": asset_id, "target_tool": target_tool, "approved": True, "effective_from_turn": eff,
            }
            return {
                "approved": True, "asset_id": asset_id, "target_tool": target_tool,
                "delay_turns": delay_turns, "effective_from_turn": eff,
            }

        priority = "normal" if name == "create_maintenance_workorder" else "urgent_shutdown"
        wo_id = f"WO-{turn_idx}-{step_idx}"
        wo_asset = str(args.get("asset_id") or default_asset)
        wo = {
            "workorder_id": wo_id, "asset_id": wo_asset, "fault_type": str(args.get("fault_type", "")),
            "priority": priority, "status": "open", "created_at": replay_time, "summary": str(args.get("summary", "")),
        }
        self.world_state["workorders"].append(wo)
        asset_patch: dict[str, Any] = {"last_workorder_id": wo_id, "last_workorder_action": "created", "workorder_status": "open"}
        if name == "create_urgent_shutdown_workorder":
            asset_patch.update({"status": "shutdown_scheduled", "urgent_shutdown": True})
        self.world_state["assets"].setdefault(wo_asset, {}).update(asset_patch)
        return {"workorder_id": wo_id, "priority": priority, "status": "open", "asset_id": wo_asset, "fault_type": wo["fault_type"], "summary": wo["summary"]}
