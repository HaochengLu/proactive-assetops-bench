"""Thread-local HTTP Keep-Alive streaming VLLMClient and full-feature MockClient."""

from __future__ import annotations

import http.client
import json
import re
import socket
import threading
import time
import urllib.parse
from collections import defaultdict
from typing import Any

from framework import parse_iso_utc, recursive_subset_match

_RETRYABLE_CONN_ERRORS = (
    http.client.RemoteDisconnected,
    http.client.BadStatusLine,
    http.client.CannotSendRequest,
    http.client.ResponseNotReady,
    ConnectionError,
)


def strip_markdown_fences(text: str) -> str:
    """Strip a single outer ```json ... ``` or ``` ... ``` fence if present."""
    stripped = text.strip()
    m = re.match(r"^```(?:json)?\s*([\s\S]*?)\s*```$", stripped)
    return m.group(1).strip() if m else stripped


def parse_action_json(raw_text: str) -> tuple[dict[str, Any] | None, str | None]:
    """Parse and strictly validate Architect action JSON from raw model output."""
    try:
        obj = json.loads(strip_markdown_fences(raw_text))
    except json.JSONDecodeError as exc:
        return None, f"Invalid JSON: {exc}"
    if not isinstance(obj, dict) or not isinstance(obj.get("decision"), str):
        return None, "Action JSON must be an object with string 'decision'"

    reason = "" if obj.get("reason") is None else obj.get("reason")
    if not isinstance(reason, str):
        return None, "Action 'reason' must be a string"

    evidence = [] if obj.get("evidence_obs_ids") is None else obj.get("evidence_obs_ids")
    if not isinstance(evidence, list) or not all(isinstance(x, str) for x in evidence):
        return None, "Action 'evidence_obs_ids' must be a list of strings"

    needs = [] if obj.get("reported_needs") is None else obj.get("reported_needs")
    if not isinstance(needs, list) or not all(
        isinstance(r, dict) and isinstance(r.get("asset_id", ""), str) and isinstance(r.get("fault_type", ""), str)
        for r in needs
    ):
        return None, "Action 'reported_needs' must be a list of objects with string asset_id/fault_type"

    tc = obj.get("tool_call")
    if obj["decision"] == "wait" and tc is not None:
        return None, "Action 'tool_call' must be null when decision is 'wait'"
    if tc is not None:
        if not isinstance(tc, dict) or not isinstance(tc.get("name"), str):
            return None, "Action 'tool_call' must be null or {name, args}"
        args = tc.get("args")
        if args is not None and not isinstance(args, dict):
            return None, "Action 'tool_call.args' must be an object or null"
        tc = {**tc, "args": args or {}}

    obj["reason"], obj["evidence_obs_ids"], obj["reported_needs"], obj["tool_call"] = reason, evidence, needs, tc
    return obj, None


class VLLMClient:
    """OpenAI-compatible SSE streaming client using thread-local HTTP Keep-Alive."""

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str = "",
        temperature: float = 0.0,
        max_tokens: int = 256,
        seed: int = 42,
        timeout: float = 60.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        parsed = urllib.parse.urlsplit(self.base_url)
        self._scheme = parsed.scheme or "http"
        self._host = parsed.hostname or "127.0.0.1"
        self._port = parsed.port or (443 if self._scheme == "https" else 80)
        self._req_path = f"{parsed.path.rstrip('/')}/chat/completions"
        self.model, self.api_key = model, api_key
        self.temperature, self.max_tokens, self.seed, self.timeout = temperature, max_tokens, seed, timeout
        self._local = threading.local()
        self._lock = threading.Lock()
        self._all_conns: list[http.client.HTTPConnection] = []

    def _get_conn(self) -> http.client.HTTPConnection:
        conn = getattr(self._local, "conn", None)
        if conn is None:
            cls = http.client.HTTPSConnection if self._scheme == "https" else http.client.HTTPConnection
            conn = cls(self._host, self._port, timeout=self.timeout)
            self._local.conn = conn
            with self._lock:
                self._all_conns.append(conn)
        return conn

    def _reset_conn(self) -> None:
        conn = getattr(self._local, "conn", None)
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass
            self._local.conn = None

    def close(self) -> None:
        """Close thread-local and tracked HTTP Keep-Alive connections."""
        self._reset_conn()
        with self._lock:
            for c in self._all_conns:
                try:
                    c.close()
                except Exception:
                    pass
            self._all_conns.clear()

    def generate(self, messages: list[dict[str, str]]) -> dict[str, Any]:
        payload = json.dumps({
            "model": self.model, "messages": messages, "temperature": self.temperature,
            "max_tokens": self.max_tokens, "seed": self.seed,
            "stream": True, "stream_options": {"include_usage": True},
        }, ensure_ascii=False).encode("utf-8")
        headers = {"Content-Type": "application/json", "Accept": "text/event-stream", "Connection": "keep-alive"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        start_unix_ms = int(time.time() * 1000)
        t0 = time.perf_counter_ns()

        for attempt in range(2):
            ttft_ms: float | None = None
            content_parts: list[str] = []
            prompt_tokens = completion_tokens = cached_tokens = None
            received_any_chunk = False
            try:
                conn = self._get_conn()
                conn.request("POST", self._req_path, body=payload, headers=headers)
                resp = conn.getresponse()
                http_code = resp.status
                if http_code != 200:
                    err_body = resp.read().decode("utf-8", errors="replace")
                    if http_code >= 500 or http_code in {408, 504}:
                        self._reset_conn()
                    status = "api_timeout" if http_code in {408, 504} else "api_error"
                    return self._err(start_unix_ms, t0, status, http_code, f"HTTP {http_code}: {err_body[:200]}")

                while True:
                    raw_line = resp.readline()
                    if not raw_line:
                        break
                    line = raw_line.decode("utf-8", errors="replace").strip()
                    if not line.startswith("data:"):
                        continue
                    data_str = line[5:].strip()
                    if data_str == "[DONE]":
                        resp.read()
                        break
                    received_any_chunk = True
                    chunk = json.loads(data_str)
                    if chunk.get("usage"):
                        u = chunk["usage"]
                        prompt_tokens, completion_tokens = u.get("prompt_tokens"), u.get("completion_tokens")
                        details = u.get("prompt_tokens_details") or {}
                        if details.get("cached_tokens") is not None:
                            cached_tokens = details["cached_tokens"]
                    choices = chunk.get("choices") or []
                    if choices:
                        delta = choices[0].get("delta") or {}
                        piece, reasoning = delta.get("content"), delta.get("reasoning_content")
                        if ttft_ms is None and (bool(piece) or bool(reasoning)):
                            ttft_ms = round((time.perf_counter_ns() - t0) / 1e6, 3)
                        if piece:
                            content_parts.append(piece)

                latency_ms = round((time.perf_counter_ns() - t0) / 1e6, 3)
                raw_response = "".join(content_parts)
                action, parse_err = parse_action_json(raw_response)
                err_status = "ok" if parse_err is None else "invalid_action_json"
                return {
                    "raw_response": raw_response,
                    "parse_ok": parse_err is None,
                    "action": action,
                    "timing_ms": {"request_start_unix_ms": start_unix_ms, "ttft_ms": ttft_ms, "latency_ms": latency_ms},
                    "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens, "cached_tokens": cached_tokens},
                    "error": {"status": err_status, "http_code": http_code, "message": parse_err},
                }
            except _RETRYABLE_CONN_ERRORS as exc:
                self._reset_conn()
                if attempt == 0 and not received_any_chunk:
                    t0 = time.perf_counter_ns()
                    continue
                return self._err(start_unix_ms, t0, "api_error", None, str(exc))
            except (TimeoutError, socket.timeout) as exc:
                self._reset_conn()
                return self._err(start_unix_ms, t0, "api_timeout", None, str(exc))
            except Exception as exc:
                self._reset_conn()
                return self._err(start_unix_ms, t0, "api_error", None, str(exc))
        return self._err(start_unix_ms, t0, "api_error", None, "Connection retry exhausted")

    @staticmethod
    def _err(start_unix_ms: int, t0: int, status: str, http_code: int | None, msg: str) -> dict[str, Any]:
        return {
            "raw_response": "",
            "parse_ok": False,
            "action": None,
            "timing_ms": {"request_start_unix_ms": start_unix_ms, "ttft_ms": None, "latency_ms": round((time.perf_counter_ns() - t0) / 1e6, 3)},
            "usage": {"prompt_tokens": None, "completion_tokens": None, "cached_tokens": None},
            "error": {"status": status, "http_code": http_code, "message": msg},
        }


def _extract_json_marker(text: str, marker: str) -> dict[str, Any] | None:
    idx = text.find(marker)
    if idx == -1:
        return None
    rest = text[idx + len(marker) :].lstrip()
    return json.loads(rest.split("\n", 1)[0]) if rest else None


def _is_planned_suppression(obs: dict[str, Any]) -> bool:
    if obs.get("modality") != "maintenance_log":
        return False
    p = obs.get("payload") or {}
    return bool(
        p.get("planned_load_shift") is True
        or p.get("expected_transient") is True
        or p.get("event") in {"planned_load_shift", "scheduled_switchover", "planned_maintenance"}
    )


class MockClient:
    """Zero-hardcoding multi-asset AssetOpsBench heuristic client reading ONLY `messages`."""

    def __init__(self, mode: str = "heuristic") -> None:
        if mode not in {"heuristic", "eager_unauth"}:
            raise ValueError(f"Unsupported mock mode: {mode}")
        self.mode = mode

    def generate(self, messages: list[dict[str, str]]) -> dict[str, Any]:
        start_unix_ms, t0 = int(time.time() * 1000), time.perf_counter_ns()
        ctx = _extract_json_marker(messages[0]["content"], "SCENARIO_CONTEXT_JSON:") or {}
        action = self._decide(ctx, messages[1:])
        elapsed_ms = round((time.perf_counter_ns() - t0) / 1e6, 3)
        return {
            "raw_response": json.dumps(action, ensure_ascii=False),
            "parse_ok": True,
            "action": action,
            "timing_ms": {"request_start_unix_ms": start_unix_ms, "ttft_ms": elapsed_ms, "latency_ms": elapsed_ms},
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "cached_tokens": None},
            "error": {"status": "ok", "http_code": 200, "message": None},
        }

    def _decide(self, ctx: dict[str, Any], history: list[dict[str, str]]) -> dict[str, Any]:
        asset_info = ctx.get("asset_info", {})
        assets = ctx.get("assets") or ([asset_info] if asset_info else [])
        default_asset = asset_info.get("asset_id", "")
        asset_meta_map = {a["asset_id"]: a for a in assets if a.get("asset_id")}
        custom_tools = {t["name"]: t for t in (ctx.get("custom_tools") or []) if isinstance(t, dict) and t.get("name")}
        perms = ctx.get("permissions", {})
        allowed_decisions = set(perms.get("allowed_decisions") or [])
        approval_req_list = perms.get("approval_required_tools") or []
        forbidden_list = perms.get("forbidden_tools") or []
        approval_req = set(approval_req_list)
        forbidden = set(forbidden_list)
        approval_policy = perms.get("approval_policy") or {}
        taxonomy = ctx.get("fault_taxonomy") or []

        mock_world_state: dict[str, Any] = {
            "assets": {
                aid: {"status": m.get("initial_status", "running"), **(m.get("initial_state") or {})}
                for aid, m in asset_meta_map.items()
            },
            "workorders": [],
            "approvals": {},
        }

        turn_obs: list[tuple[int, dict[str, Any]]] = []
        suppressed_turns: dict[str, set[int]] = defaultdict(set)
        saw_wo_history_obs: set[str] = set()
        reported_keys: set[tuple[str, str]] = set()
        tsfm_status: dict[tuple[str, str], tuple[int, str]] = {}
        forecast_status: dict[tuple[str, str], tuple[int, dict[str, Any]]] = {}
        unlocked_manuals: dict[str, list[dict[str, Any]]] = {}
        wo_queried_assets: set[str] = set()
        open_wo_by_asset: dict[str, dict[str, Any]] = {}
        approved_from_turn: dict[tuple[str, str], int] = {}
        executed_assets: set[str] = set()
        pending_ast: dict[str, Any] | None = None
        ast_turn_idx = current_turn_idx = 0

        for msg in history:
            if msg["role"] == "user":
                prev_res = _extract_json_marker(msg["content"], "PREV_TOOL_RESULT_JSON:")
                if prev_res and pending_ast and prev_res.get("executed") and prev_res.get("authorized"):
                    tc = pending_ast.get("tool_call") or {}
                    t_name, t_args = tc.get("name", ""), tc.get("args") or {}
                    t_asset = str(t_args.get("asset_id") or default_asset)
                    out = prev_res.get("output") or {}
                    if t_name == "check_tsfm_anomaly":
                        tsfm_status[(t_asset, str(t_args.get("metric", "")))] = (ast_turn_idx, str(out.get("pattern", "nominal")))
                    elif t_name == "forecast_tsfm_metric":
                        forecast_status[(t_asset, str(t_args.get("metric", "")))] = (ast_turn_idx, out)
                    elif t_name == "lookup_fmea_manual":
                        a_type = str(out.get("asset_type") or asset_meta_map.get(t_asset, {}).get("asset_type", ""))
                        unlocked_manuals[a_type] = out.get("manuals") or []
                    elif t_name == "query_workorder_history":
                        wo_queried_assets.add(t_asset)
                        open_wos = [w for w in (out.get("workorders") or []) if w.get("status") in {"open", "in_progress"}]
                        if open_wos:
                            open_wo_by_asset[t_asset] = open_wos[-1]
                    elif t_name == "submit_approval_request" and out.get("approved"):
                        target = str(t_args.get("target_tool", ""))
                        eff = int(out.get("effective_from_turn", ast_turn_idx + 1))
                        approved_from_turn[(t_asset, target)] = eff
                        mock_world_state["approvals"][f"{t_asset}:{target}"] = {"approved": True, "effective_from_turn": eff}
                    elif t_name in {"create_maintenance_workorder", "create_urgent_shutdown_workorder", "update_workorder"}:
                        wo_aid = str(out.get("asset_id") or t_asset)
                        executed_assets.add(wo_aid)
                        mock_world_state["assets"].setdefault(wo_aid, {}).update({"workorder_status": out.get("status", "open")})
                    elif t_name in custom_tools:
                        applied = out.get("applied_effects") or {}
                        mock_world_state["assets"].setdefault(t_asset, {}).update(applied)
                        if custom_tools[t_name].get("terminal", True):
                            executed_assets.add(t_asset)

                turn_data = _extract_json_marker(msg["content"], "TURN_OBSERVATIONS_JSON:") or {}
                current_turn_idx = int(turn_data.get("turn_idx", 0))
                for obs in turn_data.get("new_observations", []):
                    turn_obs.append((current_turn_idx, obs))
                    if _is_planned_suppression(obs) and obs.get("asset_id"):
                        suppressed_turns[obs["asset_id"]].add(current_turn_idx)
                    if obs.get("modality") == "workorder_history" and obs.get("asset_id"):
                        saw_wo_history_obs.add(obs["asset_id"])
            elif msg["role"] == "assistant":
                parsed, _ = parse_action_json(msg["content"])
                pending_ast = parsed
                ast_turn_idx = current_turn_idx
                for rep in (parsed or {}).get("reported_needs") or []:
                    reported_keys.add((str(rep.get("asset_id", "")), str(rep.get("fault_type", "")).lower()))

        if self.mode == "eager_unauth":
            if current_turn_idx == 0:
                target = approval_req_list[0] if approval_req_list else (forbidden_list[0] if forbidden_list else "create_maintenance_workorder")
                fault = taxonomy[0] if taxonomy else "unverified_fault"
                return {
                    "decision": "execute", "reason": "eager_unauth preemptive execution",
                    "evidence_obs_ids": [], "reported_needs": [],
                    "tool_call": {"name": target, "args": {"asset_id": default_asset, "fault_type": fault, "summary": "eager_unauth"}},
                }
            return {"decision": "wait", "reason": "eager_unauth idle", "evidence_obs_ids": [], "reported_needs": [], "tool_call": None}

        for asset_id, asset_meta in asset_meta_map.items():
            if asset_id in executed_assets:
                continue
            ranges = asset_meta.get("normal_ranges", {})
            asset_type = asset_meta.get("asset_type", "")
            horizon = float(asset_meta.get("predictive_horizon_minutes", 30.0))
            metric_series: dict[str, list[tuple[str, str, float]]] = defaultdict(list)
            for o_turn, obs in turn_obs:
                if obs.get("asset_id") != asset_id or obs.get("modality") != "telemetry" or o_turn in suppressed_turns[asset_id]:
                    continue
                payload = obs.get("payload") or {}
                for metric in ranges:
                    val = payload.get(metric)
                    if isinstance(val, (int, float)) and not isinstance(val, bool):
                        metric_series[metric].append((obs["obs_id"], obs["release_time"], float(val)))

            breaching: list[tuple[str, float, float, list[str]]] = []
            pred_candidates: list[tuple[str, list[str]]] = []
            for metric, series in metric_series.items():
                bounds = ranges.get(metric)
                if not series or not isinstance(bounds, list) or len(bounds) != 2:
                    continue
                low, high = float(bounds[0]), float(bounds[1])
                latest_val = series[-1][2]
                if latest_val < low or latest_val > high:
                    span = max(high - low, 1e-6)
                    norm_dev = (low - latest_val) / span if latest_val < low else (latest_val - high) / span
                    streak_ids = []
                    for oid, _, v in reversed(series):
                        if v < low or v > high:
                            streak_ids.append(oid)
                        else:
                            break
                    breaching.append((metric, latest_val, norm_dev, list(reversed(streak_ids))))
                elif self._is_trending_toward_breach(series, low, high, horizon, bool(asset_meta.get("enable_predictive_check"))):
                    pred_candidates.append((metric, [oid for oid, _, _ in series[-min(len(series), 4) :]]))

            if not breaching and not pred_candidates:
                continue

            # Determine active anomaly channel (current breaches take priority over predictive trends)
            breach_map: dict[str, float] = {}
            all_ev_ids: list[str] = []
            if breaching:
                sustained = {m for m, _, _, _ in breaching if tsfm_status.get((asset_id, m), (-1, ""))[1] == "sustained_anomaly"}
                if not sustained:
                    unchecked = [
                        item for item in sorted(breaching, key=lambda x: -x[2])
                        if tsfm_status.get((asset_id, item[0]), (-1, ""))[0] != current_turn_idx
                    ]
                    if not unchecked:
                        continue
                    best_m, _, _, ev_ids = unchecked[0]
                    return {
                        "decision": "investigate",
                        "reason": f"Verifying out-of-range {best_m} on {asset_id} via check_tsfm_anomaly.",
                        "evidence_obs_ids": ev_ids, "reported_needs": [],
                        "tool_call": {"name": "check_tsfm_anomaly", "args": {"asset_id": asset_id, "metric": best_m}},
                    }
                breach_map = {m: v for m, v, _, _ in breaching}
                all_ev_ids = list(dict.fromkeys(oid for _, _, _, ids in breaching for oid in ids))
            else:
                pred_breached = [
                    (m, forecast_status[(asset_id, m)][1], ev)
                    for m, ev in pred_candidates
                    if forecast_status.get((asset_id, m), (-1, {}))[1].get("predicted_breach") is True
                ]
                if not pred_breached:
                    unchecked_p = [
                        (m, ev) for m, ev in pred_candidates
                        if forecast_status.get((asset_id, m), (-1, {}))[0] != current_turn_idx
                    ]
                    if not unchecked_p:
                        continue
                    pm, ev_ids = unchecked_p[0]
                    return {
                        "decision": "investigate",
                        "reason": f"Forecasting monotonic trend of {pm} on {asset_id} via forecast_tsfm_metric.",
                        "evidence_obs_ids": ev_ids, "reported_needs": [],
                        "tool_call": {"name": "forecast_tsfm_metric", "args": {"asset_id": asset_id, "metric": pm, "horizon_minutes": horizon}},
                    }
                for m, f_out, ev in pred_breached:
                    breach_map[m] = float(f_out.get("predicted_value") or 0.0)
                    all_ev_ids.extend(ev)
                all_ev_ids = list(dict.fromkeys(all_ev_ids))

            if asset_type not in unlocked_manuals:
                return {
                    "decision": "investigate",
                    "reason": f"Anomaly confirmed on {asset_id}; looking up {asset_type} FMEA manual.",
                    "evidence_obs_ids": all_ev_ids, "reported_needs": [],
                    "tool_call": {"name": "lookup_fmea_manual", "args": {"asset_type": asset_type}},
                }

            matched = self._match_manual(unlocked_manuals[asset_type], breach_map)
            fault_type = (matched or {}).get("fault_type") or f"anomaly_{next(iter(breach_map))}"
            rec_tool = self._infer_recommended_tool(matched, custom_tools, forbidden)
            reported = [] if (asset_id, fault_type.lower()) in reported_keys else [
                {"asset_id": asset_id, "fault_type": fault_type, "summary": f"Confirmed {fault_type} from {breach_map}"}
            ]

            if "execute" not in allowed_decisions:
                if reported:
                    return {
                        "decision": "wait",
                        "reason": f"Diagnosis-only mode: reporting {fault_type} on {asset_id} without execution.",
                        "evidence_obs_ids": all_ev_ids, "reported_needs": reported, "tool_call": None,
                    }
                continue

            if (perms.get("open_workorder_check") or asset_id in saw_wo_history_obs) and asset_id not in wo_queried_assets:
                return {
                    "decision": "query",
                    "reason": f"Checking open workorder history for {asset_id} before issuing workorder.",
                    "evidence_obs_ids": all_ev_ids, "reported_needs": [],
                    "tool_call": {"name": "query_workorder_history", "args": {"asset_id": asset_id, "status": "open"}},
                }

            if asset_id in open_wo_by_asset and "update_workorder" not in forbidden:
                wo = open_wo_by_asset[asset_id]
                return {
                    "decision": "execute",
                    "reason": f"Updating existing open workorder {wo['workorder_id']} for {asset_id} ({fault_type}).",
                    "evidence_obs_ids": all_ev_ids, "reported_needs": reported,
                    "tool_call": {
                        "name": "update_workorder",
                        "args": {"workorder_id": wo["workorder_id"], "status": "in_progress", "priority": "high", "note": f"Confirmed {fault_type}: {breach_map}"},
                    },
                }

            if rec_tool in approval_req:
                eff_turn = approved_from_turn.get((asset_id, rec_tool))
                if eff_turn is None:
                    policy = approval_policy.get(rec_tool) or approval_policy.get("*") or {}
                    req_ws = policy.get("required_world_state")
                    if req_ws and not recursive_subset_match(req_ws, mock_world_state):
                        prereq = self._find_prerequisite_custom_tool(req_ws, mock_world_state, custom_tools, asset_id)
                        if prereq is not None:
                            ct_name, ct_decision, ct_args = prereq
                            return {
                                "decision": ct_decision,
                                "reason": f"Satisfying prerequisite world_state before {rec_tool} via {ct_name}.",
                                "evidence_obs_ids": all_ev_ids, "reported_needs": [],
                                "tool_call": {"name": ct_name, "args": ct_args},
                            }
                    return {
                        "decision": "request_approval",
                        "reason": f"Requesting approval for {rec_tool} on {asset_id} ({fault_type}).",
                        "evidence_obs_ids": all_ev_ids, "reported_needs": reported,
                        "tool_call": {"name": "submit_approval_request", "args": {"asset_id": asset_id, "target_tool": rec_tool, "reason": str(breach_map)}},
                    }
                if current_turn_idx < eff_turn:
                    continue

            return {
                "decision": "execute",
                "reason": f"Executing {rec_tool} on {asset_id} for confirmed {fault_type}.",
                "evidence_obs_ids": all_ev_ids, "reported_needs": reported,
                "tool_call": {"name": rec_tool, "args": {"asset_id": asset_id, "fault_type": fault_type, "summary": str(breach_map)}},
            }

        return {"decision": "wait", "reason": "No sustained unexplained anomalies requiring action.", "evidence_obs_ids": [], "reported_needs": [], "tool_call": None}

    @staticmethod
    def _is_trending_toward_breach(
        series: list[tuple[str, str, float]], low: float, high: float, horizon_min: float, force_check: bool
    ) -> bool:
        if len(series) < (2 if force_check else 3):
            return False
        vals = [v for _, _, v in series]
        if not force_check:
            inc = vals[-3] < vals[-2] < vals[-1]
            dec = vals[-3] > vals[-2] > vals[-1]
            if not (inc or dec):
                return False
        elif vals[-2] == vals[-1]:
            return False

        dts = [parse_iso_utc(ts) for _, ts, _ in series]
        t_mins = [(dt - dts[0]).total_seconds() / 60.0 for dt in dts]
        t_mean, v_mean = sum(t_mins) / len(t_mins), sum(vals) / len(vals)
        denom = sum((t - t_mean) ** 2 for t in t_mins)
        if denom <= 1e-12:
            return False
        slope = sum((t - t_mean) * (v - v_mean) for t, v in zip(t_mins, vals)) / denom
        pred = (v_mean - slope * t_mean) + slope * (t_mins[-1] + horizon_min)
        return pred < low or pred > high

    @staticmethod
    def _match_manual(manuals: list[dict[str, Any]], breach_map: dict[str, float]) -> dict[str, Any] | None:
        for m in manuals:
            sym = m.get("symptom_metric")
            if sym and sym in breach_map:
                val = breach_map[sym]
                th_hi, th_lo = m.get("threshold_high"), m.get("threshold_low")
                if th_hi is not None and val < float(th_hi):
                    continue
                if th_lo is not None and val > float(th_lo):
                    continue
                return m
        # Free-text FMSR fallback matching metric tokens / keywords
        metric_tokens = [tok for m in breach_map for tok in m.lower().split("_") if len(tok) > 1]
        best_m, best_score = None, 0
        for m in manuals:
            blob = " ".join(
                str(m.get(k, "")) for k in ("fault_type", "failure_mode", "symptoms", "cause", "description", "remediation")
            ).lower()
            score = sum(1 for tok in metric_tokens if tok in blob)
            if score > best_score:
                best_score, best_m = score, m
        return best_m or next((m for m in manuals if m.get("symptom_metric") in breach_map), manuals[0] if manuals else None)

    @staticmethod
    def _infer_recommended_tool(
        matched: dict[str, Any] | None, custom_tools: dict[str, dict[str, Any]], forbidden: set[str]
    ) -> str:
        if matched and matched.get("recommended_tool"):
            return str(matched["recommended_tool"])
        rem_text = " ".join(
            str((matched or {}).get(k, "")) for k in ("remediation", "description", "failure_mode")
        ).lower()
        for ct_name in custom_tools:
            if ct_name.lower() in rem_text:
                return ct_name
        if ("urgent" in rem_text or "shutdown" in rem_text) and "create_urgent_shutdown_workorder" not in forbidden:
            return "create_urgent_shutdown_workorder"
        return "create_maintenance_workorder"

    @staticmethod
    def _find_prerequisite_custom_tool(
        req_ws: dict[str, Any],
        curr_ws: dict[str, Any],
        custom_tools: dict[str, dict[str, Any]],
        fallback_asset: str,
    ) -> tuple[str, str, dict[str, Any]] | None:
        req_assets = req_ws.get("assets") or {}
        for target_aid, expected_state in req_assets.items():
            actual_state = (curr_ws.get("assets") or {}).get(target_aid, {})
            if recursive_subset_match(expected_state, actual_state):
                continue
            for ct in custom_tools.values():
                effects = ct.get("state_effects") or {}
                bound_args: dict[str, Any] = {}
                matches = True
                for k, v in expected_state.items():
                    eff_v = effects.get(k)
                    if isinstance(eff_v, str) and eff_v.startswith("$args."):
                        bound_args[eff_v[6:]] = v
                    elif eff_v != v:
                        matches = False
                        break
                if not matches:
                    continue
                args = {req: bound_args.get(req, target_aid if req == "asset_id" else "") for req in (ct.get("required_args") or ["asset_id"])}
                args.update(bound_args)
                if "asset_id" not in args:
                    args["asset_id"] = target_aid or fallback_asset
                return ct["name"], str(ct["decision"]), args
        return None
