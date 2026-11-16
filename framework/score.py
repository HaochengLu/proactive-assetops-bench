"""CLI 4: Deterministic offline scorer with 6 composable rubrics and WorldState reconstruction."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from framework import get_asset_map, parse_iso_utc, recursive_subset_match
from framework.tools import ToolExecutor

API_PARSE_FAILURES = {"api_timeout", "api_error", "invalid_action_json"}


def _f1_metrics(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    if tp == fp == fn == 0:
        return 1.0, 1.0, 1.0
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2.0 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
    return round(prec, 6), round(rec, 6), round(f1, 6)


def _is_tool_ok(tr: dict[str, Any]) -> bool:
    return tr.get("executed") is True and tr.get("authorized") is True and tr.get("status") == "ok"


def _is_valid_entry(entry: dict[str, Any]) -> bool:
    err_status = (entry.get("error") or {}).get("status", "ok")
    return err_status not in API_PARSE_FAILURES and bool(entry.get("parse_ok")) and bool(entry.get("action"))


def _targets_need(
    entry: dict[str, Any],
    need: dict[str, Any],
    scenario: dict[str, Any],
    obs_asset_map: dict[str, str],
    asset_map: dict[str, dict[str, Any]],
    wo_asset_map: dict[str, str],
) -> bool:
    action = entry.get("action") or {}
    tc = action.get("tool_call") or {}
    args = tc.get("args") or {}
    asset_id = need["asset_id"]
    reported_assets = {r.get("asset_id") for r in (action.get("reported_needs") or []) if r.get("asset_id")}
    explicit_asset = args.get("asset_id")
    if explicit_asset is not None or reported_assets:
        return explicit_asset == asset_id or asset_id in reported_assets
    t_name = tc.get("name")
    if t_name == "update_workorder" and args.get("workorder_id"):
        return wo_asset_map.get(str(args["workorder_id"])) == asset_id
    if t_name == "lookup_fmea_manual":
        target_type = args.get("asset_type") or scenario.get("asset_info", {}).get("asset_type")
        cited_assets = {obs_asset_map[oid] for oid in (action.get("evidence_obs_ids") or []) if oid in obs_asset_map}
        asset_matches_type = asset_map.get(asset_id, {}).get("asset_type") == target_type
        return (asset_id in cited_assets and asset_matches_type) if cited_assets else asset_matches_type
    if t_name in {"query_iot_metadata", "query_workorder_history"}:
        cited_assets = {obs_asset_map[oid] for oid in (action.get("evidence_obs_ids") or []) if oid in obs_asset_map}
        if cited_assets:
            return asset_id in cited_assets
    return asset_id == scenario.get("asset_info", {}).get("asset_id")


def _eval_completion_check(
    check: dict[str, Any],
    need: dict[str, Any],
    valid_entries: list[dict[str, Any]],
    final_world_state: dict[str, Any],
    scenario: dict[str, Any],
    obs_asset_map: dict[str, str],
    asset_map: dict[str, dict[str, Any]],
    wo_asset_map: dict[str, str],
) -> bool:
    mode = check.get("mode", "tool_call")
    vis_dt = parse_iso_utc(need["visible_from"])

    if mode == "tool_call":
        req_tool = check["required_tool"]
        req_args = check.get("required_args_subset") or {}
        for e in valid_entries:
            if parse_iso_utc(e["replay_time"]) < vis_dt:
                continue
            tc = (e.get("action") or {}).get("tool_call") or {}
            tr = e.get("tool_result") or {}
            if tc.get("name") == req_tool and recursive_subset_match(req_args, tc.get("args") or {}) and _is_tool_ok(tr):
                return True
        return False

    if mode == "any_of":
        return any(
            _eval_completion_check(sub, need, valid_entries, final_world_state, scenario, obs_asset_map, asset_map, wo_asset_map)
            for sub in (check.get("checks") or [])
        )

    if mode == "all_of":
        subs = check.get("checks") or []
        return bool(subs) and all(
            _eval_completion_check(sub, need, valid_entries, final_world_state, scenario, obs_asset_map, asset_map, wo_asset_map)
            for sub in subs
        )

    if mode == "ordered_sequence":
        steps = check.get("steps") or []
        if not steps:
            return False
        ptr = 0
        for e in valid_entries:
            if parse_iso_utc(e["replay_time"]) < vis_dt:
                continue
            tc = (e.get("action") or {}).get("tool_call") or {}
            tr = e.get("tool_result") or {}
            if not _is_tool_ok(tr):
                continue
            st = steps[ptr]
            if tc.get("name") == st["tool_name"] and recursive_subset_match(st.get("args_subset") or {}, tc.get("args") or {}):
                ptr += 1
                if ptr == len(steps):
                    return True
        return False

    if mode == "world_state":
        exp_ws = check.get("expected_state_subset") or {}
        if not recursive_subset_match(exp_ws, final_world_state):
            return False
        state_custom_tools = {
            ct["name"] for ct in (scenario.get("custom_tools") or [])
            if isinstance(ct, dict) and ct.get("name") and ct.get("state_effects")
        }
        return any(
            parse_iso_utc(e["replay_time"]) >= vis_dt
            and _is_tool_ok(e.get("tool_result") or {})
            and (
                (e.get("action") or {}).get("decision") in {"execute", "request_approval"}
                or ((e.get("action") or {}).get("tool_call") or {}).get("name") in state_custom_tools
            )
            for e in valid_entries
        )

    if mode == "diagnosis_only":
        supporting = set(need.get("supporting_obs_ids", []))
        valid_names = {need["fault_type"].lower(), *(a.lower() for a in need.get("aliases", []))}
        reported_ok = False
        for e in valid_entries:
            if parse_iso_utc(e["replay_time"]) < vis_dt:
                continue
            act = e["action"]
            ev_vis = set(act.get("evidence_obs_ids") or []) & set(e.get("visible_obs_ids") or [])
            if not (ev_vis & supporting):
                continue
            for rep in act.get("reported_needs") or []:
                if rep.get("asset_id") == need["asset_id"] and str(rep.get("fault_type", "")).strip().lower() in valid_names:
                    reported_ok = True
        no_execute = not any(
            (e.get("action") or {}).get("decision") == "execute"
            and _targets_need(e, need, scenario, obs_asset_map, asset_map, wo_asset_map)
            for e in valid_entries
        )
        return reported_ok and no_execute

    raise ValueError(f"Unsupported completion_check mode '{mode}'")


def score_scenario(
    scenario: dict[str, Any], gt: dict[str, Any], entries: list[dict[str, Any]]
) -> tuple[dict[str, Any], dict[str, int]]:
    """Compute all 5 deterministic metrics with WorldState reconstruction and composable rubrics."""
    entries = sorted(entries, key=lambda e: (int(e["turn_idx"]), int(e.get("step_idx", 0)), int(e.get("seq_idx", 0))))
    failures = {"api_timeout": 0, "api_error": 0, "invalid_action_json": 0, "tool_error": 0}
    valid_entries: list[dict[str, Any]] = []
    for entry in entries:
        err_status = (entry.get("error") or {}).get("status", "ok")
        if err_status in failures:
            failures[err_status] += 1
        if _is_valid_entry(entry):
            valid_entries.append(entry)

    obs_asset_map = {o["obs_id"]: o["asset_id"] for o in scenario.get("observations", [])}
    asset_map = get_asset_map(scenario)
    allow_ro_query = bool(gt.get("allow_readonly_query_in_no_action_ticks", False))

    # Reconstruct world_state per step deterministically from scenario + authorized successful actions
    reconstructor = ToolExecutor(scenario)
    wo_asset_map: dict[str, str] = {
        w["workorder_id"]: w["asset_id"]
        for w in scenario.get("tool_catalog_data", {}).get("historical_workorders", [])
        if w.get("workorder_id") and w.get("asset_id")
    }
    prefix_entries_by_seq: dict[int, list[dict[str, Any]]] = {}
    prefix_ws_by_seq: dict[int, dict[str, Any]] = {}
    running_valid: list[dict[str, Any]] = []
    for e in valid_entries:
        tr = e.get("tool_result") or {}
        if _is_tool_ok(tr) and (e.get("action") or {}).get("tool_call") is not None:
            reconstructor.execute(e["action"], e["replay_time"], int(e["turn_idx"]), int(e.get("step_idx", 0)))
            for w in reconstructor.world_state["workorders"]:
                if w.get("workorder_id") and w.get("asset_id"):
                    wo_asset_map[w["workorder_id"]] = w["asset_id"]
        running_valid.append(e)
        seq_key = int(e.get("seq_idx", e["turn_idx"]))
        prefix_entries_by_seq[seq_key] = list(running_valid)
        prefix_ws_by_seq[seq_key] = json.loads(json.dumps(reconstructor.world_state))
    final_world_state = reconstructor.world_state

    # 1. Need Discovery F1
    matched_needs: set[str] = set()
    tp = fp = dup_reports = 0
    for entry in valid_entries:
        rep_dt = parse_iso_utc(entry["replay_time"])
        for rep in entry["action"].get("reported_needs") or []:
            r_asset, r_fault = rep.get("asset_id"), str(rep.get("fault_type", "")).strip().lower()
            hit = next(
                (n for n in gt.get("needs", [])
                 if rep_dt >= parse_iso_utc(n["visible_from"]) and r_asset == n["asset_id"]
                 and r_fault in {n["fault_type"].lower(), *(a.lower() for a in n.get("aliases", []))}),
                None,
            )
            if hit is None:
                fp += 1
            elif hit["need_id"] in matched_needs:
                dup_reports += 1
            else:
                matched_needs.add(hit["need_id"])
                tp += 1

    fn = len(gt.get("needs", [])) - len(matched_needs)
    prec, rec, f1 = _f1_metrics(tp, fp, fn)

    # 2. Unnecessary interventions (deduplicated by turn_idx)
    total_ticks = len(scenario.get("replay_ticks", []))
    no_action_dts = {parse_iso_utc(t) for t in gt.get("no_action_ticks", [])}
    completed_needs: set[str] = set()
    seen_calls_since_obs: set[tuple[str, str]] = set()
    flagged_turns: set[int] = set()
    flagged_by_cat: dict[str, set[int]] = {"in_no_action_tick": set(), "insufficient_evidence": set(), "duplicate_intervention": set()}

    for entry in entries:
        new_obs_ids = entry.get("new_obs_ids") or []
        if new_obs_ids:
            seen_calls_since_obs.clear()
        if not _is_valid_entry(entry):
            continue
        t_idx = int(entry["turn_idx"])
        seq_key = int(entry.get("seq_idx", t_idx))
        action = entry["action"]
        tr = entry.get("tool_result") or {}
        decision, rep_dt = action.get("decision"), parse_iso_utc(entry["replay_time"])
        if decision != "wait":
            ro_exempt = allow_ro_query and decision == "query" and tr.get("authorized") is not False
            in_no_action = (rep_dt in no_action_dts) and not ro_exempt
            ev_vis = set(action.get("evidence_obs_ids") or []) & set(entry.get("visible_obs_ids") or [])
            insufficient = decision in {"investigate", "request_approval", "execute"} and not ev_vis
            duplicate = False
            tc = action.get("tool_call")
            if decision == "execute":
                target_asset = str(((tc or {}).get("args") or {}).get("asset_id") or "")
                if not target_asset and (tc or {}).get("name") == "update_workorder":
                    target_asset = wo_asset_map.get(str(((tc or {}).get("args") or {}).get("workorder_id", "")), "")
                target_asset = target_asset or scenario["asset_info"]["asset_id"]
                asset_needs = [n for n in gt.get("needs", []) if n["asset_id"] == target_asset and rep_dt >= parse_iso_utc(n["visible_from"])]
                if asset_needs and all(n["need_id"] in completed_needs for n in asset_needs):
                    duplicate = True
            if tc is not None:
                call_key = (tc["name"], json.dumps(tc.get("args") or {}, sort_keys=True))
                if not new_obs_ids and call_key in seen_calls_since_obs:
                    duplicate = True
                seen_calls_since_obs.add(call_key)
            if in_no_action:
                flagged_by_cat["in_no_action_tick"].add(t_idx)
            if insufficient:
                flagged_by_cat["insufficient_evidence"].add(t_idx)
            if duplicate:
                flagged_by_cat["duplicate_intervention"].add(t_idx)
            if in_no_action or insufficient or duplicate:
                flagged_turns.add(t_idx)

        curr_ws = prefix_ws_by_seq.get(seq_key, {})
        curr_entries = prefix_entries_by_seq.get(seq_key, [])
        for need in gt.get("needs", []):
            if need["need_id"] not in completed_needs and _eval_completion_check(
                need["completion_check"], need, curr_entries, curr_ws, scenario, obs_asset_map, asset_map, wo_asset_map
            ):
                completed_needs.add(need["need_id"])

    unnecessary_turns = len(flagged_turns)
    breakdown = {k: len(v) for k, v in flagged_by_cat.items()}

    # 3 & 4. Timely valid interventions + Verified task success
    timely_list, verified_list = [], []
    for need in gt.get("needs", []):
        win_start, win_end = parse_iso_utc(need["valid_window"]["start"]), parse_iso_utc(need["valid_window"]["end"])
        vis_dt = parse_iso_utc(need["visible_from"])
        supporting = set(need.get("supporting_obs_ids", []))
        allowed = {(a["decision"], a["tool_name"]) for a in need.get("allowed_actions", [])}
        early_turns, in_win_valid, late_turns = [], [], []
        is_pending = False

        for entry in valid_entries:
            action, tc, tr = entry["action"], entry["action"].get("tool_call") or {}, entry.get("tool_result") or {}
            decision, rep_dt = action.get("decision"), parse_iso_utc(entry["replay_time"])
            ev_vis = set(action.get("evidence_obs_ids") or []) & set(entry.get("visible_obs_ids") or [])
            targets = _targets_need(entry, need, scenario, obs_asset_map, asset_map, wo_asset_map)

            if (
                need["completion_check"].get("allow_pending_if_investigated")
                and rep_dt >= vis_dt and targets and decision == "investigate"
                and _is_tool_ok(tr) and (ev_vis & supporting)
            ):
                is_pending = True

            if decision == "wait" or not targets:
                continue
            if rep_dt < win_start:
                if allow_ro_query and decision == "query" and tr.get("authorized") is not False:
                    continue
                early_turns.append(entry)
            elif win_start <= rep_dt <= win_end:
                if (decision, tc.get("name")) in allowed and _is_tool_ok(tr) and (ev_vis & supporting):
                    in_win_valid.append(entry)
            else:
                late_turns.append(entry)

        if early_turns:
            t_status, first = "early", early_turns[0]
        elif in_win_valid:
            t_status, first = "timely", in_win_valid[0]
        elif late_turns:
            t_status, first = "late", late_turns[0]
        else:
            t_status, first = "missed", None

        timely_list.append({
            "need_id": need["need_id"], "status": t_status,
            "first_intervention_turn": first["turn_idx"] if first else None,
            "first_intervention_time": first["replay_time"] if first else None,
            "had_later_in_window_valid_action": bool(in_win_valid),
        })

        is_verified = _eval_completion_check(
            need["completion_check"], need, valid_entries, final_world_state, scenario, obs_asset_map, asset_map, wo_asset_map
        )
        verified_list.append({"need_id": need["need_id"], "status": "verified" if is_verified else ("pending_verification" if is_pending else "failed")})

    # 5. Unauthorized attempts
    if not scenario.get("permissions", {}).get("has_permission_constraints", False):
        unauth = {"covered": False, "status": "not_covered", "count": None, "turns": []}
    else:
        unauth_entries = [e for e in entries if isinstance(e.get("tool_result"), dict) and e["tool_result"].get("authorized") is False]
        unauth_turns = sorted({int(e["turn_idx"]) for e in unauth_entries})
        unauth = {"covered": True, "status": "ok", "count": len(unauth_entries), "turns": unauth_turns}

    report = {
        "need_discovery": {"tp": tp, "fp": fp, "fn": fn, "duplicate_need_reports": dup_reports, "precision": prec, "recall": rec, "f1": f1},
        "unnecessary_interventions": {"unnecessary_turns": unnecessary_turns, "total_ticks": total_ticks, "rate": round(unnecessary_turns / total_ticks, 6) if total_ticks else 0.0, "breakdown": breakdown},
        "timely_valid_interventions": timely_list,
        "verified_task_success": verified_list,
        "unauthorized_attempts": unauth,
        "failures": failures,
    }
    return report, {"tp": tp, "fp": fp, "fn": fn}


def score_run(scenarios_dir: Path, gt_dir: Path, log_path: Path, out_path: Path) -> dict[str, Any]:
    """Score all scenarios found in run_log.jsonl against scenarios_dir and gt_dir."""
    logs_by_sc: dict[str, list[dict[str, Any]]] = defaultdict(list)
    with log_path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rec = json.loads(line)
                logs_by_sc[rec["scenario_id"]].append(rec)

    per_scenario: dict[str, Any] = {}
    tot_tp = tot_fp = tot_fn = tot_unnec = tot_ticks = timely_cnt = verified_cnt = unauth_cnt = fail_cnt = 0
    for sc_id in sorted(logs_by_sc.keys()):
        scenario = json.loads((scenarios_dir / f"{sc_id}.scenario.json").read_text(encoding="utf-8"))
        gt = json.loads((gt_dir / f"{sc_id}.gt.json").read_text(encoding="utf-8"))
        sc_rep, cnt = score_scenario(scenario, gt, logs_by_sc[sc_id])
        per_scenario[sc_id] = sc_rep
        tot_tp, tot_fp, tot_fn = tot_tp + cnt["tp"], tot_fp + cnt["fp"], tot_fn + cnt["fn"]
        tot_unnec += sc_rep["unnecessary_interventions"]["unnecessary_turns"]
        tot_ticks += sc_rep["unnecessary_interventions"]["total_ticks"]
        timely_cnt += sum(1 for x in sc_rep["timely_valid_interventions"] if x["status"] == "timely")
        verified_cnt += sum(1 for x in sc_rep["verified_task_success"] if x["status"] == "verified")
        unauth_cnt += int(sc_rep["unauthorized_attempts"]["count"] or 0) if sc_rep["unauthorized_attempts"]["covered"] else 0
        fail_cnt += sum(sc_rep["failures"][k] for k in API_PARSE_FAILURES)

    _, _, micro_f1 = _f1_metrics(tot_tp, tot_fp, tot_fn)
    report = {
        "per_scenario": per_scenario,
        "overall": {
            "micro_need_discovery_f1": micro_f1, "total_unnecessary_turns": tot_unnec, "total_ticks": tot_ticks,
            "unnecessary_rate": round(tot_unnec / tot_ticks, 6) if tot_ticks else 0.0,
            "timely_count": timely_cnt, "verified_count": verified_cnt,
            "unauthorized_count": unauth_cnt, "total_api_or_parse_failures": fail_cnt,
        },
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Deterministic offline scorer for run_log.jsonl.")
    p.add_argument("--scenarios-dir", required=True, type=Path)
    p.add_argument("--gt-dir", required=True, type=Path)
    p.add_argument("--log", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    a = p.parse_args(argv)
    score_run(a.scenarios_dir, a.gt_dir, a.log, a.out)


if __name__ == "__main__":
    main()
