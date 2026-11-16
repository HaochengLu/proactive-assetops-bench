"""CLI 2: Static validator for scenario/GT pairing, schema, references, and zero-leakage."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from framework import SUPPORTED_SCHEMA_VERSIONS, VALID_MODALITIES, get_asset_map, parse_iso_utc
from framework.convert import FORBIDDEN_LABEL_COLUMNS
from framework.prompt import build_system_message
from framework.tools import BUILTIN_TOOLS

SPOILER_PROMPT_KEYS = (
    "fmea_manuals", "symptom_metric", "threshold_high",
    "threshold_low", "recommended_tool", "historical_workorders",
)
FORBIDDEN_ASSET_KEYS = {"role", "is_faulty", "ground_truth", "fault_type"}
FORBIDDEN_META_KEYS = FORBIDDEN_LABEL_COLUMNS | {"answer", "expected_output", "gold_tool"}
VALID_CHECK_MODES = {"tool_call", "any_of", "all_of", "ordered_sequence", "world_state", "diagnosis_only"}
VALID_DECISIONS = {"wait", "query", "investigate", "request_approval", "execute"}


def _validate_completion_check(
    check: Any, known_tools: set[str], errors: list[str], prefix: str
) -> None:
    if not isinstance(check, dict):
        errors.append(f"{prefix}: completion_check must be a dict")
        return
    mode = check.get("mode", "tool_call")
    if mode not in VALID_CHECK_MODES:
        errors.append(f"{prefix}: unknown completion_check mode '{mode}'")
        return
    if mode == "tool_call":
        req_tool = check.get("required_tool")
        if not isinstance(req_tool, str) or req_tool not in known_tools:
            errors.append(f"{prefix}: invalid required_tool '{req_tool}'")
    elif mode in {"any_of", "all_of"}:
        sub = check.get("checks")
        if not isinstance(sub, list) or not sub:
            errors.append(f"{prefix}: '{mode}' requires non-empty checks list")
        else:
            for idx, item in enumerate(sub):
                _validate_completion_check(item, known_tools, errors, f"{prefix}.checks[{idx}]")
    elif mode == "ordered_sequence":
        steps = check.get("steps")
        if not isinstance(steps, list) or not steps:
            errors.append(f"{prefix}: 'ordered_sequence' requires non-empty steps list")
        else:
            for idx, st in enumerate(steps):
                if not isinstance(st, dict) or st.get("tool_name") not in known_tools:
                    errors.append(f"{prefix}.steps[{idx}]: unknown tool_name '{(st or {}).get('tool_name') if isinstance(st, dict) else st}'")
    elif mode == "world_state":
        if not isinstance(check.get("expected_state_subset"), dict) or not check.get("expected_state_subset"):
            errors.append(f"{prefix}: 'world_state' requires non-empty expected_state_subset dict")


def validate_pair(scenario: dict[str, Any], gt: dict[str, Any]) -> list[str]:
    """Validate one (scenario, gt) pair and return a list of error messages."""
    errors: list[str] = []
    sc_id = scenario.get("scenario_id", "<unknown>")
    if scenario.get("scenario_id") != gt.get("scenario_id"):
        errors.append(f"scenario_id mismatch: {sc_id!r} != {gt.get('scenario_id')!r}")
    if scenario.get("schema_version") not in SUPPORTED_SCHEMA_VERSIONS:
        errors.append(f"Unsupported scenario schema_version: {scenario.get('schema_version')!r}")
    if gt.get("schema_version") not in SUPPORTED_SCHEMA_VERSIONS:
        errors.append(f"Unsupported gt schema_version: {gt.get('schema_version')!r}")

    meta = scenario.get("original_assetops_meta")
    if isinstance(meta, dict) and ({k.lower() for k in meta} & FORBIDDEN_META_KEYS):
        errors.append("original_assetops_meta contains forbidden spoiler/answer keys")

    asset_map = get_asset_map(scenario)
    if not asset_map:
        errors.append("Missing asset_info / assets")
    for aid, asset in asset_map.items():
        if {k.lower() for k in asset} & FORBIDDEN_ASSET_KEYS:
            errors.append(f"Asset '{aid}' contains forbidden spoiler keys")
        for metric, bounds in (asset.get("normal_ranges") or {}).items():
            if not isinstance(bounds, list) or len(bounds) != 2 or float(bounds[0]) > float(bounds[1]):
                errors.append(f"Asset '{aid}' metric '{metric}' has invalid normal_ranges {bounds!r}")

    ticks = scenario.get("replay_ticks") or []
    if not ticks:
        errors.append("replay_ticks must be non-empty")
    tick_dts = []
    for i, t in enumerate(ticks):
        try:
            dt = parse_iso_utc(t)
            if tick_dts and dt <= tick_dts[-1]:
                errors.append(f"replay_ticks not strictly ascending at index {i}: {t}")
            tick_dts.append(dt)
        except Exception as exc:
            errors.append(f"Invalid replay_tick '{t}': {exc}")

    obs_ids: set[str] = set()
    for o in scenario.get("observations") or []:
        oid = o.get("obs_id")
        if not oid or oid in obs_ids:
            errors.append(f"Duplicate or missing obs_id: {oid!r}")
        obs_ids.add(str(oid))
        if o.get("modality") not in VALID_MODALITIES:
            errors.append(f"Observation '{oid}' has invalid modality {o.get('modality')!r}")
        try:
            parse_iso_utc(o["release_time"])
        except Exception as exc:
            errors.append(f"Observation '{oid}' has invalid release_time: {exc}")
        payload = o.get("payload") or {}
        leaked = {k.lower() for k in payload} & FORBIDDEN_LABEL_COLUMNS
        if leaked:
            errors.append(f"Observation '{oid}' payload leaks forbidden label keys: {sorted(leaked)}")

    sys_msg = build_system_message(scenario)["content"]
    for key in SPOILER_PROMPT_KEYS:
        if key in sys_msg:
            errors.append(f"messages[0] leaks spoiler key '{key}'")

    custom_names: set[str] = set()
    for ct in scenario.get("custom_tools") or []:
        cname = ct.get("name")
        if not cname or cname in custom_names or cname in BUILTIN_TOOLS:
            errors.append(f"Invalid or duplicate custom_tool name '{cname}'")
        custom_names.add(str(cname))
        if ct.get("decision") not in VALID_DECISIONS - {"wait"}:
            errors.append(f"custom_tool '{cname}' has invalid decision {ct.get('decision')!r}")
    known_tools = BUILTIN_TOOLS | custom_names

    perms = scenario.get("permissions")
    if not isinstance(perms, dict):
        errors.append("permissions must be a dict")
    else:
        allowed_decs = perms.get("allowed_decisions")
        if not isinstance(allowed_decs, list) or not allowed_decs or any(d not in VALID_DECISIONS for d in allowed_decs):
            errors.append(f"permissions.allowed_decisions invalid: {allowed_decs!r}")
        forbidden_tools = set(perms.get("forbidden_tools") or [])
        valid_perm_tools = known_tools | forbidden_tools
        for tool in perms.get("approval_required_tools") or []:
            if tool not in valid_perm_tools:
                errors.append(f"permissions.approval_required_tools unknown tool '{tool}'")
        for p_key in (perms.get("approval_policy") or {}):
            if p_key != "*" and p_key not in valid_perm_tools:
                errors.append(f"permissions.approval_policy unknown tool key '{p_key}'")

    tick_set = set(ticks)
    for nat in gt.get("no_action_ticks") or []:
        if nat not in tick_set:
            errors.append(f"no_action_tick '{nat}' not in replay_ticks")

    for need in gt.get("needs") or []:
        nid = need.get("need_id", "<need>")
        if need.get("asset_id") not in asset_map:
            errors.append(f"Need '{nid}' references unknown asset_id '{need.get('asset_id')}'")
        try:
            vis = parse_iso_utc(need["visible_from"])
            ws = parse_iso_utc(need["valid_window"]["start"])
            we = parse_iso_utc(need["valid_window"]["end"])
            if not (vis <= ws <= we):
                errors.append(f"Need '{nid}' violates visible_from <= valid_window.start <= valid_window.end")
        except Exception as exc:
            errors.append(f"Need '{nid}' has invalid window timestamps: {exc}")
        for oid in need.get("supporting_obs_ids") or []:
            if oid not in obs_ids:
                errors.append(f"Need '{nid}' supporting_obs_id '{oid}' not in observations")
        for act in need.get("allowed_actions") or []:
            if act.get("decision") not in VALID_DECISIONS - {"wait"}:
                errors.append(f"Need '{nid}' allowed_action invalid decision '{act.get('decision')}'")
            if act.get("tool_name") not in known_tools:
                errors.append(f"Need '{nid}' allowed_action unknown tool '{act.get('tool_name')}'")
        _validate_completion_check(need.get("completion_check"), known_tools, errors, f"Need '{nid}'")
    return errors


def validate_directory(scenarios_dir: Path, gt_dir: Path, out_path: Path | None = None) -> dict[str, Any]:
    """Validate all scenario/GT files across directories."""
    sc_files = {p.name.replace(".scenario.json", ""): p for p in sorted(scenarios_dir.glob("*.scenario.json"))}
    gt_files = {p.name.replace(".gt.json", ""): p for p in sorted(gt_dir.glob("*.gt.json"))}
    errors: list[dict[str, str]] = []

    for sc_id in sorted(set(sc_files) - set(gt_files)):
        errors.append({"scenario_id": sc_id, "message": f"Missing GT file for scenario '{sc_id}'"})
    for sc_id in sorted(set(gt_files) - set(sc_files)):
        errors.append({"scenario_id": sc_id, "message": f"Missing scenario file for GT '{sc_id}'"})

    paired = sorted(set(sc_files) & set(gt_files))
    for sc_id in paired:
        sc_obj = json.loads(sc_files[sc_id].read_text(encoding="utf-8"))
        gt_obj = json.loads(gt_files[sc_id].read_text(encoding="utf-8"))
        for msg in validate_pair(sc_obj, gt_obj):
            errors.append({"scenario_id": sc_id, "message": msg})

    report = {
        "valid": len(errors) == 0,
        "checked_scenarios": len(paired),
        "error_count": len(errors),
        "errors": errors,
    }
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Validate scenarios/public and scenarios/gt integrity and zero-leakage.")
    p.add_argument("--scenarios-dir", required=True, type=Path)
    p.add_argument("--gt-dir", required=True, type=Path)
    p.add_argument("--out", default=None, type=Path)
    a = p.parse_args(argv)
    report = validate_directory(a.scenarios_dir, a.gt_dir, a.out)
    print(json.dumps(report, ensure_ascii=False))
    if not report["valid"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
