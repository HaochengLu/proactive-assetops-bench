"""CLI 1: Wide/long CSV converter with external catalogs, tick_grid, and zero-leakage checks."""

from __future__ import annotations

import argparse
import csv
import json
import math
from datetime import timedelta, timezone
from pathlib import Path
from typing import Any

from framework import SCHEMA_VERSION, VALID_MODALITIES, parse_iso_utc

FORBIDDEN_LABEL_COLUMNS = {
    "label", "anomaly", "is_anomaly", "anomaly_label", "fault",
    "fault_type", "fault_mode", "gt", "ground_truth", "target", "state_label",
}
DEFAULT_TIMESTAMP_COLS = ("timestamp", "Timestamp", "datetime", "DateTime", "time")
DEFAULT_METRIC_NAME_COLS = ("metric_name", "metric", "sensor", "tag")
DEFAULT_METRIC_VAL_COLS = ("metric_value", "value", "reading")


def _parse_payload_value(raw: str) -> float | bool | str:
    s = raw.strip()
    try:
        val = float(s)
        if math.isnan(val) or math.isinf(val):
            raise ValueError(f"Non-finite numeric value: {raw!r}")
        return val
    except ValueError as exc:
        if "Non-finite" in str(exc):
            raise
    low = s.lower()
    if low == "true":
        return True
    if low == "false":
        return False
    return s


def _pick_col(fields: list[str], configured: str | None, fallbacks: tuple[str, ...], csv_path: Path, required: bool = True) -> str | None:
    if configured:
        if configured not in fields:
            raise ValueError(f"Configured column '{configured}' not found in {csv_path}")
        return configured
    found = next((c for c in fallbacks if c in fields), None)
    if required and not found:
        raise ValueError(f"Required column from {fallbacks} not found in {csv_path}; headers={fields}")
    return found


def _load_telemetry_csv(
    csv_path: Path, default_asset_id: str, convert_cfg: dict[str, Any]
) -> list[dict[str, Any]]:
    """Parse wide or long telemetry.csv with flexible mapping and mandatory label stripping."""
    if not csv_path.exists():
        return []
    csv_format = convert_cfg.get("csv_format", "wide")
    if csv_format not in {"wide", "long"}:
        raise ValueError(f"Unsupported csv_format '{csv_format}'")

    col_map = convert_cfg.get("column_mapping") or {}
    metrics_map: dict[str, str] = col_map.get("metrics") or {}
    strip_set = {c.strip().lower() for c in (convert_cfg.get("strip_columns") or [])}
    whitelist = set(convert_cfg.get("sensor_columns") or [])

    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames or []
        ts_col = _pick_col(fields, col_map.get("timestamp"), DEFAULT_TIMESTAMP_COLS, csv_path)
        assert ts_col is not None
        asset_col = col_map.get("asset_id", "asset_id")
        src_type_col = col_map.get("source_type", "source_type")
        src_ref_col = col_map.get("source_ref", "source_ref")

        if csv_format == "wide":
            meta_cols = {ts_col, asset_col, src_type_col, src_ref_col}
            obs_list: list[dict[str, Any]] = []
            for row in reader:
                payload: dict[str, Any] = {}
                for orig_col, raw_val in row.items():
                    if orig_col in meta_cols or raw_val is None or raw_val.strip() == "":
                        continue
                    mapped_col = metrics_map.get(orig_col, orig_col)
                    if (
                        orig_col.strip().lower() in FORBIDDEN_LABEL_COLUMNS
                        or mapped_col.strip().lower() in FORBIDDEN_LABEL_COLUMNS
                        or orig_col.strip().lower() in strip_set
                        or mapped_col.strip().lower() in strip_set
                    ):
                        continue
                    if whitelist and orig_col not in whitelist and mapped_col not in whitelist:
                        continue
                    payload[mapped_col] = _parse_payload_value(raw_val)
                obs: dict[str, Any] = {
                    "release_time": row[ts_col].strip(),
                    "asset_id": (row.get(asset_col) or "").strip() or default_asset_id,
                    "modality": "telemetry",
                    "source_type": (row.get(src_type_col) or "").strip() or "adapted",
                    "payload": payload,
                }
                if (row.get(src_ref_col) or "").strip():
                    obs["source_ref"] = row[src_ref_col].strip()
                obs_list.append(obs)
            return obs_list

        # Long format: group rows by (release_time, asset_id, source_type, source_ref)
        m_name_col = _pick_col(fields, col_map.get("metric_name"), DEFAULT_METRIC_NAME_COLS, csv_path)
        m_val_col = _pick_col(fields, col_map.get("metric_value"), DEFAULT_METRIC_VAL_COLS, csv_path)
        assert m_name_col is not None and m_val_col is not None
        grouped: dict[tuple[str, str, str, str], dict[str, Any]] = {}
        for row in reader:
            rel_time = row[ts_col].strip()
            aid = (row.get(asset_col) or "").strip() or default_asset_id
            src_type = (row.get(src_type_col) or "").strip() or "adapted"
            src_ref = (row.get(src_ref_col) or "").strip()
            key = (rel_time, aid, src_type, src_ref)
            if key not in grouped:
                obs = {"release_time": rel_time, "asset_id": aid, "modality": "telemetry", "source_type": src_type, "payload": {}}
                if src_ref:
                    obs["source_ref"] = src_ref
                grouped[key] = obs
            orig_m = (row.get(m_name_col) or "").strip()
            raw_v = row.get(m_val_col)
            if not orig_m or raw_v is None or raw_v.strip() == "":
                continue
            mapped_m = metrics_map.get(orig_m, orig_m)
            if (
                orig_m.lower() in FORBIDDEN_LABEL_COLUMNS
                or mapped_m.lower() in FORBIDDEN_LABEL_COLUMNS
                or orig_m.lower() in strip_set
                or mapped_m.lower() in strip_set
            ):
                continue
            if whitelist and orig_m not in whitelist and mapped_m not in whitelist:
                continue
            grouped[key]["payload"][mapped_m] = _parse_payload_value(raw_v)
        return list(grouped.values())


def _generate_tick_grid(tick_grid: dict[str, Any], observations: list[dict[str, Any]]) -> list[str]:
    step_sec = int(tick_grid["step_seconds"])
    if step_sec <= 0:
        raise ValueError("tick_grid.step_seconds must be > 0")
    step = timedelta(seconds=step_sec)
    if tick_grid.get("auto_from_observations"):
        if not observations:
            raise ValueError("Cannot auto-generate tick_grid from empty observations")
        dts = [parse_iso_utc(o["release_time"]).astimezone(timezone.utc) for o in observations]
        cur, max_dt = min(dts), max(dts)
        ticks: list[str] = []
        last_dt = cur
        while cur <= max_dt:
            ticks.append(cur.strftime("%Y-%m-%dT%H:%M:%SZ"))
            last_dt = cur
            cur += step
        if last_dt < max_dt:
            ticks.append((last_dt + step).strftime("%Y-%m-%dT%H:%M:%SZ"))
        return ticks

    cur = parse_iso_utc(tick_grid["start"]).astimezone(timezone.utc)
    end_dt = parse_iso_utc(tick_grid["end"]).astimezone(timezone.utc)
    ticks = []
    while cur <= end_dt:
        ticks.append(cur.strftime("%Y-%m-%dT%H:%M:%SZ"))
        cur += step
    return ticks


def convert_scenario(scenario_dir: Path, out_dir: Path) -> tuple[Path, Path]:
    """Convert one raw scenario folder into public .scenario.json and hidden .gt.json."""
    spec = json.loads((scenario_dir / "raw_task_spec.json").read_text(encoding="utf-8"))
    pub = dict(spec["public_spec"])
    gt_spec = dict(spec["gt_spec"])
    convert_cfg = spec.get("convert_config") or pub.pop("convert_config", None) or {}
    pub.pop("convert_config", None)
    tick_grid = pub.pop("tick_grid", None) or convert_cfg.get("tick_grid")
    scenario_id: str = pub["scenario_id"]

    raw_obs = _load_telemetry_csv(
        scenario_dir / "telemetry.csv", pub["asset_info"]["asset_id"], convert_cfg
    ) + [dict(e) for e in (pub.pop("extra_events", []) or [])]

    indexed = sorted(enumerate(raw_obs), key=lambda item: (parse_iso_utc(item[1]["release_time"]), item[0]))
    observations: list[dict[str, Any]] = []
    for idx, (_, item) in enumerate(indexed, start=1):
        modality = item.get("modality", "telemetry")
        payload = item["payload"]
        if modality not in VALID_MODALITIES:
            raise ValueError(f"Invalid modality '{modality}' in {scenario_id}")
        leaked = {k.lower() for k in payload} & FORBIDDEN_LABEL_COLUMNS
        if leaked:
            raise ValueError(f"Zero-leakage assertion failed in {scenario_id}: {sorted(leaked)}")
        obs: dict[str, Any] = {
            "obs_id": f"{scenario_id}:obs:{idx:04d}",
            "release_time": item["release_time"],
            "asset_id": item["asset_id"],
            "modality": modality,
            "source_type": item.get("source_type", "adapted"),
        }
        if item.get("source_ref"):
            obs["source_ref"] = item["source_ref"]
        obs["payload"] = payload
        observations.append(obs)

    replay_ticks = pub.get("replay_ticks") or (_generate_tick_grid(tick_grid, observations) if tick_grid else [])
    if not replay_ticks:
        raise ValueError(f"Missing replay_ticks or tick_grid in {scenario_id}")

    tool_catalog = dict(pub.get("tool_catalog_data") or {})
    if (scenario_dir / "fmea_manuals.json").exists():
        ext_fmea = json.loads((scenario_dir / "fmea_manuals.json").read_text(encoding="utf-8"))
        tool_catalog["fmea_manuals"] = [*tool_catalog.get("fmea_manuals", []), *ext_fmea]
    if (scenario_dir / "workorders.json").exists():
        ext_wo = json.loads((scenario_dir / "workorders.json").read_text(encoding="utf-8"))
        tool_catalog["historical_workorders"] = [*tool_catalog.get("historical_workorders", []), *ext_wo]

    scenario_obj: dict[str, Any] = {"schema_version": SCHEMA_VERSION, "scenario_id": scenario_id}
    for opt_key in ("tags", "original_assetops_meta"):
        if opt_key in pub:
            scenario_obj[opt_key] = pub[opt_key]
    scenario_obj["asset_info"] = pub["asset_info"]
    if pub.get("assets"):
        scenario_obj["assets"] = pub["assets"]
    for opt_key in ("extra_fault_taxonomy", "custom_tools"):
        if opt_key in pub:
            scenario_obj[opt_key] = pub[opt_key]
    scenario_obj.update({
        "persistent_goals": pub["persistent_goals"],
        "permissions": pub["permissions"],
        "replay_ticks": replay_ticks,
        "observations": observations,
        "tool_catalog_data": tool_catalog,
    })

    resolved_needs: list[dict[str, Any]] = []
    for need in gt_spec.get("needs", []):
        need_out = dict(need)
        supporting_ts = set(need_out.pop("supporting_timestamps", []) or [])
        if supporting_ts:
            need_out["supporting_obs_ids"] = [
                o["obs_id"] for o in observations
                if o["release_time"] in supporting_ts and o["asset_id"] == need_out["asset_id"]
            ]
        elif not need_out.get("supporting_obs_ids"):
            vis_dt, end_dt = parse_iso_utc(need_out["visible_from"]), parse_iso_utc(need_out["valid_window"]["end"])
            need_out["supporting_obs_ids"] = [
                o["obs_id"] for o in observations
                if o["asset_id"] == need_out["asset_id"] and vis_dt <= parse_iso_utc(o["release_time"]) <= end_dt
            ]
        resolved_needs.append(need_out)

    gt_obj: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "scenario_id": scenario_id,
        "no_action_ticks": gt_spec.get("no_action_ticks", []),
        "needs": resolved_needs,
    }
    if "allow_readonly_query_in_no_action_ticks" in gt_spec:
        gt_obj["allow_readonly_query_in_no_action_ticks"] = bool(gt_spec["allow_readonly_query_in_no_action_ticks"])

    pub_path = out_dir / "public" / f"{scenario_id}.scenario.json"
    gt_path = out_dir / "gt" / f"{scenario_id}.gt.json"
    pub_path.parent.mkdir(parents=True, exist_ok=True)
    gt_path.parent.mkdir(parents=True, exist_ok=True)
    pub_path.write_text(json.dumps(scenario_obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    gt_path.write_text(json.dumps(gt_obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return pub_path, gt_path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Convert raw_data into public scenarios and hidden GT.")
    parser.add_argument("--raw-dir", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    args = parser.parse_args(argv)
    for entry in sorted(args.raw_dir.iterdir()):
        if entry.is_dir() and (entry / "raw_task_spec.json").exists():
            convert_scenario(entry, args.out_dir)


if __name__ == "__main__":
    main()
