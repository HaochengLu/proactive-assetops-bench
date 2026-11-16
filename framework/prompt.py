"""Zero-spoiler dynamic prompt and strict role-alternating, prefix-cache-invariant messages."""

from __future__ import annotations

import json
from typing import Any

PLACEHOLDER_ASSISTANT_JSON = (
    '{"decision":"wait","reason":"prev_turn_api_error",'
    '"evidence_obs_ids":[],"reported_needs":[],"tool_call":null}'
)

GLOBAL_FAULT_TAXONOMY = [
    "evaporator_waterside_fouling",
    "condenser_tube_scaling",
    "refrigerant_undercharge",
    "chilled_water_flow_restriction",
    "compressor_surge",
    "bearing_wear_overheat",
    "pump_cavitation",
    "impeller_erosion",
    "seal_leakage",
    "ahu_damper_stuck",
    "cooling_coil_valve_leak",
    "supply_fan_belt_slip",
    "filter_clogging_pressure_drop",
    "cooling_tower_fan_imbalance",
    "cooling_tower_fill_fouling",
    "motor_stator_winding_overheat",
    "lubrication_oil_degradation",
    "sensor_calibration_drift",
    "sensor_frozen_flatline",
    "thermostat_setpoint_override",
]

GLOBAL_SYSTEM_PREFIX = (
    "You are the fixed Architect agent in Proactive AssetOpsBench.\n"
    "At each replay tick/step, inspect only released observations and prior tool results, "
    "decide whether to wait or take one action, and respond with a single JSON object matching:\n"
    '{"decision": "wait|query|investigate|request_approval|execute", '
    '"reason": "<brief evidence summary>", "evidence_obs_ids": ["<obs_id>"], '
    '"reported_needs": [{"asset_id": "<id>", "fault_type": "<fault_type>", "summary": "<text>"}], '
    '"tool_call": {"name": "<tool_name>", "args": {...}} | null}\n'
    "Available built-in tools (plus any scenario custom_tools in SCENARIO_CONTEXT_JSON):\n"
    "1. query_iot_metadata(asset_id: str | null=null) -> [decision: query]\n"
    "2. query_iot_history(asset_id: str, limit: int=5, modality: str=\"all\", metrics: list[str] | null=null, start_time: str | null=null, aggregation: str=\"raw\") -> [decision: query]\n"
    "3. check_tsfm_anomaly(asset_id: str, metric: str, window: int=8) -> [decision: investigate]\n"
    "4. forecast_tsfm_metric(asset_id: str, metric: str, horizon_minutes: float=30.0, window: int=8) -> [decision: investigate]\n"
    "5. lookup_fmea_manual(asset_type: str | null=null, keyword: str | null=null, fault_type: str | null=null) -> [decision: investigate]\n"
    "6. query_workorder_history(asset_id: str | null=null, status: str=\"all\", limit: int=10) -> [decision: query]\n"
    "7. update_workorder(workorder_id: str, status: str | null=null, note: str=\"\", priority: str | null=null) -> [decision: execute]\n"
    "8. create_maintenance_workorder(asset_id: str, fault_type: str, summary: str) -> [decision: execute]\n"
    "9. create_urgent_shutdown_workorder(asset_id: str, fault_type: str, summary: str) -> [decision: execute]\n"
    "10. submit_approval_request(asset_id: str, target_tool: str, reason: str) -> [decision: request_approval]\n"
    "Rules: stay in 'wait' (tool_call=null) when metrics are normal, a maintenance_log explains a "
    "planned transient, or check_tsfm_anomaly shows transient_spike; use forecast_tsfm_metric on "
    "monotonic trends approaching bounds; verify FMEA via lookup_fmea_manual; check open workorders "
    "before creating duplicates; if 'execute' is disallowed, report needs in 'wait' without executing; "
    "cite released obs_ids; obey permissions and prerequisites; output raw JSON only."
)


def build_system_message(scenario: dict[str, Any]) -> dict[str, str]:
    """Build zero-spoiler, byte-invariant messages[0] without FMEA thresholds or workorders."""
    custom_tools = [
        {k: t[k] for k in ("name", "decision", "description", "required_args", "terminal", "state_effects") if k in t}
        for t in (scenario.get("custom_tools") or [])
        if isinstance(t, dict)
    ]
    taxonomy = sorted(set(GLOBAL_FAULT_TAXONOMY) | set(scenario.get("extra_fault_taxonomy") or []))
    header = {
        "scenario_id": scenario["scenario_id"],
        "asset_info": scenario["asset_info"],
        "assets": scenario.get("assets") or [scenario["asset_info"]],
        "custom_tools": custom_tools,
        "persistent_goals": scenario["persistent_goals"],
        "permissions": scenario["permissions"],
        "fault_taxonomy": taxonomy,
    }
    return {
        "role": "system",
        "content": f"{GLOBAL_SYSTEM_PREFIX}\n\nSCENARIO_CONTEXT_JSON:\n{json.dumps(header, ensure_ascii=False, sort_keys=True)}",
    }


def format_user_turn(
    turn_idx: int,
    step_idx: int,
    replay_time: str,
    new_observations: list[dict[str, Any]],
    prev_tool_result: dict[str, Any] | None = None,
) -> str:
    """Format one user step with optional prior synchronous tool_result and new observations."""
    parts: list[str] = []
    if prev_tool_result is not None:
        parts.append(f"PREV_TOOL_RESULT_JSON:\n{json.dumps(prev_tool_result, ensure_ascii=False, sort_keys=True)}")
    turn_payload = {
        "new_observations": new_observations,
        "replay_time": replay_time,
        "step_idx": step_idx,
        "turn_idx": turn_idx,
    }
    parts.append(f"TURN_OBSERVATIONS_JSON:\n{json.dumps(turn_payload, ensure_ascii=False, sort_keys=True)}")
    return "\n".join(parts)


def build_initial_messages(
    scenario: dict[str, Any], replay_time: str, new_observations: list[dict[str, Any]]
) -> list[dict[str, str]]:
    """Construct seq_idx=0 messages: [system, user]."""
    return [build_system_message(scenario), {"role": "user", "content": format_user_turn(0, 0, replay_time, new_observations)}]


def append_turn_messages(
    prev_messages: list[dict[str, str]],
    prev_raw_response: str | None,
    prev_tool_result: dict[str, Any] | None,
    turn_idx: int,
    step_idx: int,
    replay_time: str,
    new_observations: list[dict[str, Any]],
) -> list[dict[str, str]]:
    """Strictly append [assistant, user] without mutating prev_messages (prefix invariance)."""
    assistant_text = prev_raw_response if (prev_raw_response and prev_raw_response.strip()) else PLACEHOLDER_ASSISTANT_JSON
    return [
        *prev_messages,
        {"role": "assistant", "content": assistant_text},
        {"role": "user", "content": format_user_turn(turn_idx, step_idx, replay_time, new_observations, prev_tool_result)},
    ]
