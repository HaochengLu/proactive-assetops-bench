"""Comprehensive unittest suite covering v2.0 backward compatibility and all v3.0 AssetOpsBench cases."""

from __future__ import annotations

import json
import socket
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from framework.client import VLLMClient, parse_action_json
from framework.compare import compare_runs
from framework.convert import FORBIDDEN_LABEL_COLUMNS, convert_scenario
from framework.export import export_logs
from framework.prompt import GLOBAL_FAULT_TAXONOMY, PLACEHOLDER_ASSISTANT_JSON, build_system_message
from framework.replay import run_benchmark
from framework.run import run_replay
from framework.score import score_run, score_scenario
from framework.tools import ToolExecutor
from framework.validate import validate_directory, validate_pair

REPO_ROOT = Path(__file__).resolve().parent.parent


class _CountingHTTPServer(ThreadingHTTPServer):
    def __init__(self, server_address: tuple[str, int], RequestHandlerClass: type[BaseHTTPRequestHandler]) -> None:
        self.accepted_connections = 0
        super().__init__(server_address, RequestHandlerClass)

    def get_request(self) -> tuple[Any, Any]:
        conn, addr = super().get_request()
        self.accepted_connections += 1
        return conn, addr


class _MockVLLMHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    scenarios: list[dict[str, Any]] = []
    call_idx: int = 0
    captured_requests: list[dict[str, Any]] = []
    lock = threading.Lock()
    replay_events: list[tuple[str, int, int, int]] = []
    replay_delay_sec: float = 0.0

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        return

    def do_POST(self) -> None:  # noqa: N802
        start_ns = time.perf_counter_ns()
        length = int(self.headers.get("Content-Length", "0"))
        body = json.loads(self.rfile.read(length).decode("utf-8"))

        if type(self).replay_delay_sec > 0:
            time.sleep(type(self).replay_delay_sec)
            last_user = body["messages"][-1]["content"]
            turn_marker = "TURN_OBSERVATIONS_JSON:\n"
            t_idx = last_user.find(turn_marker)
            turn_info = json.loads(last_user[t_idx + len(turn_marker) :].split("\n", 1)[0])
            sys_marker = "SCENARIO_CONTEXT_JSON:\n"
            s_idx = body["messages"][0]["content"].find(sys_marker)
            sc_id = json.loads(body["messages"][0]["content"][s_idx + len(sys_marker) :].split("\n", 1)[0])["scenario_id"]
            end_ns = time.perf_counter_ns()
            with type(self).lock:
                type(self).replay_events.append((sc_id, int(turn_info["turn_idx"]) * 10 + int(turn_info["step_idx"]), start_ns, end_ns))

            act = json.dumps({"decision": "wait", "reason": "ok", "evidence_obs_ids": [], "reported_needs": [], "tool_call": None})
            payload = (
                f'data: {json.dumps({"choices": [{"delta": {"content": act}}]})}\n\n'
                f'data: {json.dumps({"choices": [], "usage": {"prompt_tokens": 100, "completion_tokens": 15, "prompt_tokens_details": {"cached_tokens": 60}}})}\n\n'
                "data: [DONE]\n\n"
            ).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            self.wfile.flush()
            return

        with type(self).lock:
            type(self).captured_requests.append(body)
            idx = type(self).call_idx
            type(self).call_idx += 1
            spec = type(self).scenarios[idx]

        if spec.get("drop_socket"):
            self.close_connection = True
            try:
                self.request.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            self.request.close()
            return
        if spec.get("delay_sec"):
            time.sleep(spec["delay_sec"])
        if spec.get("http_status", 200) != 200:
            err = b'{"error": "server error"}'
            self.send_response(spec["http_status"])
            self.send_header("Content-Length", str(len(err)))
            self.end_headers()
            self.wfile.write(err)
            return

        chunks_bytes = b"".join(f"data: {json.dumps(c, ensure_ascii=False)}\n\n".encode("utf-8") for c in spec["chunks"]) + b"data: [DONE]\n\n"
        try:
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Content-Length", str(len(chunks_bytes)))
            self.end_headers()
            if spec.get("chunk_delay_sec"):
                for c in spec["chunks"]:
                    self.wfile.write(f"data: {json.dumps(c, ensure_ascii=False)}\n\n".encode("utf-8"))
                    self.wfile.flush()
                    time.sleep(spec["chunk_delay_sec"])
                self.wfile.write(b"data: [DONE]\n\n")
                self.wfile.flush()
            else:
                self.wfile.write(chunks_bytes)
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            self.close_connection = True


class TestPipelineV3(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self.tmp.name)
        self.sc_dir = self.tmp_path / "scenarios"
        for sub in sorted((REPO_ROOT / "raw_data").iterdir()):
            if sub.is_dir() and (sub / "raw_task_spec.json").exists():
                convert_scenario(sub, self.sc_dir)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def _read_jsonl(self, path: Path) -> list[dict[str, Any]]:
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]

    def test_v2_compatibility_baseline_and_tags_filter(self) -> None:
        val_rep = validate_directory(self.sc_dir / "public", self.sc_dir / "gt")
        self.assertTrue(val_rep["valid"])
        self.assertEqual(val_rep["checked_scenarios"], 2)

        # Verify 20 taxonomy codes & zero spoiler in messages[0]
        t1_sc = json.loads((self.sc_dir / "public" / "task_01_chiller6.scenario.json").read_text(encoding="utf-8"))
        sys_txt = build_system_message(t1_sc)["content"]
        self.assertEqual(len(GLOBAL_FAULT_TAXONOMY), 20)
        for key in ("fmea_manuals", "historical_workorders", "symptom_metric", "threshold_high", "recommended_tool"):
            self.assertNotIn(key, sys_txt)

        # Run via --scenarios-dir + --tags governance (filters to task_02_pump3 only)
        gov_dir = self.tmp_path / "out_gov"
        gov_log, _ = run_replay(None, gov_dir, scenarios_dir=self.sc_dir / "public", tags=["governance"])
        gov_rows = self._read_jsonl(gov_log)
        self.assertEqual({r["scenario_id"] for r in gov_rows}, {"task_02_pump3"})

        # Full run on both scenarios
        out_dir = self.tmp_path / "out_base"
        log_path, summary_path = run_replay(None, out_dir, scenarios_dir=self.sc_dir / "public", backend="mock")
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        self.assertEqual(summary["schema_version"], "3.0")
        self.assertIn("final_world_state", summary["per_scenario"]["task_01_chiller6"])
        self.assertTrue(summary["per_scenario"]["task_02_pump3"]["final_world_state"]["assets"]["pump_3"]["urgent_shutdown"])

        report = score_run(self.sc_dir / "public", self.sc_dir / "gt", log_path, out_dir / "score.json")
        self.assertEqual(report["overall"]["micro_need_discovery_f1"], 1.0)
        self.assertEqual(report["overall"]["total_unnecessary_turns"], 0)
        self.assertEqual(report["overall"]["timely_count"], 2)
        self.assertEqual(report["overall"]["verified_count"], 2)
        self.assertEqual(report["overall"]["unauthorized_count"], 0)

        # Verify eager_unauth still catches early/unnecessary/unauthorized on task_02_pump3
        eager_dir = self.tmp_path / "out_eager"
        eager_log, _ = run_replay(None, eager_dir, scenarios_dir=self.sc_dir / "public", backend="mock", mock_mode="eager_unauth")
        eager_rep = score_run(self.sc_dir / "public", self.sc_dir / "gt", eager_log, eager_dir / "score.json")
        self.assertEqual(eager_rep["per_scenario"]["task_02_pump3"]["timely_valid_interventions"][0]["status"], "early")
        self.assertGreater(eager_rep["per_scenario"]["task_02_pump3"]["unauthorized_attempts"]["count"], 0)

    def test_case1_iot_metadata_and_windowed_aggregation(self) -> None:
        t1_sc = json.loads((self.sc_dir / "public" / "task_01_chiller6.scenario.json").read_text(encoding="utf-8"))
        ex = ToolExecutor(t1_sc)

        meta_all = ex.execute({"decision": "query", "tool_call": {"name": "query_iot_metadata", "args": {}}}, "2026-05-10T08:15:00Z", 1)
        self.assertEqual(meta_all["output"]["count"], 2)
        self.assertEqual(meta_all["output"]["assets"][0]["sensor_units"]["evap_approach_temp_c"], "C")
        self.assertEqual(ex.execute({"decision": "query", "tool_call": {"name": "query_iot_metadata", "args": {"asset_id": "missing"}}}, "2026-05-10T08:15:00Z", 1)["status"], "tool_error")

        # Raw with metrics projection & start_time
        raw_proj = ex.execute(
            {"decision": "query", "tool_call": {"name": "query_iot_history", "args": {"asset_id": "chiller_6", "start_time": "2026-05-10T08:40:00Z", "metrics": ["evap_approach_temp_c"]}}},
            "2026-05-10T08:45:00Z", 3,
        )
        self.assertEqual(raw_proj["output"]["count"], 2)
        self.assertEqual(set(raw_proj["output"]["records"][0]["payload"].keys()), {"evap_approach_temp_c"})

        # Mean/max aggregation over [start_time, replay_time] without explicit limit
        agg_mean = ex.execute(
            {"decision": "query", "tool_call": {"name": "query_iot_history", "args": {"asset_id": "chiller_6", "modality": "telemetry", "start_time": "2026-05-10T08:40:00Z", "aggregation": "mean"}}},
            "2026-05-10T08:45:00Z", 3,
        )
        self.assertEqual(agg_mean["output"]["sample_count"], 2)
        self.assertEqual(agg_mean["output"]["metrics"]["evap_approach_temp_c"], 3.4)
        agg_max = ex.execute(
            {"decision": "query", "tool_call": {"name": "query_iot_history", "args": {"asset_id": "chiller_6", "modality": "telemetry", "start_time": "2026-05-10T08:40:00Z", "aggregation": "max"}}},
            "2026-05-10T08:45:00Z", 3,
        )
        self.assertEqual(agg_max["output"]["metrics"]["evap_approach_temp_c"], 3.5)

    def test_case2_tsfm_forecast_and_diagnosis_only_and_case3_freetext_fmsr(self) -> None:
        pub_dir, gt_dir = self.tmp_path / "p23", self.tmp_path / "g23"
        pub_dir.mkdir()
        gt_dir.mkdir()

        # Sensor stays inside [1.0, 2.5] (1.6 -> 1.9 -> 2.2 -> 2.4) at slope 0.016/min -> 30m pred ~ 2.88 > 2.5!
        # Plus free-text FMEA entry (no symptom_metric!) matching via FMSR keyword search
        sc = {
            "schema_version": "3.0",
            "scenario_id": "task_forecast_diag",
            "asset_info": {"site_id": "p1", "asset_id": "chiller_8", "asset_type": "centrifugal_chiller", "normal_ranges": {"evap_approach_temp_c": [1.0, 2.5]}},
            "assets": [{"site_id": "p1", "asset_id": "chiller_8", "asset_type": "centrifugal_chiller", "normal_ranges": {"evap_approach_temp_c": [1.0, 2.5]}, "predictive_horizon_minutes": 30.0}],
            "persistent_goals": ["Forecast impending fouling and report diagnosis only."],
            "permissions": {"has_permission_constraints": True, "allowed_decisions": ["wait", "query", "investigate"], "approval_required_tools": [], "forbidden_tools": []},
            "replay_ticks": ["2026-08-01T08:00:00Z", "2026-08-01T08:15:00Z", "2026-08-01T08:30:00Z"],
            "observations": [
                {"obs_id": "task_forecast_diag:obs:0001", "release_time": "2026-08-01T08:00:00Z", "asset_id": "chiller_8", "modality": "telemetry", "source_type": "adapted", "payload": {"evap_approach_temp_c": 1.6}},
                {"obs_id": "task_forecast_diag:obs:0002", "release_time": "2026-08-01T08:15:00Z", "asset_id": "chiller_8", "modality": "telemetry", "source_type": "adapted", "payload": {"evap_approach_temp_c": 1.9}},
                {"obs_id": "task_forecast_diag:obs:0003", "release_time": "2026-08-01T08:30:00Z", "asset_id": "chiller_8", "modality": "telemetry", "source_type": "adapted", "payload": {"evap_approach_temp_c": 2.3}},
            ],
            "tool_catalog_data": {"fmea_manuals": [
                {"fault_type": "condenser_tube_scaling", "asset_type": "centrifugal_chiller", "failure_mode": "Condenser Scaling", "symptoms": "Rising condenser approach temp", "recommended_tool": "create_maintenance_workorder"},
                {"fault_type": "evaporator_waterside_fouling", "asset_type": "centrifugal_chiller", "failure_mode": "Evaporator Fouling", "symptoms": "Steadily climbing evap approach temp C", "remediation": "Schedule tube cleaning"},
            ]},
        }
        gt = {
            "schema_version": "3.0",
            "scenario_id": "task_forecast_diag",
            "no_action_ticks": ["2026-08-01T08:00:00Z", "2026-08-01T08:15:00Z"],
            "needs": [{
                "need_id": "need_01", "asset_id": "chiller_8", "fault_type": "evaporator_waterside_fouling", "aliases": [],
                "visible_from": "2026-08-01T08:30:00Z",
                "supporting_obs_ids": ["task_forecast_diag:obs:0001", "task_forecast_diag:obs:0002", "task_forecast_diag:obs:0003"],
                "valid_window": {"start": "2026-08-01T08:30:00Z", "end": "2026-08-01T08:30:00Z"},
                "allowed_actions": [
                    {"decision": "investigate", "tool_name": "forecast_tsfm_metric"},
                    {"decision": "investigate", "tool_name": "lookup_fmea_manual"},
                ],
                "completion_check": {"mode": "diagnosis_only"},
            }],
        }
        sc_p = pub_dir / "task_forecast_diag.scenario.json"
        sc_p.write_text(json.dumps(sc), encoding="utf-8")
        (gt_dir / "task_forecast_diag.gt.json").write_text(json.dumps(gt), encoding="utf-8")
        self.assertTrue(validate_directory(pub_dir, gt_dir)["valid"])

        # Case 3 direct FMSR keyword token search check
        ex = ToolExecutor(sc)
        kw_res = ex.execute({"decision": "investigate", "tool_call": {"name": "lookup_fmea_manual", "args": {"asset_type": "centrifugal_chiller", "keyword": "climbing evap"}}}, "2026-08-01T08:30:00Z", 2)
        self.assertEqual(kw_res["output"]["count"], 1)
        self.assertEqual(kw_res["output"]["manuals"][0]["fault_type"], "evaporator_waterside_fouling")

        out_dir = self.tmp_path / "out_23"
        log_p, _ = run_replay([sc_p], out_dir, backend="mock", mock_mode="heuristic")
        rows = self._read_jsonl(log_p)
        self.assertTrue(any((r.get("action") or {}).get("tool_call", {}) and r["action"]["tool_call"]["name"] == "forecast_tsfm_metric" for r in rows))
        self.assertFalse(any((r.get("action") or {}).get("decision") == "execute" for r in rows))

        rep = score_run(pub_dir, gt_dir, log_p, out_dir / "score.json")
        self.assertEqual(rep["overall"]["micro_need_discovery_f1"], 1.0)
        self.assertEqual(rep["overall"]["timely_count"], 1)
        self.assertEqual(rep["overall"]["verified_count"], 1)
        self.assertEqual(rep["overall"]["unauthorized_count"], 0)

    def test_case4_open_workorder_dedup_update_any_of_and_world_state(self) -> None:
        pub_dir, gt_dir = self.tmp_path / "p4", self.tmp_path / "g4"
        pub_dir.mkdir()
        gt_dir.mkdir()

        sc = {
            "schema_version": "3.0",
            "scenario_id": "task_open_wo",
            "asset_info": {"site_id": "p1", "asset_id": "ahu_5", "asset_type": "air_handling_unit", "normal_ranges": {"filter_dp_pa": [50.0, 250.0]}},
            "assets": [{"site_id": "p1", "asset_id": "ahu_5", "asset_type": "air_handling_unit", "normal_ranges": {"filter_dp_pa": [50.0, 250.0]}}],
            "persistent_goals": ["Check open workorders and update rather than duplicating."],
            "permissions": {"has_permission_constraints": False, "open_workorder_check": True, "allowed_decisions": ["wait", "query", "investigate", "execute"], "approval_required_tools": [], "forbidden_tools": []},
            "replay_ticks": ["2026-08-02T08:00:00Z", "2026-08-02T08:15:00Z", "2026-08-02T08:30:00Z"],
            "observations": [
                {"obs_id": "task_open_wo:obs:0001", "release_time": "2026-08-02T08:00:00Z", "asset_id": "ahu_5", "modality": "telemetry", "source_type": "adapted", "payload": {"filter_dp_pa": 120.0}},
                {"obs_id": "task_open_wo:obs:0002", "release_time": "2026-08-02T08:25:00Z", "asset_id": "ahu_5", "modality": "telemetry", "source_type": "adapted", "payload": {"filter_dp_pa": 310.0}},
                {"obs_id": "task_open_wo:obs:0003", "release_time": "2026-08-02T08:30:00Z", "asset_id": "ahu_5", "modality": "telemetry", "source_type": "adapted", "payload": {"filter_dp_pa": 335.0}},
            ],
            "tool_catalog_data": {
                "fmea_manuals": [{"fault_type": "filter_clogging_pressure_drop", "asset_type": "air_handling_unit", "symptom_metric": "filter_dp_pa", "threshold_high": 280.0, "recommended_tool": "create_maintenance_workorder"}],
                "historical_workorders": [{"workorder_id": "WO-OPEN-77", "asset_id": "ahu_5", "fault_type": "filter_clogging_pressure_drop", "priority": "normal", "status": "open", "created_at": "2026-08-01T12:00:00Z", "summary": "Initial filter check"}],
            },
        }
        gt = {
            "schema_version": "3.0",
            "scenario_id": "task_open_wo",
            "no_action_ticks": ["2026-08-02T08:00:00Z"],
            "needs": [{
                "need_id": "need_01", "asset_id": "ahu_5", "fault_type": "filter_clogging_pressure_drop", "aliases": [],
                "visible_from": "2026-08-02T08:30:00Z",
                "supporting_obs_ids": ["task_open_wo:obs:0002", "task_open_wo:obs:0003"],
                "valid_window": {"start": "2026-08-02T08:30:00Z", "end": "2026-08-02T08:30:00Z"},
                "allowed_actions": [
                    {"decision": "investigate", "tool_name": "check_tsfm_anomaly"},
                    {"decision": "investigate", "tool_name": "lookup_fmea_manual"},
                    {"decision": "query", "tool_name": "query_workorder_history"},
                    {"decision": "execute", "tool_name": "update_workorder"},
                ],
                "completion_check": {
                    "mode": "all_of",
                    "checks": [
                        {"mode": "any_of", "checks": [
                            {"mode": "tool_call", "required_tool": "update_workorder", "required_args_subset": {"workorder_id": "WO-OPEN-77"}},
                            {"mode": "tool_call", "required_tool": "create_maintenance_workorder", "required_args_subset": {"asset_id": "ahu_5"}},
                        ]},
                        {"mode": "world_state", "expected_state_subset": {
                            "assets": {"ahu_5": {"last_workorder_id": "WO-OPEN-77", "last_workorder_action": "updated", "workorder_status": "in_progress"}},
                            "workorders": [{"workorder_id": "WO-OPEN-77", "status": "in_progress", "priority": "high"}],
                        }},
                    ],
                },
            }],
        }
        sc_p = pub_dir / "task_open_wo.scenario.json"
        sc_p.write_text(json.dumps(sc), encoding="utf-8")
        (gt_dir / "task_open_wo.gt.json").write_text(json.dumps(gt), encoding="utf-8")
        self.assertTrue(validate_directory(pub_dir, gt_dir)["valid"])

        out_dir = self.tmp_path / "out_4"
        log_p, _ = run_replay([sc_p], out_dir, backend="mock", max_steps_per_tick=5)
        rows = self._read_jsonl(log_p)
        exec_tools = [r["action"]["tool_call"]["name"] for r in rows if (r.get("action") or {}).get("decision") == "execute"]
        self.assertEqual(exec_tools, ["update_workorder"])
        rep = score_run(pub_dir, gt_dir, log_p, out_dir / "score.json")
        self.assertEqual(rep["overall"]["verified_count"], 1)
        self.assertEqual(rep["overall"]["micro_need_discovery_f1"], 1.0)

    def test_case5_custom_tool_conditional_approval_and_ordered_sequence(self) -> None:
        pub_dir, gt_dir = self.tmp_path / "p5", self.tmp_path / "g5"
        pub_dir.mkdir()
        gt_dir.mkdir()

        def _make_sc(delay_turns: int) -> dict[str, Any]:
            return {
                "schema_version": "3.0",
                "scenario_id": f"task_standby_{delay_turns}",
                "asset_info": {"site_id": "p1", "asset_id": "pump_3", "asset_type": "centrifugal_pump", "normal_ranges": {"vibration_mm_s": [0.5, 4.5]}},
                "assets": [
                    {"site_id": "p1", "asset_id": "pump_3", "asset_type": "centrifugal_pump", "initial_status": "running", "normal_ranges": {"vibration_mm_s": [0.5, 4.5]}},
                    {"site_id": "p1", "asset_id": "pump_standby", "asset_type": "centrifugal_pump", "initial_status": "standby", "normal_ranges": {"vibration_mm_s": [0.0, 4.5]}},
                ],
                "custom_tools": [{
                    "name": "start_standby_pump", "decision": "execute", "description": "Start standby pump before shutting down primary.",
                    "required_args": ["asset_id"], "terminal": False, "state_effects": {"status": "running"}, "static_output": {"started": True},
                }],
                "persistent_goals": ["Start standby pump before requesting shutdown approval."],
                "permissions": {
                    "has_permission_constraints": True,
                    "allowed_decisions": ["wait", "query", "investigate", "request_approval", "execute"],
                    "approval_required_tools": ["create_urgent_shutdown_workorder"],
                    "forbidden_tools": [],
                    "approval_policy": {
                        "create_urgent_shutdown_workorder": {
                            "delay_turns": delay_turns, "auto_approve": True,
                            "required_world_state": {"assets": {"pump_standby": {"status": "running"}}},
                        }
                    },
                },
                "replay_ticks": ["2026-08-03T08:00:00Z", "2026-08-03T08:15:00Z", "2026-08-03T08:30:00Z"],
                "observations": [
                    {"obs_id": f"task_standby_{delay_turns}:obs:0001", "release_time": "2026-08-03T08:00:00Z", "asset_id": "pump_3", "modality": "telemetry", "source_type": "adapted", "payload": {"vibration_mm_s": 2.0}},
                    {"obs_id": f"task_standby_{delay_turns}:obs:0002", "release_time": "2026-08-03T08:10:00Z", "asset_id": "pump_3", "modality": "telemetry", "source_type": "adapted", "payload": {"vibration_mm_s": 7.5}},
                    {"obs_id": f"task_standby_{delay_turns}:obs:0003", "release_time": "2026-08-03T08:15:00Z", "asset_id": "pump_3", "modality": "telemetry", "source_type": "adapted", "payload": {"vibration_mm_s": 8.0}},
                ],
                "tool_catalog_data": {"fmea_manuals": [
                    {"fault_type": "bearing_wear_overheat", "asset_type": "centrifugal_pump", "symptom_metric": "vibration_mm_s", "threshold_high": 6.5, "recommended_tool": "create_urgent_shutdown_workorder"}
                ]},
            }

        def _make_gt(delay_turns: int) -> dict[str, Any]:
            return {
                "schema_version": "3.0",
                "scenario_id": f"task_standby_{delay_turns}",
                "no_action_ticks": ["2026-08-03T08:00:00Z"],
                "needs": [{
                    "need_id": "need_01", "asset_id": "pump_3", "fault_type": "bearing_wear_overheat", "aliases": [],
                    "visible_from": "2026-08-03T08:15:00Z",
                    "supporting_obs_ids": [f"task_standby_{delay_turns}:obs:0002", f"task_standby_{delay_turns}:obs:0003"],
                    "valid_window": {"start": "2026-08-03T08:15:00Z", "end": "2026-08-03T08:30:00Z"},
                    "allowed_actions": [
                        {"decision": "investigate", "tool_name": "check_tsfm_anomaly"},
                        {"decision": "investigate", "tool_name": "lookup_fmea_manual"},
                        {"decision": "execute", "tool_name": "start_standby_pump"},
                        {"decision": "request_approval", "tool_name": "submit_approval_request"},
                        {"decision": "execute", "tool_name": "create_urgent_shutdown_workorder"},
                    ],
                    "completion_check": {
                        "mode": "all_of",
                        "checks": [
                            {"mode": "ordered_sequence", "steps": [
                                {"tool_name": "start_standby_pump", "args_subset": {"asset_id": "pump_standby"}},
                                {"tool_name": "create_urgent_shutdown_workorder", "args_subset": {"asset_id": "pump_3"}},
                            ]},
                            {"mode": "world_state", "expected_state_subset": {"assets": {
                                "pump_standby": {"status": "running"},
                                "pump_3": {"status": "shutdown_scheduled", "urgent_shutdown": True},
                            }}},
                        ],
                    },
                }],
            }

        # Direct approval without starting standby pump must be unauthorized
        sc_d1 = _make_sc(1)
        ex = ToolExecutor(sc_d1)
        denied = ex.execute({"decision": "request_approval", "tool_call": {"name": "submit_approval_request", "args": {"asset_id": "pump_3", "target_tool": "create_urgent_shutdown_workorder", "reason": "test"}}}, "2026-08-03T08:15:00Z", 1)
        self.assertFalse(denied["authorized"])

        # Test both delay_turns == 0 (same-tick loop completion) and delay_turns == 1 (cross-tick approval)
        for d in (0, 1):
            sc_obj, gt_obj = _make_sc(d), _make_gt(d)
            sc_p = pub_dir / f"task_standby_{d}.scenario.json"
            sc_p.write_text(json.dumps(sc_obj), encoding="utf-8")
            (gt_dir / f"task_standby_{d}.gt.json").write_text(json.dumps(gt_obj), encoding="utf-8")

        self.assertTrue(validate_directory(pub_dir, gt_dir)["valid"])
        out_5 = self.tmp_path / "out_5"
        log_p, _ = run_replay(None, out_5, scenarios_dir=pub_dir, backend="mock", max_steps_per_tick=6)
        rep = score_run(pub_dir, gt_dir, log_p, out_5 / "score.json")
        self.assertEqual(rep["overall"]["verified_count"], 2)
        self.assertEqual(rep["overall"]["unauthorized_count"], 0)
        self.assertEqual(rep["overall"]["micro_need_discovery_f1"], 1.0)

        rows = self._read_jsonl(log_p)
        d0_exec = [r for r in rows if r["scenario_id"] == "task_standby_0" and (r.get("action") or {}).get("decision") == "execute"]
        self.assertEqual([r["turn_idx"] for r in d0_exec], [1, 1])  # both start_standby_pump & shutdown completed in turn 1!
        d1_exec = [r for r in rows if r["scenario_id"] == "task_standby_1" and (r.get("action") or {}).get("decision") == "execute"]
        self.assertEqual([r["turn_idx"] for r in d1_exec], [1, 2])  # cross-tick approval effective at turn 2!

    def test_case6_long_csv_tick_grid_external_catalogs_and_validate_cli(self) -> None:
        raw_d = self.tmp_path / "raw_long"
        raw_d.mkdir()
        (raw_d / "telemetry.csv").write_text(
            "Timestamp,asset_id,metric_name,metric_value\n"
            "2026-09-01T00:00:00Z,tower_1,fan_vib_mm_s,1.5\n"
            "2026-09-01T00:00:00Z,tower_1,is_anomaly,0\n"
            "2026-09-01T00:15:00Z,tower_1,fan_vib_mm_s,5.8\n"
            "2026-09-01T00:15:00Z,tower_1,is_anomaly,1\n"
            "2026-09-01T00:30:00Z,tower_1,fan_vib_mm_s,6.2\n",
            encoding="utf-8",
        )
        (raw_d / "fmea_manuals.json").write_text(json.dumps([
            {"fault_type": "cooling_tower_fan_imbalance", "asset_type": "cooling_tower", "symptom_metric": "fan_vib_mm_s", "threshold_high": 5.0, "recommended_tool": "create_maintenance_workorder"}
        ]), encoding="utf-8")
        (raw_d / "workorders.json").write_text(json.dumps([
            {"workorder_id": "HIST-CT-1", "asset_id": "tower_1", "fault_type": "cooling_tower_fill_fouling", "priority": "normal", "status": "closed", "created_at": "2026-08-10T00:00:00Z", "summary": "Closed"}
        ]), encoding="utf-8")
        (raw_d / "raw_task_spec.json").write_text(json.dumps({
            "convert_config": {"csv_format": "long", "tick_grid": {"auto_from_observations": True, "step_seconds": 900}},
            "public_spec": {
                "scenario_id": "task_long_ct",
                "asset_info": {"site_id": "p1", "asset_id": "tower_1", "asset_type": "cooling_tower", "normal_ranges": {"fan_vib_mm_s": [0.5, 4.0]}},
                "persistent_goals": ["Monitor cooling tower."],
                "permissions": {"has_permission_constraints": False, "allowed_decisions": ["wait", "query", "investigate", "execute"], "approval_required_tools": [], "forbidden_tools": []},
                "tool_catalog_data": {},
            },
            "gt_spec": {
                "no_action_ticks": ["2026-09-01T00:00:00Z"],
                "allow_readonly_query_in_no_action_ticks": True,
                "needs": [{
                    "need_id": "need_01", "asset_id": "tower_1", "fault_type": "cooling_tower_fan_imbalance", "aliases": [],
                    "visible_from": "2026-09-01T00:30:00Z",
                    "supporting_timestamps": ["2026-09-01T00:15:00Z", "2026-09-01T00:30:00Z"],
                    "valid_window": {"start": "2026-09-01T00:30:00Z", "end": "2026-09-01T00:30:00Z"},
                    "allowed_actions": [
                        {"decision": "investigate", "tool_name": "check_tsfm_anomaly"},
                        {"decision": "investigate", "tool_name": "lookup_fmea_manual"},
                        {"decision": "execute", "tool_name": "create_maintenance_workorder"},
                    ],
                    "completion_check": {"mode": "tool_call", "required_tool": "create_maintenance_workorder", "required_args_subset": {"asset_id": "tower_1"}},
                }],
            },
        }), encoding="utf-8")

        conv_out = self.tmp_path / "conv_long"
        pub_p, gt_p = convert_scenario(raw_d, conv_out)
        sc_obj = json.loads(pub_p.read_text(encoding="utf-8"))
        self.assertEqual(sc_obj["replay_ticks"], ["2026-09-01T00:00:00Z", "2026-09-01T00:15:00Z", "2026-09-01T00:30:00Z"])
        self.assertEqual(len(sc_obj["observations"]), 3)
        for o in sc_obj["observations"]:
            self.assertFalse(set(o["payload"]) & FORBIDDEN_LABEL_COLUMNS)
        self.assertEqual(len(sc_obj["tool_catalog_data"]["fmea_manuals"]), 1)
        self.assertEqual(len(sc_obj["tool_catalog_data"]["historical_workorders"]), 1)

        # CLI validate positive (exit 0)
        ok_proc = subprocess.run(
            [sys.executable, "-m", "framework.validate", "--scenarios-dir", str(conv_out / "public"), "--gt-dir", str(conv_out / "gt")],
            cwd=REPO_ROOT, capture_output=True, text=True, check=False,
        )
        self.assertEqual(ok_proc.returncode, 0, ok_proc.stderr)

        # Test allow_readonly_query_in_no_action_ticks=True permits query at t0 without early/unnecessary penalty
        gt_obj = json.loads(gt_p.read_text(encoding="utf-8"))
        ro_rep, _ = score_scenario(sc_obj, gt_obj, [{
            "turn_idx": 0, "step_idx": 0, "seq_idx": 0, "replay_time": "2026-09-01T00:00:00Z",
            "new_obs_ids": ["task_long_ct:obs:0001"], "visible_obs_ids": ["task_long_ct:obs:0001"],
            "parse_ok": True, "error": {"status": "ok"},
            "action": {"decision": "query", "evidence_obs_ids": ["task_long_ct:obs:0001"], "reported_needs": [], "tool_call": {"name": "query_iot_metadata", "args": {}}},
            "tool_result": {"executed": True, "authorized": True, "status": "ok", "output": {}},
        }])
        self.assertEqual(ro_rep["unnecessary_interventions"]["unnecessary_turns"], 0)
        self.assertEqual(ro_rep["timely_valid_interventions"][0]["status"], "missed")

        # Negative fault injection into scenario -> validate CLI must exit non-zero
        bad_dir = self.tmp_path / "bad_val"
        (bad_dir / "public").mkdir(parents=True)
        (bad_dir / "gt").mkdir(parents=True)
        bad_sc = json.loads(json.dumps(sc_obj))
        bad_sc["observations"][0]["payload"]["label"] = 1
        bad_sc["replay_ticks"].append("2026-09-01T00:10:00Z")
        (bad_dir / "public" / "task_long_ct.scenario.json").write_text(json.dumps(bad_sc), encoding="utf-8")
        (bad_dir / "gt" / "task_long_ct.gt.json").write_text(gt_p.read_text(encoding="utf-8"), encoding="utf-8")
        bad_proc = subprocess.run(
            [sys.executable, "-m", "framework.validate", "--scenarios-dir", str(bad_dir / "public"), "--gt-dir", str(bad_dir / "gt")],
            cwd=REPO_ROOT, capture_output=True, text=True, check=False,
        )
        self.assertNotEqual(bad_proc.returncode, 0)

    def test_case7_keep_alive_reuse_reconnect_replay_and_compare(self) -> None:
        _MockVLLMHandler.replay_delay_sec = 0.0
        _MockVLLMHandler.call_idx = 0
        _MockVLLMHandler.captured_requests = []
        valid_wait = json.dumps({"decision": "wait", "reason": "ok", "evidence_obs_ids": [], "reported_needs": [], "tool_call": None})
        _MockVLLMHandler.scenarios = [
            {"chunks": [{"choices": [{"delta": {"reasoning_content": "t"}}]}, {"choices": [{"delta": {"content": f"```json\n{valid_wait}\n```"}}]}, {"choices": [], "usage": {"prompt_tokens": 100, "completion_tokens": 20, "prompt_tokens_details": {"cached_tokens": 50}}}]},
            {"chunks": [{"choices": [{"delta": {"content": valid_wait}}]}, {"choices": [], "usage": {"prompt_tokens": 120, "completion_tokens": 20, "prompt_tokens_details": {"cached_tokens": 90}}}]},
            # Simulate server closing idle socket -> VLLMClient must reconnect once transparently!
            {"drop_socket": True},
            {"chunks": [{"choices": [{"delta": {"content": valid_wait}}]}, {"choices": [], "usage": {"prompt_tokens": 130, "completion_tokens": 20}}]},
            {"http_status": 500},
            {"delay_sec": 0.3, "chunks": []},
        ]

        server = _CountingHTTPServer(("127.0.0.1", 0), _MockVLLMHandler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            base_url = f"http://127.0.0.1:{server.server_port}/v1"
            client = VLLMClient(base_url=base_url, model="test-model", timeout=0.15)
            msgs = [{"role": "system", "content": "sys"}, {"role": "user", "content": "usr"}]

            r0 = client.generate(msgs)
            r1 = client.generate(msgs)
            self.assertTrue(r0["parse_ok"] and r1["parse_ok"])
            self.assertEqual(r0["usage"]["cached_tokens"], 50)
            self.assertEqual(r1["usage"]["cached_tokens"], 90)
            # First two requests MUST reuse the exact same single TCP connection!
            self.assertEqual(server.accepted_connections, 1)

            # Third request hits dropped socket and automatically reconnects (1 extra TCP accept)
            r2 = client.generate(msgs)
            self.assertTrue(r2["parse_ok"])
            self.assertIsNone(r2["usage"]["cached_tokens"])
            self.assertEqual(server.accepted_connections, 2)

            self.assertEqual(client.generate(msgs)["error"]["status"], "api_error")
            self.assertEqual(client.generate(msgs)["error"]["status"], "api_timeout")
            client.close()

            # Concurrent replay.py ordering test
            prep_dir = self.tmp_path / "prep_r"
            log_p, _ = run_replay(None, prep_dir, scenarios_dir=self.sc_dir / "public", backend="mock")
            reqs_p = prep_dir / "reqs.jsonl"
            export_logs(log_p, reqs_p, prep_dir / "acts.jsonl")
            _MockVLLMHandler.replay_events = []
            _MockVLLMHandler.replay_delay_sec = 0.01
            bench = run_benchmark(reqs_p, prep_dir / "bench.json", backend="vllm", base_url=base_url, cache_mode="on", concurrency=2, warmup_sessions=1, repeats=2)
            self.assertEqual(bench["aggregate"]["ok_requests"], 28)
            self.assertEqual(bench["aggregate"]["usage_totals"]["cache_hit_rate"], 0.6)
        finally:
            server.shutdown()
            server.server_close()

        # Compare CLI JSON + Markdown verification
        comp_dir = self.tmp_path / "comp"
        comp_dir.mkdir()
        off_sum = {"median_ttft_ms": 100.0, "p95_ttft_ms": 150.0, "median_latency_ms": 300.0, "p95_latency_ms": 400.0, "successful_rps": 2.0, "total_wall_ms": 2000.0, "usage_totals": {"prompt_tokens": 1000, "completion_tokens": 100, "cached_tokens": 0}, "prefix_reuse_stats": {"intra_session_reuse_ratio": 0.79}}
        on_sum = {"median_ttft_ms": 50.0, "p95_ttft_ms": 80.0, "median_latency_ms": 200.0, "p95_latency_ms": 260.0, "successful_rps": 3.0, "total_wall_ms": 1333.0, "usage_totals": {"prompt_tokens": 1000, "completion_tokens": 100, "cached_tokens": 760}, "prefix_reuse_stats": {"intra_session_reuse_ratio": 0.79}}
        (comp_dir / "off_sum.json").write_text(json.dumps(off_sum), encoding="utf-8")
        (comp_dir / "on_sum.json").write_text(json.dumps(on_sum), encoding="utf-8")
        off_sc = {"overall": {"micro_need_discovery_f1": 1.0, "verified_count": 2, "total_unnecessary_turns": 0, "unauthorized_count": 0}}
        on_sc = {"overall": {"micro_need_discovery_f1": 1.0, "verified_count": 2, "total_unnecessary_turns": 0, "unauthorized_count": 0}}
        (comp_dir / "off_sc.json").write_text(json.dumps(off_sc), encoding="utf-8")
        (comp_dir / "on_sc.json").write_text(json.dumps(on_sc), encoding="utf-8")

        c_rep = compare_runs(
            comp_dir / "off_sum.json", comp_dir / "on_sum.json", comp_dir / "cmp.json",
            off_log_path=log_p, on_log_path=log_p,
            off_score_path=comp_dir / "off_sc.json", on_score_path=comp_dir / "on_sc.json",
            target_ttft_reduction_pct=20.0, out_md_path=comp_dir / "cmp.md",
        )
        self.assertEqual(c_rep["latency_and_throughput"]["median_ttft_reduction_pct"], 50.0)
        self.assertEqual(c_rep["cache_efficiency"]["on_empirical_cache_hit_rate"], 0.76)
        self.assertEqual(c_rep["trajectory_determinism"]["exact_request_match_rate"], 1.0)
        self.assertTrue(c_rep["hypothesis_evaluation"]["overall_hypothesis_pass"])
        self.assertIn("Median TTFT (ms)", (comp_dir / "cmp.md").read_text(encoding="utf-8"))

        # Check parse_action_json guardrails
        act, err = parse_action_json('{"decision": "wait", "tool_call": {"name": "query_iot_metadata", "args": {}}}')
        self.assertIsNone(act)
        self.assertIsNotNone(err)

    def test_verifier_edge_cases_validation_world_state_freetext_and_args_interp(self) -> None:
        # 1. validate_pair catches invalid permissions.allowed_decisions, approval_required_tools, approval_policy, allowed_actions.decision
        base_sc = json.loads((self.sc_dir / "public" / "task_01_chiller6.scenario.json").read_text(encoding="utf-8"))
        base_gt = json.loads((self.sc_dir / "gt" / "task_01_chiller6.gt.json").read_text(encoding="utf-8"))
        bad_sc = json.loads(json.dumps(base_sc))
        bad_sc["permissions"]["allowed_decisions"] = ["wait", "investgate_typo"]
        bad_sc["permissions"]["approval_required_tools"] = ["typo_tool"]
        bad_sc["permissions"]["approval_policy"] = {"typo_policy_tool": {"delay_turns": 1}}
        bad_gt = json.loads(json.dumps(base_gt))
        bad_gt["needs"][0]["allowed_actions"][0]["decision"] = "typo_decision"
        errs = validate_pair(bad_sc, bad_gt)
        self.assertEqual(len(errs), 4, errs)

        # 2. world_state rubric passes when auxiliary actuator custom_tool acts for primary asset need
        aux_sc = {
            "schema_version": "3.0",
            "scenario_id": "task_aux_ws",
            "asset_info": {"site_id": "p1", "asset_id": "zone_1", "asset_type": "thermal_zone", "normal_ranges": {"zone_temp_c": [20.0, 24.0]}},
            "assets": [
                {"site_id": "p1", "asset_id": "zone_1", "asset_type": "thermal_zone", "normal_ranges": {"zone_temp_c": [20.0, 24.0]}},
                {"site_id": "p1", "asset_id": "damper_1", "asset_type": "damper", "initial_status": "closed", "normal_ranges": {}},
            ],
            "custom_tools": [{"name": "set_damper_position", "decision": "execute", "required_args": ["asset_id", "pos_pct"], "terminal": True, "state_effects": {"pos_pct": "$args.pos_pct", "status": "open"}}],
            "persistent_goals": ["Open damper_1 if zone_1 overheats."],
            "permissions": {"has_permission_constraints": False, "allowed_decisions": ["wait", "investigate", "execute"], "approval_required_tools": [], "forbidden_tools": []},
            "replay_ticks": ["2026-09-10T08:00:00Z"],
            "observations": [{"obs_id": "task_aux_ws:obs:0001", "release_time": "2026-09-10T08:00:00Z", "asset_id": "zone_1", "modality": "telemetry", "source_type": "synthetic", "payload": {"zone_temp_c": 28.0}}],
            "tool_catalog_data": {"fmea_manuals": []},
        }
        aux_gt = {
            "schema_version": "3.0",
            "scenario_id": "task_aux_ws",
            "no_action_ticks": [],
            "needs": [{
                "need_id": "need_01", "asset_id": "zone_1", "fault_type": "ahu_damper_stuck", "aliases": [],
                "visible_from": "2026-09-10T08:00:00Z", "supporting_obs_ids": ["task_aux_ws:obs:0001"],
                "valid_window": {"start": "2026-09-10T08:00:00Z", "end": "2026-09-10T08:00:00Z"},
                "allowed_actions": [{"decision": "execute", "tool_name": "set_damper_position"}],
                "completion_check": {"mode": "world_state", "expected_state_subset": {"assets": {"damper_1": {"pos_pct": 100, "status": "open"}}}},
            }],
        }
        aux_rep, _ = score_scenario(aux_sc, aux_gt, [{
            "turn_idx": 0, "step_idx": 0, "seq_idx": 0, "replay_time": "2026-09-10T08:00:00Z",
            "new_obs_ids": ["task_aux_ws:obs:0001"], "visible_obs_ids": ["task_aux_ws:obs:0001"],
            "parse_ok": True, "error": {"status": "ok"},
            "action": {"decision": "execute", "evidence_obs_ids": ["task_aux_ws:obs:0001"], "reported_needs": [], "tool_call": {"name": "set_damper_position", "args": {"asset_id": "damper_1", "pos_pct": 100}}},
            "tool_result": {"executed": True, "authorized": True, "status": "ok", "output": {}},
        }])
        self.assertEqual(aux_rep["verified_task_success"][0]["status"], "verified")

        # 3. Free-text remediation infers create_urgent_shutdown_workorder + $args.* prerequisite custom_tool binding
        p_dir, g_dir = self.tmp_path / "p_interp", self.tmp_path / "g_interp"
        p_dir.mkdir()
        g_dir.mkdir()
        sc_interp = {
            "schema_version": "3.0",
            "scenario_id": "task_interp_ft",
            "asset_info": {"site_id": "p1", "asset_id": "pump_3", "asset_type": "centrifugal_pump", "normal_ranges": {"vibration_mm_s": [0.5, 4.5]}},
            "assets": [
                {"site_id": "p1", "asset_id": "pump_3", "asset_type": "centrifugal_pump", "initial_status": "running", "normal_ranges": {"vibration_mm_s": [0.5, 4.5]}},
                {"site_id": "p1", "asset_id": "pump_standby", "asset_type": "centrifugal_pump", "initial_status": "standby", "normal_ranges": {}},
            ],
            "custom_tools": [{
                "name": "set_pump_mode", "decision": "execute", "required_args": ["asset_id", "target_status"],
                "terminal": False, "state_effects": {"status": "$args.target_status"},
            }],
            "persistent_goals": ["Set standby pump running before urgent shutdown."],
            "permissions": {
                "has_permission_constraints": True,
                "allowed_decisions": ["wait", "investigate", "request_approval", "execute"],
                "approval_required_tools": ["create_urgent_shutdown_workorder"],
                "forbidden_tools": [],
                "approval_policy": {"create_urgent_shutdown_workorder": {"delay_turns": 0, "auto_approve": True, "required_world_state": {"assets": {"pump_standby": {"status": "running"}}}}},
            },
            "replay_ticks": ["2026-09-10T08:00:00Z"],
            "observations": [
                {"obs_id": "task_interp_ft:obs:0001", "release_time": "2026-09-10T07:55:00Z", "asset_id": "pump_3", "modality": "telemetry", "source_type": "synthetic", "payload": {"vibration_mm_s": 7.5}},
                {"obs_id": "task_interp_ft:obs:0002", "release_time": "2026-09-10T08:00:00Z", "asset_id": "pump_3", "modality": "telemetry", "source_type": "synthetic", "payload": {"vibration_mm_s": 8.0}},
            ],
            "tool_catalog_data": {"fmea_manuals": [{
                "fault_type": "bearing_wear_overheat", "asset_type": "centrifugal_pump",
                "failure_mode": "Excessive vibration", "remediation": "Perform urgent shutdown immediately",
            }]},
        }
        gt_interp = {
            "schema_version": "3.0",
            "scenario_id": "task_interp_ft",
            "no_action_ticks": [],
            "needs": [{
                "need_id": "need_01", "asset_id": "pump_3", "fault_type": "bearing_wear_overheat", "aliases": [],
                "visible_from": "2026-09-10T08:00:00Z",
                "supporting_obs_ids": ["task_interp_ft:obs:0001", "task_interp_ft:obs:0002"],
                "valid_window": {"start": "2026-09-10T08:00:00Z", "end": "2026-09-10T08:00:00Z"},
                "allowed_actions": [
                    {"decision": "investigate", "tool_name": "check_tsfm_anomaly"},
                    {"decision": "investigate", "tool_name": "lookup_fmea_manual"},
                    {"decision": "execute", "tool_name": "set_pump_mode"},
                    {"decision": "request_approval", "tool_name": "submit_approval_request"},
                    {"decision": "execute", "tool_name": "create_urgent_shutdown_workorder"},
                ],
                "completion_check": {"mode": "ordered_sequence", "steps": [
                    {"tool_name": "set_pump_mode", "args_subset": {"asset_id": "pump_standby", "target_status": "running"}},
                    {"tool_name": "create_urgent_shutdown_workorder", "args_subset": {"asset_id": "pump_3"}},
                ]},
            }],
        }
        sc_ip = p_dir / "task_interp_ft.scenario.json"
        sc_ip.write_text(json.dumps(sc_interp), encoding="utf-8")
        (g_dir / "task_interp_ft.gt.json").write_text(json.dumps(gt_interp), encoding="utf-8")
        self.assertTrue(validate_directory(p_dir, g_dir)["valid"])
        log_i, _ = run_replay([sc_ip], self.tmp_path / "out_interp", backend="mock", max_steps_per_tick=6)
        rep_i = score_run(p_dir, g_dir, log_i, self.tmp_path / "out_interp" / "score.json")
        self.assertEqual(rep_i["overall"]["verified_count"], 1)
        self.assertEqual(rep_i["overall"]["unauthorized_count"], 0)


if __name__ == "__main__":
    unittest.main()
