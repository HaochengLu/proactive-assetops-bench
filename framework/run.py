"""CLI 3: Bounded intra-tick multi-step replay runner with directory/tag filtering & WorldState."""

from __future__ import annotations

import argparse
import json
import math
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from framework import SCHEMA_VERSION, parse_iso_utc
from framework.client import MockClient, VLLMClient
from framework.prompt import GLOBAL_SYSTEM_PREFIX, append_turn_messages, build_initial_messages
from framework.tools import ToolExecutor


def _stat(values: list[float], pct: float | None = None) -> float | None:
    if not values:
        return None
    s = sorted(values)
    if pct is None:
        mid = len(s) // 2
        val = s[mid] if len(s) % 2 == 1 else (s[mid - 1] + s[mid]) / 2.0
    else:
        val = s[max(0, min(len(s) - 1, math.ceil(pct * len(s)) - 1))]
    return round(val, 3)


def _msg_bytes(msg: dict[str, str]) -> int:
    return len(json.dumps(msg, ensure_ascii=False, sort_keys=True).encode("utf-8"))


def _shared_prefix_bytes(prev_msgs: list[dict[str, str]] | None, curr_msgs: list[dict[str, str]]) -> int:
    if not prev_msgs:
        return 0
    shared = 0
    for a, b in zip(prev_msgs, curr_msgs):
        if a == b:
            shared += _msg_bytes(a)
        else:
            break
    return shared


def _resolve_scenarios(scenarios: list[Path] | None, scenarios_dir: Path | None, tags: list[str] | None) -> list[Path]:
    paths: list[Path] = []
    if scenarios:
        paths.extend(scenarios)
    if scenarios_dir:
        paths.extend(sorted(scenarios_dir.glob("*.scenario.json")))
    seen: set[Path] = set()
    unique: list[Path] = []
    for p in paths:
        rp = p.resolve()
        if rp not in seen:
            seen.add(rp)
            unique.append(p)
    if tags:
        tag_set = set(tags)
        filtered = []
        for p in unique:
            sc_tags = set(json.loads(p.read_text(encoding="utf-8")).get("tags") or [])
            if tag_set & sc_tags:
                filtered.append(p)
        return filtered
    return unique


def _run_one_scenario(
    sc_path: Path,
    resolved_run_id: str,
    cache_mode: str,
    max_steps_per_tick: int,
    inference_config: dict[str, Any],
    client: MockClient | VLLMClient,
) -> tuple[str, list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    scenario = json.loads(sc_path.read_text(encoding="utf-8"))
    sc_id = scenario["scenario_id"]
    session_id = f"{resolved_run_id}:{sc_id}"
    executor = ToolExecutor(scenario)

    obs_sorted = sorted(scenario.get("observations", []), key=lambda o: parse_iso_utc(o["release_time"]))
    obs_dts = [parse_iso_utc(o["release_time"]) for o in obs_sorted]
    visible_obs_ids: list[str] = []
    prev_tick_dt: datetime | None = None

    messages: list[dict[str, str]] | None = None
    prev_messages: list[dict[str, str]] | None = None
    prev_raw: str | None = None
    pending_tool_result: dict[str, Any] | None = None
    prev_req_id: str | None = None

    records: list[dict[str, Any]] = []
    sc_t0 = time.perf_counter_ns()
    seq_idx = sc_ok = sc_fail = prompt_bytes_sum = shared_bytes_sum = 0
    ticks = scenario.get("replay_ticks", [])

    for turn_idx, tick_str in enumerate(ticks):
        tick_dt = parse_iso_utc(tick_str)
        new_obs = [
            obs for obs, rel_dt in zip(obs_sorted, obs_dts)
            if (prev_tick_dt is None or rel_dt > prev_tick_dt) and rel_dt <= tick_dt
        ]
        prev_tick_dt = tick_dt

        for step_idx in range(max_steps_per_tick):
            step_t0 = time.perf_counter_ns()
            step_obs = new_obs if step_idx == 0 else []
            step_obs_ids = [o["obs_id"] for o in step_obs]
            if step_idx == 0:
                visible_obs_ids.extend(step_obs_ids)

            messages = (
                build_initial_messages(scenario, tick_str, step_obs)
                if seq_idx == 0
                else append_turn_messages(messages or [], prev_raw, pending_tool_result, turn_idx, step_idx, tick_str, step_obs)
            )
            pending_tool_result = None

            prompt_bytes_sum += sum(_msg_bytes(m) for m in messages)
            shared_bytes_sum += _shared_prefix_bytes(prev_messages, messages)
            prev_messages = messages

            resp = client.generate(messages)
            error, action = dict(resp["error"]), resp["action"]

            tool_result, tool_ms = None, 0.0
            if resp["parse_ok"] and action is not None:
                tool_t0 = time.perf_counter_ns()
                tool_result = executor.execute(action, tick_str, turn_idx, step_idx)
                tool_ms = round((time.perf_counter_ns() - tool_t0) / 1e6, 3)
                if tool_result["status"] == "tool_error" and error["status"] == "ok":
                    error = {"status": "tool_error", "http_code": error.get("http_code"), "message": (tool_result.get("output") or {}).get("reason", "tool_error")}

            request_id = f"{sc_id}:req:{seq_idx:04d}"
            records.append({
                "schema_version": SCHEMA_VERSION, "run_id": resolved_run_id, "session_id": session_id,
                "scenario_id": sc_id, "cache_mode": cache_mode,
                "turn_idx": turn_idx, "step_idx": step_idx, "seq_idx": seq_idx,
                "request_id": request_id, "prev_request_id": prev_req_id, "replay_time": tick_str,
                "new_obs_ids": step_obs_ids, "visible_obs_ids": list(visible_obs_ids),
                "inference_config": inference_config, "llm_request_messages": messages,
                "raw_response": resp["raw_response"], "parse_ok": resp["parse_ok"],
                "action": action, "tool_result": tool_result,
                "timing_ms": {**resp["timing_ms"], "tool_ms": tool_ms, "turn_wall_ms": round((time.perf_counter_ns() - step_t0) / 1e6, 3)},
                "usage": resp["usage"], "error": error,
            })

            if error["status"] == "ok":
                sc_ok += 1
            else:
                sc_fail += 1

            prev_req_id, prev_raw = request_id, resp["raw_response"]
            if resp["parse_ok"] and action is not None and action.get("tool_call") is not None:
                pending_tool_result = tool_result
            seq_idx += 1

            if (
                not resp["parse_ok"] or action is None or error["status"] != "ok"
                or action.get("decision") == "wait" or action.get("tool_call") is None
                or tool_result is None or tool_result.get("executed") is False
            ):
                break

            decision = action.get("decision")
            tool_name = (action.get("tool_call") or {}).get("name", "")
            if decision == "request_approval":
                eff = int((tool_result.get("output") or {}).get("effective_from_turn", turn_idx + 1))
                if eff > turn_idx:
                    break
            elif decision == "execute" and executor.is_terminal_tool(tool_name):
                break

    sc_summary = {
        "turns": len(ticks), "requests": seq_idx,
        "workflow_wall_ms": round((time.perf_counter_ns() - sc_t0) / 1e6, 3),
        "ok_requests": sc_ok, "failed_requests": sc_fail,
        "final_world_state": executor.world_state,
    }
    byte_stats = {"total_prompt_bytes": prompt_bytes_sum, "shared_prefix_bytes": shared_bytes_sum}
    return sc_id, records, sc_summary, byte_stats


def run_replay(
    scenario_paths: list[Path] | None,
    out_dir: Path,
    backend: str = "mock",
    mock_mode: str = "heuristic",
    cache_mode: str = "off",
    base_url: str = "http://127.0.0.1:8000/v1",
    model: str | None = None,
    api_key: str = "",
    temperature: float = 0.0,
    max_tokens: int = 256,
    seed: int = 42,
    timeout: float = 60.0,
    max_steps_per_tick: int = 3,
    concurrency: int = 1,
    gpu_tag: str = "local_cpu",
    dtype: str = "na",
    repeat_idx: int = 0,
    run_id: str | None = None,
    scenarios_dir: Path | None = None,
    tags: list[str] | None = None,
) -> tuple[Path, Path]:
    """Replay scenarios strictly in tick order with bounded intra-tick steps."""
    resolved_paths = _resolve_scenarios(scenario_paths, scenarios_dir, tags)
    if not resolved_paths:
        raise ValueError("No scenario files matched --scenarios / --scenarios-dir / --tags")

    out_dir.mkdir(parents=True, exist_ok=True)
    resolved_run_id = run_id or f"run_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
    resolved_model = model or (f"mock-{mock_mode}" if backend == "mock" else "Qwen/Qwen2.5-7B-Instruct")

    client: MockClient | VLLMClient
    if backend == "mock":
        client = MockClient(mode=mock_mode)
    elif backend == "vllm":
        client = VLLMClient(base_url, resolved_model, api_key, temperature, max_tokens, seed, timeout)
    else:
        raise ValueError(f"Unsupported backend: {backend}")

    inference_config = {
        "backend": backend, "model": resolved_model, "temperature": temperature,
        "max_tokens": max_tokens, "seed": seed, "stream": True, "stream_options": {"include_usage": True},
    }

    log_path, summary_path = out_dir / "run_log.jsonl", out_dir / "run_summary.json"
    total_t0 = time.perf_counter_ns()
    total_reqs = ok_reqs = failed_reqs = prompt_sum = comp_sum = cached_sum = 0
    total_prompt_bytes = shared_prefix_bytes = 0
    saw_cached = False
    ttft_vals: list[float] = []
    lat_vals: list[float] = []
    per_scenario: dict[str, dict[str, Any]] = {}

    def _consume_result(res: tuple[str, list[dict[str, Any]], dict[str, Any], dict[str, Any]], log_f: Any) -> None:
        nonlocal total_reqs, ok_reqs, failed_reqs, prompt_sum, comp_sum, cached_sum, saw_cached, total_prompt_bytes, shared_prefix_bytes
        sc_id, records, sc_sum, b_stats = res
        per_scenario[sc_id] = sc_sum
        total_prompt_bytes += b_stats["total_prompt_bytes"]
        shared_prefix_bytes += b_stats["shared_prefix_bytes"]
        for rec in records:
            log_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            total_reqs += 1
            if rec["error"]["status"] == "ok":
                ok_reqs += 1
            else:
                failed_reqs += 1
            if rec["timing_ms"]["ttft_ms"] is not None:
                ttft_vals.append(float(rec["timing_ms"]["ttft_ms"]))
            if rec["timing_ms"]["latency_ms"] is not None:
                lat_vals.append(float(rec["timing_ms"]["latency_ms"]))
            prompt_sum += int(rec["usage"].get("prompt_tokens") or 0)
            comp_sum += int(rec["usage"].get("completion_tokens") or 0)
            if rec["usage"].get("cached_tokens") is not None:
                saw_cached = True
                cached_sum += int(rec["usage"]["cached_tokens"])
        log_f.flush()

    try:
        with log_path.open("w", encoding="utf-8") as log_f:
            if concurrency <= 1:
                for sc_path in resolved_paths:
                    _consume_result(
                        _run_one_scenario(sc_path, resolved_run_id, cache_mode, max_steps_per_tick, inference_config, client),
                        log_f,
                    )
            else:
                with ThreadPoolExecutor(max_workers=concurrency) as pool:
                    futures = [
                        pool.submit(_run_one_scenario, p, resolved_run_id, cache_mode, max_steps_per_tick, inference_config, client)
                        for p in resolved_paths
                    ]
                    for fut in futures:
                        _consume_result(fut.result(), log_f)
    finally:
        if isinstance(client, VLLMClient):
            client.close()

    total_wall_ms = round((time.perf_counter_ns() - total_t0) / 1e6, 3)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "run_id": resolved_run_id,
        "backend": backend,
        "cache_mode": cache_mode,
        "experiment_meta": {
            "max_steps_per_tick": max_steps_per_tick,
            "concurrency": concurrency,
            "gpu_tag": gpu_tag,
            "dtype": dtype,
            "repeat_idx": repeat_idx,
        },
        "total_wall_ms": total_wall_ms,
        "total_requests": total_reqs,
        "ok_requests": ok_reqs,
        "failed_requests": failed_reqs,
        "median_ttft_ms": _stat(ttft_vals),
        "p95_ttft_ms": _stat(ttft_vals, 0.95),
        "median_latency_ms": _stat(lat_vals),
        "p95_latency_ms": _stat(lat_vals, 0.95),
        "successful_rps": round(ok_reqs / (total_wall_ms / 1000.0), 2) if total_wall_ms > 0 else 0.0,
        "usage_totals": {
            "prompt_tokens": prompt_sum,
            "completion_tokens": comp_sum,
            "cached_tokens": cached_sum if saw_cached else None,
        },
        "prefix_reuse_stats": {
            "total_prompt_bytes": total_prompt_bytes,
            "intra_session_shared_prefix_bytes": shared_prefix_bytes,
            "intra_session_reuse_ratio": round(shared_prefix_bytes / total_prompt_bytes, 6) if total_prompt_bytes > 0 else 0.0,
            "global_system_prefix_bytes": len(GLOBAL_SYSTEM_PREFIX.encode("utf-8")),
        },
        "per_scenario": per_scenario,
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return log_path, summary_path


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Time-stepped bounded multi-step proactive replay runner.")
    p.add_argument("--scenarios", nargs="+", default=None, type=Path)
    p.add_argument("--scenarios-dir", default=None, type=Path)
    p.add_argument("--tags", nargs="+", default=None)
    p.add_argument("--backend", choices=["mock", "vllm"], default="mock")
    p.add_argument("--mock-mode", choices=["heuristic", "eager_unauth"], default="heuristic")
    p.add_argument("--cache-mode", choices=["off", "on", "na"], default="off")
    p.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    p.add_argument("--model", default=None)
    p.add_argument("--api-key", default="")
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--max-tokens", type=int, default=256)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--timeout", type=float, default=60.0)
    p.add_argument("--max-steps-per-tick", type=int, default=3)
    p.add_argument("--concurrency", type=int, default=1)
    p.add_argument("--gpu-tag", default="local_cpu")
    p.add_argument("--dtype", default="na")
    p.add_argument("--repeat-idx", type=int, default=0)
    p.add_argument("--run-id", default=None)
    p.add_argument("--out-dir", required=True, type=Path)
    a = p.parse_args(argv)
    if not a.scenarios and not a.scenarios_dir:
        p.error("Provide --scenarios and/or --scenarios-dir")
    run_replay(
        a.scenarios, a.out_dir, a.backend, a.mock_mode, a.cache_mode, a.base_url, a.model, a.api_key,
        a.temperature, a.max_tokens, a.seed, a.timeout, a.max_steps_per_tick, a.concurrency,
        a.gpu_tag, a.dtype, a.repeat_idx, a.run_id, a.scenarios_dir, a.tags,
    )


if __name__ == "__main__":
    main()
