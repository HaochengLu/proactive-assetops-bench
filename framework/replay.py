"""CLI 5: Concurrent fixed-trajectory replay benchmarker with strict intra-session ordering."""

from __future__ import annotations

import argparse
import json
import math
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from framework import SCHEMA_VERSION
from framework.client import MockClient, VLLMClient


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


def _run_session(reqs: list[dict[str, Any]], client: MockClient | VLLMClient) -> dict[str, Any]:
    t0 = time.perf_counter_ns()
    results = [client.generate(r["messages"]) for r in reqs]
    return {"wall_ms": round((time.perf_counter_ns() - t0) / 1e6, 3), "results": results}


def run_benchmark(
    requests_path: Path,
    out_path: Path,
    backend: str = "mock",
    base_url: str = "http://127.0.0.1:8000/v1",
    model: str | None = None,
    api_key: str = "",
    cache_mode: str = "off",
    concurrency: int = 1,
    warmup_sessions: int = 0,
    repeats: int = 1,
    gpu_tag: str = "local_cpu",
    dtype: str = "na",
    timeout: float = 60.0,
) -> dict[str, Any]:
    """Replay fixed request trajectories concurrently across sessions while preserving intra-session order."""
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    first_cfg: dict[str, Any] = {}
    with requests_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            if not first_cfg:
                first_cfg = row.get("inference_config") or {}
            grouped[row["session_id"]].append(row)

    sessions: list[list[dict[str, Any]]] = []
    for reqs in grouped.values():
        reqs.sort(key=lambda r: (int(r.get("seq_idx", r.get("replay_seq", r.get("turn_idx", 0)))), int(r.get("replay_seq", 0))))
        sessions.append(reqs)
    if not sessions:
        raise ValueError(f"No sessions found in {requests_path}")

    resolved_model = model or first_cfg.get("model") or ("mock-heuristic" if backend == "mock" else "Qwen/Qwen2.5-7B-Instruct")
    temperature = float(first_cfg.get("temperature", 0.0))
    max_tokens = int(first_cfg.get("max_tokens", 256))
    seed = int(first_cfg.get("seed", 42))

    client: MockClient | VLLMClient
    if backend == "mock":
        client = MockClient(mode="heuristic")
    elif backend == "vllm":
        client = VLLMClient(base_url, resolved_model, api_key, temperature, max_tokens, seed, timeout)
    else:
        raise ValueError(f"Unsupported backend: {backend}")

    def _dispatch(sess_list: list[list[dict[str, Any]]]) -> list[dict[str, Any]]:
        if concurrency <= 1:
            return [_run_session(s, client) for s in sess_list]
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            return [f.result() for f in [pool.submit(_run_session, s, client) for s in sess_list]]

    all_ttft: list[float] = []
    all_lat: list[float] = []
    all_sess_wall: list[float] = []
    repeat_walls: list[float] = []
    repeats_detail: list[dict[str, Any]] = []
    tot_reqs = tot_ok = tot_fail = tot_prompt = tot_comp = tot_cached = 0
    saw_cached = False

    try:
        if warmup_sessions > 0:
            warmup_list = [sessions[i % len(sessions)] for i in range(warmup_sessions)]
            _dispatch(warmup_list)

        for rep_idx in range(repeats):
            rep_t0 = time.perf_counter_ns()
            sess_outcomes = _dispatch(sessions)
            rep_wall = round((time.perf_counter_ns() - rep_t0) / 1e6, 3)
            repeat_walls.append(rep_wall)

            rep_ttft: list[float] = []
            rep_lat: list[float] = []
            rep_ok = rep_fail = rep_cached = 0
            rep_saw_cached = False
            for so in sess_outcomes:
                all_sess_wall.append(so["wall_ms"])
                for res in so["results"]:
                    tot_reqs += 1
                    if res["error"]["status"] == "ok":
                        rep_ok += 1
                        tot_ok += 1
                    else:
                        rep_fail += 1
                        tot_fail += 1
                    if res["timing_ms"]["ttft_ms"] is not None:
                        v = float(res["timing_ms"]["ttft_ms"])
                        rep_ttft.append(v)
                        all_ttft.append(v)
                    if res["timing_ms"]["latency_ms"] is not None:
                        v = float(res["timing_ms"]["latency_ms"])
                        rep_lat.append(v)
                        all_lat.append(v)
                    tot_prompt += int(res["usage"].get("prompt_tokens") or 0)
                    tot_comp += int(res["usage"].get("completion_tokens") or 0)
                    if res["usage"].get("cached_tokens") is not None:
                        c_val = int(res["usage"]["cached_tokens"])
                        rep_saw_cached = saw_cached = True
                        rep_cached += c_val
                        tot_cached += c_val

            repeats_detail.append({
                "repeat_idx": rep_idx,
                "wall_ms": rep_wall,
                "ok_requests": rep_ok,
                "failed_requests": rep_fail,
                "median_ttft_ms": _stat(rep_ttft),
                "p95_ttft_ms": _stat(rep_ttft, 0.95),
                "median_latency_ms": _stat(rep_lat),
                "p95_latency_ms": _stat(rep_lat, 0.95),
                "successful_rps": round(rep_ok / (rep_wall / 1000.0), 2) if rep_wall > 0 else 0.0,
                "cached_tokens": rep_cached if rep_saw_cached else None,
            })
    finally:
        if isinstance(client, VLLMClient):
            client.close()

    total_wall_ms = round(sum(repeat_walls), 3)
    hit_rate = round(tot_cached / tot_prompt, 6) if (saw_cached and tot_prompt > 0) else None
    report = {
        "schema_version": SCHEMA_VERSION,
        "backend": backend,
        "model": resolved_model,
        "cache_mode": cache_mode,
        "experiment_meta": {
            "concurrency": concurrency,
            "warmup_sessions": warmup_sessions,
            "repeats": repeats,
            "gpu_tag": gpu_tag,
            "dtype": dtype,
        },
        "trajectory_stats": {
            "num_sessions": len(sessions),
            "requests_per_repeat": sum(len(s) for s in sessions),
        },
        "aggregate": {
            "total_timed_requests": tot_reqs,
            "ok_requests": tot_ok,
            "failed_requests": tot_fail,
            "total_wall_ms": total_wall_ms,
            "mean_repeat_wall_ms": round(total_wall_ms / repeats, 3) if repeats else 0.0,
            "median_ttft_ms": _stat(all_ttft),
            "p95_ttft_ms": _stat(all_ttft, 0.95),
            "median_latency_ms": _stat(all_lat),
            "p95_latency_ms": _stat(all_lat, 0.95),
            "mean_session_wall_ms": round(sum(all_sess_wall) / len(all_sess_wall), 3) if all_sess_wall else 0.0,
            "p95_session_wall_ms": _stat(all_sess_wall, 0.95),
            "successful_rps": round(tot_ok / (total_wall_ms / 1000.0), 2) if total_wall_ms > 0 else 0.0,
            "usage_totals": {
                "prompt_tokens": tot_prompt,
                "completion_tokens": tot_comp,
                "cached_tokens": tot_cached if saw_cached else None,
                "cache_hit_rate": hit_rate,
            },
        },
        "repeats_detail": repeats_detail,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Concurrent fixed-trajectory replay benchmarker.")
    p.add_argument("--requests", required=True, type=Path)
    p.add_argument("--backend", choices=["mock", "vllm"], default="mock")
    p.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    p.add_argument("--model", default=None)
    p.add_argument("--api-key", default="")
    p.add_argument("--cache-mode", choices=["off", "on", "na"], default="off")
    p.add_argument("--concurrency", type=int, default=1)
    p.add_argument("--warmup-sessions", type=int, default=0)
    p.add_argument("--repeats", type=int, default=1)
    p.add_argument("--gpu-tag", default="local_cpu")
    p.add_argument("--dtype", default="na")
    p.add_argument("--timeout", type=float, default=60.0)
    p.add_argument("--out", required=True, type=Path)
    a = p.parse_args(argv)
    run_benchmark(
        a.requests, a.out, a.backend, a.base_url, a.model, a.api_key, a.cache_mode,
        a.concurrency, a.warmup_sessions, a.repeats, a.gpu_tag, a.dtype, a.timeout,
    )


if __name__ == "__main__":
    main()
