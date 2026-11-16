"""CLI 7: Cache OFF vs Cache ON performance, cache efficiency, determinism, and quality comparator."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from framework import SCHEMA_VERSION


def _reduction_pct(off_val: float | None, on_val: float | None) -> float | None:
    if off_val is None or on_val is None or off_val == 0:
        return None
    return round((off_val - on_val) / off_val * 100.0, 3)


def _empirical_hit_rate(summary: dict[str, Any], replay: dict[str, Any] | None) -> float | None:
    if replay is not None:
        return (replay.get("aggregate") or {}).get("usage_totals", {}).get("cache_hit_rate")
    usage = summary.get("usage_totals") or {}
    cached, prompt = usage.get("cached_tokens"), usage.get("prompt_tokens")
    if cached is None or not prompt:
        return None
    return round(float(cached) / float(prompt), 6)


def _load_log_map(log_path: Path) -> dict[tuple[str, int], dict[str, Any]]:
    rows: dict[tuple[str, int], dict[str, Any]] = {}
    with log_path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rec = json.loads(line)
                rows[(rec["scenario_id"], int(rec.get("seq_idx", rec["turn_idx"])))] = rec
    return rows


def _render_markdown(report: dict[str, Any]) -> str:
    lt = report["latency_and_throughput"]
    ce = report["cache_efficiency"]
    td = report["trajectory_determinism"]
    bp = report["behavioral_parity"]
    he = report["hypothesis_evaluation"]
    lines = [
        "# Cache OFF vs Cache ON Comparison Report",
        "",
        f"- **Target TTFT Reduction**: `{report['target_ttft_reduction_pct']}%`",
        f"- **Overall Hypothesis Pass**: `{he['overall_hypothesis_pass']}` (`ttft_target_met={he['ttft_target_met']}`, `quality_parity_pass={he['quality_parity_pass']}`)",
        "",
        "## Latency & Throughput",
        f"Source: `{lt['source']}`",
        "",
        "| Metric | Cache OFF | Cache ON | Delta / Speedup |",
        "| :--- | ---: | ---: | ---: |",
        f"| Median TTFT (ms) | {lt['off_median_ttft_ms']} | {lt['on_median_ttft_ms']} | {lt['median_ttft_reduction_pct']}% reduction |",
        f"| P95 TTFT (ms) | {lt['off_p95_ttft_ms']} | {lt['on_p95_ttft_ms']} | {lt['p95_ttft_reduction_pct']}% reduction |",
        f"| Median Latency (ms) | {lt['off_median_latency_ms']} | {lt['on_median_latency_ms']} | {lt['median_latency_reduction_pct']}% reduction |",
        f"| Successful RPS | {lt['off_rps']} | {lt['on_rps']} | {lt['rps_speedup_ratio']}x |",
        f"| Total Wall (ms) | {lt['off_wall_ms']} | {lt['on_wall_ms']} | {lt['wall_ms_reduction_pct']}% reduction |",
        "",
        "## Cache Efficiency & Trajectory Determinism",
        f"- Theoretical intra-session reuse ratio: `{ce['theoretical_intra_session_reuse_ratio']}`",
        f"- Empirical cache hit rate: OFF=`{ce['off_empirical_cache_hit_rate']}`, ON=`{ce['on_empirical_cache_hit_rate']}`",
        f"- Paired requests: `{td['paired_requests']}/{td['total_max_requests']}`, exact request match: `{td['exact_request_match_rate']}`, action agreement: `{td['action_agreement_rate']}`",
        "",
        "## Behavioral Parity",
        f"- Micro F1: OFF=`{bp['off_micro_f1']}`, ON=`{bp['on_micro_f1']}` (delta=`{bp['micro_f1_delta']}`)",
        f"- Verified count: OFF=`{bp['off_verified_count']}`, ON=`{bp['on_verified_count']}` (non-regression=`{bp['verified_non_regression']}`)",
        f"- Unnecessary turns: OFF=`{bp['off_unnecessary_turns']}`, ON=`{bp['on_unnecessary_turns']}`",
        f"- Unauthorized count: OFF=`{bp['off_unauthorized_count']}`, ON=`{bp['on_unauthorized_count']}`",
    ]
    return "\n".join(lines) + "\n"


def compare_runs(
    off_summary_path: Path,
    on_summary_path: Path,
    out_path: Path,
    off_log_path: Path | None = None,
    on_log_path: Path | None = None,
    off_score_path: Path | None = None,
    on_score_path: Path | None = None,
    off_replay_path: Path | None = None,
    on_replay_path: Path | None = None,
    target_ttft_reduction_pct: float = 20.0,
    out_md_path: Path | None = None,
) -> dict[str, Any]:
    """Compare Cache OFF vs Cache ON summaries, logs, scores, and replay benchmarks."""
    off_sum = json.loads(off_summary_path.read_text(encoding="utf-8"))
    on_sum = json.loads(on_summary_path.read_text(encoding="utf-8"))
    off_rep = json.loads(off_replay_path.read_text(encoding="utf-8")) if off_replay_path else None
    on_rep = json.loads(on_replay_path.read_text(encoding="utf-8")) if on_replay_path else None

    if off_rep is not None and on_rep is not None:
        source = "replay_benchmark"
        off_m, on_m = off_rep["aggregate"], on_rep["aggregate"]
    else:
        source = "run_summary"
        off_m, on_m = off_sum, on_sum

    off_med_ttft, on_med_ttft = off_m.get("median_ttft_ms"), on_m.get("median_ttft_ms")
    off_p95_ttft, on_p95_ttft = off_m.get("p95_ttft_ms"), on_m.get("p95_ttft_ms")
    off_med_lat, on_med_lat = off_m.get("median_latency_ms"), on_m.get("median_latency_ms")
    off_rps, on_rps = off_m.get("successful_rps"), on_m.get("successful_rps")
    off_wall, on_wall = off_m.get("total_wall_ms"), on_m.get("total_wall_ms")

    med_ttft_red = _reduction_pct(off_med_ttft, on_med_ttft)
    rps_speedup = round(float(on_rps) / float(off_rps), 3) if (off_rps and on_rps is not None and float(off_rps) > 0) else None

    latency_and_throughput = {
        "source": source,
        "off_median_ttft_ms": off_med_ttft,
        "on_median_ttft_ms": on_med_ttft,
        "median_ttft_reduction_pct": med_ttft_red,
        "off_p95_ttft_ms": off_p95_ttft,
        "on_p95_ttft_ms": on_p95_ttft,
        "p95_ttft_reduction_pct": _reduction_pct(off_p95_ttft, on_p95_ttft),
        "off_median_latency_ms": off_med_lat,
        "on_median_latency_ms": on_med_lat,
        "median_latency_reduction_pct": _reduction_pct(off_med_lat, on_med_lat),
        "off_rps": off_rps,
        "on_rps": on_rps,
        "rps_speedup_ratio": rps_speedup,
        "off_wall_ms": off_wall,
        "on_wall_ms": on_wall,
        "wall_ms_reduction_pct": _reduction_pct(off_wall, on_wall),
    }

    cache_efficiency = {
        "theoretical_intra_session_reuse_ratio": (on_sum.get("prefix_reuse_stats") or {}).get("intra_session_reuse_ratio"),
        "off_empirical_cache_hit_rate": _empirical_hit_rate(off_sum, off_rep),
        "on_empirical_cache_hit_rate": _empirical_hit_rate(on_sum, on_rep),
    }

    if off_log_path and on_log_path:
        off_rows, on_rows = _load_log_map(off_log_path), _load_log_map(on_log_path)
        paired_keys = sorted(set(off_rows) & set(on_rows))
        max_reqs = max(len(off_rows), len(on_rows))
        req_matches = sum(1 for k in paired_keys if off_rows[k].get("llm_request_messages") == on_rows[k].get("llm_request_messages"))
        act_matches = sum(1 for k in paired_keys if off_rows[k].get("action") == on_rows[k].get("action"))
        trajectory_determinism = {
            "paired_requests": len(paired_keys),
            "total_max_requests": max_reqs,
            "exact_request_match_rate": round(req_matches / max_reqs, 6) if max_reqs else 1.0,
            "action_agreement_rate": round(act_matches / max_reqs, 6) if max_reqs else 1.0,
        }
    else:
        trajectory_determinism = {
            "paired_requests": 0, "total_max_requests": 0,
            "exact_request_match_rate": None, "action_agreement_rate": None,
        }

    if off_score_path and on_score_path:
        off_ov = json.loads(off_score_path.read_text(encoding="utf-8"))["overall"]
        on_ov = json.loads(on_score_path.read_text(encoding="utf-8"))["overall"]
        off_f1, on_f1 = float(off_ov["micro_need_discovery_f1"]), float(on_ov["micro_need_discovery_f1"])
        off_ver, on_ver = int(off_ov["verified_count"]), int(on_ov["verified_count"])
        off_unnec, on_unnec = int(off_ov["total_unnecessary_turns"]), int(on_ov["total_unnecessary_turns"])
        off_unauth, on_unauth = int(off_ov["unauthorized_count"]), int(on_ov["unauthorized_count"])
        ver_non_reg = on_ver >= off_ver
        quality_pass = ver_non_reg and (on_f1 >= off_f1 - 1e-6) and (on_unauth <= off_unauth)
        behavioral_parity = {
            "off_micro_f1": off_f1, "on_micro_f1": on_f1, "micro_f1_delta": round(on_f1 - off_f1, 6),
            "off_verified_count": off_ver, "on_verified_count": on_ver, "verified_non_regression": ver_non_reg,
            "off_unnecessary_turns": off_unnec, "on_unnecessary_turns": on_unnec,
            "off_unauthorized_count": off_unauth, "on_unauthorized_count": on_unauth,
            "quality_parity_pass": quality_pass,
        }
    else:
        quality_pass = True
        behavioral_parity = {
            "off_micro_f1": None, "on_micro_f1": None, "micro_f1_delta": None,
            "off_verified_count": None, "on_verified_count": None, "verified_non_regression": None,
            "off_unnecessary_turns": None, "on_unnecessary_turns": None,
            "off_unauthorized_count": None, "on_unauthorized_count": None,
            "quality_parity_pass": None,
        }

    ttft_met = med_ttft_red is not None and med_ttft_red >= target_ttft_reduction_pct
    report = {
        "schema_version": SCHEMA_VERSION,
        "target_ttft_reduction_pct": target_ttft_reduction_pct,
        "latency_and_throughput": latency_and_throughput,
        "cache_efficiency": cache_efficiency,
        "trajectory_determinism": trajectory_determinism,
        "behavioral_parity": behavioral_parity,
        "hypothesis_evaluation": {
            "ttft_target_met": ttft_met,
            "quality_parity_pass": quality_pass if (off_score_path and on_score_path) else None,
            "overall_hypothesis_pass": ttft_met and quality_pass,
        },
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if out_md_path is not None:
        out_md_path.parent.mkdir(parents=True, exist_ok=True)
        out_md_path.write_text(_render_markdown(report), encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Compare Cache OFF vs Cache ON benchmark results.")
    p.add_argument("--off-summary", required=True, type=Path)
    p.add_argument("--on-summary", required=True, type=Path)
    p.add_argument("--off-log", default=None, type=Path)
    p.add_argument("--on-log", default=None, type=Path)
    p.add_argument("--off-score", default=None, type=Path)
    p.add_argument("--on-score", default=None, type=Path)
    p.add_argument("--off-replay", default=None, type=Path)
    p.add_argument("--on-replay", default=None, type=Path)
    p.add_argument("--target-ttft-reduction-pct", type=float, default=20.0)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--out-md", default=None, type=Path)
    a = p.parse_args(argv)
    compare_runs(
        a.off_summary, a.on_summary, a.out, a.off_log, a.on_log,
        a.off_score, a.on_score, a.off_replay, a.on_replay,
        a.target_ttft_reduction_pct, a.out_md,
    )


if __name__ == "__main__":
    main()
