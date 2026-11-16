"""CLI 4: Stream-export replay_requests.jsonl and exported_actions.jsonl from run_log.jsonl."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def export_logs(log_path: Path, out_requests: Path, out_actions: Path) -> tuple[int, int]:
    """Stream run_log.jsonl into order-preserving replay requests and compact action summaries."""
    out_requests.parent.mkdir(parents=True, exist_ok=True)
    out_actions.parent.mkdir(parents=True, exist_ok=True)
    seq = 0
    with (
        log_path.open("r", encoding="utf-8") as src,
        out_requests.open("w", encoding="utf-8") as req_f,
        out_actions.open("w", encoding="utf-8") as act_f,
    ):
        for line in src:
            if not line.strip():
                continue
            rec = json.loads(line)
            step_idx = rec.get("step_idx", 0)
            seq_idx = rec.get("seq_idx", rec["turn_idx"])
            req_f.write(json.dumps({
                "replay_seq": seq, "session_id": rec["session_id"], "scenario_id": rec["scenario_id"],
                "turn_idx": rec["turn_idx"], "step_idx": step_idx, "seq_idx": seq_idx,
                "request_id": rec["request_id"], "prev_request_id": rec["prev_request_id"],
                "inference_config": rec["inference_config"], "messages": rec["llm_request_messages"],
            }, ensure_ascii=False) + "\n")
            act_f.write(json.dumps({
                "scenario_id": rec["scenario_id"], "turn_idx": rec["turn_idx"],
                "step_idx": step_idx, "seq_idx": seq_idx, "replay_time": rec["replay_time"],
                "new_obs_ids": rec["new_obs_ids"], "action": rec["action"],
                "tool_result": rec["tool_result"], "timing_ms": rec["timing_ms"], "error": rec["error"],
            }, ensure_ascii=False) + "\n")
            seq += 1
    return seq, seq


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Export replay requests and compact action summaries.")
    p.add_argument("--log", required=True, type=Path)
    p.add_argument("--out-requests", required=True, type=Path)
    p.add_argument("--out-actions", required=True, type=Path)
    a = p.parse_args(argv)
    export_logs(a.log, a.out_requests, a.out_actions)


if __name__ == "__main__":
    main()
