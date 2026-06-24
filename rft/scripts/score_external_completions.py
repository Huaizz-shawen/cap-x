#!/usr/bin/env python3
"""Score externally generated CaP-X completions with the standard evaluator."""

from __future__ import annotations

import argparse
import functools
import hashlib
import json
import os
import statistics
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from typing import Any

os.environ.setdefault("MUJOCO_GL", "egl")

from eval_capx_fair import (  # noqa: E402
    _build_prompt_messages,
    _error_bucket,
    _jsonable,
    _score_records_parallel,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Raw candidate JSONL from collect_api_completions.py.")
    parser.add_argument("--output-dir", required=True, help="Directory for scored records and summary.")
    parser.add_argument("--label", default=None, help="Override label stored in output records.")
    parser.add_argument("--data-source", default="franka_lift_code_env")
    parser.add_argument("--prompt-summary", default=None, help="Optional summary.json containing prompt_messages.")
    parser.add_argument("--reward-workers", type=int, default=1)
    return parser.parse_args()


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _load_prompt_messages(args: argparse.Namespace) -> list[dict[str, str]]:
    if args.prompt_summary:
        summary = json.loads(Path(args.prompt_summary).read_text(encoding="utf-8"))
        return [
            {"role": str(item.get("role", "user")), "content": str(item.get("content", ""))}
            for item in summary["prompt_messages"]
        ]
    return _build_prompt_messages(args.data_source)


def _summary(
    *,
    args: argparse.Namespace,
    prompt_messages: list[dict[str, str]],
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    total = len(records)
    successes = sum(1 for r in records if r["task_success"])
    valid = sum(1 for r in records if r["valid_execution"])
    scores = [float(r["score"]) for r in records]
    errors = Counter(_error_bucket(str(r.get("error", ""))) for r in records)
    by_seed: dict[int, list[dict[str, Any]]] = {}
    for record in records:
        by_seed.setdefault(int(record["seed"]), []).append(record)
    pass_at_k = {
        str(seed): any(item["task_success"] for item in items)
        for seed, items in sorted(by_seed.items())
    }
    model_names = sorted(
        {
            str(r.get("teacher_model") or r.get("model_path") or "")
            for r in records
            if r.get("teacher_model") or r.get("model_path")
        }
    )
    generation_values = [r.get("generation") for r in records if isinstance(r.get("generation"), dict)]
    return {
        "label": args.label or (records[0].get("label") if records else "external"),
        "model_path": model_names[0] if len(model_names) == 1 else model_names,
        "data_source": args.data_source,
        "num_trials": len(by_seed),
        "samples_per_seed": max((len(items) for items in by_seed.values()), default=0),
        "num_records": total,
        "success_count": successes,
        "success_rate": successes / total if total else 0.0,
        "pass_at_k_success_count": sum(1 for ok in pass_at_k.values() if ok),
        "pass_at_k_success_rate": (
            sum(1 for ok in pass_at_k.values() if ok) / len(pass_at_k) if pass_at_k else 0.0
        ),
        "valid_execution_count": valid,
        "valid_execution_rate": valid / total if total else 0.0,
        "mean_score": statistics.fmean(scores) if scores else 0.0,
        "error_buckets": dict(errors),
        "seeds": sorted(by_seed),
        "generation": generation_values[0] if len(generation_values) == 1 else generation_values[:5],
        "prompt_sha256": hashlib.sha256(
            json.dumps(prompt_messages, sort_keys=True, ensure_ascii=False).encode("utf-8")
        ).hexdigest(),
        "prompt_messages": prompt_messages,
        "input": str(args.input),
    }


def main() -> None:
    args = _parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_records = _load_jsonl(Path(args.input))
    records: list[dict[str, Any]] = []
    for idx, row in enumerate(raw_records):
        record = dict(row)
        record["data_source"] = str(record.get("data_source") or args.data_source)
        if args.label:
            record["label"] = args.label
        else:
            record["label"] = str(record.get("label") or "external")
        record["seed"] = int(record.get("seed", idx))
        record["sample_idx"] = int(record.get("sample_idx", 0))
        record["completion"] = str(record.get("completion", ""))
        records.append(record)

    scored = _score_records_parallel(records, num_workers=args.reward_workers)
    prompt_messages = _load_prompt_messages(args)
    summary = _summary(args=args, prompt_messages=prompt_messages, records=scored)

    with (output_dir / "records.jsonl").open("w", encoding="utf-8") as f:
        for record in scored:
            f.write(json.dumps(_jsonable(record), ensure_ascii=False) + "\n")
    (output_dir / "summary.json").write_text(
        json.dumps(_jsonable(summary), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(_jsonable(summary), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
