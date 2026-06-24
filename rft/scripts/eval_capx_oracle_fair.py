#!/usr/bin/env python3
"""Fair CaP-X oracle-code evaluation without model generation."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from collections import Counter
from pathlib import Path
from typing import Any

from eval_capx_fair import (
    _build_prompt_messages,
    _error_bucket,
    _jsonable,
    _patch_robosuite_body_name_compat,
    _score_records_parallel,
    _seed_list,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--label", default="oracle")
    parser.add_argument("--data-source", default="franka_lift_code_env")
    parser.add_argument("--seed-base", type=int, default=60000)
    parser.add_argument("--num-trials", type=int, default=100)
    parser.add_argument("--seeds", default=None)
    parser.add_argument("--reward-workers", type=int, default=1)
    return parser.parse_args()


def _oracle_code(data_source: str) -> str:
    _patch_robosuite_body_name_compat()
    from capx.envs.tasks import get_exec_env

    env_cls = get_exec_env(data_source)
    return str(env_cls.oracle_code).strip()


def _records(args: argparse.Namespace, code: str, seeds: list[int]) -> list[dict[str, Any]]:
    return [
        {
            "label": args.label,
            "data_source": args.data_source,
            "seed": seed,
            "sample_idx": 0,
            "model_path": "oracle_code",
            "completion": code,
            "finish_reason": "oracle",
            "token_ids": [],
        }
        for seed in seeds
    ]


def _summary(
    args: argparse.Namespace,
    prompt_messages: list[dict[str, str]],
    oracle_code: str,
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
    return {
        "label": args.label,
        "model_path": "oracle_code",
        "data_source": args.data_source,
        "num_trials": len(by_seed),
        "samples_per_seed": 1,
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
        "oracle_code_sha256": hashlib.sha256(oracle_code.encode("utf-8")).hexdigest(),
        "oracle_code": oracle_code,
        "prompt_messages": prompt_messages,
    }


def main() -> None:
    args = _parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    prompt_messages = _build_prompt_messages(args.data_source)
    code = _oracle_code(args.data_source)
    seeds = _seed_list(args)
    scored = _score_records_parallel(_records(args, code, seeds), num_workers=args.reward_workers)
    summary = _summary(args, prompt_messages, code, scored)

    records_path = output_dir / "records.jsonl"
    with records_path.open("w", encoding="utf-8") as f:
        for record in scored:
            f.write(json.dumps(_jsonable(record), ensure_ascii=False) + "\n")
    (output_dir / "summary.json").write_text(
        json.dumps(_jsonable(summary), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(_jsonable(summary), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
