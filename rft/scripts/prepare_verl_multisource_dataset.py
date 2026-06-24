#!/usr/bin/env python3
"""Build a balanced VeRL dataset from multiple CaP-X code environments."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--data-sources",
        required=True,
        help="Comma-separated CaP-X task sources, e.g. franka_lift_code_env,franka_restack_code_env.",
    )
    parser.add_argument("--train-size-per-source", type=int, default=256)
    parser.add_argument("--val-size-per-source", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--val-seed-offset", type=int, default=10000)
    parser.add_argument("--shuffle", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--write-jsonl",
        action="store_true",
        help="Also write train.jsonl/test.jsonl. If pyarrow is unavailable, JSONL is still written.",
    )
    parser.add_argument(
        "--jsonl-only",
        action="store_true",
        help="Write train.jsonl/test.jsonl and skip parquet output. Useful for local smoke checks.",
    )
    return parser.parse_args()


def _normalise_messages(messages: list[dict[str, Any]]) -> list[dict[str, str]]:
    normalised: list[dict[str, str]] = []
    for msg in messages:
        content = msg.get("content", "")
        if isinstance(content, list):
            parts = [
                str(part.get("text", ""))
                for part in content
                if isinstance(part, dict) and part.get("type") == "text"
            ]
            content = "\n".join(parts)
        normalised.append({"role": str(msg.get("role", "user")), "content": str(content)})
    return normalised


def _row(
    *,
    data_source: str,
    split: str,
    source_index: int,
    env_seed: int,
) -> dict[str, Any]:
    from capx.envs.tasks import get_config, get_exec_env

    env = get_exec_env(data_source)(get_config(data_source))
    try:
        obs, _ = env.reset(seed=env_seed)
        prompt = _normalise_messages(obs["full_prompt"])
        oracle_code = getattr(env, "oracle_code", None)
    finally:
        env.close()

    return {
        "data_source": data_source,
        "prompt": prompt,
        "ability": "agent",
        "reward_model": {
            "style": "sim_code",
            "ground_truth": {"program": oracle_code},
        },
        "extra_info": {
            "split": split,
            "source_index": source_index,
            "seed": env_seed,
        },
    }


def _build_rows(
    *,
    data_sources: list[str],
    split: str,
    size_per_source: int,
    seed_base: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for source_offset, data_source in enumerate(data_sources):
        source_seed_base = seed_base + source_offset * 1_000_000
        for source_index in range(size_per_source):
            rows.append(
                _row(
                    data_source=data_source,
                    split=split,
                    source_index=source_index,
                    env_seed=source_seed_base + source_index,
                )
            )
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _write_parquet(output_dir: Path, train_rows: list[dict[str, Any]], val_rows: list[dict[str, Any]]) -> None:
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "pyarrow is required for parquet output. Install pyarrow or pass --write-jsonl "
            "to at least materialize JSONL rows for local smoke checks."
        ) from exc

    pq.write_table(pa.Table.from_pylist(train_rows), output_dir / "train.parquet")
    pq.write_table(pa.Table.from_pylist(val_rows), output_dir / "test.parquet")


def main() -> None:
    args = _parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data_sources = [part.strip() for part in args.data_sources.split(",") if part.strip()]
    if not data_sources:
        raise SystemExit("At least one --data-sources entry is required")

    train_rows = _build_rows(
        data_sources=data_sources,
        split="train",
        size_per_source=args.train_size_per_source,
        seed_base=args.seed,
    )
    val_rows = _build_rows(
        data_sources=data_sources,
        split="val",
        size_per_source=args.val_size_per_source,
        seed_base=args.seed + args.val_seed_offset,
    )

    if args.shuffle:
        rng = random.Random(args.seed)
        rng.shuffle(train_rows)
        rng.shuffle(val_rows)

    if args.write_jsonl or args.jsonl_only:
        _write_jsonl(output_dir / "train.jsonl", train_rows)
        _write_jsonl(output_dir / "test.jsonl", val_rows)

    if not args.jsonl_only:
        _write_parquet(output_dir, train_rows, val_rows)

    manifest = {
        "data_sources": data_sources,
        "train_size_per_source": args.train_size_per_source,
        "val_size_per_source": args.val_size_per_source,
        "train": len(train_rows),
        "val": len(val_rows),
        "seed": args.seed,
        "val_seed_offset": args.val_seed_offset,
        "shuffle": args.shuffle,
        "jsonl_only": args.jsonl_only,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
