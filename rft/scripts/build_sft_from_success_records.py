#!/usr/bin/env python3
"""Build a format-stabilized SFT dataset from successful CaP-X rollout records."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", required=True, help="Input records.jsonl from eval_capx_fair.py.")
    parser.add_argument("--summary", required=True, help="Input summary.json with prompt_messages.")
    parser.add_argument("--output", required=True, help="Output SFT JSONL.")
    parser.add_argument("--summary-output", required=True, help="Output dataset summary JSON.")
    parser.add_argument("--max-examples", type=int, default=0, help="Optional cap after deduplication.")
    parser.add_argument(
        "--no-dedup",
        action="store_true",
        help="Keep duplicate successful completions instead of deduplicating by normalized completion text.",
    )
    return parser.parse_args()


def _strip_code_fence(text: str) -> str:
    text = text.strip()
    fence = re.fullmatch(r"```(?:python)?\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    if fence:
        return fence.group(1).strip()
    text = re.sub(r"^\s*```(?:python)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```\s*$", "", text)
    return text.strip()


def _normalise_completion(text: str) -> str:
    text = _strip_code_fence(text)
    lines = [line.rstrip() for line in text.splitlines()]
    text = "\n".join(lines).strip()
    if "np." in text and "import numpy as np" not in text:
        text = "import numpy as np\n\n" + text
    return text + "\n"


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def main() -> None:
    args = _parse_args()
    records_path = Path(args.records)
    summary_path = Path(args.summary)
    output_path = Path(args.output)
    summary_output_path = Path(args.summary_output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_output_path.parent.mkdir(parents=True, exist_ok=True)

    prompt_messages = json.loads(summary_path.read_text(encoding="utf-8"))["prompt_messages"]
    rows = _load_jsonl(records_path)

    examples: list[dict[str, Any]] = []
    seen: set[str] = set()
    skipped = {"not_success": 0, "empty": 0, "duplicate": 0}
    for row in rows:
        if not row.get("task_success"):
            skipped["not_success"] += 1
            continue
        completion = _normalise_completion(str(row.get("completion", "")))
        if not completion.strip():
            skipped["empty"] += 1
            continue
        key = hashlib.sha256(completion.encode("utf-8")).hexdigest()
        if not args.no_dedup and key in seen:
            skipped["duplicate"] += 1
            continue
        seen.add(key)
        examples.append(
            {
                "messages": [
                    *prompt_messages,
                    {"role": "assistant", "content": completion},
                ],
                "source": {
                    "record_label": row.get("label"),
                    "seed": row.get("seed"),
                    "sample_idx": row.get("sample_idx"),
                    "score": row.get("score"),
                    "task_success": row.get("task_success"),
                    "completion_sha256": key,
                },
            }
        )
        if args.max_examples and len(examples) >= args.max_examples:
            break

    with output_path.open("w", encoding="utf-8") as f:
        for example in examples:
            f.write(json.dumps(example, ensure_ascii=False) + "\n")

    summary = {
        "records": str(records_path),
        "summary": str(summary_path),
        "output": str(output_path),
        "input_records": len(rows),
        "num_examples": len(examples),
        "unique_completion_count": len(seen),
        "dedup": not args.no_dedup,
        "skipped": skipped,
        "max_examples": args.max_examples,
    }
    summary_output_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
