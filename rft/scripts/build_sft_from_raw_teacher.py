#!/usr/bin/env python3
"""Build SFT data directly from raw teacher API completions."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", required=True, help="Raw JSONL from collect_api_completions.py.")
    parser.add_argument("--prompt-summary", required=True, help="summary.json containing prompt_messages.")
    parser.add_argument("--output", required=True, help="Output SFT JSONL.")
    parser.add_argument("--summary-output", required=True, help="Output dataset summary JSON.")
    parser.add_argument("--max-examples", type=int, default=0)
    parser.add_argument("--dedup", action="store_true", help="Deduplicate by normalized completion.")
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
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    args = _parse_args()
    raw_path = Path(args.raw)
    output_path = Path(args.output)
    summary_output_path = Path(args.summary_output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_output_path.parent.mkdir(parents=True, exist_ok=True)

    prompt_messages = json.loads(Path(args.prompt_summary).read_text(encoding="utf-8"))[
        "prompt_messages"
    ]
    rows = _load_jsonl(raw_path)

    examples: list[dict[str, Any]] = []
    seen: set[str] = set()
    skipped = {"empty": 0, "duplicate": 0}
    raw_unique: set[str] = set()
    normalized_unique: set[str] = set()
    for row in rows:
        raw_completion = str(row.get("completion", ""))
        raw_key = hashlib.sha256(raw_completion.strip().encode("utf-8")).hexdigest()
        raw_unique.add(raw_key)
        completion = _normalise_completion(raw_completion)
        if not completion.strip():
            skipped["empty"] += 1
            continue
        key = hashlib.sha256(completion.encode("utf-8")).hexdigest()
        normalized_unique.add(key)
        if args.dedup and key in seen:
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
                    "teacher_model": row.get("teacher_model"),
                    "completion_sha256": key,
                    "raw_teacher_unscored": True,
                },
            }
        )
        if args.max_examples and len(examples) >= args.max_examples:
            break

    with output_path.open("w", encoding="utf-8") as f:
        for example in examples:
            f.write(json.dumps(example, ensure_ascii=False) + "\n")

    summary = {
        "raw": str(raw_path),
        "prompt_summary": args.prompt_summary,
        "output": str(output_path),
        "input_records": len(rows),
        "num_examples": len(examples),
        "raw_unique_completion_count": len(raw_unique),
        "normalized_unique_completion_count": len(normalized_unique),
        "dedup": args.dedup,
        "skipped": skipped,
        "max_examples": args.max_examples,
        "note": "Built from raw teacher completions without full environment scoring.",
    }
    summary_output_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
