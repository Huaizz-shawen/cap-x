#!/usr/bin/env python3
"""Collect teacher-model completions through an OpenAI-compatible chat API.

This script is intended to run on a machine with external network access. It
does not evaluate trajectories. The output JSONL can be uploaded to Inspire and
scored offline with score_external_completions.py.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import copy
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--prompt-summary",
        default=None,
        help="summary.json containing prompt_messages. If omitted, build prompt from local CaP-X.",
    )
    parser.add_argument("--output", required=True, help="Output raw candidate JSONL.")
    parser.add_argument("--model", default=None, help="Teacher API model name.")
    parser.add_argument("--label", default="teacher_api")
    parser.add_argument("--data-source", default="franka_lift_code_env")
    parser.add_argument(
        "--client",
        choices=("capx", "openai"),
        default="capx",
        help="Use capx.llm.client when available, or a minimal OpenAI-compatible fallback.",
    )
    parser.add_argument("--capx-root", default="/media/user/B29202FA9202C2B91/cap-x")
    parser.add_argument(
        "--api-args-script",
        default=None,
        help="Optional CaP-X .capx_api script that prints launch.py-style args.",
    )
    parser.add_argument("--api-base", default=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"))
    parser.add_argument(
        "--server-url",
        default=None,
        help="Full chat/responses endpoint for capx client. Defaults to API_BASE/chat/completions.",
    )
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--seed-base", type=int, default=70000)
    parser.add_argument("--num-trials", type=int, default=128)
    parser.add_argument("--seeds", default=None, help="Optional comma-separated seed list.")
    parser.add_argument("--samples-per-seed", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--max-retries", type=int, default=6)
    parser.add_argument("--retry-sleep", type=float, default=2.0)
    parser.add_argument("--workers", type=int, default=1, help="Number of concurrent API requests.")
    parser.add_argument("--resume", action="store_true", help="Append to output and skip existing seed/sample pairs.")
    parser.add_argument(
        "--send-seed",
        action="store_true",
        help="Send an OpenAI-compatible seed field. Leave off for providers that reject it.",
    )
    return parser.parse_args()


def _seed_list(args: argparse.Namespace) -> list[int]:
    if args.seeds:
        return [int(part.strip()) for part in args.seeds.split(",") if part.strip()]
    return [args.seed_base + idx for idx in range(args.num_trials)]


def _load_prompt_messages(path: Path) -> list[dict[str, str]]:
    summary = json.loads(path.read_text(encoding="utf-8"))
    messages = summary["prompt_messages"]
    return [
        {"role": str(item.get("role", "user")), "content": str(item.get("content", ""))}
        for item in messages
    ]


def _build_prompt_messages_from_capx(capx_root: str, data_source: str) -> list[dict[str, str]]:
    if capx_root and capx_root not in sys.path:
        sys.path.insert(0, capx_root)
    from capx.envs.tasks import get_config, get_exec_env

    old_cwd = os.getcwd()
    os.chdir(capx_root)
    env = get_exec_env(data_source)(get_config(data_source))
    try:
        obs = env._get_observation()
        messages = obs["full_prompt"]
    finally:
        env.close()
        os.chdir(old_cwd)
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


def _load_api_args_script(path: str) -> dict[str, str]:
    import subprocess

    items = subprocess.check_output(["bash", path], text=True).splitlines()
    return dict(zip(items[0::2], items[1::2]))


def _request_chat_completion(
    *,
    api_base: str,
    api_key: str,
    payload: dict[str, Any],
    timeout: float,
    max_retries: int,
    retry_sleep: float,
) -> dict[str, Any]:
    url = api_base.rstrip("/") + "/chat/completions"
    body = json.dumps(payload).encode("utf-8")
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    last_error = ""
    for attempt in range(max_retries + 1):
        request = urllib.request.Request(url, data=body, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            last_error = f"HTTP {exc.code}: {detail}"
            if exc.code < 500 and exc.code not in {408, 409, 429}:
                break
        except Exception as exc:  # noqa: BLE001
            last_error = repr(exc)
        if attempt < max_retries:
            time.sleep(retry_sleep * (2**min(attempt, 4)))
    raise RuntimeError(last_error)


def _request_capx_completion(
    *,
    capx_root: str,
    server_url: str,
    api_key: str,
    model: str,
    messages: list[dict[str, str]],
    temperature: float,
    max_tokens: int,
    timeout: float,
    max_retries: int,
) -> dict[str, Any]:
    if capx_root and capx_root not in sys.path:
        sys.path.insert(0, capx_root)
    from capx.llm.client import ModelQueryArgs, query_model

    _ = timeout, max_retries
    result = query_model(
        ModelQueryArgs(
            model=model,
            server_url=server_url,
            api_key=api_key,
            temperature=temperature,
            max_tokens=max_tokens,
        ),
        copy.deepcopy(messages),
    )

    return {
        "content": str(result.get("content", "")),
        "finish_reason": result.get("finish_reason"),
        "reasoning": result.get("reasoning"),
        "raw_response": result.get("raw_response"),
        "stream_fallback_used": bool(result.get("stream_fallback_used", False)),
    }


def _load_existing_keys(path: Path) -> set[tuple[int, int]]:
    if not path.exists():
        return set()
    keys: set[tuple[int, int]] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
            keys.add((int(row["seed"]), int(row["sample_idx"])))
        except Exception:
            continue
    return keys


def _collect_one(
    *,
    args: argparse.Namespace,
    api_key: str,
    server_url: str,
    prompt_messages: list[dict[str, str]],
    seed: int,
    sample_idx: int,
) -> dict[str, Any]:
    if args.client == "capx":
        response = _request_capx_completion(
            capx_root=args.capx_root,
            server_url=server_url,
            api_key=api_key,
            model=args.model,
            messages=prompt_messages,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            timeout=args.timeout,
            max_retries=args.max_retries,
        )
        completion = response["content"]
        finish_reason = str(response.get("finish_reason") or "")
        usage = (response.get("raw_response") or {}).get("usage", {})
        response_id = (response.get("raw_response") or {}).get("id")
        raw_response = response.get("raw_response")
        reasoning = response.get("reasoning")
    else:
        payload: dict[str, Any] = {
            "model": args.model,
            "messages": prompt_messages,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "max_tokens": args.max_tokens,
        }
        if args.send_seed:
            payload["seed"] = seed * 1000 + sample_idx
        response = _request_chat_completion(
            api_base=args.api_base,
            api_key=api_key,
            payload=payload,
            timeout=args.timeout,
            max_retries=args.max_retries,
            retry_sleep=args.retry_sleep,
        )
        choice = response.get("choices", [{}])[0]
        message = choice.get("message") or {}
        completion = str(message.get("content", ""))
        finish_reason = str(choice.get("finish_reason", ""))
        usage = response.get("usage", {})
        response_id = response.get("id")
        raw_response = response
        reasoning = message.get("reasoning")

    return {
        "label": args.label,
        "data_source": args.data_source,
        "seed": seed,
        "sample_idx": sample_idx,
        "teacher_model": args.model,
        "api_base": args.api_base,
        "server_url": server_url,
        "client": args.client,
        "completion": completion,
        "finish_reason": finish_reason,
        "usage": usage,
        "response_id": response_id,
        "reasoning": reasoning,
        "raw_response": raw_response,
        "generation": {
            "max_tokens": args.max_tokens,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "send_seed": args.send_seed,
        },
    }


def main() -> None:
    args = _parse_args()
    script_params = _load_api_args_script(args.api_args_script) if args.api_args_script else {}
    if script_params.get("--model"):
        args.model = script_params["--model"]
    if script_params.get("--server-url") and not args.server_url:
        args.server_url = script_params["--server-url"]
    api_key = script_params.get("--api-key") or os.environ.get(args.api_key_env)
    if not api_key:
        raise SystemExit(f"Missing API key in --api-args-script or ${args.api_key_env}")
    if not args.model:
        raise SystemExit("Missing model in --model or --api-args-script")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if args.prompt_summary:
        prompt_messages = _load_prompt_messages(Path(args.prompt_summary))
    else:
        prompt_messages = _build_prompt_messages_from_capx(args.capx_root, args.data_source)
    seeds = _seed_list(args)
    server_url = args.server_url or (args.api_base.rstrip("/") + "/chat/completions")
    os.environ["CAPX_MODEL_REQUEST_TIMEOUT_S"] = str(args.timeout)
    os.environ["CAPX_MODEL_RETRY_MAX_ATTEMPTS"] = str(args.max_retries)

    existing = _load_existing_keys(output_path) if args.resume else set()
    requests = [
        (seed, sample_idx)
        for seed in seeds
        for sample_idx in range(args.samples_per_seed)
        if (seed, sample_idx) not in existing
    ]
    mode = "a" if args.resume and output_path.exists() else "w"
    print(
        json.dumps(
            {
                "output": str(output_path),
                "model": args.model,
                "client": args.client,
                "server_url": server_url,
                "total_requested": len(seeds) * args.samples_per_seed,
                "skipped_existing": len(existing),
                "to_collect": len(requests),
                "workers": args.workers,
            },
            ensure_ascii=False,
        )
    )

    with output_path.open(mode, encoding="utf-8") as f:
        if args.workers <= 1:
            for seed, sample_idx in requests:
                record = _collect_one(
                    args=args,
                    api_key=api_key,
                    server_url=server_url,
                    prompt_messages=prompt_messages,
                    seed=seed,
                    sample_idx=sample_idx,
                )
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                f.flush()
        else:
            with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
                futures = [
                    executor.submit(
                        _collect_one,
                        args=args,
                        api_key=api_key,
                        server_url=server_url,
                        prompt_messages=prompt_messages,
                        seed=seed,
                        sample_idx=sample_idx,
                    )
                    for seed, sample_idx in requests
                ]
                for future in concurrent.futures.as_completed(futures):
                    record = future.result()
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    f.flush()


if __name__ == "__main__":
    main()
