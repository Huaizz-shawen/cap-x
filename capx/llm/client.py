"""LLM client utilities for querying language models.

Extracted from capx/utils/launch_utils.py to separate LLM query logic
from launch/config utilities.
"""

from __future__ import annotations

import concurrent.futures
import copy
import hashlib
import json
import os
import random
import time
from collections.abc import Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import requests

if TYPE_CHECKING:
    from capx.envs.launch import LaunchArgs

# ---------------------------------------------------------------------------
# Model constants
# ---------------------------------------------------------------------------

GPT_MODELS = [
    "gpt-5",
    "azure/openai/gpt-5.1",
    "gpt-5.1",
    "openai/openai/gpt-5.1-codex",
    "openai/openai/gpt-5.1-codex-max",
    "azure/openai/o4-mini",
    "azure/openai/gpt-5.1-codex",
]
VLM_MODELS = [
    "gemini-3-pro",
    "gemini-3-flash-preview",
    "gemini-3-pro-preview",
    "gemini-2.5-flash-lite",
    "google/gemini-3.1-pro-preview",
    "google/gemini-3.1-pro",
    "google/gemini-2.5-flash-lite",
    "aws/anthropic/claude-opus-4-5",
    "aws/anthropic/claude-haiku-4-5-v1",
    "azure/openai/gpt-5.2",
    "azure/openai/gpt-5.1",
    "azure/openai/o1",
    "azure/openai/o4-mini",
    "openai/openai/gpt-5.2",
    "openai/openai/gpt-5.1",
    "nvdev/deepseek-ai/deepseek-v3-0324",
    "deepseek-ai/deepseek-v3.1-terminus",
    "deepseek-ai/deepseek-r1-0528",
    "deepseek-ai/deepseek-v3.1",
    "nvdev/deepseek-ai/deepseek-r1",
    "nvdev/qwen/qwen-235b",
    "moonshotai/kimi-k2-instruct-0905",
    "moonshotai/kimi-k2-instruct",
    "openai/openai/gpt-5.1-codex",
    "openai/openai/gpt-5.1-codex-max",
    "azure/openai/gpt-5.1-codex",
]
CLAUDE_MODELS = ["aws/anthropic/claude-opus-4-5", "aws/anthropic/claude-haiku-4-5-v1"]
OSS_MODELS = [
    "nvdev/deepseek-ai/deepseek-v3-0324",
    "deepseek-ai/deepseek-v3.1-terminus",
    "deepseek-ai/deepseek-r1-0528",
    "deepseek-ai/deepseek-v3.1",
    "nvdev/deepseek-ai/deepseek-r1",
    "nvdev/qwen/qwen-235b",
    "moonshotai/kimi-k2-instruct-0905",
    "moonshotai/kimi-k2-instruct",
]
OPENROUTER_MODELS = [
    "openrouter/google/gemini-2.5-pro-preview",
    "openrouter/google/gemini-2.5-flash-preview",
    "openrouter/anthropic/claude-sonnet-4",
    "openrouter/anthropic/claude-opus-4",
    "openrouter/deepseek/deepseek-r1",
    "openrouter/deepseek/deepseek-chat-v3-0324",
    "openrouter/openai/gpt-4.1",
    "openrouter/openai/o4-mini",
    "openrouter/meta-llama/llama-4-maverick",
    "openrouter/qwen/qwen3-235b-a22b",
]
OPENROUTER_SERVER_URL = "http://localhost:8110/chat/completions"
RETRYABLE_STATUS_CODES = {408, 409, 429, 500, 502, 503, 504, 520, 521, 522, 523, 524, 525, 526}
DEFAULT_REQUEST_TIMEOUT_S = 240.0
DEFAULT_RETRY_MAX_ATTEMPTS = 12
DEFAULT_RETRY_MAX_WALLTIME_S = 7200.0
DEFAULT_RETRY_INITIAL_S = 15.0
DEFAULT_RETRY_MAX_SLEEP_S = 240.0

# ---------------------------------------------------------------------------
# Ensemble configuration
# ---------------------------------------------------------------------------

ENSEMBLE_CONFIGS = [
    # Gemini-3-Pro only — best single model per CaP-Bench (Figure 1).
    # 3 temps for diversity; synthesis still uses Gemini-3-Pro.
    # ~45% faster than full multimodel (no Claude/GPT latency bottleneck).
    ("azure/openai/gpt-5.2", [0.1, 0.5, 0.9]),
]

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def is_openrouter_model(model: str) -> bool:
    """Return True if the model should be routed through the OpenRouter proxy."""
    return model.startswith("openrouter/") or model in OPENROUTER_MODELS


@dataclass
class ModelQueryArgs:
    """Arguments for querying a model."""

    model: str
    server_url: str
    api_key: str | None = None
    temperature: float = 0.2
    max_tokens: int = 4096
    reasoning_effort: str = "medium"
    debug: bool = False


@dataclass
class RequestRetryConfig:
    request_timeout_s: float = DEFAULT_REQUEST_TIMEOUT_S
    max_attempts: int = DEFAULT_RETRY_MAX_ATTEMPTS
    max_retry_walltime_s: float = DEFAULT_RETRY_MAX_WALLTIME_S
    retry_initial_s: float = DEFAULT_RETRY_INITIAL_S
    retry_max_sleep_s: float = DEFAULT_RETRY_MAX_SLEEP_S


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        parsed = float(value)
    except ValueError:
        return default
    if parsed <= 0:
        return default
    return parsed


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        parsed = int(value)
    except ValueError:
        return default
    if parsed <= 0:
        return default
    return parsed


def _env_float_list(name: str, default: list[float]) -> list[float]:
    value = os.getenv(name)
    if value is None:
        return default
    parsed: list[float] = []
    for token in value.split(","):
        token = token.strip()
        if not token:
            continue
        try:
            parsed.append(float(token))
        except ValueError:
            return default
    return parsed if parsed else default


def _load_request_retry_config() -> RequestRetryConfig:
    return RequestRetryConfig(
        request_timeout_s=_env_float("CAPX_MODEL_REQUEST_TIMEOUT_S", DEFAULT_REQUEST_TIMEOUT_S),
        max_attempts=_env_int("CAPX_MODEL_RETRY_MAX_ATTEMPTS", DEFAULT_RETRY_MAX_ATTEMPTS),
        max_retry_walltime_s=_env_float(
            "CAPX_MODEL_RETRY_MAX_WALLTIME_S",
            DEFAULT_RETRY_MAX_WALLTIME_S,
        ),
        retry_initial_s=_env_float("CAPX_MODEL_RETRY_INITIAL_S", DEFAULT_RETRY_INITIAL_S),
        retry_max_sleep_s=_env_float("CAPX_MODEL_RETRY_MAX_SLEEP_S", DEFAULT_RETRY_MAX_SLEEP_S),
    )


def _compute_retry_sleep_seconds(attempt: int, config: RequestRetryConfig) -> float:
    base = min(config.retry_initial_s * (2 ** max(0, attempt - 1)), config.retry_max_sleep_s)
    jitter = random.uniform(0.0, min(base * 0.25, 15.0))
    return min(base + jitter, config.retry_max_sleep_s)


def _is_retryable_exception(exc: requests.RequestException) -> bool:
    retryable = (
        requests.exceptions.Timeout,
        requests.exceptions.ConnectionError,
        requests.exceptions.ChunkedEncodingError,
    )
    return isinstance(exc, retryable)


def _is_retryable_status_code(status_code: int) -> bool:
    return status_code in RETRYABLE_STATUS_CODES


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _payload_structure_summary(payload: dict[str, Any]) -> dict[str, Any]:
    """Return a sanitized, structural summary for a model payload.

    The summary intentionally avoids logging raw prompt text or API keys.
    """
    compact = json.dumps(payload, separators=(",", ":"), ensure_ascii=False, default=str)
    summary: dict[str, Any] = {
        "model": payload.get("model"),
        "top_keys": sorted(payload.keys()),
        "payload_bytes": len(compact.encode("utf-8")),
        "payload_sha256_16": hashlib.sha256(compact.encode("utf-8")).hexdigest()[:16],
    }

    def collect_content_stats(items: list[Any], stats: dict[str, Any]) -> None:
        for item in items:
            if isinstance(item, str):
                stats["string_items"] += 1
                if stats["first_text_len"] is None and item:
                    stats["first_text_len"] = len(item)
                continue
            if not isinstance(item, dict):
                stats["other_items"] += 1
                continue
            item_type = item.get("type")
            if isinstance(item_type, str):
                stats["content_type_counts"][item_type] = stats["content_type_counts"].get(item_type, 0) + 1
                if item_type in {"image_url", "input_image"}:
                    stats["image_items"] += 1
            if stats["first_text_len"] is None:
                txt = item.get("text")
                if not isinstance(txt, str):
                    txt = item.get("input_text")
                if isinstance(txt, str) and txt:
                    stats["first_text_len"] = len(txt)

    if isinstance(payload.get("messages"), list):
        messages = payload["messages"]
        stats: dict[str, Any] = {
            "messages_count": len(messages),
            "content_items_count": 0,
            "content_type_counts": {},
            "image_items": 0,
            "string_items": 0,
            "other_items": 0,
            "first_text_len": None,
        }
        for msg in messages:
            content = msg.get("content") if isinstance(msg, dict) else None
            if isinstance(content, list):
                stats["content_items_count"] += len(content)
                collect_content_stats(content, stats)
            elif isinstance(content, str):
                stats["string_items"] += 1
                if stats["first_text_len"] is None and content:
                    stats["first_text_len"] = len(content)
        summary["messages"] = stats

    if isinstance(payload.get("input"), list):
        input_items = payload["input"]
        stats = {
            "input_items_count": len(input_items),
            "content_items_count": 0,
            "content_type_counts": {},
            "image_items": 0,
            "string_items": 0,
            "other_items": 0,
            "first_text_len": None,
        }
        for item in input_items:
            content = item.get("content") if isinstance(item, dict) else None
            if isinstance(content, list):
                stats["content_items_count"] += len(content)
                collect_content_stats(content, stats)
            elif isinstance(content, str):
                stats["string_items"] += 1
                if stats["first_text_len"] is None and content:
                    stats["first_text_len"] = len(content)
        summary["input"] = stats

    return summary


def collapse_text_image_inputs(messages: list[dict]) -> list[dict]:
    """
    Collapse a list of messages with sequential text into a single text input, images are still in the same relative position
    """
    new_prompt = []
    current_text_input = ""
    for message in messages:
        if message["type"] == "text":
            current_text_input += message["text"] + "\n"
        else:
            if current_text_input != "":
                new_prompt.append({"type": "text", "text": current_text_input})
                current_text_input = ""
            new_prompt.append(message)
    if current_text_input != "":
        new_prompt.append({"type": "text", "text": current_text_input})
    return new_prompt


def _completions_to_responses_convert_prompt(prompt: list[dict]) -> list[dict]:
    """Convert completions api format to responses api format.

    Args:
        prompt: The prompt in completions api format

    Returns:
        The prompt in responses api format

    Switch prompt structure to api responses api format e.g.:
    From
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Describe the image in detail."},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}"
                    }
                }
            ]
        }
    ]
    To
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "input_text", "text": "Describe the image in detail."},
                {
                    "type": "input_image",
                    "image_url": f"data:image/jpeg;base64,{base64_image}"
                }
            ]
        }
    ]
    """

    for message in prompt:
        for content in message["content"]:
            if type(content) == str:
                continue
            if content.get("type") == "text":
                content["type"] = "input_text"
                content["text"] = content.pop("text")

            elif content.get("type") == "image_url":
                content["type"] = "input_image"
                content["image_url"] = content["image_url"]["url"]
    return prompt


# ---------------------------------------------------------------------------
# Core query functions
# ---------------------------------------------------------------------------


def query_model(args: "LaunchArgs | ModelQueryArgs", prompt: list[dict]) -> str:
    """Query vLLM server for code generation.

    Args:
        args: Configuration with server URL and model settings
        prompt: Full prompt containing environment observation and possibly multi-turn decision prompt
    Returns:
        Model response content
    """

    # Route OpenRouter models to the OpenRouter proxy server
    if is_openrouter_model(args.model):
        server_url = OPENROUTER_SERVER_URL
    else:
        server_url = args.server_url

    if args.model in GPT_MODELS:
        if "codex" in args.model:
            prompt = _completions_to_responses_convert_prompt(prompt)
            payload = {
                "model": args.model,
                "input": prompt,
            }
        else:
            payload = {
                "model": args.model,
                "reasoning_effort": args.reasoning_effort,
                "max_completion_tokens": args.max_tokens,  # Total completion tokens = reasoning + output tokens
                "messages": prompt,
            }
    elif is_openrouter_model(args.model):
        payload = {
            "model": args.model,
            "messages": prompt,
            "temperature": args.temperature,
            "max_tokens": args.max_tokens,
        }
    elif args.model in CLAUDE_MODELS:
        payload = {
            "model": args.model,
            "temperature": args.temperature,
            "max_tokens": args.max_tokens,
            "thinking": {"type": "enabled", "budget_tokens": 4096},
            "messages": prompt,
        }
    elif args.model in OSS_MODELS:
        payload = {
            "model": args.model,
            "messages": prompt,
            "temperature": args.temperature,
            "max_tokens": args.max_tokens,
        }
    else:
        payload = {
            "model": args.model,
            "temperature": args.temperature,
            "max_tokens": args.max_tokens,
            "messages": prompt,
        }
    headers = {"Content-Type": "application/json"}
    if args.api_key:
        headers["Authorization"] = f"Bearer {args.api_key}"
    elif os.getenv("OPENAI_API_KEY") is not None and args.model in GPT_MODELS:
        headers["Authorization"] = f"Bearer {os.getenv('OPENAI_API_KEY')}"
    log_payload_summary = _env_flag("CAPX_LOG_PLANNER_PAYLOAD_SUMMARY", False)
    payload_summary = _payload_structure_summary(payload)
    if log_payload_summary:
        print(
            "[PlannerPayloadSummary] "
            f"server_url={server_url} summary={json.dumps(payload_summary, ensure_ascii=False)}"
        )
    start_time = time.time()
    retry_config = _load_request_retry_config()
    response = None
    last_error: BaseException | None = None

    for attempt in range(1, retry_config.max_attempts + 1):
        try:
            response = requests.post(
                server_url,
                headers=headers,
                data=json.dumps(payload),
                timeout=(30, retry_config.request_timeout_s),
            )
            if _is_retryable_status_code(response.status_code):
                if log_payload_summary:
                    body_head = (response.text or "")[:300].replace("\n", "\\n")
                    print(
                        "[PlannerPayloadError] "
                        f"attempt={attempt} status={response.status_code} body_head={body_head!r}"
                    )
                last_error = requests.HTTPError(
                    f"Model query failed with status code {response.status_code}",
                    response=response,
                )
            else:
                response.raise_for_status()
                break
        except requests.RequestException as exc:
            if log_payload_summary:
                print(
                    "[PlannerPayloadError] "
                    f"attempt={attempt} exception={type(exc).__name__} detail={str(exc)[:300]!r}"
                )
            if not _is_retryable_exception(exc):
                raise
            last_error = exc

        elapsed = time.time() - start_time
        if attempt >= retry_config.max_attempts or elapsed >= retry_config.max_retry_walltime_s:
            if response is not None and _is_retryable_status_code(response.status_code):
                response.raise_for_status()
            if last_error is not None:
                raise last_error
            raise RuntimeError("Model query failed after exhausting retries without a recorded error.")

        sleep_time = _compute_retry_sleep_seconds(attempt, retry_config)
        sleep_time = min(
            sleep_time,
            max(0.0, retry_config.max_retry_walltime_s - elapsed),
        )
        if sleep_time <= 0:
            if last_error is not None:
                raise last_error
            raise RuntimeError("Model query exceeded retry wall-clock budget.")
        print(
            f"Retry {attempt}. Model query failed: {last_error}. "
            f"Retrying in {sleep_time:.1f} seconds..."
        )
        time.sleep(sleep_time)
    else:
        if last_error is not None:
            raise last_error
        raise RuntimeError("Model query failed unexpectedly without producing a response.")

    end_time = time.time()
    print(f"Time taken to query model: {end_time - start_time:.2f} seconds")
    if response is None:
        raise RuntimeError("Model query finished without a response object.")
    response.raise_for_status()
    body = response.json()
    out = {}
    if args.debug:
        print(json.dumps(body, indent=2))
    try:
        if args.model in GPT_MODELS and "codex" in args.model:
            out["content"] = body["output_text"]
        else:
            out["content"] = body["choices"][0]["message"]["content"]
    except (KeyError, IndexError) as exc:
        raise RuntimeError(f"Unexpected response format: {body}") from exc
    if body.get("choices") is not None:
        out["reasoning"] = body.get("choices")[0].get("message").get("reasoning", None)
    else:
        out["reasoning"] = None
    return out  # type: ignore[return-value]


def query_model_streaming(
    args: "LaunchArgs | ModelQueryArgs",
    prompt: list[dict],
) -> Iterable[dict]:
    """Query model with streaming enabled, yielding partial responses.

    Yields dictionaries with:
      - {"type": "content_delta", "content": "partial text"}
      - {"type": "reasoning_delta", "content": "partial reasoning"} (if supported)
      - {"type": "done", "content": "full content", "reasoning": "full reasoning or None"}

    Args:
        args: Configuration with server URL and model settings
        prompt: Full prompt containing environment observation

    Yields:
        Partial response chunks as they arrive
    """
    if args.model in GPT_MODELS:
        payload = {
            "model": args.model,
            "reasoning_effort": args.reasoning_effort,
            "max_completion_tokens": args.max_tokens,
            "messages": prompt,
            "stream": True,
        }
    elif args.model in CLAUDE_MODELS:
        payload = {
            "model": args.model,
            "temperature": args.temperature,
            "max_tokens": args.max_tokens,
            "thinking": {"type": "enabled", "budget_tokens": 4096},
            "messages": prompt,
            "stream": True,
        }
    else:
        payload = {
            "model": args.model,
            "temperature": args.temperature,
            "max_tokens": args.max_tokens,
            "messages": prompt,
            "stream": True,
        }

    headers = {"Content-Type": "application/json"}
    if args.api_key:
        headers["Authorization"] = f"Bearer {args.api_key}"
    elif os.getenv("OPENAI_API_KEY") is not None and args.model in GPT_MODELS:
        headers["Authorization"] = f"Bearer {os.getenv('OPENAI_API_KEY')}"

    full_content = ""
    full_reasoning = ""

    start_time = time.time()

    with requests.post(
        args.server_url,
        headers=headers,
        data=json.dumps(payload),
        timeout=200,
        stream=True,
    ) as response:
        response.raise_for_status()

        content_type = response.headers.get("content-type", "")
        is_sse = "text/event-stream" in content_type
        is_json = "application/json" in content_type

        # If it's a regular JSON response (server doesn't support streaming),
        # fall back to non-streaming behavior
        if is_json and not is_sse:
            print("Warning: Server returned JSON instead of SSE stream, falling back to non-streaming")
            body = response.json()
            try:
                full_content = body["choices"][0]["message"]["content"]
                full_reasoning = body.get("choices", [{}])[0].get("message", {}).get("reasoning")
                if full_reasoning:
                    print(f"Reasoning extracted ({len(full_reasoning)} chars)")
                else:
                    print("No reasoning returned by model")
            except (KeyError, IndexError) as exc:
                raise RuntimeError(f"Unexpected response format: {body}") from exc

            yield {"type": "content_delta", "content": full_content}
            yield {
                "type": "done",
                "content": full_content,
                "reasoning": full_reasoning if full_reasoning else None,
            }
            end_time = time.time()
            print(f"Time taken to query model (streaming fallback): {end_time - start_time:.2f} seconds")
            return

        for line in response.iter_lines():
            if not line:
                continue

            line_str = line.decode("utf-8")

            # SSE format: "data: {...}" or "data: [DONE]"
            if line_str.startswith("data: "):
                data_str = line_str[6:]  # Remove "data: " prefix

                if data_str == "[DONE]":
                    break

                try:
                    data = json.loads(data_str)
                    choices = data.get("choices", [])
                    if not choices:
                        continue

                    delta = choices[0].get("delta", {})

                    # Handle content delta
                    content_delta = delta.get("content", "")
                    if content_delta:
                        full_content += content_delta
                        yield {"type": "content_delta", "content": content_delta}

                    # Handle reasoning delta (some APIs support this)
                    reasoning_delta = delta.get("reasoning", "")
                    if reasoning_delta:
                        full_reasoning += reasoning_delta
                        yield {"type": "reasoning_delta", "content": reasoning_delta}

                except json.JSONDecodeError:
                    continue
            else:
                # Try parsing as raw JSON (non-SSE format)
                try:
                    data = json.loads(line_str)
                    choices = data.get("choices", [])
                    if choices:
                        delta = choices[0].get("delta", {})
                        content_delta = delta.get("content", "")
                        if content_delta:
                            full_content += content_delta
                            yield {"type": "content_delta", "content": content_delta}
                except json.JSONDecodeError:
                    continue

    end_time = time.time()
    print(f"Time taken to query model (streaming): {end_time - start_time:.2f} seconds")
    if full_reasoning:
        print(f"Reasoning extracted ({len(full_reasoning)} chars)")
    else:
        print("No reasoning returned by model")

    yield {
        "type": "done",
        "content": full_content,
        "reasoning": full_reasoning if full_reasoning else None,
    }


def query_model_ensemble(
    args: "LaunchArgs | ModelQueryArgs",
    prompt: list[dict],
    synthesis_model: str = "azure/openai/gpt-5.2",
    is_multiturn = False
) -> dict[str, Any]:
    """Query 9 models (3 models x 3 temperatures) and synthesize final output."""

    def query_single(model: str, temp: float) -> dict:
        query_args = ModelQueryArgs(
            model=model,
            server_url=args.server_url,
            api_key=args.api_key,
            temperature=temp,
            max_tokens=args.max_tokens,
            reasoning_effort=getattr(args, "reasoning_effort", "medium"),
        )
        try:
            result = query_model(query_args, copy.deepcopy(prompt))
            return {"model": model, "temp": temp, "content": result["content"], "ok": True}
        except Exception as e:
            error_msg = str(e)
            print(f"[Multimodel Ensemble] {model} temp={temp} FAILED: {error_msg}")
            return {"model": model, "temp": temp, "content": error_msg, "ok": False}

    # Build all (model, temp) pairs and query in parallel
    tasks = [(m, t) for m, temps in ENSEMBLE_CONFIGS for t in temps]
    responses = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=9) as executor:
        futures = {executor.submit(query_single, m, t): (m, t) for m, t in tasks}
        for future in concurrent.futures.as_completed(futures):
            resp = future.result()
            responses.append(resp)
            if resp['ok']:
                print(f"[Multimodel Ensemble] {resp['model']} temp={resp['temp']} ok={resp['ok']}")

    successful = [r for r in responses if r["ok"]]
    if not successful:
        # Print all errors for debugging
        print("\n=== All ensemble queries failed. Errors: ===")
        for r in responses:
            print(f"  {r['model']} temp={r['temp']}: {r['content']}")
        raise RuntimeError("All ensemble queries failed")

    # Build synthesis prompt
    original_text = ""
    for msg in prompt:
        if msg["role"] == "user":
            c = msg["content"]
            if isinstance(c, list):
                original_text += "".join(x.get("text", "") for x in c if isinstance(x, dict))
            elif isinstance(c, str):
                original_text += c

    candidates = "\n\n".join(
        f"--- Candidate ({r['model']}, temp={r['temp']}) ---\n{r['content']}"
        for r in successful
    )

    # Detect if this is a multiturn decision (candidates contain REGENERATE/FINISH)
    regenerate_count = sum(1 for r in successful if isinstance(r.get("content"), str) and "REGENERATE" in r["content"])
    finish_count = sum(1 for r in successful if isinstance(r.get("content"), str) and "FINISH" in r["content"])

    if is_multiturn:
        synthesis_system_prompt = f"""You are synthesizing {len(successful)} candidate responses for a multi-turn robot control task.

    DECISION ANALYSIS:
    - {regenerate_count} candidates voted REGENERATE
    - {finish_count} candidates voted FINISH

    SYNTHESIS RULES:
    1. Analyze critically and assume no candidate is fully correct
    2. Prefer explicit checks over assumptions
    3. Combine the best ideas from multiple candidates when appropriate
    4. If candidates disagree fundamentally, choose the more robust approach
    5. Combine best code ideas from REGENERATE candidates

    OUTPUT FORMAT (strict):
    - You may include brief reasoning first
    - Then output "REGENERATE" on its own line followed by exactly ONE fenced code block, OR output "FINISH" on its own line
    """
    else:
        synthesis_system_prompt = f"""You are synthesizing {len(successful)} candidate Python solutions into one optimal program.

    SYNTHESIS RULES:
    1. Analyze critically and assume no candidate is fully correct
    2. Prefer explicit checks over assumptions
    3. Combine the best ideas from multiple candidates when appropriate
    4. If candidates disagree fundamentally, choose the more robust approach

    OUTPUT FORMAT (strict):
    You may include reasoning before the fenced code block.
    Output ONLY ONE fenced code block (```python...```) containing the complete final solution.
    Do NOT include any other code blocks or code snippets outside this single block.
    """

    synthesis_user_prompt = f"""Synthesize the best solution.

    <original_task_description>
    {original_text}
    </original_task_description>

    <candidate_solutions>
    {candidates}
    </candidate_solutions>
    """

    synthesis_prompt = [
        {
            "role": "system",
            "content": synthesis_system_prompt,
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": synthesis_user_prompt,
                }
            ],
        },
    ]

    synth_args = ModelQueryArgs(
        model=synthesis_model,
        server_url=args.server_url,
        api_key=args.api_key,
        temperature=0.2,
        max_tokens=args.max_tokens,
    )
    final = query_model(synth_args, synthesis_prompt)

    # Build text content for saving
    candidates_txt = "\n\n".join(
        f"{'='*60}\nModel: {r['model']}\nTemperature: {r['temp']}\nSuccess: {r['ok']}\n{'='*60}\n{r['content']}"
        for r in responses
    )
    synthesis_txt = f"Model: {synthesis_model}\n\n"
    synthesis_txt += f"{'='*60}\nREASONING\n{'='*60}\n{final.get('reasoning') or '(none)'}\n\n"
    synthesis_txt += f"{'='*60}\nOUTPUT\n{'='*60}\n{final['content']}"

    return {
        "content": final["content"],
        "reasoning": final.get("reasoning"),
        "all_responses": responses,
        "ensemble_candidates_txt": candidates_txt,
        "ensemble_synthesis_txt": synthesis_txt,
    }


def query_single_model_ensemble(
    args: "LaunchArgs | ModelQueryArgs",
    prompt: list[dict],
    model: str,
    is_multiturn = False,
) -> dict[str, Any]:
    """Query the same model 9 times (with temperatures 0.1 to 0.9) and synthesize final output.

    Args:
        args: Configuration with server URL and model settings
        prompt: Full prompt containing environment observation and possibly multi-turn decision prompt
        model: The model to use for both candidate generation and synthesis

    Returns:
        Dictionary containing synthesized content, reasoning, all responses, and text artifacts
    """

    def query_single(temp: float) -> dict:
        query_args = ModelQueryArgs(
            model=model,
            server_url=args.server_url,
            api_key=args.api_key,
            temperature=temp,
            max_tokens=args.max_tokens,
            reasoning_effort=getattr(args, "reasoning_effort", "medium"),
        )
        try:
            result = query_model(query_args, copy.deepcopy(prompt))
            return {"model": model, "temp": temp, "content": result["content"], "ok": True}
        except Exception as e:
            error_msg = str(e)
            print(f"[Single Model Ensemble] {model} temp={temp} FAILED: {error_msg}")
            return {"model": model, "temp": temp, "content": error_msg, "ok": False}

    # Query same model with configurable temperatures.
    # Default keeps legacy behavior (9 candidates), while CAPX_SINGLE_MODEL_ENSEMBLE_TEMPS
    # lets us run lighter 2-3 candidate ensembles without code changes.
    temperatures = _env_float_list(
        "CAPX_SINGLE_MODEL_ENSEMBLE_TEMPS",
        [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    )
    responses = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, len(temperatures))) as executor:
        futures = {executor.submit(query_single, t): t for t in temperatures}
        for future in concurrent.futures.as_completed(futures):
            resp = future.result()
            responses.append(resp)
            if resp['ok']:
                print(f"[Single Model Ensemble] {resp['model']} temp={resp['temp']} ok={resp['ok']}")

    successful = [r for r in responses if r["ok"]]
    if not successful:
        # Print all errors for debugging
        print("\n=== All single model ensemble queries failed. Errors: ===")
        for r in responses:
            print(f"  {r['model']} temp={r['temp']}: {r['content']}")
        raise RuntimeError("All single model ensemble queries failed")

    # Build synthesis prompt
    original_text = ""
    for msg in prompt:
        if msg["role"] == "user":
            c = msg["content"]
            if isinstance(c, list):
                original_text += "".join(x.get("text", "") for x in c if isinstance(x, dict))
            elif isinstance(c, str):
                original_text += c

    candidates = "\n\n".join(
        f"--- Candidate (temp={r['temp']}) ---\n{r['content']}"
        for r in successful
    )

    # Detect if this is a multiturn decision (candidates contain REGENERATE/FINISH)
    regenerate_count = sum(1 for r in successful if isinstance(r.get("content"), str) and "REGENERATE" in r["content"])
    finish_count = sum(1 for r in successful if isinstance(r.get("content"), str) and "FINISH" in r["content"])

    if is_multiturn:
        synthesis_system_prompt = f"""You are synthesizing {len(successful)} candidate responses for a multi-turn robot control task.

    DECISION ANALYSIS:
    - {regenerate_count} candidates voted REGENERATE
    - {finish_count} candidates voted FINISH

    SYNTHESIS RULES:
    1. Analyze critically and assume no candidate is fully correct
    2. Prefer explicit checks over assumptions
    3. Combine the best ideas from multiple candidates when appropriate
    4. If candidates disagree fundamentally, choose the more robust approach
    5. Combine best code ideas from REGENERATE candidates

    OUTPUT FORMAT (strict):
    - You may include brief reasoning first
    - Then output "REGENERATE" on its own line followed by exactly ONE fenced code block, OR output "FINISH" on its own line
    """
    else: # first generation has no REGEN/FINISH candidates
        synthesis_system_prompt = f"""You are synthesizing {len(successful)} candidate Python solutions into one optimal program.

    SYNTHESIS RULES:
    1. Analyze critically and assume no candidate is fully correct
    2. Prefer explicit checks over assumptions
    3. Combine the best ideas from multiple candidates when appropriate
    4. If candidates disagree fundamentally, choose the more robust approach

    OUTPUT FORMAT (strict):
    You may include reasoning before the fenced code block.
    Output ONLY ONE fenced code block (```python...```) containing the complete final solution.
    Do NOT include any other code blocks or code snippets outside this single block.
    """

    synthesis_user_prompt = f"""Synthesize the best solution.

    <original_task_description>
    {original_text}
    </original_task_description>

    <candidate_solutions>
    {candidates}
    </candidate_solutions>
    """

    synthesis_prompt = [
        {
            "role": "system",
            "content": synthesis_system_prompt,
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": synthesis_user_prompt,
                }
            ],
        },
    ]

    # Use the same model for synthesis
    synth_args = ModelQueryArgs(
        model=model,
        server_url=args.server_url,
        api_key=args.api_key,
        temperature=0.2,
        max_tokens=args.max_tokens,
    )
    final = query_model(synth_args, synthesis_prompt)

    # Build text content for saving
    candidates_txt = "\n\n".join(
        f"{'='*60}\nModel: {r['model']}\nTemperature: {r['temp']}\nSuccess: {r['ok']}\n{'='*60}\n{r['content']}"
        for r in responses
    )
    synthesis_txt = f"Model: {model}\n\n"
    synthesis_txt += f"{'='*60}\nREASONING\n{'='*60}\n{final.get('reasoning') or '(none)'}\n\n"
    synthesis_txt += f"{'='*60}\nOUTPUT\n{'='*60}\n{final['content']}"

    return {
        "content": final["content"],
        "reasoning": final.get("reasoning"),
        "all_responses": responses,
        "ensemble_candidates_txt": candidates_txt,
        "ensemble_synthesis_txt": synthesis_txt,
    }


# ---------------------------------------------------------------------------
# Backward-compatible aliases (underscore-prefixed names)
# ---------------------------------------------------------------------------

_query_model = query_model
_query_model_streaming = query_model_streaming
_query_model_ensemble = query_model_ensemble
_query_single_model_ensemble = query_single_model_ensemble
