from __future__ import annotations

import os
import signal
import time
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

os.environ.setdefault("MUJOCO_GL", "osmesa")

from capx.envs.tasks import get_config, get_exec_env

initialized_envs = {}


def _extract_code(content: str) -> str:
    fence_start = "```python\n"
    fence_end = "```"
    start_idx = content.find(fence_start)
    end_idx = content.rfind(fence_end)
    if start_idx == -1 or end_idx == -1:
        return content.strip()
    return content[start_idx + len(fence_start) : end_idx].strip()


@contextmanager
def _time_limit(seconds: float) -> Iterator[None]:
    if seconds <= 0:
        yield
        return

    def _raise_timeout(_signum: int, _frame: Any) -> None:  # type: ignore[override]
        raise TimeoutError(f"Environment step exceeded {seconds:.1f}s")

    previous_handler = signal.getsignal(signal.SIGALRM)
    try:
        signal.signal(signal.SIGALRM, _raise_timeout)
        signal.setitimer(signal.ITIMER_REAL, seconds)
        yield
    finally:
        try:
            signal.setitimer(signal.ITIMER_REAL, 0.0)
        finally:
            signal.signal(signal.SIGALRM, previous_handler)


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: dict[str, Any] | str,
    extra_info: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if data_source not in initialized_envs:
        cfg = get_config(data_source)
        if cfg.privileged:
            cfg.enable_render = False
        initialized_envs[data_source] = get_exec_env(data_source)(cfg)
    env = initialized_envs[data_source]
    seed = int((extra_info or {}).get("seed", 0))
    code = _extract_code(solution_str)

    try:
        env.reset(seed=seed)
        start_time = time.time()
        with _time_limit(90.0):
            _, raw_reward, terminated, truncated, info = env.step(code)
        success = bool(terminated)
        sandbox_ok = int(info.get("sandbox_rc", 1)) == 0
        if success:
            score = 1.0
        elif sandbox_ok:
            score = 0.05 * max(float(raw_reward), 0.0)
        else:
            score = 0.0
        return {
            "score": float(score),
            "success": success,
            "terminated": success,
            "truncated": bool(truncated),
            "raw_reward": float(raw_reward),
            "sandbox_ok": sandbox_ok,
            "error": str(info.get("stderr", "")) if not sandbox_ok else "",
            "eval_seconds": time.time() - start_time,
        }
    except TimeoutError as exc:
        return {
            "score": 0.0,
            "success": False,
            "terminated": False,
            "truncated": True,
            "raw_reward": 0.0,
            "sandbox_ok": False,
            "error": repr(exc),
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "score": 0.0,
            "success": False,
            "terminated": False,
            "truncated": False,
            "raw_reward": 0.0,
            "sandbox_ok": False,
            "error": repr(exc),
        }
