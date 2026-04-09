"""Single-trial execution for CaP-X environments.

This module handles single trial execution including code generation,
multi-turn decisions, and visual feedback. It contains the core trial
loop extracted from launch.py, covering:

- Initial code generation and oracle code handling
- Code block execution with multi-turn regeneration
- Visual feedback capture and image/video differencing
- Trial artifact saving (code, logs, per-turn videos, combined video)
"""

from __future__ import annotations

import base64
import copy
import gc
import io
import json
import os
import re
import signal
import time
import traceback
from typing import Any

import numpy as np
from PIL import Image

from capx.envs.configs.instantiate import instantiate
from capx.envs.phase_candidates import infer_phase_candidates, phase_tags_from_candidates
from capx.envs.tasks.base import CodeExecutionEnvBase
from capx.envs.trajectory_buffer import (
    append_event,
    append_snapshot,
    create_trajectory_buffer,
)

from capx.llm.client import (
    VLM_MODELS,
    ModelQueryArgs,
    query_model as _query_model,
    query_model_ensemble as _query_model_ensemble,
    query_single_model_ensemble as _query_single_model_ensemble,
)
from capx.utils.launch_utils import (
    TrialSummary,
    _build_multi_turn_decision_prompt,
    _build_multi_turn_decision_prompt_legacy,
    _count_nonempty_code_blocks,
    _extract_code,
    _get_visual_feedback,
    _parse_multi_turn_decision,
    _save_trial_artifacts,
)
from capx.utils.video_utils import _encode_video_base64, _write_video

# Use TYPE_CHECKING to avoid circular imports for type hints only
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from capx.envs.launch import LaunchArgs


MULTITURN_LIMIT = 10
PRE_CODEGEN_TIMEOUT_SECONDS = 300

# ---------------------------------------------------------------------------
# Shared formatting helpers
# ---------------------------------------------------------------------------

def _annotate_code_blocks(
    code_blocks: list[str],
    code_block_metadata: list[dict[str, Any]],
) -> str:
    """Join code blocks into a single string with ``# Code block N`` headers."""
    annotated = []
    for i, (block, metadata) in enumerate(zip(code_blocks, code_block_metadata, strict=False)):
        annotated.append(f"# Code block {i}\n{block}")
    return "\n\n".join(annotated)


def _build_log_lines(
    final_code: str,
    info_step: dict[str, Any],
    reward: float,
    terminated: bool,
    truncated: bool,
    num_regenerations: int,
    num_finishes: int,
    num_code_blocks: int,
    *,
    prefix: str = "",
    stderr_override: str | None = None,
) -> list[str]:
    """Build the standard log-line list used for both normal and timeout summaries."""
    stderr = stderr_override if stderr_override is not None else info_step.get("stderr", "")
    lines = ["-" * 100]
    if prefix:
        lines.append(prefix)
    lines.extend([
        "Generated program:",
        final_code if final_code else "(no program available)",
        "\n\nEnvironment response:",
        f"  Sandbox failed: {info_step.get('sandbox_rc', 1)}",
        f"  Stdout: {info_step.get('stdout', '')}",
        f"  Stderr: {stderr}",
        f"  Reward: {reward}",
        f"  Task Completed: {info_step.get('task_completed', False)}",
        f"  Terminated: {terminated}, Truncated: {truncated}",
        f"  Num Regenerations: {num_regenerations}",
        f"  Num Finishes: {num_finishes}",
        f"  Num Code Blocks: {num_code_blocks}",
        "-" * 100,
    ])
    return lines


def _run_with_phase_timeout(
    trial: int,
    phase_name: str,
    timeout_seconds: int,
    fn,
):
    """Run a callable under a shorter SIGALRM budget while preserving the outer trial alarm."""
    if timeout_seconds <= 0:
        return fn()

    previous_handler = signal.getsignal(signal.SIGALRM)
    outer_remaining = signal.alarm(0)
    if outer_remaining <= 0:
        signal.signal(signal.SIGALRM, previous_handler)
        return fn()

    phase_budget = max(1, min(int(timeout_seconds), int(outer_remaining)))

    def _phase_timeout_handler(signum: int, frame) -> None:  # type: ignore[override]
        raise TimeoutError(f"Trial {trial} exceeded {phase_name} timeout of {phase_budget} seconds")

    phase_start = time.monotonic()
    signal.signal(signal.SIGALRM, _phase_timeout_handler)
    signal.alarm(phase_budget)
    try:
        return fn()
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous_handler)
        elapsed = max(0, int(time.monotonic() - phase_start))
        remaining_total = max(1, int(outer_remaining) - elapsed)
        signal.alarm(remaining_total)


# ---------------------------------------------------------------------------
# Trial video directory helper
# ---------------------------------------------------------------------------

def _trial_video_dir(
    config: dict[str, Any],
    trial: int,
    attempt_idx: int,
) -> str:
    """Return the trial output directory path used for video saving."""
    return os.path.join(config["output_dir"], f"trial_{trial:02d}", f"attempt_{attempt_idx:02d}")


def _save_trial_video(
    env: CodeExecutionEnvBase,
    config: dict[str, Any],
    trial: int,
    attempt_idx: int,
    info_step: dict[str, Any],
    reward: float,
    num_code_blocks: int,
    *,
    suffix_extra: str = "",
) -> None:
    """Save recorded video frames from the environment, if available."""
    if not config["record_video"] or not hasattr(env, "get_video_frames"):
        return
    frames = env.get_video_frames(clear=True)
    if not frames or not config["output_dir"]:
        return

    base_dir = _trial_video_dir(config, trial, attempt_idx)
    suffix = f"{reward:.3f}"
    if suffix_extra:
        suffix += f"_{suffix_extra}"

    if isinstance(frames, list):
        _write_video(frames, base_dir, suffix=suffix)
    elif isinstance(frames, dict):
        for key, frame in frames.items():
            _write_video(frame, base_dir, suffix=f"{suffix}_{key}")


def _save_turn_and_combined_videos(
    env: CodeExecutionEnvBase,
    config: dict[str, Any],
    trial: int,
    attempt_idx: int,
    info_step: dict[str, Any],
    reward: float,
    turn_frame_ranges: list[tuple[int, int]],
) -> None:
    """Save per-turn videos and a combined video of all turns.

    Gets all frames from the environment (clearing the buffer), then writes:
      - ``video_turn_00.mp4``, ``video_turn_01.mp4``, ... for each turn
      - ``video_combined.mp4`` for the full trial
      - If wrist camera is enabled: ``video_turn_00_wrist.mp4``, etc.
    """
    if not config["record_video"] or not config["output_dir"]:
        return
    if not hasattr(env, "get_video_frames"):
        return

    all_frames = env.get_video_frames(clear=True)
    if not all_frames:
        return

    base_dir = _trial_video_dir(config, trial, attempt_idx)

    # all_frames may be a list (Robosuite) or a dict of lists (R1Pro multi-camera).
    # Normalise to a list for slicing; dict case is handled by _write_multi_video.
    if isinstance(all_frames, dict):
        # Multi-camera: write each camera stream as a combined video
        for key, frames in all_frames.items():
            if frames:
                _write_video(frames, base_dir, suffix=f"combined_{key}")
        return

    # Per-turn videos
    for i, (start, end) in enumerate(turn_frame_ranges):
        turn_frames = all_frames[start:end]
        if turn_frames:
            _write_video(turn_frames, base_dir, suffix=f"turn_{i:02d}")

    # Combined video
    _write_video(all_frames, base_dir, suffix="combined")

    # Wrist camera videos
    if config.get("use_wrist_camera") and hasattr(env, "get_wrist_video_frames"):
        wrist_frames = env.get_wrist_video_frames(clear=True)
        if wrist_frames:
            for i, (start, end) in enumerate(turn_frame_ranges):
                wrist_turn = wrist_frames[start:end]
                if wrist_turn:
                    _write_video(wrist_turn, base_dir, suffix=f"turn_{i:02d}_wrist")
            _write_video(wrist_frames, base_dir, suffix="combined_wrist")


# ---------------------------------------------------------------------------
# Visual feedback and image differencing
# ---------------------------------------------------------------------------

def _capture_initial_visual_feedback(
    env: CodeExecutionEnvBase,
    obs: dict[str, Any],
    config: dict[str, Any],
    args: LaunchArgs,
    visual_differencing_args: ModelQueryArgs,
) -> tuple[list, list[str], str, dict[str, Any] | None]:
    """Capture the initial environment image and optionally describe it.

    Returns:
        (visual_feedback_imgs, visual_feedback_base64_history, task_description, initial_scene_artifact)
    """
    visual_feedback_imgs: list = []
    visual_feedback_base64_history: list[str] = []
    task_description = ""
    initial_scene_artifact: dict[str, Any] | None = None

    use_wrist = config.get("use_wrist_camera", False)

    needs_visual = (
        (config["use_visual_feedback"] and args.model in VLM_MODELS)
        or (config["use_img_differencing"] and visual_differencing_args.model in VLM_MODELS)
        or config.get("use_video_differencing", False)
    )
    if not (needs_visual and hasattr(env, "render")):
        return visual_feedback_imgs, visual_feedback_base64_history, task_description, initial_scene_artifact

    initial_base64, initial_img = _get_visual_feedback(env)
    visual_feedback_imgs.append(initial_img)
    visual_feedback_base64_history.append(initial_base64)
    task_description = copy.deepcopy(obs["full_prompt"][-1]["content"][0]["text"])

    # Also capture wrist camera image for multiview initial description
    initial_wrist_base64 = None
    if use_wrist and hasattr(env, "render_wrist"):
        wrist_img = env.render_wrist()
        if wrist_img is not None:
            pil_wrist = Image.fromarray(wrist_img)
            buf = io.BytesIO()
            pil_wrist.save(buf, format="png")
            initial_wrist_base64 = (
                f"data:image/png;base64,"
                f"{base64.b64encode(buf.getvalue()).decode('utf-8')}"
            )
            visual_feedback_imgs.append(pil_wrist)

    # Append image to the prompt for VLM visual feedback
    if config["use_visual_feedback"]:
        obs["full_prompt"][-1]["content"][0]["text"] += (
            "\n\nIncluded below is an image of the initial state of the environment."
        )
        obs["full_prompt"][-1]["content"].append(
            {"type": "image_url", "image_url": {"url": initial_base64}}
        )
        if initial_wrist_base64 is not None:
            obs["full_prompt"][-1]["content"].append(
                {
                    "type": "text",
                    "text": "Included below is an image from the robot's wrist camera.",
                }
            )
            obs["full_prompt"][-1]["content"].append(
                {"type": "image_url", "image_url": {"url": initial_wrist_base64}}
            )

    # Image differencing: ask a VLM to describe the initial scene
    if config["use_img_differencing"] or config.get("use_video_differencing", False):
        initial_scene_artifact = _describe_initial_scene(
            visual_differencing_args, task_description, initial_base64,
            wrist_image_base64=initial_wrist_base64,
        )
        description = initial_scene_artifact["content"]
        feedback = f"The initial state of the environment is described as follows:\n{description}"
        obs["full_prompt"][-1]["content"][0]["text"] += f"\n\n{feedback}"
        if args.debug:
            print(description)

    return (
        visual_feedback_imgs,
        visual_feedback_base64_history,
        task_description,
        initial_scene_artifact,
    )


def _describe_initial_scene(
    visual_differencing_args: ModelQueryArgs,
    task_description: str,
    image_base64: str,
    wrist_image_base64: str | None = None,
) -> dict[str, Any]:
    """Query a VLM to describe the initial environment state."""
    user_content: list[dict[str, Any]] = [
        {"type": "text", "text": task_description},
        {
            "type": "text",
            "text": (
                "Describe the initial state of the environment with the goal of the "
                "task in mind. You should try to provide objective information and no "
                "assumptions. Do *NOT* write any code."
            ),
        },
        {"type": "text", "text": "Main camera view:"},
        {"type": "image_url", "image_url": {"url": image_base64}},
    ]
    if wrist_image_base64 is not None:
        user_content.extend([
            {"type": "text", "text": "Wrist camera view:"},
            {"type": "image_url", "image_url": {"url": wrist_image_base64}},
        ])

    prompt = [
        {
            "role": "system",
            "content": (
                "You are a helpful assistant that describes the initial state of the "
                "environment with the goal of the task in mind. You should try to provide "
                "objective information and no assumptions. Do *NOT* write any code."
            ),
        },
        {"role": "user", "content": user_content},
    ]
    response = _query_model(visual_differencing_args, prompt)
    return {
        "model": visual_differencing_args.model,
        "prompt": prompt,
        "content": response["content"],
        "reasoning": response.get("reasoning"),
        "task_description": task_description,
    }


def _get_visual_differencing_feedback(
    visual_differencing_args: ModelQueryArgs,
    task_description: str,
    visual_feedback_base64_history: list[str],
    wrist_base64_history: list[str] | None = None,
) -> str | None:
    """Query a VLM to describe what changed between the two most recent frames.

    Args:
        wrist_base64_history: Optional history of wrist camera images.  When provided
            and has >=2 entries, the before/after wrist images are included in the prompt.
    """
    if len(visual_feedback_base64_history) < 2:
        return None

    user_content: list[dict[str, Any]] = [
        {"type": "text", "text": task_description},
        {
            "type": "text",
            "text": (
                "Describe the difference between the current state of the "
                "environment and the previous state of the environment with the "
                "goal of the task in mind and whether the task has been completed. "
                "You should try to provide objective information and no assumptions. "
                "Do *NOT* write any code.."
            ),
        },
        {"type": "text", "text": "Previous state (main camera):"},
        {"type": "image_url", "image_url": {"url": visual_feedback_base64_history[-2]}},
        {"type": "text", "text": "Current state (main camera):"},
        {"type": "image_url", "image_url": {"url": visual_feedback_base64_history[-1]}},
    ]

    if wrist_base64_history and len(wrist_base64_history) >= 2:
        user_content.extend([
            {"type": "text", "text": "Previous state (wrist camera):"},
            {"type": "image_url", "image_url": {"url": wrist_base64_history[-2]}},
            {"type": "text", "text": "Current state (wrist camera):"},
            {"type": "image_url", "image_url": {"url": wrist_base64_history[-1]}},
        ])

    prompt = [
        {
            "role": "system",
            "content": (
                "You are a helpful assistant that describes the difference between the "
                "current state of the environment and the previous state of the environment "
                "with the goal of the task in mind and whether the task has been completed. "
                "You should try to provide objective information and no assumptions. "
                "Do *NOT* write any code."
            ),
        },
        {"role": "user", "content": user_content},
    ]
    return _query_model(visual_differencing_args, prompt)["content"]


# ---------------------------------------------------------------------------
# Video differencing
# ---------------------------------------------------------------------------

def _get_video_differencing_feedback(
    visual_differencing_args: ModelQueryArgs,
    task_description: str,
    turn_frames: list[np.ndarray],
    wrist_turn_frames: list[np.ndarray] | None = None,
) -> str | None:
    """Query a VLM with a video of the turn execution to describe what happened.

    Args:
        visual_differencing_args: Model query args for the VDM model.
        task_description: The task goal.
        turn_frames: RGB frames from the main camera for this turn.
        wrist_turn_frames: RGB frames from the wrist camera for this turn (optional).

    Returns:
        Text description of the execution, or None if no frames.
    """
    if not turn_frames:
        return None

    video_base64 = _encode_video_base64(turn_frames)

    user_content: list[dict[str, Any]] = [
        {"type": "text", "text": task_description},
        {
            "type": "text",
            "text": (
                "The following video shows the robot executing code in the "
                "environment from the main camera view. Describe what happened "
                "during execution, including what actions the robot took, how "
                "the objects in the scene changed, and whether the task appears "
                "to have been completed. Provide objective information and no "
                "assumptions. Do *NOT* write any code."
            ),
        },
        {"type": "text", "text": "Main camera video:"},
        {"type": "image_url", "image_url": {"url": video_base64}},
    ]

    if wrist_turn_frames:
        wrist_video_base64 = _encode_video_base64(wrist_turn_frames)
        user_content.extend([
            {
                "type": "text",
                "text": (
                    "The following video shows the same execution from the "
                    "robot's wrist-mounted camera (eye-in-hand view), providing "
                    "a close-up perspective of the gripper and objects being "
                    "manipulated."
                ),
            },
            {"type": "text", "text": "Wrist camera video:"},
            {"type": "image_url", "image_url": {"url": wrist_video_base64}},
        ])

    prompt = [
        {
            "role": "system",
            "content": (
                "You are a helpful assistant that analyzes robot execution "
                "videos. You describe what happened during the robot's code "
                "execution, what actions were taken, how the environment "
                "changed, and whether the task appears to have been completed. "
                "Provide objective information and no assumptions. "
                "Do *NOT* write any code."
            ),
        },
        {"role": "user", "content": user_content},
    ]
    return _query_model(visual_differencing_args, prompt)["content"]


# ---------------------------------------------------------------------------
# Initial code generation
# ---------------------------------------------------------------------------

def _query_initial_code(
    args: LaunchArgs,
    config: dict[str, Any],
    obs: dict[str, Any],
) -> tuple[str, str | None, dict | None, dict[str, Any] | None]:
    """Query the model for the initial code generation.

    Returns:
        (raw_code, reasoning, ensemble_data, raw_model_response)
    """
    # Save the initial prompt
    with open(os.path.join(config["output_dir"], "initial_prompt.txt"), "w") as f:
        f.write(str(obs["full_prompt"]))

    ensemble_data = None
    raw_model_response = None
    if config["use_parallel_ensemble"]:
        if config.get("use_multimodel", False):
            print("RUNNING MULTIMODEL ENSEMBLE QUERY")
            out = _query_model_ensemble(args, obs["full_prompt"], is_multiturn=False)
        else:
            print("RUNNING SINGLE MODEL ENSEMBLE QUERY")
            out = _query_single_model_ensemble(args, obs["full_prompt"], args.model, is_multiturn=False)
        ensemble_data = {
            "ensemble_candidates_txt": out["ensemble_candidates_txt"],
            "ensemble_synthesis_txt": out["ensemble_synthesis_txt"],
        }
    else:
        out = _query_model(args, obs["full_prompt"])
        raw_model_response = {
            "model": args.model,
            "finish_reason": out.get("finish_reason"),
            "raw_response": out.get("raw_response"),
        }

    return out["content"], out["reasoning"], ensemble_data, raw_model_response


def _build_recovery_prompt(
    obs: dict[str, Any],
    *,
    executed_code: str,
    console_stdout: str,
    console_stderr: str,
    recovery_reason: str,
) -> list[dict[str, Any]]:
    prompt = copy.deepcopy(obs["full_prompt"])
    prompt[-1]["content"].append(
        {
            "type": "text",
            "text": (
                "The previous attempt did not successfully complete the task. "
                "The environment has already been restored to a safe earlier "
                "snapshot before the failed action. Generate exactly one new "
                "Python code block that recovers from the current state and "
                "continues the task.\n\n"
                f"Recovery reason: {recovery_reason}\n\n"
                "Previously executed code:\n"
                f"```python\n{executed_code}\n```\n"
                f"Console stdout:\n```\n{console_stdout}\n```\n"
                f"Console stderr:\n```\n{console_stderr}\n```\n"
                "Respond with only executable Python code in a fenced "
                "```python``` block. Do not write REGENERATE or FINISH."
            ),
        }
    )
    return prompt


def _query_recovery_code(
    args: LaunchArgs,
    obs: dict[str, Any],
    *,
    executed_code: str,
    console_stdout: str,
    console_stderr: str,
    recovery_reason: str,
) -> tuple[str, str | None, list[dict[str, Any]]]:
    recovery_prompt = _build_recovery_prompt(
        obs,
        executed_code=executed_code,
        console_stdout=console_stdout,
        console_stderr=console_stderr,
        recovery_reason=recovery_reason,
    )
    out = _query_model(args, recovery_prompt)
    return out["content"], out["reasoning"], recovery_prompt


def _extract_latest_prompt_text(obs: dict[str, Any]) -> str:
    for message in reversed(obs.get("full_prompt", [])):
        content = message.get("content")
        if isinstance(content, str) and content.strip():
            return content.strip()
        if isinstance(content, list):
            texts = [
                item.get("text", "").strip()
                for item in content
                if isinstance(item, dict) and item.get("type") == "text" and item.get("text")
            ]
            joined = "\n".join(text for text in texts if text)
            if joined:
                return joined
    return ""


def _build_snapshot_selection_prompt(
    obs: dict[str, Any],
    *,
    purpose: str,
    candidates: list[dict[str, Any]],
    preferred_phases: list[str],
    executed_code: str,
    console_stdout: str,
    console_stderr: str,
) -> list[dict[str, Any]]:
    candidate_lines = []
    for candidate in candidates:
        phase_tags = ", ".join(candidate.get("phase_tags", [])) or "none"
        summary = candidate.get("summary", {})
        candidate_lines.append(
            (
                f"- {candidate['snapshot_id']}: type={candidate['snapshot_type']}, "
                f"code_block_idx={candidate['code_block_idx']}, "
                f"open_gripper={candidate['is_open_gripper']}, "
                f"phase_tags={phase_tags}, "
                f"reward={summary.get('reward')}, done={summary.get('done')}"
            )
        )

    task_text = _extract_latest_prompt_text(obs)
    prompt_text = (
        "Select a single simulator snapshot to recover from.\n\n"
        f"Task context:\n{task_text}\n\n"
        f"Selection purpose: {purpose}\n"
        "You must choose exactly one snapshot_id from the provided candidates. "
        "Prefer uncontaminated earlier states that are safe to resume from. "
        "Open-gripper snapshots and phases aligned with the preferred phase list "
        "are usually safer.\n\n"
        f"Preferred phases: {', '.join(preferred_phases)}\n\n"
        "Executed code up to the failure point:\n"
        f"```python\n{executed_code}\n```\n"
        f"Console stdout:\n```\n{console_stdout}\n```\n"
        f"Console stderr:\n```\n{console_stderr}\n```\n"
        "Candidate snapshots:\n"
        f"{os.linesep.join(candidate_lines)}\n\n"
        'Respond with JSON only, for example: {"snapshot_id": "snap_0001", "reason": "..."}. '
        "Do not invent snapshot ids outside the candidate list."
    )
    return [
        {
            "role": "system",
            "content": "You are a recovery-point selector for a robot simulator.",
        },
        {
            "role": "user",
            "content": [{"type": "text", "text": prompt_text}],
        },
    ]


def _parse_snapshot_selection_response(
    content: str,
    *,
    valid_snapshot_ids: set[str],
) -> tuple[str | None, str | None]:
    stripped = content.strip()
    parsed_reason = None

    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError:
        parsed = None

    if isinstance(parsed, dict):
        snapshot_id = parsed.get("snapshot_id")
        if isinstance(snapshot_id, str) and snapshot_id in valid_snapshot_ids:
            reason = parsed.get("reason")
            parsed_reason = reason if isinstance(reason, str) else None
            return snapshot_id, parsed_reason

    if stripped in valid_snapshot_ids:
        return stripped, parsed_reason

    match = re.search(r"\bsnap_\d+\b", stripped)
    if match is not None:
        snapshot_id = match.group(0)
        if snapshot_id in valid_snapshot_ids:
            return snapshot_id, parsed_reason

    return None, parsed_reason


def _query_snapshot_selection(
    args: LaunchArgs,
    obs: dict[str, Any],
    *,
    purpose: str,
    candidates: list[dict[str, Any]],
    preferred_phases: list[str],
    executed_code: str,
    console_stdout: str,
    console_stderr: str,
) -> tuple[str | None, str | None, str | None, list[dict[str, Any]], str | None]:
    selection_prompt = _build_snapshot_selection_prompt(
        obs,
        purpose=purpose,
        candidates=candidates,
        preferred_phases=preferred_phases,
        executed_code=executed_code,
        console_stdout=console_stdout,
        console_stderr=console_stderr,
    )
    out = _query_model(args, selection_prompt)
    selected_snapshot_id, parsed_reason = _parse_snapshot_selection_response(
        out["content"],
        valid_snapshot_ids={candidate["snapshot_id"] for candidate in candidates},
    )
    return selected_snapshot_id, out["content"], out["reasoning"], selection_prompt, parsed_reason


# ---------------------------------------------------------------------------
# Multi-turn decision handling
# ---------------------------------------------------------------------------

def _handle_multi_turn_step(
    env: CodeExecutionEnvBase,
    obs: dict[str, Any],
    args: LaunchArgs,
    config: dict[str, Any],
    visual_differencing_args: ModelQueryArgs,
    multi_turn_prompt: str,
    code_blocks: list[str],
    code_block_idx: int,
    info_step: dict[str, Any],
    task_description: str,
    visual_feedback_imgs: list,
    visual_feedback_base64_history: list[str],
    stderr_history: list[str],
    turn_frames: list[np.ndarray] | None = None,
    wrist_turn_frames: list[np.ndarray] | None = None,
    wrist_base64_history: list[str] | None = None,
) -> tuple[str, str | None, str | None, dict | None, list | None]:
    """Execute one multi-turn decision step.

    Captures visual feedback, builds the decision prompt, queries the model,
    and returns the parsed decision.

    Args:
        turn_frames: Frames from the main camera for this turn (for video differencing).
        wrist_turn_frames: Frames from the wrist camera for this turn (for video differencing).
        wrist_base64_history: History of wrist camera base64 images for image-based
            differencing with multiview.

    Returns:
        (decision, new_code, reasoning, multiturn_ensemble_entry)
        where decision is "regenerate", "finish", or "continue".
    """
    use_wrist = config.get("use_wrist_camera", False)

    executed_code = "\n".join(code_blocks[:code_block_idx])
    complete_multi_turn_prompt = multi_turn_prompt.format(
        executed_code=executed_code,
        console_stdout=info_step["stdout"],
        console_stderr=info_step["stderr"],
    )

    if info_step["stderr"] != "":
        stderr_history.append(info_step["stderr"])

    # Capture visual feedback if applicable
    visual_feedback_base64 = None
    needs_visual = (
        (config["use_visual_feedback"] and args.model in VLM_MODELS)
        or (config["use_img_differencing"] and visual_differencing_args.model in VLM_MODELS)
    )
    if needs_visual and hasattr(env, "render"):
        vf_base64, vf_img = _get_visual_feedback(env)
        visual_feedback_imgs.append(vf_img)
        visual_feedback_base64_history.append(vf_base64)

        # Also capture wrist camera snapshot for image-based multiview
        if use_wrist and hasattr(env, "render_wrist") and wrist_base64_history is not None:
            wrist_result = _get_visual_feedback(env, use_wrist_camera=True)
            if wrist_result[0] is not None and isinstance(wrist_result[0], list) and len(wrist_result[0]) > 1:
                wrist_base64_history.append(wrist_result[0][1])  # index 1 = wrist image

    # Determine differencing feedback
    differencing_feedback = None
    is_video_feedback = False

    if config.get("use_video_differencing") and turn_frames:
        # Video-based differencing: pass video of this turn to VDM
        differencing_feedback = _get_video_differencing_feedback(
            visual_differencing_args, task_description, turn_frames, wrist_turn_frames,
        )
        is_video_feedback = True
    elif config["use_img_differencing"] and len(visual_feedback_base64_history) >= 2:
        # Image-based differencing: pass before/after images to VDM
        differencing_feedback = _get_visual_differencing_feedback(
            visual_differencing_args, task_description, visual_feedback_base64_history,
            wrist_base64_history=wrist_base64_history,
        )

    # Only pass visual feedback to prompt if visual_feedback is enabled
    if not config["use_visual_feedback"]:
        visual_feedback_base64 = None
    elif needs_visual and hasattr(env, "render"):
        visual_feedback_base64 = visual_feedback_base64_history[-1] if visual_feedback_base64_history else None

    # Build decision prompt
    if args.use_legacy_multi_turn_decision_prompt:
        print("Using legacy multi-turn decision prompt")
        decision_prompt = _build_multi_turn_decision_prompt_legacy(
            obs, complete_multi_turn_prompt, visual_feedback_base64, differencing_feedback,
            is_video_feedback=is_video_feedback,
        )
    else:
        decision_prompt = _build_multi_turn_decision_prompt(
            obs, complete_multi_turn_prompt, visual_feedback_base64, differencing_feedback,
            is_video_feedback=is_video_feedback,
        )

    # Query model
    multiturn_ensemble_entry = None
    if config["use_parallel_ensemble"]:
        if config.get("use_multimodel", False):
            print("RUNNING MULTITURN MULTIMODEL ENSEMBLE QUERY")
            content = _query_model_ensemble(args, decision_prompt, is_multiturn=True)
        else:
            print("RUNNING MULTITURN SINGLE MODEL ENSEMBLE QUERY")
            content = _query_single_model_ensemble(args, decision_prompt, args.model, is_multiturn=True)
        multiturn_ensemble_entry = {
            "ensemble_candidates_txt": content.get("ensemble_candidates_txt", ""),
            "ensemble_synthesis_txt": content.get("ensemble_synthesis_txt", ""),
        }
    else:
        content = _query_model(args, decision_prompt)

    reasoning = content["reasoning"]
    decision, new_code = _parse_multi_turn_decision(content["content"])

    return decision, new_code, reasoning, multiturn_ensemble_entry, decision_prompt


# ---------------------------------------------------------------------------
# Core single-trial execution
# ---------------------------------------------------------------------------

def _run_single_trial(
    env: CodeExecutionEnvBase,
    trial: int,
    attempt_idx: int,
    args: LaunchArgs,
    config: dict[str, Any],
    multi_turn_prompt: str | None,
    partial_artifacts: dict[str, Any] | None = None,
) -> TrialSummary:
    """Execute a single trial end-to-end.

    Steps:
        1. Reset the environment.
        2. Capture initial visual feedback (if configured).
        3. Query the model for initial code generation.
        4. Execute code blocks one-by-one, with optional multi-turn regeneration.
        5. Save artifacts (code, logs, per-turn videos, combined video) and return a TrialSummary.
    """
    trial_start_time = time.time()

    use_video_diff = config.get("use_video_differencing", False)
    use_wrist = config.get("use_wrist_camera", False)
    enable_eap_rollback = config.get("enable_eap_rollback", False)
    enable_eap_recovery = config.get("enable_eap_recovery", False)
    enable_eap_model_snapshot_selection = config.get("enable_eap_model_snapshot_selection", False)

    # --- 1. Reset environment ---
    obs, _ = env.reset(options={"trial": trial}, seed=trial)
    # Reset the SIGALRM timer AFTER env.reset() so the timeout only covers
    # actual task execution, not scene loading / cuRobo JIT compilation.
    import signal
    remaining = signal.alarm(0)  # cancel current alarm
    if remaining > 0:
        signal.alarm(1000)  # restart fresh 1000s from now
    obs["full_prompt"] = copy.deepcopy(obs["full_prompt"])
    _patch_libero_goal(env, obs)

    if config["record_video"] and hasattr(env, "enable_video_capture"):
        env.enable_video_capture(True, clear=True, wrist_camera=use_wrist)
    elif use_video_diff and hasattr(env, "enable_video_capture"):
        # Video differencing needs frame recording even without record_video
        env.enable_video_capture(True, clear=True, wrist_camera=use_wrist)

    # --- Shared trial state ---
    code_blocks: list[str] = []
    code_block_metadata: list[dict[str, Any]] = []
    all_responses: list[dict[str, Any]] = []
    stderr_history: list[str] = []
    num_regenerations = 0
    num_finishes = 0
    num_recoveries = 0
    info_step: dict[str, Any] = {"sandbox_rc": -1, "stdout": "", "stderr": ""}
    reward = 0.0
    terminated = truncated = False
    sandbox_rc_override = None
    ensemble_data = None
    multiturn_ensemble_data: list[dict[str, Any]] = []

    # Per-turn frame tracking (for video differencing and per-turn video saving)
    turn_frame_ranges: list[tuple[int, int]] = []

    # Wrist camera base64 history for image-based multiview differencing
    wrist_base64_history: list[str] | None = [] if use_wrist else None

    visual_differencing_args = ModelQueryArgs(
        model=args.visual_differencing_model,
        server_url=args.visual_differencing_model_server_url,
        api_key=args.visual_differencing_model_api_key,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        reasoning_effort=args.reasoning_effort,
        debug=args.debug,
    )

    if config["use_img_differencing"] or use_video_diff:
        assert visual_differencing_args.model in VLM_MODELS, (
            "Image/video differencing model must be in the list of VLM models"
        )

    if partial_artifacts is not None:
        partial_artifacts["pre_codegen_phase"] = True
        partial_artifacts["phase_timeout_name"] = "pre-codegen"
        partial_artifacts["phase_timeout_seconds"] = PRE_CODEGEN_TIMEOUT_SECONDS

    def _prepare_initial_prompt_state():
        visual_feedback = _capture_initial_visual_feedback(
            env, obs, config, args, visual_differencing_args
        )
        if config["use_oracle_code"]:
            generated_raw_code = env.oracle_code
            with open(os.path.join(config["output_dir"], "oracle_code.py"), "w") as f:
                f.write(generated_raw_code)
            generated_reasoning = None
            generated_ensemble_data = None
            generated_raw_model_response = None
        else:
            (
                generated_raw_code,
                generated_reasoning,
                generated_ensemble_data,
                generated_raw_model_response,
            ) = _query_initial_code(
                args, config, obs
            )
        return (
            visual_feedback,
            generated_raw_code,
            generated_reasoning,
            generated_ensemble_data,
            generated_raw_model_response,
        )

    # --- 2. Capture initial visual feedback and first code generation ---
    (
        (
            visual_feedback_imgs,
            visual_feedback_base64_history,
            task_description,
            initial_scene_artifact,
        ),
        raw_code,
        reasoning,
        ensemble_data,
        initial_raw_model_response,
    ) = _run_with_phase_timeout(
        trial,
        "pre-codegen",
        PRE_CODEGEN_TIMEOUT_SECONDS,
        _prepare_initial_prompt_state,
    )

    trajectory_data = create_trajectory_buffer(
        trial=trial,
        attempt=attempt_idx,
        config_path=args.config_path,
        task_prompt=obs["full_prompt"][-1]["content"][0]["text"],
    )
    transition_dataset = env.get_transition_dataset() if hasattr(env, "get_transition_dataset") else None
    if transition_dataset is not None:
        transition_dataset["trial"] = trial
        transition_dataset["attempt"] = attempt_idx
        transition_dataset["config_path"] = args.config_path
        transition_dataset["task_prompt"] = obs["full_prompt"][-1]["content"][0]["text"]
    reset_state = env.get_reset_state() if hasattr(env, "get_reset_state") else None
    reset_snapshot_id = append_snapshot(
        trajectory_data,
        state=reset_state,
        env=env if reset_state is None else None,
        snapshot_type="reset",
        label="canonical_reset_state",
        phase_tags=["reset", "home"],
    )
    append_event(
        trajectory_data,
        "reset_complete",
        snapshot_id=reset_snapshot_id,
        task_description=task_description,
    )

    # Seed wrist base64 history with initial wrist image
    if use_wrist and wrist_base64_history is not None and hasattr(env, "render_wrist"):
        wrist_img = env.render_wrist()
        if wrist_img is not None:
            pil_wrist = Image.fromarray(wrist_img)
            buf = io.BytesIO()
            pil_wrist.save(buf, format="png")
            wrist_base64_history.append(
                f"data:image/png;base64,"
                f"{base64.b64encode(buf.getvalue()).decode('utf-8')}"
            )

    # Initialize partial artifacts for timeout recovery
    if partial_artifacts is not None:
        partial_artifacts["pre_codegen_phase"] = False
        partial_artifacts.pop("phase_timeout_name", None)
        partial_artifacts.pop("phase_timeout_seconds", None)
        partial_artifacts.update({
            "attempt_idx": attempt_idx,
            "raw_code": raw_code,
            "code_blocks": code_blocks,
            "code_block_metadata": code_block_metadata,
            "all_responses": all_responses,
            "visual_feedback_imgs": visual_feedback_imgs,
            "info_step": info_step,
            "reward": reward,
            "terminated": terminated,
            "truncated": truncated,
            "num_regenerations": num_regenerations,
            "num_finishes": num_finishes,
            "num_code_blocks": 0,
            "ensemble_data": ensemble_data,
            "multiturn_ensemble_data": multiturn_ensemble_data,
            "trajectory_data": trajectory_data,
            "transition_dataset": transition_dataset,
        })

    def _build_trial_metadata() -> dict[str, Any]:
        stderr_value = info_step.get("stderr", "")
        num_nonempty_code_blocks = _count_nonempty_code_blocks(code_blocks)
        exclude_from_training = bool(
            truncated or "executing action in terminated episode" in stderr_value
        )
        exclusion_reason = "sim_limit_reset" if exclude_from_training else None
        if num_nonempty_code_blocks <= 0:
            exclude_from_training = True
            exclusion_reason = "empty_codegen"
        return {
            "trial": trial,
            "attempt": attempt_idx,
            "sandbox_rc": int(info_step.get("sandbox_rc", 1)),
            "reward": float(reward),
            "task_completed": bool(info_step.get("task_completed", False)),
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "success": bool(info_step.get("sandbox_rc", 1) == 0),
            "exclude_from_training": exclude_from_training,
            "exclusion_reason": exclusion_reason,
            "num_code_blocks": len(code_blocks),
            "num_nonempty_code_blocks": num_nonempty_code_blocks,
        }

    # Parse initial code into blocks
    initial_blocks = _extract_code(raw_code)
    all_responses.append({
        "block_idx": [0],
        "code_blocks": initial_blocks,
        "decision": "initial",
        "initial_prompt": copy.deepcopy(obs["full_prompt"]),
        "reasoning": reasoning if reasoning is not None else "",
        "initial_model_finish_reason": (
            initial_raw_model_response.get("finish_reason")
            if initial_raw_model_response is not None
            else None
        ),
        "initial_model_raw_response": (
            copy.deepcopy(initial_raw_model_response.get("raw_response"))
            if initial_raw_model_response is not None
            else None
        ),
    })
    if initial_scene_artifact is not None:
        all_responses.append({
            "initial_scene_model": initial_scene_artifact.get("model", ""),
            "initial_scene_task_description": initial_scene_artifact.get("task_description", ""),
            "initial_scene_prompt": copy.deepcopy(initial_scene_artifact.get("prompt", [])),
            "initial_scene_raw_response": initial_scene_artifact.get("content", ""),
            "initial_scene_reasoning": initial_scene_artifact.get("reasoning") or "",
        })
    append_event(
        trajectory_data,
        "initial_generation",
        block_idx=[0],
        code_blocks=initial_blocks,
        reasoning=reasoning if reasoning is not None else "",
        used_oracle_code=config["use_oracle_code"],
    )
    if _count_nonempty_code_blocks(initial_blocks) > 0:
        code_blocks.extend(initial_blocks)
        code_block_metadata.extend([{"generation": 0, "regenerated": False}] * len(initial_blocks))
    else:
        info_step = {
            "sandbox_rc": 1,
            "stdout": "",
            "stderr": "Model returned no executable code in initial generation.",
            "task_completed": False,
        }
        append_event(
            trajectory_data,
            "empty_codegen",
            phase="initial_generation",
            raw_code=raw_code,
        )

    with open(os.path.join(config["output_dir"], "all_responses.json"), "w") as f:
        json.dump(all_responses, f)

    if args.debug:
        with open(os.path.join(config["output_dir"], "code_init.txt"), "w") as f:
            f.write("\n".join(initial_blocks))

    # --- 4. Execute code blocks (with optional multi-turn) ---
    info_step = {"sandbox_rc": -1, "stdout": "", "stderr": ""}
    reward = 0.0
    terminated = truncated = False
    code_block_idx = 0

    if not code_blocks:
        info_step = {
            "sandbox_rc": 1,
            "stdout": "",
            "stderr": "Model returned no executable code in initial generation.",
            "task_completed": False,
        }
        final_code = _annotate_code_blocks(code_blocks, code_block_metadata)
        log_lines = _build_log_lines(
            final_code,
            info_step,
            reward,
            terminated,
            truncated,
            num_regenerations,
            num_finishes,
            0,
            stderr_override="Model returned no executable code in initial generation.",
        )
        code_path = _save_trial_artifacts(
            config, trial, attempt_idx, info_step["sandbox_rc"], reward,
            False, final_code, raw_code, all_responses, log_lines, visual_feedback_imgs,
            ensemble_data=ensemble_data,
            multiturn_ensemble_data=multiturn_ensemble_data,
            trajectory_data=trajectory_data,
            transition_dataset=transition_dataset,
            trial_metadata=_build_trial_metadata(),
        )
        final_summary = TrialSummary(
            trial=trial,
            success=False,
            reward=reward,
            terminated=terminated,
            truncated=truncated,
            sandbox_rc=info_step["sandbox_rc"],
            log="\n".join(log_lines),
            task_completed=False,
            code_path=code_path,
            num_regenerations=num_regenerations,
            num_finishes=num_finishes,
            num_code_blocks=0,
        )
        if partial_artifacts is not None:
            partial_artifacts["final_summary"] = final_summary
        return final_summary

    # Track whether we're recording frames (for video diff or record_video)
    recording_frames = (
        (config["record_video"] or use_video_diff)
        and hasattr(env, "get_video_frame_count")
    )

    while code_block_idx < len(code_blocks) and code_block_idx <= MULTITURN_LIMIT:
        code = code_blocks[code_block_idx]
        current_block_idx = code_block_idx
        code_block_idx += 1
        phase_candidates = infer_phase_candidates(code)
        phase_tags = phase_tags_from_candidates(phase_candidates)

        pre_snapshot_id = append_snapshot(
            trajectory_data,
            env=env,
            snapshot_type="pre_code",
            code_block_idx=current_block_idx,
            label=f"before_code_block_{current_block_idx:02d}",
            phase_candidates=phase_candidates,
            phase_tags=phase_tags,
        )

        # Record frame index before step
        frame_start = env.get_video_frame_count() if recording_frames else 0

        try:
            obs_next, reward, terminated, truncated, info_step = env.step(code)
        except Exception as exc:
            # Keep trial bookkeeping alive even if a simulator step raises, so we
            # can still save attempt artifacts instead of dropping the whole trial.
            info_step = {
                "sandbox_rc": 1,
                "stdout": "",
                "stderr": traceback.format_exc(),
                "task_completed": False,
            }
            reward = float(reward)
            terminated = False
            truncated = "terminated episode" in str(exc).lower()
            append_event(
                trajectory_data,
                "step_exception",
                code_block_idx=current_block_idx,
                code=code,
                phase_candidates=phase_candidates,
                phase_tags=phase_tags,
                snapshot_before=pre_snapshot_id,
                error_type=type(exc).__name__,
                error_message=str(exc),
                stderr=info_step["stderr"],
            )
            break

        # Record frame index after step
        frame_end = env.get_video_frame_count() if recording_frames else 0
        turn_frame_ranges.append((frame_start, frame_end))

        post_snapshot_id = append_snapshot(
            trajectory_data,
            env=env,
            snapshot_type="post_code",
            code_block_idx=current_block_idx,
            label=f"after_code_block_{current_block_idx:02d}",
            phase_candidates=phase_candidates,
            phase_tags=phase_tags,
        )
        append_event(
            trajectory_data,
            "code_execution",
            code_block_idx=current_block_idx,
            code=code,
            phase_candidates=phase_candidates,
            phase_tags=phase_tags,
            snapshot_before=pre_snapshot_id,
            snapshot_after=post_snapshot_id,
            reward=reward,
            terminated=terminated,
            truncated=truncated,
            sandbox_rc=info_step.get("sandbox_rc"),
            task_completed=info_step.get("task_completed"),
            stdout=info_step.get("stdout", ""),
            stderr=info_step.get("stderr", ""),
        )

        if partial_artifacts is not None:
            partial_artifacts.update({
                "info_step": info_step,
                "reward": reward,
                "terminated": terminated,
                "truncated": truncated,
            "trajectory_data": trajectory_data,
            "transition_dataset": transition_dataset,
        })

        obs = obs_next

        if info_step.get("task_completed", False):
            append_event(
                trajectory_data,
                "task_completed_auto_finish",
                code_block_idx=current_block_idx,
                reward=reward,
                terminated=terminated,
                truncated=truncated,
            )
            break

        # Multi-turn decision
        if multi_turn_prompt:
            if "terminated episode" in info_step["stderr"]:
                truncated = True
                break

            # Get turn frames for video differencing
            turn_frames = None
            wrist_turn_frames = None
            if use_video_diff and recording_frames:
                turn_frames = env.get_video_frames_range(frame_start, frame_end)
                if use_wrist and hasattr(env, "get_wrist_video_frames_range"):
                    wrist_turn_frames = env.get_wrist_video_frames_range(
                        frame_start, frame_end,
                    )

            decision, new_code, mt_reasoning, mt_ensemble, decision_prompt = _handle_multi_turn_step(
                env, obs, args, config, visual_differencing_args,
                multi_turn_prompt, code_blocks, code_block_idx, info_step,
                task_description, visual_feedback_imgs, visual_feedback_base64_history,
                stderr_history,
                turn_frames=turn_frames,
                wrist_turn_frames=wrist_turn_frames,
                wrist_base64_history=wrist_base64_history,
            )

            if mt_ensemble is not None:
                mt_ensemble["regeneration"] = num_regenerations + 1
                multiturn_ensemble_data.append(mt_ensemble)

            if decision == "regenerate":
                print("Model chose to regenerate code")
                new_blocks = _extract_code(new_code)
                if _count_nonempty_code_blocks(new_blocks) <= 0:
                    all_responses.append({
                        "multi_turn_prompt": decision_prompt if config.get("save_multiturn_prompts", False) else None,
                        "block_idx": [current_block_idx],
                        "code_blocks": new_blocks,
                        "decision": "regenerate_empty",
                        "reasoning": mt_reasoning if mt_reasoning is not None else "",
                    })
                    append_event(
                        trajectory_data,
                        "empty_codegen",
                        phase="regenerate",
                        code_block_idx=current_block_idx,
                        raw_code=new_code,
                    )
                    info_step["sandbox_rc"] = 1
                    info_step["stderr"] = "Model returned no executable code during regenerate."
                    break
                rollback_performed = False
                rollback_snapshot_id = None
                safe_snapshot_details = None
                model_snapshot_selection_details = None
                insert_idx = code_block_idx
                if enable_eap_rollback:
                    executed_code_for_selection = "\n".join(code_blocks[: current_block_idx + 1])
                    if enable_eap_model_snapshot_selection:
                        (
                            selected_snapshot_id,
                            safe_snapshot_details,
                            model_snapshot_selection_details,
                        ) = _select_safe_snapshot_with_model(
                            args,
                            obs,
                            trajectory_data,
                            current_code_block_idx=current_block_idx,
                            purpose="regenerate",
                            preferred_snapshot_id=pre_snapshot_id,
                            executed_code=executed_code_for_selection,
                            console_stdout=info_step.get("stdout", ""),
                            console_stderr=info_step.get("stderr", ""),
                        )
                        append_event(
                            trajectory_data,
                            "model_snapshot_selection",
                            code_block_idx=current_block_idx,
                            purpose="regenerate",
                            selection_prompt=model_snapshot_selection_details.get("selection_prompt")
                            if config.get("save_multiturn_prompts", False)
                            else None,
                            **{
                                key: value
                                for key, value in model_snapshot_selection_details.items()
                                if key not in {"selection_prompt", "purpose"}
                            },
                        )
                    else:
                        selected_snapshot_id, safe_snapshot_details = _select_safe_snapshot(
                            trajectory_data,
                            current_code_block_idx=current_block_idx,
                            purpose="regenerate",
                            preferred_snapshot_id=pre_snapshot_id,
                        )
                    append_event(
                        trajectory_data,
                        "safe_snapshot_selected",
                        code_block_idx=current_block_idx,
                        purpose="regenerate",
                        **safe_snapshot_details,
                    )
                else:
                    selected_snapshot_id = pre_snapshot_id

                if enable_eap_rollback and selected_snapshot_id is not None:
                    restored_obs, rollback_snapshot_id = _restore_rollback_snapshot(
                        env,
                        trajectory_data,
                        selected_snapshot_id,
                        code_block_idx=current_block_idx,
                        reason="regenerate",
                    )
                    if restored_obs is not None:
                        obs = restored_obs
                        rollback_performed = True
                        insert_idx = current_block_idx
                all_responses.append({
                    "multi_turn_prompt": decision_prompt if config.get("save_multiturn_prompts", False) else None,
                    "block_idx": [insert_idx],
                    "code_blocks": new_blocks,
                    "decision": "regenerate",
                    "reasoning": mt_reasoning if mt_reasoning is not None else "",
                })
                append_event(
                    trajectory_data,
                    "multi_turn_decision",
                    code_block_idx=current_block_idx,
                    decision="regenerate",
                    reasoning=mt_reasoning if mt_reasoning is not None else "",
                    replacement_code_blocks=new_blocks,
                    rollback_performed=rollback_performed,
                    rollback_snapshot_id=rollback_snapshot_id,
                    safe_snapshot_details=safe_snapshot_details,
                    model_snapshot_selection_details=model_snapshot_selection_details,
                )
                del code_blocks[insert_idx:]
                del code_block_metadata[insert_idx:]
                code_blocks.extend(new_blocks)
                code_block_metadata.extend(
                    [{"generation": num_regenerations + 1, "regenerated": True,
                      "regenerated_at_idx": insert_idx}]
                    * len(new_blocks)
                )
                code_block_idx = insert_idx
                num_regenerations += 1
                if partial_artifacts is not None:
                    partial_artifacts["num_regenerations"] = num_regenerations

            elif decision == "finish":
                attempted_recovery = False
                model_snapshot_selection_details = None
                if (
                    enable_eap_recovery
                    and num_recoveries < 1
                    and not info_step.get("task_completed", False)
                ):
                    executed_code_for_selection = "\n".join(code_blocks[: current_block_idx + 1])
                    if enable_eap_model_snapshot_selection:
                        (
                            selected_snapshot_id,
                            safe_snapshot_details,
                            model_snapshot_selection_details,
                        ) = _select_safe_snapshot_with_model(
                            args,
                            obs,
                            trajectory_data,
                            current_code_block_idx=current_block_idx,
                            purpose="finish_incomplete_recovery",
                            preferred_snapshot_id=pre_snapshot_id,
                            executed_code=executed_code_for_selection,
                            console_stdout=info_step.get("stdout", ""),
                            console_stderr=info_step.get("stderr", ""),
                        )
                        append_event(
                            trajectory_data,
                            "model_snapshot_selection",
                            code_block_idx=current_block_idx,
                            purpose="finish_incomplete_recovery",
                            selection_prompt=model_snapshot_selection_details.get("selection_prompt")
                            if config.get("save_multiturn_prompts", False)
                            else None,
                            **{
                                key: value
                                for key, value in model_snapshot_selection_details.items()
                                if key not in {"selection_prompt", "purpose"}
                            },
                        )
                    else:
                        selected_snapshot_id, safe_snapshot_details = _select_safe_snapshot(
                            trajectory_data,
                            current_code_block_idx=current_block_idx,
                            purpose="finish_incomplete_recovery",
                            preferred_snapshot_id=pre_snapshot_id,
                        )
                    append_event(
                        trajectory_data,
                        "safe_snapshot_selected",
                        code_block_idx=current_block_idx,
                        purpose="finish_incomplete_recovery",
                        **safe_snapshot_details,
                    )
                    if selected_snapshot_id is not None:
                        restored_obs, rollback_snapshot_id = _restore_rollback_snapshot(
                            env,
                            trajectory_data,
                            selected_snapshot_id,
                            code_block_idx=current_block_idx,
                            reason="finish_incomplete_recovery",
                        )
                    else:
                        restored_obs, rollback_snapshot_id = None, None

                    if restored_obs is not None:
                        obs = restored_obs
                        recovery_raw_code, recovery_reasoning, recovery_prompt = _query_recovery_code(
                            args,
                            obs,
                            executed_code="\n".join(code_blocks[:code_block_idx]),
                            console_stdout=info_step.get("stdout", ""),
                            console_stderr=info_step.get("stderr", ""),
                            recovery_reason="model_finished_but_task_incomplete",
                        )
                        recovery_blocks = _extract_code(recovery_raw_code)
                        if _count_nonempty_code_blocks(recovery_blocks) > 0:
                            insert_idx = current_block_idx
                            all_responses.append({
                                "multi_turn_prompt": recovery_prompt if config.get("save_multiturn_prompts", False) else None,
                                "block_idx": [insert_idx],
                                "code_blocks": recovery_blocks,
                                "decision": "recovery",
                                "reasoning": recovery_reasoning if recovery_reasoning is not None else "",
                            })
                            append_event(
                                trajectory_data,
                                "recovery_generation",
                                code_block_idx=current_block_idx,
                                reason="model_finished_but_task_incomplete",
                                rollback_snapshot_id=rollback_snapshot_id,
                                safe_snapshot_details=safe_snapshot_details,
                                model_snapshot_selection_details=model_snapshot_selection_details,
                                replacement_code_blocks=recovery_blocks,
                            )
                            del code_blocks[insert_idx:]
                            del code_block_metadata[insert_idx:]
                            code_blocks.extend(recovery_blocks)
                            code_block_metadata.extend(
                                [{"generation": num_regenerations + num_recoveries + 1, "regenerated": True,
                                  "regenerated_at_idx": insert_idx, "recovery": True}]
                                * len(recovery_blocks)
                            )
                            code_block_idx = insert_idx
                            num_recoveries += 1
                            attempted_recovery = True
                        else:
                            append_event(
                                trajectory_data,
                                "empty_codegen",
                                phase="recovery",
                                code_block_idx=current_block_idx,
                                raw_code=recovery_raw_code,
                            )

                if attempted_recovery:
                    continue

                all_responses.append({
                    "decision": "finish",
                    "reasoning": mt_reasoning if mt_reasoning is not None else (new_code or ""),
                })
                append_event(
                    trajectory_data,
                    "multi_turn_decision",
                    code_block_idx=current_block_idx,
                    decision="finish",
                    reasoning=mt_reasoning if mt_reasoning is not None else (new_code or ""),
                    attempted_recovery=attempted_recovery,
                )
                print("Model chose to finish")
                num_finishes += 1
                if partial_artifacts is not None:
                    partial_artifacts["num_finishes"] = num_finishes
                break

        print(f"Code block {code_block_idx} done")
        print(f"Number of code blocks: {len(code_blocks)}")

        # Save intermediate artifacts (code, logs) per code block
        final_code = _annotate_code_blocks(code_blocks, code_block_metadata)
        _save_trial_artifacts(
            config, trial, attempt_idx, info_step["sandbox_rc"], reward,
            info_step.get("task_completed", False), final_code, raw_code,
            all_responses, ["-" * 100, "Generated program:", final_code],
            visual_feedback_imgs,
            trial_metadata=_build_trial_metadata(),
        )

        # Only save intermediate video if NOT doing per-turn saving
        # (per-turn saving is deferred to after the loop to avoid clearing the buffer)
        if not recording_frames:
            _save_trial_video(
                env, config, trial, attempt_idx, info_step, reward, len(code_blocks),
                suffix_extra=str(len(code_blocks)),
            )

    print("Code blocks done")

    # --- 5. Build final summary ---
    final_code = _annotate_code_blocks(code_blocks, code_block_metadata)
    num_code_blocks = len(code_blocks)

    if partial_artifacts is not None:
        partial_artifacts["final_code"] = final_code
        partial_artifacts["num_code_blocks"] = num_code_blocks

    # Override sandbox_rc for terminated-episode stderr
    if "executing action in terminated episode" in info_step["stderr"]:
        sandbox_rc_override = 0
    if sandbox_rc_override is not None:
        info_step["sandbox_rc"] = sandbox_rc_override

    stderr = "\n\n".join(stderr_history) if stderr_history else info_step["stderr"]
    log_lines = _build_log_lines(
        final_code, info_step, reward, terminated, truncated,
        num_regenerations, num_finishes, num_code_blocks,
        stderr_override=stderr,
    )

    success = info_step["sandbox_rc"] == 0
    append_event(
        trajectory_data,
        "trial_complete",
        success=success,
        reward=reward,
        terminated=terminated,
        truncated=truncated,
        sandbox_rc=info_step["sandbox_rc"],
        task_completed=info_step.get("task_completed", False),
        num_regenerations=num_regenerations,
        num_finishes=num_finishes,
        num_code_blocks=num_code_blocks,
    )
    trial_metadata = _build_trial_metadata()

    code_path = _save_trial_artifacts(
        config, trial, attempt_idx, info_step["sandbox_rc"], reward,
        info_step.get("task_completed", False), final_code, raw_code,
        all_responses, log_lines, visual_feedback_imgs,
        ensemble_data=ensemble_data,
        multiturn_ensemble_data=multiturn_ensemble_data,
        trajectory_data=trajectory_data,
        transition_dataset=transition_dataset,
        trial_metadata=trial_metadata,
    )

    final_summary = TrialSummary(
        trial=trial,
        success=success,
        reward=reward,
        terminated=terminated,
        truncated=truncated,
        sandbox_rc=info_step["sandbox_rc"],
        log="\n".join(log_lines),
        task_completed=info_step.get("task_completed", None),
        code_path=code_path,
        num_regenerations=num_regenerations,
        num_finishes=num_finishes,
        num_code_blocks=num_code_blocks,
    )
    if partial_artifacts is not None:
        partial_artifacts["final_summary"] = final_summary

    # Save per-turn and combined videos
    if recording_frames and turn_frame_ranges:
        _save_turn_and_combined_videos(
            env, config, trial, attempt_idx, info_step, reward, turn_frame_ranges,
        )
    else:
        _save_trial_video(env, config, trial, attempt_idx, info_step, reward, num_code_blocks)

    # --- Evolving skill library integration (opt-in) ---
    if config.get("evolve_skill_library", False) and info_step.get("task_completed", False):
        try:
            from capx.skills import SkillLibrary

            skill_lib_path = config.get("skill_library_path", None)
            skill_lib = SkillLibrary(path=skill_lib_path)
            task_name = config.get("task_name", f"trial_{trial}")
            new_skills = skill_lib.extract_from_code(final_code, task_name=task_name)
            skill_lib.save()
            if new_skills:
                print(f"[SkillLibrary] Extracted {len(new_skills)} new skill(s): {new_skills}")
        except Exception as exc:
            print(f"[SkillLibrary] Skill extraction failed: {exc}")

    print(f"Trial {trial} took {time.time() - trial_start_time:.2f} seconds")

    gc.collect()

    return final_summary


def _patch_libero_goal(env: CodeExecutionEnvBase, obs: dict[str, Any]) -> None:
    """Inject the LIBERO task language into the prompt template if applicable."""
    if not hasattr(env.low_level_env, "handle"):
        return
    handle = env.low_level_env.handle
    if (
        hasattr(handle, "task_language")
        and "libero_environment_goal" in obs["full_prompt"][-1]["content"][0]["text"]
    ):
        goal = getattr(handle, "task_language")
        obs["full_prompt"][-1]["content"][0]["text"] = (
            obs["full_prompt"][-1]["content"][0]["text"].format(
                libero_environment_goal=goal
            )
        )


def _restore_rollback_snapshot(
    env: CodeExecutionEnvBase,
    trajectory_data: dict[str, Any],
    snapshot_id: str,
    *,
    code_block_idx: int,
    reason: str,
) -> tuple[dict[str, Any] | None, str | None]:
    """Restore a previously captured snapshot and append rollback metadata."""
    snapshot_payloads = trajectory_data.get("_snapshot_payloads", {})
    snapshot_state = snapshot_payloads.get(snapshot_id)
    if snapshot_state is None:
        return None, None
    source_snapshot_meta = next(
        (snapshot for snapshot in trajectory_data.get("snapshots", []) if snapshot.get("snapshot_id") == snapshot_id),
        None,
    )
    source_phase_candidates = source_snapshot_meta.get("phase_candidates", []) if source_snapshot_meta else []
    source_phase_tags = source_snapshot_meta.get("phase_tags", []) if source_snapshot_meta else []

    env.restore_state(snapshot_state)
    restored_snapshot_id = append_snapshot(
        trajectory_data,
        env=env,
        snapshot_type="rollback_restore",
        code_block_idx=code_block_idx,
        label=f"rollback_after_code_block_{code_block_idx:02d}",
        phase_candidates=source_phase_candidates,
        phase_tags=source_phase_tags,
    )
    append_event(
        trajectory_data,
        "rollback",
        code_block_idx=code_block_idx,
        reason=reason,
        source_snapshot_id=snapshot_id,
        restored_snapshot_id=restored_snapshot_id,
    )

    obs = env._get_observation()
    _patch_libero_goal(env, obs)
    return obs, restored_snapshot_id


def _is_open_gripper_state(state: dict[str, Any]) -> bool:
    gripper_fraction = state.get("gripper_fraction")
    if gripper_fraction is not None:
        return float(gripper_fraction) >= 0.9

    current_obs = state.get("current_obs")
    if isinstance(current_obs, dict):
        robot_joint_pos = current_obs.get("robot_joint_pos")
        if robot_joint_pos is not None and len(robot_joint_pos) > 0:
            return float(np.asarray(robot_joint_pos)[-1]) >= 0.9
    return False


def _collect_safe_snapshot_candidates(
    trajectory_data: dict[str, Any],
    *,
    current_code_block_idx: int,
) -> list[dict[str, Any]]:
    snapshots = trajectory_data.get("snapshots", [])
    snapshot_payloads = trajectory_data.get("_snapshot_payloads", {})

    allowed_types = {"reset", "pre_code", "rollback_restore"}
    candidates: list[dict[str, Any]] = []
    for snapshot_meta in snapshots:
        snapshot_id = snapshot_meta.get("snapshot_id")
        if snapshot_id is None or snapshot_meta.get("snapshot_type") not in allowed_types:
            continue
        code_block_idx = snapshot_meta.get("code_block_idx")
        if code_block_idx is not None and code_block_idx > current_code_block_idx:
            continue
        state = snapshot_payloads.get(snapshot_id)
        if state is None:
            continue
        candidates.append(
            {
                "snapshot_id": snapshot_id,
                "snapshot_type": snapshot_meta.get("snapshot_type"),
                "code_block_idx": code_block_idx,
                "is_open_gripper": _is_open_gripper_state(state),
                "phase_tags": snapshot_meta.get("phase_tags", []),
                "summary": snapshot_meta.get("summary", {}),
            }
        )
    return candidates


def _preferred_safe_snapshot_phases(purpose: str) -> list[str]:
    preferred_phases_by_purpose = {
        "regenerate": ["prepare_grasp", "perceive", "plan_grasp", "home"],
        "finish_incomplete_recovery": ["prepare_grasp", "perceive", "plan_grasp", "home", "release"],
    }
    return preferred_phases_by_purpose.get(
        purpose,
        ["prepare_grasp", "perceive", "plan_grasp", "home"],
    )


def _select_safe_snapshot(
    trajectory_data: dict[str, Any],
    *,
    current_code_block_idx: int,
    purpose: str,
    preferred_snapshot_id: str | None = None,
) -> tuple[str | None, dict[str, Any]]:
    candidates = _collect_safe_snapshot_candidates(
        trajectory_data,
        current_code_block_idx=current_code_block_idx,
    )
    preferred_phases = _preferred_safe_snapshot_phases(purpose)

    selection_details = {
        "candidate_count": len(candidates),
        "candidate_snapshot_ids": [candidate["snapshot_id"] for candidate in candidates],
        "preferred_snapshot_id": preferred_snapshot_id,
        "preferred_phases": preferred_phases,
        "strategy": "latest_preferred_phase_and_open_gripper_else_latest_open_gripper_else_latest_candidate",
    }
    if not candidates:
        return None, selection_details

    phase_and_open_candidates = [
        candidate
        for candidate in candidates
        if candidate["is_open_gripper"]
        and any(tag in preferred_phases for tag in candidate.get("phase_tags", []))
    ]
    open_candidates = [candidate for candidate in candidates if candidate["is_open_gripper"]]
    if phase_and_open_candidates:
        selected = phase_and_open_candidates[-1]
    elif open_candidates:
        selected = open_candidates[-1]
    else:
        selected = candidates[-1]

    if preferred_snapshot_id is not None:
        preferred = next((c for c in candidates if c["snapshot_id"] == preferred_snapshot_id), None)
        if preferred is not None and preferred["is_open_gripper"]:
            selected = preferred
            selection_details["strategy"] = "preferred_snapshot_is_safe"

    selection_details["selected_snapshot_id"] = selected["snapshot_id"]
    selection_details["selected_snapshot_type"] = selected["snapshot_type"]
    selection_details["selected_snapshot_code_block_idx"] = selected["code_block_idx"]
    selection_details["selected_snapshot_open_gripper"] = selected["is_open_gripper"]
    selection_details["selected_snapshot_phase_tags"] = selected.get("phase_tags", [])
    return selected["snapshot_id"], selection_details


def _select_safe_snapshot_with_model(
    args: LaunchArgs,
    obs: dict[str, Any],
    trajectory_data: dict[str, Any],
    *,
    current_code_block_idx: int,
    purpose: str,
    preferred_snapshot_id: str | None = None,
    executed_code: str,
    console_stdout: str,
    console_stderr: str,
) -> tuple[str | None, dict[str, Any], dict[str, Any]]:
    heuristic_selected_snapshot_id, heuristic_details = _select_safe_snapshot(
        trajectory_data,
        current_code_block_idx=current_code_block_idx,
        purpose=purpose,
        preferred_snapshot_id=preferred_snapshot_id,
    )
    candidates = _collect_safe_snapshot_candidates(
        trajectory_data,
        current_code_block_idx=current_code_block_idx,
    )
    preferred_phases = _preferred_safe_snapshot_phases(purpose)
    model_details: dict[str, Any] = {
        "purpose": purpose,
        "candidate_count": len(candidates),
        "candidate_snapshot_ids": [candidate["snapshot_id"] for candidate in candidates],
        "heuristic_selected_snapshot_id": heuristic_selected_snapshot_id,
    }
    if not candidates:
        model_details["fallback_used"] = True
        model_details["fallback_reason"] = "no_candidates"
        return heuristic_selected_snapshot_id, heuristic_details, model_details

    model_selected_snapshot_id, raw_content, reasoning, selection_prompt, parsed_reason = _query_snapshot_selection(
        args,
        obs,
        purpose=purpose,
        candidates=candidates,
        preferred_phases=preferred_phases,
        executed_code=executed_code,
        console_stdout=console_stdout,
        console_stderr=console_stderr,
    )
    model_details.update({
        "raw_response": raw_content,
        "reasoning": reasoning,
        "selection_prompt": selection_prompt,
        "parsed_reason": parsed_reason,
        "model_selected_snapshot_id": model_selected_snapshot_id,
    })

    candidate_ids = {candidate["snapshot_id"] for candidate in candidates}
    if model_selected_snapshot_id not in candidate_ids:
        model_details["fallback_used"] = True
        model_details["fallback_reason"] = "invalid_model_selection"
        return heuristic_selected_snapshot_id, heuristic_details, model_details

    selected_candidate = next(
        candidate for candidate in candidates if candidate["snapshot_id"] == model_selected_snapshot_id
    )
    selection_details = dict(heuristic_details)
    selection_details.update({
        "strategy": "model_selected_candidate",
        "selected_snapshot_id": selected_candidate["snapshot_id"],
        "selected_snapshot_type": selected_candidate["snapshot_type"],
        "selected_snapshot_code_block_idx": selected_candidate["code_block_idx"],
        "selected_snapshot_open_gripper": selected_candidate["is_open_gripper"],
        "selected_snapshot_phase_tags": selected_candidate.get("phase_tags", []),
        "model_selected_snapshot_id": model_selected_snapshot_id,
        "heuristic_selected_snapshot_id": heuristic_selected_snapshot_id,
    })
    model_details["fallback_used"] = False
    model_details["fallback_reason"] = None
    model_details["final_selected_snapshot_id"] = model_selected_snapshot_id
    return model_selected_snapshot_id, selection_details, model_details
