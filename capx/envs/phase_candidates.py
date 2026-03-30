from __future__ import annotations

import re
from typing import Any


PERCEPTION_CALLS = {
    "get_observation",
    "get_object_pose",
    "get_all_object_poses",
    "get_object_3d_points_and_masks_from_language",
    "segment_sam3_text_prompt",
    "segment_sam3_point_prompt",
    "point_prompt_molmo",
}

PLANNING_CALLS = {
    "sample_grasp_pose",
    "plan_grasp_trajectory",
    "plan_with_grasped_object",
    "get_oriented_bounding_box_from_3d_points",
    "select_top_down_grasp",
}

MOTION_CALLS = {
    "goto_pose",
    "move_to_joints_blocking",
    "execute_joint_trajectory",
}

STATE_CALLS = {
    "open_gripper",
    "close_gripper",
    "goto_home_joint_position",
}

KNOWN_CALLS = sorted(PERCEPTION_CALLS | PLANNING_CALLS | MOTION_CALLS | STATE_CALLS, key=len, reverse=True)
CALL_PATTERN = re.compile(r"\b(" + "|".join(re.escape(name) for name in KNOWN_CALLS) + r")\s*\(")


def _line_number_from_offset(code: str, offset: int) -> int:
    return code.count("\n", 0, offset) + 1


def infer_phase_candidates(code: str) -> list[dict[str, Any]]:
    """Infer coarse task phases from ordered API calls in generated code."""
    matches = list(CALL_PATTERN.finditer(code))
    if not matches:
        return []

    candidates: list[dict[str, Any]] = []
    last_phase: str | None = None
    gripper_closed = False

    def push(phase: str, call_name: str, line_no: int) -> None:
        nonlocal last_phase
        if candidates and candidates[-1]["phase"] == phase:
            candidates[-1]["evidence_calls"].append(call_name)
            candidates[-1]["line_end"] = line_no
        else:
            candidates.append(
                {
                    "phase": phase,
                    "line_start": line_no,
                    "line_end": line_no,
                    "evidence_calls": [call_name],
                }
            )
        last_phase = phase

    for match in matches:
        call_name = match.group(1)
        line_no = _line_number_from_offset(code, match.start())

        if call_name in PERCEPTION_CALLS:
            push("perceive", call_name, line_no)
            continue
        if call_name in PLANNING_CALLS:
            push("plan_grasp", call_name, line_no)
            continue
        if call_name == "open_gripper":
            phase = "release" if gripper_closed else "prepare_grasp"
            gripper_closed = False
            push(phase, call_name, line_no)
            continue
        if call_name == "close_gripper":
            gripper_closed = True
            push("grasp", call_name, line_no)
            continue
        if call_name == "goto_home_joint_position":
            push("home", call_name, line_no)
            continue
        if call_name in MOTION_CALLS:
            if gripper_closed:
                phase = "transport"
            elif last_phase in {"perceive", "plan_grasp", "prepare_grasp"}:
                phase = "approach"
            elif last_phase == "release":
                phase = "retreat"
            else:
                phase = "move"
            push(phase, call_name, line_no)
            continue

    return candidates


def phase_tags_from_candidates(phase_candidates: list[dict[str, Any]]) -> list[str]:
    tags: list[str] = []
    for candidate in phase_candidates:
        phase = candidate.get("phase")
        if isinstance(phase, str) and phase not in tags:
            tags.append(phase)
    return tags
