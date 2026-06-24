#!/usr/bin/env python3
"""Smoke-test the shared RoboCasa GR1 skill interface across target tasks.

This bypasses model code generation and exercises the same high-level skill
shape on PnPPouring, TwoArmLift, and a drawer-opening task:

    observe_scene -> probe_eef_directions -> select_gr1_skill
    -> execute_gr1_skill -> verify_success_or_contact

The script is a harness validation tool. A task may fail its success predicate
while the shared interface still passes if it creates transitions, videos, and a
diagnostic summary without task-specific call-site logic.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

import numpy as np

from capx.envs.configs.instantiate import instantiate
from capx.envs.transition_dataset import save_transition_dataset


TASK_SPECS: dict[str, dict[str, Any]] = {
    "pnp_pouring": {
        "env_name": "PnPPouring",
        "skill_type": "pnp_pouring",
        "arm": "right",
        "camera_names": ["egoview", "robot0_agentview_center"],
        "render_camera": "egoview",
        "camera_widths": 320,
        "camera_heights": 240,
        "camera_fovy_overrides": {"egoview": 100},
        "name_hints": ["ball", "bowl", "container", "cup", "obj"],
        "visual_anchors": [
            {
                "name": "source_cup",
                "method": "sam3_rgbd",
                "camera_name": "egoview",
                "text_prompt": "cup",
                "arm": "right",
                "debug_subdir": "visual_anchor_source_cup",
                "required_for_control": False,
            },
            {
                "name": "target_container",
                "method": "sam3_rgbd",
                "camera_name": "egoview",
                "text_prompt": "bowl",
                "arm": "right",
                "debug_subdir": "visual_anchor_target_container",
                "required_for_control": False,
            },
        ],
        "remaining_oracle_dependencies": [
            "execute_gr1_skill(pnp_pouring) still resolves ball/cup true poses for contact metrics and posthoc evaluation",
            "source_cup/target_container control positions use visual anchors when available",
            "transport uses a fixed-horizon visual-target trajectory when distance feedback is disabled",
            "final pour still uses a fixed robot-joint heuristic until a visual/kinematic wrist policy is validated",
        ],
        "visual_control_offsets": {
            # SAM3/RGBD returns visible mask centroids. These offsets convert
            # the current egoview cup/bowl centroids into the control points
            # used by the validated PnPPouring grasp and pour primitive.
            "source_cup_grasp_pos": [0.0042, -0.0074, 0.0],
            "target_container_pos": [-0.0235, -0.0169, -0.0082],
        },
        "skill_overrides": {
            "hand_pose_family": "cylindrical",
            "approach_steps": 30,
            "close_steps": 22,
            "lift_steps": 18,
            "move_steps": 95,
            "pour_steps": 30,
            "cup_grasp_z_offset": -0.045,
            "target_z_offset": 0.47,
            "target_xy_offset": [0.08, 0.0],
            "use_object_pose_place": False,
            "use_transport_distance_feedback": False,
            "object_place_axis_angle_delta": [0.0, 0.40, 0.0],
            "pivot_forward": 0.06,
            "pivot_drop": 0.08,
            "pivot_lift": 0.33,
            "cup_grasp_profile": "cylindrical_lower",
            "use_final_pour_kinematic_policy": False,
            "final_pour_kinematic_steps": 75,
            "final_pour_kinematic_source_axis": [0.0, -1.0, 0.0],
            "final_pour_kinematic_down_bias": 0.35,
            "final_pour_kinematic_orientation_gain": 0.85,
            "final_pour_kinematic_max_axis_angle": 0.22,
            "final_pour_kinematic_target_z_offset": 0.24,
            "final_pour_joint_deltas": {
                "robot0_r_wrist_yaw": 1.57,
                "robot0_r_wrist_roll": 1.75,
                "robot0_r_shoulder_yaw": 0.25,
            },
            "final_pour_joint_steps": 75,
        },
    },
    "two_arm_lift": {
        "env_name": "TwoArmLift",
        "skill_type": "two_arm_lift",
        "arm": "both",
        "camera_names": ["frontview", "sideview"],
        "render_camera": "frontview",
        "camera_widths": 224,
        "camera_heights": 224,
        "name_hints": ["pot", "handle", "fixture"],
        "visual_anchors": [
            {
                "name": "pot_handle_front",
                "method": "sam3_rgbd",
                "camera_name": "frontview",
                "text_prompt": "pot handle",
                "arm": "right",
                "debug_subdir": "visual_anchor_pot_handle_front",
                "required_for_control": False,
            },
        ],
        "remaining_oracle_dependencies": [
            "TwoArmLift execution still falls back to MuJoCo pot handle geoms when visual anchors are diagnostic-only",
            "initialization_adapter intentionally moves pot_joint0 into the arms-only reach envelope",
        ],
        "initialization_adapter": {
            "free_joint": "pot_joint0",
            # Current arms-only GR1 controller cannot reach the default right
            # handle position; move the pot into the measured reach envelope.
            "pos_delta": [-0.16, 0.0, 0.0],
            "reason": "arms_only_gr1_reachability",
        },
        "skill_overrides": {
            "two_arm_plan_params": {
                "x_offset": 0.0,
                "lateral_offset": 0.04,
                "z_offset": -0.01,
                "approach_height": 0.07,
                "preset": "hook",
            },
            "approach_steps": 70,
            "grasp_steps": 80,
            "close_steps": 24,
            "lift_steps": 80,
            "scale": 0.045,
        },
    },
    "drawer": {
        "env_name": "TabletopOpenDrawerDoor",
        "skill_type": "drawer",
        "arm": "right",
        "camera_names": ["egoview", "robot0_frontview", "robot0_behindhead"],
        "render_camera": "egoview",
        "camera_widths": 224,
        "camera_heights": 224,
        "camera_fovy_overrides": {"egoview": 100},
        "name_hints": ["drawer", "handle", "fixture"],
        "visual_anchor": {
            "method": "sam3_rgbd",
            "name": "drawer_handle",
            "camera_name": "egoview",
            "text_prompt": "drawer handle",
            "crop_roi_xyxy": None,
            "debug_subdir": "visual_anchor_debug",
            "required_for_control": True,
        },
        "remaining_oracle_dependencies": [
            "privileged handle pose is used only for posthoc handle_before/handle_after evaluation",
        ],
        "skill_overrides": {
            "wrist_grasp_offset": [0.0, 0.30, -0.12],
            "wrist_pregrasp_offset": [0.0, 0.18, -0.03],
            "wrist_axis_angle_delta": [0.0, -0.45, 0.0],
            "wrist_orientation_frame": "world",
            "pregrasp_steps": 36,
            "approach_steps": 80,
            "close_steps": 20,
            "pull_steps": 80,
            "pregrasp_scale": 0.10,
            "approach_scale": 0.12,
            "pull_scale": 0.045,
        },
    },
}


def _parse_vec3(value: str, *, name: str) -> list[float]:
    parts = [part.strip() for part in str(value).split(",") if part.strip()]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(f"{name} must contain exactly 3 comma-separated values")
    try:
        return [float(part) for part in parts]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"{name} must contain numeric values") from exc


def _parse_optional_roi(value: str | None) -> list[float] | None:
    if value is None or str(value).strip() == "":
        return None
    parts = [part.strip() for part in str(value).split(",") if part.strip()]
    if len(parts) != 4:
        raise argparse.ArgumentTypeError("ROI must contain exactly 4 comma-separated values: x1,y1,x2,y2")
    try:
        return [float(part) for part in parts]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("ROI must contain numeric values") from exc


def _set_if_not_none(mapping: dict[str, Any], key: str, value: Any) -> None:
    if value is not None:
        mapping[key] = value


def _apply_initialization_adapter(env: Any, adapter: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(adapter, dict) or not adapter:
        return {"applied": False}
    low_level = getattr(env, "low_level_env", None)
    robosuite_env = getattr(low_level, "robosuite_env", None)
    sim = getattr(robosuite_env, "sim", None)
    joint_name = str(adapter.get("free_joint", "")).strip()
    if sim is None or not joint_name:
        return {"applied": False, "reason": "missing_sim_or_joint", "adapter": adapter}
    before = np.asarray(sim.data.get_joint_qpos(joint_name), dtype=np.float64).reshape(-1).copy()
    after = before.copy()
    pos_delta = np.asarray(adapter.get("pos_delta", [0.0, 0.0, 0.0]), dtype=np.float64).reshape(-1)
    if before.size < 3 or pos_delta.size < 3:
        return {"applied": False, "reason": "invalid_free_joint_or_delta", "adapter": adapter}
    after[:3] += pos_delta[:3]
    sim.data.set_joint_qpos(joint_name, after)
    sim.forward()
    if hasattr(robosuite_env, "_get_observations") and low_level is not None:
        low_level._current_obs = low_level._normalize_raw_obs(robosuite_env._get_observations())
    return {
        "applied": True,
        "free_joint": joint_name,
        "pos_delta": pos_delta[:3].tolist(),
        "before_qpos": before.tolist(),
        "after_qpos": after.tolist(),
        "reason": adapter.get("reason"),
    }


def _deprivileged_scene_sanitize_code(enabled: bool) -> str:
    if not enabled:
        return "SCENE = RAW_SCENE\n"
    return r'''
SCENE = dict(RAW_SCENE)
SCENE["scene_poses"] = {"bodies": {}, "sites": {}, "geoms": {}}
TASK_STATE = dict(SCENE.get("task_state", {}))
TASK_STATE["objects"] = {}
TASK_STATE["scene_poses"] = {"bodies": {}, "sites": {}, "geoms": {}}
SCENE["task_state"] = TASK_STATE
'''


def _visual_anchor_code(spec: dict[str, Any], out_dir: Path) -> str:
    anchors: list[dict[str, Any]] = []
    legacy_anchor = spec.get("visual_anchor") if isinstance(spec.get("visual_anchor"), dict) else None
    if legacy_anchor:
        anchors.append(legacy_anchor)
    extra_anchors = spec.get("visual_anchors") if isinstance(spec.get("visual_anchors"), list) else []
    anchors.extend(anchor for anchor in extra_anchors if isinstance(anchor, dict))
    if not bool(spec.get("visual_anchor_diagnostics", False)):
        anchors = [anchor for anchor in anchors if bool(anchor.get("required_for_control", False))]
    lines = [
        "VISUAL_ANCHORS = {}",
        "VISUAL_GRASP = None",
        "ANCHOR_DIAGNOSTICS = {'anchor_source': 'none', 'anchors': {}, 'privilege_mode': "
        f"{str(spec.get('privilege_mode', 'privileged_scene'))!r}}}",
    ]
    for index, anchor in enumerate(anchors):
        if anchor.get("method") != "sam3_rgbd":
            continue
        name = str(anchor.get("name") or f"anchor_{index}")
        debug_dir = out_dir / str(anchor.get("debug_subdir", f"visual_anchor_{name}"))
        crop_roi = anchor.get("crop_roi_xyxy")
        crop_expr = "None" if crop_roi is None else repr(tuple(float(v) for v in crop_roi))
        required = bool(anchor.get("required_for_control", False))
        lines.append(
            f'''
try:
    _anchor = estimate_sam3_rgbd_grasp_pose(
        camera_name={str(anchor.get("camera_name"))!r},
        text_prompt={str(anchor.get("text_prompt", name))!r},
        arm={str(anchor.get("arm", spec.get("arm", "right")))!r},
        approach_distance={float(anchor.get("approach_distance", 0.10))!r},
        crop_roi_xyxy={crop_expr},
        debug_output_dir={str(debug_dir)!r},
    )
    VISUAL_ANCHORS[{name!r}] = _anchor
    ANCHOR_DIAGNOSTICS["anchors"][{name!r}] = {{
        "ok": True,
        "method": _anchor.get("method"),
        "camera_name": _anchor.get("camera_name"),
        "text_prompt": {str(anchor.get("text_prompt", name))!r},
        "source": _anchor.get("source"),
        "grasp_pos": _anchor.get("grasp_pos"),
        "valid_depth_point_count": _anchor.get("valid_depth_point_count"),
        "mask_pixel_count": _anchor.get("mask_pixel_count"),
    }}
    if {required!r}:
        VISUAL_GRASP = _anchor
        ANCHOR_DIAGNOSTICS["anchor_source"] = {name!r}
except Exception as exc:
    ANCHOR_DIAGNOSTICS["anchors"][{name!r}] = {{
        "ok": False,
        "method": "sam3_rgbd",
        "camera_name": {str(anchor.get("camera_name"))!r},
        "text_prompt": {str(anchor.get("text_prompt", name))!r},
        "error": repr(exc),
    }}
    if {required!r}:
        raise
'''
        )
    return "\n".join(lines)


def _posthoc_eval_code(task_name: str) -> str:
    return f'''
def _posthoc_find(scene, *needles):
    lower_needles = [str(item).lower() for item in needles]
    for section in ("sites", "bodies", "geoms"):
        for name, entry in (scene.get("scene_poses", {{}}).get(section, {{}}) or {{}}).items():
            lower_name = str(name).lower()
            if all(needle in lower_name for needle in lower_needles):
                return {{"section": section, "name": name, "pos": entry.get("pos"), "quat": entry.get("quat")}}
    return None

def _posthoc_delta(before, after):
    if before is None or after is None:
        return None
    try:
        return np.asarray(after["pos"], dtype=float).reshape(3) - np.asarray(before["pos"], dtype=float).reshape(3)
    except Exception:
        return None

POSTHOC_EVAL = {{"task": {task_name!r}, "privilege_level": "posthoc_only_not_passed_to_selector_or_control"}}
if {task_name!r} == "drawer":
    POSTHOC_EVAL["drawer_handle_before"] = _posthoc_find(POSTHOC_SCENE_BEFORE, "drawer_tabletop", "door_handle")
    POSTHOC_EVAL["drawer_handle_after"] = _posthoc_find(POSTHOC_SCENE_AFTER, "drawer_tabletop", "door_handle")
    POSTHOC_EVAL["drawer_handle_delta"] = _posthoc_delta(POSTHOC_EVAL["drawer_handle_before"], POSTHOC_EVAL["drawer_handle_after"])
elif {task_name!r} == "pnp_pouring":
    POSTHOC_EVAL["source_cup_before"] = _posthoc_find(POSTHOC_SCENE_BEFORE, "obj_container")
    POSTHOC_EVAL["source_cup_after"] = _posthoc_find(POSTHOC_SCENE_AFTER, "obj_container")
    POSTHOC_EVAL["target_container_before"] = _posthoc_find(POSTHOC_SCENE_BEFORE, "container")
    POSTHOC_EVAL["ball_before"] = _posthoc_find(POSTHOC_SCENE_BEFORE, "ball_obj")
    POSTHOC_EVAL["ball_after"] = _posthoc_find(POSTHOC_SCENE_AFTER, "ball_obj")
elif {task_name!r} == "two_arm_lift":
    POSTHOC_EVAL["pot_before"] = _posthoc_find(POSTHOC_SCENE_BEFORE, "pot_root")
    POSTHOC_EVAL["pot_after"] = _posthoc_find(POSTHOC_SCENE_AFTER, "pot_root")
    POSTHOC_EVAL["pot_delta"] = _posthoc_delta(POSTHOC_EVAL["pot_before"], POSTHOC_EVAL["pot_after"])
'''


def _build_task_specs(args: argparse.Namespace) -> dict[str, dict[str, Any]]:
    specs = copy.deepcopy(TASK_SPECS)
    for spec in specs.values():
        spec["deprivileged_observation"] = bool(args.deprivileged_observation)
        spec["visual_anchor_diagnostics"] = bool(args.visual_anchor_diagnostics or args.deprivileged_observation)
        spec["privilege_mode"] = (
            "deprivileged_observation_with_visual_anchor_diagnostics"
            if bool(args.deprivileged_observation)
            else "privileged_scene_observation"
        )
    drawer = specs["drawer"]
    drawer_anchor = drawer.setdefault("visual_anchor", {})
    drawer_overrides = drawer.setdefault("skill_overrides", {})
    if args.drawer_anchor_camera:
        drawer_anchor["camera_name"] = args.drawer_anchor_camera
    if args.drawer_anchor_prompt:
        drawer_anchor["text_prompt"] = args.drawer_anchor_prompt
    if args.drawer_anchor_roi is not None:
        drawer_anchor["crop_roi_xyxy"] = args.drawer_anchor_roi
    if args.drawer_render_camera:
        drawer["render_camera"] = args.drawer_render_camera
    if args.drawer_wrist_grasp_offset is not None:
        drawer_overrides["wrist_grasp_offset"] = args.drawer_wrist_grasp_offset
    if args.drawer_wrist_pregrasp_offset is not None:
        drawer_overrides["wrist_pregrasp_offset"] = args.drawer_wrist_pregrasp_offset
    if args.drawer_wrist_axis_angle_delta is not None:
        drawer_overrides["wrist_axis_angle_delta"] = args.drawer_wrist_axis_angle_delta
    for arg_name in (
        "drawer_pregrasp_steps",
        "drawer_approach_steps",
        "drawer_close_steps",
        "drawer_pull_steps",
        "drawer_pregrasp_scale",
        "drawer_approach_scale",
        "drawer_pull_scale",
    ):
        value = getattr(args, arg_name)
        if value is not None:
            drawer_overrides[arg_name.removeprefix("drawer_")] = value
    two_arm = specs["two_arm_lift"]
    two_arm_overrides = two_arm.setdefault("skill_overrides", {})
    two_arm_plan = two_arm_overrides.setdefault("two_arm_plan_params", {})
    _set_if_not_none(two_arm_plan, "x_offset", args.two_arm_x_offset)
    _set_if_not_none(two_arm_plan, "lateral_offset", args.two_arm_lateral_offset)
    _set_if_not_none(two_arm_plan, "z_offset", args.two_arm_z_offset)
    _set_if_not_none(two_arm_plan, "approach_height", args.two_arm_approach_height)
    _set_if_not_none(two_arm_plan, "preset", args.two_arm_preset)
    if args.two_arm_scene_shift_x is not None:
        two_arm["initialization_adapter"] = {
            "free_joint": "pot_joint0",
            "pos_delta": [float(args.two_arm_scene_shift_x), 0.0, 0.0],
            "reason": "cli_override",
        }
    _set_if_not_none(two_arm_overrides, "hand_pose_family", args.two_arm_preset)
    if args.two_arm_wrist_axis_angle_delta is not None:
        two_arm_overrides["wrist_axis_angle_delta"] = args.two_arm_wrist_axis_angle_delta
    _set_if_not_none(two_arm_overrides, "approach_steps", args.two_arm_approach_steps)
    _set_if_not_none(two_arm_overrides, "grasp_steps", args.two_arm_grasp_steps)
    _set_if_not_none(two_arm_overrides, "close_steps", args.two_arm_close_steps)
    _set_if_not_none(two_arm_overrides, "lift_steps", args.two_arm_lift_steps)
    _set_if_not_none(two_arm_overrides, "scale", args.two_arm_scale)
    if args.task_spec_override_json is not None:
        override_path = Path(args.task_spec_override_json)
        overrides = json.loads(override_path.read_text(encoding="utf-8"))
        if not isinstance(overrides, dict):
            raise ValueError(f"Task spec override must be a JSON object: {override_path}")
        for task_name, task_override in overrides.items():
            if task_name not in specs:
                raise ValueError(f"Unknown task in override file {override_path}: {task_name!r}")
            if not isinstance(task_override, dict):
                raise ValueError(f"Override for task {task_name!r} must be a JSON object")
            _deep_update(specs[task_name], task_override)
    return specs


def _deep_update(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = value
    return base


def _jsonify(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonify(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonify(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    try:
        json.dumps(value)
        return value
    except TypeError:
        return repr(value)


def _write_video(path: Path, frames: list[np.ndarray], fps: int) -> None:
    if not frames:
        return
    import imageio.v2 as imageio

    path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(path, frames, fps=fps)


def _build_env_config(spec: dict[str, Any], *, seed: int, max_steps: int) -> dict[str, Any]:
    low_level = {
        "_target_": "capx.envs.simulators.robocasa_gr1.GR1RobocasaLowLevel",
        "env_name": spec["env_name"],
        "robots": "GR1ArmsOnlyFourierHands",
        "controller_config": "WHOLE_BODY_MINK_IK",
        "camera_names": list(spec["camera_names"]),
        "render_camera": spec["render_camera"],
        "camera_widths": int(spec["camera_widths"]),
        "camera_heights": int(spec["camera_heights"]),
        "max_steps": int(max_steps),
        "seed": int(seed),
        "record_transition_fps": 20.0,
        "record_transition_depth": False,
    }
    if spec.get("camera_fovy_overrides"):
        low_level["camera_fovy_overrides"] = dict(spec["camera_fovy_overrides"])
    return {
        "_target_": "capx.envs.tasks.gr1.robocasa.GR1RobocasaCodeEnv",
        "cfg": {
            "_target_": "capx.envs.tasks.base.CodeExecEnvConfig",
            "low_level": low_level,
            "privileged": False,
            "apis": ["GR1RobocasaControlApi"],
        },
    }


def _run_task(
    *,
    task_name: str,
    spec: dict[str, Any],
    out_dir: Path,
    seed: int,
    fps: int,
    max_steps: int,
    smoke: bool,
    save_videos: bool,
) -> dict[str, Any]:
    env = instantiate(_build_env_config(spec, seed=seed, max_steps=max_steps))
    try:
        env.reset(seed=seed)
        low_level = env.low_level_env
        initialization_diagnostics = _apply_initialization_adapter(
            env,
            spec.get("initialization_adapter") if isinstance(spec.get("initialization_adapter"), dict) else None,
        )
        if hasattr(low_level, "enable_video_capture"):
            low_level.enable_video_capture(save_videos, clear=True, wrist_camera=True)

        visual_code = _visual_anchor_code(spec, out_dir)
        scene_sanitize_code = _deprivileged_scene_sanitize_code(bool(spec.get("deprivileged_observation", False)))
        skill_override_lines = []
        overrides = spec.get("skill_overrides") if isinstance(spec.get("skill_overrides"), dict) else {}
        for key, value in overrides.items():
            if key == "wrist_grasp_offset":
                skill_override_lines.append(
                    f"SKILL['wrist_grasp_pos'] = np.asarray(SKILL['handle_pos'], dtype=float) + np.array({value!r}, dtype=float)"
                )
            elif key == "wrist_pregrasp_offset":
                skill_override_lines.append(
                    f"SKILL['wrist_pregrasp_pos'] = np.asarray(SKILL['handle_pos'], dtype=float) + np.array({value!r}, dtype=float)"
                )
            elif key == "wrist_axis_angle_delta":
                skill_override_lines.append(f"SKILL[{key!r}] = np.array({value!r}, dtype=float)")
            elif key == "two_arm_plan_params":
                skill_override_lines.append(
                    "SKILL['plan'] = plan_two_arm_lift_handles(scene=SCENE.get('scene_poses', SCENE), **"
                    f"{value!r})"
                )
                skill_override_lines.append(f"SKILL['selection_diagnostics']['adapter_plan_params'] = {value!r}")
            else:
                skill_override_lines.append(f"SKILL[{key!r}] = {value!r}")
        skill_override_code = "\n".join(skill_override_lines)

        code = f"""
import numpy as np
POSTHOC_SCENE_BEFORE = observe_scene(name_hints={list(spec["name_hints"])!r}, include_geoms=True)
RAW_SCENE = observe_scene(name_hints={list(spec["name_hints"])!r}, include_geoms={not bool(spec.get("deprivileged_observation", False))!r})
{scene_sanitize_code}
CALIBRATION = probe_eef_directions(
    arm={spec["arm"]!r},
    step=0.025,
    axes=("+x", "-x", "+y", "-y", "+z", "-z"),
    steps_per_axis=3,
)
{visual_code}
SKILL = select_gr1_skill(
    task_type={spec["skill_type"]!r},
    arm={spec["arm"]!r},
    scene=SCENE,
    calibration=CALIBRATION,
    visual_grasp=VISUAL_GRASP,
)
SKILL["visual_anchors"] = VISUAL_ANCHORS
if {task_name!r} == "pnp_pouring":
    _visual_control_offsets = {dict(spec.get("visual_control_offsets", {}))!r}
    _source_cup_anchor = VISUAL_ANCHORS.get("source_cup") if isinstance(VISUAL_ANCHORS.get("source_cup"), dict) else None
    _target_container_anchor = (
        VISUAL_ANCHORS.get("target_container")
        if isinstance(VISUAL_ANCHORS.get("target_container"), dict)
        else None
    )
    if _source_cup_anchor is not None and _source_cup_anchor.get("grasp_pos") is not None:
        _source_cup_offset = np.asarray(
            _visual_control_offsets.get("source_cup_grasp_pos", [0.0, 0.0, 0.0]),
            dtype=np.float64,
        ).reshape(3)
        SKILL["source_cup_anchor_raw_pos"] = _source_cup_anchor.get("grasp_pos")
        SKILL["source_cup_anchor_offset"] = _source_cup_offset
        SKILL["source_cup_grasp_pos"] = np.asarray(_source_cup_anchor.get("grasp_pos"), dtype=np.float64).reshape(3) + _source_cup_offset
    if _target_container_anchor is not None:
        _target_container_pos = _target_container_anchor.get("grasp_pos", _target_container_anchor.get("centroid"))
        if _target_container_pos is not None:
            _target_container_offset = np.asarray(
                _visual_control_offsets.get("target_container_pos", [0.0, 0.0, 0.0]),
                dtype=np.float64,
            ).reshape(3)
            SKILL["target_container_anchor_raw_pos"] = _target_container_pos
            SKILL["target_container_anchor_offset"] = _target_container_offset
            SKILL["target_container_pos"] = np.asarray(_target_container_pos, dtype=np.float64).reshape(3) + _target_container_offset
SKILL["privilege_audit"] = {{
    "privilege_mode": {str(spec.get("privilege_mode", "privileged_scene_observation"))!r},
    "agent_scene_contains_sim_scene_poses": {not bool(spec.get("deprivileged_observation", False))!r},
    "visual_anchor_names": list(VISUAL_ANCHORS.keys()),
    "known_remaining_oracle_dependencies": {list(spec.get("remaining_oracle_dependencies", []))!r},
}}
{skill_override_code}
RESULT = execute_gr1_skill(SKILL, smoke={bool(smoke)!r})
VERIFY = verify_success_or_contact(RESULT)
POSTHOC_SCENE_AFTER = observe_scene(name_hints={list(spec["name_hints"])!r}, include_geoms=True)
{_posthoc_eval_code(task_name)}
print({{"task": {task_name!r}, "skill_type": SKILL["skill_type"], "verify": VERIFY}})
"""
        _obs, reward, terminated, truncated, info = env.step(code)
        transition_dataset = env.get_transition_dataset() if hasattr(env, "get_transition_dataset") else None
        if transition_dataset is not None:
            transition_dataset["trial"] = 1
            transition_dataset["attempt"] = 1
            transition_dataset["diagnostic_task"] = task_name
            transition_dataset["diagnostic_skill_spec"] = spec
            transition_dataset["diagnostic_initialization_adapter"] = initialization_diagnostics
            save_transition_dataset(out_dir, transition_dataset)

        frames = low_level.get_video_frames() if hasattr(low_level, "get_video_frames") else []
        wrist_frames = low_level.get_wrist_video_frames() if hasattr(low_level, "get_wrist_video_frames") else []
        if save_videos:
            _write_video(out_dir / f"{task_name}_frontview.mp4", frames, fps=fps)
            _write_video(out_dir / f"{task_name}_wrist.mp4", wrist_frames, fps=fps)

        namespace = getattr(env, "_exec_globals", {})
        summary = {
            "task": task_name,
            "env_name": spec["env_name"],
            "ok": int(info.get("sandbox_rc", 1)) == 0,
            "reward": float(reward),
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "task_completed": bool(info.get("task_completed", False)),
            "stdout": info.get("stdout", ""),
            "stderr": info.get("stderr", ""),
            "scene": _jsonify(namespace.get("SCENE")),
            "calibration": _jsonify(namespace.get("CALIBRATION")),
            "anchor_diagnostics": _jsonify(namespace.get("ANCHOR_DIAGNOSTICS")),
            "initialization_diagnostics": _jsonify(initialization_diagnostics),
            "skill": _jsonify(namespace.get("SKILL")),
            "result": _jsonify(namespace.get("RESULT")),
            "verify": _jsonify(namespace.get("VERIFY")),
            "posthoc_eval": _jsonify(namespace.get("POSTHOC_EVAL")),
            "frame_count": len(frames),
            "wrist_frame_count": len(wrist_frames),
            "transition_count": (
                len(transition_dataset.get("transitions", [])) if transition_dataset is not None else 0
            ),
        }
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        return summary
    finally:
        if hasattr(env, "close"):
            env.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=["pnp_pouring", "two_arm_lift", "drawer"],
        choices=sorted(TASK_SPECS),
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--full", action="store_true", help="Run longer skill motions instead of short smoke motions.")
    parser.add_argument("--save-videos", action="store_true")
    parser.add_argument(
        "--deprivileged-observation",
        action="store_true",
        help="Remove sim object/site/geom poses from the scene object passed to skill selection.",
    )
    parser.add_argument(
        "--visual-anchor-diagnostics",
        action="store_true",
        help="Estimate and record configured RGBD/SAM3 anchors even when they are not required for control.",
    )
    parser.add_argument("--drawer-anchor-camera", default=None)
    parser.add_argument("--drawer-anchor-prompt", default=None)
    parser.add_argument("--drawer-anchor-roi", type=_parse_optional_roi, default=None)
    parser.add_argument("--drawer-render-camera", default=None)
    parser.add_argument(
        "--drawer-wrist-grasp-offset",
        type=lambda value: _parse_vec3(value, name="--drawer-wrist-grasp-offset"),
        default=None,
        help="Drawer wrist grasp offset from selected handle/visual anchor, as dx,dy,dz.",
    )
    parser.add_argument(
        "--drawer-wrist-pregrasp-offset",
        type=lambda value: _parse_vec3(value, name="--drawer-wrist-pregrasp-offset"),
        default=None,
        help="Drawer wrist pregrasp offset from selected handle/visual anchor, as dx,dy,dz.",
    )
    parser.add_argument(
        "--drawer-wrist-axis-angle-delta",
        type=lambda value: _parse_vec3(value, name="--drawer-wrist-axis-angle-delta"),
        default=None,
        help="Drawer wrist world/local orientation delta, as rx,ry,rz.",
    )
    parser.add_argument("--drawer-pregrasp-steps", type=int, default=None)
    parser.add_argument("--drawer-approach-steps", type=int, default=None)
    parser.add_argument("--drawer-close-steps", type=int, default=None)
    parser.add_argument("--drawer-pull-steps", type=int, default=None)
    parser.add_argument("--drawer-pregrasp-scale", type=float, default=None)
    parser.add_argument("--drawer-approach-scale", type=float, default=None)
    parser.add_argument("--drawer-pull-scale", type=float, default=None)
    parser.add_argument("--two-arm-x-offset", type=float, default=None)
    parser.add_argument("--two-arm-lateral-offset", type=float, default=None)
    parser.add_argument("--two-arm-z-offset", type=float, default=None)
    parser.add_argument("--two-arm-approach-height", type=float, default=None)
    parser.add_argument("--two-arm-scene-shift-x", type=float, default=None)
    parser.add_argument("--two-arm-preset", default=None)
    parser.add_argument("--two-arm-wrist-axis-angle-delta", type=lambda value: _parse_vec3(value, name="--two-arm-wrist-axis-angle-delta"), default=None)
    parser.add_argument("--two-arm-approach-steps", type=int, default=None)
    parser.add_argument("--two-arm-grasp-steps", type=int, default=None)
    parser.add_argument("--two-arm-close-steps", type=int, default=None)
    parser.add_argument("--two-arm-lift-steps", type=int, default=None)
    parser.add_argument("--two-arm-scale", type=float, default=None)
    parser.add_argument(
        "--task-spec-override-json",
        default=None,
        help="JSON object of task-specific spec overrides for generated adapter search candidates.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.output_dir)
    root.mkdir(parents=True, exist_ok=True)
    task_specs = _build_task_specs(args)

    summaries = []
    for task_name in args.tasks:
        task_dir = root / task_name
        summary = _run_task(
            task_name=task_name,
            spec=task_specs[task_name],
            out_dir=task_dir,
            seed=args.seed,
            fps=args.fps,
            max_steps=args.max_steps,
            smoke=not args.full,
            save_videos=args.save_videos,
        )
        summaries.append(summary)
        print(json.dumps({"task": task_name, "ok": summary["ok"], "transition_count": summary["transition_count"]}, sort_keys=True))

    aggregate = {
        "seed": int(args.seed),
        "smoke": not bool(args.full),
        "tasks": [summary["task"] for summary in summaries],
        "all_ok": all(bool(summary["ok"]) for summary in summaries),
        "summaries": _jsonify(summaries),
    }
    (root / "skill_interface_summary.json").write_text(json.dumps(aggregate, indent=2), encoding="utf-8")
    if not aggregate["all_ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
