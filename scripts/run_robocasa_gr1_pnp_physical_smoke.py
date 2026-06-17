#!/usr/bin/env python3
"""Run a deterministic RoboCasa GR1 PnPPouring physical-control smoke.

This bypasses code generation and directly calls the high-level GR1 harness.
It is meant to validate grasp/place control changes before spending a model
trial.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from capx.envs.configs.instantiate import instantiate
from capx.envs.configs.loader import DictLoader
from capx.envs.transition_dataset import save_transition_dataset


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="env_configs/robocasa/gr1_robocasa_pnp_pouring_mink_smoke.yaml",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument(
        "--sweep-preset",
        choices=[
            "single",
            "contact_v1",
            "cup_grasp_v1",
            "cup_tilt_v1",
            "cup_geom_tilt_v1",
            "cup_pivot_v1",
            "cup_object_pose_v1",
            "cup_axis_retreat_v1",
            "cup_lower_body_compare_v1",
            "cup_cylindrical_tune_v1",
            "cup_final_rotate_v1",
            "cup_joint_posture_rotate_v1",
            "cup_joint_posture_high_v1",
            "cup_joint_posture_shoulder_high_v1",
            "cup_case7_plus025_v1",
            "cup_case7_high_wrist_pitch_v1",
            "cup_case7_yaw_combo_v1",
            "cup_case7_yaw_widecam_debug_v1",
            "cup_case7_yaw_roll_refine_v1",
            "cup_case7_yaw_roll_refine2_v1",
            "cup_case7_default_v1",
        ],
        default="cup_case7_default_v1",
        help="Run a deterministic parameter sweep instead of the default single baseline.",
    )
    parser.add_argument("--render-camera", default=None, help="Override the primary render camera for video/debug runs.")
    parser.add_argument("--camera-names", nargs="+", default=None, help="Override low-level camera_names.")
    parser.add_argument("--camera-width", type=int, default=None, help="Override all low-level camera widths.")
    parser.add_argument("--camera-height", type=int, default=None, help="Override all low-level camera heights.")
    parser.add_argument(
        "--camera-fovy",
        action="append",
        default=[],
        metavar="NAME=DEGREES",
        help="Override MuJoCo camera fovy for a named camera. Can be passed multiple times.",
    )
    parser.add_argument(
        "--save-case-videos",
        action="store_true",
        help="Save per-case videos during sweeps. Disabled by default to keep diagnostics lightweight.",
    )
    return parser.parse_args()


def _parse_camera_fovy_overrides(values: list[str]) -> dict[str, float]:
    overrides: dict[str, float] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"--camera-fovy must use NAME=DEGREES, got {value!r}")
        name, degrees = value.split("=", 1)
        overrides[name.strip()] = float(degrees)
    return overrides


def _apply_camera_overrides(config: dict[str, Any], args: argparse.Namespace) -> None:
    low_level = config["env"]["cfg"]["low_level"]
    if args.camera_names is not None:
        low_level["camera_names"] = list(args.camera_names)
    if args.render_camera is not None:
        low_level["render_camera"] = str(args.render_camera)
    if args.camera_width is not None:
        low_level["camera_widths"] = int(args.camera_width)
    if args.camera_height is not None:
        low_level["camera_heights"] = int(args.camera_height)
    fovy_overrides = _parse_camera_fovy_overrides(args.camera_fovy)
    if fovy_overrides:
        low_level["camera_fovy_overrides"] = fovy_overrides


def _run_case(
    *,
    config: dict[str, Any],
    config_path: str,
    out_dir: Path,
    seed: int,
    fps: int,
    params: dict[str, Any],
    save_video: bool,
) -> dict[str, Any]:
    env = instantiate(config["env"])
    try:
        env.reset(seed=seed)
        low_level = env.low_level_env
        if hasattr(low_level, "enable_video_capture"):
            low_level.enable_video_capture(save_video, clear=True, wrist_camera=True)

        if str(params.get("mode", "ball")) == "cup":
            code = f"""
state = get_task_state()
print("initial_task_completed", state["task_completed"])
RESULT = solve_pnp_pouring_cup_physical(
    cup_name="obj_container",
    ball_name="ball_obj",
    target_name="container",
    arm={params["arm"]!r},
    preset={params["preset"]!r},
    approach_steps={int(params["approach_steps"])},
    close_steps={int(params["close_steps"])},
    lift_steps={int(params["lift_steps"])},
    move_steps={int(params["move_steps"])},
    pour_steps={int(params["pour_steps"])},
    release_steps={int(params["release_steps"])},
    scale={float(params["scale"])!r},
    cup_grasp_z_offset={float(params["cup_grasp_z_offset"])!r},
    target_z_offset={float(params["target_z_offset"])!r},
    target_xy_offset={tuple(params["target_xy_offset"])!r},
    pour_axis_angle_delta={tuple(params["pour_axis_angle_delta"])!r},
    use_geometry_pour={bool(params["use_geometry_pour"])!r},
    pour_down_bias={float(params["pour_down_bias"])!r},
    pour_orientation_gain={float(params["pour_orientation_gain"])!r},
    use_pivot_pour={bool(params["use_pivot_pour"])!r},
    pivot_forward={float(params["pivot_forward"])!r},
    pivot_drop={float(params["pivot_drop"])!r},
    pivot_lift={float(params["pivot_lift"])!r},
    use_object_pose_place={bool(params["use_object_pose_place"])!r},
    object_place_axis_angle_delta={tuple(params["object_place_axis_angle_delta"])!r},
    cup_grasp_profile={params["cup_grasp_profile"]!r},
    cup_grasp_side_offset={float(params["cup_grasp_side_offset"])!r},
    cup_grasp_z_floor={float(params["cup_grasp_z_floor"])!r},
    final_pour_axis_angle_delta={tuple(params["final_pour_axis_angle_delta"])!r},
    final_pour_steps={int(params["final_pour_steps"])},
    final_pour_joint_deltas={dict(params["final_pour_joint_deltas"])!r},
    final_pour_joint_steps={int(params["final_pour_joint_steps"])},
)
print("phase", RESULT["phase"])
print("ball_distance_to_target", RESULT["ball_distance_to_target"])
print("cup_distance_to_target", RESULT["cup_distance_to_target"])
print("task_completed", RESULT["task_completed"])
print("reward", RESULT["reward"])
print("cup_grasp_stage", RESULT.get("cup_grasp_stage"))
"""
        else:
            code = f"""
state = get_task_state()
print("initial_task_completed", state["task_completed"])
RESULT = solve_pnp_pouring_physical(
    object_name="ball_obj",
    target_name="container",
    arm={params["arm"]!r},
    preset={params["preset"]!r},
    approach_steps={int(params["approach_steps"])},
    close_steps={int(params["close_steps"])},
    lift_steps={int(params["lift_steps"])},
    move_steps={int(params["move_steps"])},
    release_steps={int(params["release_steps"])},
    scale={float(params["scale"])!r},
    target_z_offset={float(params["target_z_offset"])!r},
)
print("distance_to_target", RESULT["distance_to_target"])
print("task_completed", RESULT["task_completed"])
print("reward", RESULT["reward"])
print("place_summary", RESULT.get("place_summary"))
"""
        _obs, reward, terminated, truncated, info = env.step(code)
        result = env._exec_globals.get("RESULT")  # diagnostic script; use the persistent namespace intentionally.
        transition_dataset = env.get_transition_dataset() if hasattr(env, "get_transition_dataset") else None
        if transition_dataset is not None:
            transition_dataset["trial"] = 1
            transition_dataset["attempt"] = 1
            transition_dataset["config_path"] = config_path
            transition_dataset["diagnostic_params"] = params
            save_transition_dataset(out_dir, transition_dataset)

        frames = low_level.get_video_frames() if hasattr(low_level, "get_video_frames") else []
        wrist_frames = low_level.get_wrist_video_frames() if hasattr(low_level, "get_wrist_video_frames") else []
        if save_video:
            _write_video(out_dir / "pnp_physical_frontview.mp4", frames, fps=fps)
            _write_video(out_dir / "pnp_physical_wrist.mp4", wrist_frames, fps=fps)

        summary = {
            "params": _jsonify(params),
            "ok": int(info.get("sandbox_rc", 1)) == 0,
            "reward": float(reward),
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "task_completed": bool(info.get("task_completed", False)),
            "stdout": info.get("stdout", ""),
            "stderr": info.get("stderr", ""),
            "result": _jsonify(result),
            "frame_count": len(frames),
            "wrist_frame_count": len(wrist_frames),
            "transition_count": (
                len(transition_dataset.get("transitions", [])) if transition_dataset is not None else 0
            ),
        }
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        return summary
    finally:
        if hasattr(env, "close"):
            env.close()


def _sweep_cases(preset: str) -> list[dict[str, Any]]:
    base = {
        "mode": "ball",
        "arm": "right",
        "preset": "power",
        "approach_steps": 30,
        "close_steps": 22,
        "lift_steps": 18,
        "move_steps": 90,
        "pour_steps": 18,
        "release_steps": 12,
        "scale": 0.07,
        "cup_grasp_z_offset": -0.055,
        "target_z_offset": 0.12,
        "target_xy_offset": (0.0, 0.0),
        "pour_axis_angle_delta": (0.0, 0.0, 0.0),
        "use_geometry_pour": False,
        "pour_down_bias": 0.35,
        "pour_orientation_gain": 1.0,
        "use_pivot_pour": False,
        "pivot_forward": 0.06,
        "pivot_drop": 0.08,
        "pivot_lift": 0.04,
        "use_object_pose_place": False,
        "object_place_axis_angle_delta": (0.0, 0.0, 0.0),
        "cup_grasp_profile": "center",
        "cup_grasp_side_offset": 0.035,
        "cup_grasp_z_floor": -0.075,
        "final_pour_axis_angle_delta": (0.0, 0.0, 0.0),
        "final_pour_steps": 0,
        "final_pour_joint_deltas": {},
        "final_pour_joint_steps": 0,
    }
    if preset == "single":
        return [base]
    if preset == "contact_v1":
        variants = [
            {"scale": 0.055, "target_z_offset": 0.10, "move_steps": 90, "release_steps": 0},
            {"scale": 0.055, "target_z_offset": 0.12, "move_steps": 90, "release_steps": 0},
            {"scale": 0.055, "target_z_offset": 0.14, "move_steps": 100, "release_steps": 0},
            {"scale": 0.065, "target_z_offset": 0.10, "move_steps": 90, "release_steps": 0},
            {"scale": 0.065, "target_z_offset": 0.12, "move_steps": 100, "release_steps": 0},
            {"scale": 0.065, "target_z_offset": 0.16, "move_steps": 110, "release_steps": 0},
            {"scale": 0.075, "target_z_offset": 0.12, "move_steps": 100, "release_steps": 0},
            {"scale": 0.075, "target_z_offset": 0.16, "move_steps": 120, "release_steps": 0},
        ]
        return [{**base, **variant} for variant in variants]
    if preset == "cup_grasp_v1":
        cup_base = {
            **base,
            "mode": "cup",
            "release_steps": 0,
            "move_steps": 95,
            "pour_steps": 18,
            "scale": 0.055,
            "target_z_offset": 0.18,
        }
        variants = [
            {"preset": "support", "cup_grasp_z_offset": -0.045, "target_xy_offset": (0.04, 0.00)},
            {"preset": "support", "cup_grasp_z_offset": -0.065, "target_xy_offset": (0.06, 0.00)},
            {"preset": "power", "cup_grasp_z_offset": -0.045, "target_xy_offset": (0.04, 0.00)},
            {"preset": "power", "cup_grasp_z_offset": -0.065, "target_xy_offset": (0.06, 0.00)},
            {"preset": "pinch", "cup_grasp_z_offset": -0.045, "target_xy_offset": (0.04, 0.00)},
            {"preset": "pinch", "cup_grasp_z_offset": -0.065, "target_xy_offset": (0.06, 0.00)},
        ]
        return [{**cup_base, **variant} for variant in variants]
    if preset == "cup_tilt_v1":
        cup_base = {
            **base,
            "mode": "cup",
            "preset": "power",
            "release_steps": 0,
            "move_steps": 95,
            "pour_steps": 24,
            "scale": 0.055,
            "cup_grasp_z_offset": -0.045,
            "target_z_offset": 0.18,
            "target_xy_offset": (0.04, 0.0),
        }
        variants = [
            {"pour_axis_angle_delta": (0.35, 0.0, 0.0)},
            {"pour_axis_angle_delta": (-0.35, 0.0, 0.0)},
            {"pour_axis_angle_delta": (0.0, 0.35, 0.0)},
            {"pour_axis_angle_delta": (0.0, -0.35, 0.0)},
            {"pour_axis_angle_delta": (0.0, 0.0, 0.35)},
            {"pour_axis_angle_delta": (0.0, 0.0, -0.35)},
            {"pour_axis_angle_delta": (0.55, 0.0, 0.0)},
            {"pour_axis_angle_delta": (0.0, -0.55, 0.0)},
        ]
        return [{**cup_base, **variant} for variant in variants]
    if preset == "cup_geom_tilt_v1":
        cup_base = {
            **base,
            "mode": "cup",
            "preset": "power",
            "release_steps": 0,
            "move_steps": 95,
            "pour_steps": 24,
            "scale": 0.055,
            "cup_grasp_z_offset": -0.045,
            "target_z_offset": 0.18,
            "target_xy_offset": (0.04, 0.0),
            "pour_axis_angle_delta": (0.0, 0.0, 0.0),
            "use_geometry_pour": True,
        }
        variants = [
            {"pour_down_bias": 0.20, "pour_orientation_gain": 0.6},
            {"pour_down_bias": 0.35, "pour_orientation_gain": 0.6},
            {"pour_down_bias": 0.50, "pour_orientation_gain": 0.6},
            {"pour_down_bias": 0.20, "pour_orientation_gain": 1.0},
            {"pour_down_bias": 0.35, "pour_orientation_gain": 1.0},
            {"pour_down_bias": 0.50, "pour_orientation_gain": 1.0},
            {"pour_down_bias": 0.35, "pour_orientation_gain": 1.4},
            {"pour_down_bias": 0.50, "pour_orientation_gain": 1.4},
        ]
        return [{**cup_base, **variant} for variant in variants]
    if preset == "cup_pivot_v1":
        cup_base = {
            **base,
            "mode": "cup",
            "preset": "power",
            "release_steps": 0,
            "move_steps": 95,
            "pour_steps": 30,
            "scale": 0.055,
            "cup_grasp_z_offset": -0.045,
            "target_z_offset": 0.18,
            "target_xy_offset": (0.04, 0.0),
            "pour_axis_angle_delta": (0.0, -0.55, 0.0),
            "use_geometry_pour": False,
            "use_pivot_pour": True,
        }
        variants = [
            {"pivot_forward": 0.03, "pivot_drop": 0.04, "pivot_lift": 0.02},
            {"pivot_forward": 0.05, "pivot_drop": 0.06, "pivot_lift": 0.02},
            {"pivot_forward": 0.07, "pivot_drop": 0.08, "pivot_lift": 0.02},
            {"pivot_forward": 0.03, "pivot_drop": 0.08, "pivot_lift": 0.05},
            {"pivot_forward": 0.05, "pivot_drop": 0.10, "pivot_lift": 0.05},
            {"pivot_forward": 0.07, "pivot_drop": 0.12, "pivot_lift": 0.05},
            {"pivot_forward": -0.03, "pivot_drop": 0.08, "pivot_lift": 0.05},
            {"pivot_forward": -0.05, "pivot_drop": 0.10, "pivot_lift": 0.05},
        ]
        return [{**cup_base, **variant} for variant in variants]
    if preset == "cup_object_pose_v1":
        cup_base = {
            **base,
            "mode": "cup",
            "preset": "power",
            "release_steps": 0,
            "move_steps": 95,
            "pour_steps": 30,
            "scale": 0.055,
            "cup_grasp_z_offset": -0.045,
            "target_z_offset": 0.18,
            "target_xy_offset": (0.04, 0.0),
            "pour_axis_angle_delta": (0.0, 0.0, 0.0),
            "use_geometry_pour": False,
            "use_pivot_pour": True,
            "use_object_pose_place": True,
            "pivot_forward": 0.07,
            "pivot_drop": 0.08,
            "pivot_lift": 0.02,
        }
        variants = [
            {"object_place_axis_angle_delta": (0.0, -0.35, 0.0)},
            {"object_place_axis_angle_delta": (0.0, -0.55, 0.0)},
            {"object_place_axis_angle_delta": (0.0, -0.75, 0.0)},
            {"object_place_axis_angle_delta": (0.20, -0.55, 0.0)},
            {"object_place_axis_angle_delta": (-0.20, -0.55, 0.0)},
            {"target_xy_offset": (0.06, 0.0), "object_place_axis_angle_delta": (0.0, -0.55, 0.0)},
            {"target_xy_offset": (0.02, 0.0), "object_place_axis_angle_delta": (0.0, -0.55, 0.0)},
            {"pivot_forward": 0.09, "pivot_drop": 0.08, "object_place_axis_angle_delta": (0.0, -0.55, 0.0)},
        ]
        return [{**cup_base, **variant} for variant in variants]
    if preset == "cup_axis_retreat_v1":
        cup_base = {
            **base,
            "mode": "cup",
            "preset": "power",
            "release_steps": 0,
            "move_steps": 95,
            "pour_steps": 30,
            "scale": 0.055,
            "cup_grasp_z_offset": -0.045,
            "target_z_offset": 0.18,
            "target_xy_offset": (0.04, 0.0),
            "pour_axis_angle_delta": (0.0, 0.0, 0.0),
            "use_geometry_pour": False,
            "use_pivot_pour": True,
            "use_object_pose_place": True,
            "object_place_axis_angle_delta": (0.0, 0.20, 0.0),
            "pivot_drop": 0.08,
            "pivot_lift": 0.02,
        }
        variants = [
            {"pivot_forward": 0.04, "target_xy_offset": (0.04, 0.0)},
            {"pivot_forward": 0.05, "target_xy_offset": (0.04, 0.0)},
            {"pivot_forward": 0.06, "target_xy_offset": (0.04, 0.0)},
            {"pivot_forward": 0.07, "target_xy_offset": (0.04, 0.0)},
            {"pivot_forward": 0.05, "target_xy_offset": (0.06, 0.0)},
            {"pivot_forward": 0.06, "target_xy_offset": (0.06, 0.0)},
            {"pivot_forward": 0.07, "target_xy_offset": (0.06, 0.0)},
            {"pivot_forward": 0.05, "target_xy_offset": (0.08, 0.0)},
            {"pivot_forward": 0.06, "target_xy_offset": (0.08, 0.0)},
            {"pivot_forward": 0.07, "target_xy_offset": (0.08, 0.0)},
        ]
        return [{**cup_base, **variant} for variant in variants]
    if preset == "cup_lower_body_compare_v1":
        cup_base = {
            **base,
            "mode": "cup",
            "release_steps": 0,
            "move_steps": 95,
            "pour_steps": 30,
            "scale": 0.055,
            "cup_grasp_z_offset": -0.045,
            "target_z_offset": 0.18,
            "target_xy_offset": (0.08, 0.0),
            "pour_axis_angle_delta": (0.0, 0.0, 0.0),
            "use_geometry_pour": False,
            "use_pivot_pour": True,
            "use_object_pose_place": True,
            "object_place_axis_angle_delta": (0.0, 0.25, 0.0),
            "pivot_forward": 0.06,
            "pivot_drop": 0.08,
            "pivot_lift": 0.02,
        }
        variants = [
            {"preset": "power", "cup_grasp_profile": "center"},
            {"preset": "power", "cup_grasp_profile": "lower_body"},
            {"preset": "cylindrical", "cup_grasp_profile": "cylindrical_lower"},
        ]
        return [{**cup_base, **variant} for variant in variants]
    if preset == "cup_cylindrical_tune_v1":
        cup_base = {
            **base,
            "mode": "cup",
            "preset": "cylindrical",
            "cup_grasp_profile": "cylindrical_lower",
            "release_steps": 0,
            "move_steps": 95,
            "pour_steps": 30,
            "scale": 0.055,
            "cup_grasp_z_offset": -0.045,
            "cup_grasp_z_floor": -0.075,
            "cup_grasp_side_offset": 0.035,
            "target_z_offset": 0.18,
            "target_xy_offset": (0.08, 0.0),
            "pour_axis_angle_delta": (0.0, 0.0, 0.0),
            "use_geometry_pour": False,
            "use_pivot_pour": True,
            "use_object_pose_place": True,
            "object_place_axis_angle_delta": (0.0, 0.25, 0.0),
            "pivot_forward": 0.06,
            "pivot_drop": 0.08,
            "pivot_lift": 0.02,
        }
        variants = [
            {"object_place_axis_angle_delta": (0.0, 0.25, 0.0)},
            {"object_place_axis_angle_delta": (0.0, 0.30, 0.0)},
            {"object_place_axis_angle_delta": (0.0, 0.35, 0.0)},
            {"object_place_axis_angle_delta": (0.0, 0.40, 0.0)},
            {"target_xy_offset": (0.10, 0.0), "object_place_axis_angle_delta": (0.0, 0.35, 0.0)},
            {"target_xy_offset": (0.10, 0.0), "object_place_axis_angle_delta": (0.0, 0.40, 0.0)},
            {"cup_grasp_side_offset": 0.020, "object_place_axis_angle_delta": (0.0, 0.35, 0.0)},
            {"cup_grasp_side_offset": 0.050, "object_place_axis_angle_delta": (0.0, 0.35, 0.0)},
            {"cup_grasp_z_floor": -0.065, "object_place_axis_angle_delta": (0.0, 0.35, 0.0)},
            {"cup_grasp_z_floor": -0.085, "object_place_axis_angle_delta": (0.0, 0.35, 0.0)},
        ]
        return [{**cup_base, **variant} for variant in variants]
    if preset == "cup_final_rotate_v1":
        cup_base = {
            **base,
            "mode": "cup",
            "preset": "cylindrical",
            "cup_grasp_profile": "cylindrical_lower",
            "release_steps": 0,
            "move_steps": 95,
            "pour_steps": 30,
            "scale": 0.055,
            "cup_grasp_z_offset": -0.045,
            "cup_grasp_z_floor": -0.075,
            "cup_grasp_side_offset": 0.035,
            "target_z_offset": 0.18,
            "target_xy_offset": (0.08, 0.0),
            "pour_axis_angle_delta": (0.0, 0.0, 0.0),
            "use_geometry_pour": False,
            "use_pivot_pour": True,
            "use_object_pose_place": True,
            "object_place_axis_angle_delta": (0.0, 0.40, 0.0),
            "pivot_forward": 0.06,
            "pivot_drop": 0.08,
            "pivot_lift": 0.02,
            "final_pour_steps": 18,
        }
        variants = [
            {"final_pour_axis_angle_delta": (0.0, 0.0, 0.0), "final_pour_steps": 0},
            {"final_pour_axis_angle_delta": (0.8, 0.0, 0.0)},
            {"final_pour_axis_angle_delta": (-0.8, 0.0, 0.0)},
            {"final_pour_axis_angle_delta": (0.0, 0.8, 0.0)},
            {"final_pour_axis_angle_delta": (0.0, -0.8, 0.0)},
            {"final_pour_axis_angle_delta": (0.0, 0.0, 0.8)},
            {"final_pour_axis_angle_delta": (0.0, 0.0, -0.8)},
        ]
        return [{**cup_base, **variant} for variant in variants]
    if preset == "cup_joint_posture_rotate_v1":
        cup_base = {
            **base,
            "mode": "cup",
            "preset": "cylindrical",
            "cup_grasp_profile": "cylindrical_lower",
            "release_steps": 0,
            "move_steps": 95,
            "pour_steps": 30,
            "scale": 0.055,
            "cup_grasp_z_offset": -0.045,
            "cup_grasp_z_floor": -0.075,
            "cup_grasp_side_offset": 0.035,
            "target_z_offset": 0.18,
            "target_xy_offset": (0.08, 0.0),
            "pour_axis_angle_delta": (0.0, 0.0, 0.0),
            "use_geometry_pour": False,
            "use_pivot_pour": True,
            "use_object_pose_place": True,
            "object_place_axis_angle_delta": (0.0, 0.40, 0.0),
            "pivot_forward": 0.06,
            "pivot_drop": 0.08,
            "pivot_lift": 0.02,
            "final_pour_joint_steps": 40,
        }
        variants = [
            {"final_pour_joint_deltas": {}, "final_pour_joint_steps": 0},
            {"final_pour_joint_deltas": {"robot0_r_wrist_yaw": 1.57}},
            {"final_pour_joint_deltas": {"robot0_r_wrist_pitch": 1.57}},
            {"final_pour_joint_deltas": {"robot0_r_shoulder_yaw": 1.57}},
            {"final_pour_joint_deltas": {"robot0_r_elbow_pitch": 1.57}},
        ]
        return [{**cup_base, **variant} for variant in variants]
    if preset == "cup_joint_posture_high_v1":
        cup_base = {
            **base,
            "mode": "cup",
            "preset": "cylindrical",
            "cup_grasp_profile": "cylindrical_lower",
            "release_steps": 0,
            "move_steps": 95,
            "pour_steps": 30,
            "scale": 0.055,
            "cup_grasp_z_offset": -0.045,
            "cup_grasp_z_floor": -0.075,
            "cup_grasp_side_offset": 0.035,
            "target_z_offset": 0.22,
            "target_xy_offset": (0.08, 0.0),
            "pour_axis_angle_delta": (0.0, 0.0, 0.0),
            "use_geometry_pour": False,
            "use_pivot_pour": True,
            "use_object_pose_place": True,
            "object_place_axis_angle_delta": (0.0, 0.40, 0.0),
            "pivot_forward": 0.06,
            "pivot_drop": 0.08,
            "pivot_lift": 0.05,
            "final_pour_joint_steps": 40,
        }
        rotation_variants = [
            {"final_pour_joint_deltas": {"robot0_r_wrist_yaw": 1.57}},
            {"final_pour_joint_deltas": {"robot0_r_wrist_pitch": 1.57}},
            {"final_pour_joint_deltas": {"robot0_r_shoulder_yaw": 1.57}},
        ]
        height_variants = [
            {"target_z_offset": 0.22, "pivot_lift": 0.05},
            {"target_z_offset": 0.26, "pivot_lift": 0.05},
            {"target_z_offset": 0.22, "pivot_lift": 0.08},
            {"target_z_offset": 0.26, "pivot_lift": 0.08},
        ]
        return [
            {**cup_base, **height_variant, **rotation_variant}
            for rotation_variant in rotation_variants
            for height_variant in height_variants
        ]
    if preset == "cup_joint_posture_shoulder_high_v1":
        cup_base = {
            **base,
            "mode": "cup",
            "preset": "cylindrical",
            "cup_grasp_profile": "cylindrical_lower",
            "release_steps": 0,
            "move_steps": 95,
            "pour_steps": 30,
            "scale": 0.055,
            "cup_grasp_z_offset": -0.045,
            "cup_grasp_z_floor": -0.075,
            "cup_grasp_side_offset": 0.035,
            "target_xy_offset": (0.08, 0.0),
            "pour_axis_angle_delta": (0.0, 0.0, 0.0),
            "use_geometry_pour": False,
            "use_pivot_pour": True,
            "use_object_pose_place": True,
            "object_place_axis_angle_delta": (0.0, 0.40, 0.0),
            "pivot_forward": 0.06,
            "pivot_drop": 0.08,
            "final_pour_joint_deltas": {"robot0_r_shoulder_yaw": 1.57},
            "final_pour_joint_steps": 40,
        }
        height_variants = [
            {"target_z_offset": 0.28, "pivot_lift": 0.10},
            {"target_z_offset": 0.30, "pivot_lift": 0.10},
            {"target_z_offset": 0.28, "pivot_lift": 0.12},
            {"target_z_offset": 0.30, "pivot_lift": 0.12},
            {"target_z_offset": 0.28, "pivot_lift": 0.14},
            {"target_z_offset": 0.30, "pivot_lift": 0.14},
        ]
        return [{**cup_base, **height_variant} for height_variant in height_variants]
    if preset == "cup_case7_high_wrist_pitch_v1":
        cup_base = {
            **base,
            "mode": "cup",
            "preset": "cylindrical",
            "cup_grasp_profile": "cylindrical_lower",
            "release_steps": 0,
            "move_steps": 95,
            "pour_steps": 30,
            "scale": 0.055,
            "cup_grasp_z_offset": -0.045,
            "cup_grasp_z_floor": -0.075,
            "cup_grasp_side_offset": 0.035,
            "target_z_offset": 0.47,
            "target_xy_offset": (0.08, 0.0),
            "pour_axis_angle_delta": (0.0, 0.0, 0.0),
            "use_geometry_pour": False,
            "use_pivot_pour": True,
            "use_object_pose_place": True,
            "object_place_axis_angle_delta": (0.0, 0.40, 0.0),
            "pivot_forward": 0.06,
            "pivot_drop": 0.08,
            "pivot_lift": 0.33,
            "final_pour_joint_steps": 40,
        }
        return [
            {**cup_base, "final_pour_joint_deltas": {"robot0_r_wrist_pitch": pitch}}
            for pitch in (2.10, 2.30, 2.50)
        ]
    if preset == "cup_case7_plus025_v1":
        cup_base = {
            **base,
            "mode": "cup",
            "preset": "cylindrical",
            "cup_grasp_profile": "cylindrical_lower",
            "release_steps": 0,
            "move_steps": 95,
            "pour_steps": 30,
            "scale": 0.055,
            "cup_grasp_z_offset": -0.045,
            "cup_grasp_z_floor": -0.075,
            "cup_grasp_side_offset": 0.035,
            "target_z_offset": 0.47,
            "target_xy_offset": (0.08, 0.0),
            "pour_axis_angle_delta": (0.0, 0.0, 0.0),
            "use_geometry_pour": False,
            "use_pivot_pour": True,
            "use_object_pose_place": True,
            "object_place_axis_angle_delta": (0.0, 0.40, 0.0),
            "pivot_forward": 0.06,
            "pivot_drop": 0.08,
            "pivot_lift": 0.33,
            "final_pour_joint_deltas": {"robot0_r_wrist_pitch": 1.57},
            "final_pour_joint_steps": 40,
        }
        return [cup_base]
    if preset == "cup_case7_yaw_combo_v1":
        cup_base = {
            **base,
            "mode": "cup",
            "preset": "cylindrical",
            "cup_grasp_profile": "cylindrical_lower",
            "release_steps": 0,
            "move_steps": 95,
            "pour_steps": 30,
            "scale": 0.055,
            "cup_grasp_z_offset": -0.045,
            "cup_grasp_z_floor": -0.075,
            "cup_grasp_side_offset": 0.035,
            "target_z_offset": 0.47,
            "target_xy_offset": (0.08, 0.0),
            "pour_axis_angle_delta": (0.0, 0.0, 0.0),
            "use_geometry_pour": False,
            "use_pivot_pour": True,
            "use_object_pose_place": True,
            "object_place_axis_angle_delta": (0.0, 0.40, 0.0),
            "pivot_forward": 0.06,
            "pivot_drop": 0.08,
            "pivot_lift": 0.33,
            "final_pour_joint_steps": 50,
        }
        yaw_variants = [
            {"robot0_r_wrist_yaw": 1.57},
            {"robot0_r_wrist_yaw": 2.10},
            {"robot0_r_wrist_yaw": 1.57, "robot0_r_wrist_roll": 0.80},
            {"robot0_r_wrist_yaw": 1.57, "robot0_r_wrist_roll": 1.20},
            {"robot0_r_wrist_yaw": 1.57, "robot0_r_shoulder_yaw": 0.40},
            {"robot0_r_wrist_yaw": 1.57, "robot0_r_elbow_pitch": 0.80},
        ]
        return [{**cup_base, "final_pour_joint_deltas": joints} for joints in yaw_variants]
    if preset == "cup_case7_yaw_widecam_debug_v1":
        cup_base = {
            **base,
            "mode": "cup",
            "preset": "cylindrical",
            "cup_grasp_profile": "cylindrical_lower",
            "release_steps": 0,
            "move_steps": 95,
            "pour_steps": 30,
            "scale": 0.055,
            "cup_grasp_z_offset": -0.045,
            "cup_grasp_z_floor": -0.075,
            "cup_grasp_side_offset": 0.035,
            "target_z_offset": 0.47,
            "target_xy_offset": (0.08, 0.0),
            "pour_axis_angle_delta": (0.0, 0.0, 0.0),
            "use_geometry_pour": False,
            "use_pivot_pour": True,
            "use_object_pose_place": True,
            "object_place_axis_angle_delta": (0.0, 0.40, 0.0),
            "pivot_forward": 0.06,
            "pivot_drop": 0.08,
            "pivot_lift": 0.33,
            "final_pour_joint_steps": 50,
        }
        yaw_variants = [
            {"robot0_r_wrist_yaw": 1.57, "robot0_r_wrist_roll": 1.20},
            {"robot0_r_wrist_yaw": 1.57, "robot0_r_shoulder_yaw": 0.40},
        ]
        return [{**cup_base, "final_pour_joint_deltas": joints} for joints in yaw_variants]
    if preset == "cup_case7_yaw_roll_refine_v1":
        cup_base = {
            **base,
            "mode": "cup",
            "preset": "cylindrical",
            "cup_grasp_profile": "cylindrical_lower",
            "release_steps": 0,
            "move_steps": 95,
            "pour_steps": 30,
            "scale": 0.055,
            "cup_grasp_z_offset": -0.045,
            "cup_grasp_z_floor": -0.075,
            "cup_grasp_side_offset": 0.035,
            "target_z_offset": 0.47,
            "target_xy_offset": (0.08, 0.0),
            "pour_axis_angle_delta": (0.0, 0.0, 0.0),
            "use_geometry_pour": False,
            "use_pivot_pour": True,
            "use_object_pose_place": True,
            "object_place_axis_angle_delta": (0.0, 0.40, 0.0),
            "pivot_forward": 0.06,
            "pivot_drop": 0.08,
            "pivot_lift": 0.33,
            "final_pour_joint_steps": 50,
        }
        yaw_variants = [
            {"robot0_r_wrist_yaw": 1.57, "robot0_r_wrist_roll": 1.20},
            {"robot0_r_wrist_yaw": 1.57, "robot0_r_wrist_roll": 1.35},
            {"robot0_r_wrist_yaw": 1.57, "robot0_r_wrist_roll": 1.50},
            {"robot0_r_wrist_yaw": 1.70, "robot0_r_wrist_roll": 1.35},
            {"robot0_r_wrist_yaw": 1.57, "robot0_r_wrist_roll": 1.35, "robot0_r_shoulder_yaw": 0.15},
            {"robot0_r_wrist_yaw": 1.57, "robot0_r_wrist_roll": 1.50, "robot0_r_shoulder_yaw": 0.15},
        ]
        return [{**cup_base, "final_pour_joint_deltas": joints} for joints in yaw_variants]
    if preset == "cup_case7_yaw_roll_refine2_v1":
        cup_base = {
            **base,
            "mode": "cup",
            "preset": "cylindrical",
            "cup_grasp_profile": "cylindrical_lower",
            "release_steps": 0,
            "move_steps": 95,
            "pour_steps": 30,
            "scale": 0.055,
            "cup_grasp_z_offset": -0.045,
            "cup_grasp_z_floor": -0.075,
            "cup_grasp_side_offset": 0.035,
            "target_z_offset": 0.47,
            "target_xy_offset": (0.08, 0.0),
            "pour_axis_angle_delta": (0.0, 0.0, 0.0),
            "use_geometry_pour": False,
            "use_pivot_pour": True,
            "use_object_pose_place": True,
            "object_place_axis_angle_delta": (0.0, 0.40, 0.0),
            "pivot_forward": 0.06,
            "pivot_drop": 0.08,
            "pivot_lift": 0.33,
            "final_pour_joint_steps": 55,
        }
        yaw_variants = [
            {"robot0_r_wrist_yaw": 1.57, "robot0_r_wrist_roll": 1.60, "robot0_r_shoulder_yaw": 0.15},
            {"robot0_r_wrist_yaw": 1.57, "robot0_r_wrist_roll": 1.70, "robot0_r_shoulder_yaw": 0.15},
            {"robot0_r_wrist_yaw": 1.57, "robot0_r_wrist_roll": 1.50, "robot0_r_shoulder_yaw": 0.25},
            {"robot0_r_wrist_yaw": 1.57, "robot0_r_wrist_roll": 1.65, "robot0_r_shoulder_yaw": 0.25},
            {"robot0_r_wrist_yaw": 1.70, "robot0_r_wrist_roll": 1.50, "robot0_r_shoulder_yaw": 0.15},
            {"robot0_r_wrist_yaw": 1.70, "robot0_r_wrist_roll": 1.50, "robot0_r_shoulder_yaw": 0.25},
        ]
        return [{**cup_base, "final_pour_joint_deltas": joints} for joints in yaw_variants]
    if preset == "cup_case7_default_v1":
        cup_base = {
            **base,
            "mode": "cup",
            "preset": "cylindrical",
            "cup_grasp_profile": "cylindrical_lower",
            "release_steps": 0,
            "move_steps": 95,
            "pour_steps": 30,
            "scale": 0.055,
            "cup_grasp_z_offset": -0.045,
            "cup_grasp_z_floor": -0.075,
            "cup_grasp_side_offset": 0.035,
            "target_z_offset": 0.47,
            "target_xy_offset": (0.08, 0.0),
            "pour_axis_angle_delta": (0.0, 0.0, 0.0),
            "use_geometry_pour": False,
            "use_pivot_pour": True,
            "use_object_pose_place": True,
            "object_place_axis_angle_delta": (0.0, 0.40, 0.0),
            "pivot_forward": 0.06,
            "pivot_drop": 0.08,
            "pivot_lift": 0.33,
            "final_pour_joint_deltas": {
                "robot0_r_wrist_yaw": 1.57,
                "robot0_r_wrist_roll": 1.50,
                "robot0_r_shoulder_yaw": 0.25,
            },
            "final_pour_joint_steps": 55,
        }
        return [cup_base]
    raise ValueError(f"unknown sweep preset: {preset}")


def _quat_wxyz_to_matrix(quat: Any) -> np.ndarray | None:
    if quat is None:
        return None
    arr = np.asarray(quat, dtype=float)
    if arr.shape != (4,):
        return None
    norm = float(np.linalg.norm(arr))
    if norm <= 1e-9:
        return None
    w, x, y, z = arr / norm
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=float,
    )


def _final_axis_dot_down(summary: dict[str, Any]) -> dict[str, float] | None:
    result = summary.get("result") or {}
    if not isinstance(result, dict):
        return None
    final_state = result.get("final_state") or {}
    scene_poses = final_state.get("scene_poses") if isinstance(final_state, dict) else {}
    bodies = scene_poses.get("bodies") if isinstance(scene_poses, dict) else {}
    cup_body = bodies.get("obj_container_main") if isinstance(bodies, dict) else {}
    rotation = _quat_wxyz_to_matrix(cup_body.get("quat") if isinstance(cup_body, dict) else None)
    if rotation is None:
        return None
    down = np.array([0.0, 0.0, -1.0], dtype=float)
    axes = {
        "+X": np.array([1.0, 0.0, 0.0], dtype=float),
        "-X": np.array([-1.0, 0.0, 0.0], dtype=float),
        "+Y": np.array([0.0, 1.0, 0.0], dtype=float),
        "-Y": np.array([0.0, -1.0, 0.0], dtype=float),
        "+Z": np.array([0.0, 0.0, 1.0], dtype=float),
        "-Z": np.array([0.0, 0.0, -1.0], dtype=float),
    }
    return {name: float((rotation @ axis).dot(down)) for name, axis in axes.items()}


def _case_metrics(summary: dict[str, Any]) -> dict[str, Any]:
    result = summary.get("result") or {}
    grasp_attempts = result.get("grasp_attempts") or []
    best_lift = 0.0
    best_grasp_distance = None
    if isinstance(grasp_attempts, list):
        for attempt in grasp_attempts:
            if isinstance(attempt, dict):
                best_lift = max(best_lift, float(attempt.get("object_lifted_z") or 0.0))
                distance = attempt.get("distance_to_target_after_grasp")
                if distance is not None:
                    best_grasp_distance = (
                        float(distance)
                        if best_grasp_distance is None
                        else min(best_grasp_distance, float(distance))
                    )
    place_summary = result.get("place_summary") if isinstance(result, dict) else {}
    return {
        "phase": result.get("phase") if isinstance(result, dict) else None,
        "distance_to_target": result.get("distance_to_target") if isinstance(result, dict) else None,
        "ball_distance_to_target": result.get("ball_distance_to_target") if isinstance(result, dict) else None,
        "cup_distance_to_target": result.get("cup_distance_to_target") if isinstance(result, dict) else None,
        "ball_xy_distance_to_target": result.get("ball_xy_distance_to_target") if isinstance(result, dict) else None,
        "cup_xy_distance_to_target": result.get("cup_xy_distance_to_target") if isinstance(result, dict) else None,
        "ball_vertical_offset_to_target": (
            result.get("ball_vertical_offset_to_target") if isinstance(result, dict) else None
        ),
        "cup_vertical_offset_to_target": (
            result.get("cup_vertical_offset_to_target") if isinstance(result, dict) else None
        ),
        "cup_ball_distance": result.get("cup_ball_distance") if isinstance(result, dict) else None,
        "cup_target_collision_risk": result.get("cup_target_collision_risk") if isinstance(result, dict) else None,
        "task_completed": result.get("task_completed") if isinstance(result, dict) else None,
        "reward": result.get("reward") if isinstance(result, dict) else None,
        "best_grasp_lift_z": best_lift,
        "best_distance_after_grasp": best_grasp_distance,
        "best_transport_distance": (
            place_summary.get("best_distance_to_target")
            if isinstance(place_summary, dict)
            else None
        ),
        "distance_before_release": (
            place_summary.get("distance_to_target_before_release")
            if isinstance(place_summary, dict)
            else None
        ),
        "final_axis_dot_down": _final_axis_dot_down(summary),
    }


def _primary_distance(summary: dict[str, Any]) -> float:
    metrics = summary.get("metrics", {})
    params = summary.get("params", {})
    if isinstance(params, dict) and params.get("mode") == "cup":
        value = metrics.get("ball_xy_distance_to_target") if isinstance(metrics, dict) else None
        if value is not None:
            return float(value)
    for key in ("distance_to_target", "ball_distance_to_target", "cup_distance_to_target"):
        value = metrics.get(key) if isinstance(metrics, dict) else None
        if value is not None:
            return float(value)
    return 1e9


def _case_digest(summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "case_idx": summary["case_idx"],
        "case_dir": summary["case_dir"],
        "params": summary["params"],
        "metrics": summary["metrics"],
    }


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    config = DictLoader.load(args.config)
    _apply_camera_overrides(config, args)
    case_summaries = []
    for idx, params in enumerate(_sweep_cases(args.sweep_preset), start=1):
        case_dir = out_dir / f"case_{idx:02d}"
        case_dir.mkdir(parents=True, exist_ok=True)
        summary = _run_case(
            config=config,
            config_path=args.config,
            out_dir=case_dir,
            seed=args.seed,
            fps=args.fps,
            params=params,
            save_video=(args.save_case_videos or args.sweep_preset == "single"),
        )
        summary["case_idx"] = idx
        summary["case_dir"] = str(case_dir)
        summary["metrics"] = _case_metrics(summary)
        case_summaries.append(summary)
        print(json.dumps({
            "case_idx": idx,
            "params": summary["params"],
            "metrics": summary["metrics"],
            "ok": summary["ok"],
            "task_completed": summary["task_completed"],
        }, indent=2))

    ranked = sorted(case_summaries, key=_primary_distance)
    aggregate = {
        "config": args.config,
        "seed": args.seed,
        "sweep_preset": args.sweep_preset,
        "num_cases": len(case_summaries),
        "best_case": _case_digest(ranked[0]) if ranked else None,
        "cases": [
            {
                "case_idx": summary["case_idx"],
                "case_dir": summary["case_dir"],
                "params": summary["params"],
                "metrics": summary["metrics"],
                "ok": summary["ok"],
                "task_completed": summary["task_completed"],
                "truncated": summary["truncated"],
                "transition_count": summary["transition_count"],
            }
            for summary in case_summaries
        ],
    }
    (out_dir / "sweep_summary.json").write_text(json.dumps(aggregate, indent=2), encoding="utf-8")
    print(json.dumps(aggregate, indent=2))


if __name__ == "__main__":
    main()
