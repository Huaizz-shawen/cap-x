#!/usr/bin/env python3
"""Run a depth-guided closed-loop drawer handle smoke for RoboCasa GR1."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import numpy as np

from capx.envs.configs.instantiate import instantiate
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
    path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(path, frames, fps=fps)


def _write_keyframes(path: Path, frames: list[np.ndarray], prefix: str) -> dict[str, str]:
    if not frames:
        return {}
    path.mkdir(parents=True, exist_ok=True)
    last = len(frames) - 1
    indices = {
        "start": 0,
        "approach": int(round(last * 0.45)),
        "close": int(round(last * 0.70)),
        "pull": last,
    }
    written: dict[str, str] = {}
    for name, idx in indices.items():
        idx = max(0, min(last, int(idx)))
        out = path / f"{prefix}_{name}_{idx:04d}.png"
        imageio.imwrite(out, frames[idx])
        written[name] = str(out)
    return written


def _build_env_config(*, seed: int, max_steps: int) -> dict[str, Any]:
    return {
        "_target_": "capx.envs.tasks.gr1.robocasa.GR1RobocasaCodeEnv",
        "cfg": {
            "_target_": "capx.envs.tasks.base.CodeExecEnvConfig",
            "low_level": {
                "_target_": "capx.envs.simulators.robocasa_gr1.GR1RobocasaLowLevel",
                "env_name": "TabletopOpenDrawerDoor",
                "robots": "GR1ArmsOnlyFourierHands",
                "controller_config": "WHOLE_BODY_MINK_IK",
                "camera_names": ["robot0_frontview", "robot0_agentview_center"],
                "render_camera": "robot0_frontview",
                "camera_widths": 224,
                "camera_heights": 224,
                "max_steps": int(max_steps),
                "seed": int(seed),
                "record_transition_fps": 20.0,
                "record_transition_depth": False,
            },
            "privileged": False,
            "apis": ["GR1RobocasaControlApi"],
        },
    }


def _parse_vec3(text: str | None, default: tuple[float, float, float]) -> tuple[float, float, float]:
    if text is None or not str(text).strip():
        return default
    parts = [float(item.strip()) for item in str(text).split(",")]
    if len(parts) != 3:
        raise ValueError(f"Expected comma-separated vec3, got {text!r}")
    return (parts[0], parts[1], parts[2])


def run(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    env = instantiate(_build_env_config(seed=args.seed, max_steps=args.max_steps))
    try:
        env.reset(seed=args.seed)
        low_level = env.low_level_env
        if hasattr(low_level, "enable_video_capture"):
            low_level.enable_video_capture(True, clear=True, wrist_camera=True)
        wrist_grasp_offset = _parse_vec3(args.wrist_grasp_offset, (0.0, -0.035, 0.10))
        wrist_pregrasp_offset = _parse_vec3(args.wrist_pregrasp_offset, (0.0, -0.14, 0.12))
        wrist_axis_angle_delta = _parse_vec3(args.wrist_axis_angle_delta, (0.0, 0.0, 0.0))
        anchor_source = str(args.anchor_source).lower().strip()
        if anchor_source == "privileged":
            anchor_code = """
SCENE = observe_scene(name_hints=["drawer_tabletop", "door_handle", "handle"], include_geoms=True)
CALIBRATION = probe_eef_directions(
    arm="right",
    step=0.025,
    axes=("+x", "-x", "+y", "-y", "+z", "-z"),
    steps_per_axis=3,
)
SKILL = select_gr1_skill(
    task_type="drawer",
    arm="right",
    hand_pose_family={hand_preset!r},
    scene=SCENE,
    calibration=CALIBRATION,
    use_depth_grasp=True,
    depth_camera_name={camera_name!r},
)
ANCHOR_DIAGNOSTICS = {{"anchor_source": "privileged"}}
""".format(hand_preset=args.hand_preset, camera_name=args.depth_camera_name)
        elif anchor_source == "sam3":
            anchor_code = """
SCENE = {{"privilege_level": "not_used_for_anchor"}}
CALIBRATION = {{"axis_source": "fixed_task_prior_-y"}}
VISUAL_GRASP = estimate_sam3_rgbd_grasp_pose(
    camera_name={camera_name!r},
    text_prompt={sam3_text_prompt!r},
    arm="right",
    approach_distance=0.10,
)
SKILL = {{
    "skill_type": "drawer_pull",
    "object_name": "drawer_handle_visual_sam3",
    "target_name": None,
    "arm": "right",
    "hand_pose_family": {hand_preset!r},
    "handle_affordance": None,
    "ranked_affordance_candidates": [],
    "handle_pos": np.asarray(VISUAL_GRASP["grasp_pos"], dtype=float),
    "visual_grasp": VISUAL_GRASP,
    "desired_pull_axis": np.array([0.0, -1.0, 0.0], dtype=float),
    "pull_axis": np.array([0.0, -1.0, 0.0], dtype=float),
    "axis_source": "fixed_task_prior_-y",
    "axis_diagnostics": {{"selected_probe": "fixed_task_prior_-y"}},
    "anchor_source": "sam3_rgbd_mask",
    "privilege_level": "rgbd_sam3_no_sim_object_pose_for_anchor",
}}
ANCHOR_DIAGNOSTICS = {{
    "anchor_source": "sam3",
    "sam3_text_prompt": {sam3_text_prompt!r},
    "visual_grasp_method": VISUAL_GRASP.get("method"),
    "visual_grasp_source": VISUAL_GRASP.get("source"),
}}
""".format(
                camera_name=args.depth_camera_name,
                sam3_text_prompt=args.sam3_text_prompt,
                hand_preset=args.hand_preset,
            )
        elif anchor_source == "rgbd_roi":
            anchor_code = """
SCENE = {{"privilege_level": "not_used_for_anchor"}}
CALIBRATION = {{"axis_source": "fixed_task_prior_-y"}}
VISUAL_GRASP = estimate_rgbd_roi_grasp_pose(
    camera_name={camera_name!r},
    roi_xyxy=({roi_xyxy}),
    arm="right",
    approach_distance=0.10,
)
SKILL = {{
    "skill_type": "drawer_pull",
    "object_name": "drawer_handle_visual_rgbd_roi",
    "target_name": None,
    "arm": "right",
    "hand_pose_family": {hand_preset!r},
    "handle_affordance": None,
    "ranked_affordance_candidates": [],
    "handle_pos": np.asarray(VISUAL_GRASP["grasp_pos"], dtype=float),
    "visual_grasp": VISUAL_GRASP,
    "desired_pull_axis": np.array([0.0, -1.0, 0.0], dtype=float),
    "pull_axis": np.array([0.0, -1.0, 0.0], dtype=float),
    "axis_source": "fixed_task_prior_-y",
    "axis_diagnostics": {{"selected_probe": "fixed_task_prior_-y"}},
    "anchor_source": "rgbd_roi_mask",
    "privilege_level": "rgbd_roi_no_sim_object_pose_for_anchor",
}}
ANCHOR_DIAGNOSTICS = {{
    "anchor_source": "rgbd_roi",
    "roi_xyxy": [{roi_json}],
    "visual_grasp_method": VISUAL_GRASP.get("method"),
    "visual_grasp_source": VISUAL_GRASP.get("source"),
}}
""".format(
                camera_name=args.depth_camera_name,
                roi_xyxy=", ".join(str(float(v.strip())) for v in args.rgbd_roi_xyxy.split(",")),
                roi_json=", ".join(str(float(v.strip())) for v in args.rgbd_roi_xyxy.split(",")),
                hand_preset=args.hand_preset,
            )
        else:
            raise ValueError(f"Unsupported anchor source: {args.anchor_source!r}")
        code = f"""
import numpy as np
{anchor_code}
SKILL["wrist_grasp_pos"] = SKILL["handle_pos"] + np.array({wrist_grasp_offset!r}, dtype=float)
SKILL["wrist_pregrasp_pos"] = SKILL["handle_pos"] + np.array({wrist_pregrasp_offset!r}, dtype=float)
SKILL["wrist_axis_angle_delta"] = np.array({wrist_axis_angle_delta!r}, dtype=float)
SKILL["wrist_orientation_frame"] = {args.wrist_orientation_frame!r}
EVAL_HANDLE_BEFORE = None
EVAL_HANDLE_AFTER = None
EVAL_HANDLE_DELTA = None
try:
    _EVAL_SCENE_BEFORE = observe_scene(
        name_hints=["drawer_tabletop", "door_handle", "handle"],
        include_geoms=True,
    )
    EVAL_HANDLE_BEFORE = select_gr1_skill(
        task_type="drawer",
        arm="right",
        hand_pose_family={args.hand_preset!r},
        scene=_EVAL_SCENE_BEFORE,
    )["handle_pos"]
except Exception as exc:
    EVAL_HANDLE_BEFORE = {{"error": repr(exc)}}
RESULT = execute_gr1_skill(SKILL, smoke={bool(args.smoke)!r})
try:
    _EVAL_SCENE_AFTER = observe_scene(
        name_hints=["drawer_tabletop", "door_handle", "handle"],
        include_geoms=True,
    )
    EVAL_HANDLE_AFTER = select_gr1_skill(
        task_type="drawer",
        arm="right",
        hand_pose_family={args.hand_preset!r},
        scene=_EVAL_SCENE_AFTER,
    )["handle_pos"]
    if not isinstance(EVAL_HANDLE_BEFORE, dict):
        EVAL_HANDLE_DELTA = np.asarray(EVAL_HANDLE_AFTER, dtype=float) - np.asarray(EVAL_HANDLE_BEFORE, dtype=float)
except Exception as exc:
    EVAL_HANDLE_AFTER = {{"error": repr(exc)}}
VERIFY = verify_success_or_contact(RESULT)
print({{"verify": VERIFY, "eef_to_handle_after_close": RESULT.get("eef_to_handle_after_close"), "handle_delta": RESULT.get("handle_delta"), "posthoc_handle_delta": EVAL_HANDLE_DELTA, "wrist_target_quat_xyzw": RESULT.get("wrist_target_quat_xyzw"), "visual_target": SKILL.get("visual_grasp", {{}}).get("target")}})
"""
        _obs, reward, terminated, truncated, info = env.step(code)
        transition_dataset = env.get_transition_dataset() if hasattr(env, "get_transition_dataset") else None
        if transition_dataset is not None:
            transition_dataset["trial"] = 1
            transition_dataset["attempt"] = 1
            transition_dataset["diagnostic_task"] = "drawer_depth_closed_loop"
            save_transition_dataset(out_dir, transition_dataset)
        frames = low_level.get_video_frames() if hasattr(low_level, "get_video_frames") else []
        wrist_frames = low_level.get_wrist_video_frames() if hasattr(low_level, "get_wrist_video_frames") else []
        _write_video(out_dir / "drawer_depth_closed_loop_frontview.mp4", frames, fps=args.fps)
        _write_video(out_dir / "drawer_depth_closed_loop_wrist.mp4", wrist_frames, fps=args.fps)
        keyframes = {
            "frontview": _write_keyframes(out_dir / "keyframes", frames, "frontview"),
            "wrist": _write_keyframes(out_dir / "keyframes", wrist_frames, "wrist"),
        }
        namespace = getattr(env, "_exec_globals", {})
        summary = {
            "ok": int(info.get("sandbox_rc", 1)) == 0,
            "anchor_source": anchor_source,
            "reward": float(reward),
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "task_completed": bool(info.get("task_completed", False)),
            "stdout": info.get("stdout", ""),
            "stderr": info.get("stderr", ""),
            "scene": _jsonify(namespace.get("SCENE")),
            "calibration": _jsonify(namespace.get("CALIBRATION")),
            "anchor_diagnostics": _jsonify(namespace.get("ANCHOR_DIAGNOSTICS")),
            "skill": _jsonify(namespace.get("SKILL")),
            "result": _jsonify(namespace.get("RESULT")),
            "verify": _jsonify(namespace.get("VERIFY")),
            "posthoc_privileged_eval": {
                "handle_before": _jsonify(namespace.get("EVAL_HANDLE_BEFORE")),
                "handle_after": _jsonify(namespace.get("EVAL_HANDLE_AFTER")),
                "handle_delta": _jsonify(namespace.get("EVAL_HANDLE_DELTA")),
                "note": "Evaluation only; these privileged poses are not used to choose or execute the action anchor.",
            },
            "frame_count": len(frames),
            "wrist_frame_count": len(wrist_frames),
            "keyframes": keyframes,
            "transition_count": len(transition_dataset.get("transitions", [])) if transition_dataset is not None else 0,
        }
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        return summary
    finally:
        if hasattr(env, "close"):
            env.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=360)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--smoke", action="store_true", help="Use shorter approach/pull motion.")
    parser.add_argument("--hand-preset", default="hook")
    parser.add_argument(
        "--anchor-source",
        choices=("privileged", "sam3", "rgbd_roi"),
        default="privileged",
        help="How to choose the drawer anchor. sam3/rgbd_roi avoid simulator object poses for the anchor.",
    )
    parser.add_argument("--depth-camera-name", default="robot0_frontview")
    parser.add_argument("--sam3-text-prompt", default="drawer handle")
    parser.add_argument(
        "--rgbd-roi-xyxy",
        default="0.34,0.42,0.66,0.72",
        help="Normalized image ROI fallback used by --anchor-source rgbd_roi.",
    )
    parser.add_argument("--wrist-grasp-offset", default="0.0,-0.035,0.10")
    parser.add_argument("--wrist-pregrasp-offset", default="0.0,-0.14,0.12")
    parser.add_argument(
        "--wrist-axis-angle-delta",
        default="0.0,0.0,0.0",
        help="Axis-angle delta used to construct an absolute target wrist orientation during approach/close/pull.",
    )
    parser.add_argument(
        "--wrist-orientation-frame",
        choices=("world", "local"),
        default="world",
        help="Reference frame for applying --wrist-axis-angle-delta before passing an absolute target orientation to IK.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = run(args)
    result = summary.get("result") or {}
    print(json.dumps({
        "ok": summary["ok"],
        "anchor_source": summary["anchor_source"],
        "task_completed": summary["task_completed"],
        "transition_count": summary["transition_count"],
        "eef_to_handle_after_close": result.get("eef_to_handle_after_close"),
        "handle_delta": result.get("handle_delta"),
        "posthoc_handle_delta": summary.get("posthoc_privileged_eval", {}).get("handle_delta"),
        "wrist_axis_angle_delta": result.get("wrist_axis_angle_delta"),
        "wrist_orientation_frame": result.get("wrist_orientation_frame"),
        "wrist_target_quat_xyzw": result.get("wrist_target_quat_xyzw"),
    }, sort_keys=True))
    if not summary["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
