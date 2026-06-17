#!/usr/bin/env python3
"""Validate RGBD -> world point cloud -> grasp-pose estimation for RoboCasa GR1.

This is a perception-bridge smoke test, not a task solver. It verifies that the
simulator can provide metric depth and camera calibration, and that the GR1 API
can turn a target affordance into a compact GraspNet-compatible candidate pose.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from capx.envs.configs.instantiate import instantiate


TASK_SPECS: dict[str, dict[str, Any]] = {
    "drawer": {
        "env_name": "TabletopOpenDrawerDoor",
        "camera_names": ["robot0_frontview", "robot0_agentview_center"],
        "render_camera": "robot0_frontview",
        "camera_widths": 224,
        "camera_heights": 224,
        "camera_name": "robot0_frontview",
        "name_hints": ["drawer_tabletop", "door_handle"],
        "arm": "right",
        "crop_radius": 0.14,
    },
    "two_arm_lift": {
        "env_name": "TwoArmLift",
        "camera_names": ["frontview", "sideview"],
        "render_camera": "frontview",
        "camera_widths": 224,
        "camera_heights": 224,
        "camera_name": "frontview",
        "name_hints": ["pot_handle0_c", "pot_handle"],
        "arm": "right",
        "crop_radius": 0.16,
    },
    "pnp_pouring": {
        "env_name": "PnPPouring",
        "camera_names": ["egoview", "robot0_agentview_center"],
        "render_camera": "egoview",
        "camera_widths": 320,
        "camera_heights": 240,
        "camera_fovy_overrides": {"egoview": 100},
        "camera_name": "egoview",
        "object_name": "obj_container",
        "name_hints": ["obj_container", "cup"],
        "arm": "right",
        "crop_radius": 0.14,
    },
}


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
        "record_transition_fps": 0.0,
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


def _run_task(task_name: str, spec: dict[str, Any], out_dir: Path, *, seed: int, max_steps: int) -> dict[str, Any]:
    env = instantiate(_build_env_config(spec, seed=seed, max_steps=max_steps))
    try:
        env.reset(seed=seed)
        code = f"""
SCENE = observe_scene(name_hints={list(spec['name_hints'])!r}, include_geoms=True)
GRASP = estimate_depth_grasp_pose(
    camera_name={spec['camera_name']!r},
    object_name={spec.get('object_name')!r},
    name_hints={list(spec['name_hints'])!r},
    arm={spec['arm']!r},
    crop_radius={float(spec['crop_radius'])!r},
    approach_distance=0.10,
    subsample_factor=3,
)
print({{"task": {task_name!r}, "method": GRASP["method"], "target": GRASP["target"], "crop_point_count": GRASP["crop_point_count"], "grasp_pos": GRASP["grasp_pos"], "pregrasp_pos": GRASP["pregrasp_pos"]}})
"""
        _obs, reward, terminated, truncated, info = env.step(code)
        namespace = getattr(env, "_exec_globals", {})
        summary = {
            "task": task_name,
            "env_name": spec["env_name"],
            "ok": int(info.get("sandbox_rc", 1)) == 0,
            "reward": float(reward),
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "stdout": info.get("stdout", ""),
            "stderr": info.get("stderr", ""),
            "grasp": _jsonify(namespace.get("GRASP")),
            "scene_task_completed": _jsonify(namespace.get("SCENE", {}).get("task_state", {}).get("task_completed") if isinstance(namespace.get("SCENE"), dict) else None),
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
    parser.add_argument("--tasks", nargs="+", default=["drawer", "two_arm_lift"], choices=sorted(TASK_SPECS))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=80)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.output_dir)
    root.mkdir(parents=True, exist_ok=True)
    summaries = []
    for task in args.tasks:
        summary = _run_task(task, TASK_SPECS[task], root / task, seed=args.seed, max_steps=args.max_steps)
        summaries.append(summary)
        grasp = summary.get("grasp") or {}
        print(json.dumps({"task": task, "ok": summary["ok"], "crop_point_count": grasp.get("crop_point_count")}, sort_keys=True))
    aggregate = {"seed": int(args.seed), "tasks": args.tasks, "all_ok": all(s["ok"] for s in summaries), "summaries": summaries}
    (root / "depth_grasp_summary.json").write_text(json.dumps(_jsonify(aggregate), indent=2), encoding="utf-8")
    if not aggregate["all_ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
