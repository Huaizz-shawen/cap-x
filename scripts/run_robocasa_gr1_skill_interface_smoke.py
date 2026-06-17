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
    },
    "drawer": {
        "env_name": "TabletopOpenDrawerDoor",
        "skill_type": "drawer",
        "arm": "right",
        "camera_names": ["robot0_frontview", "robot0_agentview_center"],
        "render_camera": "robot0_frontview",
        "camera_widths": 224,
        "camera_heights": 224,
        "name_hints": ["drawer", "handle", "fixture"],
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
        if hasattr(low_level, "enable_video_capture"):
            low_level.enable_video_capture(save_videos, clear=True, wrist_camera=True)

        code = f"""
SCENE = observe_scene(name_hints={list(spec["name_hints"])!r}, include_geoms=True)
CALIBRATION = probe_eef_directions(
    arm={spec["arm"]!r},
    step=0.025,
    axes=("+x", "-x", "+y", "-y", "+z", "-z"),
    steps_per_axis=3,
)
SKILL = select_gr1_skill(
    task_type={spec["skill_type"]!r},
    arm={spec["arm"]!r},
    scene=SCENE,
    calibration=CALIBRATION,
)
RESULT = execute_gr1_skill(SKILL, smoke={bool(smoke)!r})
VERIFY = verify_success_or_contact(RESULT)
print({{"task": {task_name!r}, "skill_type": SKILL["skill_type"], "verify": VERIFY}})
"""
        _obs, reward, terminated, truncated, info = env.step(code)
        transition_dataset = env.get_transition_dataset() if hasattr(env, "get_transition_dataset") else None
        if transition_dataset is not None:
            transition_dataset["trial"] = 1
            transition_dataset["attempt"] = 1
            transition_dataset["diagnostic_task"] = task_name
            transition_dataset["diagnostic_skill_spec"] = spec
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
            "skill": _jsonify(namespace.get("SKILL")),
            "result": _jsonify(namespace.get("RESULT")),
            "verify": _jsonify(namespace.get("VERIFY")),
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.output_dir)
    root.mkdir(parents=True, exist_ok=True)

    summaries = []
    for task_name in args.tasks:
        task_dir = root / task_name
        summary = _run_task(
            task_name=task_name,
            spec=TASK_SPECS[task_name],
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
