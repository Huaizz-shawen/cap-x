#!/usr/bin/env python3
"""Search task-specific RoboCasa GR1 skill adapters.

This script intentionally runs small deterministic sweeps and records ranked
candidate adapters. It does not change the stable collection defaults; promotion
is a separate validation decision.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from capx.skills.adapter_registry import SkillAdapterRecord, SkillAdapterRegistry


DEFAULT_REGISTRY = REPO_ROOT / "outputs" / "robocasa-gr1" / "skill_adapters" / "registry.json"


BASE_PNP_OVERRIDES: dict[str, Any] = {
    "deprivileged_observation": True,
    "visual_anchor_diagnostics": True,
    "skill_overrides": {
        "use_object_pose_place": False,
        "use_transport_distance_feedback": False,
    },
}


PNP_POUR_PIVOT_CANDIDATES: list[dict[str, Any]] = [
    {
        "name": "stable_fixed_joint_v1",
        "template": "pour_pivot",
        "assumptions": [
            "visual anchors provide source cup and target container control points",
            "transport is fixed horizon toward visual target",
            "final pour uses current validated fixed joint heuristic",
        ],
        "params": {
            "target_xy_offset": [0.08, 0.0],
            "target_z_offset": 0.47,
            "pivot_forward": 0.06,
            "pivot_drop": 0.08,
            "pivot_lift": 0.33,
            "final_pour_joint_deltas": {
                "robot0_r_wrist_yaw": 1.57,
                "robot0_r_wrist_roll": 1.75,
                "robot0_r_shoulder_yaw": 0.25,
            },
            "final_pour_joint_steps": 75,
            "use_final_pour_kinematic_policy": False,
        },
    },
    {
        "name": "lower_pivot_more_drop",
        "template": "pour_pivot",
        "assumptions": ["lower held cup during final approach while keeping validated final joint heuristic"],
        "params": {
            "target_xy_offset": [0.08, 0.0],
            "target_z_offset": 0.42,
            "pivot_forward": 0.07,
            "pivot_drop": 0.12,
            "pivot_lift": 0.18,
            "final_pour_joint_deltas": {
                "robot0_r_wrist_yaw": 1.57,
                "robot0_r_wrist_roll": 1.75,
                "robot0_r_shoulder_yaw": 0.25,
            },
            "final_pour_joint_steps": 75,
            "use_final_pour_kinematic_policy": False,
        },
    },
    {
        "name": "visual_kinematic_axis_pos_y",
        "template": "pour_pivot",
        "assumptions": ["experimental visual-target wrist axis alignment; no direct joint posture"],
        "params": {
            "target_xy_offset": [0.08, 0.0],
            "target_z_offset": 0.47,
            "pivot_forward": 0.06,
            "pivot_drop": 0.08,
            "pivot_lift": 0.33,
            "use_final_pour_kinematic_policy": True,
            "final_pour_kinematic_source_axis": [0.0, 1.0, 0.0],
            "final_pour_kinematic_down_bias": 0.45,
            "final_pour_kinematic_orientation_gain": 1.0,
            "final_pour_kinematic_max_axis_angle": 0.55,
            "final_pour_kinematic_target_z_offset": 0.08,
            "final_pour_joint_deltas": {},
            "final_pour_joint_steps": 0,
        },
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", default=str(REPO_ROOT / "outputs" / "robocasa-gr1" / "adapter_search"))
    parser.add_argument("--registry-path", default=str(DEFAULT_REGISTRY))
    parser.add_argument("--tasks", choices=["pnp_pouring"], default="pnp_pouring")
    parser.add_argument("--template", choices=["pour_pivot"], default="pour_pivot")
    parser.add_argument("--max-candidates", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=900)
    parser.add_argument("--save-videos", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.tasks != "pnp_pouring" or args.template != "pour_pivot":
        raise ValueError("Only pnp_pouring/pour_pivot is implemented in the minimal adapter search")
    candidates = PNP_POUR_PIVOT_CANDIDATES[: args.max_candidates]
    output_root = Path(args.output_root)
    registry = SkillAdapterRegistry(args.registry_path)
    results: list[dict[str, Any]] = []
    for candidate in candidates:
        result = run_candidate(candidate, output_root=output_root, args=args, registry=registry)
        results.append(result)
        print(json.dumps(result, sort_keys=True))
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "adapter_search_summary.json").write_text(json.dumps(results, indent=2, sort_keys=True), encoding="utf-8")


def run_candidate(
    candidate: dict[str, Any],
    *,
    output_root: Path,
    args: argparse.Namespace,
    registry: SkillAdapterRegistry,
) -> dict[str, Any]:
    candidate_name = str(candidate["name"])
    run_id = f"{candidate_name}_{time.strftime('%Y%m%d_%H%M%S')}"
    run_dir = output_root / run_id
    override = copy.deepcopy(BASE_PNP_OVERRIDES)
    override["skill_overrides"].update(copy.deepcopy(candidate["params"]))
    override_path = run_dir / "task_spec_override.json"
    override_path.parent.mkdir(parents=True, exist_ok=True)
    override_path.write_text(json.dumps({"pnp_pouring": override}, indent=2, sort_keys=True), encoding="utf-8")
    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "run_robocasa_gr1_skill_interface_smoke.py"),
        "--output-dir",
        str(run_dir),
        "--tasks",
        "pnp_pouring",
        "--seed",
        str(args.seed),
        "--max-steps",
        str(args.max_steps),
        "--deprivileged-observation",
        "--visual-anchor-diagnostics",
        "--task-spec-override-json",
        str(override_path),
    ]
    if args.save_videos:
        cmd.append("--save-videos")
    if args.dry_run:
        return {
            "candidate": candidate_name,
            "adapter_id": adapter_id(candidate),
            "command": cmd,
            "override_path": str(override_path),
            "dry_run": True,
        }
    env = os.environ.copy()
    env.setdefault("PYTHONPATH", ".:third_party/robocasa_gr1/robocasa-gr1-tabletop-tasks:capx/third_party/robosuite")
    env.setdefault("CAPX_SAM3_SERVICE_URL", "http://127.0.0.1:8114")
    env.setdefault("MUJOCO_GL", "egl")
    env.setdefault("PYOPENGL_PLATFORM", "egl")
    proc = subprocess.run(cmd, cwd=REPO_ROOT, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    summary_path = run_dir / "pnp_pouring" / "summary.json"
    validation = {
        "run_dir": str(run_dir),
        "summary_path": str(summary_path),
        "returncode": int(proc.returncode),
        "stdout_tail": proc.stdout[-4000:],
    }
    score = -1.0
    success = False
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        result = summary.get("result", {})
        success = bool(summary.get("task_completed", False)) and float(summary.get("reward", 0.0)) > 0.0
        ball_distance = _optional_float(result.get("ball_distance_to_target"))
        cup_distance = _optional_float(result.get("cup_distance_to_target"))
        transition_count = int(summary.get("transition_count", 0) or 0)
        score = score_candidate(success=success, ball_distance=ball_distance, cup_distance=cup_distance, transition_count=transition_count)
        validation.update(
            {
                "task_completed": bool(summary.get("task_completed", False)),
                "reward": float(summary.get("reward", 0.0)),
                "transition_count": transition_count,
                "ball_distance_to_target": ball_distance,
                "cup_distance_to_target": cup_distance,
                "phase": result.get("phase"),
                "use_object_pose_place": result.get("effective_params", {}).get("use_object_pose_place"),
                "use_transport_distance_feedback": result.get("effective_params", {}).get("use_transport_distance_feedback"),
                "use_final_pour_kinematic_policy": result.get("effective_params", {}).get("use_final_pour_kinematic_policy"),
            }
        )
    record = SkillAdapterRecord(
        adapter_id=adapter_id(candidate),
        environment="robocasa",
        task="pnp_pouring",
        robot="GR1ArmsOnlyFourierHands",
        template=str(candidate["template"]),
        params=copy.deepcopy(candidate["params"]),
        score=float(score),
        success=bool(success),
        validation=validation,
        assumptions=list(candidate.get("assumptions", [])),
    )
    registry.upsert(record)
    return {
        "candidate": candidate_name,
        "adapter_id": record.adapter_id,
        "success": success,
        "score": score,
        "validation": validation,
    }


def adapter_id(candidate: dict[str, Any]) -> str:
    return f"robocasa.pnp_pouring.GR1ArmsOnlyFourierHands.{candidate['template']}.{candidate['name']}"


def score_candidate(*, success: bool, ball_distance: float | None, cup_distance: float | None, transition_count: int) -> float:
    score = 100.0 if success else 0.0
    if ball_distance is not None:
        score += max(0.0, 1.0 - ball_distance) * 10.0
    if cup_distance is not None:
        score += max(0.0, 1.0 - cup_distance) * 2.0
    score += min(max(int(transition_count), 0), 200) / 200.0
    return float(score)


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


if __name__ == "__main__":
    main()
