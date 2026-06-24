#!/usr/bin/env python3
"""Validate a registered RoboCasa GR1 skill adapter across multiple seeds."""

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
from scripts.search_robocasa_gr1_skill_adapters import BASE_PNP_OVERRIDES, score_candidate


DEFAULT_REGISTRY = REPO_ROOT / "outputs" / "robocasa-gr1" / "skill_adapters" / "registry.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry-path", default=str(DEFAULT_REGISTRY))
    parser.add_argument("--adapter-id", default=None)
    parser.add_argument("--environment", default="robocasa")
    parser.add_argument("--task", choices=["pnp_pouring"], default="pnp_pouring")
    parser.add_argument("--robot", default="GR1ArmsOnlyFourierHands")
    parser.add_argument("--template", choices=["pour_pivot"], default="pour_pivot")
    parser.add_argument("--seeds", default="0,1,2,3,4")
    parser.add_argument("--output-root", default=str(REPO_ROOT / "outputs" / "robocasa-gr1" / "adapter_validation"))
    parser.add_argument("--max-steps", type=int, default=900)
    parser.add_argument("--save-videos", action="store_true")
    parser.add_argument("--update-registry", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    registry = SkillAdapterRegistry(args.registry_path)
    adapter = _select_adapter(registry, args)
    seeds = _parse_seeds(args.seeds)
    run_root = Path(args.output_root) / f"{_safe_name(adapter['adapter_id'])}_{time.strftime('%Y%m%d_%H%M%S')}"
    aggregate = validate_adapter(adapter, seeds=seeds, run_root=run_root, args=args)
    run_root.mkdir(parents=True, exist_ok=True)
    (run_root / "adapter_validation_summary.json").write_text(json.dumps(aggregate, indent=2, sort_keys=True), encoding="utf-8")
    if args.update_registry and not args.dry_run:
        _update_registry(registry, adapter, aggregate)
    print(json.dumps(aggregate, indent=2, sort_keys=True))


def validate_adapter(adapter: dict[str, Any], *, seeds: list[int], run_root: Path, args: argparse.Namespace) -> dict[str, Any]:
    per_seed: list[dict[str, Any]] = []
    for seed in seeds:
        per_seed.append(run_one_seed(adapter, seed=seed, run_root=run_root, args=args))
    successes = [item for item in per_seed if item.get("success")]
    ball_distances = [_float(item.get("ball_distance_to_target")) for item in per_seed]
    cup_distances = [_float(item.get("cup_distance_to_target")) for item in per_seed]
    aggregate: dict[str, Any] = {
        "adapter_id": adapter["adapter_id"],
        "environment": adapter.get("environment"),
        "task": adapter.get("task"),
        "robot": adapter.get("robot"),
        "template": adapter.get("template"),
        "seeds": seeds,
        "run_root": str(run_root),
        "success_count": len(successes),
        "trial_count": len(per_seed),
        "success_rate": len(successes) / len(per_seed) if per_seed else 0.0,
        "mean_score": _mean([_float(item.get("score")) for item in per_seed]),
        "mean_ball_distance_to_target": _mean(ball_distances),
        "max_ball_distance_to_target": _max(ball_distances),
        "mean_cup_distance_to_target": _mean(cup_distances),
        "max_cup_distance_to_target": _max(cup_distances),
        "per_seed": per_seed,
    }
    return aggregate


def run_one_seed(adapter: dict[str, Any], *, seed: int, run_root: Path, args: argparse.Namespace) -> dict[str, Any]:
    run_dir = run_root / f"seed_{seed:03d}"
    override = copy.deepcopy(BASE_PNP_OVERRIDES)
    override["skill_overrides"].update(copy.deepcopy(adapter.get("params", {})))
    override_path = run_dir / "task_spec_override.json"
    override_path.parent.mkdir(parents=True, exist_ok=True)
    override_path.write_text(json.dumps({args.task: override}, indent=2, sort_keys=True), encoding="utf-8")
    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "run_robocasa_gr1_skill_interface_smoke.py"),
        "--output-dir",
        str(run_dir),
        "--tasks",
        args.task,
        "--seed",
        str(seed),
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
        return {"seed": seed, "command": cmd, "override_path": str(override_path), "dry_run": True}
    env = os.environ.copy()
    env.setdefault("PYTHONPATH", ".:third_party/robocasa_gr1/robocasa-gr1-tabletop-tasks:capx/third_party/robosuite")
    env.setdefault("CAPX_SAM3_SERVICE_URL", "http://127.0.0.1:8114")
    env.setdefault("MUJOCO_GL", "egl")
    env.setdefault("PYOPENGL_PLATFORM", "egl")
    proc = subprocess.run(cmd, cwd=REPO_ROOT, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    summary_path = run_dir / args.task / "summary.json"
    result: dict[str, Any] = {
        "seed": seed,
        "run_dir": str(run_dir),
        "summary_path": str(summary_path),
        "returncode": int(proc.returncode),
        "stdout_tail": proc.stdout[-4000:],
        "success": False,
        "score": -1.0,
    }
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        task_result = summary.get("result", {})
        success = bool(summary.get("task_completed", False)) and float(summary.get("reward", 0.0)) > 0.0
        ball_distance = _optional_float(task_result.get("ball_distance_to_target"))
        cup_distance = _optional_float(task_result.get("cup_distance_to_target"))
        transition_count = int(summary.get("transition_count", 0) or 0)
        result.update(
            {
                "success": success,
                "task_completed": bool(summary.get("task_completed", False)),
                "reward": float(summary.get("reward", 0.0)),
                "transition_count": transition_count,
                "ball_distance_to_target": ball_distance,
                "cup_distance_to_target": cup_distance,
                "phase": task_result.get("phase"),
                "score": score_candidate(
                    success=success,
                    ball_distance=ball_distance,
                    cup_distance=cup_distance,
                    transition_count=transition_count,
                ),
            }
        )
    return result


def _select_adapter(registry: SkillAdapterRegistry, args: argparse.Namespace) -> dict[str, Any]:
    data = registry.load()
    if args.adapter_id:
        for item in data["adapters"]:
            if isinstance(item, dict) and item.get("adapter_id") == args.adapter_id:
                return item
        raise ValueError(f"Adapter not found in registry: {args.adapter_id}")
    best = registry.best(
        environment=args.environment,
        task=args.task,
        robot=args.robot,
        template=args.template,
        require_success=True,
    )
    if best is None:
        raise ValueError("No successful adapter found; pass --adapter-id or run search first")
    return best


def _update_registry(registry: SkillAdapterRegistry, adapter: dict[str, Any], aggregate: dict[str, Any]) -> None:
    record = SkillAdapterRecord(
        adapter_id=str(adapter["adapter_id"]),
        environment=str(adapter.get("environment", "")),
        task=str(adapter.get("task", "")),
        robot=str(adapter.get("robot", "")),
        template=str(adapter.get("template", "")),
        params=copy.deepcopy(adapter.get("params", {})),
        score=float(aggregate.get("mean_score") or 0.0),
        success=bool(aggregate.get("success_rate") == 1.0),
        validation={"multi_seed": aggregate},
        assumptions=list(adapter.get("assumptions", [])),
    )
    registry.upsert(record)


def _parse_seeds(value: str) -> list[int]:
    seeds = [int(part.strip()) for part in value.split(",") if part.strip()]
    if not seeds:
        raise ValueError("--seeds must contain at least one integer")
    return seeds


def _safe_name(value: str) -> str:
    return "".join(char if char.isalnum() or char in "._-" else "_" for char in value)


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _float(value: Any) -> float | None:
    return _optional_float(value)


def _mean(values: list[float | None]) -> float | None:
    valid = [value for value in values if value is not None]
    if not valid:
        return None
    return float(sum(valid) / len(valid))


def _max(values: list[float | None]) -> float | None:
    valid = [value for value in values if value is not None]
    if not valid:
        return None
    return float(max(valid))


if __name__ == "__main__":
    main()
