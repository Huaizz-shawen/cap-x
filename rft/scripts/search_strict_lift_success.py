#!/usr/bin/env python3
"""Search for strict terminated=True Franka lift programs."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from eval_capx_fair import _jsonable, _patch_robosuite_body_name_compat


@dataclass(frozen=True)
class ProgramTemplate:
    name: str
    code: str
    uses_env: bool = False


def _templates() -> list[ProgramTemplate]:
    api_templates: list[ProgramTemplate] = [
        ProgramTemplate(
            name="strict_home_single_hold_approach0.08_lift0.45",
            code="""import numpy as np

home_pose()
open_gripper()
grasp_pos, grasp_quat = sample_grasp_pose("red cube")
goto_pose(grasp_pos, grasp_quat, z_approach=0.08)
close_gripper()
for _ in range(80):
    env._step_once()
lift_pos = np.array(grasp_pos, dtype=float) + np.array([0.0, 0.0, 0.45])
goto_pose(lift_pos, grasp_quat)
for _ in range(80):
    env._step_once()
""",
            uses_env=True,
        ),
        ProgramTemplate(
            name="strict_nohome_single_hold_approach0.08_lift0.45",
            code="""import numpy as np

open_gripper()
grasp_pos, grasp_quat = sample_grasp_pose("red cube")
goto_pose(grasp_pos, grasp_quat, z_approach=0.08)
close_gripper()
for _ in range(80):
    env._step_once()
lift_pos = np.array(grasp_pos, dtype=float) + np.array([0.0, 0.0, 0.45])
goto_pose(lift_pos, grasp_quat)
for _ in range(80):
    env._step_once()
""",
            uses_env=True,
        ),
    ]
    for approach in (0.05, 0.10, 0.15):
        for pre_delta in (0.0, -0.01, -0.02, -0.04):
            for lift in (0.10, 0.20, 0.30, 0.45):
                api_templates.append(
                    ProgramTemplate(
                        name=f"api_approach{approach:.2f}_predz{pre_delta:+.2f}_lift{lift:.2f}",
                        code=f"""import numpy as np

home_pose()
open_gripper()
grasp_pos, grasp_quat = sample_grasp_pose("red cube")
grasp_pos = np.array(grasp_pos, dtype=float)
grasp_pos[2] += {pre_delta!r}
goto_pose(grasp_pos, grasp_quat, z_approach={approach!r})
goto_pose(grasp_pos, grasp_quat)
close_gripper()
lift_pos = grasp_pos + np.array([0.0, 0.0, {lift!r}])
goto_pose(lift_pos, grasp_quat)
""",
                    )
                )

    api_templates.extend(
        [
            ProgramTemplate(
                name="api_oracle",
                code="""import numpy as np

grasp_pos, grasp_quat = sample_grasp_pose("red cube")
open_gripper()
goto_pose(grasp_pos, grasp_quat, z_approach=0.1)
goto_pose(grasp_pos, grasp_quat)
close_gripper()
lift_pos = grasp_pos + np.array([0.0, 0.0, 0.1])
goto_pose(lift_pos, grasp_quat)
""",
            ),
            ProgramTemplate(
                name="api_hold_close_then_lift_high",
                code="""import numpy as np

home_pose()
open_gripper()
grasp_pos, grasp_quat = sample_grasp_pose("red cube")
goto_pose(grasp_pos, grasp_quat, z_approach=0.05)
goto_pose(grasp_pos, grasp_quat)
close_gripper()
for _ in range(80):
    env._step_once()
lift_pos = grasp_pos + np.array([0.0, 0.0, 0.45])
goto_pose(lift_pos, grasp_quat)
for _ in range(80):
    env._step_once()
""",
                uses_env=True,
            ),
            ProgramTemplate(
                name="env_joint_hold_after_oracle",
                code="""import numpy as np

home_pose()
open_gripper()
grasp_pos, grasp_quat = sample_grasp_pose("red cube")
goto_pose(grasp_pos, grasp_quat, z_approach=0.05)
goto_pose(grasp_pos, grasp_quat)
close_gripper()
for _ in range(160):
    env._step_once()
lift_pos = np.array(grasp_pos, dtype=float)
lift_pos[2] += 0.35
goto_pose(lift_pos, grasp_quat)
env._set_gripper(0.0)
for _ in range(160):
    env._step_once()
""",
                uses_env=True,
            ),
        ]
    )
    return api_templates


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--data-source", default="franka_lift_code_env")
    parser.add_argument("--seed-base", type=int, default=60000)
    parser.add_argument("--num-seeds", type=int, default=20)
    parser.add_argument("--max-templates", type=int, default=0, help="0 means all templates.")
    parser.add_argument("--stop-after-successes", type=int, default=5)
    return parser.parse_args()


def _cube_z(obs: dict[str, Any]) -> float | None:
    try:
        return float(obs["cube_pos"][2])
    except Exception:
        pass
    try:
        return float(obs["cube_poses"]["primary"][2])
    except Exception:
        return None


def _run_one(env: Any, seed: int, template: ProgramTemplate) -> dict[str, Any]:
    start = time.time()
    obs0, _ = env.reset(seed=seed)
    initial_z = _cube_z(obs0)
    obs, raw_reward, terminated, truncated, info = env.step(template.code)
    final_z = _cube_z(obs)
    task_completed = info.get("task_completed")
    return {
        "template": template.name,
        "uses_env": template.uses_env,
        "seed": seed,
        "raw_reward": float(raw_reward),
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "task_completed": bool(task_completed) if task_completed is not None else None,
        "sandbox_rc": int(info.get("sandbox_rc", -1)),
        "stderr": str(info.get("stderr", "")),
        "initial_cube_z": initial_z,
        "final_cube_z": final_z,
        "delta_cube_z": None if initial_z is None or final_z is None else final_z - initial_z,
        "sim_steps": int(getattr(env.low_level_env, "_sim_step_count", -1)),
        "eval_seconds": time.time() - start,
        "code": template.code,
    }


def main() -> None:
    args = _parse_args()
    _patch_robosuite_body_name_compat()

    from capx.envs.tasks import get_config, get_exec_env

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    records_path = output_dir / "records.jsonl"

    templates = _templates()
    if args.max_templates > 0:
        templates = templates[: args.max_templates]
    seeds = [args.seed_base + idx for idx in range(args.num_seeds)]

    cfg = get_config(args.data_source)
    if cfg.privileged:
        cfg.enable_render = False
    env = get_exec_env(args.data_source)(cfg)

    records: list[dict[str, Any]] = []
    success_count = 0
    try:
        with records_path.open("w", encoding="utf-8") as f:
            for template in templates:
                for seed in seeds:
                    record = _run_one(env, seed, template)
                    records.append(record)
                    f.write(json.dumps(_jsonable(record), ensure_ascii=False) + "\n")
                    f.flush()
                    print(
                        json.dumps(
                            {
                                "template": record["template"],
                                "seed": record["seed"],
                                "terminated": record["terminated"],
                                "raw_reward": record["raw_reward"],
                                "delta_cube_z": record["delta_cube_z"],
                                "sandbox_rc": record["sandbox_rc"],
                            },
                            ensure_ascii=False,
                        )
                    )
                    if record["terminated"]:
                        success_count += 1
                        if args.stop_after_successes > 0 and success_count >= args.stop_after_successes:
                            raise StopIteration
    except StopIteration:
        pass
    finally:
        env.close()

    by_template: dict[str, dict[str, Any]] = {}
    for template in templates:
        rows = [r for r in records if r["template"] == template.name]
        if not rows:
            continue
        by_template[template.name] = {
            "num_records": len(rows),
            "success_count": sum(1 for r in rows if r["terminated"]),
            "max_raw_reward": max(float(r["raw_reward"]) for r in rows),
            "max_delta_cube_z": max(
                float(r["delta_cube_z"] or 0.0) for r in rows
            ),
        }

    summary = {
        "data_source": args.data_source,
        "seed_base": args.seed_base,
        "num_seeds": args.num_seeds,
        "num_templates": len(templates),
        "num_records": len(records),
        "success_count": sum(1 for r in records if r["terminated"]),
        "success_rate": (
            sum(1 for r in records if r["terminated"]) / len(records) if records else 0.0
        ),
        "max_raw_reward": max((float(r["raw_reward"]) for r in records), default=0.0),
        "max_delta_cube_z": max((float(r["delta_cube_z"] or 0.0) for r in records), default=0.0),
        "by_template": by_template,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(_jsonable(summary), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(_jsonable(summary), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
