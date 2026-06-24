#!/usr/bin/env python3
"""Local diagnostics for the Franka lift primitive and strict success signal."""

from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np

os.environ.setdefault("MUJOCO_GL", "egl")


def _patches() -> None:
    from eval_capx_fair import _patch_robosuite_body_name_compat

    _patch_robosuite_body_name_compat()


def _port_ready(port: int) -> bool:
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=1.0):
            return True
    except OSError:
        return False


def _ensure_pyroki(port: int, log_path: Path, timeout_s: int) -> subprocess.Popen[str] | None:
    if _port_ready(port):
        return None
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_file = log_path.open("w", encoding="utf-8")
    proc = subprocess.Popen(
        [
            sys.executable,
            "-c",
            (
                "from capx.serving.launch_pyroki_server import main; "
                f"main(port={port}, host='127.0.0.1')"
            ),
        ],
        stdout=log_file,
        stderr=subprocess.STDOUT,
        text=True,
    )
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if _port_ready(port):
            return proc
        if proc.poll() is not None:
            raise RuntimeError(f"PyRoKi server exited early, see {log_path}")
        time.sleep(1)
    proc.terminate()
    raise TimeoutError(f"PyRoKi server did not become ready on port {port}, see {log_path}")


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _state(env: Any, label: str) -> dict[str, Any]:
    low = env.low_level_env
    rs = low.robosuite_env
    sim = rs.sim
    obs = low.get_observation()
    cube_world = np.asarray(sim.data.body_xpos[rs.cube_body_id], dtype=float).copy()
    table_height = float(rs.model.mujoco_arena.table_offset[2])
    eef_site_id = rs.robots[0].eef_site_id
    if isinstance(eef_site_id, dict):
        eef_site_id = next(iter(eef_site_id.values()))
    if isinstance(eef_site_id, (list, tuple)):
        eef_site_id = eef_site_id[0]
    eef_site = np.asarray(sim.data.site_xpos[int(eef_site_id)], dtype=float).copy()
    gripper_body = np.asarray(sim.data.xpos[low.gripper_link_idx], dtype=float).copy()
    robot_cartesian = np.asarray(obs.get("robot_cartesian_pos", []), dtype=float)
    cube_robot = np.asarray(obs["cube_poses"]["primary"][:3], dtype=float).copy()
    return {
        "label": label,
        "sim_steps": int(getattr(low, "_sim_step_count", -1)),
        "reward": float(low.compute_reward()),
        "success": bool(low.task_completed()),
        "cube_world": cube_world,
        "cube_robot": cube_robot,
        "cube_height_margin": float(cube_world[2] - table_height),
        "table_height": table_height,
        "eef_site": eef_site,
        "gripper_body": gripper_body,
        "robot_cartesian_pos": robot_cartesian[:3] if robot_cartesian.size >= 3 else None,
        "robot_gripper_fraction": (
            float(robot_cartesian[-1]) if robot_cartesian.size >= 1 else None
        ),
        "eef_to_cube": float(np.linalg.norm(eef_site - cube_world)),
        "gripper_body_to_cube": float(np.linalg.norm(gripper_body - cube_world)),
        "qpos_head": np.asarray(sim.data.qpos[:9], dtype=float).copy(),
    }


def _run_sequence(
    env: Any,
    *,
    seed: int,
    name: str,
    quat_wxyz: np.ndarray,
    tcp_offset: np.ndarray,
    target_z_delta: float,
    lift: float,
    z_approach: float,
    hold_steps: int,
    home_first: bool,
) -> dict[str, Any]:
    obs, _ = env.reset(seed=seed)
    api = next(iter(env._apis.values()))
    api._TCP_OFFSET = np.asarray(tcp_offset, dtype=np.float64)

    states: list[dict[str, Any]] = [_state(env, "reset")]
    grasp_pos = np.asarray(obs["cube_poses"]["primary"][:3], dtype=np.float64)
    grasp_pos[2] += target_z_delta
    quat_wxyz = np.asarray(quat_wxyz, dtype=np.float64)
    states.append(
        {
            "label": "target",
            "grasp_pos": grasp_pos,
            "quat_wxyz": quat_wxyz,
            "tcp_offset": np.asarray(tcp_offset, dtype=float),
            "target_z_delta": target_z_delta,
            "lift": lift,
            "z_approach": z_approach,
        }
    )

    if home_first:
        api.home_pose()
        states.append(_state(env, "after_home"))
    api.open_gripper()
    states.append(_state(env, "after_open"))
    api.goto_pose(grasp_pos, quat_wxyz, z_approach=z_approach)
    states.append(_state(env, "after_goto_grasp"))
    api.close_gripper()
    states.append(_state(env, "after_close"))
    for _ in range(hold_steps):
        env.low_level_env._step_once()
    states.append(_state(env, "after_hold_closed"))
    lift_pos = grasp_pos + np.array([0.0, 0.0, lift], dtype=np.float64)
    api.goto_pose(lift_pos, quat_wxyz)
    states.append(_state(env, "after_lift"))
    for _ in range(hold_steps):
        env.low_level_env._step_once()
    states.append(_state(env, "after_hold_lift"))

    final = states[-1]
    initial = states[0]
    return {
        "name": name,
        "seed": seed,
        "terminated": bool(final["success"]),
        "reward": float(final["reward"]),
        "cube_world_z_delta": float(final["cube_world"][2] - initial["cube_world"][2]),
        "cube_height_margin": float(final["cube_height_margin"]),
        "min_eef_to_cube": min(
            float(s["eef_to_cube"]) for s in states if "eef_to_cube" in s
        ),
        "home_first": home_first,
        "states": states,
    }


def _cases() -> list[dict[str, Any]]:
    quats = {
        "q_0010": [0.0, 0.0, 1.0, 0.0],
        "q_0100": [0.0, 1.0, 0.0, 0.0],
        "q_1000": [1.0, 0.0, 0.0, 0.0],
        "q_0001": [0.0, 0.0, 0.0, 1.0],
    }
    offsets = {
        "tcp_neg": [0.0, 0.0, -0.107],
        "tcp_zero": [0.0, 0.0, 0.0],
        "tcp_pos": [0.0, 0.0, 0.107],
    }
    cases: list[dict[str, Any]] = []
    for q_name, quat in quats.items():
        for off_name, tcp_offset in offsets.items():
            for dz in (0.0, 0.025, -0.025):
                for home_first in (False, True):
                    prefix = "home" if home_first else "nohome"
                    cases.append(
                        {
                            "name": f"{prefix}_{q_name}_{off_name}_dz{dz:+.3f}",
                            "quat_wxyz": quat,
                            "tcp_offset": tcp_offset,
                            "target_z_delta": dz,
                            "home_first": home_first,
                        }
                    )
    return cases


def _run_env_step_success_check(env: Any, seed: int) -> dict[str, Any]:
    code = _env_step_programs()["success_home_single_hold_approach008"]
    return _run_env_step_program(env, seed, code)


def _run_env_step_program(env: Any, seed: int, code: str) -> dict[str, Any]:
    env.reset(seed=seed)
    low = env.low_level_env
    rs = low.robosuite_env
    z0 = float(rs.sim.data.body_xpos[rs.cube_body_id][2])
    obs, reward, terminated, truncated, info = env.step(code)
    z1 = float(rs.sim.data.body_xpos[rs.cube_body_id][2])
    return {
        "seed": seed,
        "reward": float(reward),
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "task_completed": bool(info.get("task_completed", False)),
        "sandbox_rc": int(info.get("sandbox_rc", -1)),
        "z_delta": z1 - z0,
        "stderr": str(info.get("stderr", "")),
        "code": code,
    }


def _env_step_programs() -> dict[str, str]:
    return {
        "success_home_single_hold_approach008": """import numpy as np

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
        "old_sweep_home_double_nohold_approach005": """import numpy as np

home_pose()
open_gripper()
grasp_pos, grasp_quat = sample_grasp_pose("red cube")
grasp_pos = np.array(grasp_pos, dtype=float)
goto_pose(grasp_pos, grasp_quat, z_approach=0.05)
goto_pose(grasp_pos, grasp_quat)
close_gripper()
lift_pos = grasp_pos + np.array([0.0, 0.0, 0.45])
goto_pose(lift_pos, grasp_quat)
""",
        "old_plus_hold": """import numpy as np

home_pose()
open_gripper()
grasp_pos, grasp_quat = sample_grasp_pose("red cube")
grasp_pos = np.array(grasp_pos, dtype=float)
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
        "single_nohold_approach008": """import numpy as np

home_pose()
open_gripper()
grasp_pos, grasp_quat = sample_grasp_pose("red cube")
goto_pose(grasp_pos, grasp_quat, z_approach=0.08)
close_gripper()
lift_pos = np.array(grasp_pos, dtype=float) + np.array([0.0, 0.0, 0.45])
goto_pose(lift_pos, grasp_quat)
""",
        "single_hold_approach005": """import numpy as np

home_pose()
open_gripper()
grasp_pos, grasp_quat = sample_grasp_pose("red cube")
goto_pose(grasp_pos, grasp_quat, z_approach=0.05)
close_gripper()
for _ in range(80):
    env._step_once()
lift_pos = np.array(grasp_pos, dtype=float) + np.array([0.0, 0.0, 0.45])
goto_pose(lift_pos, grasp_quat)
for _ in range(80):
    env._step_once()
""",
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="diagnostics/lift_primitive_local")
    parser.add_argument("--seed", type=int, default=60000)
    parser.add_argument("--max-cases", type=int, default=0, help="0 means all cases.")
    parser.add_argument("--lift", type=float, default=0.45)
    parser.add_argument("--z-approach", type=float, default=0.08)
    parser.add_argument("--hold-steps", type=int, default=80)
    parser.add_argument("--skip-env-step-check", action="store_true")
    parser.add_argument("--run-program-ablation", action="store_true")
    parser.add_argument("--pyroki-port", type=int, default=8116)
    parser.add_argument("--pyroki-timeout-s", type=int, default=240)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    proc = _ensure_pyroki(
        args.pyroki_port,
        output_dir / "pyroki.log",
        args.pyroki_timeout_s,
    )
    _patches()

    from capx.envs.tasks import get_config, get_exec_env

    cfg = get_config("franka_lift_code_env")
    cfg.enable_render = False
    env = get_exec_env("franka_lift_code_env")(cfg)
    cases = _cases()
    if args.max_cases > 0:
        cases = cases[: args.max_cases]

    rows: list[dict[str, Any]] = []
    try:
        if not args.skip_env_step_check:
            env_step = _run_env_step_success_check(env, args.seed)
            (output_dir / "env_step_success_check.json").write_text(
                json.dumps(_jsonable(env_step), indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            print(json.dumps({"env_step_success_check": _jsonable(env_step)}, ensure_ascii=False))

        if args.run_program_ablation:
            program_rows = []
            for name, code in _env_step_programs().items():
                row = _run_env_step_program(env, args.seed, code)
                row["name"] = name
                program_rows.append(row)
                print(
                    json.dumps(
                        {
                            "program_ablation": name,
                            "terminated": row["terminated"],
                            "reward": row["reward"],
                            "z_delta": row["z_delta"],
                            "sandbox_rc": row["sandbox_rc"],
                            "stderr": row["stderr"][:200],
                        },
                        ensure_ascii=False,
                    )
                )
            (output_dir / "program_ablation.json").write_text(
                json.dumps(_jsonable(program_rows), indent=2, ensure_ascii=False),
                encoding="utf-8",
            )

        records_path = output_dir / "records.jsonl"
        with records_path.open("w", encoding="utf-8") as f:
            for case in cases:
                started = time.time()
                try:
                    row = _run_sequence(
                        env,
                        seed=args.seed,
                        name=case["name"],
                        quat_wxyz=np.asarray(case["quat_wxyz"], dtype=float),
                        tcp_offset=np.asarray(case["tcp_offset"], dtype=float),
                        target_z_delta=float(case["target_z_delta"]),
                        lift=args.lift,
                        z_approach=args.z_approach,
                        hold_steps=args.hold_steps,
                        home_first=bool(case["home_first"]),
                    )
                    row["error"] = ""
                except Exception as exc:  # noqa: BLE001
                    row = {
                        "name": case["name"],
                        "seed": args.seed,
                        "terminated": False,
                        "reward": 0.0,
                        "cube_world_z_delta": 0.0,
                        "cube_height_margin": 0.0,
                        "min_eef_to_cube": None,
                        "states": [],
                        "error": repr(exc),
                        "traceback": traceback.format_exc(),
                        "home_first": bool(case["home_first"]),
                    }
                else:
                    row["traceback"] = ""
                row["seconds"] = time.time() - started
                rows.append(row)
                f.write(json.dumps(_jsonable(row), ensure_ascii=False) + "\n")
                f.flush()
                print(
                    json.dumps(
                        _jsonable(
                            {
                                "name": row["name"],
                                "terminated": row["terminated"],
                                "reward": row["reward"],
                                "cube_world_z_delta": row["cube_world_z_delta"],
                                "cube_height_margin": row["cube_height_margin"],
                                "min_eef_to_cube": row["min_eef_to_cube"],
                                "error": row["error"],
                            }
                        ),
                        ensure_ascii=False,
                    )
                )
    finally:
        env.close()
        if proc is not None:
            proc.terminate()

    successes = [r for r in rows if r["terminated"]]
    summary = {
        "seed": args.seed,
        "num_cases": len(rows),
        "success_count": len(successes),
        "best_by_z_delta": sorted(
            [
                {
                    "name": r["name"],
                    "terminated": r["terminated"],
                    "reward": r["reward"],
                    "cube_world_z_delta": r["cube_world_z_delta"],
                    "cube_height_margin": r["cube_height_margin"],
                    "min_eef_to_cube": r["min_eef_to_cube"],
                    "error": r["error"],
                }
                for r in rows
            ],
            key=lambda x: (bool(x["terminated"]), float(x["cube_world_z_delta"])),
            reverse=True,
        )[:10],
    }
    (output_dir / "summary.json").write_text(
        json.dumps(_jsonable(summary), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(_jsonable(summary), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
