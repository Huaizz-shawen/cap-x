#!/usr/bin/env python3
"""Validate GR1 Fourier hand presets and geometric grasp helpers.

This is a lightweight Phase-0/Phase-1 diagnostic for RoboCasa GR1. It avoids
model calls and directly exercises the CaP-X control API in one simulator reset.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-path", default="env_configs/robocasa/gr1_robocasa_smoke.yaml")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--object", default="payload", dest="object_name")
    parser.add_argument("--arm", default="auto")
    parser.add_argument("--preset", default="power")
    parser.add_argument("--skip-presets", action="store_true")
    parser.add_argument("--skip-grasp", action="store_true")
    parser.add_argument("--diagnose-motion", action="store_true")
    parser.add_argument("--output-json", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.environ.setdefault("MUJOCO_GL", "osmesa")
    os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")

    from capx.envs.configs.instantiate import instantiate
    from capx.envs.configs.loader import DictLoader

    cfg = DictLoader.load([args.config_path])
    env = instantiate(cfg["env"])
    try:
        env.reset(seed=args.seed)
        api = env._apis["GR1RobocasaControlApi"]  # noqa: SLF001 - diagnostic script.
        result: dict[str, Any] = {
            "config_path": args.config_path,
            "seed": args.seed,
            "task_state_before": api.get_task_state(),
            "hand_spec": api.get_fourier_hand_spec(),
            "grasp_plan": api.plan_fourier_grasp(
                object_name=args.object_name,
                arm=args.arm,
                preset=args.preset,
            ),
        }
        if not args.skip_presets:
            result["preset_validation"] = api.validate_hand_presets(arm="both")
        if args.diagnose_motion:
            result["motion_response"] = api.diagnose_arm_motion_response(
                arm=result["grasp_plan"]["arm"],
            )
        if not args.skip_grasp:
            result["grasp_summary"] = api.grasp_object(
                object_name=args.object_name,
                arm=result["grasp_plan"]["arm"],
                preset=result["grasp_plan"]["preset"],
            )
        result["task_state_after"] = api.get_task_state()
    finally:
        env.close()

    payload = json.dumps(_jsonify(result), indent=2, sort_keys=True)
    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(payload + "\n", encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()
