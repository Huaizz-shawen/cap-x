#!/usr/bin/env python3
"""Inspect RoboCasa task/robot compatibility for Fourier GR1.

Run this inside the Inspire notebook or reference RoboCasa repo environment:

    python scripts/inspect_robocasa_gr1.py --steps 1

The script avoids CaP-X dependencies so it can be copied into the reference
repo and used to discover the exact task, robot, camera and action-space names.
"""

from __future__ import annotations

import argparse
import inspect
import json
from typing import Any

import numpy as np


def _jsonify(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonify(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonify(v) for v in value]
    if isinstance(value, np.ndarray):
        return {
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "min": float(np.nanmin(value)) if value.size else None,
            "max": float(np.nanmax(value)) if value.size else None,
        }
    if isinstance(value, np.generic):
        return value.item()
    try:
        json.dumps(value)
        return value
    except TypeError:
        return repr(value)


DEFAULT_GR1_GYM_ID = (
    "robocasa_gr1_arms_only_fourier_hands/"
    "TwoArmTransport_GR1ArmsOnlyFourierHands_Env"
)


def _patch_robocasa_gr1_env_kwargs() -> None:
    """Filter wrapper kwargs against the selected robosuite env signature.

    The GR1 gymnasium wrapper forwards a small set of convenience kwargs into
    every robosuite task. Some tasks, e.g. TwoArmTransport, do not accept all of
    them in the current RoboCasa GR1 repo. Filtering here keeps the smoke tool
    useful while avoiding a persistent third-party source edit.
    """
    try:
        import robocasa.utils.gym_utils.gymnasium_basic as gym_basic
        import robocasa.utils.gym_utils.gymnasium_groot as gym_groot  # noqa: F401
        import robosuite
        from robosuite.environments.base import REGISTERED_ENVS
    except Exception:
        return

    if getattr(robosuite.make, "_capx_filters_kwargs", False):
        return

    # Patch the helper by wrapping robosuite.make, because the original helper
    # constructs env_kwargs internally before calling robosuite.make.
    original_make = robosuite.make

    def make_compat(*args, **kwargs):
        env_name = kwargs.get("env_name")
        if env_name in REGISTERED_ENVS:
            signature = inspect.signature(REGISTERED_ENVS[env_name].__init__)
            if not any(param.kind == inspect.Parameter.VAR_KEYWORD for param in signature.parameters.values()):
                filtered = {"env_name": env_name}
                filtered.update({key: value for key, value in kwargs.items() if key in signature.parameters})
                kwargs = filtered
        return original_make(*args, **kwargs)

    make_compat._capx_filters_kwargs = True  # type: ignore[attr-defined]
    robosuite.make = make_compat


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task",
        default=DEFAULT_GR1_GYM_ID,
        help="RoboCasa task/env name or gymnasium env id",
    )
    parser.add_argument("--robot", default="GR1", help="robosuite robot name")
    parser.add_argument(
        "--camera",
        action="append",
        default=None,
        help="Camera name; can be repeated",
    )
    parser.add_argument("--width", type=int, default=224)
    parser.add_argument("--height", type=int, default=224)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--action-scale", type=float, default=0.01)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.camera is None:
        args.camera = ["robot0_eye_in_left_hand", "robot0_eye_in_right_hand"]

    import robocasa  # noqa: F401  registers RoboCasa envs
    _patch_robocasa_gr1_env_kwargs()

    if "/" in args.task:
        import gymnasium as gym
        from robocasa.utils.gym_utils import GrootRoboCasaEnv  # noqa: F401

        env = gym.make(args.task, enable_render=True)
    else:
        import robosuite
        from robosuite.controllers import load_composite_controller_config

        controller_config = load_composite_controller_config(controller=None, robot=args.robot)
        env = robosuite.make(
            env_name=args.task,
            robots=args.robot,
            controller_configs=controller_config,
            camera_names=args.camera,
            camera_widths=args.width,
            camera_heights=args.height,
            has_renderer=False,
            has_offscreen_renderer=True,
            ignore_done=True,
            use_object_obs=True,
            use_camera_obs=True,
            camera_depths=False,
            seed=args.seed,
            translucent_robot=False,
            render_camera=args.camera[0],
        )

    try:
        try:
            retval = env.reset(seed=args.seed)
        except TypeError:
            retval = env.reset()
        obs = retval[0] if isinstance(retval, tuple) and len(retval) == 2 else retval
        action_space = getattr(env, "action_space", None)
        action_shape = list(action_space.shape) if action_space is not None and hasattr(action_space, "shape") else None
        action_dim = int(np.prod(action_space.shape)) if action_shape else int(getattr(env, "action_dim", 7))

        step_summaries = []
        for step_idx in range(max(0, args.steps)):
            if action_space is not None and hasattr(action_space, "sample") and not action_shape:
                action = action_space.sample()
            else:
                action = np.random.uniform(-args.action_scale, args.action_scale, size=(action_dim,))
            step_retval = env.step(action)
            if len(step_retval) == 5:
                obs, reward, done, truncated, info = step_retval
            else:
                obs, reward, done, info = step_retval
                truncated = False
            step_summaries.append(
                {
                    "step": step_idx,
                    "reward": float(reward),
                    "done": bool(done),
                    "truncated": bool(truncated),
                    "info": _jsonify(info),
                }
            )

        summary = {
            "task": args.task,
            "robot": args.robot,
            "cameras": args.camera,
            "action_shape": action_shape,
            "action_space": repr(action_space),
            "obs_keys": sorted(obs.keys()),
            "obs_summary": {key: _jsonify(value) for key, value in obs.items()},
            "steps": step_summaries,
        }
        print(json.dumps(summary, indent=2, sort_keys=True))
    finally:
        if hasattr(env, "close"):
            env.close()


if __name__ == "__main__":
    main()
