"""RoboCasa low-level adapter for Fourier GR1 experiments.

This module intentionally keeps the RoboCasa-specific surface thin. RoboCasa
can expose either robosuite task names or gymnasium ids, so the adapter creates
the underlying simulator through the matching backend and exposes a small CaP-X
``BaseEnv`` facade for smoke tests and early data collection.
"""

from __future__ import annotations

import copy
import inspect
import json
import os
import sys
import time
import types
from typing import Any

import numpy as np
from robosuite.utils.camera_utils import get_real_depth_map

from capx.envs.base import BaseEnv
from capx.envs.transition_dataset import (
    append_transition,
    create_transition_dataset,
    set_initial_observation,
)

os.environ.setdefault("MUJOCO_GL", "egl")


class GR1RobocasaLowLevel(BaseEnv):
    """Minimal RoboCasa/GR1 low-level environment.

    Parameters are deliberately YAML-friendly so the exact task, robot name and
    camera list can be adjusted once the Inspire reference repo is available.
    """

    def __init__(
        self,
        *,
        env_name: str = "TwoArmTransport",
        robots: str | list[str] = "GR1ArmsOnlyFourierHands",
        controller_config: str | dict[str, Any] | None = None,
        camera_names: list[str] | tuple[str, ...] | None = None,
        camera_widths: int | list[int] = 224,
        camera_heights: int | list[int] = 224,
        render_camera: str | None = None,
        camera_fovy_overrides: dict[str, float] | None = None,
        max_steps: int = 300,
        seed: int | None = None,
        has_renderer: bool = False,
        has_offscreen_renderer: bool = True,
        use_object_obs: bool = True,
        use_camera_obs: bool = True,
        camera_depths: bool = False,
        ignore_done: bool = True,
        record_transition_fps: float | None = 20.0,
        record_transition_depth: bool = False,
        env_kwargs: dict[str, Any] | None = None,
        privileged: bool = False,
        enable_render: bool = False,
        viser_debug: bool = False,
    ) -> None:
        super().__init__()
        del privileged, enable_render, viser_debug

        self.env_name = env_name
        self.robots = robots
        self.controller_config = controller_config
        self.camera_names = list(camera_names or ["robot0_eye_in_left_hand", "robot0_eye_in_right_hand"])
        self.camera_widths = camera_widths
        self.camera_heights = camera_heights
        self.render_camera = render_camera or self.camera_names[0]
        self.camera_fovy_overrides = dict(camera_fovy_overrides or {})
        self.max_steps = int(max_steps)
        self.seed = seed
        self.has_renderer = has_renderer
        self.has_offscreen_renderer = has_offscreen_renderer
        self.use_object_obs = use_object_obs
        self.use_camera_obs = use_camera_obs
        self.camera_depths = camera_depths
        self.ignore_done = ignore_done
        self.env_kwargs = dict(env_kwargs or {})

        self._record_transition_fps = (
            None
            if record_transition_fps is None or float(record_transition_fps) <= 0.0
            else float(record_transition_fps)
        )
        self._record_transition_interval_s = (
            None if self._record_transition_fps is None else 1.0 / self._record_transition_fps
        )
        self._record_transition_depth = bool(record_transition_depth)
        self._transition_dataset: dict[str, Any] | None = None
        self._last_recorded_transition_timestamp_s: float | None = None

        self._step_count = 0
        self._sim_step_count = 0
        self._start_time = time.monotonic()
        self._current_obs: dict[str, Any] | None = None
        self._current_reward: float | None = None
        self._current_done: bool | None = None
        self._current_truncated: bool | None = None
        self._last_action: np.ndarray | None = None
        self._reset_state: dict[str, Any] | None = None

        self._record_frames = False
        self._frame_buffer: list[np.ndarray] = []
        self._wrist_frame_buffer: list[np.ndarray] = []
        self._record_wrist_camera = False

        self.backend = "gymnasium" if "/" in self.env_name else "robosuite"
        self.robosuite_env = self._create_env()
        self._apply_camera_overrides()

    def _load_controller_config(self) -> Any:
        if self.controller_config is not None:
            if isinstance(self.controller_config, str):
                controller = self.controller_config.strip()
                if controller == "WHOLE_BODY_MINK_IK":
                    return self._load_mink_controller_config()
                if os.path.exists(controller):
                    with open(controller, encoding="utf-8") as f:
                        return json.load(f)
                try:
                    from robosuite.controllers import load_composite_controller_config

                    return load_composite_controller_config(controller=controller, robot=self.robots)
                except Exception:
                    # Preserve the previous behavior for unknown string values so
                    # callers get the original robosuite-side error context.
                    return self.controller_config
            return self.controller_config
        try:
            from robosuite.controllers import load_composite_controller_config

            return load_composite_controller_config(controller=None, robot=self.robots)
        except Exception as exc:
            raise RuntimeError(
                "Failed to load a robosuite composite controller config for "
                f"robot={self.robots!r}. Pass controller_config explicitly if "
                "the reference RoboCasa repo uses a custom GR1 controller."
            ) from exc

    def _load_mink_controller_config(self) -> Any:
        """Load the registered Mink whole-body IK config for the active GR1 robot.

        The robosuite example controller expects an older mink import path
        (``mink.tasks.exceptions``). Current mink exposes the same exception from
        ``mink.exceptions``. Installing a narrow module alias keeps this adapter
        independent of local edits to the vendored robosuite example file.
        """
        self._install_mink_import_compat()
        import robosuite.examples.third_party_controller.mink_controller  # noqa: F401
        from robosuite.controllers import load_composite_controller_config

        config = load_composite_controller_config(controller="WHOLE_BODY_MINK_IK", robot=self.robots)
        self._tune_gr1_mink_controller_config(config)
        return config

    def _tune_gr1_mink_controller_config(self, config: dict[str, Any]) -> None:
        """Apply GR1-ArmsOnly specific Mink tuning for Cartesian hand motion."""
        robot_names = self.robots if isinstance(self.robots, (list, tuple)) else [self.robots]
        if "GR1ArmsOnlyFourierHands" not in {str(name) for name in robot_names}:
            return
        specific = config.setdefault("composite_controller_specific_configs", {})
        # Factory Mink config is tuned for full-body posture/orientation control.
        # On GR1ArmsOnly it pulls the wrist upward regardless of requested XYZ
        # direction. Initial-qpos posture plus position-only hand tasks gives a
        # stable local Cartesian action surface for early data collection.
        specific["ik_hand_ori_cost"] = 0.0
        specific["initial_qpos_as_posture_target"] = True

    @staticmethod
    def _install_mink_import_compat() -> None:
        try:
            import mink.tasks.exceptions  # noqa: F401
            return
        except ModuleNotFoundError:
            pass

        try:
            from mink.exceptions import TargetNotSet
        except Exception:
            return

        module = types.ModuleType("mink.tasks.exceptions")
        module.TargetNotSet = TargetNotSet
        sys.modules.setdefault("mink.tasks.exceptions", module)

    def _create_env(self) -> Any:
        if self.backend == "gymnasium":
            try:
                import robocasa  # noqa: F401  registers RoboCasa envs
                from robocasa.utils.gym_utils import GrootRoboCasaEnv  # noqa: F401
                import gymnasium as gym
                self._patch_gr1_gym_wrapper_kwargs()
            except Exception as exc:
                raise RuntimeError(
                    "RoboCasa gymnasium environment is not importable. Install the "
                    "Isaac-GR00T-robocasa environment or run inside the reference notebook image."
                ) from exc

            try:
                return gym.make(self.env_name, enable_render=True)
            except Exception as exc:
                raise RuntimeError(
                    "Failed to create RoboCasa gym env with gym.make("
                    f"{self.env_name!r}). Confirm the env id in the reference repo."
                ) from exc

        try:
            import robocasa  # noqa: F401  Robocasa registers tasks on import.
            import robosuite
        except Exception as exc:
            raise RuntimeError(
                "RoboCasa/robosuite is not importable. Install the RoboCasa environment "
                "or run this adapter inside the Inspire notebook image that contains it."
            ) from exc

        kwargs = {
            "env_name": self.env_name,
            "robots": self.robots,
            "controller_configs": self._load_controller_config(),
            "camera_names": self.camera_names,
            "camera_widths": self.camera_widths,
            "camera_heights": self.camera_heights,
            "has_renderer": self.has_renderer,
            "has_offscreen_renderer": self.has_offscreen_renderer,
            "ignore_done": self.ignore_done,
            "use_object_obs": self.use_object_obs,
            "use_camera_obs": self.use_camera_obs,
            "camera_depths": self.camera_depths,
            "seed": self.seed,
            "translucent_robot": False,
            "render_camera": self.render_camera,
        }
        kwargs.update(self.env_kwargs)
        try:
            self._filter_robosuite_kwargs(kwargs)
            return robosuite.make(**kwargs)
        except Exception as exc:
            raise RuntimeError(
                "Failed to create RoboCasa GR1 env with robosuite.make("
                f"env_name={self.env_name!r}, robots={self.robots!r}). "
                "Run scripts/inspect_robocasa_gr1.py in the reference repo to "
                "confirm the available task names, robot name and camera names."
            ) from exc

    @staticmethod
    def _filter_robosuite_kwargs(kwargs: dict[str, Any]) -> None:
        """Remove kwargs unsupported by the selected robosuite task class."""
        try:
            from robosuite.environments.base import REGISTERED_ENVS
        except Exception:
            return
        env_name = kwargs.get("env_name")
        if env_name not in REGISTERED_ENVS:
            return
        signature = inspect.signature(REGISTERED_ENVS[env_name].__init__)
        if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in signature.parameters.values()):
            return
        for key in list(kwargs):
            if key != "env_name" and key not in signature.parameters:
                kwargs.pop(key, None)

    def _apply_camera_overrides(self) -> None:
        """Apply debug-only camera model overrides after env creation/reset."""
        if not self.camera_fovy_overrides or self.backend != "robosuite":
            return
        model = getattr(getattr(self.robosuite_env, "sim", None), "model", None)
        if model is None:
            return
        for camera_name, fovy in self.camera_fovy_overrides.items():
            try:
                camera_id = model.camera_name2id(str(camera_name))
            except Exception:
                continue
            model.cam_fovy[camera_id] = float(fovy)

    @classmethod
    def _patch_gr1_gym_wrapper_kwargs(cls) -> None:
        """Patch RoboCasa GR1 gym wrapper to tolerate task-specific signatures."""
        try:
            import robosuite
        except Exception:
            return
        if getattr(robosuite.make, "_capx_filters_kwargs", False):
            return
        original_make = robosuite.make

        def make_compat(*args: Any, **kwargs: Any) -> Any:
            cls._filter_robosuite_kwargs(kwargs)
            return original_make(*args, **kwargs)

        make_compat._capx_filters_kwargs = True  # type: ignore[attr-defined]
        robosuite.make = make_compat

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        options = options or {}
        reset_kwargs: dict[str, Any] = {}
        effective_seed = seed if seed is not None else self.seed
        if effective_seed is not None:
            reset_kwargs["seed"] = effective_seed
        reset_kwargs.update(options)

        try:
            retval = self.robosuite_env.reset(**reset_kwargs)
        except TypeError:
            # Robosuite-style RoboCasa envs often accept seed only at creation time.
            retval = self.robosuite_env.reset()
        self._apply_camera_overrides()
        obs = retval[0] if isinstance(retval, tuple) and len(retval) == 2 else retval
        self._current_obs = self._normalize_raw_obs(obs)
        self._current_reward = None
        self._current_done = False
        self._current_truncated = False
        self._last_action = None
        self._step_count = 0
        self._sim_step_count = 0
        self._start_time = time.monotonic()

        transition_obs = self._build_transition_observation()
        self._transition_dataset = create_transition_dataset(trial=0)
        self._transition_dataset["record_transition_fps"] = self._record_transition_fps
        self._transition_dataset["trial_metadata"].update(
            {
                "benchmark": "robocasa",
                "robot": self.robots,
                "env_name": self.env_name,
                "camera_names": list(self.camera_names),
            }
        )
        set_initial_observation(self._transition_dataset, transition_obs)
        self._last_recorded_transition_timestamp_s = None

        if self._record_frames:
            self._record_frame()
        self._reset_state = self.capture_state()
        return self.get_observation(), {"raw_obs_keys": sorted(self._current_obs.keys())}

    def step(self, action: Any) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        normalized_action = self._normalize_action(action)
        step_retval = self.robosuite_env.step(normalized_action)
        if len(step_retval) == 5:
            raw_obs, reward, done, truncated, info = step_retval
        else:
            raw_obs, reward, done, info = step_retval
            truncated = False
        self._current_obs = self._normalize_raw_obs(raw_obs)
        self._current_reward = float(reward)
        self._current_done = bool(done)
        self._step_count += 1
        self._sim_step_count += 1
        self._current_truncated = bool(truncated) or bool(self._step_count >= self.max_steps)
        self._last_action = self._flatten_action(normalized_action)

        if self._record_frames:
            self._record_frame()
        self._record_transition(
            action=self._last_action,
            source="robocasa_step",
            metadata={"info": info},
        )

        return (
            self.get_observation(),
            float(reward),
            bool(done),
            self._current_truncated,
            dict(info),
        )

    def get_observation(self) -> dict[str, Any]:
        if self._current_obs is None:
            if hasattr(self.robosuite_env, "_get_observations"):
                raw_obs = self.robosuite_env._get_observations()
            else:
                raw_obs = {}
            self._current_obs = self._normalize_raw_obs(raw_obs)
        obs = dict(self._current_obs)
        obs["robot_joint_pos"] = self._extract_robot_joint_pos(obs)
        obs["robot_cartesian_pos"] = self._extract_robot_cartesian_pos(obs)
        obs["low_level_observation"] = dict(self._current_obs)
        obs["task_description"] = self.env_name
        return obs

    def compute_reward(self) -> float:
        if hasattr(self.robosuite_env, "_check_success"):
            try:
                return 1.0 if bool(self.robosuite_env._check_success()) else 0.0
            except Exception:
                pass
        return float(self._current_reward or 0.0)

    def task_completed(self) -> bool:
        if hasattr(self.robosuite_env, "_check_success"):
            try:
                return bool(self.robosuite_env._check_success())
            except Exception:
                pass
        return bool(self._current_done)

    def enable_video_capture(
        self,
        enabled: bool = True,
        *,
        clear: bool = True,
        wrist_camera: bool = False,
    ) -> None:
        self._record_frames = enabled
        self._record_wrist_camera = wrist_camera
        if clear:
            self._frame_buffer.clear()
            self._wrist_frame_buffer.clear()
        if enabled:
            self._record_frame()

    def get_video_frames(self, *, clear: bool = False) -> list[np.ndarray]:
        frames = [frame.copy() for frame in self._frame_buffer]
        if clear:
            self._frame_buffer.clear()
        return frames

    def get_video_frame_count(self) -> int:
        return len(self._frame_buffer)

    def get_video_frames_range(self, start: int, end: int) -> list[np.ndarray]:
        return [frame.copy() for frame in self._frame_buffer[start:end]]

    def get_wrist_video_frames(self, *, clear: bool = False) -> list[np.ndarray]:
        frames = [frame.copy() for frame in self._wrist_frame_buffer]
        if clear:
            self._wrist_frame_buffer.clear()
        return frames

    def get_wrist_video_frames_range(self, start: int, end: int) -> list[np.ndarray]:
        return [frame.copy() for frame in self._wrist_frame_buffer[start:end]]

    def render(self, mode: str = "rgb_array") -> np.ndarray:  # type: ignore[override]
        if mode != "rgb_array":
            raise ValueError("Only rgb_array render mode is supported")
        return self._render_camera(self.render_camera)

    def render_wrist(self) -> np.ndarray | None:
        if len(self.camera_names) < 2:
            return None
        return self._render_camera(self.camera_names[1])

    def render_camera_rgbd(self, camera_name: str | None = None) -> dict[str, Any]:
        """Render RGBD and camera calibration for one RoboCasa camera.

        The returned ``pose_mat`` maps camera-frame points from
        ``capx.utils.depth_utils.depth_to_pointcloud`` into MuJoCo world frame.
        This is intentionally low-level so GR1 APIs can build GraspNet-style
        point-cloud inputs without depending on task observations exposing depth.
        """
        if self.backend == "gymnasium":
            raise NotImplementedError("RGBD camera calibration is only implemented for robosuite backend")
        camera = str(camera_name or self.render_camera)
        height = self.camera_heights[0] if isinstance(self.camera_heights, list) else self.camera_heights
        width = self.camera_widths[0] if isinstance(self.camera_widths, list) else self.camera_widths
        rgb, depth_buffer = self.robosuite_env.sim.render(
            camera_name=camera,
            width=int(width),
            height=int(height),
            depth=True,
        )
        rgb = np.asarray(rgb, dtype=np.uint8)[::-1].copy()
        depth = get_real_depth_map(self.robosuite_env.sim, np.asarray(depth_buffer, dtype=np.float32)[::-1]).astype(np.float32)
        return {
            "camera_name": camera,
            "rgb": rgb,
            "depth": depth,
            "intrinsics": self._camera_intrinsics(camera, width=int(width), height=int(height)),
            "pose_mat": self._camera_pose_mat(camera),
        }

    def close(self) -> None:
        if hasattr(self.robosuite_env, "close"):
            self.robosuite_env.close()

    def get_transition_dataset(self) -> dict[str, Any] | None:
        return self._transition_dataset

    def _copy_state_value(self, value: Any) -> Any:
        if isinstance(value, np.ndarray):
            return value.copy()
        if isinstance(value, dict):
            return {key: self._copy_state_value(val) for key, val in value.items()}
        if isinstance(value, list):
            return [self._copy_state_value(item) for item in value]
        if isinstance(value, tuple):
            return tuple(self._copy_state_value(item) for item in value)
        return copy.deepcopy(value)

    def _get_underlying_sim(self) -> Any:
        env = self.robosuite_env
        visited: set[int] = set()
        while env is not None and id(env) not in visited:
            visited.add(id(env))
            sim = getattr(env, "sim", None)
            if sim is not None:
                return sim
            env = getattr(env, "env", None) or getattr(env, "unwrapped", None)
        raise NotImplementedError(f"{type(self).__name__} does not expose a robosuite sim")

    def capture_state(self) -> dict[str, Any]:
        sim = self._get_underlying_sim()
        return {
            "sim_state": sim.get_state().flatten().copy(),
            "current_obs": self._copy_state_value(self._current_obs),
            "current_reward": self._current_reward,
            "current_done": self._current_done,
            "current_truncated": self._current_truncated,
            "last_action": None if self._last_action is None else self._last_action.copy(),
            "step_count": int(self._step_count),
            "sim_step_count": int(self._sim_step_count),
            "last_recorded_transition_timestamp_s": self._last_recorded_transition_timestamp_s,
        }

    def restore_state(self, state: dict[str, Any]) -> None:
        sim = self._get_underlying_sim()
        sim.set_state_from_flattened(np.asarray(state["sim_state"], dtype=np.float64))
        sim.forward()
        self._current_obs = self._copy_state_value(state.get("current_obs"))
        if self._current_obs is None and hasattr(self.robosuite_env, "_get_observations"):
            self._current_obs = self._normalize_raw_obs(self.robosuite_env._get_observations())
        self._current_reward = state.get("current_reward")
        self._current_done = state.get("current_done")
        self._current_truncated = state.get("current_truncated")
        last_action = state.get("last_action")
        self._last_action = None if last_action is None else np.asarray(last_action, dtype=np.float64).copy()
        self._step_count = int(state.get("step_count", 0))
        self._sim_step_count = int(state.get("sim_step_count", 0))
        self._last_recorded_transition_timestamp_s = state.get("last_recorded_transition_timestamp_s")

    def get_reset_state(self) -> dict[str, Any] | None:
        if self._reset_state is None:
            return None
        return self._copy_state_value(self._reset_state)

    def sample_action(self, scale: float = 0.05) -> np.ndarray:
        """Return a small random action compatible with the current action space."""
        space = getattr(self.robosuite_env, "action_space", None)
        if space is not None and hasattr(space, "sample") and not hasattr(space, "shape"):
            return self._flatten_action(space.sample())
        if space is not None and hasattr(space, "shape"):
            dim = int(np.prod(space.shape))
        elif hasattr(self.robosuite_env, "action_dim"):
            dim = int(getattr(self.robosuite_env, "action_dim"))
        elif self._last_action is not None:
            dim = int(self._last_action.size)
        else:
            dim = 7
        return np.random.uniform(-float(scale), float(scale), size=(dim,)).astype(np.float64)

    def _normalize_action(self, action: Any) -> Any:
        if isinstance(action, dict):
            return {key: self._normalize_action(value) for key, value in action.items()}
        if isinstance(action, (list, tuple)):
            return np.asarray(action, dtype=np.float64)
        if isinstance(action, np.ndarray):
            return action.astype(np.float64, copy=False)
        return action

    def _flatten_action(self, action: Any) -> np.ndarray:
        if isinstance(action, dict):
            chunks = [self._flatten_action(value) for _, value in sorted(action.items())]
            if not chunks:
                return np.zeros(0, dtype=np.float64)
            return np.concatenate(chunks)
        try:
            return np.asarray(action, dtype=np.float64).reshape(-1)
        except Exception:
            return np.zeros(0, dtype=np.float64)

    def _extract_robot_joint_pos(self, obs: dict[str, Any]) -> np.ndarray:
        for key in ("robot0_joint_pos", "robot0_joint_pos_cos", "robot0_proprio-state"):
            if key in obs:
                arr = np.asarray(obs[key], dtype=np.float32).reshape(-1)
                if key == "robot0_joint_pos_cos" and "robot0_joint_pos_sin" in obs:
                    sin = np.asarray(obs["robot0_joint_pos_sin"], dtype=np.float32).reshape(-1)
                    arr = np.arctan2(sin, arr)
                return self._pad_or_trim(arr, 8)
        return np.zeros(8, dtype=np.float32)

    def _extract_robot_cartesian_pos(self, obs: dict[str, Any]) -> np.ndarray:
        pos = self._first_array(
            obs,
            ("robot0_eef_pos", "robot0_base_to_eef_pos", "robot0_left_eef_pos", "robot0_right_eef_pos"),
            size=3,
        )
        quat = self._first_array(
            obs,
            ("robot0_eef_quat", "robot0_base_to_eef_quat", "robot0_left_eef_quat", "robot0_right_eef_quat"),
            size=4,
        )
        if quat is None:
            quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        gripper = self._first_array(
            obs,
            ("robot0_gripper_qpos", "robot0_left_gripper_qpos", "robot0_right_gripper_qpos"),
            size=1,
        )
        if gripper is None:
            gripper = np.zeros(1, dtype=np.float32)
        if pos is None:
            pos = np.zeros(3, dtype=np.float32)
        return np.concatenate([pos[:3], quat[:4], gripper[:1]]).astype(np.float32)

    @staticmethod
    def _first_array(obs: dict[str, Any], keys: tuple[str, ...], size: int) -> np.ndarray | None:
        for key in keys:
            if key in obs:
                arr = np.asarray(obs[key], dtype=np.float32).reshape(-1)
                return GR1RobocasaLowLevel._pad_or_trim(arr, size)
        return None

    @staticmethod
    def _pad_or_trim(value: np.ndarray, size: int) -> np.ndarray:
        arr = np.asarray(value, dtype=np.float32).reshape(-1)
        if arr.size >= size:
            return arr[:size]
        return np.pad(arr, (0, size - arr.size), mode="constant")

    def _build_transition_observation(self) -> dict[str, Any]:
        obs = self.get_observation()
        low_level = dict(obs.get("low_level_observation", {}))
        self._add_export_aliases(low_level)
        if not self._record_transition_depth:
            for key in list(low_level):
                if key.endswith("_depth"):
                    low_level.pop(key, None)
        return {
            "low_level_observation": low_level,
            "robot_joint_pos": np.asarray(obs["robot_joint_pos"], dtype=np.float32),
            "robot_cartesian_pos": np.asarray(obs["robot_cartesian_pos"], dtype=np.float32),
        }

    def _record_transition(
        self,
        *,
        action: np.ndarray,
        source: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        if self._transition_dataset is None or self._current_obs is None:
            return
        timestamp_s = time.monotonic() - self._start_time
        if (
            self._record_transition_interval_s is not None
            and self._last_recorded_transition_timestamp_s is not None
            and (timestamp_s - self._last_recorded_transition_timestamp_s)
            < (self._record_transition_interval_s - 1e-9)
        ):
            return
        append_transition(
            self._transition_dataset,
            timestamp_s=timestamp_s,
            wall_time_s=time.time(),
            sim_step_count=int(self._sim_step_count),
            action=np.asarray(action, dtype=np.float64).copy(),
            observation=self._build_transition_observation(),
            reward=float(self._current_reward) if self._current_reward is not None else None,
            done=bool(self._current_done) if self._current_done is not None else None,
            truncated=bool(self._current_truncated),
            source=source,
            metadata=metadata,
        )
        self._last_recorded_transition_timestamp_s = timestamp_s

    def _record_frame(self) -> None:
        self._frame_buffer.append(self._render_camera(self.render_camera))
        if self._record_wrist_camera and len(self.camera_names) > 1:
            self._wrist_frame_buffer.append(self._render_camera(self.camera_names[1]))

    def _render_camera(self, camera_name: str) -> np.ndarray:
        if self.backend == "gymnasium":
            del camera_name
            frame = self.robosuite_env.render()
            return np.asarray(frame, dtype=np.uint8)

        height = self.camera_heights[0] if isinstance(self.camera_heights, list) else self.camera_heights
        width = self.camera_widths[0] if isinstance(self.camera_widths, list) else self.camera_widths
        frame = self.robosuite_env.sim.render(
            camera_name=camera_name,
            width=int(width),
            height=int(height),
            depth=False,
        )
        return np.asarray(frame, dtype=np.uint8)[::-1]

    def _camera_intrinsics(self, camera_name: str, *, width: int, height: int) -> np.ndarray:
        cam_id = self.robosuite_env.sim.model.camera_name2id(str(camera_name))
        fovy = float(self.robosuite_env.sim.model.cam_fovy[cam_id])
        f = 0.5 * float(height) / np.tan(fovy * np.pi / 360.0)
        return np.array(
            [[f, 0.0, 0.5 * float(width)], [0.0, f, 0.5 * float(height)], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )

    def _camera_pose_mat(self, camera_name: str) -> np.ndarray:
        sim = self.robosuite_env.sim
        camera = str(camera_name)
        camera_xmat = np.asarray(sim.data.get_camera_xmat(camera), dtype=np.float64).reshape(3, 3)
        camera_xpos = np.asarray(sim.data.get_camera_xpos(camera), dtype=np.float64).reshape(3)
        pose = np.eye(4, dtype=np.float64)
        # MuJoCo camera xmat uses OpenGL camera convention. The two fixed
        # rotations align deprojected depth points (+z forward, +y down) with
        # the MuJoCo world frame, matching the existing robosuite adapters.
        rot_y_pi = np.diag([-1.0, 1.0, -1.0])
        rot_z_pi = np.diag([-1.0, -1.0, 1.0])
        pose[:3, :3] = camera_xmat @ rot_y_pi @ rot_z_pi
        pose[:3, 3] = camera_xpos
        return pose

    def _normalize_raw_obs(self, obs: Any) -> dict[str, Any]:
        if isinstance(obs, dict):
            return dict(obs)
        if isinstance(obs, tuple) and len(obs) == 2 and isinstance(obs[1], dict):
            return dict(obs[1])
        return {"raw_observation": obs}

    def _add_export_aliases(self, low_level: dict[str, Any]) -> None:
        """Add Franka-style camera aliases used by the current LeRobot exporter."""
        if "agentview_image" not in low_level:
            for camera_name in self.camera_names:
                key = f"{camera_name}_image"
                if key in low_level and "eye_in_hand" not in camera_name:
                    low_level["agentview_image"] = low_level[key]
                    break
        if "robot0_eye_in_hand_image" not in low_level:
            for camera_name in self.camera_names:
                key = f"{camera_name}_image"
                if key in low_level and "eye_in_hand" in camera_name:
                    low_level["robot0_eye_in_hand_image"] = low_level[key]
                    break


__all__ = ["GR1RobocasaLowLevel"]
