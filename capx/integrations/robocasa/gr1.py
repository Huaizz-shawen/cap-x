from __future__ import annotations

from typing import Any

import numpy as np

from capx.envs.base import BaseEnv
from capx.integrations.base_api import ApiBase
from capx.utils.depth_utils import depth_color_to_pointcloud, depth_to_pointcloud


FOURIER_HAND_COMMAND_SPEC: tuple[dict[str, Any], ...] = (
    {"index": 0, "name": "thumb_yaw", "meaning": "thumb lateral spread/opposition"},
    {"index": 1, "name": "thumb_curl", "meaning": "thumb flexion"},
    {"index": 2, "name": "index_curl", "meaning": "index finger flexion"},
    {"index": 3, "name": "middle_curl", "meaning": "middle finger flexion"},
    {"index": 4, "name": "ring_curl", "meaning": "ring finger flexion"},
    {"index": 5, "name": "pinky_curl", "meaning": "pinky finger flexion"},
)

FOURIER_HAND_PRESETS: dict[str, np.ndarray] = {
    "open": np.array([-1.0, -1.0, -1.0, -1.0, -1.0, -1.0], dtype=np.float64),
    "power": np.array([0.45, 0.90, 0.90, 0.90, 0.90, 0.90], dtype=np.float64),
    "cylindrical": np.array([0.20, 0.65, 0.72, 0.72, 0.70, 0.68], dtype=np.float64),
    "pinch": np.array([0.35, 0.85, 0.85, 0.65, 0.15, 0.15], dtype=np.float64),
    "hook": np.array([0.10, 0.25, 0.25, 0.95, 0.95, 0.95], dtype=np.float64),
    "support": np.array([0.05, 0.20, 0.20, 0.45, 0.75, 0.75], dtype=np.float64),
}


class GR1RobocasaControlApi(ApiBase):
    """Stable harness for early RoboCasa/GR1 collection runs.

    This API intentionally exposes a small, structured surface instead of making
    the model parse the raw RoboCasa observation dictionary. The underlying GR1
    controller is still low-level, so helpers favor conservative bounded steps
    that produce useful transitions without relying on a full IK stack.
    """

    _OBJECT_ALIASES: dict[str, tuple[str, ...]] = {
        "payload": ("payload", "pot", "target object"),
        "trash": ("trash", "trash object", "distractor"),
        "lid_handle": ("lid_handle", "lid handle", "handle"),
        "target_bin": ("target_bin", "target bin", "target receptacle"),
        "trash_bin": ("trash_bin", "trash bin"),
        "ball_obj": ("ball_obj", "ball", "sphere", "pouring object"),
        "container": ("container", "target_container", "pouring container", "destination container"),
        "obj_container": ("obj_container", "object_container", "source container", "start container"),
    }

    _SCENE_OBJECT_CANDIDATES: dict[str, tuple[str, ...]] = {
        "ball_obj": ("ball_obj_main", "ball_obj_default_site"),
        "container": ("container_main", "container_default_site"),
        "obj_container": ("obj_container_main", "obj_container_default_site"),
    }

    def __init__(self, env: BaseEnv) -> None:
        super().__init__(env)

    def functions(self) -> dict[str, Any]:
        return {
            "get_task_state": self.get_task_state,
            "get_object_pose": self.get_object_pose,
            "get_named_positions": self.get_named_positions,
            "get_scene_poses": self.get_scene_poses,
            "get_robot_state": self.get_robot_state,
            "get_action_layout": self.get_action_layout,
            "get_fourier_hand_spec": self.get_fourier_hand_spec,
            "get_fourier_hand_state": self.get_fourier_hand_state,
            "observe_scene": self.observe_scene,
            "get_rgbd_observation": self.get_rgbd_observation,
            "estimate_depth_grasp_pose": self.estimate_depth_grasp_pose,
            "estimate_rgbd_roi_grasp_pose": self.estimate_rgbd_roi_grasp_pose,
            "estimate_sam3_rgbd_grasp_pose": self.estimate_sam3_rgbd_grasp_pose,
            "probe_eef_directions": self.probe_eef_directions,
            "select_gr1_skill": self.select_gr1_skill,
            "execute_gr1_skill": self.execute_gr1_skill,
            "verify_success_or_contact": self.verify_success_or_contact,
            "validate_hand_presets": self.validate_hand_presets,
            "diagnose_arm_motion_response": self.diagnose_arm_motion_response,
            "step_action": self.step_action,
            "step_delta_action": self.step_delta_action,
            "hold_still": self.hold_still,
            "move_towards_object": self.move_towards_object,
            "move_towards_position": self.move_towards_position,
            "set_hand": self.set_hand,
            "set_hand_preset": self.set_hand_preset,
            "open_hand": self.open_hand,
            "close_hand": self.close_hand,
            "plan_fourier_grasp": self.plan_fourier_grasp,
            "grasp_object": self.grasp_object,
            "plan_two_arm_lift_handles": self.plan_two_arm_lift_handles,
            "lift_pot_by_handles": self.lift_pot_by_handles,
            "lift_hand": self.lift_hand,
            "place_object": self.place_object,
            "solve_pnp_pouring_physical": self.solve_pnp_pouring_physical,
            "solve_pnp_pouring_cup_physical": self.solve_pnp_pouring_cup_physical,
            "teleport_object_to_target": self.teleport_object_to_target,
            "solve_transport_privileged": self.solve_transport_privileged,
            "sample_random_action": self.sample_random_action,
            "get_env_observation": self.get_env_observation,
            "write_video": self.write_video,
        }

    def get_task_state(self) -> dict[str, Any]:
        """Return a compact task-centric state summary.

        Returns:
            Dictionary with task name, completion flags, named object poses,
            robot/gripper positions, action dimension, and success booleans.
            Use this first instead of manually parsing ``get_env_observation``.
        """
        obs = self._env.get_observation()
        return {
            "task_description": str(obs.get("task_description", "robocasa")),
            "task_completed": bool(self._env.task_completed()),
            "reward": float(self._env.compute_reward()),
            "action_dim": int(self._action_dim()),
            "action_layout": self.get_action_layout(),
            "objects": self.get_named_positions(),
            "scene_poses": self.get_scene_poses(),
            "robot": self.get_robot_state(),
            "success_flags": self._success_flags(obs),
        }

    def get_env_observation(self) -> dict[str, Any]:
        """Return the raw RoboCasa observation dictionary.

        Prefer ``get_task_state`` and ``get_named_positions`` for control code.
        Raw ``object-state`` is usually a flat numpy vector, not a dict.
        """
        return self._env.get_observation()

    def get_named_positions(self) -> dict[str, dict[str, Any]]:
        """Return known object and target positions in a model-friendly format.

        Returns:
            Mapping from names such as ``payload``, ``trash``, ``lid_handle``,
            ``target_bin`` and ``trash_bin`` to dictionaries with ``pos`` and
            optional ``quat`` numpy arrays. Missing objects are omitted.
        """
        obs = self._env.get_observation()
        result: dict[str, dict[str, Any]] = {}
        for name in self._OBJECT_ALIASES:
            pos = self._first_obs_array(obs, (f"{name}_pos",), size=3)
            quat = self._first_obs_array(obs, (f"{name}_quat",), size=4)
            if pos is None:
                scene_entry = self._scene_object_pose(name)
                if scene_entry is not None:
                    result[name] = scene_entry
                    continue
                continue
            entry: dict[str, Any] = {"pos": pos}
            if quat is not None:
                entry["quat"] = quat
            result[name] = entry
        return result

    def get_scene_poses(
        self,
        name_hints: tuple[str, ...] | list[str] | None = None,
        include_geoms: bool = False,
        max_items: int = 80,
    ) -> dict[str, dict[str, dict[str, Any]]]:
        """Return MuJoCo scene poses for RoboCasa objects and fixtures.

        Some RoboCasa tasks, such as ``PnPPouring``, do not expose object poses
        in the observation dictionary. This helper gives the planner a compact
        privileged scene map from MuJoCo names so it can reason about cups,
        containers, balls, pots, handles and relevant fixtures without parsing
        hundreds of robot geoms.
        """
        default_hints = (
            "ball",
            "bowl",
            "bottle",
            "container",
            "cup",
            "fixture",
            "handle",
            "mug",
            "obj",
            "pan",
            "pitcher",
            "pot",
            "target",
        )
        hints = tuple(str(item).lower().strip() for item in (name_hints or default_hints) if str(item).strip())
        robosuite_env = getattr(self._env, "robosuite_env", None)
        sim = getattr(robosuite_env, "sim", None)
        model = getattr(sim, "model", None)
        data = getattr(sim, "data", None)
        if model is None or data is None:
            return {"bodies": {}, "sites": {}, "geoms": {}}

        def keep(name: str) -> bool:
            lower = str(name).lower()
            if lower.startswith(("robot", "left_eef_target", "right_eef_target")):
                return False
            return any(hint in lower for hint in hints)

        def take(names: tuple[str, ...] | list[str], getter: Any, quat_getter: Any | None = None) -> dict[str, dict[str, Any]]:
            result: dict[str, dict[str, Any]] = {}
            for name in names:
                if len(result) >= max(0, int(max_items)):
                    break
                name_str = str(name)
                if not keep(name_str):
                    continue
                try:
                    pos = np.asarray(getter(name_str), dtype=np.float64).reshape(3).copy()
                except Exception:
                    continue
                entry: dict[str, Any] = {"pos": pos}
                if quat_getter is not None:
                    try:
                        entry["quat"] = np.asarray(quat_getter(name_str), dtype=np.float64).reshape(4).copy()
                    except Exception:
                        pass
                result[name_str] = entry
            return result

        body_names = tuple(getattr(model, "body_names", ()) or ())
        site_names = tuple(getattr(model, "site_names", ()) or ())
        geom_names = tuple(getattr(model, "geom_names", ()) or ())
        scene = {
            "bodies": take(
                body_names,
                lambda name: data.body_xpos[model.body_name2id(name)],
                lambda name: data.body_xquat[model.body_name2id(name)],
            ),
            "sites": take(site_names, lambda name: data.site_xpos[model.site_name2id(name)]),
            "geoms": {},
        }
        if include_geoms:
            scene["geoms"] = take(geom_names, lambda name: data.geom_xpos[model.geom_name2id(name)])
        return scene

    def _scene_object_pose(self, canonical_name: str) -> dict[str, Any] | None:
        candidates = self._SCENE_OBJECT_CANDIDATES.get(str(canonical_name), ())
        if not candidates:
            return None
        scene = self.get_scene_poses()
        for section in ("bodies", "sites"):
            entries = scene.get(section, {})
            for candidate in candidates:
                entry = entries.get(candidate)
                if entry is not None:
                    return dict(entry)
        return None

    def _find_scene_affordance(
        self,
        name_hints: tuple[str, ...] | list[str],
        *,
        include_geoms: bool = True,
        prefer_names: tuple[str, ...] | list[str] = (),
        avoid_names: tuple[str, ...] | list[str] = (),
        reference_pos: np.ndarray | list[float] | tuple[float, float, float] | None = None,
        z_range: tuple[float, float] | None = None,
        scene: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        """Return the best scene entry matching the requested affordance.

        Names alone are ambiguous in RoboCasa fixture-heavy scenes: e.g. drawer
        tasks may contain several cabinet handles, but only one tabletop drawer
        handle is reachable by GR1. This scorer keeps the API task-generic while
        making the selection explainable in the returned payload.
        """
        ranked = self._rank_scene_affordances(
            name_hints,
            include_geoms=include_geoms,
            prefer_names=prefer_names,
            avoid_names=avoid_names,
            reference_pos=reference_pos,
            z_range=z_range,
            scene=scene,
        )
        return ranked[0] if ranked else None

    def _rank_scene_affordances(
        self,
        name_hints: tuple[str, ...] | list[str],
        *,
        include_geoms: bool = True,
        prefer_names: tuple[str, ...] | list[str] = (),
        avoid_names: tuple[str, ...] | list[str] = (),
        reference_pos: np.ndarray | list[float] | tuple[float, float, float] | None = None,
        z_range: tuple[float, float] | None = None,
        scene: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Rank matching MuJoCo scene entries with transparent score terms."""
        hints = tuple(str(hint).lower().strip() for hint in name_hints if str(hint).strip())
        prefers = tuple(str(item).lower().strip() for item in prefer_names if str(item).strip())
        avoids = tuple(str(item).lower().strip() for item in avoid_names if str(item).strip())
        scene_map = scene or self.get_scene_poses(name_hints=hints, include_geoms=include_geoms, max_items=240)
        ref = None if reference_pos is None else np.asarray(reference_pos, dtype=np.float64).reshape(3)
        ranked: list[dict[str, Any]] = []
        for section in ("sites", "bodies", "geoms"):
            section_bias = {"sites": 0.15, "bodies": 0.05, "geoms": 0.0}.get(section, 0.0)
            for name, entry in scene_map.get(section, {}).items():
                lower = str(name).lower()
                if hints and not any(hint in lower for hint in hints):
                    continue
                try:
                    pos = np.asarray(entry.get("pos"), dtype=np.float64).reshape(3)
                except Exception:
                    continue
                matched = [hint for hint in hints if hint in lower]
                preferred = [item for item in prefers if item in lower]
                avoided = [item for item in avoids if item in lower]
                score = float(len(matched)) + 2.0 * float(len(preferred)) - 3.0 * float(len(avoided)) + section_bias
                distance = None
                if ref is not None:
                    distance = float(np.linalg.norm(pos - ref))
                    score -= 0.25 * distance
                z_penalty = 0.0
                if z_range is not None:
                    z_min, z_max = float(z_range[0]), float(z_range[1])
                    if pos[2] < z_min:
                        z_penalty = z_min - float(pos[2])
                    elif pos[2] > z_max:
                        z_penalty = float(pos[2]) - z_max
                    score -= 5.0 * z_penalty
                result = dict(entry)
                result.update(
                    {
                        "name": str(name),
                        "section": section,
                        "pos": pos.copy(),
                        "score": score,
                        "score_terms": {
                            "matched_hints": matched,
                            "preferred_terms": preferred,
                            "avoided_terms": avoided,
                            "section_bias": section_bias,
                            "distance_to_reference": distance,
                            "z_penalty": z_penalty,
                        },
                    }
                )
                ranked.append(result)
        ranked.sort(key=lambda item: float(item.get("score", 0.0)), reverse=True)
        return ranked

    def _axis_from_named_points(
        self,
        positive_name_hints: tuple[str, ...] | list[str],
        negative_name_hints: tuple[str, ...] | list[str],
        *,
        scene: dict[str, Any] | None = None,
    ) -> np.ndarray | None:
        """Infer a horizontal axis from two ranked scene affordances."""
        positive = self._find_scene_affordance(positive_name_hints, scene=scene, include_geoms=True)
        negative = self._find_scene_affordance(negative_name_hints, scene=scene, include_geoms=True)
        if positive is None or negative is None:
            return None
        axis = np.asarray(positive["pos"], dtype=np.float64).reshape(3) - np.asarray(negative["pos"], dtype=np.float64).reshape(3)
        axis[2] = 0.0
        norm = float(np.linalg.norm(axis))
        if norm < 1e-9:
            return None
        return axis / norm

    @staticmethod
    def _calibrated_requested_axis(
        desired_world_axis: np.ndarray | list[float] | tuple[float, float, float],
        calibration: dict[str, Any] | None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Choose the requested controller axis that best realizes a world axis."""
        desired = np.asarray(desired_world_axis, dtype=np.float64).reshape(3)
        desired[2] = 0.0 if abs(float(desired[2])) < 1e-9 else desired[2]
        desired_norm = float(np.linalg.norm(desired))
        if desired_norm < 1e-9:
            desired = np.array([0.0, -1.0, 0.0], dtype=np.float64)
            desired_norm = 1.0
        desired = desired / desired_norm
        best_axis = desired.copy()
        best_score = -float("inf")
        best_name = "desired_world_axis"
        direction_map = (calibration or {}).get("direction_motion_map", {}) if isinstance(calibration, dict) else {}
        for name, probe in direction_map.items():
            try:
                actual = np.asarray(probe.get("actual_delta"), dtype=np.float64).reshape(3)
            except Exception:
                continue
            actual_norm = float(np.linalg.norm(actual))
            if actual_norm < 1e-9:
                continue
            score = float(np.dot(actual / actual_norm, desired))
            if score > best_score:
                requested = np.asarray(probe.get("requested_delta"), dtype=np.float64).reshape(3)
                req_norm = float(np.linalg.norm(requested))
                if req_norm < 1e-9:
                    continue
                best_axis = requested / req_norm
                best_score = score
                best_name = str(name)
        return best_axis, {
            "desired_world_axis": desired.copy(),
            "selected_requested_axis": best_axis.copy(),
            "selected_probe": best_name,
            "alignment_score": None if best_score == -float("inf") else float(best_score),
        }

    @staticmethod
    def _transform_points(pose_mat: np.ndarray, points: np.ndarray) -> np.ndarray:
        points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        if len(points) == 0:
            return points.copy()
        pose = np.asarray(pose_mat, dtype=np.float64).reshape(4, 4)
        homo = np.concatenate([points, np.ones((len(points), 1), dtype=np.float64)], axis=1)
        return (homo @ pose.T)[:, :3]

    @staticmethod
    def _array_stats(values: np.ndarray) -> dict[str, float | int | None]:
        arr = np.asarray(values, dtype=np.float64).reshape(-1)
        if arr.size == 0:
            return {"count": 0, "min": None, "max": None, "mean": None}
        return {
            "count": int(arr.size),
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
            "mean": float(np.mean(arr)),
        }

    def _resolve_grasp_target_pos(
        self,
        *,
        object_name: str | None,
        name_hints: tuple[str, ...] | list[str] | None,
        target_pos: tuple[float, float, float] | list[float] | np.ndarray | None,
    ) -> dict[str, Any]:
        if target_pos is not None:
            return {
                "source": "explicit_target_pos",
                "name": object_name or "explicit_target",
                "pos": np.asarray(target_pos, dtype=np.float64).reshape(3).copy(),
            }
        if object_name is not None:
            try:
                pos, quat = self.get_object_pose(str(object_name))
                return {
                    "source": "object_pose",
                    "name": str(object_name),
                    "pos": np.asarray(pos, dtype=np.float64).reshape(3).copy(),
                    "quat": np.asarray(quat, dtype=np.float64).reshape(4).copy(),
                }
            except Exception:
                pass
        hints = tuple(name_hints or (() if object_name is None else (str(object_name),)))
        affordance = self._find_scene_affordance(
            hints or ("handle", "object", "target"),
            include_geoms=True,
            prefer_names=("default_site", "_c", "handle"),
            avoid_names=("visual", "connector"),
            z_range=(0.4, 1.4),
        )
        if affordance is None:
            raise ValueError("Could not resolve depth grasp target from object_name/name_hints/target_pos")
        return {
            "source": "scene_affordance",
            "name": affordance.get("name"),
            "section": affordance.get("section"),
            "pos": np.asarray(affordance["pos"], dtype=np.float64).reshape(3).copy(),
            "score_terms": affordance.get("score_terms", {}),
        }

    def _copy_drawer_handle_pose(self, skill: dict[str, Any]) -> dict[str, Any] | None:
        affordance = skill.get("handle_affordance") if isinstance(skill.get("handle_affordance"), dict) else None
        if affordance is None:
            return None
        name = str(affordance.get("name", ""))
        section = str(affordance.get("section", ""))
        if not name or not section:
            return None
        robosuite_env = getattr(self._env, "robosuite_env", None)
        sim = getattr(robosuite_env, "sim", None)
        model = getattr(sim, "model", None)
        data = getattr(sim, "data", None)
        if model is None or data is None:
            return None
        try:
            if section == "sites":
                pos = np.asarray(data.site_xpos[model.site_name2id(name)], dtype=np.float64).reshape(3).copy()
            elif section == "geoms":
                pos = np.asarray(data.geom_xpos[model.geom_name2id(name)], dtype=np.float64).reshape(3).copy()
            elif section == "bodies":
                pos = np.asarray(data.body_xpos[model.body_name2id(name)], dtype=np.float64).reshape(3).copy()
            else:
                return None
        except Exception:
            return None
        return {"name": name, "section": section, "pos": pos}

    @staticmethod
    def _principal_axes(points: np.ndarray) -> np.ndarray:
        pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        if len(pts) < 3:
            return np.eye(3, dtype=np.float64)
        centered = pts - pts.mean(axis=0, keepdims=True)
        cov = centered.T @ centered / max(1, len(pts) - 1)
        try:
            values, vectors = np.linalg.eigh(cov)
        except np.linalg.LinAlgError:
            return np.eye(3, dtype=np.float64)
        order = np.argsort(values)[::-1]
        axes = vectors[:, order].T
        for idx in range(3):
            norm = float(np.linalg.norm(axes[idx]))
            if norm > 1e-9:
                axes[idx] = axes[idx] / norm
        return axes

    @staticmethod
    def _grasp_pose_matrix(
        grasp_pos: np.ndarray,
        approach_axis: np.ndarray,
        closing_axis: np.ndarray,
    ) -> np.ndarray:
        z_axis = -np.asarray(approach_axis, dtype=np.float64).reshape(3)
        z_norm = float(np.linalg.norm(z_axis))
        if z_norm < 1e-9:
            z_axis = np.array([0.0, 0.0, -1.0], dtype=np.float64)
            z_norm = 1.0
        z_axis = z_axis / z_norm
        x_axis = np.asarray(closing_axis, dtype=np.float64).reshape(3)
        x_axis = x_axis - z_axis * float(np.dot(x_axis, z_axis))
        if float(np.linalg.norm(x_axis)) < 1e-9:
            x_axis = np.cross(np.array([0.0, 0.0, 1.0], dtype=np.float64), z_axis)
        if float(np.linalg.norm(x_axis)) < 1e-9:
            x_axis = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        x_axis = x_axis / float(np.linalg.norm(x_axis))
        y_axis = np.cross(z_axis, x_axis)
        y_axis = y_axis / max(float(np.linalg.norm(y_axis)), 1e-9)
        pose = np.eye(4, dtype=np.float64)
        pose[:3, 0] = x_axis
        pose[:3, 1] = y_axis
        pose[:3, 2] = z_axis
        pose[:3, 3] = np.asarray(grasp_pos, dtype=np.float64).reshape(3)
        return pose

    def get_object_pose(self, object_name: str) -> tuple[np.ndarray, np.ndarray]:
        """Get a named object's pose from privileged RoboCasa state.

        Args:
            object_name: One of ``payload``, ``trash``, ``lid_handle``,
                ``target_bin`` or ``trash_bin``. Natural aliases like
                ``target bin`` and ``lid handle`` are accepted.

        Returns:
            ``(position, quaternion_wxyz)``. If the observation does not expose
            a quaternion for the target, identity quaternion is returned.
        """
        canonical = self._canonical_object_name(object_name)
        obs = self._env.get_observation()
        pos = self._first_obs_array(obs, (f"{canonical}_pos",), size=3)
        if pos is None:
            scene_entry = self._scene_object_pose(canonical)
            if scene_entry is None:
                raise ValueError(f"Object {object_name!r} is not exposed by this RoboCasa observation or scene")
            pos = np.asarray(scene_entry["pos"], dtype=np.float32)
            quat = np.asarray(scene_entry.get("quat", np.array([1.0, 0.0, 0.0, 0.0])), dtype=np.float32)
            return pos, quat
        quat = self._first_obs_array(obs, (f"{canonical}_quat",), size=4)
        if quat is None:
            quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        return pos, quat

    def get_robot_state(self) -> dict[str, Any]:
        """Return normalized robot state vectors and end-effector hints.

        Returns:
            Dictionary with ``joint_pos``, ``cartesian_pos``, left/right EEF
            positions when present, and gripper-to-object vectors when present.
        """
        obs = self._env.get_observation()
        state: dict[str, Any] = {
            "joint_pos": np.asarray(obs.get("robot_joint_pos", []), dtype=np.float32),
            "cartesian_pos": np.asarray(obs.get("robot_cartesian_pos", []), dtype=np.float32),
        }
        for key in (
            "robot0_left_eef_pos",
            "robot0_right_eef_pos",
            "robot0_base_to_left_eef_pos",
            "robot0_base_to_right_eef_pos",
            "robot0_base_to_left_eef_quat",
            "robot0_base_to_right_eef_quat",
            "robot0_base_to_left_eef_quat_site",
            "robot0_base_to_right_eef_quat_site",
            "gripper0_to_payload",
            "gripper1_to_payload",
            "gripper0_to_lid_handle",
            "gripper1_to_trash",
        ):
            if key in obs:
                state[key] = np.asarray(obs[key], dtype=np.float32).reshape(-1)
        return state

    def get_action_layout(self) -> dict[str, tuple[int, int]]:
        """Return the active GR1 action slices.

        The current RoboCasa GR1 controller exposes a whole-body-IK action
        vector, not a simple left/right half split. In the validated setup the
        layout is ``right:0:6``, ``left:6:12``, ``right_gripper:12:18`` and
        ``left_gripper:18:24``. This method asks the live controller first and
        falls back to that canonical layout if introspection is unavailable.
        """
        robosuite_env = getattr(self._env, "robosuite_env", None)
        robots = getattr(robosuite_env, "robots", []) if robosuite_env is not None else []
        if robots:
            controller = getattr(robots[0], "composite_controller", None)
            split = getattr(controller, "_whole_body_controller_action_split_indexes", None)
            if split:
                return {str(name): (int(bounds[0]), int(bounds[1])) for name, bounds in split.items()}

        dim = self._action_dim()
        if dim >= 24:
            return {
                "right": (0, 6),
                "left": (6, 12),
                "right_gripper": (12, 18),
                "left_gripper": (18, 24),
            }
        half = max(1, dim // 2)
        return {"right": (0, min(half, dim)), "left": (half, dim)}

    def get_fourier_hand_spec(self) -> dict[str, Any]:
        """Return the validated Fourier hand command surface.

        Returns:
            A model-readable description of the current 6D Fourier hand action
            space, presets, and live action slices. Use this before writing
            low-level hand commands.
        """
        return {
            "command_dim": 6,
            "command_spec": [dict(item) for item in FOURIER_HAND_COMMAND_SPEC],
            "presets": {name: values.copy() for name, values in FOURIER_HAND_PRESETS.items()},
            "action_layout": self.get_action_layout(),
            "notes": (
                "Each hand accepts a 6D normalized command that RoboCasa expands "
                "to the 11 actuated Fourier hand joints. Use presets unless you "
                "are explicitly debugging hand control."
            ),
        }

    def get_fourier_hand_state(self, arm: str = "both") -> dict[str, Any]:
        """Return observable Fourier hand state for one or both hands.

        The exact joint names differ across RoboCasa/GR1 builds, so this method
        returns both the action slices and a best-effort MuJoCo joint snapshot.
        """
        return {arm_name: self._hand_joint_snapshot(arm_name) for arm_name in self._selected_arms(arm)}

    def observe_scene(
        self,
        name_hints: tuple[str, ...] | list[str] | None = None,
        include_geoms: bool = True,
        max_items: int = 120,
    ) -> dict[str, Any]:
        """Return the shared object-centric scene state used by GR1 skills.

        This is the first stable interface layer for RoboCasa GR1 skills. It is
        intentionally privileged today, but the returned shape is designed so a
        later RGBD/detector implementation can fill the same fields.
        """
        return {
            "task_state": self.get_task_state(),
            "scene_poses": self.get_scene_poses(
                name_hints=name_hints,
                include_geoms=include_geoms,
                max_items=max_items,
            ),
            "robot_state": self.get_robot_state(),
            "hand_state": self.get_fourier_hand_state("both"),
            "hand_spec": self.get_fourier_hand_spec(),
        }

    def get_rgbd_observation(
        self,
        camera_name: str | None = None,
        *,
        subsample_factor: int = 4,
        max_points: int = 20000,
        include_arrays: bool = False,
    ) -> dict[str, Any]:
        """Render RGBD and a world-frame point cloud from a RoboCasa camera.

        This is the first bridge toward GraspNet-style perception. By default
        it returns compact metadata and a bounded point cloud instead of dumping
        full RGB/depth arrays into model context.
        """
        if not hasattr(self._env, "render_camera_rgbd"):
            raise RuntimeError("Underlying low-level env does not expose render_camera_rgbd")
        rgbd = self._env.render_camera_rgbd(camera_name)
        rgb = np.asarray(rgbd["rgb"], dtype=np.uint8)
        depth = np.asarray(rgbd["depth"], dtype=np.float32)
        intrinsics = np.asarray(rgbd["intrinsics"], dtype=np.float64).reshape(3, 3)
        pose_mat = np.asarray(rgbd["pose_mat"], dtype=np.float64).reshape(4, 4)
        subsample = max(1, int(subsample_factor))
        if subsample > 1:
            height, width = depth.shape[:2]
            trimmed_height = max(subsample, (height // subsample) * subsample)
            trimmed_width = max(subsample, (width // subsample) * subsample)
            rgb = rgb[:trimmed_height, :trimmed_width]
            depth = depth[:trimmed_height, :trimmed_width]
        points_camera, colors = depth_color_to_pointcloud(
            depth,
            rgb,
            intrinsics,
            subsample_factor=subsample,
            depth_clip_range=(0.02, 5.0),
        )
        points_world = self._transform_points(pose_mat, points_camera)
        if int(max_points) > 0 and len(points_world) > int(max_points):
            idx = np.linspace(0, len(points_world) - 1, int(max_points)).astype(np.int64)
            points_world = points_world[idx]
            colors = colors[idx]
        payload: dict[str, Any] = {
            "camera_name": str(rgbd["camera_name"]),
            "image_shape": tuple(int(x) for x in rgb.shape),
            "depth_shape": tuple(int(x) for x in depth.shape),
            "intrinsics": intrinsics,
            "pose_mat": pose_mat,
            "point_cloud_world": points_world,
            "point_cloud_colors": colors,
            "point_count": int(len(points_world)),
            "depth_stats": self._array_stats(depth[np.isfinite(depth)]),
        }
        if include_arrays:
            payload["rgb"] = rgb
            payload["depth"] = depth
        return payload

    def estimate_depth_grasp_pose(
        self,
        *,
        camera_name: str | None = None,
        object_name: str | None = None,
        name_hints: tuple[str, ...] | list[str] | None = None,
        target_pos: tuple[float, float, float] | list[float] | np.ndarray | None = None,
        arm: str = "right",
        crop_radius: float = 0.10,
        approach_distance: float = 0.10,
        subsample_factor: int = 3,
    ) -> dict[str, Any]:
        """Estimate a compact GraspNet-compatible grasp pose from depth.

        This is not a learned GraspNet model. It validates the perception bridge:
        RGBD -> world point cloud -> target crop -> candidate wrist pregrasp and
        grasp pose. A learned GraspNet/AnyGrasp backend can later replace the
        crop/PCA estimator while preserving this output shape.
        """
        target = self._resolve_grasp_target_pos(
            object_name=object_name,
            name_hints=name_hints,
            target_pos=target_pos,
        )
        rgbd = self.get_rgbd_observation(
            camera_name=camera_name,
            subsample_factor=subsample_factor,
            max_points=60000,
            include_arrays=False,
        )
        points = np.asarray(rgbd["point_cloud_world"], dtype=np.float64).reshape(-1, 3)
        target_vec = np.asarray(target["pos"], dtype=np.float64).reshape(3)
        deltas = points - target_vec
        distances = np.linalg.norm(deltas, axis=1) if len(points) else np.zeros(0, dtype=np.float64)
        mask = distances <= float(crop_radius)
        crop = points[mask]
        used_fallback = False
        if len(crop) < 12:
            # Keep the estimator deterministic and diagnosable when the target
            # is partially occluded or outside the selected camera view.
            if len(points) > 0:
                nearest = np.argsort(distances)[: min(len(points), 64)]
                crop = points[nearest]
            else:
                crop = target_vec.reshape(1, 3)
            used_fallback = True
        centroid = crop.mean(axis=0)
        bbox_min = crop.min(axis=0)
        bbox_max = crop.max(axis=0)
        principal_axes = self._principal_axes(crop)
        selected_arm = arm if arm in {"left", "right"} else "right"
        eef = self._eef_world_pos(selected_arm)
        approach_axis = eef - centroid
        if float(np.linalg.norm(approach_axis)) < 1e-9:
            approach_axis = np.array([0.0, -1.0, 0.0], dtype=np.float64)
        approach_axis = approach_axis / float(np.linalg.norm(approach_axis))
        grasp_pos = centroid.copy()
        pregrasp_pos = grasp_pos + approach_axis * float(approach_distance)
        pregrasp_pos[2] += 0.03
        closing_axis = principal_axes[0]
        pose_mat = self._grasp_pose_matrix(grasp_pos, approach_axis, closing_axis)
        return {
            "method": "depth_crop_pca_v0",
            "camera_name": rgbd["camera_name"],
            "target": target,
            "arm": selected_arm,
            "point_count": int(len(points)),
            "crop_point_count": int(len(crop)),
            "used_nearest_fallback": bool(used_fallback),
            "crop_radius": float(crop_radius),
            "centroid": centroid,
            "bbox_min": bbox_min,
            "bbox_max": bbox_max,
            "principal_axes": principal_axes,
            "grasp_pos": grasp_pos,
            "pregrasp_pos": pregrasp_pos,
            "approach_axis_world": approach_axis,
            "closing_axis_world": closing_axis,
            "pose_mat": pose_mat,
            "notes": [
                "This is a geometry estimator used to validate RGBD/world calibration.",
                "Replace method with GraspNet/AnyGrasp once camera transform and target crop are verified.",
            ],
        }

    def _estimate_masked_rgbd_grasp_pose(
        self,
        *,
        rgbd: dict[str, Any],
        mask: np.ndarray,
        arm: str,
        approach_distance: float,
        method: str,
        source: dict[str, Any],
        min_points: int = 16,
    ) -> dict[str, Any]:
        rgb = np.asarray(rgbd["rgb"], dtype=np.uint8)
        depth = np.asarray(rgbd["depth"], dtype=np.float32)
        intrinsics = np.asarray(rgbd["intrinsics"], dtype=np.float64).reshape(3, 3)
        pose_mat = np.asarray(rgbd["pose_mat"], dtype=np.float64).reshape(4, 4)
        mask_bool = np.asarray(mask, dtype=bool)
        if mask_bool.shape != depth.shape:
            raise ValueError(f"RGBD mask shape {mask_bool.shape} does not match depth shape {depth.shape}")
        points_camera = depth_to_pointcloud(
            depth,
            intrinsics,
            subsample_factor=1,
            depth_clip_range=(0.02, 5.0),
            filter_invalid=False,
        ).reshape(depth.shape[0], depth.shape[1], 3)
        z = points_camera[..., 2]
        valid = mask_bool & np.isfinite(z) & (z >= 0.02) & (z <= 5.0)
        if int(np.count_nonzero(valid)) < int(min_points):
            raise ValueError(
                f"RGBD mask produced too few valid depth points: {int(np.count_nonzero(valid))}"
            )
        crop = self._transform_points(pose_mat, points_camera[valid])
        centroid = crop.mean(axis=0)
        bbox_min = crop.min(axis=0)
        bbox_max = crop.max(axis=0)
        principal_axes = self._principal_axes(crop)
        selected_arm = arm if arm in {"left", "right"} else "right"
        eef = self._eef_world_pos(selected_arm)
        approach_axis = eef - centroid
        if float(np.linalg.norm(approach_axis)) < 1e-9:
            approach_axis = np.array([0.0, -1.0, 0.0], dtype=np.float64)
        approach_axis = approach_axis / float(np.linalg.norm(approach_axis))
        grasp_pos = centroid.copy()
        pregrasp_pos = grasp_pos + approach_axis * float(approach_distance)
        pregrasp_pos[2] += 0.03
        closing_axis = principal_axes[0]
        ys, xs = np.nonzero(valid)
        pose_mat_out = self._grasp_pose_matrix(grasp_pos, approach_axis, closing_axis)
        return {
            "method": method,
            "camera_name": rgbd["camera_name"],
            "arm": selected_arm,
            "source": source,
            "image_shape": tuple(int(x) for x in rgb.shape),
            "mask_pixel_count": int(np.count_nonzero(mask_bool)),
            "valid_depth_point_count": int(len(crop)),
            "mask_box_xyxy": [
                int(xs.min()),
                int(ys.min()),
                int(xs.max() + 1),
                int(ys.max() + 1),
            ],
            "centroid": centroid,
            "bbox_min": bbox_min,
            "bbox_max": bbox_max,
            "principal_axes": principal_axes,
            "grasp_pos": grasp_pos,
            "pregrasp_pos": pregrasp_pos,
            "approach_axis_world": approach_axis,
            "closing_axis_world": closing_axis,
            "pose_mat": pose_mat_out,
            "privilege_level": "rgbd_mask_no_sim_object_pose_for_anchor",
            "notes": [
                "The grasp anchor is estimated from rendered RGBD and an image mask.",
                "No simulator object/site/geom pose is used to choose the anchor in this method.",
            ],
        }

    def estimate_rgbd_roi_grasp_pose(
        self,
        *,
        camera_name: str | None = None,
        roi_xyxy: tuple[float, float, float, float] | list[float] = (0.34, 0.42, 0.66, 0.72),
        arm: str = "right",
        approach_distance: float = 0.10,
    ) -> dict[str, Any]:
        """Estimate a grasp pose from RGBD inside a normalized image ROI.

        This is a deliberately labeled fallback for environments where SAM3 is
        not running. It uses only camera/depth observations plus a task-camera
        prior, not simulator object poses.
        """
        rgbd = self.get_rgbd_observation(
            camera_name=camera_name,
            subsample_factor=1,
            max_points=0,
            include_arrays=True,
        )
        depth = np.asarray(rgbd["depth"], dtype=np.float32)
        h, w = depth.shape
        x1f, y1f, x2f, y2f = [float(v) for v in roi_xyxy]
        x1 = int(np.clip(round(x1f * w), 0, w - 1))
        y1 = int(np.clip(round(y1f * h), 0, h - 1))
        x2 = int(np.clip(round(x2f * w), x1 + 1, w))
        y2 = int(np.clip(round(y2f * h), y1 + 1, h))
        mask = np.zeros((h, w), dtype=bool)
        mask[y1:y2, x1:x2] = True
        return self._estimate_masked_rgbd_grasp_pose(
            rgbd=rgbd,
            mask=mask,
            arm=arm,
            approach_distance=approach_distance,
            method="rgbd_roi_mask_pca_v0",
            source={
                "type": "normalized_roi",
                "roi_xyxy": [x1f, y1f, x2f, y2f],
                "pixel_box_xyxy": [x1, y1, x2, y2],
            },
        )

    def estimate_sam3_rgbd_grasp_pose(
        self,
        *,
        camera_name: str | None = None,
        text_prompt: str = "drawer handle",
        arm: str = "right",
        approach_distance: float = 0.10,
        max_masks: int = 5,
    ) -> dict[str, Any]:
        """Estimate a grasp pose from rendered RGBD plus a SAM3 image mask.

        The only target-selection input is the text prompt against the rendered
        RGB image. This is the minimal de-privileged ablation for replacing the
        earlier privileged ``observe_scene(...include_geoms=True)`` handle pose.
        """
        from capx.integrations.vision.sam3 import init_sam3

        rgbd = self.get_rgbd_observation(
            camera_name=camera_name,
            subsample_factor=1,
            max_points=0,
            include_arrays=True,
        )
        rgb = np.asarray(rgbd["rgb"], dtype=np.uint8)
        segment = init_sam3()
        results = segment(rgb, str(text_prompt))
        if not results:
            raise RuntimeError(f"SAM3 returned no masks for prompt {text_prompt!r}")
        candidates: list[tuple[float, dict[str, Any], dict[str, Any]]] = []
        for idx, result in enumerate(results[: max(1, int(max_masks))]):
            mask = np.asarray(result.get("mask"), dtype=bool)
            try:
                estimate = self._estimate_masked_rgbd_grasp_pose(
                    rgbd=rgbd,
                    mask=mask,
                    arm=arm,
                    approach_distance=approach_distance,
                    method="sam3_rgbd_mask_pca_v0",
                    source={
                        "type": "sam3_text_prompt",
                        "text_prompt": str(text_prompt),
                        "result_index": int(idx),
                        "score": float(result.get("score", 0.0)),
                        "box": [float(x) for x in result.get("box", [])],
                        "label": str(result.get("label", text_prompt)),
                    },
                )
            except Exception:
                continue
            score = float(result.get("score", 0.0))
            candidates.append((score, result, estimate))
        if not candidates:
            raise RuntimeError(f"SAM3 masks for prompt {text_prompt!r} had no valid depth points")
        candidates.sort(key=lambda item: item[0], reverse=True)
        best = candidates[0][2]
        best["candidate_count"] = int(len(candidates))
        return best

    def probe_eef_directions(
        self,
        arm: str = "right",
        step: float = 0.04,
        axes: tuple[str, ...] | list[str] = ("+x", "-x", "+y", "-y", "+z", "-z"),
        steps_per_axis: int = 6,
    ) -> dict[str, Any]:
        """Probe small EEF moves and restore the simulator state.

        The existing ``diagnose_arm_motion_response`` already implements the
        capture/restore mechanics. This wrapper normalizes its output into the
        skill-library vocabulary from ``docs/robocasa-gr1-skill-library.md``.
        """
        raw = self.diagnose_arm_motion_response(
            arm=arm,
            steps=max(1, int(steps_per_axis)),
            scale=float(step),
        )
        requested = {str(axis).lower().strip() for axis in axes}
        probes = {
            axis: probe
            for axis, probe in raw.get("probes", {}).items()
            if not requested or axis in requested
        }
        return {
            "arm": raw.get("arm", arm),
            "origin_eef_pose": raw.get("baseline_eef_pos"),
            "reachable_axes": {
                axis: bool(float(probe.get("actual_norm", 0.0)) > 1e-4)
                for axis, probe in probes.items()
            },
            "direction_motion_map": probes,
        }

    def select_gr1_skill(
        self,
        task_type: str,
        object_name: str | None = None,
        target_name: str | None = None,
        arm: str = "right",
        hand_pose_family: str = "auto",
        scene: dict[str, Any] | None = None,
        calibration: dict[str, Any] | None = None,
        visual_grasp: dict[str, Any] | None = None,
        use_depth_grasp: bool = False,
        depth_camera_name: str | None = None,
    ) -> dict[str, Any]:
        """Select a task-level GR1 skill through one shared function shape.

        The returned dictionary is executable by ``execute_gr1_skill``. Task
        variants should differ primarily through parameters and affordances, not
        through custom low-level controller code at the call site.
        """
        task = str(task_type).lower().strip()
        if task in {"pnp_pouring", "pnp", "pouring"}:
            return {
                "skill_type": "pnp_pouring",
                "object_name": object_name or "obj_container",
                "ball_name": "ball_obj",
                "target_name": target_name or "container",
                "arm": arm,
                "hand_pose_family": "cylindrical" if hand_pose_family == "auto" else hand_pose_family,
            }
        if task in {"two_arm_lift", "lift", "twoarmlift"}:
            plan = self.plan_two_arm_lift_handles(
                preset="hook" if hand_pose_family == "auto" else hand_pose_family,
                scene=scene.get("scene_poses") if isinstance(scene, dict) and "scene_poses" in scene else scene,
            )
            return {
                "skill_type": "two_arm_lift",
                "object_name": object_name or "pot",
                "target_name": target_name,
                "arm": "both",
                "hand_pose_family": "hook" if hand_pose_family == "auto" else hand_pose_family,
                "plan": plan,
                "selection_diagnostics": {
                    "left_handle": plan.get("left_handle_name"),
                    "right_handle": plan.get("right_handle_name"),
                    "handle_separation": plan.get("handle_separation"),
                },
            }
        if task in {"drawer", "drawer_pull", "open_drawer", "tabletop_open_drawer"}:
            robot = self.get_robot_state()
            selected_arm = arm if arm in {"left", "right"} else "right"
            eef_key = "robot0_left_eef_pos" if selected_arm == "left" else "robot0_right_eef_pos"
            reference_pos = robot.get(eef_key, self._eef_world_pos(selected_arm))
            scene_map = scene.get("scene_poses") if isinstance(scene, dict) and "scene_poses" in scene else scene
            affordances = self._rank_scene_affordances(
                ("drawer", "handle"),
                include_geoms=True,
                prefer_names=("drawer_tabletop", "door_handle", "default_site"),
                avoid_names=("stack_", "connector", "visual", "trim"),
                reference_pos=reference_pos,
                z_range=(0.82, 1.28),
                scene=scene_map,
            )
            affordance = affordances[0] if affordances else None
            handle_pos = affordance.get("pos") if affordance is not None else self._eef_world_pos("right")
            drawer_axis = None
            axis_source = "fallback_-y"
            if affordance is not None:
                handle_name = str(affordance.get("name", ""))
                if "drawer_tabletop" in handle_name:
                    drawer_axis = self._axis_from_named_points(
                        ("drawer_tabletop", "door_handle"),
                        ("drawer_tabletop", "default_site"),
                        scene=scene_map,
                    )
                    axis_source = "drawer_handle_minus_drawer_center"
            desired_axis = drawer_axis if drawer_axis is not None else np.array([0.0, -1.0, 0.0], dtype=np.float64)
            pull_axis, axis_diagnostics = self._calibrated_requested_axis(desired_axis, calibration)
            depth_grasp = visual_grasp
            if depth_grasp is None and bool(use_depth_grasp):
                depth_grasp = self.estimate_depth_grasp_pose(
                    camera_name=depth_camera_name,
                    name_hints=("drawer_tabletop", "door_handle"),
                    arm=selected_arm,
                    crop_radius=0.14,
                    approach_distance=0.10,
                    subsample_factor=3,
                )
            if depth_grasp is not None:
                handle_pos = np.asarray(depth_grasp.get("grasp_pos", handle_pos), dtype=np.float64).reshape(3)
            wrist_grasp_pos = None
            wrist_pregrasp_pos = None
            if depth_grasp is not None:
                # The depth grasp is an object/handle pose. The GR1 action
                # target is a wrist frame, so keep the wrist slightly outside
                # and above the handle and let the Fourier hook fingers do the
                # final contact. Directly driving the wrist to handle center is
                # too low for the current whole-body IK posture.
                wrist_grasp_pos = np.asarray(handle_pos, dtype=np.float64).reshape(3) + np.array(
                    [0.0, -0.035, 0.10], dtype=np.float64
                )
                wrist_pregrasp_pos = np.asarray(handle_pos, dtype=np.float64).reshape(3) + np.array(
                    [0.0, -0.14, 0.12], dtype=np.float64
                )
            return {
                "skill_type": "drawer_pull",
                "object_name": object_name or affordance.get("name", "drawer_handle") if affordance else object_name or "drawer_handle",
                "target_name": target_name,
                "arm": selected_arm,
                "hand_pose_family": "hook" if hand_pose_family == "auto" else hand_pose_family,
                "handle_affordance": affordance,
                "ranked_affordance_candidates": affordances[:8],
                "handle_pos": np.asarray(handle_pos, dtype=np.float64).reshape(3).copy(),
                "visual_grasp": depth_grasp,
                "wrist_grasp_pos": None if wrist_grasp_pos is None else wrist_grasp_pos.copy(),
                "wrist_pregrasp_pos": None if wrist_pregrasp_pos is None else wrist_pregrasp_pos.copy(),
                "desired_pull_axis": np.asarray(desired_axis, dtype=np.float64).reshape(3).copy(),
                "pull_axis": np.asarray(pull_axis, dtype=np.float64).reshape(3).copy(),
                "axis_source": axis_source,
                "axis_diagnostics": axis_diagnostics,
            }
        raise ValueError(f"Unsupported GR1 skill task_type={task_type!r}")

    def execute_gr1_skill(
        self,
        skill: dict[str, Any],
        *,
        smoke: bool = True,
    ) -> dict[str, Any]:
        """Execute a skill selected by ``select_gr1_skill``.

        ``smoke=True`` keeps motions short for interface validation. Full
        task-solving runs can pass ``smoke=False`` once the shared shape is
        proven on all target tasks.
        """
        skill_type = str(skill.get("skill_type", "")).lower().strip()
        if skill_type == "pnp_pouring":
            return self.solve_pnp_pouring_cup_physical(
                cup_name=str(skill.get("object_name", "obj_container")),
                ball_name=str(skill.get("ball_name", "ball_obj")),
                target_name=str(skill.get("target_name", "container")),
                arm=str(skill.get("arm", "right")),
                preset=str(skill.get("hand_pose_family", "cylindrical")),
                approach_steps=18 if smoke else 30,
                close_steps=16 if smoke else 22,
                lift_steps=12 if smoke else 18,
                move_steps=32 if smoke else 95,
                pour_steps=12 if smoke else 30,
                final_pour_joint_steps=14 if smoke else 55,
            )
        if skill_type == "two_arm_lift":
            return self.lift_pot_by_handles(
                preset=str(skill.get("hand_pose_family", "hook")),
                plan=skill.get("plan") if isinstance(skill.get("plan"), dict) else None,
                approach_steps=12 if smoke else 30,
                grasp_steps=12 if smoke else 30,
                close_steps=10 if smoke else 24,
                lift_steps=16 if smoke else 55,
                scale=0.035 if smoke else 0.04,
            )
        if skill_type == "drawer_pull":
            arm = str(skill.get("arm", "right"))
            preset = str(skill.get("hand_pose_family", "hook"))
            handle_pos = np.asarray(skill.get("handle_pos", self._eef_world_pos(arm)), dtype=np.float64).reshape(3)
            visual_grasp = skill.get("visual_grasp") if isinstance(skill.get("visual_grasp"), dict) else None
            pregrasp_pos = None
            wrist_grasp_pos = None
            if skill.get("wrist_grasp_pos") is not None:
                wrist_grasp_pos = np.asarray(skill["wrist_grasp_pos"], dtype=np.float64).reshape(3)
            if skill.get("wrist_pregrasp_pos") is not None:
                pregrasp_pos = np.asarray(skill["wrist_pregrasp_pos"], dtype=np.float64).reshape(3)
            if visual_grasp is not None and "pregrasp_pos" in visual_grasp:
                pregrasp_pos = pregrasp_pos if pregrasp_pos is not None else np.asarray(visual_grasp["pregrasp_pos"], dtype=np.float64).reshape(3)
            wrist_axis_angle_delta = np.asarray(
                skill.get("wrist_axis_angle_delta", [0.0, 0.0, 0.0]),
                dtype=np.float64,
            ).reshape(-1)
            if wrist_axis_angle_delta.size < 3:
                wrist_axis_angle_delta = np.pad(wrist_axis_angle_delta, (0, 3 - wrist_axis_angle_delta.size), mode="constant")
            wrist_axis_angle_delta = np.clip(wrist_axis_angle_delta[:3], -0.9, 0.9)
            wrist_target_quat_xyzw: np.ndarray | None = None
            if skill.get("wrist_target_quat_xyzw") is not None:
                wrist_target_quat_xyzw = np.asarray(skill["wrist_target_quat_xyzw"], dtype=np.float64).reshape(4)
            pull_axis = np.asarray(skill.get("pull_axis", [0.0, -1.0, 0.0]), dtype=np.float64).reshape(3)
            if float(np.linalg.norm(pull_axis)) > 1e-9:
                pull_axis = pull_axis / float(np.linalg.norm(pull_axis))
            handle_before = self._copy_drawer_handle_pose(skill)
            eef_before = self._eef_world_pos(arm)
            eef_quat_before = self._eef_world_quat_xyzw(arm)
            if wrist_target_quat_xyzw is None and float(np.linalg.norm(wrist_axis_angle_delta)) > 1e-9:
                wrist_target_quat_xyzw = self._offset_world_quat_xyzw(
                    eef_quat_before,
                    wrist_axis_angle_delta,
                    frame=str(skill.get("wrist_orientation_frame", "world")),
                )
            self.open_hand(arm=arm, steps=3 if smoke else 6)
            if pregrasp_pos is not None:
                self.move_towards_position(
                    pregrasp_pos,
                    arm=arm,
                    steps=10 if smoke else 24,
                    scale=0.045,
                    wrist_axis_angle_delta=wrist_axis_angle_delta,
                    wrist_target_quat_xyzw=wrist_target_quat_xyzw,
                )
            else:
                self.move_towards_position(
                    handle_pos + np.array([0.0, 0.0, 0.06]),
                    arm=arm,
                    steps=8 if smoke else 18,
                    scale=0.045,
                    wrist_axis_angle_delta=wrist_axis_angle_delta,
                    wrist_target_quat_xyzw=wrist_target_quat_xyzw,
                )
            contact_target = wrist_grasp_pos if wrist_grasp_pos is not None else handle_pos
            approach_summary = self.move_towards_position(
                contact_target,
                arm=arm,
                steps=10 if smoke else 24,
                scale=0.035,
                wrist_axis_angle_delta=wrist_axis_angle_delta,
                wrist_target_quat_xyzw=wrist_target_quat_xyzw,
            )
            eef_after_approach = self._eef_world_pos(arm)
            eef_quat_after_approach = self._eef_world_quat_xyzw(arm)
            close_summary = self._set_hand_while_holding(
                arm=arm,
                preset=preset,
                steps=8 if smoke else 18,
                wrist_axis_angle_delta=wrist_axis_angle_delta,
                wrist_target_quat_xyzw=wrist_target_quat_xyzw,
            )
            eef_after_close = self._eef_world_pos(arm)
            eef_quat_after_close = self._eef_world_quat_xyzw(arm)
            pull_summary = self.step_delta_action(
                arm=arm,
                delta_xyz=pull_axis,
                steps=10 if smoke else 40,
                scale=0.025 if smoke else 0.045,
                hand_preset=preset,
                wrist_axis_angle_delta=wrist_axis_angle_delta,
                wrist_target_quat_xyzw=wrist_target_quat_xyzw,
            )
            handle_after = self._copy_drawer_handle_pose(skill)
            eef_after_pull = self._eef_world_pos(arm)
            eef_quat_after_pull = self._eef_world_quat_xyzw(arm)
            final_state = self.get_task_state()
            return {
                "skill_type": skill_type,
                "handle_affordance": skill.get("handle_affordance"),
                "visual_grasp": visual_grasp,
                "wrist_grasp_pos": wrist_grasp_pos,
                "wrist_pregrasp_pos": pregrasp_pos,
                "wrist_axis_angle_delta": wrist_axis_angle_delta,
                "wrist_target_quat_xyzw": wrist_target_quat_xyzw,
                "wrist_orientation_frame": str(skill.get("wrist_orientation_frame", "world")),
                "handle_before": handle_before,
                "handle_after": handle_after,
                "handle_delta": None
                if handle_before is None or handle_after is None
                else np.asarray(handle_after["pos"], dtype=np.float64) - np.asarray(handle_before["pos"], dtype=np.float64),
                "eef_before": eef_before,
                "eef_quat_before": eef_quat_before,
                "eef_after_approach": eef_after_approach,
                "eef_quat_after_approach": eef_quat_after_approach,
                "eef_after_close": eef_after_close,
                "eef_quat_after_close": eef_quat_after_close,
                "eef_after_pull": eef_after_pull,
                "eef_quat_after_pull": eef_quat_after_pull,
                "eef_to_handle_after_approach": float(np.linalg.norm(eef_after_approach - handle_pos)),
                "eef_to_handle_after_close": float(np.linalg.norm(eef_after_close - handle_pos)),
                "approach_summary": approach_summary,
                "close_summary": close_summary,
                "pull_summary": pull_summary,
                "final_state": final_state,
                "task_completed": bool(final_state.get("task_completed", False)),
                "reward": float(final_state.get("reward", 0.0)),
            }
        raise ValueError(f"Unsupported skill_type={skill_type!r}")

    def verify_success_or_contact(
        self,
        result: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Return the shared verification payload for GR1 skill execution."""
        state = self.get_task_state()
        result = result or {}
        return {
            "task_completed": bool(state.get("task_completed", False)),
            "reward": float(state.get("reward", 0.0)),
            "success_flags": state.get("success_flags", {}),
            "result_task_completed": bool(result.get("task_completed", False)) if isinstance(result, dict) else False,
            "result_reward": float(result.get("reward", 0.0)) if isinstance(result, dict) else 0.0,
            "objects": state.get("objects", {}),
            "first_failure_phase": result.get("phase") if isinstance(result, dict) else None,
        }

    def validate_hand_presets(
        self,
        arm: str = "both",
        presets: tuple[str, ...] | list[str] = ("open", "power", "cylindrical", "pinch", "hook", "support"),
        steps_per_preset: int = 8,
    ) -> dict[str, Any]:
        """Exercise Fourier hand presets and return before/after joint snapshots.

        This is the Phase-0 harness: it verifies that the preset commands reach
        the simulator and perturb hand joints before we rely on any grasp logic.
        """
        selected = self._selected_arms(arm)
        results: dict[str, Any] = {
            "action_layout": self.get_action_layout(),
            "before": self.get_fourier_hand_state(arm),
            "presets": {},
        }
        for preset in presets:
            preset_name = str(preset).lower().strip()
            before = self.get_fourier_hand_state(arm)
            summary = self.set_hand(arm=arm, preset=preset_name, strength=1.0, steps=steps_per_preset)
            after = self.get_fourier_hand_state(arm)
            results["presets"][preset_name] = {
                "command": self._hand_preset(preset_name).copy(),
                "before": before,
                "after": after,
                "max_abs_joint_delta": self._max_hand_delta(before, after, selected),
                "summary": summary,
            }
        results["after"] = self.get_fourier_hand_state(arm)
        return results

    def diagnose_arm_motion_response(
        self,
        arm: str = "right",
        steps: int = 8,
        scale: float = 0.05,
    ) -> dict[str, Any]:
        """Measure how requested XYZ directions move the live GR1 EEF.

        The current RoboCasa GR1 WholeBodyIK surface is not guaranteed to align
        with world-frame deltas. This diagnostic restores the same simulator
        state before each direction probe and records actual EEF displacement.
        """
        selected = "right" if str(arm).lower().strip() in {"auto", "nearest", "both"} else str(arm).lower().strip()
        if selected not in {"left", "right"}:
            selected = "right"
        if not hasattr(self._env, "capture_state") or not hasattr(self._env, "restore_state"):
            raise RuntimeError("Underlying env does not support state capture/restore")
        baseline = self._env.capture_state()
        directions: dict[str, list[float]] = {
            "+x": [1.0, 0.0, 0.0],
            "-x": [-1.0, 0.0, 0.0],
            "+y": [0.0, 1.0, 0.0],
            "-y": [0.0, -1.0, 0.0],
            "+z": [0.0, 0.0, 1.0],
            "-z": [0.0, 0.0, -1.0],
        }
        probes: dict[str, Any] = {}
        try:
            for name, vector in directions.items():
                self._env.restore_state(baseline)
                before = self._eef_world_pos(selected)
                summary = self.step_delta_action(arm=selected, delta_xyz=vector, steps=steps, scale=scale)
                after = self._eef_world_pos(selected)
                actual = after - before
                requested = np.asarray(vector, dtype=np.float64)
                probes[name] = {
                    "requested_delta": requested.copy(),
                    "before_eef_pos": before.copy(),
                    "after_eef_pos": after.copy(),
                    "actual_delta": actual.copy(),
                    "actual_norm": float(np.linalg.norm(actual)),
                    "dot_with_requested": float(np.dot(actual, requested)),
                    "summary": summary,
                }
        finally:
            self._env.restore_state(baseline)
        return {
            "arm": selected,
            "steps": int(steps),
            "scale": float(scale),
            "probes": probes,
            "baseline_eef_pos": self._eef_world_pos(selected),
        }

    def step_action(self, action: np.ndarray | list[float]) -> dict[str, Any]:
        """Step the simulator with one full low-level action.

        Args:
            action: Full action vector. For the current GR1 RoboCasa setup this
                is usually 24 values. Values are clipped to ``[-1, 1]`` for
                safety before stepping.

        Returns:
            Step summary containing reward, termination flags and task state.
        """
        action_arr = self._safe_full_action(action)
        obs, reward, terminated, truncated, info = self._env.step(action_arr)
        return self._step_summary(obs, reward, terminated, truncated, info)

    def step_delta_action(
        self,
        arm: str = "both",
        delta_xyz: np.ndarray | list[float] | tuple[float, float, float] | None = None,
        gripper: float = 0.0,
        steps: int = 1,
        scale: float = 0.15,
        hand_preset: str | None = None,
        hand_strength: float = 1.0,
        wrist_axis_angle_delta: np.ndarray | list[float] | tuple[float, float, float] | None = None,
        wrist_target_quat_xyzw: np.ndarray | list[float] | tuple[float, float, float, float] | None = None,
    ) -> dict[str, Any]:
        """Apply a conservative Cartesian-like delta action for one or both arms.

        For Mink IK this sends absolute controller-frame EEF targets, recomputed from
        the current world-frame EEF pose at each step. This is intentionally
        different from the earlier relative-delta heuristic, which was not
        aligned with the controller contract and made directions hard to reason
        about.

        Args:
            arm: ``"left"``, ``"right"`` or ``"both"``.
            delta_xyz: Desired small direction vector. It is normalized/clipped.
            gripper: Scalar gripper command in ``[-1, 1]`` placed at likely
                gripper control slots when available.
            steps: Number of simulator steps to repeat the action.
            scale: Maximum per-axis action magnitude.

        Returns:
            Final step summary after repeated actions.
        """
        delta = np.zeros(3, dtype=np.float64) if delta_xyz is None else np.asarray(delta_xyz, dtype=np.float64).reshape(3)
        selected = self._selected_arms(arm)
        summary: dict[str, Any] | None = None
        for _ in range(max(1, int(steps))):
            action = self._hold_pose_action()
            for arm_name in selected:
                current = self._eef_world_pos(arm_name)
                step_delta = self._clipped_step_delta(delta, max_step=float(scale))
                target = current + step_delta
                self._set_arm_absolute_target(action, arm_name, target)
                if wrist_target_quat_xyzw is not None:
                    self._set_arm_absolute_orientation(action, arm_name, np.asarray(wrist_target_quat_xyzw, dtype=np.float64).reshape(4))
                elif wrist_axis_angle_delta is not None:
                    self._add_arm_axis_angle_delta(action, arm_name, wrist_axis_angle_delta)
                if hand_preset is not None:
                    self._set_hand_action(action, arm_name, hand_preset, hand_strength)
                elif gripper != 0.0:
                    grip_slice = self._action_slice(f"{arm_name}_gripper")
                    action[grip_slice[0] : grip_slice[1]] = self._hand_preset("power", strength=gripper)
            summary = self._repeat_action(action, steps=1)
            if summary["terminated"] or summary["truncated"] or summary["task_completed"]:
                break
        assert summary is not None
        return summary

    def hold_still(self, steps: int = 1) -> dict[str, Any]:
        """Step zero action for a small number of steps.

        Args:
            steps: Number of no-op simulator steps.

        Returns:
            Final step summary.
        """
        return self._repeat_action(np.zeros(self._action_dim(), dtype=np.float64), steps=steps)

    def move_towards_object(
        self,
        object_name: str,
        arm: str = "both",
        steps: int = 5,
        scale: float = 0.08,
        hand_preset: str | None = None,
        hand_strength: float = 1.0,
        target_z_offset: float = 0.0,
    ) -> dict[str, Any]:
        """Move selected arm command channels in the direction of a named object.

        Args:
            object_name: Target object name accepted by ``get_object_pose``.
            arm: ``"left"``, ``"right"`` or ``"both"``. If ``"both"``, the
                nearest exposed end-effector to the object is selected.
            steps: Number of repeated low-level steps.
            scale: Maximum action magnitude.

        Returns:
            Final step summary.
        """
        target_pos, _ = self.get_object_pose(object_name)
        target_pos = np.asarray(target_pos, dtype=np.float64).reshape(3).copy()
        target_pos[2] += float(target_z_offset)
        robot = self.get_robot_state()
        selected_arm = arm
        if arm == "both":
            left = robot.get("robot0_left_eef_pos")
            right = robot.get("robot0_right_eef_pos")
            if left is not None and right is not None:
                selected_arm = "left" if np.linalg.norm(target_pos - left[:3]) <= np.linalg.norm(target_pos - right[:3]) else "right"
        eef_key = "robot0_left_eef_pos" if selected_arm == "left" else "robot0_right_eef_pos"
        eef_pos = robot.get(eef_key)
        if eef_pos is None:
            # Fall back to task-provided gripper vector if direct EEF pose is absent.
            vector_key = "gripper0_to_payload" if selected_arm == "left" else "gripper1_to_payload"
            delta = robot.get(vector_key, np.zeros(3, dtype=np.float32))
        else:
            delta = target_pos - np.asarray(eef_pos[:3], dtype=np.float32)
        return self.move_towards_position(
            target_pos,
            arm=selected_arm,
            steps=steps,
            scale=scale,
            hand_preset=hand_preset,
            hand_strength=hand_strength,
        )

    def move_towards_position(
        self,
        position: np.ndarray | list[float] | tuple[float, float, float],
        arm: str = "right",
        steps: int = 5,
        scale: float = 0.08,
        hand_preset: str | None = None,
        hand_strength: float = 1.0,
        wrist_axis_angle_delta: np.ndarray | list[float] | tuple[float, float, float] | None = None,
        wrist_target_quat_xyzw: np.ndarray | list[float] | tuple[float, float, float, float] | None = None,
    ) -> dict[str, Any]:
        """Move selected arm toward an explicit world-frame position.

        Args:
            position: Target XYZ in the same frame as RoboCasa object poses.
            arm: ``"left"``, ``"right"`` or ``"both"``.
            steps: Number of repeated control steps.
            scale: Maximum per-step Cartesian intent magnitude.
        """
        target_pos = np.asarray(position, dtype=np.float64).reshape(3)
        selected_arm = "right" if str(arm).lower().strip() in {"auto", "nearest", "both"} else str(arm).lower().strip()
        summary: dict[str, Any] | None = None
        for _ in range(max(1, int(steps))):
            current = self._eef_world_pos(selected_arm)
            delta = target_pos - current
            if float(np.linalg.norm(delta)) < 1e-3:
                break
            action = self._hold_pose_action()
            step_target = current + self._clipped_step_delta(delta, max_step=float(scale))
            self._set_arm_absolute_target(action, selected_arm, step_target)
            if wrist_target_quat_xyzw is not None:
                self._set_arm_absolute_orientation(action, selected_arm, np.asarray(wrist_target_quat_xyzw, dtype=np.float64).reshape(4))
            elif wrist_axis_angle_delta is not None:
                self._add_arm_axis_angle_delta(action, selected_arm, wrist_axis_angle_delta)
            if hand_preset is not None:
                self._set_hand_action(action, selected_arm, hand_preset, hand_strength)
            summary = self._repeat_action(action, steps=1)
            if summary["terminated"] or summary["truncated"] or summary["task_completed"]:
                break
        if summary is None:
            summary = self.hold_still(steps=1)
        return summary

    def set_hand(
        self,
        arm: str = "right",
        preset: str = "open",
        strength: float = 1.0,
        steps: int = 8,
    ) -> dict[str, Any]:
        """Apply a Fourier hand preset while keeping the arms still.

        Args:
            arm: ``"left"``, ``"right"`` or ``"both"``.
            preset: ``"open"``, ``"power"``, ``"pinch"`` or ``"hook"``.
            strength: Multiplier in ``[-1, 1]``. Positive closes toward the
                preset; negative opens away from it.
            steps: Number of simulator steps to repeat the command.

        Returns:
            Final step summary.
        """
        # Hand-only commands should leave arm channels at zero. The current GR1
        # controller does not treat our reconstructed absolute hold-pose action
        # as a stable no-op, so using it here makes the wrists drift while only
        # trying to open/close the hands.
        action = np.zeros(self._action_dim(), dtype=np.float64)
        for arm_name in self._selected_arms(arm):
            start, end = self._action_slice(f"{arm_name}_gripper")
            if end > start:
                action[start:end] = self._hand_preset(preset, strength=strength)[: end - start]
        return self._repeat_action(action, steps=steps)

    def set_hand_preset(
        self,
        arm: str = "right",
        preset: str = "open",
        strength: float = 1.0,
        steps: int = 8,
    ) -> dict[str, Any]:
        """Alias for ``set_hand`` with a model-facing name."""
        return self.set_hand(arm=arm, preset=preset, strength=strength, steps=steps)

    def open_hand(self, arm: str = "both", steps: int = 8) -> dict[str, Any]:
        """Open one or both Fourier hands."""
        return self.set_hand(arm=arm, preset="open", strength=1.0, steps=steps)

    def close_hand(
        self,
        arm: str = "right",
        preset: str = "power",
        strength: float = 1.0,
        steps: int = 10,
    ) -> dict[str, Any]:
        """Close one Fourier hand using a named grasp preset."""
        return self.set_hand(arm=arm, preset=preset, strength=strength, steps=steps)

    def _set_hand_while_holding(
        self,
        arm: str = "right",
        preset: str = "hook",
        strength: float = 1.0,
        steps: int = 10,
        wrist_axis_angle_delta: np.ndarray | list[float] | tuple[float, float, float] | None = None,
        wrist_target_quat_xyzw: np.ndarray | list[float] | tuple[float, float, float, float] | None = None,
    ) -> dict[str, Any]:
        """Apply a hand preset while commanding the current wrist pose.

        The generic hand-only helper intentionally uses zero arm action, which
        can let the GR1 wrist drift away from a handle during closure. Drawer
        pulling needs closure at the reached pose, so use a hold-pose action here.
        """
        summary: dict[str, Any] | None = None
        for _ in range(max(1, int(steps))):
            action = self._hold_pose_action()
            for arm_name in self._selected_arms(arm):
                if wrist_target_quat_xyzw is not None:
                    self._set_arm_absolute_orientation(action, arm_name, np.asarray(wrist_target_quat_xyzw, dtype=np.float64).reshape(4))
                elif wrist_axis_angle_delta is not None:
                    self._add_arm_axis_angle_delta(action, arm_name, wrist_axis_angle_delta)
                self._set_hand_action(action, arm_name, preset, strength)
            summary = self._repeat_action(action, steps=1)
            if summary["terminated"] or summary["truncated"] or summary["task_completed"]:
                break
        assert summary is not None
        return summary

    def grasp_object(
        self,
        object_name: str,
        arm: str = "auto",
        preset: str = "power",
        approach_steps: int = 20,
        close_steps: int = 12,
        approach_scale: float = 0.08,
        approach_height: float = 0.10,
        grasp_height_offset: float = 0.025,
        grasp_position_offset: tuple[float, float, float] | list[float] | np.ndarray | None = None,
    ) -> dict[str, Any]:
        """Run a geometric Fourier-hand grasp primitive.

        This is a first physical-control baseline: choose a hand, stage above
        the object, descend toward a grasp pose, close a preset, then lift. It
        does not teleport objects and does not solve full-body IK.
        """
        plan = self.plan_fourier_grasp(
            object_name=object_name,
            arm=arm,
            preset=preset,
            approach_height=approach_height,
            grasp_height_offset=grasp_height_offset,
            grasp_position_offset=grasp_position_offset,
        )
        selected = str(plan["arm"])
        preset_name = str(plan["preset"])
        self.open_hand(arm=selected, steps=max(2, min(6, close_steps // 2)))
        self.move_towards_position(plan["pregrasp_pos"], arm=selected, steps=max(1, approach_steps // 2), scale=approach_scale)
        summary = self.move_towards_position(plan["grasp_pos"], arm=selected, steps=max(1, approach_steps // 2), scale=approach_scale)
        if not summary["task_completed"]:
            summary = self.close_hand(arm=selected, preset=preset_name, steps=close_steps)
        if not summary["task_completed"]:
            summary = self.lift_hand(
                arm=selected,
                steps=max(3, close_steps // 2),
                scale=min(0.08, approach_scale),
                hand_preset=preset_name,
            )
        summary["object"] = str(plan["object"])
        summary["arm"] = selected
        summary["preset"] = preset_name
        summary["grasp_plan"] = plan
        return summary

    def plan_two_arm_lift_handles(
        self,
        x_offset: float = 0.06,
        lateral_offset: float = 0.04,
        z_offset: float = -0.01,
        approach_height: float = 0.07,
        preset: str = "hook",
        scene: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Plan a two-hand hook grasp around the TwoArmLift pot handles.

        The default Transport object sits outside the GR1 reachable workspace.
        TwoArmLift exposes pot handles near the table center, so this primitive
        targets the MuJoCo handle geoms directly instead of the pot center.
        """
        handles = self._select_two_arm_lift_handles(scene=scene)
        left_handle = np.asarray(handles["left_handle_pos"], dtype=np.float64).reshape(3)
        right_handle = np.asarray(handles["right_handle_pos"], dtype=np.float64).reshape(3)
        left_grasp = left_handle + np.array([float(x_offset), float(lateral_offset), float(z_offset)], dtype=np.float64)
        right_grasp = right_handle + np.array([float(x_offset), -float(lateral_offset), float(z_offset)], dtype=np.float64)
        return {
            "task": "TwoArmLift",
            "left_handle_geom": handles["left_handle_name"],
            "right_handle_geom": handles["right_handle_name"],
            "left_handle_name": handles["left_handle_name"],
            "right_handle_name": handles["right_handle_name"],
            "left_handle_pos": left_handle,
            "right_handle_pos": right_handle,
            "left_pregrasp_pos": left_grasp + np.array([0.0, 0.0, float(approach_height)], dtype=np.float64),
            "right_pregrasp_pos": right_grasp + np.array([0.0, 0.0, float(approach_height)], dtype=np.float64),
            "left_grasp_pos": left_grasp,
            "right_grasp_pos": right_grasp,
            "x_offset": float(x_offset),
            "lateral_offset": float(lateral_offset),
            "z_offset": float(z_offset),
            "approach_height": float(approach_height),
            "preset": str(preset),
            "hand_command": self._hand_preset(str(preset)).copy(),
            "handle_separation": float(np.linalg.norm(left_handle - right_handle)),
            "selection_diagnostics": handles.get("selection_diagnostics", {}),
        }

    def _select_two_arm_lift_handles(self, scene: dict[str, Any] | None = None) -> dict[str, Any]:
        """Select the two pot handles from scene geometry with fallback names."""
        ranked = self._rank_scene_affordances(
            ("pot_handle",),
            include_geoms=True,
            prefer_names=("_c", "default_site", "pot_handle"),
            avoid_names=("visual", "connector"),
            z_range=(0.82, 1.08),
            scene=scene,
        )
        unique: list[dict[str, Any]] = []
        seen: set[str] = set()
        for item in ranked:
            name = str(item.get("name", ""))
            # Prefer handle center entries when available; fall back to sites.
            if name.endswith("_vis") or "connector" in name or "visual" in name:
                continue
            family = "handle1" if "handle1" in name else "handle0" if "handle0" in name else name
            if family in seen:
                continue
            seen.add(family)
            unique.append(item)
            if len(unique) >= 2:
                break
        if len(unique) < 2:
            left_pos = self._geom_world_pos("pot_handle1_c")
            right_pos = self._geom_world_pos("pot_handle0_c")
            return {
                "left_handle_name": "pot_handle1_c",
                "right_handle_name": "pot_handle0_c",
                "left_handle_pos": left_pos,
                "right_handle_pos": right_pos,
                "selection_diagnostics": {"source": "fallback_hardcoded_geoms", "ranked_candidates": ranked[:6]},
            }
        first, second = unique[0], unique[1]
        # In RoboCasa's GR1 frame, positive-Y handle is closest to the left hand.
        left, right = (first, second) if float(first["pos"][1]) >= float(second["pos"][1]) else (second, first)
        return {
            "left_handle_name": str(left["name"]),
            "right_handle_name": str(right["name"]),
            "left_handle_pos": np.asarray(left["pos"], dtype=np.float64).reshape(3).copy(),
            "right_handle_pos": np.asarray(right["pos"], dtype=np.float64).reshape(3).copy(),
            "selection_diagnostics": {
                "source": "ranked_scene_affordances",
                "ranked_candidates": ranked[:8],
            },
        }

    def lift_pot_by_handles(
        self,
        x_offset: float = 0.06,
        lateral_offset: float = 0.04,
        z_offset: float = -0.01,
        preset: str = "hook",
        plan: dict[str, Any] | None = None,
        approach_steps: int = 30,
        grasp_steps: int = 30,
        close_steps: int = 24,
        lift_steps: int = 55,
        scale: float = 0.04,
    ) -> dict[str, Any]:
        """Attempt a physical TwoArmLift grasp by targeting both pot handles.

        Returns the final task state plus the measured pot-root displacement.
        This is a smoke primitive for controller tuning, not a solved policy.
        """
        pot_before = self._body_world_pos("pot_root")
        plan = plan or self.plan_two_arm_lift_handles(
            x_offset=x_offset,
            lateral_offset=lateral_offset,
            z_offset=z_offset,
            preset=preset,
        )
        self.open_hand(arm="both", steps=4)
        self._move_bimanual_towards(
            plan["left_pregrasp_pos"],
            plan["right_pregrasp_pos"],
            steps=approach_steps,
            scale=max(float(scale), 0.045),
        )
        self._move_bimanual_towards(
            plan["left_grasp_pos"],
            plan["right_grasp_pos"],
            steps=grasp_steps,
            scale=float(scale),
        )
        close_summary = self.close_hand(arm="both", preset=preset, steps=close_steps)
        lift_summary = self.step_delta_action(arm="both", delta_xyz=[0.0, 0.0, 1.0], steps=lift_steps, scale=float(scale))
        pot_after = self._body_world_pos("pot_root")
        final_state = self.get_task_state()
        return {
            "plan": plan,
            "pot_before": pot_before,
            "pot_after": pot_after,
            "pot_delta": pot_after - pot_before,
            "close_summary": close_summary,
            "lift_summary": lift_summary,
            "final_state": final_state,
            "task_completed": bool(final_state.get("task_completed", False)),
            "reward": float(final_state.get("reward", 0.0)),
        }

    def plan_fourier_grasp(
        self,
        object_name: str,
        arm: str = "auto",
        preset: str = "auto",
        approach_height: float = 0.10,
        grasp_height_offset: float = 0.025,
        grasp_position_offset: tuple[float, float, float] | list[float] | np.ndarray | None = None,
    ) -> dict[str, Any]:
        """Plan a simple world-frame Fourier hand grasp around an object pose.

        The output is intentionally transparent so failures can be diagnosed
        without looking inside the simulator.
        """
        object_key = self._canonical_object_name(object_name)
        object_pos, object_quat = self.get_object_pose(object_key)
        selected = self._nearest_arm(object_key) if str(arm).lower().strip() in {"auto", "nearest", "both"} else str(arm).lower().strip()
        preset_name = self._choose_grasp_preset(object_key, preset)
        grasp_pos = np.asarray(object_pos, dtype=np.float64).reshape(3).copy()
        grasp_pos[2] += float(grasp_height_offset)
        position_offset = np.zeros(3, dtype=np.float64)
        if grasp_position_offset is not None:
            offset = np.asarray(grasp_position_offset, dtype=np.float64).reshape(-1)
            if offset.size < 3:
                offset = np.pad(offset, (0, 3 - offset.size), mode="constant")
            position_offset = offset[:3].copy()
            grasp_pos += position_offset
        pregrasp_pos = grasp_pos.copy()
        pregrasp_pos[2] += float(approach_height)
        lift_pos = grasp_pos.copy()
        lift_pos[2] += max(float(approach_height), 0.12)
        return {
            "object": object_key,
            "arm": selected,
            "preset": preset_name,
            "object_pos": object_pos,
            "object_quat": object_quat,
            "pregrasp_pos": pregrasp_pos,
            "grasp_pos": grasp_pos,
            "lift_pos": lift_pos,
            "grasp_position_offset": position_offset,
            "hand_command": self._hand_preset(preset_name).copy(),
            "assumptions": [
                "world-frame object pose is usable as a wrist staging target",
                "WholeBodyIK can approximately move the selected wrist toward staged positions",
                "preset closure is enough to create first-contact grasp data",
            ],
        }

    def lift_hand(
        self,
        arm: str = "right",
        steps: int = 15,
        scale: float = 0.06,
        hand_preset: str | None = None,
        hand_strength: float = 1.0,
    ) -> dict[str, Any]:
        """Lift the selected hand upward after a grasp attempt."""
        selected = "right" if str(arm).lower().strip() in {"auto", "nearest", "both"} else str(arm).lower().strip()
        summary: dict[str, Any] | None = None
        for _ in range(max(1, int(steps))):
            current = self._eef_world_pos(selected)
            action = self._hold_pose_action()
            self._set_arm_absolute_target(action, selected, current + np.array([0.0, 0.0, float(scale)], dtype=np.float64))
            if hand_preset is not None:
                self._set_hand_action(action, selected, hand_preset, hand_strength)
            summary = self._repeat_action(action, steps=1)
            if summary["terminated"] or summary["truncated"] or summary["task_completed"]:
                break
        assert summary is not None
        return summary

    def place_object(
        self,
        target_name: str = "target_bin",
        arm: str = "right",
        object_name: str | None = None,
        holding_preset: str = "power",
        lift_steps: int = 12,
        move_steps: int = 30,
        release_steps: int = 10,
        scale: float = 0.08,
        target_z_offset: float = 0.12,
        release_z_offset: float = 0.06,
        release_distance_threshold: float | None = None,
        target_offsets: tuple[tuple[float, float, float], ...] | None = None,
    ) -> dict[str, Any]:
        """Move a grasped object toward a target and release it.

        Args:
            target_name: Target receptacle such as ``target_bin`` or
                ``trash_bin``.
            arm: Carrying hand. Use the same arm chosen by ``grasp_object``.
            lift_steps: Steps to move upward before translation.
            move_steps: Steps to move toward the target pose.
            release_steps: Steps to open the hand at the target.
            scale: Maximum per-axis command magnitude.

        Returns:
            Final step summary.
        """
        selected = "right" if str(arm).lower().strip() in {"auto", "nearest", "both"} else str(arm).lower().strip()
        if int(lift_steps) > 0:
            self.lift_hand(
                arm=selected,
                steps=lift_steps,
                scale=scale,
                hand_preset=holding_preset,
            )
        if object_name is not None:
            can_restore = hasattr(self._env, "capture_state") and hasattr(self._env, "restore_state")
            offsets = target_offsets or ((0.0, 0.0, 0.0),)
            if can_restore and len(offsets) > 1:
                start_state = self._env.capture_state()
                best_state = start_state
                best_distance = float("inf")
                best_summary: dict[str, Any] | None = None
                sweep_results: list[dict[str, Any]] = []
                for offset_idx, offset in enumerate(offsets):
                    # Preserve live contact/velocity continuity for the first
                    # candidate. RoboCasa GR1 PnPPouring currently benefits
                    # from contact-rich pushing; restoring before candidate 0
                    # can erase that transient state and regress transport.
                    if offset_idx > 0:
                        self._env.restore_state(start_state)
                    candidate = self.transport_held_object_towards_target(
                        object_name=object_name,
                        target_name=target_name,
                        arm=selected,
                        steps=move_steps,
                        scale=min(float(scale), 0.045),
                        holding_preset=holding_preset,
                        target_z_offset=float(target_z_offset),
                        target_offset=offset,
                    )
                    object_pos, _ = self.get_object_pose(object_name)
                    target_pos, _ = self.get_object_pose(target_name)
                    distance = float(
                        np.linalg.norm(
                            np.asarray(object_pos, dtype=np.float64).reshape(3)
                            - np.asarray(target_pos, dtype=np.float64).reshape(3)
                        )
                    )
                    result = {
                        "target_offset": tuple(float(x) for x in offset),
                        "distance_to_target": distance,
                        "task_completed": bool(candidate.get("task_completed", False)),
                        "reward": float(candidate.get("reward", 0.0)),
                    }
                    sweep_results.append(result)
                    if distance < best_distance:
                        best_distance = distance
                        best_summary = dict(candidate)
                        best_summary["selected_target_offset"] = result["target_offset"]
                        best_summary["target_offset_sweep"] = sweep_results.copy()
                        best_state = self._env.capture_state()
                self._env.restore_state(best_state)
                summary = best_summary if best_summary is not None else self.hold_still(steps=1)
                summary["best_distance_to_target"] = float(best_distance)
                summary["target_offset_sweep"] = sweep_results
            else:
                summary = self.transport_held_object_towards_target(
                    object_name=object_name,
                    target_name=target_name,
                    arm=selected,
                    steps=move_steps,
                    scale=min(float(scale), 0.045),
                    holding_preset=holding_preset,
                    target_z_offset=float(target_z_offset),
                    target_offset=offsets[0],
                )
            summary["object_hand_alignment"] = self.measure_object_hand_alignment(object_name, selected)
        else:
            summary = self.move_towards_object(
                target_name,
                arm=selected,
                steps=move_steps,
                scale=scale,
                hand_preset=holding_preset,
                target_z_offset=target_z_offset,
            )
        should_release = not summary["task_completed"]
        if should_release and release_distance_threshold is not None and object_name is not None:
            object_pos, _ = self.get_object_pose(object_name)
            target_pos, _ = self.get_object_pose(target_name)
            distance = float(np.linalg.norm(np.asarray(object_pos, dtype=np.float64) - np.asarray(target_pos, dtype=np.float64)))
            summary["distance_to_target_before_release"] = distance
            should_release = distance <= float(release_distance_threshold)
        if should_release:
            summary = self.open_hand(arm=selected, steps=release_steps)
        elif not summary["task_completed"]:
            summary["release_skipped"] = True
        return summary

    def carry_object_towards_target(
        self,
        object_name: str,
        target_name: str,
        arm: str = "right",
        holding_preset: str = "power",
        steps: int = 50,
        scale: float = 0.06,
        target_z_offset: float = 0.16,
    ) -> dict[str, Any]:
        """Move the wrist by the observed object-to-target error while holding.

        Directly commanding the wrist to the target pose fails when the object
        is only partially captured by the hand. This helper closes the loop on
        the object pose: at every step it measures where the object actually is
        and moves the carrying wrist in the direction that would reduce the
        object's error to an above-target carry point.
        """
        object_key = self._canonical_object_name(object_name)
        target_key = self._canonical_object_name(target_name)
        selected = "right" if str(arm).lower().strip() in {"auto", "nearest", "both"} else str(arm).lower().strip()
        summary: dict[str, Any] | None = None
        for _ in range(max(1, int(steps))):
            object_pos, _ = self.get_object_pose(object_key)
            target_pos, _ = self.get_object_pose(target_key)
            desired_object_pos = np.asarray(target_pos, dtype=np.float64).reshape(3).copy()
            desired_object_pos[2] += float(target_z_offset)
            error = desired_object_pos - np.asarray(object_pos, dtype=np.float64).reshape(3)
            if float(np.linalg.norm(error[:2])) < 0.025 and abs(float(error[2])) < 0.06:
                break
            current = self._eef_world_pos(selected)
            action = self._hold_pose_action()
            step_target = current + self._clipped_step_delta(error, max_step=float(scale))
            self._set_arm_absolute_target(action, selected, step_target)
            self._set_hand_action(action, selected, holding_preset, 1.0)
            summary = self._repeat_action(action, steps=1)
            if summary["terminated"] or summary["truncated"] or summary["task_completed"]:
                break
        if summary is None:
            summary = self.hold_still(steps=1)
        summary["object"] = object_key
        summary["target"] = target_key
        summary["arm"] = selected
        summary["object_hand_alignment"] = self.measure_object_hand_alignment(object_key, selected)
        return summary

    def transport_held_object_towards_target(
        self,
        object_name: str,
        target_name: str,
        arm: str = "right",
        holding_preset: str = "power",
        steps: int = 90,
        scale: float = 0.02,
        target_z_offset: float = 0.08,
        patience: int = 6,
        min_improvement: float = 0.003,
        target_offset: tuple[float, float, float] | np.ndarray | None = None,
        restore_best_state: bool = True,
    ) -> dict[str, Any]:
        """Conservatively move a held object and restore the best distance state.

        The GR1 hand can push or sling the ball past the target. This transport
        loop measures object-to-target distance after every low-level step,
        captures the best simulator state, and restores it if later steps
        overshoot. This keeps the generated transition useful even when the
        final success predicate is not reached.
        """
        object_key = self._canonical_object_name(object_name)
        target_key = self._canonical_object_name(target_name)
        selected = "right" if str(arm).lower().strip() in {"auto", "nearest", "both"} else str(arm).lower().strip()
        can_restore = hasattr(self._env, "capture_state") and hasattr(self._env, "restore_state")
        target_pos, _ = self.get_object_pose(target_key)
        target_pos = np.asarray(target_pos, dtype=np.float64).reshape(3)
        offset = np.zeros(3, dtype=np.float64) if target_offset is None else np.asarray(target_offset, dtype=np.float64).reshape(3)
        command_target_pos = target_pos + offset

        def distance() -> float:
            object_pos, _ = self.get_object_pose(object_key)
            return float(np.linalg.norm(np.asarray(object_pos, dtype=np.float64).reshape(3) - target_pos))

        best_distance = distance()
        best_state = self._env.capture_state() if can_restore else None
        best_summary: dict[str, Any] | None = None
        stale_steps = 0
        summary: dict[str, Any] | None = None
        for _ in range(max(1, int(steps))):
            current = self._eef_world_pos(selected)
            wrist_target = command_target_pos.copy()
            wrist_target[2] += float(target_z_offset)
            action = self._hold_pose_action()
            step_target = current + self._clipped_step_delta(wrist_target - current, max_step=float(scale))
            self._set_arm_absolute_target(action, selected, step_target)
            self._set_hand_action(action, selected, holding_preset, 1.0)
            summary = self._repeat_action(action, steps=1)
            current_distance = distance()
            if current_distance + float(min_improvement) < best_distance:
                best_distance = current_distance
                best_summary = dict(summary)
                best_state = self._env.capture_state() if can_restore else None
                stale_steps = 0
            else:
                stale_steps += 1
            if summary["terminated"] or summary["truncated"] or summary["task_completed"]:
                break
            if stale_steps >= max(1, int(patience)):
                break
        if bool(restore_best_state) and can_restore and best_state is not None:
            self._env.restore_state(best_state)
            summary = best_summary if best_summary is not None else self.hold_still(steps=1)
        elif summary is None:
            summary = self.hold_still(steps=1)
        summary["object"] = object_key
        summary["target"] = target_key
        summary["arm"] = selected
        summary["target_offset"] = tuple(float(x) for x in offset)
        summary["best_distance_to_target"] = float(best_distance)
        summary["restore_best_state"] = bool(restore_best_state)
        summary["object_hand_alignment"] = self.measure_object_hand_alignment(object_key, selected)
        return summary

    def measure_object_hand_alignment(self, object_name: str, arm: str = "right") -> dict[str, Any]:
        """Return object, wrist and distance metrics for grasp diagnostics."""
        object_key = self._canonical_object_name(object_name)
        selected = "right" if str(arm).lower().strip() in {"auto", "nearest", "both"} else str(arm).lower().strip()
        object_pos, _ = self.get_object_pose(object_key)
        eef_pos = self._eef_world_pos(selected)
        delta = np.asarray(object_pos, dtype=np.float64).reshape(3) - np.asarray(eef_pos, dtype=np.float64).reshape(3)
        return {
            "object": object_key,
            "arm": selected,
            "object_pos": np.asarray(object_pos, dtype=np.float64).copy(),
            "eef_pos": np.asarray(eef_pos, dtype=np.float64).copy(),
            "object_to_eef_delta": delta,
            "object_to_eef_distance": float(np.linalg.norm(delta)),
            "vertical_gap_object_minus_eef": float(delta[2]),
        }

    def solve_pnp_pouring_physical(
        self,
        object_name: str = "ball_obj",
        target_name: str = "container",
        arm: str = "right",
        preset: str = "power",
        approach_steps: int = 30,
        close_steps: int = 18,
        lift_steps: int = 18,
        move_steps: int = 50,
        release_steps: int = 12,
        scale: float = 0.07,
        target_z_offset: float = 0.12,
    ) -> dict[str, Any]:
        """Physical PnPPouring baseline using scene-pose object aliases.

        The current PnPPouring task exposes ``ball_obj`` as the grasped object,
        ``obj_container`` as the source container, and ``container`` as the
        target. This helper keeps the Fourier hand closed while lifting and
        moving, which is critical for dexterous-hand contact data.
        """
        object_key = self._canonical_object_name(object_name)
        target_key = self._canonical_object_name(target_name)
        object_before, _ = self.get_object_pose(object_key)
        target_pos, _ = self.get_object_pose(target_key)
        grasp_attempts: list[dict[str, Any]] = []
        grasp_summary: dict[str, Any] | None = None
        selected = "right" if str(arm).lower().strip() in {"auto", "nearest", "both"} else str(arm).lower().strip()
        requested_params = {
            "approach_steps": int(approach_steps),
            "close_steps": int(close_steps),
            "lift_steps": int(lift_steps),
            "move_steps": int(move_steps),
            "release_steps": int(release_steps),
            "scale": float(scale),
            "target_z_offset": float(target_z_offset),
        }
        effective_params = {
            "approach_steps_first": max(10, int(approach_steps)),
            "approach_steps_retry": max(10, int(approach_steps) // 2),
            "close_steps": max(int(close_steps), 22),
            "lift_steps": max(0, int(lift_steps)),
            "move_steps": max(1, int(move_steps)),
            "release_steps": max(0, int(release_steps)),
            "grasp_scale": float(np.clip(float(scale), 0.01, 0.08)),
            "transport_scale": float(np.clip(float(scale), 0.005, 0.08)),
            "target_z_offset": float(np.clip(float(target_z_offset), 0.02, 0.25)),
            "release_xy_distance_threshold": 0.12,
            "contact_lift_threshold": 0.015,
            "contact_move_threshold": 0.03,
            "contact_alignment_threshold": 0.16,
            "contact_vertical_gap_min": -0.16,
        }
        can_restore = hasattr(self._env, "capture_state") and hasattr(self._env, "restore_state")
        start_state = self._env.capture_state() if can_restore else None
        best_attempt: dict[str, Any] | None = None
        best_state = start_state
        best_score = -float("inf")
        # Try progressively lower wrist staging targets. The previous default
        # often closed above the ball/source cup, producing contact without
        # capture. Keep this bounded so failures still generate useful data.
        for attempt_idx, grasp_height_offset in enumerate((0.005, -0.015, -0.035)):
            if can_restore and start_state is not None:
                self._env.restore_state(start_state)
            attempt_before, _ = self.get_object_pose(object_key)
            grasp_summary = self.grasp_object(
                object_name=object_key,
                arm=selected,
                preset=preset,
                approach_steps=(
                    int(effective_params["approach_steps_first"])
                    if attempt_idx == 0
                    else int(effective_params["approach_steps_retry"])
                ),
                close_steps=int(effective_params["close_steps"]),
                approach_scale=float(effective_params["grasp_scale"]),
                approach_height=0.13,
                grasp_height_offset=grasp_height_offset,
            )
            selected = str(grasp_summary.get("arm", selected))
            attempt_after, _ = self.get_object_pose(object_key)
            lifted = float(np.asarray(attempt_after, dtype=np.float64)[2] - np.asarray(attempt_before, dtype=np.float64)[2])
            moved = float(np.linalg.norm(np.asarray(attempt_after, dtype=np.float64) - np.asarray(attempt_before, dtype=np.float64)))
            target_distance = float(
                np.linalg.norm(np.asarray(attempt_after, dtype=np.float64) - np.asarray(target_pos, dtype=np.float64))
            )
            alignment = self.measure_object_hand_alignment(object_key, selected)
            alignment_distance = float(alignment["object_to_eef_distance"])
            vertical_gap = float(alignment["vertical_gap_object_minus_eef"])
            contact_likely = (
                lifted > float(effective_params["contact_lift_threshold"])
                or (
                    moved > float(effective_params["contact_move_threshold"])
                    and alignment_distance < float(effective_params["contact_alignment_threshold"])
                    and vertical_gap > float(effective_params["contact_vertical_gap_min"])
                )
                or bool(grasp_summary.get("task_completed", False))
            )
            grasp_score = (
                (100.0 if bool(grasp_summary.get("task_completed", False)) else 0.0)
                - target_distance
                - 0.35 * alignment_distance
                + 2.0 * max(lifted, 0.0)
                + 0.2 * moved
                + (1.0 if contact_likely else 0.0)
            )
            attempt = {
                "attempt": attempt_idx + 1,
                "grasp_height_offset": float(grasp_height_offset),
                "object_before": attempt_before,
                "object_after": attempt_after,
                "object_delta": np.asarray(attempt_after, dtype=np.float64) - np.asarray(attempt_before, dtype=np.float64),
                "object_moved_norm": moved,
                "object_lifted_z": lifted,
                "distance_to_target_after_grasp": target_distance,
                "contact_likely": bool(contact_likely),
                "grasp_score": grasp_score,
                "object_hand_alignment": alignment,
                "summary": grasp_summary,
            }
            grasp_attempts.append(attempt)
            if grasp_score > best_score:
                best_score = grasp_score
                best_attempt = attempt
                best_state = self._env.capture_state() if can_restore else None
            if bool(grasp_summary.get("task_completed", False)):
                break
            if contact_likely:
                break
            if not can_restore:
                self.open_hand(arm=selected, steps=max(3, int(close_steps) // 3))
                self.lift_hand(
                    arm=selected,
                    steps=4,
                    scale=min(float(effective_params["grasp_scale"]), 0.045),
                    hand_preset="open",
                )
        assert grasp_summary is not None
        if can_restore and best_state is not None:
            self._env.restore_state(best_state)
        if best_attempt is not None:
            grasp_summary = dict(best_attempt["summary"])
            selected = str(grasp_summary.get("arm", selected))
        selected = str(grasp_summary.get("arm", arm))
        phase = "transport"
        transport_summary: dict[str, Any] | None = None
        release_summary: dict[str, Any] | None = None
        contact_established = bool(best_attempt.get("contact_likely", False)) if best_attempt is not None else False
        if bool(grasp_summary.get("task_completed", False)):
            phase = "completed_after_grasp"
            place_summary = {"skipped": True, "reason": phase}
        elif not contact_established:
            phase = "grasp_failed"
            place_summary = {
                "skipped": True,
                "reason": "contact_not_established",
                "best_grasp_attempt": best_attempt,
            }
        else:
            transport_summary = self.transport_held_object_towards_target(
                object_name=object_key,
                target_name=target_key,
                arm=selected,
                holding_preset=str(grasp_summary.get("preset", preset)),
                steps=int(effective_params["move_steps"]),
                scale=float(effective_params["transport_scale"]),
                target_z_offset=float(effective_params["target_z_offset"]),
            )
            object_before_release, _ = self.get_object_pose(object_key)
            release_delta = (
                np.asarray(object_before_release, dtype=np.float64).reshape(3)
                - np.asarray(target_pos, dtype=np.float64).reshape(3)
            )
            distance_before_release = float(
                np.linalg.norm(release_delta)
            )
            xy_distance_before_release = float(np.linalg.norm(release_delta[:2]))
            vertical_offset_before_release = float(release_delta[2])
            should_release = (
                not bool(transport_summary.get("task_completed", False))
                and xy_distance_before_release <= float(effective_params["release_xy_distance_threshold"])
            )
            transport_summary["distance_to_target_before_release"] = distance_before_release
            transport_summary["xy_distance_to_target_before_release"] = xy_distance_before_release
            transport_summary["vertical_offset_to_target_before_release"] = vertical_offset_before_release
            if should_release:
                release_summary = self.open_hand(arm=selected, steps=int(effective_params["release_steps"]))
                phase = "released"
            else:
                release_summary = {
                    "release_skipped": True,
                    "reason": "outside_release_xy_threshold",
                    "distance_to_target_before_release": distance_before_release,
                    "xy_distance_to_target_before_release": xy_distance_before_release,
                    "vertical_offset_to_target_before_release": vertical_offset_before_release,
                    "release_xy_distance_threshold": float(effective_params["release_xy_distance_threshold"]),
                }
                phase = "transport_no_release"
            place_summary = dict(transport_summary)
            place_summary["release_summary"] = release_summary
        object_after, _ = self.get_object_pose(object_key)
        final_state = self.get_task_state()
        object_delta = np.asarray(object_after, dtype=np.float64) - np.asarray(object_before, dtype=np.float64)
        target_delta_after = np.asarray(object_after, dtype=np.float64).reshape(3) - np.asarray(target_pos, dtype=np.float64).reshape(3)
        return {
            "phase": phase,
            "object": object_key,
            "target": target_key,
            "arm": selected,
            "preset": str(grasp_summary.get("preset", preset)),
            "requested_params": requested_params,
            "effective_params": effective_params,
            "object_before": object_before,
            "object_after": object_after,
            "object_delta": object_delta,
            "distance_to_target": float(np.linalg.norm(target_delta_after)),
            "xy_distance_to_target": float(np.linalg.norm(target_delta_after[:2])),
            "vertical_offset_to_target": float(target_delta_after[2]),
            "grasp_attempts": grasp_attempts,
            "grasp_stage": {
                "selected_attempt": best_attempt,
                "contact_established": contact_established,
            },
            "transport_stage": transport_summary,
            "release_stage": release_summary,
            "final_object_hand_alignment": self.measure_object_hand_alignment(object_key, selected),
            "grasp_summary": grasp_summary,
            "place_summary": place_summary,
            "final_state": final_state,
            "task_completed": bool(final_state.get("task_completed", False)),
            "reward": float(final_state.get("reward", 0.0)),
        }

    def solve_pnp_pouring_cup_physical(
        self,
        cup_name: str = "obj_container",
        ball_name: str = "ball_obj",
        target_name: str = "container",
        arm: str = "right",
        preset: str = "cylindrical",
        approach_steps: int = 30,
        close_steps: int = 22,
        lift_steps: int = 18,
        move_steps: int = 95,
        pour_steps: int = 30,
        release_steps: int = 0,
        scale: float = 0.055,
        cup_grasp_z_offset: float = -0.045,
        target_z_offset: float = 0.47,
        target_xy_offset: tuple[float, float] | list[float] = (0.08, 0.0),
        pour_axis_angle_delta: tuple[float, float, float] | list[float] = (0.0, 0.0, 0.0),
        use_geometry_pour: bool = False,
        pour_down_bias: float = 0.35,
        pour_orientation_gain: float = 1.0,
        use_pivot_pour: bool = True,
        pivot_forward: float = 0.06,
        pivot_drop: float = 0.08,
        pivot_lift: float = 0.33,
        use_object_pose_place: bool = True,
        object_place_axis_angle_delta: tuple[float, float, float] | list[float] = (0.0, 0.40, 0.0),
        cup_grasp_profile: str = "cylindrical_lower",
        cup_grasp_side_offset: float = 0.035,
        cup_grasp_z_floor: float = -0.075,
        final_pour_axis_angle_delta: tuple[float, float, float] | list[float] = (0.0, 0.0, 0.0),
        final_pour_steps: int = 0,
        final_pour_joint_deltas: dict[str, float] | None = None,
        final_pour_joint_steps: int = 55,
    ) -> dict[str, Any]:
        """PnPPouring primitive that manipulates the source cup, not the ball.

        RoboCasa PnPPouring places the ball inside ``obj_container``. Grasping
        ``ball_obj`` directly is a useful contact diagnostic but not the real
        task strategy. This helper instead grasps the source cup near its lower
        side, carries it near/above the target container, and runs a small
        position-only pour/shake phase. It intentionally records cup, ball, and
        target distances separately so videos and summaries can diagnose cup
        collision versus failed ball transfer.
        """
        cup_key = self._canonical_object_name(cup_name)
        ball_key = self._canonical_object_name(ball_name)
        target_key = self._canonical_object_name(target_name)
        selected = "right" if str(arm).lower().strip() in {"auto", "nearest", "both"} else str(arm).lower().strip()
        if final_pour_joint_deltas is None:
            final_pour_joint_deltas = {
                "robot0_r_wrist_yaw": 1.57,
                "robot0_r_wrist_roll": 1.50,
                "robot0_r_shoulder_yaw": 0.25,
            }
        cup_before, _ = self.get_object_pose(cup_key)
        ball_before, _ = self.get_object_pose(ball_key)
        target_pos, _ = self.get_object_pose(target_key)
        target_pos = np.asarray(target_pos, dtype=np.float64).reshape(3)
        xy_offset = np.asarray(target_xy_offset, dtype=np.float64).reshape(-1)
        if xy_offset.size < 2:
            xy_offset = np.zeros(2, dtype=np.float64)
        target_offset = np.array([float(xy_offset[0]), float(xy_offset[1]), 0.0], dtype=np.float64)
        pour_delta = np.asarray(pour_axis_angle_delta, dtype=np.float64).reshape(-1)
        if pour_delta.size < 3:
            pour_delta = np.pad(pour_delta, (0, 3 - pour_delta.size), mode="constant")
        pour_delta = pour_delta[:3]
        object_place_delta = np.asarray(object_place_axis_angle_delta, dtype=np.float64).reshape(-1)
        if object_place_delta.size < 3:
            object_place_delta = np.pad(object_place_delta, (0, 3 - object_place_delta.size), mode="constant")
        object_place_delta = object_place_delta[:3]
        final_pour_delta = np.asarray(final_pour_axis_angle_delta, dtype=np.float64).reshape(-1)
        if final_pour_delta.size < 3:
            final_pour_delta = np.pad(final_pour_delta, (0, 3 - final_pour_delta.size), mode="constant")
        final_pour_delta = final_pour_delta[:3]
        requested_params = {
            "approach_steps": int(approach_steps),
            "close_steps": int(close_steps),
            "lift_steps": int(lift_steps),
            "move_steps": int(move_steps),
            "pour_steps": int(pour_steps),
            "release_steps": int(release_steps),
            "scale": float(scale),
            "cup_grasp_z_offset": float(cup_grasp_z_offset),
            "target_z_offset": float(target_z_offset),
            "target_xy_offset": tuple(float(x) for x in target_offset[:2]),
            "pour_axis_angle_delta": tuple(float(x) for x in pour_delta),
            "use_geometry_pour": bool(use_geometry_pour),
            "pour_down_bias": float(pour_down_bias),
            "pour_orientation_gain": float(pour_orientation_gain),
            "use_pivot_pour": bool(use_pivot_pour),
            "pivot_forward": float(pivot_forward),
            "pivot_drop": float(pivot_drop),
            "pivot_lift": float(pivot_lift),
            "use_object_pose_place": bool(use_object_pose_place),
            "object_place_axis_angle_delta": tuple(float(x) for x in object_place_delta),
            "cup_grasp_profile": str(cup_grasp_profile),
            "cup_grasp_side_offset": float(cup_grasp_side_offset),
            "cup_grasp_z_floor": float(cup_grasp_z_floor),
            "final_pour_axis_angle_delta": tuple(float(x) for x in final_pour_delta),
            "final_pour_steps": int(final_pour_steps),
            "final_pour_joint_deltas": {str(k): float(v) for k, v in (final_pour_joint_deltas or {}).items()},
            "final_pour_joint_steps": int(final_pour_joint_steps),
        }
        requested_joint_deltas = {
            str(k): float(v)
            for k, v in (final_pour_joint_deltas or {}).items()
            if str(k).strip()
        }
        profile_name = str(cup_grasp_profile).lower().strip()
        if profile_name in {"", "default"}:
            profile_name = "center"
        effective_params = {
            "approach_steps": max(8, int(approach_steps)),
            "close_steps": max(12, int(close_steps)),
            "lift_steps": max(0, int(lift_steps)),
            "move_steps": max(1, int(move_steps)),
            "pour_steps": max(0, int(pour_steps)),
            "release_steps": max(0, int(release_steps)),
            "scale": float(np.clip(float(scale), 0.005, 0.08)),
            "cup_grasp_z_offset": float(np.clip(float(cup_grasp_z_offset), -0.10, 0.02)),
            "target_z_offset": float(np.clip(float(target_z_offset), 0.08, 0.30)),
            "target_offset": tuple(float(x) for x in target_offset),
            "pour_axis_angle_delta": tuple(float(x) for x in np.clip(pour_delta, -0.8, 0.8)),
            "use_geometry_pour": bool(use_geometry_pour),
            "pour_down_bias": float(np.clip(float(pour_down_bias), 0.0, 1.0)),
            "pour_orientation_gain": float(np.clip(float(pour_orientation_gain), 0.0, 2.0)),
            "use_pivot_pour": bool(use_pivot_pour),
            "pivot_forward": float(np.clip(float(pivot_forward), -0.12, 0.16)),
            "pivot_drop": float(np.clip(float(pivot_drop), 0.0, 0.18)),
            "pivot_lift": float(np.clip(float(pivot_lift), 0.0, 0.14)),
            "use_object_pose_place": bool(use_object_pose_place),
            "object_place_axis_angle_delta": tuple(float(x) for x in np.clip(object_place_delta, -1.2, 1.2)),
            "cup_grasp_profile": profile_name,
            "cup_grasp_side_offset": float(np.clip(float(cup_grasp_side_offset), 0.0, 0.08)),
            "cup_grasp_z_floor": float(np.clip(float(cup_grasp_z_floor), -0.10, -0.02)),
            "final_pour_axis_angle_delta": tuple(float(x) for x in np.clip(final_pour_delta, -1.0, 1.0)),
            "final_pour_steps": max(0, int(final_pour_steps)),
            "final_pour_joint_deltas": {
                str(k): float(np.clip(float(v), -2.8, 2.8)) for k, v in requested_joint_deltas.items()
            },
            "final_pour_joint_steps": max(0, int(final_pour_joint_steps)),
            "cup_lift_threshold": 0.025,
            "cup_target_collision_distance": 0.11,
        }
        grasp_position_offset = np.zeros(3, dtype=np.float64)
        profile_grasp_z_offset = float(effective_params["cup_grasp_z_offset"])
        if profile_name in {"lower_body", "cylindrical_lower"}:
            # Approximate a human cylindrical cup grasp: approach the lower cup
            # wall from the hand-facing side instead of aiming at the cup center.
            profile_grasp_z_offset = min(profile_grasp_z_offset, float(effective_params["cup_grasp_z_floor"]))
            side_dir = np.asarray(self._eef_world_pos(selected), dtype=np.float64).reshape(3) - np.asarray(cup_before, dtype=np.float64).reshape(3)
            side_dir[2] = 0.0
            side_dir = self._normalized(side_dir)
            if float(np.linalg.norm(side_dir)) < 1e-9:
                side_dir = np.array([0.0, -1.0, 0.0], dtype=np.float64)
            grasp_position_offset[:2] = side_dir[:2] * float(effective_params["cup_grasp_side_offset"])
        effective_params["profile_grasp_z_offset"] = profile_grasp_z_offset
        effective_params["grasp_position_offset"] = tuple(float(x) for x in grasp_position_offset)

        grasp_summary = self.grasp_object(
            object_name=cup_key,
            arm=selected,
            preset=preset,
            approach_steps=int(effective_params["approach_steps"]),
            close_steps=int(effective_params["close_steps"]),
            approach_scale=float(effective_params["scale"]),
            approach_height=0.10,
            grasp_height_offset=profile_grasp_z_offset,
            grasp_position_offset=grasp_position_offset,
        )
        selected = str(grasp_summary.get("arm", selected))
        cup_after_grasp, cup_quat_after_grasp = self.get_object_pose(cup_key)
        ball_after_grasp, _ = self.get_object_pose(ball_key)
        cup_lifted = float(np.asarray(cup_after_grasp, dtype=np.float64)[2] - np.asarray(cup_before, dtype=np.float64)[2])
        cup_moved = float(np.linalg.norm(np.asarray(cup_after_grasp, dtype=np.float64) - np.asarray(cup_before, dtype=np.float64)))
        cup_alignment = self.measure_object_hand_alignment(cup_key, selected)
        cup_contact_established = (
            cup_lifted > float(effective_params["cup_lift_threshold"])
            or cup_moved > 0.05
            or bool(grasp_summary.get("task_completed", False))
        )
        phase = "cup_grasp_failed"
        transport_summary: dict[str, Any] | None = None
        pour_summary: dict[str, Any] | None = None
        final_pour_summary: dict[str, Any] | None = None
        final_joint_posture_summary: dict[str, Any] | None = None
        release_summary: dict[str, Any] | None = None
        if cup_contact_established and not bool(grasp_summary.get("task_completed", False)):
            gripper_object_tf: np.ndarray | None = None
            inv_gripper_object_tf: np.ndarray | None = None
            object_pose_reference: np.ndarray | None = None
            if bool(effective_params["use_object_pose_place"]):
                try:
                    gripper_pose_at_pick = self._pose_matrix_from_xyzw(
                        self._eef_world_pos(selected),
                        self._eef_world_quat_xyzw(selected),
                    )
                    object_pose_at_pick = self._pose_matrix_from_wxyz(
                        cup_after_grasp,
                        cup_quat_after_grasp,
                    )
                    gripper_object_tf = self._inverse_pose_matrix(gripper_pose_at_pick) @ object_pose_at_pick
                    inv_gripper_object_tf = self._inverse_pose_matrix(gripper_object_tf)
                    object_pose_reference = object_pose_at_pick.copy()
                except Exception:
                    gripper_object_tf = None
                    inv_gripper_object_tf = None
                    object_pose_reference = None
            transport_summary = self.transport_held_object_towards_target(
                object_name=cup_key,
                target_name=target_key,
                arm=selected,
                holding_preset=str(grasp_summary.get("preset", preset)),
                steps=int(effective_params["move_steps"]),
                scale=float(effective_params["scale"]),
                target_z_offset=float(effective_params["target_z_offset"]),
                target_offset=target_offset,
                patience=10,
                restore_best_state=False,
            )
            phase = "cup_transport"
            pour_orientation_deltas: list[dict[str, Any]] = []
            for step_idx in range(int(effective_params["pour_steps"])):
                progress = (
                    step_idx / max(1, int(effective_params["pour_steps"]) - 1)
                    if int(effective_params["pour_steps"]) > 1
                    else 1.0
                )
                if bool(effective_params["use_pivot_pour"]):
                    cup_pos_now, _ = self.get_object_pose(cup_key)
                    horizontal = target_pos - np.asarray(cup_pos_now, dtype=np.float64).reshape(3)
                    horizontal[2] = 0.0
                    pour_dir = self._normalized(horizontal)
                    if float(np.linalg.norm(pour_dir)) < 1e-9:
                        pour_dir = self._normalized(target_offset)
                    desired = target_pos + target_offset
                    desired = desired + pour_dir * float(effective_params["pivot_forward"]) * progress
                    desired[2] += (
                        float(effective_params["target_z_offset"])
                        + float(effective_params["pivot_lift"]) * progress
                        - float(effective_params["pivot_drop"]) * progress
                    )
                else:
                    # Without a validated wrist-orientation setter, use a small
                    # lateral/downward shake above the bowl to test whether cup
                    # placement alone can transfer the ball and to expose collision
                    # cases in videos.
                    sign = -1.0 if step_idx % 2 else 1.0
                    desired = target_pos + target_offset + np.array(
                        [0.025 * sign, 0.0, float(effective_params["target_z_offset"]) - 0.035],
                        dtype=np.float64,
                    )
                current = self._eef_world_pos(selected)
                action = self._hold_pose_action()
                target_gripper_pose: np.ndarray | None = None
                if bool(effective_params["use_object_pose_place"]) and inv_gripper_object_tf is not None:
                    reference_pose = object_pose_reference
                    if reference_pose is None:
                        _, current_cup_quat = self.get_object_pose(cup_key)
                        reference_pose = self._pose_matrix_from_wxyz(desired, current_cup_quat)
                    desired_object_pose = np.eye(4, dtype=np.float64)
                    object_delta_rot = self._axis_angle_to_matrix(
                        np.asarray(effective_params["object_place_axis_angle_delta"], dtype=np.float64) * progress
                    )
                    desired_object_pose[:3, :3] = object_delta_rot @ reference_pose[:3, :3]
                    desired_object_pose[:3, 3] = desired
                    target_gripper_pose = desired_object_pose @ inv_gripper_object_tf
                    desired = target_gripper_pose[:3, 3]
                self._set_arm_absolute_target(
                    action,
                    selected,
                    current + self._clipped_step_delta(desired - current, max_step=float(effective_params["scale"])),
                )
                if target_gripper_pose is not None:
                    self._set_arm_absolute_orientation(
                        action,
                        selected,
                        self._matrix_to_quat_xyzw(target_gripper_pose[:3, :3]),
                    )
                if bool(effective_params["use_geometry_pour"]):
                    cup_pos, cup_quat = self.get_object_pose(cup_key)
                    current_mouth_dir = self._quat_wxyz_rotate_vector(cup_quat, np.array([0.0, 0.0, 1.0], dtype=np.float64))
                    target_dir = target_pos - np.asarray(cup_pos, dtype=np.float64).reshape(3)
                    target_dir[2] -= float(effective_params["pour_down_bias"])
                    geometric_delta = self._axis_angle_from_vectors(current_mouth_dir, target_dir)
                    geometric_delta = self._limit_axis_angle(
                        geometric_delta * float(effective_params["pour_orientation_gain"]),
                        max_norm=0.9,
                    )
                    self._add_arm_axis_angle_delta(action, selected, geometric_delta)
                    pour_orientation_deltas.append(
                        {
                            "step": step_idx,
                            "current_mouth_dir": current_mouth_dir,
                            "target_mouth_dir": self._normalized(target_dir),
                            "geometric_delta": geometric_delta,
                        }
                    )
                self._add_arm_axis_angle_delta(
                    action,
                    selected,
                    np.asarray(effective_params["pour_axis_angle_delta"], dtype=np.float64),
                )
                self._set_hand_action(action, selected, str(grasp_summary.get("preset", preset)), 1.0)
                pour_summary = self._repeat_action(action, steps=1)
                if pour_summary["terminated"] or pour_summary["truncated"] or pour_summary["task_completed"]:
                    break
            if int(effective_params["release_steps"]) > 0 and not self._env.task_completed():
                release_summary = self.open_hand(arm=selected, steps=int(effective_params["release_steps"]))
                phase = "cup_released"
            elif pour_summary is not None:
                phase = "cup_pour_attempted"
            if pour_summary is not None:
                pour_summary["geometry_pour"] = bool(effective_params["use_geometry_pour"])
                pour_summary["pivot_pour"] = bool(effective_params["use_pivot_pour"])
                pour_summary["object_pose_place"] = bool(effective_params["use_object_pose_place"])
                pour_summary["gripper_object_tf"] = gripper_object_tf if gripper_object_tf is not None else None
                pour_summary["pour_orientation_deltas"] = pour_orientation_deltas[-5:]
            final_delta = np.asarray(effective_params["final_pour_axis_angle_delta"], dtype=np.float64)
            if int(effective_params["final_pour_steps"]) > 0 and float(np.linalg.norm(final_delta)) > 1e-9:
                for _ in range(int(effective_params["final_pour_steps"])):
                    action = self._hold_pose_action()
                    self._add_arm_axis_angle_delta(action, selected, final_delta)
                    self._set_hand_action(action, selected, str(grasp_summary.get("preset", preset)), 1.0)
                    final_pour_summary = self._repeat_action(action, steps=1)
                    if (
                        final_pour_summary["terminated"]
                        or final_pour_summary["truncated"]
                        or final_pour_summary["task_completed"]
                    ):
                        break
                if final_pour_summary is not None:
                    final_pour_summary["final_pour_axis_angle_delta"] = tuple(float(x) for x in final_delta)
                    final_pour_summary["final_pour_steps"] = int(effective_params["final_pour_steps"])
                    phase = "cup_final_rotation_attempted"
            joint_deltas = dict(effective_params["final_pour_joint_deltas"])
            if int(effective_params["final_pour_joint_steps"]) > 0 and joint_deltas:
                final_joint_posture_summary = {
                    "requested_joint_deltas": dict(requested_joint_deltas),
                    "effective_joint_deltas": dict(joint_deltas),
                    "final_pour_joint_steps": int(effective_params["final_pour_joint_steps"]),
                    "ok": False,
                }
                try:
                    robosuite_env = getattr(self._env, "robosuite_env", None)
                    robots = getattr(robosuite_env, "robots", []) if robosuite_env is not None else []
                    controller = getattr(robots[0], "composite_controller", None) if robots else None
                    joint_policy = getattr(controller, "joint_action_policy", None)
                    if joint_policy is None:
                        raise RuntimeError("active controller does not expose joint_action_policy")
                    joint_policy.update_robot_states()
                    posture_start = np.asarray(joint_policy.configuration.data.qpos, dtype=np.float64).copy()
                    posture_target = posture_start.copy()
                    before_eef = self._eef_world_pos(selected)
                    before_qpos: dict[str, float] = {}
                    target_indexes: dict[str, int] = {}
                    sim = getattr(robosuite_env, "sim", None)
                    data = getattr(sim, "data", None)
                    for joint_name, delta in joint_deltas.items():
                        target_idx = int(joint_policy.robot_model.joint(joint_name).qposadr[0])
                        target_indexes[joint_name] = target_idx
                        posture_target[target_idx] += float(delta)
                        if data is not None:
                            before_qpos[joint_name] = float(np.asarray(data.get_joint_qpos(joint_name)).reshape(-1)[0])
                    summary: dict[str, Any] | None = None
                    posture_steps = int(effective_params["final_pour_joint_steps"])
                    for posture_step in range(posture_steps):
                        progress = (posture_step + 1) / max(1, posture_steps)
                        joint_policy.set_posture_target(posture_start + (posture_target - posture_start) * progress)
                        action = self._hold_pose_action()
                        self._set_hand_action(action, selected, str(grasp_summary.get("preset", preset)), 1.0)
                        summary = self._repeat_action(action, steps=1)
                        if summary["terminated"] or summary["truncated"] or summary["task_completed"]:
                            break
                    after_qpos: dict[str, float] = {}
                    if data is not None:
                        for joint_name in joint_deltas:
                            after_qpos[joint_name] = float(np.asarray(data.get_joint_qpos(joint_name)).reshape(-1)[0])
                    after_eef = self._eef_world_pos(selected)
                    final_joint_posture_summary.update(
                        {
                            "ok": True,
                            "posture_target_interpolation": "linear",
                            "target_indexes": target_indexes,
                            "before_qpos": before_qpos,
                            "after_qpos": after_qpos,
                            "actual_joint_deltas": {
                                name: float(after_qpos[name] - before_qpos[name])
                                for name in before_qpos
                                if name in after_qpos
                            },
                            "eef_delta_norm": float(np.linalg.norm(after_eef - before_eef)),
                            "summary": summary,
                        }
                    )
                    phase = "cup_final_joint_posture_attempted"
                except Exception as exc:
                    final_joint_posture_summary["error"] = repr(exc)
        cup_after, _ = self.get_object_pose(cup_key)
        ball_after, _ = self.get_object_pose(ball_key)
        final_state = self.get_task_state()

        def dist_to_target(pos: np.ndarray) -> float:
            return float(np.linalg.norm(np.asarray(pos, dtype=np.float64).reshape(3) - target_pos))

        def xy_dist_to_target(pos: np.ndarray) -> float:
            delta = np.asarray(pos, dtype=np.float64).reshape(3) - target_pos
            return float(np.linalg.norm(delta[:2]))

        def vertical_offset_to_target(pos: np.ndarray) -> float:
            delta = np.asarray(pos, dtype=np.float64).reshape(3) - target_pos
            return float(delta[2])

        cup_target_distance = dist_to_target(cup_after)
        ball_target_distance = dist_to_target(ball_after)
        cup_target_xy_distance = xy_dist_to_target(cup_after)
        ball_target_xy_distance = xy_dist_to_target(ball_after)
        cup_target_vertical_offset = vertical_offset_to_target(cup_after)
        ball_target_vertical_offset = vertical_offset_to_target(ball_after)
        return {
            "phase": phase,
            "cup": cup_key,
            "ball": ball_key,
            "target": target_key,
            "arm": selected,
            "preset": str(grasp_summary.get("preset", preset)),
            "requested_params": requested_params,
            "effective_params": effective_params,
            "cup_before": cup_before,
            "cup_after": cup_after,
            "cup_delta": np.asarray(cup_after, dtype=np.float64) - np.asarray(cup_before, dtype=np.float64),
            "ball_before": ball_before,
            "ball_after": ball_after,
            "ball_delta": np.asarray(ball_after, dtype=np.float64) - np.asarray(ball_before, dtype=np.float64),
            "distance_to_target": ball_target_distance,
            "cup_distance_to_target": cup_target_distance,
            "ball_distance_to_target": ball_target_distance,
            "cup_xy_distance_to_target": cup_target_xy_distance,
            "ball_xy_distance_to_target": ball_target_xy_distance,
            "cup_vertical_offset_to_target": cup_target_vertical_offset,
            "ball_vertical_offset_to_target": ball_target_vertical_offset,
            "cup_ball_distance": float(np.linalg.norm(np.asarray(cup_after, dtype=np.float64) - np.asarray(ball_after, dtype=np.float64))),
            "cup_target_collision_risk": cup_target_distance < float(effective_params["cup_target_collision_distance"]),
            "cup_grasp_stage": {
                "contact_established": cup_contact_established,
                "cup_lifted_z": cup_lifted,
                "cup_moved_norm": cup_moved,
                "cup_hand_alignment": cup_alignment,
                "summary": grasp_summary,
            },
            "transport_stage": transport_summary,
            "pour_stage": pour_summary,
            "final_pour_stage": final_pour_summary,
            "final_joint_posture_stage": final_joint_posture_summary,
            "release_stage": release_summary,
            "final_state": final_state,
            "task_completed": bool(final_state.get("task_completed", False)),
            "reward": float(final_state.get("reward", 0.0)),
        }

    def teleport_object_to_target(
        self,
        object_name: str,
        target_name: str,
        z_offset: float = 0.08,
        settle_steps: int = 10,
    ) -> dict[str, Any]:
        """Privilegedly place an object into a target bin and settle physics.

        This directly edits the object's free joint in MuJoCo. It is meant as a
        deterministic data-plumbing and task-success baseline for RoboCasa/GR1
        migration, not as a physically realistic robot policy primitive.

        Args:
            object_name: ``"payload"`` or ``"trash"``.
            target_name: ``"target_bin"`` or ``"trash_bin"``.
            z_offset: Height above the target-bin base before settling.
            settle_steps: Number of zero-action steps after state injection.

        Returns:
            Final step summary after physics settling.
        """
        object_key = self._canonical_object_name(object_name)
        target_key = self._canonical_object_name(target_name)
        self._set_object_pose_to_target(object_key, target_key, z_offset=z_offset)
        return self._repeat_action(np.zeros(self._action_dim(), dtype=np.float64), steps=settle_steps)

    def solve_transport_privileged(
        self,
        settle_steps: int = 12,
        z_offset: float = 0.08,
    ) -> dict[str, Any]:
        """Privileged oracle for the default TwoArmTransport smoke task.

        The RoboSuite success check requires both ``payload_in_target_bin`` and
        ``trash_in_trash_bin``. This method moves those two free-joint objects
        into their bins, then steps the simulator so contacts, reward and the
        transition dataset are updated.
        """
        if not hasattr(getattr(self._env, "robosuite_env", None), "transport"):
            raise RuntimeError("solve_transport_privileged is only available for TwoArmTransport-like tasks")
        self._set_object_pose_to_target("payload", "target_bin", z_offset=z_offset)
        self._set_object_pose_to_target("trash", "trash_bin", z_offset=max(0.03, min(z_offset, 0.08)))
        return self._repeat_action(np.zeros(self._action_dim(), dtype=np.float64), steps=settle_steps)

    def sample_random_action(self, scale: float = 0.05) -> np.ndarray:
        """Return a small random action vector for smoke testing.

        Args:
            scale: Uniform bound for each action dimension.

        Returns:
            Full action vector matching the current environment action dimension.
        """
        if hasattr(self._env, "sample_action"):
            return np.asarray(self._env.sample_action(scale=scale), dtype=np.float64)
        return np.random.uniform(-scale, scale, size=(self._action_dim(),)).astype(np.float64)

    def write_video(self, path: str) -> None:
        """Write currently buffered main-camera frames to an MP4 file."""
        import imageio.v2 as imageio

        if not hasattr(self._env, "get_video_frames"):
            raise RuntimeError("The current environment does not expose video frames")
        frames = self._env.get_video_frames(clear=False)
        if not frames:
            raise RuntimeError("No video frames are buffered")
        writer = imageio.get_writer(path, fps=20)
        try:
            for frame in frames:
                writer.append_data(frame)
        finally:
            writer.close()

    def _repeat_action(self, action: np.ndarray, *, steps: int) -> dict[str, Any]:
        summary: dict[str, Any] | None = None
        for _ in range(max(1, int(steps))):
            obs, reward, terminated, truncated, info = self._env.step(action)
            summary = self._step_summary(obs, reward, terminated, truncated, info)
            if terminated or truncated or summary["task_completed"]:
                break
        assert summary is not None
        return summary

    def _step_summary(
        self,
        obs: dict[str, Any],
        reward: float,
        terminated: bool,
        truncated: bool,
        info: dict[str, Any],
    ) -> dict[str, Any]:
        return {
            "reward": float(reward),
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "task_completed": bool(self._env.task_completed()),
            "success_flags": self._success_flags(obs),
            "objects": self.get_named_positions(),
            "info": info,
        }

    def _safe_full_action(self, action: np.ndarray | list[float]) -> np.ndarray:
        arr = np.asarray(action, dtype=np.float64).reshape(-1)
        dim = self._action_dim()
        if arr.size < dim:
            arr = np.pad(arr, (0, dim - arr.size), mode="constant")
        elif arr.size > dim:
            arr = arr[:dim]
        return np.clip(arr, -1.0, 1.0)

    def _action_dim(self) -> int:
        if hasattr(self._env, "sample_action"):
            try:
                return int(np.asarray(self._env.sample_action(scale=0.0)).size)
            except Exception:
                pass
        obs = self._env.get_observation()
        joint_pos = np.asarray(obs.get("robot_joint_pos", []), dtype=np.float32).reshape(-1)
        if joint_pos.size:
            return int(max(24, joint_pos.size))
        return 24

    def _selected_arms(self, arm: str) -> tuple[str, ...]:
        arm_name = str(arm).lower().strip()
        if arm_name in {"left", "arm0", "0"}:
            return ("left",)
        if arm_name in {"right", "arm1", "1"}:
            return ("right",)
        return ("right", "left")

    def _action_slice(self, part_name: str) -> tuple[int, int]:
        layout = self.get_action_layout()
        if part_name not in layout:
            return (0, 0)
        start, end = layout[part_name]
        dim = self._action_dim()
        return max(0, min(start, dim)), max(0, min(end, dim))

    def _set_hand_action(
        self,
        action: np.ndarray,
        arm_name: str,
        preset: str,
        strength: float = 1.0,
    ) -> None:
        start, end = self._action_slice(f"{arm_name}_gripper")
        if end <= start:
            return
        command = self._hand_preset(preset, strength=strength)
        action[start:end] = command[: end - start]

    def _set_object_pose_to_target(self, object_name: str, target_name: str, *, z_offset: float) -> None:
        """Set a RoboSuite TransportGroup object's free-joint pose."""
        robosuite_env = getattr(self._env, "robosuite_env", None)
        transport = getattr(robosuite_env, "transport", None)
        if transport is None:
            raise RuntimeError("Underlying RoboCasa environment does not expose a transport object group")
        obj = getattr(transport, object_name, None)
        if obj is None or not getattr(obj, "joints", None):
            raise ValueError(f"Object {object_name!r} does not expose a free joint")
        if target_name == "target_bin":
            target_pos = np.asarray(transport.target_bin_pos, dtype=np.float64)
        elif target_name == "trash_bin":
            target_pos = np.asarray(transport.trash_bin_pos, dtype=np.float64)
        else:
            raise ValueError("target_name must be 'target_bin' or 'trash_bin' for the transport oracle")
        current_pose = self.get_object_pose(object_name)
        quat = np.asarray(current_pose[1], dtype=np.float64).reshape(-1)
        if quat.size < 4 or not np.all(np.isfinite(quat[:4])):
            quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        pos = target_pos.copy()
        pos[2] += float(z_offset)
        robosuite_env.sim.data.set_joint_qpos(obj.joints[0], np.concatenate([pos[:3], quat[:4]]))
        robosuite_env.sim.forward()
        if hasattr(robosuite_env, "_get_observations"):
            self._env._current_obs = self._env._normalize_raw_obs(robosuite_env._get_observations())

    def _move_bimanual_towards(
        self,
        left_target_world_pos: np.ndarray,
        right_target_world_pos: np.ndarray,
        *,
        steps: int,
        scale: float,
    ) -> dict[str, Any]:
        summary: dict[str, Any] | None = None
        targets = {
            "left": np.asarray(left_target_world_pos, dtype=np.float64).reshape(3),
            "right": np.asarray(right_target_world_pos, dtype=np.float64).reshape(3),
        }
        for _ in range(max(1, int(steps))):
            action = self._hold_pose_action()
            for arm_name, target in targets.items():
                current = self._eef_world_pos(arm_name)
                step_target = current + self._clipped_step_delta(target - current, max_step=float(scale))
                self._set_arm_absolute_target(action, arm_name, step_target)
            summary = self._repeat_action(action, steps=1)
            if summary["terminated"] or summary["truncated"] or summary["task_completed"]:
                break
        assert summary is not None
        return summary

    def _geom_world_pos(self, geom_name: str) -> np.ndarray:
        robosuite_env = getattr(self._env, "robosuite_env", None)
        sim = getattr(robosuite_env, "sim", None)
        if sim is None:
            raise RuntimeError("Underlying RoboCasa environment does not expose a MuJoCo sim")
        try:
            geom_id = sim.model.geom_name2id(str(geom_name))
            return np.asarray(sim.data.geom_xpos[geom_id], dtype=np.float64).reshape(3).copy()
        except Exception as exc:
            raise ValueError(f"Geom {geom_name!r} is not available in this task") from exc

    def _body_world_pos(self, body_name: str) -> np.ndarray:
        robosuite_env = getattr(self._env, "robosuite_env", None)
        sim = getattr(robosuite_env, "sim", None)
        if sim is None:
            raise RuntimeError("Underlying RoboCasa environment does not expose a MuJoCo sim")
        try:
            body_id = sim.model.body_name2id(str(body_name))
            return np.asarray(sim.data.body_xpos[body_id], dtype=np.float64).reshape(3).copy()
        except Exception as exc:
            raise ValueError(f"Body {body_name!r} is not available in this task") from exc

    def _nearest_arm(self, object_name: str) -> str:
        target_pos, _ = self.get_object_pose(object_name)
        robot = self.get_robot_state()
        left = robot.get("robot0_left_eef_pos")
        right = robot.get("robot0_right_eef_pos")
        if left is None and right is None:
            return "right"
        if left is None:
            return "right"
        if right is None:
            return "left"
        left_dist = float(np.linalg.norm(target_pos - np.asarray(left[:3], dtype=np.float32)))
        right_dist = float(np.linalg.norm(target_pos - np.asarray(right[:3], dtype=np.float32)))
        return "left" if left_dist <= right_dist else "right"

    def _eef_world_pos(self, arm_name: str) -> np.ndarray:
        robot = self.get_robot_state()
        key = f"robot0_{arm_name}_eef_pos"
        pos = np.asarray(robot.get(key, []), dtype=np.float64).reshape(-1)
        if pos.size >= 3:
            return pos[:3].copy()
        return np.zeros(3, dtype=np.float64)

    def _eef_world_quat_xyzw(self, arm_name: str) -> np.ndarray:
        robot = self.get_robot_state()
        key = f"robot0_{arm_name}_eef_quat"
        quat = np.asarray(robot.get(key, []), dtype=np.float64).reshape(-1)
        if quat.size >= 4 and np.all(np.isfinite(quat[:4])):
            return quat[:4].copy()
        base_key = f"robot0_base_to_{arm_name}_eef_quat"
        quat = np.asarray(robot.get(base_key, []), dtype=np.float64).reshape(-1)
        if quat.size >= 4 and np.all(np.isfinite(quat[:4])):
            return self._base_quat_xyzw_to_world_quat_xyzw(quat[:4])
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)

    def _hand_joint_snapshot(self, arm_name: str) -> dict[str, Any]:
        """Best-effort hand joint snapshot for preset validation."""
        arm = str(arm_name).lower().strip()
        snapshot: dict[str, Any] = {
            "action_slice": self._action_slice(f"{arm}_gripper"),
            "joint_qpos": {},
        }
        robosuite_env = getattr(self._env, "robosuite_env", None)
        sim = getattr(robosuite_env, "sim", None)
        model = getattr(sim, "model", None)
        data = getattr(sim, "data", None)
        joint_names = tuple(getattr(model, "joint_names", ()) or ())
        qpos: dict[str, float] = {}
        for joint_name in joint_names:
            lower = str(joint_name).lower()
            if arm not in lower:
                continue
            if not any(token in lower for token in ("hand", "finger", "thumb", "index", "middle", "ring", "pinky")):
                continue
            try:
                value = np.asarray(data.get_joint_qpos(joint_name), dtype=np.float64).reshape(-1)
            except Exception:
                continue
            if value.size:
                qpos[str(joint_name)] = float(value[0])
        snapshot["joint_qpos"] = qpos
        robot_joint_pos = np.asarray(self.get_robot_state().get("joint_pos", []), dtype=np.float64).reshape(-1)
        snapshot["robot_joint_pos_shape"] = tuple(int(v) for v in robot_joint_pos.shape)
        if qpos:
            snapshot["joint_count"] = len(qpos)
        return snapshot

    def _hold_pose_action(self) -> np.ndarray:
        """Build an action that keeps both GR1 EEF absolute IK targets steady."""
        action = np.zeros(self._action_dim(), dtype=np.float64)
        robot = self.get_robot_state()
        ref_frame = self._ik_input_ref_frame()
        for arm_name in ("right", "left"):
            start, end = self._action_slice(arm_name)
            width = end - start
            if width <= 0:
                continue
            pos_key = (
                f"robot0_{arm_name}_eef_pos"
                if ref_frame == "world"
                else f"robot0_base_to_{arm_name}_eef_pos"
            )
            quat_key = (
                f"robot0_{arm_name}_eef_quat"
                if ref_frame == "world"
                else f"robot0_base_to_{arm_name}_eef_quat"
            )
            pos = np.asarray(
                robot.get(pos_key, np.zeros(3, dtype=np.float32)),
                dtype=np.float64,
            ).reshape(-1)
            if pos.size >= 3:
                action[start : min(start + 3, end)] = pos[: min(3, width)]
            if width >= 6:
                quat = np.asarray(robot.get(quat_key, []), dtype=np.float64).reshape(-1)
                action[start + 3 : start + 6] = self._quat_to_axis_angle(quat)
        return action

    def _set_arm_absolute_target(self, action: np.ndarray, arm_name: str, target_world_pos: np.ndarray) -> None:
        """Write one arm's absolute target position into an action."""
        start, end = self._action_slice(arm_name)
        if end - start < 3:
            return
        if self._ik_input_ref_frame() == "world":
            action[start : start + 3] = np.asarray(target_world_pos, dtype=np.float64).reshape(3)
        else:
            action[start : start + 3] = self._world_to_base_pos(target_world_pos)

    def _set_arm_absolute_orientation(self, action: np.ndarray, arm_name: str, target_world_quat_xyzw: np.ndarray) -> None:
        """Write one arm's absolute IK target orientation into an action."""
        start, end = self._action_slice(arm_name)
        if end - start < 6:
            return
        quat = self._world_quat_xyzw_to_ik_quat_xyzw(target_world_quat_xyzw)
        action[start + 3 : start + 6] = np.clip(self._quat_to_axis_angle(quat), -1.0, 1.0)

    def _add_arm_axis_angle_delta(self, action: np.ndarray, arm_name: str, delta_axis_angle: np.ndarray) -> None:
        """Bias one arm's current IK orientation target by a small axis-angle delta."""
        start, end = self._action_slice(arm_name)
        if end - start < 6:
            return
        delta = np.asarray(delta_axis_angle, dtype=np.float64).reshape(3)
        action[start + 3 : start + 6] = np.clip(action[start + 3 : start + 6] + delta, -1.0, 1.0)

    @classmethod
    def _offset_world_quat_xyzw(
        cls,
        quat_xyzw: np.ndarray,
        axis_angle_delta: np.ndarray,
        *,
        frame: str = "world",
    ) -> np.ndarray:
        """Return a target world quaternion after applying a candidate rotation.

        ``frame="world"`` rotates the current EEF frame by a world-frame delta
        (``R_target = R_delta @ R_current``). ``frame="local"`` rotates around
        the current EEF local axes (``R_target = R_current @ R_delta``).
        """
        current = cls._quat_xyzw_to_matrix(quat_xyzw)
        delta = cls._axis_angle_to_matrix(axis_angle_delta)
        if str(frame).lower().strip() == "local":
            target = current @ delta
        else:
            target = delta @ current
        return cls._matrix_to_quat_xyzw(target)

    @staticmethod
    def _normalized(vector: np.ndarray) -> np.ndarray:
        arr = np.asarray(vector, dtype=np.float64).reshape(3)
        norm = float(np.linalg.norm(arr))
        if norm < 1e-9:
            return np.zeros(3, dtype=np.float64)
        return arr / norm

    @classmethod
    def _axis_angle_from_vectors(cls, source: np.ndarray, target: np.ndarray) -> np.ndarray:
        """Return the shortest axis-angle rotation from source vector to target."""
        src = cls._normalized(source)
        dst = cls._normalized(target)
        if float(np.linalg.norm(src)) < 1e-9 or float(np.linalg.norm(dst)) < 1e-9:
            return np.zeros(3, dtype=np.float64)
        dot = float(np.clip(np.dot(src, dst), -1.0, 1.0))
        if dot > 1.0 - 1e-8:
            return np.zeros(3, dtype=np.float64)
        axis = np.cross(src, dst)
        axis_norm = float(np.linalg.norm(axis))
        if axis_norm < 1e-8:
            # 180-degree case: pick any stable perpendicular axis.
            basis = np.array([1.0, 0.0, 0.0], dtype=np.float64)
            if abs(float(np.dot(src, basis))) > 0.9:
                basis = np.array([0.0, 1.0, 0.0], dtype=np.float64)
            axis = np.cross(src, basis)
            axis_norm = float(np.linalg.norm(axis))
        axis = axis / max(axis_norm, 1e-9)
        angle = float(np.arccos(dot))
        return axis * angle

    @classmethod
    def _limit_axis_angle(cls, axis_angle: np.ndarray, *, max_norm: float) -> np.ndarray:
        vec = np.asarray(axis_angle, dtype=np.float64).reshape(3)
        norm = float(np.linalg.norm(vec))
        limit = max(0.0, float(max_norm))
        if norm <= limit or norm < 1e-9:
            return vec
        return vec / norm * limit

    @staticmethod
    def _quat_wxyz_rotate_vector(quat: np.ndarray, vector: np.ndarray) -> np.ndarray:
        """Rotate a vector by a MuJoCo body quaternion in wxyz order."""
        q = np.asarray(quat, dtype=np.float64).reshape(-1)
        if q.size < 4 or not np.all(np.isfinite(q[:4])):
            return np.asarray(vector, dtype=np.float64).reshape(3)
        q = q[:4]
        norm = float(np.linalg.norm(q))
        if norm < 1e-9:
            return np.asarray(vector, dtype=np.float64).reshape(3)
        w, x, y, z = q / norm
        rot = np.array(
            [
                [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
                [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
                [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
            ],
            dtype=np.float64,
        )
        return rot @ np.asarray(vector, dtype=np.float64).reshape(3)

    @classmethod
    def _pose_matrix_from_wxyz(cls, pos: np.ndarray, quat_wxyz: np.ndarray) -> np.ndarray:
        mat = np.eye(4, dtype=np.float64)
        mat[:3, :3] = cls._quat_wxyz_to_matrix(quat_wxyz)
        mat[:3, 3] = np.asarray(pos, dtype=np.float64).reshape(3)
        return mat

    @classmethod
    def _pose_matrix_from_xyzw(cls, pos: np.ndarray, quat_xyzw: np.ndarray) -> np.ndarray:
        mat = np.eye(4, dtype=np.float64)
        mat[:3, :3] = cls._quat_xyzw_to_matrix(quat_xyzw)
        mat[:3, 3] = np.asarray(pos, dtype=np.float64).reshape(3)
        return mat

    @staticmethod
    def _inverse_pose_matrix(pose: np.ndarray) -> np.ndarray:
        mat = np.asarray(pose, dtype=np.float64).reshape(4, 4)
        inv = np.eye(4, dtype=np.float64)
        rot = mat[:3, :3]
        inv[:3, :3] = rot.T
        inv[:3, 3] = -(rot.T @ mat[:3, 3])
        return inv

    @classmethod
    def _quat_wxyz_to_matrix(cls, quat_wxyz: np.ndarray) -> np.ndarray:
        q = np.asarray(quat_wxyz, dtype=np.float64).reshape(-1)
        if q.size < 4 or not np.all(np.isfinite(q[:4])):
            return np.eye(3, dtype=np.float64)
        w, x, y, z = cls._normalized_quat_wxyz(q[:4])
        return cls._quat_components_to_matrix(w, x, y, z)

    @classmethod
    def _quat_xyzw_to_matrix(cls, quat_xyzw: np.ndarray) -> np.ndarray:
        q = np.asarray(quat_xyzw, dtype=np.float64).reshape(-1)
        if q.size < 4 or not np.all(np.isfinite(q[:4])):
            return np.eye(3, dtype=np.float64)
        x, y, z, w = q[:4]
        w, x, y, z = cls._normalized_quat_wxyz(np.array([w, x, y, z], dtype=np.float64))
        return cls._quat_components_to_matrix(w, x, y, z)

    @staticmethod
    def _quat_components_to_matrix(w: float, x: float, y: float, z: float) -> np.ndarray:
        return np.array(
            [
                [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
                [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
                [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
            ],
            dtype=np.float64,
        )

    @staticmethod
    def _normalized_quat_wxyz(quat_wxyz: np.ndarray) -> tuple[float, float, float, float]:
        q = np.asarray(quat_wxyz, dtype=np.float64).reshape(4)
        norm = float(np.linalg.norm(q))
        if norm < 1e-9:
            return (1.0, 0.0, 0.0, 0.0)
        q = q / norm
        return (float(q[0]), float(q[1]), float(q[2]), float(q[3]))

    @staticmethod
    def _matrix_to_quat_xyzw(matrix: np.ndarray) -> np.ndarray:
        m = np.asarray(matrix, dtype=np.float64).reshape(3, 3)
        trace = float(np.trace(m))
        if trace > 0.0:
            s = np.sqrt(trace + 1.0) * 2.0
            w = 0.25 * s
            x = (m[2, 1] - m[1, 2]) / s
            y = (m[0, 2] - m[2, 0]) / s
            z = (m[1, 0] - m[0, 1]) / s
        elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
            s = np.sqrt(max(1.0 + m[0, 0] - m[1, 1] - m[2, 2], 0.0)) * 2.0
            w = (m[2, 1] - m[1, 2]) / max(s, 1e-9)
            x = 0.25 * s
            y = (m[0, 1] + m[1, 0]) / max(s, 1e-9)
            z = (m[0, 2] + m[2, 0]) / max(s, 1e-9)
        elif m[1, 1] > m[2, 2]:
            s = np.sqrt(max(1.0 + m[1, 1] - m[0, 0] - m[2, 2], 0.0)) * 2.0
            w = (m[0, 2] - m[2, 0]) / max(s, 1e-9)
            x = (m[0, 1] + m[1, 0]) / max(s, 1e-9)
            y = 0.25 * s
            z = (m[1, 2] + m[2, 1]) / max(s, 1e-9)
        else:
            s = np.sqrt(max(1.0 + m[2, 2] - m[0, 0] - m[1, 1], 0.0)) * 2.0
            w = (m[1, 0] - m[0, 1]) / max(s, 1e-9)
            x = (m[0, 2] + m[2, 0]) / max(s, 1e-9)
            y = (m[1, 2] + m[2, 1]) / max(s, 1e-9)
            z = 0.25 * s
        q = np.array([x, y, z, w], dtype=np.float64)
        norm = float(np.linalg.norm(q))
        if norm < 1e-9:
            return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
        return q / norm

    @classmethod
    def _axis_angle_to_matrix(cls, axis_angle: np.ndarray) -> np.ndarray:
        vec = np.asarray(axis_angle, dtype=np.float64).reshape(3)
        angle = float(np.linalg.norm(vec))
        if angle < 1e-9:
            return np.eye(3, dtype=np.float64)
        axis = vec / angle
        x, y, z = axis
        c = float(np.cos(angle))
        s = float(np.sin(angle))
        t = 1.0 - c
        return np.array(
            [
                [t * x * x + c, t * x * y - s * z, t * x * z + s * y],
                [t * x * y + s * z, t * y * y + c, t * y * z - s * x],
                [t * x * z - s * y, t * y * z + s * x, t * z * z + c],
            ],
            dtype=np.float64,
        )

    def _ik_input_ref_frame(self) -> str:
        """Return the active IK target reference frame, defaulting to base."""
        robosuite_env = getattr(self._env, "robosuite_env", None)
        robots = getattr(robosuite_env, "robots", []) if robosuite_env is not None else []
        if robots:
            controller = getattr(robots[0], "composite_controller", None)
            config = getattr(controller, "composite_controller_specific_config", {}) or {}
            ref_frame = str(config.get("ik_input_ref_frame", "")).lower().strip()
            if ref_frame in {"world", "base"}:
                return ref_frame
        return "base"

    def _world_to_base_pos(self, world_pos: np.ndarray) -> np.ndarray:
        """Convert a world-frame XYZ target to the robot base frame."""
        pos = np.asarray(world_pos, dtype=np.float64).reshape(3)
        robosuite_env = getattr(self._env, "robosuite_env", None)
        sim = getattr(robosuite_env, "sim", None)
        data = getattr(sim, "data", None)
        try:
            base_pos = np.asarray(data.get_body_xpos("robot0_base"), dtype=np.float64).reshape(3)
            base_xmat = np.asarray(data.get_body_xmat("robot0_base"), dtype=np.float64).reshape(3, 3)
            return base_xmat.T @ (pos - base_pos)
        except Exception:
            robot = self.get_robot_state()
            for arm_name in ("right", "left"):
                world = np.asarray(robot.get(f"robot0_{arm_name}_eef_pos", []), dtype=np.float64).reshape(-1)
                base = np.asarray(robot.get(f"robot0_base_to_{arm_name}_eef_pos", []), dtype=np.float64).reshape(-1)
                if world.size >= 3 and base.size >= 3:
                    return pos - (world[:3] - base[:3])
            return pos

    def _world_quat_xyzw_to_ik_quat_xyzw(self, quat_xyzw: np.ndarray) -> np.ndarray:
        if self._ik_input_ref_frame() == "world":
            q = np.asarray(quat_xyzw, dtype=np.float64).reshape(-1)
            return q[:4] if q.size >= 4 else np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
        robosuite_env = getattr(self._env, "robosuite_env", None)
        sim = getattr(robosuite_env, "sim", None)
        data = getattr(sim, "data", None)
        try:
            base_xmat = np.asarray(data.get_body_xmat("robot0_base"), dtype=np.float64).reshape(3, 3)
            world_rot = self._quat_xyzw_to_matrix(quat_xyzw)
            return self._matrix_to_quat_xyzw(base_xmat.T @ world_rot)
        except Exception:
            q = np.asarray(quat_xyzw, dtype=np.float64).reshape(-1)
            return q[:4] if q.size >= 4 else np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)

    def _base_quat_xyzw_to_world_quat_xyzw(self, quat_xyzw: np.ndarray) -> np.ndarray:
        robosuite_env = getattr(self._env, "robosuite_env", None)
        sim = getattr(robosuite_env, "sim", None)
        data = getattr(sim, "data", None)
        try:
            base_xmat = np.asarray(data.get_body_xmat("robot0_base"), dtype=np.float64).reshape(3, 3)
            base_rot = self._quat_xyzw_to_matrix(quat_xyzw)
            return self._matrix_to_quat_xyzw(base_xmat @ base_rot)
        except Exception:
            q = np.asarray(quat_xyzw, dtype=np.float64).reshape(-1)
            return q[:4] if q.size >= 4 else np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)

    @staticmethod
    def _clipped_step_delta(delta: np.ndarray, *, max_step: float) -> np.ndarray:
        delta = np.asarray(delta, dtype=np.float64).reshape(3)
        max_step = max(0.0, float(max_step))
        norm = float(np.linalg.norm(delta))
        if norm < 1e-9 or max_step <= 0.0:
            return np.zeros(3, dtype=np.float64)
        if norm <= max_step:
            return delta
        return delta / norm * max_step

    @staticmethod
    def _hand_preset(preset: str, strength: float = 1.0) -> np.ndarray:
        strength = float(np.clip(strength, -1.0, 1.0))
        name = str(preset).lower().strip()
        if name == "release":
            name = "open"
        if name == "precision":
            name = "pinch"
        if name in {"cylinder", "cylindrical_power", "power_cylindrical"}:
            name = "cylindrical"
        base = FOURIER_HAND_PRESETS.get(name, FOURIER_HAND_PRESETS["power"])
        return np.clip(base * strength, -1.0, 1.0)

    @staticmethod
    def _choose_grasp_preset(object_name: str, preset: str) -> str:
        requested = str(preset).lower().strip()
        if requested not in {"", "auto", "nearest"}:
            if requested == "precision":
                return "pinch"
            if requested in {"cylinder", "cylindrical_power", "power_cylindrical"}:
                return "cylindrical"
            if requested in FOURIER_HAND_PRESETS:
                return requested
            return "power"
        if object_name in {"trash", "lid_handle"}:
            return "pinch"
        return "power"

    @staticmethod
    def _max_hand_delta(before: dict[str, Any], after: dict[str, Any], arms: tuple[str, ...]) -> float | None:
        max_delta = 0.0
        found = False
        for arm in arms:
            before_qpos = before.get(arm, {}).get("joint_qpos", {})
            after_qpos = after.get(arm, {}).get("joint_qpos", {})
            for name, before_value in before_qpos.items():
                if name not in after_qpos:
                    continue
                found = True
                max_delta = max(max_delta, abs(float(after_qpos[name]) - float(before_value)))
        return max_delta if found else None

    @staticmethod
    def _quat_to_axis_angle(quat: np.ndarray) -> np.ndarray:
        """Convert an observation quaternion to axis-angle conservatively.

        RoboSuite observations usually expose quaternions in xyzw order. If the
        value looks invalid, return zero rotation rather than destabilizing IK.
        """
        if quat.size < 4 or not np.all(np.isfinite(quat[:4])):
            return np.zeros(3, dtype=np.float64)
        q = np.asarray(quat[:4], dtype=np.float64)
        norm = float(np.linalg.norm(q))
        if norm < 1e-9:
            return np.zeros(3, dtype=np.float64)
        q = q / norm
        # Prefer xyzw, which matches RoboSuite transform_utils convention.
        xyz = q[:3]
        w = float(np.clip(q[3], -1.0, 1.0))
        angle = 2.0 * np.arccos(w)
        s = np.sqrt(max(1.0 - w * w, 0.0))
        if s < 1e-8 or angle < 1e-8:
            return np.zeros(3, dtype=np.float64)
        return xyz / s * angle

    def _canonical_object_name(self, object_name: str) -> str:
        key = object_name.lower().strip().replace("-", "_").replace(" ", "_")
        for canonical, aliases in self._OBJECT_ALIASES.items():
            names = {canonical, *(alias.replace(" ", "_") for alias in aliases)}
            if key in names:
                return canonical
        raise ValueError(f"Unknown object {object_name!r}; expected one of {sorted(self._OBJECT_ALIASES)}")

    @staticmethod
    def _first_obs_array(obs: dict[str, Any], keys: tuple[str, ...], *, size: int) -> np.ndarray | None:
        for key in keys:
            if key not in obs:
                continue
            arr = np.asarray(obs[key], dtype=np.float32).reshape(-1)
            if arr.size >= size:
                return arr[:size]
        return None

    @staticmethod
    def _success_flags(obs: dict[str, Any]) -> dict[str, bool]:
        flags = {
            key: bool(value)
            for key, value in obs.items()
            if key.endswith("_in_target_bin") or key.endswith("_in_trash_bin") or key.startswith("success")
        }
        if not flags:
            # Prevent model code from treating all(success_flags.values()) as
            # true for tasks that expose no boolean success flags.
            return {"no_success_flags_exposed": False}
        return flags


__all__ = ["GR1RobocasaControlApi"]
