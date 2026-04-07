from __future__ import annotations

import numpy as np

from capx.integrations.franka.libero import FrankaLiberoApi


class _FakeEnv:
    def __init__(self) -> None:
        self.steps = 0
        self._sim_step_count = 0

    def _step_once(self) -> None:
        self.steps += 1
        self._sim_step_count += 1


def _make_api() -> FrankaLiberoApi:
    api = object.__new__(FrankaLiberoApi)
    api._env = _FakeEnv()
    api._language_perception_cache = {}
    api._language_pose_cache = {}
    api._candidate_selection_args = None
    api.goto_calls = []
    api.open_calls = 0

    def _goto(position, quat, z_approach=0.0):
        api.goto_calls.append(
            (
                np.asarray(position, dtype=np.float64).copy(),
                np.asarray(quat, dtype=np.float64).copy(),
                float(z_approach),
            )
        )

    def _open():
        api.open_calls += 1

    api.goto_pose = _goto
    api.open_gripper = _open
    api.filter_noise = lambda points: (np.asarray(points, dtype=np.float64), None)
    api.get_oriented_bounding_box_from_3d_points = lambda points: {
        "center": np.asarray(points, dtype=np.float64).mean(axis=0),
        "R": np.eye(3),
    }
    return api


def test_place_on_object_center_falls_back_to_cached_perception() -> None:
    api = _make_api()
    api._set_cached_language_perception(
        "the_plate",
        {
            "agentview_mask": None,
            "wrist_mask": None,
            "points_3d": np.array(
                [
                    [0.70, 0.20, 0.00],
                    [0.74, 0.20, 0.00],
                    [0.72, 0.18, 0.01],
                    [0.72, 0.22, 0.01],
                    [0.71, 0.19, 0.00],
                    [0.73, 0.21, 0.00],
                    [0.72, 0.20, 0.01],
                    [0.72, 0.20, 0.00],
                ],
                dtype=np.float64,
            ),
            "agentview_points_3d": np.empty((0, 3)),
            "wrist_points_3d": np.empty((0, 3)),
            "agentview_score": 1.0,
            "wrist_score": None,
        },
    )

    def _raise(*args, **kwargs):
        raise ValueError("sam3 unavailable")

    api.get_object_3d_points_and_masks_from_language = _raise

    api.place_on_object_center("the_plate", grasp_quaternion_wxyz=None)

    assert len(api.goto_calls) == 3
    hover, release, retreat = api.goto_calls
    np.testing.assert_allclose(hover[1], np.array([0.0, 0.0, 1.0, 0.0]))
    np.testing.assert_allclose(release[1], np.array([0.0, 0.0, 1.0, 0.0]))
    assert release[0][2] < hover[0][2]
    assert retreat[0][2] > release[0][2]
    assert api.open_calls == 1
    assert api._env.steps == 30


def test_place_on_object_center_falls_back_to_cached_pose() -> None:
    api = _make_api()
    api._set_cached_language_pose(
        "the_plate",
        np.array([0.75, 0.18, 0.0], dtype=np.float64),
        np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float64),
    )

    def _raise(*args, **kwargs):
        raise ValueError("sam3 unavailable")

    api.get_object_3d_points_and_masks_from_language = _raise

    api.place_on_object_center("the_plate", grasp_quaternion_wxyz=None)

    assert len(api.goto_calls) == 3
    hover, release, retreat = api.goto_calls
    np.testing.assert_allclose(hover[0][:2], np.array([0.75, 0.18]))
    np.testing.assert_allclose(release[0][:2], np.array([0.75, 0.18]))
    np.testing.assert_allclose(retreat[0][:2], np.array([0.75, 0.18]))


def test_get_object_pose_falls_back_to_recent_cached_perception() -> None:
    api = _make_api()
    api._set_cached_language_perception(
        "the_bowl",
        {
            "agentview_mask": None,
            "wrist_mask": None,
            "points_3d": np.array(
                [
                    [0.60, 0.10, 0.01],
                    [0.62, 0.10, 0.01],
                    [0.61, 0.12, 0.02],
                ],
                dtype=np.float64,
            ),
            "agentview_points_3d": np.empty((0, 3)),
            "wrist_points_3d": np.empty((0, 3)),
            "agentview_score": 1.0,
            "wrist_score": None,
        },
    )

    def _raise(*args, **kwargs):
        raise ValueError("sam3 unavailable")

    api.get_object_3d_points_and_masks_from_language = _raise

    pos, quat = api.get_object_pose("the_bowl")

    np.testing.assert_allclose(pos, np.array([0.61, 0.10666667, 0.01333333]), atol=1e-6)
    np.testing.assert_allclose(quat, np.array([0.0, 0.0, 1.0, 0.0]))


def test_get_object_pose_rejects_stale_dynamic_cache() -> None:
    api = _make_api()
    api._set_cached_language_perception(
        "the_bowl",
        {
            "agentview_mask": None,
            "wrist_mask": None,
            "points_3d": np.array(
                [
                    [0.60, 0.10, 0.01],
                    [0.62, 0.10, 0.01],
                    [0.61, 0.12, 0.02],
                ],
                dtype=np.float64,
            ),
            "agentview_points_3d": np.empty((0, 3)),
            "wrist_points_3d": np.empty((0, 3)),
            "agentview_score": 1.0,
            "wrist_score": None,
        },
    )
    api._env._sim_step_count = 200

    def _raise(*args, **kwargs):
        raise ValueError("sam3 unavailable")

    api.get_object_3d_points_and_masks_from_language = _raise

    try:
        api.get_object_pose("the_bowl")
    except ValueError as exc:
        assert "sam3 unavailable" in str(exc)
    else:
        raise AssertionError("Expected stale dynamic cache to be rejected")


def test_get_object_pose_accepts_stale_static_cache() -> None:
    api = _make_api()
    api._set_cached_language_pose(
        "the_plate",
        np.array([0.75, 0.18, 0.0], dtype=np.float64),
        np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float64),
    )
    api._env._sim_step_count = 400

    def _raise(*args, **kwargs):
        raise ValueError("sam3 unavailable")

    api.get_object_3d_points_and_masks_from_language = _raise

    pos, quat = api.get_object_pose("the_plate")

    np.testing.assert_allclose(pos, np.array([0.75, 0.18, 0.0]))
    np.testing.assert_allclose(quat, np.array([0.0, 0.0, 1.0, 0.0]))
