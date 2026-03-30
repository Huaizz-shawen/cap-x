import os

import numpy as np

os.environ.setdefault("MUJOCO_GL", "egl")

from capx.envs.simulators.libero import FrankaLiberoEnv


def test_libero_capture_and_restore_state() -> None:
    env = FrankaLiberoEnv("libero_goal", 1, privileged=True, enable_render=False)

    snapshot = env.capture_state()
    initial_state = snapshot["sim_state"].copy()
    initial_obs = env.get_observation()
    initial_joint_pos = initial_obs["robot_joint_pos"].copy()
    initial_cartesian = initial_obs["robot_cartesian_pos"].copy()

    env._set_gripper(0.0)
    for _ in range(5):
        env._step_once()

    mutated_state = env.handle.env.sim.get_state().flatten().copy()
    assert not np.allclose(mutated_state, initial_state)

    env.restore_state(snapshot)

    restored_state = env.handle.env.sim.get_state().flatten().copy()
    restored_obs = env.get_observation()

    np.testing.assert_allclose(restored_state, initial_state, atol=1e-8, rtol=0.0)
    np.testing.assert_allclose(restored_obs["robot_joint_pos"], initial_joint_pos, atol=2e-4, rtol=0.0)
    np.testing.assert_allclose(
        restored_obs["robot_cartesian_pos"], initial_cartesian, atol=2e-4, rtol=0.0
    )


def test_libero_get_reset_state_returns_copy() -> None:
    env = FrankaLiberoEnv("libero_goal", 1, privileged=True, enable_render=False)

    reset_state_a = env.get_reset_state()
    reset_state_b = env.get_reset_state()

    assert reset_state_a is not None
    assert reset_state_b is not None
    assert reset_state_a is not reset_state_b

    reset_state_a["sim_state"][0] += 1.0
    assert not np.isclose(reset_state_a["sim_state"][0], reset_state_b["sim_state"][0])
