from types import SimpleNamespace

import numpy as np

from capx.envs.simulators.libero import FrankaLiberoEnv


def test_step_once_noops_after_episode_done() -> None:
    env = object.__new__(FrankaLiberoEnv)
    env._current_done = True
    env._sim_step_count = 12
    env.max_steps = 4000
    env.handle = SimpleNamespace(
        step=lambda action: (_ for _ in ()).throw(AssertionError("step should not be called"))
    )

    assert env._step_once() is False


def test_move_to_joints_blocking_stops_immediately_after_episode_done() -> None:
    env = object.__new__(FrankaLiberoEnv)
    env._current_done = True
    env._sim_step_count = 12
    env.max_steps = 4000
    env._current_joints = np.zeros(7, dtype=np.float64)
    env.handle = SimpleNamespace(
        env=SimpleNamespace(
            sim=SimpleNamespace(
                data=SimpleNamespace(qpos=np.zeros(7, dtype=np.float64)),
            )
        ),
        step=lambda action: (_ for _ in ()).throw(AssertionError("step should not be called")),
    )
    env._panda_joint_qpos_addrs = list(range(7))

    env.move_to_joints_blocking(np.ones(7, dtype=np.float64))
