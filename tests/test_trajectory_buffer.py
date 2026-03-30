import json
import pickle

import numpy as np

from capx.envs.trajectory_buffer import (
    append_event,
    append_snapshot,
    create_trajectory_buffer,
    save_trajectory_artifacts,
)


class _DummyEnv:
    def __init__(self) -> None:
        self.calls = 0

    def capture_state(self) -> dict:
        self.calls += 1
        return {
            "sim_state": np.array([self.calls, self.calls + 1], dtype=np.float64),
            "step_count": self.calls,
            "sim_step_count": self.calls * 10,
        }


def test_trajectory_buffer_saves_metadata_and_snapshots(tmp_path) -> None:
    env = _DummyEnv()
    trajectory = create_trajectory_buffer(
        trial=3,
        config_path="env_configs/libero/franka_libero_goal_1.yaml",
        task_prompt="Put the bowl on the stove.",
    )

    first_snapshot = append_snapshot(
        trajectory,
        env=env,
        snapshot_type="reset",
        label="canonical_reset_state",
    )
    second_snapshot = append_snapshot(
        trajectory,
        state={
            "sim_state": np.array([9.0, 10.0], dtype=np.float64),
            "step_count": 4,
            "sim_step_count": 40,
        },
        snapshot_type="post_code",
        code_block_idx=0,
    )
    append_event(
        trajectory,
        "code_execution",
        code_block_idx=0,
        snapshot_before=first_snapshot,
        snapshot_after=second_snapshot,
        reward=np.float64(1.0),
    )

    save_trajectory_artifacts(tmp_path, trajectory)

    metadata = json.loads((tmp_path / "trajectory" / "metadata.json").read_text())
    assert metadata["trial"] == 3
    assert len(metadata["snapshots"]) == 2
    assert metadata["events"][0]["reward"] == 1.0

    with open(tmp_path / "trajectory" / "snapshots" / f"{first_snapshot}.pkl", "rb") as handle:
        saved_snapshot = pickle.load(handle)
    np.testing.assert_allclose(saved_snapshot["sim_state"], np.array([1.0, 2.0]))
