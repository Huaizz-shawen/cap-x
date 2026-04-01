import gzip
import json
import pickle

import numpy as np

from capx.envs.transition_dataset import (
    append_transition,
    create_transition_dataset,
    save_transition_dataset,
    set_initial_observation,
)


def test_transition_dataset_save_roundtrip(tmp_path) -> None:
    dataset = create_transition_dataset(
        trial=3,
        config_path="env_configs/libero/test.yaml",
        task_prompt="put the bowl on the stove",
    )
    set_initial_observation(
        dataset,
        {
            "robot_joint_pos": np.array([1.0, 2.0, 3.0]),
        },
    )
    append_transition(
        dataset,
        timestamp_s=0.05,
        wall_time_s=100.0,
        sim_step_count=1,
        action=np.array([0.1, 0.2, 0.3]),
        observation={"robot_joint_pos": np.array([4.0, 5.0, 6.0])},
        reward=0.0,
        done=False,
        truncated=False,
        source="move_to_joints_blocking",
        action_context={"action_name": "goto_pose"},
        metadata={"controller_step_idx": 0},
    )

    save_transition_dataset(tmp_path, dataset)

    metadata = json.loads((tmp_path / "transition_dataset" / "metadata.json").read_text())
    assert metadata["transition_count"] == 1
    assert metadata["has_initial_observation"] is True

    with gzip.open(tmp_path / "transition_dataset" / "data.pkl.gz", "rb") as handle:
        restored = pickle.load(handle)

    assert restored["trial"] == 3
    assert restored["config_path"] == "env_configs/libero/test.yaml"
    assert restored["transitions"][0]["source"] == "move_to_joints_blocking"
    np.testing.assert_allclose(restored["transitions"][0]["action"], [0.1, 0.2, 0.3])
    assert not list((tmp_path / "transition_dataset").glob("*.tmp"))
