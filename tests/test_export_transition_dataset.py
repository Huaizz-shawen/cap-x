import gzip
import json
import pickle

import numpy as np

from capx.data.export_transition_dataset import export_directory
from capx.envs.transition_dataset import (
    append_transition,
    create_transition_dataset,
    set_initial_observation,
)


def test_export_directory_writes_training_ready_episode(tmp_path) -> None:
    trial_dir = tmp_path / "trial_01_sandboxrc_0_reward_1.000_taskcompleted_1"
    (trial_dir / "transition_dataset").mkdir(parents=True)
    (trial_dir / "trajectory").mkdir(parents=True)

    dataset = create_transition_dataset(
        trial=1,
        config_path="env_configs/libero/franka_libero_goal_1_privileged.yaml",
        task_prompt="put the bowl on the stove",
    )
    set_initial_observation(
        dataset,
        {
            "robot_joint_pos": np.zeros(8, dtype=np.float32),
            "robot_cartesian_pos": np.ones(8, dtype=np.float32),
            "agentview": {"images": {"rgb": np.zeros((2, 2, 3), dtype=np.uint8)}},
        },
    )
    append_transition(
        dataset,
        timestamp_s=0.1,
        wall_time_s=100.0,
        sim_step_count=1,
        action=np.arange(8, dtype=np.float32),
        observation={
            "robot_joint_pos": np.ones(8, dtype=np.float32),
            "robot_cartesian_pos": np.full(8, 2.0, dtype=np.float32),
            "low_level_observation": {
                "agentview_image": np.zeros((2, 2, 3), dtype=np.uint8),
                "agentview_depth": np.ones((2, 2, 1), dtype=np.float32),
                "robot0_eye_in_hand_image": np.ones((2, 2, 3), dtype=np.uint8),
                "robot0_eye_in_hand_depth": np.full((2, 2, 1), 2.0, dtype=np.float32),
            },
        },
        reward=1.0,
        done=True,
        truncated=False,
        source="move_to_joints_blocking",
        action_context={"action_name": "goto_pose"},
        metadata={"controller_step_idx": 0},
    )

    with gzip.open(trial_dir / "transition_dataset" / "data.pkl.gz", "wb") as handle:
        pickle.dump(dataset, handle, protocol=pickle.HIGHEST_PROTOCOL)
    (trial_dir / "trajectory" / "metadata.json").write_text(
        json.dumps(
            {
                "events": [
                    {
                        "event_type": "trial_complete",
                        "success": True,
                        "reward": 1.0,
                        "task_completed": True,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    output_root = tmp_path / "exported"
    manifest = export_directory(
        input_root=tmp_path,
        output_root=output_root,
        include_images=True,
        include_depth=True,
    )

    assert manifest["num_episodes"] == 1
    episode_npz = output_root / "episodes" / "episode_000001.npz"
    episode_json = output_root / "episodes" / "episode_000001.json"
    assert episode_npz.exists()
    assert episode_json.exists()

    arrays = np.load(episode_npz)
    assert arrays["actions"].shape == (1, 8)
    assert arrays["robot_joint_pos"].shape == (1, 8)
    assert arrays["agentview_rgb"].shape == (1, 2, 2, 3)
    assert arrays["agentview_depth"].shape == (1, 2, 2, 1)

    metadata = json.loads(episode_json.read_text())
    assert metadata["trial_summary"]["task_completed"] is True
    assert metadata["step_metadata"]["action_context_counts"]["goto_pose"] == 1
