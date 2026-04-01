import gzip
import json
import pickle

import numpy as np
import pytest

pd = pytest.importorskip("pandas")
pytest.importorskip("pyarrow")
pytest.importorskip("imageio_ffmpeg")

from capx.data.export_lerobot_video_dataset import _infer_source_metadata, export_lerobot_dataset
from capx.envs.transition_dataset import append_transition, create_transition_dataset, set_initial_observation


def test_export_lerobot_dataset_writes_video_dataset(tmp_path) -> None:
    trial_dir = tmp_path / "trial_01" / "attempt_01"
    (trial_dir / "transition_dataset").mkdir(parents=True)
    (trial_dir / "trajectory").mkdir(parents=True)

    dataset = create_transition_dataset(
        trial=1,
        attempt=1,
        config_path="env_configs/libero/franka_libero_goal_1_privileged.yaml",
        task_prompt="Goal: put the bowl on the stove",
    )
    dataset["trial_metadata"] = {
        "trial": 1,
        "attempt": 1,
        "exclude_from_training": False,
        "exclusion_reason": None,
    }
    set_initial_observation(
        dataset,
        {
            "robot_joint_pos": np.zeros(8, dtype=np.float32),
            "robot_cartesian_pos": np.ones(8, dtype=np.float32),
            "agentview": {"images": {"rgb": np.zeros((2, 2, 3), dtype=np.uint8)}},
            "robot0_eye_in_hand": {"images": {"rgb": np.ones((2, 2, 3), dtype=np.uint8)}},
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
                "agentview_image": np.full((2, 2, 3), 10, dtype=np.uint8),
                "robot0_eye_in_hand_image": np.full((2, 2, 3), 20, dtype=np.uint8),
            },
        },
        reward=0.5,
        done=False,
        truncated=False,
        source="_step_once",
        action_context={"action_name": "goto_pose"},
        metadata={"controller_step_idx": 0},
    )
    append_transition(
        dataset,
        timestamp_s=0.2,
        wall_time_s=100.1,
        sim_step_count=2,
        action=np.full(8, 2.0, dtype=np.float32),
        observation={
            "robot_joint_pos": np.full(8, 3.0, dtype=np.float32),
            "robot_cartesian_pos": np.full(8, 4.0, dtype=np.float32),
            "low_level_observation": {
                "agentview_image": np.full((2, 2, 3), 30, dtype=np.uint8),
                "robot0_eye_in_hand_image": np.full((2, 2, 3), 40, dtype=np.uint8),
            },
        },
        reward=1.0,
        done=True,
        truncated=False,
        source="move_to_joints_blocking",
        action_context={"action_name": "close_gripper"},
        metadata={"controller_step_idx": 1},
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

    output_root = tmp_path / "lerobot_export"
    manifest = export_lerobot_dataset(
        input_root=tmp_path,
        output_root=output_root,
        robot_type="franka",
        fps=10,
        chunk_size=1000,
        crf=35,
    )

    assert manifest["num_episodes"] == 1
    assert manifest["num_frames"] == 2

    info = json.loads((output_root / "meta" / "info.json").read_text(encoding="utf-8"))
    assert info["features"]["observation.images.agentview"]["dtype"] == "video"
    assert info["features"]["observation.images.wrist"]["dtype"] == "video"
    assert info["fps"] == 10
    assert info["benchmarks"] == ["libero"]

    data_df = pd.read_parquet(output_root / "data" / "chunk-000" / "file-000.parquet")
    assert list(data_df["episode_index"]) == [0, 0]
    assert list(data_df["task_index"]) == [0, 0]
    assert "action_context" not in data_df.columns
    assert "source" not in data_df.columns

    episodes_df = pd.read_parquet(output_root / "meta" / "episodes" / "chunk-000" / "file-000.parquet")
    assert len(episodes_df) == 1
    assert int(episodes_df.iloc[0]["length"]) == 2
    assert int(episodes_df.iloc[0]["data/chunk_index"]) == 0
    assert int(episodes_df.iloc[0]["videos/observation.images.agentview/chunk_index"]) == 0
    assert float(episodes_df.iloc[0]["videos/observation.images.agentview/from_timestamp"]) == 0.0
    assert episodes_df.iloc[0]["benchmark"] == "libero"
    assert episodes_df.iloc[0]["suite_name"] == "libero_goal"
    assert int(episodes_df.iloc[0]["task_id"]) == 1
    assert int(episodes_df.iloc[0]["attempt"]) == 1
    assert bool(episodes_df.iloc[0]["excluded_from_training"]) is False

    episodes_jsonl = (output_root / "meta" / "episodes.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(episodes_jsonl) == 1
    episodes_stats_jsonl = (output_root / "meta" / "episodes_stats.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(episodes_stats_jsonl) == 1

    tasks_lines = (output_root / "meta" / "tasks.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(tasks_lines) == 1
    task_record = json.loads(tasks_lines[0])
    assert task_record["benchmark"] == "libero"
    assert task_record["suite_name"] == "libero_goal"
    assert task_record["task_id"] == 1

    agentview_video = output_root / "videos" / "observation.images.agentview" / "chunk-000" / "file-000.mp4"
    wrist_video = output_root / "videos" / "observation.images.wrist" / "chunk-000" / "file-000.mp4"
    assert agentview_video.exists()
    assert wrist_video.exists()
    assert agentview_video.stat().st_size > 0
    assert wrist_video.stat().st_size > 0


def test_export_lerobot_dataset_skips_excluded_episodes_by_default(tmp_path) -> None:
    trial_dir = tmp_path / "trial_01" / "attempt_02"
    (trial_dir / "transition_dataset").mkdir(parents=True)
    (trial_dir / "trajectory").mkdir(parents=True)

    dataset = create_transition_dataset(
        trial=1,
        attempt=2,
        config_path="env_configs/libero/franka_libero_goal_1.yaml",
        task_prompt="Goal: put the bowl on the stove",
    )
    dataset["trial_metadata"] = {
        "trial": 1,
        "attempt": 2,
        "exclude_from_training": True,
        "exclusion_reason": "sim_limit_reset",
    }
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
            "low_level_observation": {"agentview_image": np.full((2, 2, 3), 10, dtype=np.uint8)},
        },
        reward=0.0,
        done=False,
        truncated=True,
        source="_step_once",
        action_context={"action_name": "goto_pose"},
        metadata={},
    )
    with gzip.open(trial_dir / "transition_dataset" / "data.pkl.gz", "wb") as handle:
        pickle.dump(dataset, handle, protocol=pickle.HIGHEST_PROTOCOL)
    (trial_dir / "trajectory" / "metadata.json").write_text(
        json.dumps({"events": [{"event_type": "trial_complete", "truncated": True}]}),
        encoding="utf-8",
    )

    output_root = tmp_path / "lerobot_export"
    manifest = export_lerobot_dataset(
        input_root=tmp_path,
        output_root=output_root,
        robot_type="franka",
        fps=10,
        chunk_size=1000,
        crf=35,
    )

    assert manifest["num_episodes"] == 0


def test_infer_source_metadata_for_behavior1k_config(tmp_path) -> None:
    config_path = tmp_path / "b1k_cook_bacon.yaml"
    config_path.write_text(
        """
env:
  _target_: capx.envs.tasks.r1pro.r1pro_behavior.R1ProBehaviorCodeEnv
  cfg:
    _target_: capx.envs.tasks.base.CodeExecEnvConfig
    low_level:
      _target_: capx.envs.simulators.r1pro_b1k.R1ProBehaviourLowLevel
      controller_cfg: r1pro_primitives.yaml
      activity_name: cook_bacon
    privileged: false
""".strip(),
        encoding="utf-8",
    )
    metadata = _infer_source_metadata({"config_path": str(config_path)}, tmp_path)
    assert metadata["benchmark"] == "behavior1k"
    assert metadata["benchmark_variant"] == "r1pro"
    assert metadata["activity_name"] == "cook_bacon"
    assert metadata["privileged"] is False


def test_infer_source_metadata_for_robosuite_registry_config(tmp_path) -> None:
    config_path = tmp_path / "cube_stack.yaml"
    config_path.write_text(
        """
env:
  _target_: capx.envs.tasks.franka.franka_pick_place.FrankaPickPlaceCodeEnv
  cfg:
    _target_: capx.envs.tasks.base.CodeExecEnvConfig
    low_level: franka_robosuite_cubes_low_level
    privileged: true
""".strip(),
        encoding="utf-8",
    )
    metadata = _infer_source_metadata({"config_path": str(config_path)}, tmp_path)
    assert metadata["benchmark"] == "robosuite"
    assert metadata["task_family"] == "cube_stack"
    assert metadata["privileged"] is True
