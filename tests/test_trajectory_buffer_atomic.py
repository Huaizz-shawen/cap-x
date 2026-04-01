import json
import pickle

from capx.envs.trajectory_buffer import append_event, append_snapshot, create_trajectory_buffer, save_trajectory_artifacts


def test_save_trajectory_artifacts_roundtrip_and_no_temp_files(tmp_path) -> None:
    trajectory_data = create_trajectory_buffer(
        trial=1,
        attempt=2,
        config_path="env_configs/libero/franka_libero_goal_1.yaml",
        task_prompt="Goal: test",
    )
    append_snapshot(
        trajectory_data,
        state={
            "sim_state": [1.0, 2.0, 3.0],
            "step_count": 0,
            "sim_step_count": 10,
            "gripper_fraction": 1.0,
            "current_reward": 0.0,
            "current_done": False,
        },
        snapshot_type="reset",
        label="canonical_reset_state",
        phase_tags=["reset"],
    )
    append_event(trajectory_data, "trial_complete", success=True, task_completed=True)

    save_trajectory_artifacts(tmp_path, trajectory_data)

    metadata = json.loads((tmp_path / "trajectory" / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["trial"] == 1
    assert metadata["attempt"] == 2
    assert metadata["events"][0]["event_type"] == "trial_complete"

    with open(tmp_path / "trajectory" / "snapshots" / "snap_0000.pkl", "rb") as handle:
        payload = pickle.load(handle)
    assert payload["sim_step_count"] == 10
    assert not list((tmp_path / "trajectory").glob("*.tmp"))
    assert not list((tmp_path / "trajectory" / "snapshots").glob("*.tmp"))
