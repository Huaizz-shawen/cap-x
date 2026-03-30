import types

from capx.envs.trial import (
    _build_snapshot_selection_prompt,
    _build_recovery_prompt,
    _parse_snapshot_selection_response,
    _restore_rollback_snapshot,
    _select_safe_snapshot,
    _select_safe_snapshot_with_model,
)


class _DummyEnv:
    def __init__(self) -> None:
        self.restored_state = None
        self.low_level_env = object()

    def restore_state(self, state):
        self.restored_state = state

    def capture_state(self):
        return {
            "sim_state": [4.0, 5.0, 6.0],
            "step_count": 3,
            "sim_step_count": 30,
        }

    def _get_observation(self):
        return {
            "full_prompt": [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": "Goal: dummy"}],
                }
            ]
        }


def test_restore_rollback_snapshot_restores_and_logs_event() -> None:
    env = _DummyEnv()
    trajectory_data = {
        "snapshots": [],
        "events": [],
        "_snapshot_payloads": {
            "snap_0003": {
                "sim_state": [1.0, 2.0, 3.0],
                "step_count": 2,
                "sim_step_count": 20,
            }
        },
    }

    obs, restored_snapshot_id = _restore_rollback_snapshot(
        env,
        trajectory_data,
        "snap_0003",
        code_block_idx=1,
        reason="regenerate",
    )

    assert obs is not None
    assert restored_snapshot_id == "snap_0000"
    assert env.restored_state == trajectory_data["_snapshot_payloads"]["snap_0003"]
    assert trajectory_data["snapshots"][0]["snapshot_type"] == "rollback_restore"
    assert trajectory_data["events"][0]["event_type"] == "rollback"
    assert trajectory_data["events"][0]["source_snapshot_id"] == "snap_0003"


def test_build_recovery_prompt_adds_recovery_instruction() -> None:
    obs = {
        "full_prompt": [
            {"role": "system", "content": "system"},
            {
                "role": "user",
                "content": [{"type": "text", "text": "Goal: dummy task"}],
            },
        ]
    }

    prompt = _build_recovery_prompt(
        obs,
        executed_code="print('hello')",
        console_stdout="hello",
        console_stderr="",
        recovery_reason="model_finished_but_task_incomplete",
    )

    assert len(prompt[-1]["content"]) == 2
    recovery_text = prompt[-1]["content"][-1]["text"]
    assert "safe earlier snapshot" in recovery_text
    assert "Do not write REGENERATE or FINISH" in recovery_text
    assert "print('hello')" in recovery_text


def test_select_safe_snapshot_prefers_latest_open_gripper_candidate() -> None:
    trajectory_data = {
        "snapshots": [
            {"snapshot_id": "snap_0000", "snapshot_type": "reset", "code_block_idx": None},
            {"snapshot_id": "snap_0001", "snapshot_type": "pre_code", "code_block_idx": 0},
            {"snapshot_id": "snap_0002", "snapshot_type": "post_code", "code_block_idx": 0},
            {"snapshot_id": "snap_0003", "snapshot_type": "rollback_restore", "code_block_idx": 0},
        ],
        "_snapshot_payloads": {
            "snap_0000": {"gripper_fraction": 1.0},
            "snap_0001": {"gripper_fraction": 0.2},
            "snap_0002": {"gripper_fraction": 1.0},
            "snap_0003": {"gripper_fraction": 1.0},
        },
    }

    snapshot_id, details = _select_safe_snapshot(
        trajectory_data,
        current_code_block_idx=0,
        purpose="regenerate",
        preferred_snapshot_id="snap_0001",
    )

    assert snapshot_id == "snap_0003"
    assert details["selected_snapshot_open_gripper"] is True
    assert details["candidate_count"] == 3


def test_select_safe_snapshot_falls_back_to_latest_candidate_when_none_open() -> None:
    trajectory_data = {
        "snapshots": [
            {"snapshot_id": "snap_0000", "snapshot_type": "reset", "code_block_idx": None},
            {"snapshot_id": "snap_0001", "snapshot_type": "pre_code", "code_block_idx": 0},
        ],
        "_snapshot_payloads": {
            "snap_0000": {"gripper_fraction": 0.1},
            "snap_0001": {"gripper_fraction": 0.2},
        },
    }

    snapshot_id, details = _select_safe_snapshot(
        trajectory_data,
        current_code_block_idx=0,
        purpose="regenerate",
        preferred_snapshot_id="snap_0001",
    )

    assert snapshot_id == "snap_0001"
    assert details["selected_snapshot_open_gripper"] is False


def test_build_snapshot_selection_prompt_includes_candidates_and_constraints() -> None:
    obs = {
        "full_prompt": [
            {
                "role": "user",
                "content": [{"type": "text", "text": "Goal: stack the red block"}],
            }
        ]
    }

    prompt = _build_snapshot_selection_prompt(
        obs,
        purpose="finish_incomplete_recovery",
        candidates=[
            {
                "snapshot_id": "snap_0001",
                "snapshot_type": "pre_code",
                "code_block_idx": 0,
                "is_open_gripper": True,
                "phase_tags": ["perceive", "prepare_grasp"],
                "summary": {"reward": 0.0, "done": False},
            }
        ],
        preferred_phases=["prepare_grasp", "perceive"],
        executed_code="move_gripper_to_pose(home_pose())",
        console_stdout="ok",
        console_stderr="",
    )

    prompt_text = prompt[-1]["content"][0]["text"]
    assert "snap_0001" in prompt_text
    assert "prepare_grasp, perceive" in prompt_text
    assert "Do not invent snapshot ids" in prompt_text


def test_parse_snapshot_selection_response_accepts_json_and_falls_back_to_regex() -> None:
    snapshot_id, reason = _parse_snapshot_selection_response(
        '{"snapshot_id": "snap_0002", "reason": "open gripper"}',
        valid_snapshot_ids={"snap_0001", "snap_0002"},
    )
    assert snapshot_id == "snap_0002"
    assert reason == "open gripper"

    snapshot_id, reason = _parse_snapshot_selection_response(
        "Best choice is snap_0001 because it is earlier.",
        valid_snapshot_ids={"snap_0001", "snap_0002"},
    )
    assert snapshot_id == "snap_0001"
    assert reason is None

    snapshot_id, _ = _parse_snapshot_selection_response(
        '{"snapshot_id": "snap_9999"}',
        valid_snapshot_ids={"snap_0001", "snap_0002"},
    )
    assert snapshot_id is None


def test_select_safe_snapshot_with_model_falls_back_on_invalid_selection(monkeypatch) -> None:
    trajectory_data = {
        "snapshots": [
            {"snapshot_id": "snap_0000", "snapshot_type": "reset", "code_block_idx": None, "summary": {}},
            {"snapshot_id": "snap_0001", "snapshot_type": "pre_code", "code_block_idx": 0, "summary": {}},
        ],
        "_snapshot_payloads": {
            "snap_0000": {"gripper_fraction": 1.0},
            "snap_0001": {"gripper_fraction": 0.2},
        },
    }

    def _fake_query(*args, **kwargs):
        return "snap_9999", "snap_9999", None, [], None

    monkeypatch.setattr("capx.envs.trial._query_snapshot_selection", _fake_query)

    snapshot_id, details, model_details = _select_safe_snapshot_with_model(
        types.SimpleNamespace(),
        {"full_prompt": []},
        trajectory_data,
        current_code_block_idx=0,
        purpose="regenerate",
        preferred_snapshot_id="snap_0001",
        executed_code="pass",
        console_stdout="",
        console_stderr="",
    )

    assert snapshot_id == "snap_0000"
    assert details["selected_snapshot_id"] == "snap_0000"
    assert model_details["fallback_used"] is True
    assert model_details["fallback_reason"] == "invalid_model_selection"
