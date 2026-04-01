from typing import Any

from capx.envs.trial import _run_single_trial


class _CompletionEnv:
    def __init__(self) -> None:
        self.low_level_env = object()
        self.oracle_code = "print('done')"
        self._frames: list[Any] = []

    def reset(self, *, options=None, seed=None):
        obs = {
            "full_prompt": [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": "Goal: finish task"}],
                }
            ]
        }
        return obs, {}

    def step(self, code):
        obs = {
            "full_prompt": [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": "Goal: finish task"}],
                }
            ]
        }
        return obs, 1.0, True, False, {
            "sandbox_rc": 0,
            "stdout": "ok",
            "stderr": "",
            "task_completed": True,
        }

    def enable_video_capture(self, enabled=True, *, clear=True, wrist_camera=False):
        return None

    def get_video_frame_count(self) -> int:
        return 0

    def get_video_frames(self, *, clear=False):
        return []

    def get_reset_state(self):
        return None

    def get_transition_dataset(self):
        return None

    def capture_state(self):
        return {"sim_state": [1, 2, 3]}


class _Args:
    config_path = "env_configs/libero/franka_libero_goal_1.yaml"
    visual_differencing_model = "gpt-5.3-codex"
    visual_differencing_model_server_url = "http://unused"
    visual_differencing_model_api_key = "unused"
    max_tokens = 128
    temperature = 0.0
    reasoning_effort = "medium"
    debug = False
    model = "gpt-5.3-codex"
    use_legacy_multi_turn_decision_prompt = False


def test_trial_stops_immediately_after_task_completion(monkeypatch, tmp_path) -> None:
    called = {"multi_turn": 0}

    def _fail_if_called(*args, **kwargs):
        called["multi_turn"] += 1
        raise AssertionError("multi-turn should not run after task completion")

    monkeypatch.setattr("capx.envs.trial._handle_multi_turn_step", _fail_if_called)
    monkeypatch.setattr(
        "capx.envs.trial._save_trial_artifacts",
        lambda *args, **kwargs: str(tmp_path / "code.py"),
    )

    env = _CompletionEnv()
    config = {
        "record_video": False,
        "output_dir": str(tmp_path),
        "use_img_differencing": False,
        "use_video_differencing": False,
        "use_visual_feedback": False,
        "use_parallel_ensemble": False,
        "use_oracle_code": True,
        "enable_eap_rollback": False,
        "enable_eap_recovery": False,
        "enable_eap_model_snapshot_selection": False,
        "save_multiturn_prompts": False,
        "evolve_skill_library": False,
    }

    summary = _run_single_trial(
        env,
        trial=1,
        attempt_idx=1,
        args=_Args(),
        config=config,
        multi_turn_prompt="You may REGENERATE or FINISH.",
    )

    assert summary.task_completed is True
    assert summary.num_code_blocks == 1
    assert called["multi_turn"] == 0
