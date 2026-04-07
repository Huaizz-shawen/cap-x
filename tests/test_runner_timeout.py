from types import SimpleNamespace

from capx.envs.runner import _run_single_trial_with_timeout
from capx.utils.launch_utils import TrialSummary


def test_timeout_returns_finalized_summary_without_retry(monkeypatch) -> None:
    final_summary = TrialSummary(
        trial=7,
        success=True,
        reward=1.0,
        terminated=True,
        truncated=False,
        sandbox_rc=0,
        log="done",
        task_completed=True,
        code_path="/tmp/final.py",
        num_regenerations=1,
        num_finishes=1,
        num_code_blocks=1,
    )

    def _fake_run_single_trial(env, trial, attempt_idx, args, config, multi_turn_prompt, partial_artifacts):
        partial_artifacts["final_summary"] = final_summary
        raise TimeoutError("timed out while writing videos")

    monkeypatch.setattr("capx.envs.runner._run_single_trial", _fake_run_single_trial)

    summary = _run_single_trial_with_timeout(
        env=object(),
        trial=7,
        attempt_idx=2,
        args=SimpleNamespace(),
        config={"output_dir": None},
        multi_turn_prompt=None,
        timeout_s=60,
        raise_on_timeout=True,
    )

    assert summary is final_summary
    assert summary.success is True
    assert summary.task_completed is True


def test_timeout_marks_pre_codegen_timeout_as_excluded(monkeypatch, tmp_path) -> None:
    captured = {}

    def _fake_save_trial_artifacts(config, trial, attempt_idx, **kwargs):
        captured["trial_metadata"] = kwargs["trial_metadata"]
        captured["log_lines"] = kwargs["log_lines"]
        return str(tmp_path / "code.py")

    def _fake_run_single_trial(env, trial, attempt_idx, args, config, multi_turn_prompt, partial_artifacts):
        partial_artifacts["attempt_idx"] = attempt_idx
        partial_artifacts["pre_codegen_phase"] = True
        partial_artifacts["phase_timeout_name"] = "pre-codegen"
        partial_artifacts["phase_timeout_seconds"] = 300
        partial_artifacts["info_step"] = {"sandbox_rc": 1, "stdout": "", "stderr": ""}
        partial_artifacts["code_blocks"] = []
        partial_artifacts["code_block_metadata"] = []
        partial_artifacts["all_responses"] = []
        partial_artifacts["reward"] = 0.0
        partial_artifacts["terminated"] = False
        partial_artifacts["truncated"] = False
        partial_artifacts["num_regenerations"] = 0
        partial_artifacts["num_finishes"] = 0
        partial_artifacts["num_code_blocks"] = 0
        raise TimeoutError("Trial 3 exceeded pre-codegen timeout of 300 seconds")

    monkeypatch.setattr("capx.envs.runner._run_single_trial", _fake_run_single_trial)
    monkeypatch.setattr("capx.envs.runner._save_trial_artifacts", _fake_save_trial_artifacts)

    summary = _run_single_trial_with_timeout(
        env=object(),
        trial=3,
        attempt_idx=2,
        args=SimpleNamespace(),
        config={"output_dir": str(tmp_path)},
        multi_turn_prompt=None,
        timeout_s=1000,
        raise_on_timeout=False,
    )

    assert summary.success is False
    assert summary.num_code_blocks == 0
    assert captured["trial_metadata"]["exclude_from_training"] is True
    assert captured["trial_metadata"]["exclusion_reason"] == "pre_codegen_timeout"
    assert "pre-codegen timeout of 300 seconds" in "\n".join(captured["log_lines"])
