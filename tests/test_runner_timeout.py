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

    def _fake_run_single_trial(env, trial, args, config, multi_turn_prompt, partial_artifacts):
        partial_artifacts["final_summary"] = final_summary
        raise TimeoutError("timed out while writing videos")

    monkeypatch.setattr("capx.envs.runner._run_single_trial", _fake_run_single_trial)

    summary = _run_single_trial_with_timeout(
        env=object(),
        trial=7,
        args=SimpleNamespace(),
        config={"output_dir": None},
        multi_turn_prompt=None,
        timeout_s=60,
        raise_on_timeout=True,
    )

    assert summary is final_summary
    assert summary.success is True
    assert summary.task_completed is True
