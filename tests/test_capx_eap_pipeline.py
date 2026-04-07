from __future__ import annotations

import importlib.util
import json
import subprocess
from argparse import Namespace
from pathlib import Path

import yaml


def _load_pipeline_module():
    path = (
        Path(__file__).resolve().parents[1]
        / "skills"
        / "capx-eap-data-collection"
        / "scripts"
        / "capx_eap_pipeline.py"
    )
    spec = importlib.util.spec_from_file_location("capx_eap_pipeline", path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_load_api_ports_from_config(tmp_path):
    module = _load_pipeline_module()
    repo_root = tmp_path
    cfg_dir = repo_root / "env_configs" / "libero"
    cfg_dir.mkdir(parents=True)
    cfg_path = cfg_dir / "test.yaml"
    cfg_path.write_text(
        """
api_servers:
  - port: 8116
  - port: 8115
  - host: 127.0.0.1
"""
    )

    ports = module._load_api_ports(repo_root, str(cfg_path.relative_to(repo_root)))

    assert ports == [8115, 8116]


def test_strip_tmux_flags():
    module = _load_pipeline_module()
    argv = [
        "collect",
        "--config-path",
        "env_configs/libero/franka_libero_goal_1.yaml",
        "--tmux-session",
        "goal1",
        "--tmux-log-file=outputs/tmux.log",
        "--tmux-replace-existing",
        "--total-trials",
        "10",
    ]

    stripped = module._strip_tmux_flags(argv)

    assert stripped == [
        "collect",
        "--config-path",
        "env_configs/libero/franka_libero_goal_1.yaml",
        "--total-trials",
        "10",
    ]


def test_strip_detached_flags():
    module = _load_pipeline_module()
    argv = [
        "collect-manifest",
        "--manifest-path",
        "outputs/manifests/libero_standard_4_2trials.yaml",
        "--detached",
        "--detached-log-file",
        "outputs/detached.log",
        "--detached-pid-file=outputs/detached.pid",
        "--detached-replace-existing",
        "--output-root",
        "outputs/libero_spatial_10x2",
    ]

    stripped = module._strip_detached_flags(argv)

    assert stripped == [
        "collect-manifest",
        "--manifest-path",
        "outputs/manifests/libero_standard_4_2trials.yaml",
        "--output-root",
        "outputs/libero_spatial_10x2",
    ]


def test_build_collect_env_defaults():
    module = _load_pipeline_module()
    env = module._build_collect_env(
        Namespace(
            request_timeout_s=None,
            request_max_attempts=None,
            request_max_retry_walltime_s=None,
            request_retry_initial_s=None,
            request_retry_max_sleep_s=None,
        )
    )

    assert env["CAPX_MODEL_REQUEST_TIMEOUT_S"] == str(module.DEFAULT_MODEL_REQUEST_TIMEOUT_S)
    assert env["CAPX_MODEL_RETRY_MAX_ATTEMPTS"] == str(module.DEFAULT_MODEL_RETRY_MAX_ATTEMPTS)


def test_build_validate_env_uses_writable_tmp_cache(tmp_path):
    module = _load_pipeline_module()
    env = module._build_validate_env(tmp_path / "hf_cache")

    assert env["HF_HOME"].startswith(str(tmp_path))
    assert env["HF_DATASETS_CACHE"].startswith(str(tmp_path))
    assert env["HUGGINGFACE_HUB_CACHE"].startswith(str(tmp_path))
    assert Path(env["HF_HOME"]).exists()
    assert Path(env["HF_DATASETS_CACHE"]).exists()


def test_materialize_manifest_preset_contains_standard_libero_suites():
    module = _load_pipeline_module()

    manifest = module._materialize_manifest_preset("libero_standard_4", trials_per_task=3)

    assert manifest["benchmark"] == "libero_standard_4"
    assert manifest["default_trials_per_task"] == 3
    suite_names = [suite["suite_name"] for suite in manifest["suites"]]
    assert suite_names == ["libero_10", "libero_object", "libero_spatial", "libero_goal"]
    assert all(suite["task_ids"] == list(range(10)) for suite in manifest["suites"])
    assert all(suite["trials_per_task"] == 3 for suite in manifest["suites"])


def test_default_detached_paths_derive_from_output_root(tmp_path):
    module = _load_pipeline_module()

    log_path, pid_path = module._default_detached_paths(
        tmp_path,
        Namespace(
            command="collect-manifest",
            manifest_path="outputs/manifests/libero_standard_4_2trials.yaml",
            output_root="outputs/libero_spatial_10x2_overnight_v2",
            output_dir=None,
            config_path=None,
        ),
    )

    assert log_path == tmp_path / "outputs" / module.DETACHED_LOG_DIRNAME / "libero_standard_4_2trials.log"
    assert pid_path == tmp_path / "outputs" / module.DETACHED_PID_DIRNAME / "libero_standard_4_2trials.pid"


def test_manifest_orchestrator_detached_paths_prefers_env(tmp_path, monkeypatch):
    module = _load_pipeline_module()
    log_path = tmp_path / "external.log"
    pid_path = tmp_path / "external.pid"
    monkeypatch.setenv("CAPX_DETACHED_LOG_FILE", str(log_path))
    monkeypatch.setenv("CAPX_DETACHED_PID_FILE", str(pid_path))

    resolved_log_path, resolved_pid_path = module._manifest_orchestrator_detached_paths(
        tmp_path,
        Namespace(
            command="collect-manifest",
            manifest_path="outputs/manifests/libero_standard_4_2trials.yaml",
            output_root="outputs/libero_spatial_10x2_overnight_v2",
            output_dir=None,
            config_path=None,
        ),
    )

    assert resolved_log_path == log_path.resolve()
    assert resolved_pid_path == pid_path.resolve()


def test_iter_manifest_jobs_supports_suite_filter_and_trial_override():
    module = _load_pipeline_module()
    manifest = module._materialize_manifest_preset("libero_standard_4", trials_per_task=2)

    jobs = module._iter_manifest_jobs(
        manifest,
        suite_filters=["libero_spatial"],
        trials_per_task_override=4,
    )

    assert len(jobs) == 10
    assert {job["suite_name"] for job in jobs} == {"libero_spatial"}
    assert {job["trials_per_task"] for job in jobs} == {4}
    assert jobs[0]["task_id"] == 0
    assert jobs[-1]["task_id"] == 9


def test_make_manifest_job_config_overrides_suite_and_task_id(tmp_path):
    module = _load_pipeline_module()
    repo_root = tmp_path
    cfg_dir = repo_root / "env_configs" / "libero"
    cfg_dir.mkdir(parents=True)
    base_cfg = cfg_dir / "base.yaml"
    base_cfg.write_text(
        """
env:
  cfg:
    low_level:
      suite_name: libero_goal
      task_id: 1
""",
        encoding="utf-8",
    )
    scratch_dir = tmp_path / "scratch"
    scratch_dir.mkdir()

    generated = module._make_manifest_job_config(
        repo_root=repo_root,
        base_config_path=str(base_cfg.relative_to(repo_root)),
        suite_name="libero_spatial",
        task_id=7,
        scratch_dir=scratch_dir,
    )

    generated_cfg = yaml.safe_load(generated.read_text(encoding="utf-8"))
    assert generated_cfg["env"]["cfg"]["low_level"]["suite_name"] == "libero_spatial"
    assert generated_cfg["env"]["cfg"]["low_level"]["task_id"] == 7


def test_make_manifest_job_config_disables_record_video_by_default_for_manifest_runs(tmp_path):
    module = _load_pipeline_module()
    repo_root = tmp_path
    cfg_dir = repo_root / "env_configs" / "libero"
    cfg_dir.mkdir(parents=True)
    base_cfg = cfg_dir / "base.yaml"
    base_cfg.write_text(
        """
env:
  cfg:
    low_level:
      suite_name: libero_goal
      task_id: 1
record_video: true
""",
        encoding="utf-8",
    )
    scratch_dir = tmp_path / "scratch"
    scratch_dir.mkdir()

    generated = module._make_manifest_job_config(
        repo_root=repo_root,
        base_config_path=str(base_cfg.relative_to(repo_root)),
        suite_name="libero_spatial",
        task_id=0,
        scratch_dir=scratch_dir,
        record_video=False,
    )

    generated_cfg = yaml.safe_load(generated.read_text(encoding="utf-8"))
    assert generated_cfg["record_video"] is False


def test_make_manifest_job_config_writes_resume_idx(tmp_path):
    module = _load_pipeline_module()
    repo_root = tmp_path
    cfg_dir = repo_root / "env_configs" / "libero"
    cfg_dir.mkdir(parents=True)
    base_cfg = cfg_dir / "base.yaml"
    base_cfg.write_text(
        """
env:
  cfg:
    low_level:
      suite_name: libero_goal
      task_id: 1
""",
        encoding="utf-8",
    )
    scratch_dir = tmp_path / "scratch"
    scratch_dir.mkdir()

    generated = module._make_manifest_job_config(
        repo_root=repo_root,
        base_config_path=str(base_cfg.relative_to(repo_root)),
        suite_name="libero_spatial",
        task_id=0,
        scratch_dir=scratch_dir,
        resume_idx=2,
    )

    generated_cfg = yaml.safe_load(generated.read_text(encoding="utf-8"))
    assert generated_cfg["resume_idx"] == 2


def test_summarize_completed_trials_counts_successful_trials(tmp_path):
    module = _load_pipeline_module()
    output_dir = tmp_path / "run"
    for trial, attempt, task_completed in [(1, 1, True), (2, 1, False), (2, 2, True)]:
        attempt_dir = output_dir / f"trial_{trial:02d}" / f"attempt_{attempt:02d}"
        attempt_dir.mkdir(parents=True, exist_ok=True)
        (attempt_dir / "trial_metadata.json").write_text(
            json.dumps(
                {
                    "trial": trial,
                    "attempt": attempt,
                    "task_completed": task_completed,
                }
            ),
            encoding="utf-8",
        )

    summary = module._summarize_completed_trials(output_dir)

    assert summary["num_completed_trials"] == 2
    assert summary["num_successful_trials"] == 2
    assert summary["completed_trial_ids"] == [1, 2]
    assert summary["successful_trial_ids"] == [1, 2]


def test_summarize_manifest_job_reports_resume_idx(tmp_path):
    module = _load_pipeline_module()
    output_dir = tmp_path / "run"
    attempt_dir = output_dir / "trial_01" / "attempt_01"
    attempt_dir.mkdir(parents=True, exist_ok=True)
    (attempt_dir / "trial_metadata.json").write_text(
        json.dumps(
            {
                "trial": 1,
                "attempt": 1,
                "task_completed": False,
            }
        ),
        encoding="utf-8",
    )

    summary = module._summarize_manifest_job(output_dir, expected_trials=2)

    assert summary["complete"] is False
    assert summary["done_flag_exists"] is False
    assert summary["num_completed_trials"] == 1
    assert summary["next_resume_idx"] == 2


def test_collect_treats_signal_exit_as_complete_when_all_trial_metadata_exist(tmp_path, monkeypatch):
    module = _load_pipeline_module()
    repo_root = tmp_path
    (repo_root / "capx" / "envs").mkdir(parents=True)
    (repo_root / ".venv-libero" / "bin").mkdir(parents=True)
    python_bin = repo_root / ".venv-libero" / "bin" / "python"
    python_bin.write_text("", encoding="utf-8")
    python_bin.chmod(0o755)

    output_dir = repo_root / "outputs" / "collect_task"
    effective_output_dir = output_dir.parent / "gpt-5.3-codex" / output_dir.name
    for trial in [1, 2]:
        attempt_dir = effective_output_dir / f"trial_{trial:02d}" / "attempt_01"
        attempt_dir.mkdir(parents=True, exist_ok=True)
        (attempt_dir / "trial_metadata.json").write_text(
            json.dumps(
                {
                    "trial": trial,
                    "attempt": 1,
                    "task_completed": True,
                }
            ),
            encoding="utf-8",
        )

    monkeypatch.setattr(module, "_find_repo_root", lambda explicit: repo_root)
    monkeypatch.setattr(module, "_select_python", lambda rr: str(python_bin))
    monkeypatch.setattr(module, "_cleanup_services", lambda *args, **kwargs: None)
    monkeypatch.setattr(module, "_build_collect_env", lambda args: {})

    def _raise(*args, **kwargs):
        raise subprocess.CalledProcessError(
            returncode=-9,
            cmd=["python", "capx/envs/launch.py"],
        )

    monkeypatch.setattr(module, "_run", _raise)

    result = module._collect(
        Namespace(
            repo_root=str(repo_root),
            api_bash_profile=None,
            api_bash_file=None,
            config_path="env_configs/libero/franka_libero_spatial_0.yaml",
            output_dir=str(output_dir.relative_to(repo_root)),
            server_url=None,
            api_key=None,
            model=None,
            total_trials=2,
            num_workers=1,
            record_video=False,
            enable_eap_rollback=False,
            enable_eap_recovery=False,
            enable_eap_model_snapshot_selection=False,
            use_img_differencing=False,
            request_timeout_s=None,
            request_max_attempts=None,
            request_max_retry_walltime_s=None,
            request_retry_initial_s=None,
            request_retry_max_sleep_s=None,
            tmux_session=None,
            tmux_log_file=None,
            tmux_replace_existing=False,
            tmux_child=True,
            extra_arg=[],
        )
    )

    assert result["collection_output_dir"] == str(output_dir.resolve())


def test_spawn_detached_process_rejects_existing_live_pid(tmp_path, monkeypatch):
    module = _load_pipeline_module()
    repo_root = tmp_path
    (repo_root / ".venv-libero" / "bin").mkdir(parents=True)
    python_bin = repo_root / ".venv-libero" / "bin" / "python"
    python_bin.write_text("", encoding="utf-8")
    python_bin.chmod(0o755)

    pid_path = repo_root / "outputs" / module.DETACHED_PID_DIRNAME / "job.pid"
    pid_path.parent.mkdir(parents=True, exist_ok=True)
    pid_path.write_text("12345\n", encoding="utf-8")

    monkeypatch.setattr(module, "_find_repo_root", lambda explicit: repo_root)
    monkeypatch.setattr(module, "_select_python", lambda rr: str(python_bin))
    monkeypatch.setattr(module, "_build_collect_env", lambda args: {})
    monkeypatch.setattr(module, "_default_detached_paths", lambda rr, args: (repo_root / "outputs" / module.DETACHED_LOG_DIRNAME / "job.log", pid_path))
    monkeypatch.setattr(module, "_is_pid_alive", lambda pid: True)

    try:
        module._spawn_detached_process(
            Namespace(
                repo_root=str(repo_root),
                detached_log_file=None,
                detached_pid_file=None,
                detached_replace_existing=False,
                command="collect-manifest",
                manifest_path="outputs/manifests/libero_standard_4_2trials.yaml",
                output_root="outputs/libero_spatial_10x2_overnight_v2",
            ),
            ["collect-manifest", "--manifest-path", "outputs/manifests/libero_standard_4_2trials.yaml", "--detached"],
        )
    except RuntimeError as exc:
        assert "already appears to be running" in str(exc)
    else:
        raise AssertionError("Expected RuntimeError for existing live detached pid")


def test_collect_manifest_restarts_incomplete_task_until_complete(tmp_path, monkeypatch):
    module = _load_pipeline_module()
    repo_root = tmp_path
    cfg_dir = repo_root / "env_configs" / "libero"
    cfg_dir.mkdir(parents=True)
    base_cfg = cfg_dir / "base.yaml"
    base_cfg.write_text(
        """
env:
  cfg:
    low_level:
      suite_name: libero_spatial
      task_id: 0
""",
        encoding="utf-8",
    )
    manifest_path = repo_root / "manifest.yaml"
    manifest_path.write_text(
        yaml.safe_dump(
            {
                "version": 1,
                "benchmark": "libero_standard_4",
                "default_trials_per_task": 2,
                "suites": [
                    {
                        "suite_name": "libero_spatial",
                        "task_ids": [0],
                        "trials_per_task": 2,
                        "base_config_path": str(base_cfg.relative_to(repo_root)),
                    }
                ],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(module, "_find_repo_root", lambda explicit: repo_root)
    monkeypatch.setattr(module, "_select_python", lambda rr: str(repo_root / ".venv-libero" / "bin" / "python"))

    statuses = iter(
        [
            {
                "entries": [],
                "completed_trial_ids": [],
                "successful_trial_ids": [],
                "num_completed_trials": 0,
                "num_successful_trials": 0,
                "effective_output_dir": str((repo_root / "outputs" / "run" / "libero_spatial" / "task_00").resolve()),
                "expected_trials": 2,
                "done_flag_exists": False,
                "complete": False,
                "next_resume_idx": 1,
            },
            {
                "entries": [],
                "completed_trial_ids": [1],
                "successful_trial_ids": [1],
                "num_completed_trials": 1,
                "num_successful_trials": 1,
                "effective_output_dir": str((repo_root / "outputs" / "run" / "libero_spatial" / "task_00").resolve()),
                "expected_trials": 2,
                "done_flag_exists": False,
                "complete": False,
                "next_resume_idx": 2,
            },
            {
                "entries": [],
                "completed_trial_ids": [1],
                "successful_trial_ids": [1],
                "num_completed_trials": 1,
                "num_successful_trials": 1,
                "effective_output_dir": str((repo_root / "outputs" / "run" / "libero_spatial" / "task_00").resolve()),
                "expected_trials": 2,
                "done_flag_exists": False,
                "complete": False,
                "next_resume_idx": 2,
            },
            {
                "entries": [],
                "completed_trial_ids": [1, 2],
                "successful_trial_ids": [1, 2],
                "num_completed_trials": 2,
                "num_successful_trials": 2,
                "effective_output_dir": str((repo_root / "outputs" / "run" / "libero_spatial" / "task_00").resolve()),
                "expected_trials": 2,
                "done_flag_exists": True,
                "complete": True,
                "next_resume_idx": None,
            },
        ]
    )
    monkeypatch.setattr(module, "_summarize_manifest_job", lambda output_dir, expected_trials: next(statuses))

    launched: list[dict[str, object]] = []

    def _fake_launch_manifest_task_job(**kwargs):
        launched.append(kwargs)
        return {
            "detached_pid": str(1000 + len(launched)),
            "detached_pid_file": str(repo_root / "pid"),
            "detached_log_file": str(repo_root / "log"),
        }

    monkeypatch.setattr(module, "_launch_manifest_task_job", _fake_launch_manifest_task_job)
    monkeypatch.setattr(module, "_wait_for_pid_exit", lambda pid, poll_interval_s=5.0: None)

    args = Namespace(
        repo_root=str(repo_root),
        manifest_path=str(manifest_path.relative_to(repo_root)),
        output_root="outputs/run",
        suite_filter=[],
        trials_per_task=None,
        task_max_restarts=4,
        record_video=False,
        api_bash_profile=None,
        api_bash_file=None,
        server_url=None,
        api_key=None,
        model=None,
        num_workers=1,
        enable_eap_rollback=False,
        enable_eap_recovery=False,
        enable_eap_model_snapshot_selection=False,
        use_img_differencing=False,
        request_timeout_s=None,
        request_max_attempts=None,
        request_max_retry_walltime_s=None,
        request_retry_initial_s=None,
        request_retry_max_sleep_s=None,
        detached=False,
        detached_log_file=None,
        detached_pid_file=None,
        detached_replace_existing=False,
        detached_child=False,
        tmux_session=None,
        tmux_log_file=None,
        tmux_replace_existing=False,
        tmux_child=False,
        extra_arg=[],
    )

    result = module._collect_manifest(args)

    assert len(launched) == 2
    assert result["jobs"][0]["attempts"][0]["resume_idx"] == 1
    assert result["jobs"][0]["attempts"][1]["resume_idx"] == 2
    assert result["jobs"][0]["final_status"]["complete"] is True


def test_collect_manifest_detached_child_spawns_successor_for_remaining_jobs(tmp_path, monkeypatch):
    module = _load_pipeline_module()
    repo_root = tmp_path
    cfg_dir = repo_root / "env_configs" / "libero"
    cfg_dir.mkdir(parents=True)
    base_cfg = cfg_dir / "base.yaml"
    base_cfg.write_text(
        """
env:
  cfg:
    low_level:
      suite_name: libero_spatial
      task_id: 0
""",
        encoding="utf-8",
    )
    manifest_path = repo_root / "manifest.yaml"
    manifest_path.write_text(
        yaml.safe_dump(
            {
                "version": 1,
                "benchmark": "libero_standard_4",
                "default_trials_per_task": 2,
                "suites": [
                    {
                        "suite_name": "libero_spatial",
                        "task_ids": [0, 1],
                        "trials_per_task": 2,
                        "base_config_path": str(base_cfg.relative_to(repo_root)),
                    }
                ],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(module, "_find_repo_root", lambda explicit: repo_root)
    monkeypatch.setattr(module, "_select_python", lambda rr: str(repo_root / ".venv-libero" / "bin" / "python"))
    monkeypatch.setattr(module, "_write_manifest_run_summary", lambda output_root, run_summary: output_root / "manifest_run_summary.json")
    monkeypatch.setattr(module, "_wait_for_pid_exit", lambda pid, poll_interval_s=5.0: None)

    statuses: dict[tuple[int, int], list[dict[str, object]]] = {
        (0, 2): [
            {
                "entries": [],
                "completed_trial_ids": [],
                "successful_trial_ids": [],
                "num_completed_trials": 0,
                "num_successful_trials": 0,
                "effective_output_dir": str((repo_root / "outputs" / "run" / "libero_spatial" / "task_00").resolve()),
                "expected_trials": 2,
                "done_flag_exists": False,
                "complete": False,
                "next_resume_idx": 1,
            },
            {
                "entries": [],
                "completed_trial_ids": [1, 2],
                "successful_trial_ids": [1, 2],
                "num_completed_trials": 2,
                "num_successful_trials": 2,
                "effective_output_dir": str((repo_root / "outputs" / "run" / "libero_spatial" / "task_00").resolve()),
                "expected_trials": 2,
                "done_flag_exists": True,
                "complete": True,
                "next_resume_idx": None,
            },
            {
                "entries": [],
                "completed_trial_ids": [1, 2],
                "successful_trial_ids": [1, 2],
                "num_completed_trials": 2,
                "num_successful_trials": 2,
                "effective_output_dir": str((repo_root / "outputs" / "run" / "libero_spatial" / "task_00").resolve()),
                "expected_trials": 2,
                "done_flag_exists": True,
                "complete": True,
                "next_resume_idx": None,
            },
        ],
        (1, 2): [
            {
                "entries": [],
                "completed_trial_ids": [],
                "successful_trial_ids": [],
                "num_completed_trials": 0,
                "num_successful_trials": 0,
                "effective_output_dir": str((repo_root / "outputs" / "run" / "libero_spatial" / "task_01").resolve()),
                "expected_trials": 2,
                "done_flag_exists": False,
                "complete": False,
                "next_resume_idx": 1,
            },
        ],
    }

    def _fake_summarize_manifest_job(output_dir, expected_trials):
        task_id = int(str(output_dir).split("task_")[-1])
        key = (task_id, expected_trials)
        queue = statuses[key]
        if len(queue) > 1:
            return queue.pop(0)
        return queue[0]

    monkeypatch.setattr(module, "_summarize_manifest_job", _fake_summarize_manifest_job)

    launched: list[dict[str, object]] = []

    def _fake_launch_manifest_task_job(**kwargs):
        launched.append(kwargs)
        return {
            "detached_pid": str(2000 + len(launched)),
            "detached_pid_file": str(repo_root / "task.pid"),
            "detached_log_file": str(repo_root / "task.log"),
        }

    successor: list[dict[str, str]] = []

    def _fake_spawn_next_manifest_orchestrator(**kwargs):
        successor.append({"detached_pid": "3001"})
        return {"detached_pid": "3001", "detached_pid_file": str(repo_root / "manifest.pid"), "detached_log_file": str(repo_root / "manifest.log")}

    monkeypatch.setattr(module, "_launch_manifest_task_job", _fake_launch_manifest_task_job)
    monkeypatch.setattr(module, "_spawn_next_manifest_orchestrator", _fake_spawn_next_manifest_orchestrator)

    args = Namespace(
        repo_root=str(repo_root),
        manifest_path=str(manifest_path.relative_to(repo_root)),
        output_root="outputs/run",
        suite_filter=[],
        trials_per_task=None,
        task_max_restarts=4,
        record_video=False,
        api_bash_profile=None,
        api_bash_file=None,
        server_url=None,
        api_key=None,
        model=None,
        num_workers=1,
        enable_eap_rollback=False,
        enable_eap_recovery=False,
        enable_eap_model_snapshot_selection=False,
        use_img_differencing=False,
        request_timeout_s=None,
        request_max_attempts=None,
        request_max_retry_walltime_s=None,
        request_retry_initial_s=None,
        request_retry_max_sleep_s=None,
        detached=False,
        detached_log_file=None,
        detached_pid_file=None,
        detached_replace_existing=False,
        detached_child=True,
        tmux_session=None,
        tmux_log_file=None,
        tmux_replace_existing=False,
        tmux_child=False,
        extra_arg=[],
        command="collect-manifest",
    )

    result = module._collect_manifest(args)

    assert len(launched) == 0
    assert len(successor) == 1
    assert result["remaining_jobs"] == [
        {
            "suite_name": "libero_spatial",
            "task_id": 1,
            "next_resume_idx": 1,
        }
    ]
    assert result["next_detached_orchestrator"]["detached_pid"] == "3001"
