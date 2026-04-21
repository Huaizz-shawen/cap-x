#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import yaml


DEFAULT_MODEL_REQUEST_TIMEOUT_S = 240.0
DEFAULT_MODEL_RETRY_MAX_ATTEMPTS = 12
DEFAULT_MODEL_RETRY_MAX_WALLTIME_S = 7200.0
DEFAULT_MODEL_RETRY_INITIAL_S = 15.0
DEFAULT_MODEL_RETRY_MAX_SLEEP_S = 240.0
DEFAULT_API_PORTS = (8114, 8115, 8116)
DEFAULT_HF_CACHE_ROOT = Path("/tmp/capx_hf_cache")
DEFAULT_LIBERO_TASK_IDS = list(range(10))
DEFAULT_LIBERO_TRIALS_PER_TASK = 2
DETACHED_LOG_DIRNAME = "detached_logs"
DETACHED_PID_DIRNAME = "detached_pids"
MANIFEST_TASK_LOG_DIRNAME = "_task_logs"
MANIFEST_TASK_PID_DIRNAME = "_task_pids"
DEFAULT_MANIFEST_TASK_MAX_RESTARTS = 4


def _builtin_manifests() -> dict[str, dict[str, object]]:
    suites = [
        {
            "suite_name": "libero_10",
            "num_tasks": 10,
            "task_ids": DEFAULT_LIBERO_TASK_IDS,
            "base_config_path": "env_configs/libero/franka_libero.yaml",
            "privileged_base_config_path": "env_configs/libero/franka_libero_10_0_privileged.yaml",
        },
        {
            "suite_name": "libero_object",
            "num_tasks": 10,
            "task_ids": DEFAULT_LIBERO_TASK_IDS,
            "base_config_path": "env_configs/libero/franka_libero_object_0.yaml",
            "privileged_base_config_path": "env_configs/libero/franka_libero_object_0_privileged.yaml",
        },
        {
            "suite_name": "libero_spatial",
            "num_tasks": 10,
            "task_ids": DEFAULT_LIBERO_TASK_IDS,
            "base_config_path": "env_configs/libero/franka_libero_spatial_0.yaml",
            "privileged_base_config_path": "env_configs/libero/franka_libero_spatial_0_privileged.yaml",
        },
        {
            "suite_name": "libero_goal",
            "num_tasks": 10,
            "task_ids": DEFAULT_LIBERO_TASK_IDS,
            "base_config_path": "env_configs/libero/franka_libero_goal_1.yaml",
            "privileged_base_config_path": "env_configs/libero/franka_libero_goal_1_privileged.yaml",
        },
    ]
    return {
        "libero_standard_4": {
            "version": 1,
            "benchmark": "libero_standard_4",
            "description": "Standard LIBERO four-suite benchmark manifest for outer-agent orchestration.",
            "robot_type": "franka",
            "default_trials_per_task": DEFAULT_LIBERO_TRIALS_PER_TASK,
            "suites": suites,
        }
    }


def _find_repo_root(explicit: str | None) -> Path:
    if explicit is not None:
        return Path(explicit).expanduser().resolve()

    cwd = Path.cwd().resolve()
    for candidate in [cwd, *cwd.parents]:
        if (candidate / "capx").exists() and (candidate / "env_configs").exists():
            return candidate

    script_dir = Path(__file__).resolve().parent
    for candidate in script_dir.parents:
        if (candidate / "capx").exists() and (candidate / "env_configs").exists():
            return candidate

    raise FileNotFoundError("Could not infer cap-x repo root. Pass --repo-root explicitly.")


def _select_python(repo_root: Path) -> str:
    for rel in [".venv-libero/bin/python", ".venv/bin/python"]:
        candidate = repo_root / rel
        if candidate.exists():
            return str(candidate)
    return sys.executable


def _run(cmd: list[str], cwd: Path, env: dict[str, str] | None = None) -> None:
    print("$", " ".join(cmd))
    subprocess.run(cmd, cwd=cwd, env=env, check=True)


def _resolve_path(repo_root: Path, path_str: str) -> Path:
    path = Path(path_str).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (repo_root / path).resolve()


def _normalize_output_path(repo_root: Path, path_str: str) -> Path:
    return _resolve_path(repo_root, path_str)


def _load_api_ports(repo_root: Path, config_path: str) -> list[int]:
    resolved_config = _resolve_path(repo_root, config_path)
    try:
        cfg = yaml.safe_load(resolved_config.read_text()) or {}
    except Exception:
        return list(DEFAULT_API_PORTS)

    ports: list[int] = []
    for api_cfg in cfg.get("api_servers", []) or []:
        port = api_cfg.get("port")
        if port is None:
            continue
        try:
            ports.append(int(port))
        except (TypeError, ValueError):
            continue

    if not ports:
        return list(DEFAULT_API_PORTS)
    return sorted(set(ports))


def _list_listening_pids(ports: list[int]) -> dict[int, set[int]]:
    if not ports or shutil.which("ss") is None:
        return {}

    try:
        result = subprocess.run(
            ["ss", "-ltnp"],
            capture_output=True,
            text=True,
            check=True,
        )
    except subprocess.CalledProcessError:
        return {}

    port_to_pids = {port: set() for port in ports}
    for line in result.stdout.splitlines():
        for port in ports:
            if f":{port} " not in line:
                continue
            tail = line.split("users:(", 1)[-1]
            for chunk in tail.split("pid=")[1:]:
                digits = []
                for char in chunk:
                    if char.isdigit():
                        digits.append(char)
                    else:
                        break
                if digits:
                    port_to_pids[port].add(int("".join(digits)))

    return {port: pids for port, pids in port_to_pids.items() if pids}


def _wait_for_ports_to_clear(ports: list[int], timeout_s: float) -> bool:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if not _list_listening_pids(ports):
            return True
        time.sleep(0.2)
    return not _list_listening_pids(ports)


def _cleanup_port_listeners(ports: list[int], *, grace_s: float = 5.0) -> dict[str, object]:
    initial = _list_listening_pids(ports)
    initial_pids = sorted({pid for pids in initial.values() for pid in pids})
    if not initial_pids:
        return {"ports": ports, "initial_pids": [], "killed_pids": [], "remaining_pids": []}

    for pid in initial_pids:
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass

    _wait_for_ports_to_clear(ports, timeout_s=grace_s)
    remaining = _list_listening_pids(ports)
    remaining_pids = sorted({pid for pids in remaining.values() for pid in pids})
    for pid in remaining_pids:
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass

    final_remaining = _list_listening_pids(ports)
    final_remaining_pids = sorted({pid for pids in final_remaining.values() for pid in pids})
    killed_pids = sorted(set(initial_pids) - set(final_remaining_pids))
    return {
        "ports": ports,
        "initial_pids": initial_pids,
        "killed_pids": killed_pids,
        "remaining_pids": final_remaining_pids,
    }


def _build_collect_env(args: argparse.Namespace) -> dict[str, str]:
    env = os.environ.copy()
    env["CAPX_MODEL_REQUEST_TIMEOUT_S"] = str(
        args.request_timeout_s or DEFAULT_MODEL_REQUEST_TIMEOUT_S
    )
    env["CAPX_MODEL_RETRY_MAX_ATTEMPTS"] = str(
        args.request_max_attempts or DEFAULT_MODEL_RETRY_MAX_ATTEMPTS
    )
    env["CAPX_MODEL_RETRY_MAX_WALLTIME_S"] = str(
        args.request_max_retry_walltime_s or DEFAULT_MODEL_RETRY_MAX_WALLTIME_S
    )
    env["CAPX_MODEL_RETRY_INITIAL_S"] = str(
        args.request_retry_initial_s or DEFAULT_MODEL_RETRY_INITIAL_S
    )
    env["CAPX_MODEL_RETRY_MAX_SLEEP_S"] = str(
        args.request_retry_max_sleep_s or DEFAULT_MODEL_RETRY_MAX_SLEEP_S
    )
    return env


def _build_validate_env(cache_root: Path | None = None) -> dict[str, str]:
    env = os.environ.copy()
    resolved_root = (cache_root or DEFAULT_HF_CACHE_ROOT).expanduser().resolve()
    hf_home = resolved_root / "huggingface"
    datasets_cache = hf_home / "datasets"
    hub_cache = hf_home / "hub"
    for path in [resolved_root, hf_home, datasets_cache, hub_cache]:
        path.mkdir(parents=True, exist_ok=True)
    env["HF_HOME"] = str(hf_home)
    env["HF_DATASETS_CACHE"] = str(datasets_cache)
    env["HUGGINGFACE_HUB_CACHE"] = str(hub_cache)
    return env


def _materialize_manifest_preset(name: str, *, trials_per_task: int | None = None) -> dict[str, object]:
    manifests = _builtin_manifests()
    if name not in manifests:
        raise KeyError(f"Unknown manifest preset: {name}")
    manifest = json.loads(json.dumps(manifests[name]))
    resolved_trials = trials_per_task or int(manifest["default_trials_per_task"])
    manifest["default_trials_per_task"] = resolved_trials
    for suite in manifest.get("suites", []):
        suite["trials_per_task"] = resolved_trials
    return manifest


def _load_manifest(path: Path) -> dict[str, object]:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Manifest must be a mapping: {path}")
    if "suites" not in data or not isinstance(data["suites"], list):
        raise ValueError(f"Manifest must contain a 'suites' list: {path}")
    return data


def _write_yaml(path: Path, data: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(data, sort_keys=False, allow_unicode=False),
        encoding="utf-8",
    )


def _write_json(path: Path, data: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _iter_manifest_jobs(
    manifest: dict[str, object],
    *,
    suite_filters: list[str] | None = None,
    trials_per_task_override: int | None = None,
) -> list[dict[str, object]]:
    selected_suites = set(suite_filters or [])
    default_trials = int(manifest.get("default_trials_per_task", DEFAULT_LIBERO_TRIALS_PER_TASK))
    jobs: list[dict[str, object]] = []
    for suite in manifest.get("suites", []):
        if not isinstance(suite, dict):
            continue
        suite_name = str(suite["suite_name"])
        if selected_suites and suite_name not in selected_suites:
            continue
        task_ids = [int(task_id) for task_id in suite.get("task_ids", DEFAULT_LIBERO_TASK_IDS)]
        trials_per_task = (
            trials_per_task_override
            if trials_per_task_override is not None
            else int(suite.get("trials_per_task", default_trials))
        )
        for task_id in task_ids:
            jobs.append(
                {
                    "suite_name": suite_name,
                    "task_id": task_id,
                    "trials_per_task": trials_per_task,
                    "base_config_path": str(suite["base_config_path"]),
                    "privileged_base_config_path": suite.get("privileged_base_config_path"),
                }
            )
    return jobs


def _make_manifest_job_config(
    *,
    repo_root: Path,
    base_config_path: str,
    suite_name: str,
    task_id: int,
    scratch_dir: Path,
    record_video: bool | None = None,
    resume_idx: int | None = None,
) -> Path:
    base_config = yaml.safe_load(
        _resolve_path(repo_root, base_config_path).read_text(encoding="utf-8")
    )
    low_level = base_config["env"]["cfg"]["low_level"]
    low_level["suite_name"] = suite_name
    low_level["task_id"] = int(task_id)
    if record_video is not None:
        base_config["record_video"] = bool(record_video)
    if resume_idx is not None:
        base_config["resume_idx"] = int(resume_idx)
    else:
        base_config.pop("resume_idx", None)
    output_path = scratch_dir / f"{suite_name}_task_{task_id:02d}.yaml"
    _write_yaml(output_path, base_config)
    return output_path


def _resolve_effective_collection_output_dir(requested_output_dir: Path) -> Path:
    requested_output_dir = requested_output_dir.resolve()
    if requested_output_dir.exists():
        return requested_output_dir

    parent = requested_output_dir.parent
    leaf = requested_output_dir.name
    candidates = sorted(parent.glob(f"*/{leaf}"))
    if len(candidates) == 1:
        return candidates[0].resolve()

    for candidate in candidates:
        if any(candidate.glob("trial_*/*/trial_metadata.json")):
            return candidate.resolve()

    return requested_output_dir


def _collect_trial_metadata(output_dir: Path) -> list[dict[str, object]]:
    effective_dir = _resolve_effective_collection_output_dir(output_dir)
    metadata_entries: list[dict[str, object]] = []
    for path in sorted(effective_dir.glob("trial_*/attempt_*/trial_metadata.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(data, dict):
            continue
        data["_path"] = str(path)
        metadata_entries.append(data)
    return metadata_entries


def _summarize_completed_trials(output_dir: Path) -> dict[str, object]:
    entries = _collect_trial_metadata(output_dir)
    completed_trials: dict[int, list[dict[str, object]]] = {}
    successful_trials: set[int] = set()
    for entry in entries:
        try:
            trial_id = int(entry.get("trial"))
        except Exception:
            continue
        completed_trials.setdefault(trial_id, []).append(entry)
        if bool(entry.get("task_completed", False)):
            successful_trials.add(trial_id)
    return {
        "entries": entries,
        "completed_trial_ids": sorted(completed_trials.keys()),
        "successful_trial_ids": sorted(successful_trials),
        "num_completed_trials": len(completed_trials),
        "num_successful_trials": len(successful_trials),
        "effective_output_dir": str(_resolve_effective_collection_output_dir(output_dir)),
    }


def _task_done_flag_exists(output_dir: Path) -> bool:
    effective_dir = _resolve_effective_collection_output_dir(output_dir)
    return (effective_dir / "aaa_done_flag" / "aaa_done_flag.txt").exists()


def _next_missing_trial_id(expected_trials: int, completed_trial_ids: Iterable[int]) -> int | None:
    completed = {int(trial_id) for trial_id in completed_trial_ids}
    for trial_id in range(1, expected_trials + 1):
        if trial_id not in completed:
            return trial_id
    return None


def _summarize_manifest_job(output_dir: Path, expected_trials: int) -> dict[str, object]:
    completion = _summarize_completed_trials(output_dir)
    completed_trial_ids = [int(trial_id) for trial_id in completion["completed_trial_ids"]]
    next_resume_idx = _next_missing_trial_id(expected_trials, completed_trial_ids)
    complete = _task_done_flag_exists(output_dir) or next_resume_idx is None
    completion.update(
        {
            "expected_trials": int(expected_trials),
            "done_flag_exists": _task_done_flag_exists(output_dir),
            "complete": bool(complete),
            "next_resume_idx": next_resume_idx,
        }
    )
    return completion


def _strip_tmux_flags(argv: list[str]) -> list[str]:
    filtered: list[str] = []
    skip_next = False
    flags_with_values = {
        "--tmux-session",
        "--tmux-log-file",
    }
    flags_without_values = {
        "--tmux-replace-existing",
        "--tmux-child",
    }
    for token in argv:
        if skip_next:
            skip_next = False
            continue
        if token in flags_with_values:
            skip_next = True
            continue
        if any(token.startswith(f"{flag}=") for flag in flags_with_values):
            continue
        if token in flags_without_values:
            continue
        filtered.append(token)
    return filtered


def _strip_detached_flags(argv: list[str]) -> list[str]:
    filtered: list[str] = []
    skip_next = False
    flags_with_values = {
        "--detached-log-file",
        "--detached-pid-file",
    }
    flags_without_values = {
        "--detached",
        "--detached-replace-existing",
        "--detached-child",
    }
    for token in argv:
        if skip_next:
            skip_next = False
            continue
        if token in flags_with_values:
            skip_next = True
            continue
        if any(token.startswith(f"{flag}=") for flag in flags_with_values):
            continue
        if token in flags_without_values:
            continue
        filtered.append(token)
    return filtered


def _is_pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _kill_pid(pid: int, *, grace_s: float = 5.0) -> bool:
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        return True

    deadline = time.time() + grace_s
    while time.time() < deadline:
        if not _is_pid_alive(pid):
            return True
        time.sleep(0.2)

    try:
        os.kill(pid, signal.SIGKILL)
    except ProcessLookupError:
        return True
    time.sleep(0.2)
    return not _is_pid_alive(pid)


def _read_pid_file(pid_path: Path) -> int | None:
    if not pid_path.exists():
        return None
    try:
        return int(pid_path.read_text(encoding="utf-8").strip())
    except ValueError:
        return None


def _derive_detached_name(args: argparse.Namespace) -> str:
    if getattr(args, "command", None) == "collect-manifest":
        base = Path(args.manifest_path).stem
    elif getattr(args, "output_dir", None):
        base = Path(args.output_dir).name
    elif getattr(args, "output_root", None):
        base = Path(args.output_root).name
    elif getattr(args, "config_path", None):
        base = Path(args.config_path).stem
    else:
        base = getattr(args, "command", "capx-job")
    cleaned = "".join(char if char.isalnum() or char in {"-", "_"} else "-" for char in base)
    return cleaned or "capx-job"


def _default_detached_paths(repo_root: Path, args: argparse.Namespace) -> tuple[Path, Path]:
    name = _derive_detached_name(args)
    log_path = (repo_root / "outputs" / DETACHED_LOG_DIRNAME / f"{name}.log").resolve()
    pid_path = (repo_root / "outputs" / DETACHED_PID_DIRNAME / f"{name}.pid").resolve()
    return log_path, pid_path


def _spawn_detached_child_command(
    *,
    cwd: Path,
    env: dict[str, str],
    child_cmd: list[str],
    log_path: Path,
    pid_path: Path,
    replace_existing: bool = False,
) -> dict[str, str]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    pid_path.parent.mkdir(parents=True, exist_ok=True)

    existing_pid = None
    if pid_path.exists():
        try:
            existing_pid = int(pid_path.read_text(encoding="utf-8").strip())
        except ValueError:
            pid_path.unlink(missing_ok=True)
        else:
            if _is_pid_alive(existing_pid):
                if replace_existing:
                    _kill_pid(existing_pid)
                else:
                    raise RuntimeError(
                        f"Detached process '{pid_path.name}' already appears to be running "
                        f"(pid={existing_pid}). Pass --detached-replace-existing to replace it."
                    )
            pid_path.unlink(missing_ok=True)

    with log_path.open("ab") as log_handle, open(os.devnull, "rb") as devnull:
        process = subprocess.Popen(
            child_cmd,
            cwd=cwd,
            env=env,
            stdin=devnull,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            close_fds=True,
        )

    pid_path.write_text(f"{process.pid}\n", encoding="utf-8")
    return {
        "detached_pid": str(process.pid),
        "detached_pid_file": str(pid_path),
        "detached_log_file": str(log_path),
    }


def _spawn_detached_process(args: argparse.Namespace, raw_args: list[str]) -> dict[str, str]:
    repo_root = _find_repo_root(args.repo_root)
    python_bin = _select_python(repo_root)
    script_path = Path(__file__).resolve()

    default_log_path, default_pid_path = _default_detached_paths(repo_root, args)
    log_path = (
        _resolve_path(repo_root, args.detached_log_file)
        if args.detached_log_file is not None
        else default_log_path
    )
    pid_path = (
        _resolve_path(repo_root, args.detached_pid_file)
        if args.detached_pid_file is not None
        else default_pid_path
    )
    child_args = _strip_tmux_flags(raw_args)
    child_args = _strip_detached_flags(child_args)
    child_args.append("--detached-child")
    child_cmd = [python_bin, str(script_path), *child_args]
    env = _build_collect_env(args)
    env["CAPX_DETACHED_LOG_FILE"] = str(log_path)
    env["CAPX_DETACHED_PID_FILE"] = str(pid_path)
    result = _spawn_detached_child_command(
        cwd=repo_root,
        env=env,
        child_cmd=child_cmd,
        log_path=log_path,
        pid_path=pid_path,
        replace_existing=bool(args.detached_replace_existing),
    )
    result["repo_root"] = str(repo_root)
    return result


def _manifest_task_name(suite_name: str, task_id: int) -> str:
    return f"{suite_name}-task-{task_id:02d}"


def _manifest_task_detached_paths(output_root: Path, suite_name: str, task_id: int) -> tuple[Path, Path]:
    task_name = _manifest_task_name(suite_name, task_id)
    log_path = (output_root / MANIFEST_TASK_LOG_DIRNAME / f"{task_name}.log").resolve()
    pid_path = (output_root / MANIFEST_TASK_PID_DIRNAME / f"{task_name}.pid").resolve()
    return log_path, pid_path


def _manifest_orchestrator_detached_paths(repo_root: Path, args: argparse.Namespace) -> tuple[Path, Path]:
    env_log_path = os.environ.get("CAPX_DETACHED_LOG_FILE")
    env_pid_path = os.environ.get("CAPX_DETACHED_PID_FILE")
    if env_log_path and env_pid_path:
        return Path(env_log_path).expanduser().resolve(), Path(env_pid_path).expanduser().resolve()
    return _default_detached_paths(repo_root, args)


def _wait_for_pid_exit(pid: int, *, poll_interval_s: float = 5.0) -> None:
    while _is_pid_alive(pid):
        time.sleep(poll_interval_s)


def _launch_manifest_task_job(
    *,
    repo_root: Path,
    parent_args: argparse.Namespace,
    config_path: Path,
    output_dir: Path,
    total_trials: int,
    suite_name: str,
    task_id: int,
    replace_existing: bool = False,
) -> dict[str, str]:
    python_bin = _select_python(repo_root)
    script_path = Path(__file__).resolve()
    log_path, pid_path = _manifest_task_detached_paths(output_dir.parent.parent, suite_name, task_id)
    argv: list[str] = [
        python_bin,
        str(script_path),
        "collect",
        "--repo-root",
        str(repo_root),
        "--config-path",
        str(config_path),
        "--output-dir",
        str(output_dir),
        "--total-trials",
        str(total_trials),
    ]
    if parent_args.api_bash_profile is not None:
        argv.extend(["--api-bash-profile", parent_args.api_bash_profile])
    if parent_args.api_bash_file is not None:
        argv.extend(["--api-bash-file", parent_args.api_bash_file])
    if parent_args.server_url is not None:
        argv.extend(["--server-url", str(parent_args.server_url)])
    if parent_args.api_key is not None:
        argv.extend(["--api-key", str(parent_args.api_key)])
    if parent_args.model is not None:
        argv.extend(["--model", str(parent_args.model)])
    if parent_args.num_workers is not None:
        argv.extend(["--num-workers", str(parent_args.num_workers)])
    if parent_args.record_video:
        argv.append("--record-video")
    if parent_args.enable_eap_rollback:
        argv.append("--enable-eap-rollback")
    if parent_args.enable_eap_recovery:
        argv.append("--enable-eap-recovery")
    if parent_args.enable_eap_model_snapshot_selection:
        argv.append("--enable-eap-model-snapshot-selection")
    if parent_args.use_img_differencing:
        argv.append("--use-img-differencing")
    if parent_args.request_timeout_s is not None:
        argv.extend(["--request-timeout-s", str(parent_args.request_timeout_s)])
    if parent_args.request_max_attempts is not None:
        argv.extend(["--request-max-attempts", str(parent_args.request_max_attempts)])
    if parent_args.request_max_retry_walltime_s is not None:
        argv.extend(["--request-max-retry-walltime-s", str(parent_args.request_max_retry_walltime_s)])
    if parent_args.request_retry_initial_s is not None:
        argv.extend(["--request-retry-initial-s", str(parent_args.request_retry_initial_s)])
    if parent_args.request_retry_max_sleep_s is not None:
        argv.extend(["--request-retry-max-sleep-s", str(parent_args.request_retry_max_sleep_s)])
    for extra in parent_args.extra_arg:
        argv.extend(["--extra-arg", extra[0], extra[1]])
    argv.append("--detached-child")

    env = _build_collect_env(parent_args)
    result = _spawn_detached_child_command(
        cwd=repo_root,
        env=env,
        child_cmd=argv,
        log_path=log_path,
        pid_path=pid_path,
        replace_existing=replace_existing,
    )
    result.update(
        {
            "suite_name": suite_name,
            "task_id": str(task_id),
            "output_dir": str(output_dir),
            "config_path": str(config_path),
        }
    )
    return result


def _spawn_next_manifest_orchestrator(
    *,
    repo_root: Path,
    args: argparse.Namespace,
) -> dict[str, str]:
    python_bin = _select_python(repo_root)
    script_path = Path(__file__).resolve()
    log_path, pid_path = _manifest_orchestrator_detached_paths(repo_root, args)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    pid_path.parent.mkdir(parents=True, exist_ok=True)

    existing_pid = _read_pid_file(pid_path)
    current_pid = os.getpid()
    if existing_pid is not None and existing_pid != current_pid and _is_pid_alive(existing_pid):
        raise RuntimeError(
            f"Detached manifest orchestrator already appears to be running (pid={existing_pid})."
        )

    argv: list[str] = [
        python_bin,
        str(script_path),
        "collect-manifest",
        "--repo-root",
        str(repo_root),
        "--manifest-path",
        args.manifest_path,
        "--output-root",
        args.output_root,
        "--task-max-restarts",
        str(args.task_max_restarts),
        "--detached-child",
    ]
    for suite_name in args.suite_filter:
        argv.extend(["--suite-filter", suite_name])
    if args.trials_per_task is not None:
        argv.extend(["--trials-per-task", str(args.trials_per_task)])
    if args.api_bash_profile is not None:
        argv.extend(["--api-bash-profile", args.api_bash_profile])
    if args.api_bash_file is not None:
        argv.extend(["--api-bash-file", args.api_bash_file])
    if args.server_url is not None:
        argv.extend(["--server-url", str(args.server_url)])
    if args.api_key is not None:
        argv.extend(["--api-key", str(args.api_key)])
    if args.model is not None:
        argv.extend(["--model", str(args.model)])
    if args.num_workers is not None:
        argv.extend(["--num-workers", str(args.num_workers)])
    if args.record_video:
        argv.append("--record-video")
    if args.enable_eap_rollback:
        argv.append("--enable-eap-rollback")
    if args.enable_eap_recovery:
        argv.append("--enable-eap-recovery")
    if args.enable_eap_model_snapshot_selection:
        argv.append("--enable-eap-model-snapshot-selection")
    if args.use_img_differencing:
        argv.append("--use-img-differencing")
    if args.request_timeout_s is not None:
        argv.extend(["--request-timeout-s", str(args.request_timeout_s)])
    if args.request_max_attempts is not None:
        argv.extend(["--request-max-attempts", str(args.request_max_attempts)])
    if args.request_max_retry_walltime_s is not None:
        argv.extend(["--request-max-retry-walltime-s", str(args.request_max_retry_walltime_s)])
    if args.request_retry_initial_s is not None:
        argv.extend(["--request-retry-initial-s", str(args.request_retry_initial_s)])
    if args.request_retry_max_sleep_s is not None:
        argv.extend(["--request-retry-max-sleep-s", str(args.request_retry_max_sleep_s)])
    for extra in args.extra_arg:
        argv.extend(["--extra-arg", extra[0], extra[1]])

    env = _build_collect_env(args)
    env["CAPX_DETACHED_LOG_FILE"] = str(log_path)
    env["CAPX_DETACHED_PID_FILE"] = str(pid_path)

    with log_path.open("ab") as log_handle, open(os.devnull, "rb") as devnull:
        process = subprocess.Popen(
            argv,
            cwd=repo_root,
            env=env,
            stdin=devnull,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            close_fds=True,
        )
    pid_path.write_text(f"{process.pid}\n", encoding="utf-8")
    return {
        "detached_pid": str(process.pid),
        "detached_pid_file": str(pid_path),
        "detached_log_file": str(log_path),
    }


def _write_manifest_run_summary(output_root: Path, run_summary: dict[str, object]) -> Path:
    summary_path = output_root / "manifest_run_summary.json"
    _write_json(summary_path, run_summary)
    return summary_path


def _tmux_has_session(session_name: str) -> bool:
    result = subprocess.run(
        ["tmux", "has-session", "-t", session_name],
        capture_output=True,
        text=True,
        check=False,
    )
    return result.returncode == 0


def _spawn_tmux_session(args: argparse.Namespace, raw_args: list[str]) -> dict[str, str]:
    if shutil.which("tmux") is None:
        raise RuntimeError("tmux is not installed; cannot launch detached collection session.")

    repo_root = _find_repo_root(args.repo_root)
    python_bin = _select_python(repo_root)
    script_path = Path(__file__).resolve()
    session_name = args.tmux_session
    if session_name is None:
        raise ValueError("--tmux-session is required for tmux launch")

    if _tmux_has_session(session_name):
        if args.tmux_replace_existing:
            subprocess.run(["tmux", "kill-session", "-t", session_name], check=True)
        else:
            raise RuntimeError(
                f"tmux session '{session_name}' already exists. "
                "Pass --tmux-replace-existing to replace it."
            )

    log_path = (
        _resolve_path(repo_root, args.tmux_log_file)
        if args.tmux_log_file is not None
        else (repo_root / "outputs" / "tmux_logs" / f"{session_name}.log").resolve()
    )
    log_path.parent.mkdir(parents=True, exist_ok=True)

    env = _build_collect_env(args)
    env_exports = "\n".join(
        f"export {key}={shlex.quote(value)}"
        for key, value in env.items()
        if key.startswith("CAPX_MODEL_")
    )

    child_args = _strip_tmux_flags(raw_args)
    child_args.append("--tmux-child")
    child_cmd = shlex.join([python_bin, str(script_path), *child_args])
    cleanup_cmd = None
    config_path = getattr(args, "config_path", None)
    if config_path is not None:
        cleanup_cmd = shlex.join(
            [
                python_bin,
                str(script_path),
                "cleanup-services",
                "--repo-root",
                str(repo_root),
                "--config-path",
                config_path,
            ]
        )
    cleanup_body = f"  {cleanup_cmd} >/dev/null 2>&1 || true" if cleanup_cmd is not None else "  true"
    shell_cmd = "\n".join(
        [
            "set -euo pipefail",
            f"cd {shlex.quote(str(repo_root))}",
            env_exports,
            "cleanup() {",
            cleanup_body,
            "}",
            "trap cleanup EXIT INT TERM",
            "cleanup",
            f"{child_cmd} 2>&1 | tee -a {shlex.quote(str(log_path))}",
            "status=${PIPESTATUS[0]}",
            "cleanup",
            "exit ${status}",
        ]
    )
    subprocess.run(
        ["tmux", "new-session", "-d", "-s", session_name, "bash", "-lc", shell_cmd],
        check=True,
    )
    return {
        "repo_root": str(repo_root),
        "tmux_session": session_name,
        "tmux_log_file": str(log_path),
    }


def _cleanup_services(args: argparse.Namespace) -> dict[str, object]:
    repo_root = _find_repo_root(args.repo_root)
    ports = args.port or _load_api_ports(repo_root, args.config_path)
    result = _cleanup_port_listeners(sorted(set(int(port) for port in ports)))
    result["repo_root"] = str(repo_root)
    result["config_path"] = str(_resolve_path(repo_root, args.config_path))
    print(json.dumps(result, indent=2))
    return result


def _make_collect_namespace(
    args: argparse.Namespace,
    *,
    config_path: str,
    output_dir: str,
    total_trials: int | None,
) -> argparse.Namespace:
    return argparse.Namespace(
        repo_root=args.repo_root,
        api_bash_profile=args.api_bash_profile,
        api_bash_file=args.api_bash_file,
        config_path=config_path,
        output_dir=output_dir,
        server_url=args.server_url,
        api_key=args.api_key,
        model=args.model,
        total_trials=total_trials,
        num_workers=args.num_workers,
        record_video=args.record_video,
        enable_eap_rollback=args.enable_eap_rollback,
        enable_eap_recovery=args.enable_eap_recovery,
        enable_eap_model_snapshot_selection=args.enable_eap_model_snapshot_selection,
        use_img_differencing=args.use_img_differencing,
        request_timeout_s=args.request_timeout_s,
        request_max_attempts=args.request_max_attempts,
        request_max_retry_walltime_s=args.request_max_retry_walltime_s,
        request_retry_initial_s=args.request_retry_initial_s,
        request_retry_max_sleep_s=args.request_retry_max_sleep_s,
        tmux_session=None,
        tmux_log_file=None,
        tmux_replace_existing=False,
        tmux_child=True,
        detached=False,
        detached_log_file=None,
        detached_pid_file=None,
        detached_replace_existing=False,
        detached_child=True,
        extra_arg=list(args.extra_arg),
    )


def _collect(args: argparse.Namespace) -> dict[str, str]:
    repo_root = _find_repo_root(args.repo_root)
    python_bin = _select_python(repo_root)
    cmd = [
        python_bin,
        str(repo_root / "capx" / "envs" / "launch.py"),
    ]
    if args.api_bash_profile is not None:
        cmd.extend(["--api-bash-profile", args.api_bash_profile])
    if args.api_bash_file is not None:
        cmd.extend(["--api-bash-file", args.api_bash_file])
    cmd.extend([
        "--config-path",
        args.config_path,
        "--output-dir",
        args.output_dir,
    ])

    optional_pairs = [
        ("--server-url", args.server_url),
        ("--api-key", args.api_key),
        ("--model", args.model),
        ("--total-trials", args.total_trials),
        ("--num-workers", args.num_workers),
    ]
    for key, value in optional_pairs:
        if value is not None:
            cmd.extend([key, str(value)])

    if args.record_video:
        cmd.extend(["--record-video", "True"])
    if args.enable_eap_rollback:
        cmd.extend(["--enable-eap-rollback", "True"])
    if args.enable_eap_recovery:
        cmd.extend(["--enable-eap-recovery", "True"])
    if args.enable_eap_model_snapshot_selection:
        cmd.extend(["--enable-eap-model-snapshot-selection", "True"])
    if args.use_img_differencing:
        cmd.extend(["--use-img-differencing", "True"])

    for extra in args.extra_arg:
        cmd.extend(["--" + extra[0], extra[1]])

    env = _build_collect_env(args)
    _cleanup_services(
        argparse.Namespace(repo_root=str(repo_root), config_path=args.config_path, port=[])
    )
    try:
        _run(cmd, cwd=repo_root, env=env)
    except subprocess.CalledProcessError as exc:
        expected_trials = int(args.total_trials) if args.total_trials is not None else None
        requested_output_dir = _normalize_output_path(repo_root, args.output_dir)
        completion = _summarize_completed_trials(requested_output_dir)
        if (
            exc.returncode < 0
            and expected_trials is not None
            and int(completion["num_completed_trials"]) >= expected_trials
        ):
            print(
                "[collect] launch.py exited after all trial metadata had already been written; "
                "treating collection as complete.\n"
                + json.dumps(
                    {
                        "returncode": exc.returncode,
                        "expected_trials": expected_trials,
                        "num_completed_trials": completion["num_completed_trials"],
                        "num_successful_trials": completion["num_successful_trials"],
                        "effective_output_dir": completion["effective_output_dir"],
                    },
                    indent=2,
                )
            )
        else:
            raise
    finally:
        _cleanup_services(
            argparse.Namespace(repo_root=str(repo_root), config_path=args.config_path, port=[])
        )
    return {
        "repo_root": str(repo_root),
        "collection_output_dir": str(_normalize_output_path(repo_root, args.output_dir)),
    }


def _write_manifest(args: argparse.Namespace) -> dict[str, str]:
    repo_root = _find_repo_root(args.repo_root)
    manifest = _materialize_manifest_preset(
        args.preset,
        trials_per_task=args.trials_per_task,
    )
    output_path = _resolve_path(repo_root, args.output_path)
    _write_yaml(output_path, manifest)
    result = {
        "repo_root": str(repo_root),
        "manifest_path": str(output_path),
        "preset": args.preset,
        "suite_count": str(len(manifest.get("suites", []))),
    }
    print(json.dumps(result, indent=2))
    return result


def _collect_manifest(args: argparse.Namespace) -> dict[str, object]:
    repo_root = _find_repo_root(args.repo_root)
    manifest = _load_manifest(_resolve_path(repo_root, args.manifest_path))
    all_jobs = _iter_manifest_jobs(
        manifest,
        suite_filters=args.suite_filter,
        trials_per_task_override=args.trials_per_task,
    )
    jobs = list(all_jobs)
    output_root = _normalize_output_path(repo_root, args.output_root)
    scratch_dir = Path(tempfile.mkdtemp(prefix="capx_manifest_configs_", dir="/tmp")).resolve()
    run_summary: dict[str, object] = {
        "repo_root": str(repo_root),
        "manifest_path": str(_resolve_path(repo_root, args.manifest_path)),
        "output_root": str(output_root),
        "job_count": len(jobs),
        "suite_filters": list(args.suite_filter),
        "task_max_restarts": int(args.task_max_restarts),
        "jobs": [],
    }
    if getattr(args, "detached_child", False):
        for job in all_jobs:
            job_output_dir = output_root / str(job["suite_name"]) / f"task_{int(job['task_id']):02d}"
            if not bool(_summarize_manifest_job(job_output_dir, int(job["trials_per_task"]))["complete"]):
                jobs = [job]
                break
        else:
            jobs = []
        run_summary["detached_task_step_mode"] = True
    try:
        for job in jobs:
            suite_name = str(job["suite_name"])
            task_id = int(job["task_id"])
            job_output_dir = output_root / suite_name / f"task_{task_id:02d}"
            expected_trials = int(job["trials_per_task"])
            job_record: dict[str, object] = {
                "suite_name": suite_name,
                "task_id": task_id,
                "trials_per_task": expected_trials,
                "base_config_path": str(job["base_config_path"]),
                "collection_output_dir": str(job_output_dir),
                "attempts": [],
            }
            run_summary["jobs"].append(job_record)
            _write_manifest_run_summary(output_root, run_summary)

            for task_restart in range(int(args.task_max_restarts) + 1):
                task_log_path, task_pid_path = _manifest_task_detached_paths(output_root, suite_name, task_id)
                existing_pid = _read_pid_file(task_pid_path)
                if existing_pid is not None and _is_pid_alive(existing_pid):
                    job_record["waiting_on_existing_pid"] = existing_pid
                    _write_manifest_run_summary(output_root, run_summary)
                    _wait_for_pid_exit(existing_pid)

                task_status = _summarize_manifest_job(job_output_dir, expected_trials)
                job_record["status_before_attempt"] = task_status
                if bool(task_status["complete"]):
                    job_record["final_status"] = task_status
                    break

                generated_config_path = _make_manifest_job_config(
                    repo_root=repo_root,
                    base_config_path=str(job["base_config_path"]),
                    suite_name=suite_name,
                    task_id=task_id,
                    scratch_dir=scratch_dir,
                    # For long-running benchmark collection, prefer not to encode MP4s
                    # unless the operator explicitly requests them. This avoids child
                    # process deaths during video-writing after trial metadata has
                    # already been saved.
                    record_video=bool(args.record_video),
                    resume_idx=int(task_status["next_resume_idx"]) if task_status["next_resume_idx"] is not None else None,
                )

                detached_child = _launch_manifest_task_job(
                    repo_root=repo_root,
                    parent_args=args,
                    config_path=generated_config_path,
                    output_dir=job_output_dir,
                    total_trials=expected_trials,
                    suite_name=suite_name,
                    task_id=task_id,
                    replace_existing=bool(args.detached_replace_existing),
                )
                task_attempt_record: dict[str, object] = {
                    "task_restart_index": task_restart + 1,
                    "generated_config_path": str(generated_config_path),
                    "resume_idx": task_status["next_resume_idx"],
                    **detached_child,
                }
                job_record["attempts"].append(task_attempt_record)
                _write_manifest_run_summary(output_root, run_summary)

                _wait_for_pid_exit(int(detached_child["detached_pid"]))

                task_status_after = _summarize_manifest_job(job_output_dir, expected_trials)
                task_attempt_record["status_after_attempt"] = task_status_after
                job_record["final_status"] = task_status_after
                _write_manifest_run_summary(output_root, run_summary)

                if bool(task_status_after["complete"]):
                    break
            else:
                job_record["final_status"] = _summarize_manifest_job(job_output_dir, expected_trials)

            _write_manifest_run_summary(output_root, run_summary)
            if getattr(args, "detached_child", False):
                break
    finally:
        shutil.rmtree(scratch_dir, ignore_errors=True)

    if getattr(args, "detached_child", False):
        remaining_jobs: list[dict[str, object]] = []
        for job in all_jobs:
            suite_name = str(job["suite_name"])
            task_id = int(job["task_id"])
            job_output_dir = output_root / suite_name / f"task_{task_id:02d}"
            status = _summarize_manifest_job(job_output_dir, int(job["trials_per_task"]))
            if not bool(status["complete"]):
                remaining_jobs.append(
                    {
                        "suite_name": suite_name,
                        "task_id": task_id,
                        "next_resume_idx": status["next_resume_idx"],
                    }
                )
        run_summary["remaining_jobs"] = remaining_jobs
        if remaining_jobs:
            run_summary["next_detached_orchestrator"] = _spawn_next_manifest_orchestrator(
                repo_root=repo_root,
                args=args,
            )
        _write_manifest_run_summary(output_root, run_summary)

    summary_path = _write_manifest_run_summary(output_root, run_summary)
    run_summary["summary_path"] = str(summary_path)
    print(json.dumps(run_summary, indent=2))
    return run_summary


def _export(args: argparse.Namespace) -> dict[str, str]:
    repo_root = _find_repo_root(args.repo_root)
    python_bin = _select_python(repo_root)
    cmd = [
        python_bin,
        "-m",
        "capx.data.export_lerobot_video_dataset",
        "--input-root",
        args.input_root,
        "--output-root",
        args.output_root,
        "--robot-type",
        args.robot_type,
        "--crf",
        str(args.crf),
        "--chunk-size",
        str(args.chunk_size),
    ]
    if args.fps is not None:
        cmd.extend(["--fps", str(args.fps)])
    if args.include_excluded:
        cmd.append("--include-excluded")

    _run(cmd, cwd=repo_root)
    return {
        "repo_root": str(repo_root),
        "lerobot_output_dir": str(
            (repo_root / args.output_root).resolve()
            if not Path(args.output_root).is_absolute()
            else Path(args.output_root).resolve()
        ),
    }


def _export_shards(args: argparse.Namespace) -> dict[str, object]:
    """Export each transition dataset into an independent LeRobot shard directory."""
    repo_root = _find_repo_root(args.repo_root)
    python_bin = _select_python(repo_root)
    input_root = (repo_root / args.input_root).resolve() if not Path(args.input_root).is_absolute() else Path(args.input_root).resolve()
    output_root = (repo_root / args.output_root).resolve() if not Path(args.output_root).is_absolute() else Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    transition_paths = sorted(input_root.glob("**/transition_dataset/data.pkl.gz"))
    if not transition_paths:
        raise FileNotFoundError(f"No transition datasets found under {input_root}")

    shard_results: list[dict[str, object]] = []
    exported = 0
    skipped = 0
    cleaned = 0

    for transition_path in transition_paths:
        attempt_dir = transition_path.parent.parent
        rel_attempt_dir = attempt_dir.relative_to(input_root)
        shard_name = "__".join(rel_attempt_dir.parts)
        shard_root = output_root / shard_name
        manifest_path = shard_root / "manifest.json"

        result: dict[str, object] = {
            "transition_path": str(transition_path),
            "attempt_dir": str(attempt_dir),
            "shard_name": shard_name,
            "shard_root": str(shard_root),
        }

        if manifest_path.exists() and not args.reexport_existing:
            result["status"] = "skipped_existing"
            skipped += 1
            shard_results.append(result)
            continue

        cmd = [
            python_bin,
            "-m",
            "capx.data.export_lerobot_video_dataset",
            "--input-root",
            str(attempt_dir),
            "--output-root",
            str(shard_root),
            "--robot-type",
            args.robot_type,
            "--crf",
            str(args.crf),
            "--chunk-size",
            str(args.chunk_size),
        ]
        if args.fps is not None:
            cmd.extend(["--fps", str(args.fps)])
        if args.include_excluded:
            cmd.append("--include-excluded")

        _run(cmd, cwd=repo_root)
        result["status"] = "exported"
        exported += 1

        if args.cleanup_transition_dataset:
            shutil.rmtree(transition_path.parent, ignore_errors=True)
            result["transition_dataset_cleaned"] = True
            cleaned += 1

        shard_results.append(result)

    summary = {
        "repo_root": str(repo_root),
        "input_root": str(input_root),
        "shard_output_root": str(output_root),
        "num_transition_datasets": len(transition_paths),
        "num_exported": exported,
        "num_skipped_existing": skipped,
        "num_transition_datasets_cleaned": cleaned,
        "shards": shard_results,
    }
    print(json.dumps(summary, indent=2))
    return summary


def _validate(args: argparse.Namespace) -> dict[str, object]:
    dataset_root = Path(args.dataset_root).expanduser().resolve()
    validate_env = _build_validate_env(
        Path(args.hf_cache_root) if getattr(args, "hf_cache_root", None) else None
    )
    os.environ.update(validate_env)
    from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata

    meta = LeRobotDatasetMetadata(repo_id="local/capx-eap-dataset", root=dataset_root)
    dataset = LeRobotDataset(repo_id="local/capx-eap-dataset", root=dataset_root, video_backend=args.video_backend)
    sample = dataset[0]
    summary = {
        "dataset_root": str(dataset_root),
        "total_episodes": int(meta.total_episodes),
        "fps": int(meta.fps),
        "video_keys": list(meta.video_keys),
        "num_rows": int(len(dataset)),
        "sample_keys": sorted(sample.keys()),
        "hf_home": validate_env["HF_HOME"],
        "hf_datasets_cache": validate_env["HF_DATASETS_CACHE"],
    }
    print(json.dumps(summary, indent=2))
    return summary


def _add_collect_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--repo-root", default=None)
    parser.add_argument("--api-bash-profile", default=None)
    parser.add_argument("--api-bash-file", default=None)
    parser.add_argument("--config-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--server-url", default=None)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--total-trials", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--record-video", action="store_true")
    parser.add_argument("--enable-eap-rollback", action="store_true")
    parser.add_argument("--enable-eap-recovery", action="store_true")
    parser.add_argument("--enable-eap-model-snapshot-selection", action="store_true")
    parser.add_argument("--use-img-differencing", action="store_true")
    parser.add_argument("--request-timeout-s", type=float, default=None)
    parser.add_argument("--request-max-attempts", type=int, default=None)
    parser.add_argument("--request-max-retry-walltime-s", type=float, default=None)
    parser.add_argument("--request-retry-initial-s", type=float, default=None)
    parser.add_argument("--request-retry-max-sleep-s", type=float, default=None)
    parser.add_argument("--tmux-session", default=None)
    parser.add_argument("--tmux-log-file", default=None)
    parser.add_argument("--tmux-replace-existing", action="store_true")
    parser.add_argument("--tmux-child", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--detached", action="store_true")
    parser.add_argument("--detached-log-file", default=None)
    parser.add_argument("--detached-pid-file", default=None)
    parser.add_argument("--detached-replace-existing", action="store_true")
    parser.add_argument("--detached-child", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument(
        "--extra-arg",
        action="append",
        nargs=2,
        metavar=("NAME", "VALUE"),
        default=[],
        help="Pass an extra launch.py argument as NAME VALUE.",
    )


def _add_collect_like_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--repo-root", default=None)
    parser.add_argument("--api-bash-profile", default=None)
    parser.add_argument("--api-bash-file", default=None)
    parser.add_argument("--server-url", default=None)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--record-video", action="store_true")
    parser.add_argument("--enable-eap-rollback", action="store_true")
    parser.add_argument("--enable-eap-recovery", action="store_true")
    parser.add_argument("--enable-eap-model-snapshot-selection", action="store_true")
    parser.add_argument("--use-img-differencing", action="store_true")
    parser.add_argument("--request-timeout-s", type=float, default=None)
    parser.add_argument("--request-max-attempts", type=int, default=None)
    parser.add_argument("--request-max-retry-walltime-s", type=float, default=None)
    parser.add_argument("--request-retry-initial-s", type=float, default=None)
    parser.add_argument("--request-retry-max-sleep-s", type=float, default=None)
    parser.add_argument("--tmux-session", default=None)
    parser.add_argument("--tmux-log-file", default=None)
    parser.add_argument("--tmux-replace-existing", action="store_true")
    parser.add_argument("--tmux-child", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--detached", action="store_true")
    parser.add_argument("--detached-log-file", default=None)
    parser.add_argument("--detached-pid-file", default=None)
    parser.add_argument("--detached-replace-existing", action="store_true")
    parser.add_argument("--detached-child", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument(
        "--extra-arg",
        action="append",
        nargs=2,
        metavar=("NAME", "VALUE"),
        default=[],
        help="Pass an extra launch.py argument as NAME VALUE.",
    )


def _add_export_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--repo-root", default=None)
    parser.add_argument("--input-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--robot-type", default="franka")
    parser.add_argument("--fps", type=int, default=None)
    parser.add_argument("--chunk-size", type=int, default=1000)
    parser.add_argument("--crf", type=int, default=30)
    parser.add_argument("--include-excluded", action="store_true")


def _add_export_shards_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--repo-root", default=None)
    parser.add_argument("--input-root", required=True)
    parser.add_argument("--output-root", required=True, help="Root directory for per-attempt LeRobot shards.")
    parser.add_argument("--robot-type", default="franka")
    parser.add_argument("--fps", type=int, default=None)
    parser.add_argument("--chunk-size", type=int, default=1000)
    parser.add_argument("--crf", type=int, default=30)
    parser.add_argument("--include-excluded", action="store_true")
    parser.add_argument(
        "--cleanup-transition-dataset",
        action="store_true",
        help="Delete each attempt's transition_dataset directory after successful shard export.",
    )
    parser.add_argument(
        "--reexport-existing",
        action="store_true",
        help="Re-export shards even if shard_root/manifest.json already exists.",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Orchestrate CaP-X collection, LeRobot export, and validation.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    collect_parser = subparsers.add_parser("collect")
    _add_collect_args(collect_parser)

    manifest_parser = subparsers.add_parser("write-manifest")
    manifest_parser.add_argument("--repo-root", default=None)
    manifest_parser.add_argument("--preset", choices=sorted(_builtin_manifests().keys()), required=True)
    manifest_parser.add_argument("--output-path", required=True)
    manifest_parser.add_argument("--trials-per-task", type=int, default=DEFAULT_LIBERO_TRIALS_PER_TASK)

    collect_manifest_parser = subparsers.add_parser("collect-manifest")
    _add_collect_like_args(collect_manifest_parser)
    collect_manifest_parser.add_argument("--manifest-path", required=True)
    collect_manifest_parser.add_argument("--output-root", required=True)
    collect_manifest_parser.add_argument("--suite-filter", action="append", default=[])
    collect_manifest_parser.add_argument("--trials-per-task", type=int, default=None)
    collect_manifest_parser.add_argument("--task-max-restarts", type=int, default=DEFAULT_MANIFEST_TASK_MAX_RESTARTS)

    export_parser = subparsers.add_parser("export")
    _add_export_args(export_parser)

    export_shards_parser = subparsers.add_parser("export-shards")
    _add_export_shards_args(export_shards_parser)

    cleanup_parser = subparsers.add_parser("cleanup-services")
    cleanup_parser.add_argument("--repo-root", default=None)
    cleanup_parser.add_argument("--config-path", required=True)
    cleanup_parser.add_argument("--port", action="append", type=int, default=[])

    validate_parser = subparsers.add_parser("validate")
    validate_parser.add_argument("--dataset-root", required=True)
    validate_parser.add_argument("--video-backend", default="pyav")
    validate_parser.add_argument("--hf-cache-root", default=None)

    collect_export_parser = subparsers.add_parser("collect-export")
    _add_collect_args(collect_export_parser)
    collect_export_parser.add_argument("--lerobot-output-root", required=True)
    collect_export_parser.add_argument("--robot-type", default="franka")
    collect_export_parser.add_argument("--fps", type=int, default=None)
    collect_export_parser.add_argument("--chunk-size", type=int, default=1000)
    collect_export_parser.add_argument("--crf", type=int, default=30)
    collect_export_parser.add_argument("--include-excluded", action="store_true")
    collect_export_parser.add_argument("--validate", action="store_true")
    collect_export_parser.add_argument("--video-backend", default="pyav")
    collect_export_parser.add_argument("--hf-cache-root", default=None)

    args = parser.parse_args()

    if (
        args.command in {"collect", "collect-export", "collect-manifest"}
        and args.tmux_session
        and getattr(args, "detached", False)
    ):
        raise ValueError("Use either --tmux-session or --detached, not both.")

    if args.command in {"collect", "collect-export", "collect-manifest"} and args.tmux_session and not args.tmux_child:
        print(json.dumps(_spawn_tmux_session(args, sys.argv[1:]), indent=2))
        return

    if (
        args.command in {"collect", "collect-export", "collect-manifest"}
        and getattr(args, "detached", False)
        and not getattr(args, "detached_child", False)
    ):
        print(json.dumps(_spawn_detached_process(args, sys.argv[1:]), indent=2))
        return

    if args.command == "cleanup-services":
        _cleanup_services(args)
        return

    if args.command == "write-manifest":
        _write_manifest(args)
        return

    if args.command == "collect":
        print(json.dumps(_collect(args), indent=2))
        return

    if args.command == "collect-manifest":
        _collect_manifest(args)
        return

    if args.command == "export":
        print(json.dumps(_export(args), indent=2))
        return

    if args.command == "export-shards":
        _export_shards(args)
        return

    if args.command == "validate":
        _validate(args)
        return

    collect_result = _collect(args)
    export_args = argparse.Namespace(
        repo_root=args.repo_root,
        input_root=args.output_dir,
        output_root=args.lerobot_output_root,
        robot_type=args.robot_type,
        fps=args.fps,
        chunk_size=args.chunk_size,
        crf=args.crf,
        include_excluded=args.include_excluded,
    )
    export_result = _export(export_args)
    result: dict[str, object] = {
        **collect_result,
        **export_result,
    }
    if args.validate:
        result["validation"] = _validate(
            argparse.Namespace(
                dataset_root=args.lerobot_output_root,
                video_backend=args.video_backend,
                hf_cache_root=args.hf_cache_root,
            )
        )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
