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
    cleanup_cmd = shlex.join(
        [
            python_bin,
            str(script_path),
            "cleanup-services",
            "--repo-root",
            str(repo_root),
            "--config-path",
            args.config_path,
        ]
    )
    shell_cmd = "\n".join(
        [
            "set -euo pipefail",
            f"cd {shlex.quote(str(repo_root))}",
            env_exports,
            "cleanup() {",
            f"  {cleanup_cmd} >/dev/null 2>&1 || true",
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
    finally:
        _cleanup_services(
            argparse.Namespace(repo_root=str(repo_root), config_path=args.config_path, port=[])
        )
    return {
        "repo_root": str(repo_root),
        "collection_output_dir": str(_normalize_output_path(repo_root, args.output_dir)),
    }


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


def main() -> None:
    parser = argparse.ArgumentParser(description="Orchestrate CaP-X collection, LeRobot export, and validation.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    collect_parser = subparsers.add_parser("collect")
    _add_collect_args(collect_parser)

    export_parser = subparsers.add_parser("export")
    _add_export_args(export_parser)

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

    if args.command in {"collect", "collect-export"} and args.tmux_session and not args.tmux_child:
        print(json.dumps(_spawn_tmux_session(args, sys.argv[1:]), indent=2))
        return

    if args.command == "cleanup-services":
        _cleanup_services(args)
        return

    if args.command == "collect":
        print(json.dumps(_collect(args), indent=2))
        return

    if args.command == "export":
        print(json.dumps(_export(args), indent=2))
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
