#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


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


def _run(cmd: list[str], cwd: Path) -> None:
    print("$", " ".join(cmd))
    subprocess.run(cmd, cwd=cwd, check=True)


def _collect(args: argparse.Namespace) -> dict[str, str]:
    repo_root = _find_repo_root(args.repo_root)
    python_bin = _select_python(repo_root)
    cmd = [
        python_bin,
        str(repo_root / "capx" / "envs" / "launch.py"),
        "--config-path",
        args.config_path,
        "--output-dir",
        args.output_dir,
    ]

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

    for extra in args.extra_arg:
        cmd.extend(["--" + extra[0], extra[1]])

    _run(cmd, cwd=repo_root)
    return {
        "repo_root": str(repo_root),
        "collection_output_dir": str(
            (repo_root / args.output_dir).resolve()
            if not Path(args.output_dir).is_absolute()
            else Path(args.output_dir).resolve()
        ),
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
    }
    print(json.dumps(summary, indent=2))
    return summary


def _add_collect_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--repo-root", default=None)
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Orchestrate CaP-X collection, LeRobot export, and validation.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    collect_parser = subparsers.add_parser("collect")
    _add_collect_args(collect_parser)

    export_parser = subparsers.add_parser("export")
    _add_export_args(export_parser)

    validate_parser = subparsers.add_parser("validate")
    validate_parser.add_argument("--dataset-root", required=True)
    validate_parser.add_argument("--video-backend", default="pyav")

    collect_export_parser = subparsers.add_parser("collect-export")
    _add_collect_args(collect_export_parser)
    collect_export_parser.add_argument("--lerobot-output-root", required=True)
    collect_export_parser.add_argument("--robot-type", default="franka")
    collect_export_parser.add_argument("--fps", type=int, default=None)
    collect_export_parser.add_argument("--chunk-size", type=int, default=1000)
    collect_export_parser.add_argument("--crf", type=int, default=30)
    collect_export_parser.add_argument("--validate", action="store_true")
    collect_export_parser.add_argument("--video-backend", default="pyav")

    args = parser.parse_args()

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
    )
    export_result = _export(export_args)
    result: dict[str, object] = {
        **collect_result,
        **export_result,
    }
    if args.validate:
        result["validation"] = _validate(
            argparse.Namespace(dataset_root=args.lerobot_output_root, video_backend=args.video_backend)
        )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
