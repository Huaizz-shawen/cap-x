from __future__ import annotations

import argparse
import gc
import gzip
import json
import math
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import yaml


CAMERA_SPECS = {
    "observation.images.agentview": {
        "initial_path": ("agentview", "images", "rgb"),
        "transition_key": "agentview_image",
        "is_depth_map": False,
    },
    "observation.images.wrist": {
        "initial_path": ("robot0_eye_in_hand", "images", "rgb"),
        "transition_key": "robot0_eye_in_hand_image",
        "is_depth_map": False,
    },
}

ACTION_NAMES = [
    "joint_delta_1",
    "joint_delta_2",
    "joint_delta_3",
    "joint_delta_4",
    "joint_delta_5",
    "joint_delta_6",
    "joint_delta_7",
    "gripper_command",
]

STATE_NAMES = [
    "joint_pos_1",
    "joint_pos_2",
    "joint_pos_3",
    "joint_pos_4",
    "joint_pos_5",
    "joint_pos_6",
    "joint_pos_7",
    "gripper_position",
]

EE_STATE_NAMES = [
    "eef_x",
    "eef_y",
    "eef_z",
    "eef_quat_w",
    "eef_quat_x",
    "eef_quat_y",
    "eef_quat_z",
    "gripper_position",
]

STANDARD_LIBERO_SUITES = {
    "libero_10",
    "libero_object",
    "libero_spatial",
    "libero_goal",
}

ROBOSUITE_LOW_LEVEL_REGISTRY = {
    "franka_robosuite_cubes_low_level": "cube_stack",
    "franka_robosuite_cube_lift_low_level": "cube_lifting",
    "franka_robosuite_cubes_restack_low_level": "cube_restack",
    "franka_robosuite_spill_wipe_low_level": "spill_wipe",
    "franka_robosuite_nut_assembly_low_level": "nut_assembly",
    "franka_robosuite_nut_assembly_low_level_visual": "nut_assembly",
    "two_arm_handover_robosuite": "two_arm_handover",
    "two_arm_lift_robosuite": "two_arm_lift",
}


def _jsonify(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonify(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonify(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _load_pickle_gz(path: Path) -> dict[str, Any]:
    with gzip.open(path, "rb") as handle:
        return pickle.load(handle)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _extract_goal(task_prompt: str | None) -> str:
    if not task_prompt:
        return "unknown task"
    for line in task_prompt.splitlines():
        line = line.strip()
        if line.lower().startswith("goal:"):
            return line.split(":", 1)[1].strip()
    return task_prompt.strip()


def _nested_get(data: dict[str, Any], path: tuple[str, ...]) -> Any | None:
    current: Any = data
    for key in path:
        if not isinstance(current, dict) or key not in current:
            return None
        current = current[key]
    return current


def _extract_frame_from_initial(initial_observation: dict[str, Any], camera_key: str) -> np.ndarray | None:
    spec = CAMERA_SPECS[camera_key]
    frame = _nested_get(initial_observation, spec["initial_path"])
    if frame is None:
        return None
    return np.asarray(frame, dtype=np.uint8)


def _extract_frame_from_transition_observation(
    transition_observation: dict[str, Any],
    camera_key: str,
) -> np.ndarray | None:
    low_level_observation = transition_observation.get("low_level_observation", {})
    frame = low_level_observation.get(CAMERA_SPECS[camera_key]["transition_key"])
    if frame is None:
        return None
    return np.asarray(frame, dtype=np.uint8)[::-1]


def _build_row_observation_from_initial(initial_observation: dict[str, Any]) -> dict[str, Any]:
    return {
        "robot_joint_pos": np.asarray(initial_observation["robot_joint_pos"], dtype=np.float32),
        "robot_cartesian_pos": np.asarray(initial_observation["robot_cartesian_pos"], dtype=np.float32),
        "frames": {
            camera_key: _extract_frame_from_initial(initial_observation, camera_key)
            for camera_key in CAMERA_SPECS
        },
    }


def _build_row_observation_from_transition(transition: dict[str, Any]) -> dict[str, Any]:
    observation = transition["observation"]
    return {
        "robot_joint_pos": np.asarray(observation["robot_joint_pos"], dtype=np.float32),
        "robot_cartesian_pos": np.asarray(observation["robot_cartesian_pos"], dtype=np.float32),
        "frames": {
            camera_key: _extract_frame_from_transition_observation(observation, camera_key)
            for camera_key in CAMERA_SPECS
        },
    }


def _safe_bool(value: Any) -> bool:
    return bool(value) if value is not None else False


def _safe_float(value: Any) -> float:
    return float(value) if value is not None else 0.0


def _compute_fps(all_transition_datasets: list[dict[str, Any]]) -> int:
    deltas: list[float] = []
    for dataset in all_transition_datasets:
        timestamps = [float(transition["timestamp_s"]) for transition in dataset.get("transitions", [])]
        deltas.extend(
            max(b - a, 0.0)
            for a, b in zip(timestamps[:-1], timestamps[1:], strict=False)
            if b > a
        )
    if not deltas:
        return 20
    median_delta = float(np.median(np.asarray(deltas, dtype=np.float32)))
    if median_delta <= 0:
        return 20
    return max(1, int(round(1.0 / median_delta)))


def _compute_fps_from_paths(transition_paths: list[Path]) -> int:
    deltas: list[float] = []
    for path in transition_paths:
        dataset = _load_pickle_gz(path)
        timestamps = [float(transition["timestamp_s"]) for transition in dataset.get("transitions", [])]
        deltas.extend(
            max(b - a, 0.0)
            for a, b in zip(timestamps[:-1], timestamps[1:], strict=False)
            if b > a
        )
    if not deltas:
        return 20
    median_delta = float(np.median(np.asarray(deltas, dtype=np.float32)))
    if median_delta <= 0:
        return 20
    return max(1, int(round(1.0 / median_delta)))


def _video_feature_info(frame: np.ndarray, fps: int, codec: str) -> dict[str, Any]:
    height, width, channels = frame.shape
    return {
        "dtype": "video",
        "shape": [height, width, channels],
        "names": ["height", "width", "channels"],
        "info": {
            "video.height": int(height),
            "video.width": int(width),
            "video.codec": codec,
            "video.pix_fmt": "yuv420p",
            "video.is_depth_map": False,
            "video.fps": int(fps),
            "video.channels": int(channels),
            "has_audio": False,
        },
    }


def _compute_numeric_stats(values: np.ndarray) -> dict[str, Any]:
    if values.ndim == 1:
        values = values[:, None]
    return {
        "min": values.min(axis=0).tolist(),
        "max": values.max(axis=0).tolist(),
        "mean": values.mean(axis=0).tolist(),
        "std": values.std(axis=0).tolist(),
        "count": [int(values.shape[0])],
    }


def _init_running_stats(dim: int) -> dict[str, Any]:
    return {
        "min": np.full((dim,), np.inf, dtype=np.float64),
        "max": np.full((dim,), -np.inf, dtype=np.float64),
        "sum": np.zeros((dim,), dtype=np.float64),
        "sumsq": np.zeros((dim,), dtype=np.float64),
        "count": 0,
    }


def _update_running_stats(stats: dict[str, Any], value: np.ndarray) -> None:
    arr = np.asarray(value, dtype=np.float64).reshape(-1)
    stats["min"] = np.minimum(stats["min"], arr)
    stats["max"] = np.maximum(stats["max"], arr)
    stats["sum"] += arr
    stats["sumsq"] += arr * arr
    stats["count"] += 1


def _finalize_running_stats(stats: dict[str, Any]) -> dict[str, Any]:
    count = int(stats["count"])
    if count <= 0:
        return {
            "min": [],
            "max": [],
            "mean": [],
            "std": [],
            "count": [0],
        }
    mean = stats["sum"] / float(count)
    var = np.maximum(stats["sumsq"] / float(count) - np.square(mean), 0.0)
    std = np.sqrt(var)
    return {
        "min": stats["min"].tolist(),
        "max": stats["max"].tolist(),
        "mean": mean.tolist(),
        "std": std.tolist(),
        "count": [count],
    }


def _directory_size_bytes(path: Path) -> int:
    total = 0
    if not path.exists():
        return total
    for candidate in path.rglob("*"):
        if candidate.is_file():
            total += candidate.stat().st_size
    return total


def _episode_numeric_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    actions = np.stack([np.asarray(row["action"], dtype=np.float32) for row in rows], axis=0)
    joint_state = np.stack([np.asarray(row["observation.state"], dtype=np.float32) for row in rows], axis=0)
    eef_state = np.stack([np.asarray(row["observation.eef_state"], dtype=np.float32) for row in rows], axis=0)
    rewards = np.asarray([row["reward"] for row in rows], dtype=np.float32)
    timestamps = np.asarray([row["timestamp"] for row in rows], dtype=np.float32)
    return {
        "action": _compute_numeric_stats(actions),
        "observation.state": _compute_numeric_stats(joint_state),
        "observation.eef_state": _compute_numeric_stats(eef_state),
        "reward": _compute_numeric_stats(rewards),
        "timestamp": _compute_numeric_stats(timestamps),
    }


def _write_data_parquet(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.table(
        {
            "action": pa.array(
                [np.asarray(row["action"], dtype=np.float32).tolist() for row in rows],
                type=pa.list_(pa.float32(), len(ACTION_NAMES)),
            ),
            "observation.state": pa.array(
                [np.asarray(row["observation.state"], dtype=np.float32).tolist() for row in rows],
                type=pa.list_(pa.float32(), len(STATE_NAMES)),
            ),
            "observation.eef_state": pa.array(
                [np.asarray(row["observation.eef_state"], dtype=np.float32).tolist() for row in rows],
                type=pa.list_(pa.float32(), len(EE_STATE_NAMES)),
            ),
            "reward": pa.array([np.float32(row["reward"]) for row in rows], type=pa.float32()),
            "done": pa.array([bool(row["done"]) for row in rows], type=pa.bool_()),
            "truncated": pa.array([bool(row["truncated"]) for row in rows], type=pa.bool_()),
            "timestamp": pa.array([np.float32(row["timestamp"]) for row in rows], type=pa.float32()),
            "frame_index": pa.array([int(row["frame_index"]) for row in rows], type=pa.int64()),
            "episode_index": pa.array([int(row["episode_index"]) for row in rows], type=pa.int64()),
            "index": pa.array([int(row["index"]) for row in rows], type=pa.int64()),
            "task_index": pa.array([int(row["task_index"]) for row in rows], type=pa.int64()),
        }
    )
    pq.write_table(table, path)


@dataclass
class EpisodeRecord:
    episode_index: int
    task_index: int
    task: str
    source_trial_dir: str
    source_metadata: dict[str, Any]
    attempt: int | None
    task_completed: bool
    success: bool
    excluded_from_training: bool
    exclusion_reason: str | None
    rows: list[dict[str, Any]]
    frames: dict[str, list[np.ndarray]]


def _downsample_indices(length: int, stride: int) -> list[int]:
    if length <= 0:
        return []
    if stride <= 1:
        return list(range(length))
    indices = list(range(0, length, stride))
    if indices[-1] != length - 1:
        indices.append(length - 1)
    return indices


def _downsample_episode_records(
    records: list[EpisodeRecord],
    *,
    source_fps: int,
    target_fps: int | None,
) -> tuple[list[EpisodeRecord], int]:
    if target_fps is None or target_fps <= 0:
        return records, source_fps

    # Time-based downsampling is robust even when source fps metadata is unavailable or noisy.
    interval_s = 1.0 / float(target_fps)
    downsampled: list[EpisodeRecord] = []
    for record in records:
        if not record.rows:
            downsampled.append(record)
            continue

        keep: list[int] = [0]
        last_kept_ts = float(record.rows[0]["timestamp"])
        for idx in range(1, len(record.rows)):
            ts = float(record.rows[idx]["timestamp"])
            if ts - last_kept_ts >= interval_s:
                keep.append(idx)
                last_kept_ts = ts
        if keep[-1] != len(record.rows) - 1:
            keep.append(len(record.rows) - 1)
        keep_set = set(keep)

        down_rows = [row for idx, row in enumerate(record.rows) if idx in keep_set]
        down_frames: dict[str, list[np.ndarray]] = {}
        for camera_key, frames in record.frames.items():
            down_frames[camera_key] = [frame for idx, frame in enumerate(frames) if idx in keep_set]

        downsampled.append(
            EpisodeRecord(
                episode_index=record.episode_index,
                task_index=record.task_index,
                task=record.task,
                source_trial_dir=record.source_trial_dir,
                source_metadata=record.source_metadata,
                attempt=record.attempt,
                task_completed=record.task_completed,
                success=record.success,
                excluded_from_training=record.excluded_from_training,
                exclusion_reason=record.exclusion_reason,
                rows=down_rows,
                frames=down_frames,
            )
        )

    return downsampled, int(target_fps)


def _exclude_from_training(
    transition_dataset: dict[str, Any],
    trajectory_metadata: dict[str, Any],
) -> tuple[bool, str | None]:
    trial_metadata = transition_dataset.get("trial_metadata")
    if isinstance(trial_metadata, dict):
        if trial_metadata.get("exclude_from_training"):
            return True, str(trial_metadata.get("exclusion_reason") or "unknown")
        return False, None

    trial_complete = next(
        (event for event in trajectory_metadata.get("events", []) if event.get("event_type") == "trial_complete"),
        {},
    )
    if bool(trial_complete.get("truncated")):
        return True, "sim_limit_reset"
    return False, None


def _resolve_config_path(config_path: str | None, source_trial_dir: Path) -> Path | None:
    if not config_path:
        return None
    candidate = Path(config_path).expanduser()
    if candidate.is_absolute() and candidate.exists():
        return candidate
    for parent in [source_trial_dir, *source_trial_dir.parents]:
        resolved = parent / config_path
        if resolved.exists():
            return resolved
    cwd_resolved = Path.cwd() / config_path
    if cwd_resolved.exists():
        return cwd_resolved
    return None


def _annotate_libero_metadata(metadata: dict[str, Any], suite_name: Any, task_id: Any) -> None:
    metadata["benchmark"] = "libero"
    if suite_name is not None:
        suite_name = str(suite_name)
        metadata["suite_name"] = suite_name
        metadata["suite_category"] = "standard" if suite_name in STANDARD_LIBERO_SUITES else "extended"
    if task_id is not None:
        metadata["task_id"] = int(task_id)


def _infer_robosuite_task_family(simulator_name: str) -> str | None:
    if simulator_name in ROBOSUITE_LOW_LEVEL_REGISTRY:
        return ROBOSUITE_LOW_LEVEL_REGISTRY[simulator_name]
    return None


def _infer_metadata_from_low_level_target(
    low_level_target: str,
    low_level_cfg: dict[str, Any],
    metadata: dict[str, Any],
) -> None:
    metadata["low_level_target"] = low_level_target

    if low_level_target == "capx.envs.simulators.libero.FrankaLiberoEnv":
        _annotate_libero_metadata(metadata, low_level_cfg.get("suite_name"), low_level_cfg.get("task_id"))
        return

    if low_level_target == "capx.envs.simulators.r1pro_b1k.R1ProBehaviourLowLevel":
        metadata["benchmark"] = "behavior1k"
        metadata["benchmark_variant"] = "r1pro"
        if low_level_cfg.get("activity_name") is not None:
            metadata["activity_name"] = str(low_level_cfg["activity_name"])
        if low_level_cfg.get("controller_cfg") is not None:
            metadata["controller_cfg"] = str(low_level_cfg["controller_cfg"])
        return

    if low_level_target == "capx.envs.simulators.franka_real.FrankaRealLowLevel":
        metadata["benchmark"] = "real"
        return

    if ".robosuite_" in low_level_target or "Robosuite" in low_level_target:
        metadata["benchmark"] = "robosuite"
        metadata["simulator_class"] = low_level_target.rsplit(".", 1)[-1]


def _infer_metadata_from_low_level_registry(low_level_name: str, metadata: dict[str, Any]) -> None:
    metadata["low_level_registry_name"] = low_level_name

    if low_level_name.startswith("franka_libero_") and low_level_name.endswith("_low_level"):
        suffix = low_level_name[len("franka_libero_") : -len("_low_level")]
        suite_name, _, task_id = suffix.rpartition("_")
        if suite_name and task_id.isdigit():
            _annotate_libero_metadata(metadata, suite_name, int(task_id))
            return

    if low_level_name == "r1pro_b1k_low_level":
        metadata["benchmark"] = "behavior1k"
        metadata["benchmark_variant"] = "r1pro"
        return

    if low_level_name == "franka_real_low_level":
        metadata["benchmark"] = "real"
        return

    robosuite_task_family = _infer_robosuite_task_family(low_level_name)
    if robosuite_task_family is not None:
        metadata["benchmark"] = "robosuite"
        metadata["task_family"] = robosuite_task_family


def _infer_source_metadata(
    transition_dataset: dict[str, Any],
    source_trial_dir: Path,
) -> dict[str, Any]:
    config_path = transition_dataset.get("config_path")
    metadata: dict[str, Any] = {
        "config_path": config_path,
        "benchmark": "unknown",
    }
    config_path_str = str(config_path) if config_path is not None else ""
    if "/libero/" in config_path_str or config_path_str.startswith("env_configs/libero/"):
        metadata["benchmark"] = "libero"
    elif "/robosuite/" in config_path_str or "robosuite" in config_path_str:
        metadata["benchmark"] = "robosuite"

    resolved_config = _resolve_config_path(config_path, source_trial_dir)
    if resolved_config is None:
        return metadata

    try:
        config = yaml.safe_load(resolved_config.read_text(encoding="utf-8")) or {}
    except Exception:
        return metadata

    env_cfg = config.get("env", {}).get("cfg", {})
    low_level = env_cfg.get("low_level", {})

    env_target = config.get("env", {}).get("_target_")
    if isinstance(env_target, str):
        metadata["env_target"] = env_target

    if isinstance(low_level, dict):
        low_level_target = low_level.get("_target_")
        if isinstance(low_level_target, str):
            _infer_metadata_from_low_level_target(low_level_target, low_level, metadata)
    elif isinstance(low_level, str):
        _infer_metadata_from_low_level_registry(low_level, metadata)

    privileged = env_cfg.get("privileged")
    if privileged is None and isinstance(low_level, dict):
        privileged = low_level.get("privileged")
    if privileged is not None:
        metadata["privileged"] = bool(privileged)

    if metadata["benchmark"] == "unknown" and isinstance(env_target, str):
        if ".r1pro." in env_target:
            metadata["benchmark"] = "behavior1k"
            metadata.setdefault("benchmark_variant", "r1pro")
        elif ".two_arm_" in env_target or ".franka_" in env_target:
            metadata["benchmark"] = "robosuite"
    return metadata


def _build_episode_record(
    *,
    episode_index: int,
    transition_dataset: dict[str, Any],
    trajectory_summary: dict[str, Any],
    source_trial_dir: Path,
) -> EpisodeRecord:
    transitions = transition_dataset.get("transitions", [])
    if not transitions:
        raise ValueError(f"{source_trial_dir} has no transitions")

    initial_observation = transition_dataset.get("initial_observation")
    if not isinstance(initial_observation, dict):
        raise ValueError(f"{source_trial_dir} is missing initial_observation")

    task = _extract_goal(transition_dataset.get("task_prompt"))
    frames_by_camera = {camera_key: [] for camera_key in CAMERA_SPECS}
    rows: list[dict[str, Any]] = []

    previous_observation = _build_row_observation_from_initial(initial_observation)
    previous_timestamp = 0.0

    for frame_index, transition in enumerate(transitions):
        row_frames = previous_observation["frames"]
        for camera_key, frame in row_frames.items():
            if frame is not None:
                frames_by_camera[camera_key].append(frame)

        rows.append({
            "action": np.asarray(transition["action"], dtype=np.float32),
            "observation.state": np.asarray(previous_observation["robot_joint_pos"], dtype=np.float32),
            "observation.eef_state": np.asarray(previous_observation["robot_cartesian_pos"], dtype=np.float32),
            "reward": _safe_float(transition.get("reward")),
            "done": _safe_bool(transition.get("done")),
            "truncated": _safe_bool(transition.get("truncated")),
            "timestamp": float(previous_timestamp),
            "frame_index": int(frame_index),
            "source": transition.get("source", "unknown"),
            "action_context": (
                transition.get("action_context", {}).get("action_name", "none")
                if transition.get("action_context") is not None
                else "none"
            ),
        })

        previous_observation = _build_row_observation_from_transition(transition)
        previous_timestamp = float(transition["timestamp_s"])

    available_cameras = {
        camera_key: frames
        for camera_key, frames in frames_by_camera.items()
        if len(frames) == len(rows)
    }
    excluded_from_training, exclusion_reason = _exclude_from_training(
        transition_dataset,
        trajectory_summary,
    )
    trial_complete = next(
        (event for event in trajectory_summary.get("events", []) if event.get("event_type") == "trial_complete"),
        {},
    )
    return EpisodeRecord(
        episode_index=episode_index,
        task_index=-1,
        task=task,
        source_trial_dir=str(source_trial_dir.resolve()),
        source_metadata=_infer_source_metadata(transition_dataset, source_trial_dir),
        attempt=transition_dataset.get("attempt"),
        task_completed=bool(trial_complete.get("task_completed", False)),
        success=bool(trial_complete.get("success", False)),
        excluded_from_training=excluded_from_training,
        exclusion_reason=exclusion_reason,
        rows=rows,
        frames=available_cameras,
    )


def _write_video(video_path: Path, frames: list[np.ndarray], fps: int, codec: str, crf: int) -> None:
    video_path.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(
        video_path,
        fps=fps,
        codec="libx264",
        pixelformat="yuv420p",
        macro_block_size=None,
        ffmpeg_log_level="error",
        ffmpeg_params=["-crf", str(crf), "-preset", "veryfast"],
    )
    try:
        for frame in frames:
            writer.append_data(frame)
    finally:
        writer.close()


def export_lerobot_dataset(
    *,
    input_root: Path,
    output_root: Path,
    robot_type: str,
    fps: int | None,
    downsample_fps: int | None = None,
    chunk_size: int,
    crf: int,
    include_excluded: bool = False,
) -> dict[str, Any]:
    transition_paths = sorted(input_root.glob("**/transition_dataset/data.pkl.gz"))
    if not transition_paths:
        raise FileNotFoundError(f"No transition datasets found under {input_root}")

    inferred_fps: int | None = None
    if fps is not None:
        dataset_fps = int(fps)
    elif downsample_fps is not None and downsample_fps > 0:
        # Avoid scanning huge payloads only to infer source fps.
        dataset_fps = int(downsample_fps)
    else:
        inferred_fps = _compute_fps_from_paths(transition_paths)
        dataset_fps = inferred_fps

    trial_summaries = {
        path.parent.parent: _read_json(path.parent.parent / "trajectory" / "metadata.json")
        for path in input_root.glob("**/trajectory/metadata.json")
    }

    effective_fps = dataset_fps
    if downsample_fps is not None and downsample_fps > 0 and downsample_fps < dataset_fps:
        effective_fps = int(downsample_fps)
    dataset_fps = effective_fps

    task_to_index: dict[tuple[str, tuple[tuple[str, Any], ...]], int] = {}
    task_records_by_index: dict[int, dict[str, Any]] = {}

    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "meta" / "episodes").mkdir(parents=True, exist_ok=True)
    (output_root / "data").mkdir(parents=True, exist_ok=True)
    (output_root / "videos").mkdir(parents=True, exist_ok=True)

    total_frames = 0
    global_index = 0
    numeric_stats_running: dict[str, dict[str, Any]] = {
        "action": _init_running_stats(len(ACTION_NAMES)),
        "observation.state": _init_running_stats(len(STATE_NAMES)),
        "observation.eef_state": _init_running_stats(len(EE_STATE_NAMES)),
        "reward": _init_running_stats(1),
        "timestamp": _init_running_stats(1),
        "frame_index": _init_running_stats(1),
        "episode_index": _init_running_stats(1),
        "index": _init_running_stats(1),
        "task_index": _init_running_stats(1),
    }
    all_episode_rows: list[dict[str, Any]] = []
    all_episode_stats_rows: list[dict[str, Any]] = []

    first_frame_by_camera: dict[str, np.ndarray] = {}
    chunk_index = 0
    file_index = 0
    episodes_in_chunk = 0
    total_episodes = 0
    total_videos = 0

    data_rows: list[dict[str, Any]] = []
    episode_rows: list[dict[str, Any]] = []
    video_frames_by_camera: dict[str, list[np.ndarray]] = {camera_key: [] for camera_key in CAMERA_SPECS}
    video_offsets_by_camera: dict[str, int] = {camera_key: 0 for camera_key in CAMERA_SPECS}

    def flush_chunk() -> None:
        nonlocal data_rows
        nonlocal episode_rows
        nonlocal video_frames_by_camera
        nonlocal video_offsets_by_camera
        nonlocal chunk_index
        nonlocal episodes_in_chunk
        if not episode_rows:
            return

        data_chunk_dir = output_root / "data" / f"chunk-{chunk_index:03d}"
        data_chunk_dir.mkdir(parents=True, exist_ok=True)
        _write_data_parquet(data_chunk_dir / f"file-{file_index:03d}.parquet", data_rows)

        episodes_chunk_dir = output_root / "meta" / "episodes" / f"chunk-{chunk_index:03d}"
        episodes_chunk_dir.mkdir(parents=True, exist_ok=True)
        episodes_df = pd.DataFrame(episode_rows)
        episodes_df.to_parquet(
            episodes_chunk_dir / f"file-{file_index:03d}.parquet",
            index=False,
            engine="pyarrow",
        )

        for camera_key, frames in video_frames_by_camera.items():
            if not frames:
                continue
            video_chunk_dir = output_root / "videos" / camera_key / f"chunk-{chunk_index:03d}"
            _write_video(
                video_chunk_dir / f"file-{file_index:03d}.mp4",
                frames=frames,
                fps=dataset_fps,
                codec="h264",
                crf=crf,
            )

        data_rows = []
        episode_rows = []
        video_frames_by_camera = {camera_key: [] for camera_key in CAMERA_SPECS}
        video_offsets_by_camera = {camera_key: 0 for camera_key in CAMERA_SPECS}
        episodes_in_chunk = 0
        chunk_index += 1
        gc.collect()

    for transition_path in transition_paths:
        print(f"[export] loading {transition_path}", flush=True)
        transition_dataset = _load_pickle_gz(transition_path)
        trial_dir = transition_path.parent.parent
        trajectory_meta = trial_summaries.get(trial_dir, {})
        episode = _build_episode_record(
            episode_index=total_episodes,
            transition_dataset=transition_dataset,
            trajectory_summary=trajectory_meta,
            source_trial_dir=trial_dir,
        )
        if episode.excluded_from_training and not include_excluded:
            continue

        if downsample_fps is not None and downsample_fps > 0:
            source_fps = int(fps) if fps is not None else int(inferred_fps or downsample_fps)
            downsampled, _ = _downsample_episode_records(
                [episode],
                source_fps=source_fps,
                target_fps=downsample_fps,
            )
            episode = downsampled[0]

        source_metadata_items = tuple(sorted(episode.source_metadata.items()))
        task_key = (episode.task, source_metadata_items)
        if task_key not in task_to_index:
            task_index = len(task_to_index)
            task_to_index[task_key] = task_index
            task_records_by_index[task_index] = {
                "task_index": task_index,
                "task": episode.task,
                **episode.source_metadata,
            }
        episode.task_index = task_to_index[task_key]

        dataset_from_index = global_index
        episode_video_from_index = {
            camera_key: video_offsets_by_camera[camera_key]
            for camera_key in episode.frames
        }

        for row in episode.rows:
            data_rows.append({
                "action": row["action"].tolist(),
                "observation.state": row["observation.state"].tolist(),
                "observation.eef_state": row["observation.eef_state"].tolist(),
                "reward": float(row["reward"]),
                "done": bool(row["done"]),
                "truncated": bool(row["truncated"]),
                "timestamp": float(row["timestamp"]),
                "frame_index": int(row["frame_index"]),
                "episode_index": int(episode.episode_index),
                "index": int(global_index),
                "task_index": int(episode.task_index),
                "source": row["source"],
                "action_context": row["action_context"],
            })
            _update_running_stats(numeric_stats_running["action"], np.asarray(row["action"], dtype=np.float32))
            _update_running_stats(
                numeric_stats_running["observation.state"],
                np.asarray(row["observation.state"], dtype=np.float32),
            )
            _update_running_stats(
                numeric_stats_running["observation.eef_state"],
                np.asarray(row["observation.eef_state"], dtype=np.float32),
            )
            _update_running_stats(numeric_stats_running["reward"], np.asarray([row["reward"]], dtype=np.float32))
            _update_running_stats(
                numeric_stats_running["timestamp"],
                np.asarray([row["timestamp"]], dtype=np.float32),
            )
            _update_running_stats(
                numeric_stats_running["frame_index"],
                np.asarray([row["frame_index"]], dtype=np.int64),
            )
            _update_running_stats(
                numeric_stats_running["episode_index"],
                np.asarray([episode.episode_index], dtype=np.int64),
            )
            _update_running_stats(
                numeric_stats_running["index"],
                np.asarray([global_index], dtype=np.int64),
            )
            _update_running_stats(
                numeric_stats_running["task_index"],
                np.asarray([episode.task_index], dtype=np.int64),
            )
            global_index += 1

        for camera_key, frames in episode.frames.items():
            if frames and camera_key not in first_frame_by_camera:
                first_frame_by_camera[camera_key] = frames[0]
            video_frames_by_camera[camera_key].extend(frames)
            video_offsets_by_camera[camera_key] += len(frames)

        dataset_to_index = global_index - 1
        episode_row = {
            "episode_index": int(episode.episode_index),
            "task_index": int(episode.task_index),
            "length": int(len(episode.rows)),
            "dataset_from_index": int(dataset_from_index),
            "dataset_to_index": int(dataset_to_index),
            "chunk_index": int(chunk_index),
            "file_index": int(file_index),
            "data/chunk_index": int(chunk_index),
            "data/file_index": int(file_index),
            "source_trial_dir": episode.source_trial_dir,
            "attempt": None if episode.attempt is None else int(episode.attempt),
            **episode.source_metadata,
            "task_completed": bool(episode.task_completed),
            "success": bool(episode.success),
            "excluded_from_training": bool(episode.excluded_from_training),
            "exclusion_reason": episode.exclusion_reason,
        }
        for camera_key, frames in episode.frames.items():
            if not frames:
                continue
            video_from_index = int(episode_video_from_index[camera_key])
            video_to_index = int(episode_video_from_index[camera_key] + len(frames) - 1)
            episode_row[f"videos/{camera_key}/chunk_index"] = int(chunk_index)
            episode_row[f"videos/{camera_key}/file_index"] = int(file_index)
            episode_row[f"videos/{camera_key}/from_index"] = video_from_index
            episode_row[f"videos/{camera_key}/to_index"] = video_to_index
            episode_row[f"videos/{camera_key}/from_timestamp"] = float(video_from_index / dataset_fps)
            episode_row[f"videos/{camera_key}/to_timestamp"] = float(video_to_index / dataset_fps)
            episode_row[f"{camera_key}.video_from_index"] = video_from_index
            episode_row[f"{camera_key}.video_to_index"] = video_to_index
            total_videos += 1
        episode_rows.append(episode_row)
        all_episode_rows.append(episode_row)
        all_episode_stats_rows.append(
            {
                "episode_index": int(episode.episode_index),
                "stats": _episode_numeric_stats(episode.rows),
            }
        )
        total_frames += len(episode.rows)
        total_episodes += 1
        episodes_in_chunk += 1
        print(
            f"[export] prepared episode={episode.episode_index} rows={len(episode.rows)} "
            f"chunk={chunk_index} episodes_in_chunk={episodes_in_chunk}",
            flush=True,
        )

        if episodes_in_chunk >= chunk_size:
            flush_chunk()
            print(f"[export] flushed chunk={chunk_index - 1}", flush=True)

    flush_chunk()

    tasks_records = [task_records_by_index[idx] for idx in sorted(task_records_by_index)]
    tasks_jsonl_path = output_root / "meta" / "tasks.jsonl"
    with tasks_jsonl_path.open("w", encoding="utf-8") as handle:
        for record in tasks_records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    pd.DataFrame(tasks_records).to_parquet(output_root / "meta" / "tasks.parquet", index=False, engine="pyarrow")

    episodes_jsonl_path = output_root / "meta" / "episodes.jsonl"
    with episodes_jsonl_path.open("w", encoding="utf-8") as handle:
        for record in all_episode_rows:
            handle.write(json.dumps(_jsonify(record), ensure_ascii=False) + "\n")

    episodes_stats_jsonl_path = output_root / "meta" / "episodes_stats.jsonl"
    with episodes_stats_jsonl_path.open("w", encoding="utf-8") as handle:
        for record in all_episode_stats_rows:
            handle.write(json.dumps(_jsonify(record), ensure_ascii=False) + "\n")

    stats = {
        feature_name: _finalize_running_stats(feature_stats)
        for feature_name, feature_stats in numeric_stats_running.items()
        if int(feature_stats["count"]) > 0
    }
    (output_root / "meta" / "stats.json").write_text(
        json.dumps(_jsonify(stats), indent=2),
        encoding="utf-8",
    )

    features: dict[str, Any] = {
        "action": {
            "dtype": "float32",
            "shape": [len(ACTION_NAMES)],
            "names": ACTION_NAMES,
        },
        "observation.state": {
            "dtype": "float32",
            "shape": [len(STATE_NAMES)],
            "names": STATE_NAMES,
        },
        "observation.eef_state": {
            "dtype": "float32",
            "shape": [len(EE_STATE_NAMES)],
            "names": EE_STATE_NAMES,
        },
        "reward": {"dtype": "float32", "shape": [1], "names": None},
        "done": {"dtype": "bool", "shape": [1], "names": None},
        "truncated": {"dtype": "bool", "shape": [1], "names": None},
        "timestamp": {"dtype": "float32", "shape": [1], "names": None, "fps": dataset_fps},
        "frame_index": {"dtype": "int64", "shape": [1], "names": None, "fps": dataset_fps},
        "episode_index": {"dtype": "int64", "shape": [1], "names": None, "fps": dataset_fps},
        "index": {"dtype": "int64", "shape": [1], "names": None, "fps": dataset_fps},
        "task_index": {"dtype": "int64", "shape": [1], "names": None, "fps": dataset_fps},
    }
    for camera_key, frame in first_frame_by_camera.items():
        features[camera_key] = _video_feature_info(frame, fps=dataset_fps, codec="h264")

    data_size_mb = round(_directory_size_bytes(output_root / "data") / (1024 * 1024), 3)
    video_size_mb = round(_directory_size_bytes(output_root / "videos") / (1024 * 1024), 3)

    info = {
        "codebase_version": "v3.0",
        "robot_type": robot_type,
        "total_episodes": total_episodes,
        "total_frames": total_frames,
        "total_tasks": len(task_to_index),
        "benchmarks": sorted({record.get("benchmark", "unknown") for record in task_records_by_index.values()}),
        "total_videos": total_videos,
        "chunks_size": chunk_size,
        "data_files_size_in_mb": data_size_mb,
        "video_files_size_in_mb": video_size_mb,
        "fps": dataset_fps,
        "include_excluded": include_excluded,
        "splits": {"train": f"0:{total_episodes}"},
        "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
        "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4"
        if first_frame_by_camera
        else None,
        "episodes_path": "meta/episodes/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
        "tasks_path": "meta/tasks.parquet",
        "features": features,
    }
    (output_root / "meta" / "info.json").write_text(
        json.dumps(_jsonify(info), indent=2),
        encoding="utf-8",
    )

    manifest = {
        "version": 1,
        "input_root": str(input_root.resolve()),
        "output_root": str(output_root.resolve()),
        "robot_type": robot_type,
        "fps": dataset_fps,
        "include_excluded": include_excluded,
        "num_episodes": total_episodes,
        "num_tasks": len(task_to_index),
        "num_frames": total_frames,
        "cameras": sorted(first_frame_by_camera.keys()),
    }
    (output_root / "manifest.json").write_text(
        json.dumps(_jsonify(manifest), indent=2),
        encoding="utf-8",
    )
    return manifest


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export CaP-X transition datasets to LeRobot v3 video dataset format.")
    parser.add_argument("--input-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--robot-type", default="franka")
    parser.add_argument("--fps", type=int, default=None, help="Override FPS. Defaults to inferred from timestamps.")
    parser.add_argument(
        "--downsample-fps",
        type=int,
        default=None,
        help="Downsample episode steps/frames to this FPS before export. Must be <= source FPS.",
    )
    parser.add_argument("--chunk-size", type=int, default=1000, help="Episodes per data/video shard.")
    parser.add_argument("--crf", type=int, default=28, help="H.264 CRF. Higher is smaller and lower quality.")
    parser.add_argument("--include-excluded", action="store_true", help="Include episodes marked as unsuitable for training.")
    return parser


def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()
    manifest = export_lerobot_dataset(
        input_root=args.input_root,
        output_root=args.output_root,
        robot_type=args.robot_type,
        fps=args.fps,
        downsample_fps=args.downsample_fps,
        chunk_size=args.chunk_size,
        crf=args.crf,
        include_excluded=args.include_excluded,
    )
    print(json.dumps(_jsonify(manifest), indent=2))


if __name__ == "__main__":
    main()
