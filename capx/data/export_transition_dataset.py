from __future__ import annotations

import argparse
import gzip
import json
import pickle
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np


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


def _load_transition_dataset(path: Path) -> dict[str, Any]:
    with gzip.open(path, "rb") as handle:
        return pickle.load(handle)


def _load_trajectory_metadata(trial_dir: Path) -> dict[str, Any]:
    trajectory_path = trial_dir / "trajectory" / "metadata.json"
    if not trajectory_path.exists():
        return {}
    return json.loads(trajectory_path.read_text(encoding="utf-8"))


def _load_trial_summary(trial_dir: Path) -> dict[str, Any]:
    data = _load_trajectory_metadata(trial_dir)
    trial_complete = next(
        (event for event in data.get("events", []) if event.get("event_type") == "trial_complete"),
        None,
    )
    return trial_complete or {}


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


def _extract_initial_modalities(
    initial_observation: dict[str, Any],
    *,
    include_images: bool,
    include_depth: bool,
) -> dict[str, np.ndarray]:
    arrays: dict[str, np.ndarray] = {}
    if not isinstance(initial_observation, dict):
        return arrays

    if "robot_joint_pos" in initial_observation:
        arrays["initial_robot_joint_pos"] = np.asarray(initial_observation["robot_joint_pos"])
    if "robot_cartesian_pos" in initial_observation:
        arrays["initial_robot_cartesian_pos"] = np.asarray(initial_observation["robot_cartesian_pos"])

    for camera_name, prefix in (("agentview", "agentview"), ("robot0_eye_in_hand", "wrist")):
        camera = initial_observation.get(camera_name)
        if not isinstance(camera, dict):
            continue
        images = camera.get("images")
        if not isinstance(images, dict):
            continue
        if include_images and "rgb" in images:
            arrays[f"{prefix}_rgb_initial"] = np.asarray(images["rgb"])
        if include_depth and "depth" in images:
            arrays[f"{prefix}_depth_initial"] = np.asarray(images["depth"])
    return arrays


def _stack_if_present(values: list[Any], *, dtype: Any | None = None) -> np.ndarray | None:
    if not values:
        return None
    if any(value is None for value in values):
        return None
    arrays = [np.asarray(value, dtype=dtype) if dtype is not None else np.asarray(value) for value in values]
    return np.stack(arrays, axis=0)


def _build_episode_arrays(
    transition_dataset: dict[str, Any],
    *,
    include_images: bool,
    include_depth: bool,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    transitions = transition_dataset.get("transitions", [])
    if not transitions:
        raise ValueError("Transition dataset is empty")

    arrays: dict[str, np.ndarray] = {
        "timestamps_s": np.asarray([transition["timestamp_s"] for transition in transitions], dtype=np.float32),
        "wall_time_s": np.asarray([transition["wall_time_s"] for transition in transitions], dtype=np.float64),
        "sim_step_count": np.asarray([transition["sim_step_count"] for transition in transitions], dtype=np.int32),
        "actions": np.stack([np.asarray(transition["action"], dtype=np.float32) for transition in transitions], axis=0),
        "rewards": np.asarray(
            [0.0 if transition["reward"] is None else transition["reward"] for transition in transitions],
            dtype=np.float32,
        ),
        "dones": np.asarray([bool(transition["done"]) for transition in transitions], dtype=bool),
        "truncated": np.asarray([bool(transition["truncated"]) for transition in transitions], dtype=bool),
        "robot_joint_pos": np.stack(
            [np.asarray(transition["observation"]["robot_joint_pos"], dtype=np.float32) for transition in transitions],
            axis=0,
        ),
        "robot_cartesian_pos": np.stack(
            [np.asarray(transition["observation"]["robot_cartesian_pos"], dtype=np.float32) for transition in transitions],
            axis=0,
        ),
    }
    arrays.update(
        _extract_initial_modalities(
            transition_dataset.get("initial_observation", {}),
            include_images=include_images,
            include_depth=include_depth,
        )
    )

    low_level_obs = [transition["observation"]["low_level_observation"] for transition in transitions]
    if include_images:
        agentview_rgb = _stack_if_present([obs.get("agentview_image") for obs in low_level_obs], dtype=np.uint8)
        wrist_rgb = _stack_if_present([obs.get("robot0_eye_in_hand_image") for obs in low_level_obs], dtype=np.uint8)
        if agentview_rgb is not None:
            arrays["agentview_rgb"] = agentview_rgb
        if wrist_rgb is not None:
            arrays["wrist_rgb"] = wrist_rgb
    if include_depth:
        agentview_depth = _stack_if_present([obs.get("agentview_depth") for obs in low_level_obs], dtype=np.float32)
        wrist_depth = _stack_if_present([obs.get("robot0_eye_in_hand_depth") for obs in low_level_obs], dtype=np.float32)
        if agentview_depth is not None:
            arrays["agentview_depth"] = agentview_depth
        if wrist_depth is not None:
            arrays["wrist_depth"] = wrist_depth

    action_context_names = [
        transition.get("action_context", {}).get("action_name", "none")
        if transition.get("action_context") is not None
        else "none"
        for transition in transitions
    ]
    sources = [transition.get("source", "unknown") for transition in transitions]
    step_metadata = {
        "sources": sources,
        "source_counts": dict(Counter(sources)),
        "action_context_names": action_context_names,
        "action_context_counts": dict(Counter(action_context_names)),
        "action_context": [transition.get("action_context") for transition in transitions],
        "per_step_metadata": [transition.get("metadata", {}) for transition in transitions],
    }
    return arrays, step_metadata


def export_transition_dataset(
    transition_dataset_path: Path,
    *,
    output_root: Path,
    episode_idx: int,
    include_images: bool,
    include_depth: bool,
    include_excluded: bool,
) -> dict[str, Any]:
    trial_dir = transition_dataset_path.parent.parent
    transition_dataset = _load_transition_dataset(transition_dataset_path)
    trajectory_metadata = _load_trajectory_metadata(trial_dir)
    excluded_from_training, exclusion_reason = _exclude_from_training(
        transition_dataset,
        trajectory_metadata,
    )
    if excluded_from_training and not include_excluded:
        raise ValueError(f"Excluded from training ({exclusion_reason})")
    arrays, step_metadata = _build_episode_arrays(
        transition_dataset,
        include_images=include_images,
        include_depth=include_depth,
    )
    trial_summary = _load_trial_summary(trial_dir)

    episode_id = f"episode_{episode_idx:06d}"
    episodes_dir = output_root / "episodes"
    episodes_dir.mkdir(parents=True, exist_ok=True)
    npz_path = episodes_dir / f"{episode_id}.npz"
    metadata_path = episodes_dir / f"{episode_id}.json"

    np.savez_compressed(npz_path, **arrays)

    episode_metadata = {
        "episode_id": episode_id,
        "source_trial_dir": str(trial_dir.resolve()),
        "source_transition_dataset": str(transition_dataset_path.resolve()),
        "config_path": transition_dataset.get("config_path"),
        "trial": transition_dataset.get("trial"),
        "attempt": transition_dataset.get("attempt"),
        "task_prompt": transition_dataset.get("task_prompt"),
        "excluded_from_training": excluded_from_training,
        "exclusion_reason": exclusion_reason,
        "num_steps": int(arrays["actions"].shape[0]),
        "action_dim": int(arrays["actions"].shape[1]),
        "included_modalities": sorted(arrays.keys()),
        "step_metadata": _jsonify(step_metadata),
        "trial_summary": _jsonify(trial_summary),
        "npz_file": npz_path.name,
    }
    metadata_path.write_text(json.dumps(_jsonify(episode_metadata), indent=2), encoding="utf-8")
    return episode_metadata


def export_directory(
    *,
    input_root: Path,
    output_root: Path,
    include_images: bool,
    include_depth: bool,
    include_excluded: bool = False,
) -> dict[str, Any]:
    transition_paths = sorted(input_root.glob("**/transition_dataset/data.pkl.gz"))
    if not transition_paths:
        raise FileNotFoundError(f"No transition datasets found under {input_root}")

    output_root.mkdir(parents=True, exist_ok=True)
    episodes_jsonl_path = output_root / "episodes.jsonl"
    manifest_path = output_root / "manifest.json"

    exported_episodes: list[dict[str, Any]] = []
    with episodes_jsonl_path.open("w", encoding="utf-8") as jsonl_handle:
        for episode_idx, transition_path in enumerate(transition_paths, start=1):
            try:
                episode_metadata = export_transition_dataset(
                    transition_path,
                    output_root=output_root,
                    episode_idx=episode_idx,
                    include_images=include_images,
                    include_depth=include_depth,
                    include_excluded=include_excluded,
                )
            except ValueError as exc:
                if "Excluded from training" in str(exc):
                    continue
                raise
            exported_episodes.append(episode_metadata)
            jsonl_handle.write(json.dumps(_jsonify(episode_metadata), ensure_ascii=False) + "\n")

    manifest = {
        "version": 1,
        "input_root": str(input_root.resolve()),
        "output_root": str(output_root.resolve()),
        "num_episodes": len(exported_episodes),
        "include_images": include_images,
        "include_depth": include_depth,
        "include_excluded": include_excluded,
        "episodes_jsonl": episodes_jsonl_path.name,
    }
    manifest_path.write_text(json.dumps(_jsonify(manifest), indent=2), encoding="utf-8")
    return manifest


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export CaP-X transition datasets into training-ready episode files.")
    parser.add_argument("--input-root", required=True, type=Path, help="Root directory containing trial_* outputs.")
    parser.add_argument("--output-root", required=True, type=Path, help="Directory to write exported episodes.")
    parser.add_argument("--include-images", action="store_true", help="Export RGB image arrays into the episode npz.")
    parser.add_argument("--include-depth", action="store_true", help="Export depth image arrays into the episode npz.")
    parser.add_argument("--include-excluded", action="store_true", help="Include episodes marked as unsuitable for training.")
    return parser


def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()
    manifest = export_directory(
        input_root=args.input_root,
        output_root=args.output_root,
        include_images=args.include_images,
        include_depth=args.include_depth,
        include_excluded=args.include_excluded,
    )
    print(json.dumps(_jsonify(manifest), indent=2))


if __name__ == "__main__":
    main()
