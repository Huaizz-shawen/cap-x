from __future__ import annotations

import copy
import gzip
import json
import pickle
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


def _copy_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.copy()
    if isinstance(value, dict):
        return {k: _copy_value(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_copy_value(v) for v in value]
    if isinstance(value, tuple):
        return tuple(_copy_value(v) for v in value)
    return copy.deepcopy(value)


def create_transition_dataset(
    *,
    trial: int,
    config_path: str | None = None,
    task_prompt: str | None = None,
) -> dict[str, Any]:
    return {
        "version": 1,
        "trial": trial,
        "config_path": config_path,
        "task_prompt": task_prompt,
        "initial_observation": None,
        "transitions": [],
    }


def set_initial_observation(transition_dataset: dict[str, Any], observation: Any) -> None:
    transition_dataset["initial_observation"] = _copy_value(observation)


def append_transition(
    transition_dataset: dict[str, Any],
    *,
    timestamp_s: float,
    wall_time_s: float,
    sim_step_count: int | None,
    action: Any,
    observation: Any,
    reward: float | None,
    done: bool | None,
    truncated: bool | None,
    source: str,
    action_context: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
) -> None:
    transition_dataset["transitions"].append(
        {
            "transition_idx": len(transition_dataset["transitions"]),
            "timestamp_s": float(timestamp_s),
            "wall_time_s": float(wall_time_s),
            "sim_step_count": sim_step_count,
            "action": _copy_value(action),
            "observation": _copy_value(observation),
            "reward": reward,
            "done": done,
            "truncated": truncated,
            "source": source,
            "action_context": _copy_value(action_context) if action_context is not None else None,
            "metadata": _copy_value(metadata) if metadata is not None else {},
        }
    )


def save_transition_dataset(trial_dir: Path, transition_dataset: dict[str, Any]) -> None:
    dataset_dir = trial_dir / "transition_dataset"
    dataset_dir.mkdir(parents=True, exist_ok=True)

    payload_path = dataset_dir / "data.pkl.gz"
    with gzip.open(payload_path, "wb") as handle:
        pickle.dump(transition_dataset, handle, protocol=pickle.HIGHEST_PROTOCOL)

    metadata = {
        "version": transition_dataset.get("version", 1),
        "trial": transition_dataset.get("trial"),
        "config_path": transition_dataset.get("config_path"),
        "task_prompt": transition_dataset.get("task_prompt"),
        "transition_count": len(transition_dataset.get("transitions", [])),
        "has_initial_observation": transition_dataset.get("initial_observation") is not None,
        "payload_file": payload_path.name,
    }
    (dataset_dir / "metadata.json").write_text(
        json.dumps(_jsonify(metadata), indent=2),
        encoding="utf-8",
    )
