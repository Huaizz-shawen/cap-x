from __future__ import annotations

import copy
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


def create_trajectory_buffer(
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
        "snapshots": [],
        "events": [],
        "_snapshot_payloads": {},
    }


def _extract_snapshot_summary(state: dict[str, Any]) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "gripper_fraction": state.get("gripper_fraction"),
        "reward": state.get("current_reward"),
        "done": state.get("current_done"),
    }

    current_obs = state.get("current_obs")
    if isinstance(current_obs, dict):
        robot_joint_pos = current_obs.get("robot_joint_pos")
        robot_cartesian_pos = current_obs.get("robot_cartesian_pos")
        if robot_joint_pos is not None:
            summary["robot_joint_pos"] = _jsonify(robot_joint_pos)
        if robot_cartesian_pos is not None:
            summary["robot_cartesian_pos"] = _jsonify(robot_cartesian_pos)

    return summary


def append_snapshot(
    trajectory_data: dict[str, Any],
    *,
    env: Any | None = None,
    state: dict[str, Any] | None = None,
    snapshot_type: str,
    code_block_idx: int | None = None,
    label: str | None = None,
    phase_candidates: list[dict[str, Any]] | None = None,
    phase_tags: list[str] | None = None,
) -> str | None:
    if state is None:
        if env is None or not hasattr(env, "capture_state"):
            return None
        state = env.capture_state()

    snapshot_id = f"snap_{len(trajectory_data['snapshots']):04d}"
    snapshot_meta = {
        "snapshot_id": snapshot_id,
        "snapshot_type": snapshot_type,
        "code_block_idx": code_block_idx,
        "label": label,
        "step_count": state.get("step_count"),
        "sim_step_count": state.get("sim_step_count"),
        "file": f"snapshots/{snapshot_id}.pkl",
        "summary": _extract_snapshot_summary(state),
        "phase_candidates": _jsonify(phase_candidates) if phase_candidates is not None else [],
        "phase_tags": list(phase_tags) if phase_tags is not None else [],
    }
    trajectory_data["snapshots"].append(snapshot_meta)
    trajectory_data["_snapshot_payloads"][snapshot_id] = copy.deepcopy(state)
    return snapshot_id


def append_event(trajectory_data: dict[str, Any], event_type: str, **kwargs: Any) -> None:
    event = {
        "event_idx": len(trajectory_data["events"]),
        "event_type": event_type,
    }
    for key, value in kwargs.items():
        event[key] = _jsonify(value)
    trajectory_data["events"].append(event)


def save_trajectory_artifacts(trial_dir: Path, trajectory_data: dict[str, Any]) -> None:
    trajectory_dir = trial_dir / "trajectory"
    snapshots_dir = trajectory_dir / "snapshots"
    snapshots_dir.mkdir(parents=True, exist_ok=True)

    snapshot_payloads = trajectory_data.get("_snapshot_payloads", {})
    for snapshot_meta in trajectory_data.get("snapshots", []):
        snapshot_id = snapshot_meta["snapshot_id"]
        payload = snapshot_payloads.get(snapshot_id)
        if payload is None:
            continue
        with open(snapshots_dir / f"{snapshot_id}.pkl", "wb") as handle:
            pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)

    metadata = {
        key: value
        for key, value in trajectory_data.items()
        if key != "_snapshot_payloads"
    }
    (trajectory_dir / "metadata.json").write_text(
        json.dumps(_jsonify(metadata), indent=2),
        encoding="utf-8",
    )
