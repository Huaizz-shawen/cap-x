"""Compatibility helpers for mixed robosuite controller config versions.

Newer robosuite releases expose composite controller loading at
`robosuite.controllers.composite.composite_controller_factory`.
Older releases only provide `load_controller_config`.

This module provides a single loader used by simulator wrappers so they can run
on both variants.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _load_json(path: str) -> dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Controller config not found: {path}")
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def _flatten_basic_composite_cfg(cfg: dict[str, Any]) -> dict[str, Any]:
    """Extract a legacy single-arm config from BASIC composite format.

    Expected shape:
      {
        "type": "BASIC",
        "body_parts": {"arms": {"right": {...} | "left": {...}}}
      }
    """
    body_parts = cfg.get("body_parts", {})
    arms = body_parts.get("arms", {}) if isinstance(body_parts, dict) else {}

    arm_cfg: dict[str, Any] | None = None
    if isinstance(arms, dict):
        # Prefer right arm when available, else left, else first entry.
        if isinstance(arms.get("right"), dict):
            arm_cfg = arms["right"]
        elif isinstance(arms.get("left"), dict):
            arm_cfg = arms["left"]
        else:
            for v in arms.values():
                if isinstance(v, dict):
                    arm_cfg = v
                    break

    if arm_cfg is None:
        raise ValueError("Cannot flatten composite controller config: no arm config found")

    # Legacy path does not use nested gripper config in controller dict.
    flat = dict(arm_cfg)
    flat.pop("gripper", None)
    return flat


def load_controller_config_compat(controller: str) -> dict[str, Any]:
    """Load controller config compatible with both new and old robosuite APIs."""
    try:
        from robosuite.controllers.composite.composite_controller_factory import (
            load_composite_controller_config,
        )

        return load_composite_controller_config(controller=controller)
    except Exception:
        # Fallback for old robosuite releases without composite controllers.
        cfg = _load_json(controller)
        if isinstance(cfg, dict) and cfg.get("type") == "BASIC" and "body_parts" in cfg:
            return _flatten_basic_composite_cfg(cfg)
        return cfg
