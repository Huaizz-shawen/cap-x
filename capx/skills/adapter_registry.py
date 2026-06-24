"""JSON-backed registry for environment-specific skill adapters.

Adapters are intentionally small, explicit records: a reusable task template,
its parameters, and the validation result that justifies reusing it. The
registry is append/update only and does not promote adapters into runtime
defaults by itself.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any


SCHEMA_VERSION = 1


@dataclass
class SkillAdapterRecord:
    adapter_id: str
    environment: str
    task: str
    robot: str
    template: str
    params: dict[str, Any]
    score: float
    success: bool
    validation: dict[str, Any]
    assumptions: list[str] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    schema_version: int = SCHEMA_VERSION

    def to_jsonable(self) -> dict[str, Any]:
        return _jsonable(asdict(self))


class SkillAdapterRegistry:
    def __init__(self, path: str | Path):
        self.path = Path(path)

    def load(self) -> dict[str, Any]:
        if not self.path.exists():
            return {"schema_version": SCHEMA_VERSION, "adapters": []}
        data = json.loads(self.path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError(f"Adapter registry must be a JSON object: {self.path}")
        data.setdefault("schema_version", SCHEMA_VERSION)
        data.setdefault("adapters", [])
        if not isinstance(data["adapters"], list):
            raise ValueError(f"Adapter registry 'adapters' must be a list: {self.path}")
        return data

    def save(self, data: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(_jsonable(data), indent=2, sort_keys=True), encoding="utf-8")

    def upsert(self, record: SkillAdapterRecord) -> None:
        data = self.load()
        record_data = record.to_jsonable()
        adapters = data["adapters"]
        for index, existing in enumerate(adapters):
            if isinstance(existing, dict) and existing.get("adapter_id") == record.adapter_id:
                adapters[index] = record_data
                break
        else:
            adapters.append(record_data)
        adapters.sort(key=lambda item: (not bool(item.get("success")), -float(item.get("score", 0.0)), str(item.get("adapter_id", ""))))
        self.save(data)

    def best(
        self,
        *,
        environment: str,
        task: str,
        robot: str,
        template: str | None = None,
        require_success: bool = True,
    ) -> dict[str, Any] | None:
        candidates: list[dict[str, Any]] = []
        for item in self.load()["adapters"]:
            if not isinstance(item, dict):
                continue
            if item.get("environment") != environment or item.get("task") != task or item.get("robot") != robot:
                continue
            if template is not None and item.get("template") != template:
                continue
            if require_success and not bool(item.get("success")):
                continue
            candidates.append(item)
        if not candidates:
            return None
        candidates.sort(key=lambda item: (-float(item.get("score", 0.0)), str(item.get("adapter_id", ""))))
        return candidates[0]


def _jsonable(value: Any) -> Any:
    try:
        import numpy as np
    except Exception:  # pragma: no cover - numpy is optional for this helper.
        np = None  # type: ignore[assignment]
    if isinstance(value, dict):
        return {str(key): _jsonable(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if np is not None and isinstance(value, np.ndarray):
        return value.tolist()
    if np is not None and isinstance(value, np.generic):
        return value.item()
    try:
        json.dumps(value)
        return value
    except TypeError:
        return repr(value)
