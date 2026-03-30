from __future__ import annotations

import importlib
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"

STANDARD_LIBERO_SUITES = (
    "libero_10",
    "libero_object",
    "libero_spatial",
    "libero_goal",
)


def _maybe_add_vendor_roots() -> None:
    here = Path(__file__).resolve()
    vendor_roots = (
        here.parents[2] / "third_party" / "LIBERO-PRO",
        here.parents[2] / "third_party" / "LIBERO-PRO" / "libero",
    )
    for vendor_root in vendor_roots:
        vendor_root_str = str(vendor_root)
        if vendor_root.is_dir() and vendor_root_str not in sys.path:
            sys.path.append(vendor_root_str)


def _import_first(module_names: tuple[str, ...]) -> Any:
    last_error: Exception | None = None
    for module_name in module_names:
        try:
            return importlib.import_module(module_name)
        except Exception as exc:  # pragma: no cover - exercised via fallback order
            last_error = exc
    raise ModuleNotFoundError(
        f"Could not import any of the expected LIBERO modules: {module_names}"
    ) from last_error


def _import_libero_modules() -> tuple[Any, Any, Any]:
    _maybe_add_vendor_roots()
    benchmark_module = _import_first(("libero.benchmark", "libero.libero.benchmark"))
    envs_module = _import_first(("libero.envs", "libero.libero.envs"))
    path_module_names = ("libero.utils", "libero", "libero.libero.utils", "libero.libero")
    get_libero_path = None
    last_path_module = None
    for module_name in path_module_names:
        try:
            path_module = importlib.import_module(module_name)
        except Exception:
            continue
        last_path_module = path_module
        get_libero_path = getattr(path_module, "get_libero_path", None)
        if get_libero_path is not None:
            break
    if get_libero_path is None:
        module_name = "<unimportable>" if last_path_module is None else last_path_module.__name__
        raise AttributeError(f"Module {module_name} does not export get_libero_path")
    return benchmark_module, envs_module.OffScreenRenderEnv, get_libero_path


def get_libero_benchmark_dict(*, help: bool = False) -> dict[str, Any]:
    benchmark_module, _, _ = _import_libero_modules()
    return benchmark_module.get_benchmark_dict(help=help)


def _resolve_libero_asset_path(
    asset_key: str, problem_folder: str, filename: str, get_libero_path: Any
) -> str:
    asset_dir_by_key = {
        "bddl_files": "bddl_files",
        "init_states": "init_files",
    }
    if asset_key not in asset_dir_by_key:
        raise KeyError(f"Unsupported LIBERO asset key: {asset_key}")

    candidates: list[Path] = []
    try:
        root_path = Path(get_libero_path(asset_key))
        candidates.append(root_path / problem_folder / filename)
    except Exception:
        pass

    here = Path(__file__).resolve()
    vendor_root = here.parents[2] / "third_party" / "LIBERO-PRO" / "libero" / "libero"
    candidates.append(vendor_root / asset_dir_by_key[asset_key] / problem_folder / filename)

    for candidate in candidates:
        if candidate.exists():
            return str(candidate)

    searched = ", ".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(
        f"Could not locate LIBERO asset {filename!r} for suite folder {problem_folder!r}. "
        f"Searched: {searched}"
    )


@dataclass
class LiberoHandle:
    env: Any
    suite_name: str
    task_id: int
    task_language: str
    init_states: Any

    def reset(self, seed: int | None = None) -> tuple[Any, dict[str, Any]]:
        self.env.seed(seed)
        obs = self.env.reset()
        if self.init_states is not None:
            self.env.set_init_state(self.init_states[0])
        return obs, {}

    def step(self, action: list[float]) -> tuple[Any, float, bool, dict[str, Any]]:
        obs, reward, done, info = self.env.step(action)
        return obs, float(reward), bool(done), info


def _extract_language_from_bddl(bddl_path: str) -> str | None:
    try:
        with open(bddl_path, "r") as f:
            content = f.read()
        match = re.search(r"\(:language\s+(.*?)\)", content, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()
    except Exception as e:
        print(f"Warning: Could not extract language from {bddl_path}: {e}")
    return None


def load_libero_task(
    suite_name: str,
    task_id: int,
    cam_w: int = 128,
    cam_h: int = 128,
    controller: str = "OSC_POSE",
    horizon: int = 1000,
    control_freq: int = 20,
    camera_depths: bool = True,
) -> LiberoHandle:
    """Load a LIBERO task using OffScreenRenderEnv.

    Reference: https://github.com/Lifelong-Robot-Learning/LIBERO
    """
    try:
        benchmark, OffScreenRenderEnv, get_libero_path = _import_libero_modules()
    except Exception as e:  # pragma: no cover - optional dependency
        raise ModuleNotFoundError(
            "LIBERO not available; add submodule or run `uv sync --extra libero`."
        ) from e

    # setting help=True will print the available benchmarks
    benchmark_dict = benchmark.get_benchmark_dict(help=False)
    if suite_name not in benchmark_dict:
        available = ", ".join(sorted(benchmark_dict.keys()))
        raise KeyError(f"Unknown LIBERO suite {suite_name!r}. Available suites: {available}")
    task_suite = benchmark_dict[suite_name]()
    task = task_suite.get_task(task_id)

    bddl_file_path = _resolve_libero_asset_path(
        "bddl_files",
        task.problem_folder,
        task.bddl_file,
        get_libero_path,
    )

    env_args = {
        "bddl_file_name": bddl_file_path,
        "camera_heights": cam_h,
        "camera_widths": cam_w,
        "controller": controller,
        "horizon": horizon,
        "control_freq": control_freq,
        "camera_depths": camera_depths,
    }
    env = OffScreenRenderEnv(**env_args)
    env.seed(0)

    # Try to extract language from BDDL file directly
    task_language = _extract_language_from_bddl(bddl_file_path)
    if not task_language:
        task_language = task.language

    # Handle init states path resolution
    # Libero's get_task_init_states uses get_libero_path("init_states") internally
    # We need to manually load them if the default path fails
    try:
        init_states = task_suite.get_task_init_states(task_id)
        print(f"Loaded init states for task {task_id} in suite {suite_name}")
    except Exception:
        print(f"Warning: Could not load init states for task {task_id} in suite {suite_name}")
        init_states_path = _resolve_libero_asset_path(
            "init_states",
            task.problem_folder,
            task.init_states_file,
            get_libero_path,
        )
        import torch

        init_states = torch.load(init_states_path)

    handle = LiberoHandle(
        env=env,
        suite_name=suite_name,
        task_id=task_id,
        task_language=task_language,
        init_states=init_states,
    )
    return handle
