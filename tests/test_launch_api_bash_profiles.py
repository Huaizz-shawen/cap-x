import importlib.util
import sys
import types
from pathlib import Path

import pytest


def _load_launch_module():
    tyro = types.ModuleType("tyro")
    tyro.cli = lambda *args, **kwargs: None

    launch_utils = types.ModuleType("capx.utils.launch_utils")
    launch_utils._load_config = lambda args: None

    capx_pkg = types.ModuleType("capx")
    capx_pkg.__path__ = []
    utils_pkg = types.ModuleType("capx.utils")
    utils_pkg.__path__ = []

    sys.modules["tyro"] = tyro
    sys.modules["capx"] = capx_pkg
    sys.modules["capx.utils"] = utils_pkg
    sys.modules["capx.utils.launch_utils"] = launch_utils

    launch_path = Path(__file__).resolve().parents[1] / "capx" / "envs" / "launch.py"
    spec = importlib.util.spec_from_file_location("capx_envs_launch_test", launch_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_prepare_cli_args_expands_bash_file_before_user_args(tmp_path: Path) -> None:
    launch = _load_launch_module()
    script = tmp_path / "provider.sh"
    script.write_text(
        "#!/usr/bin/env bash\n"
        "printf '%s\\n' \\\n"
        "  --server-url \\\n"
        "  'https://example.test/v1/chat/completions' \\\n"
        "  --api-key \\\n"
        "  'secret key' \\\n"
        "  --model \\\n"
        "  'azure/openai/gpt-5.1'\n",
        encoding="utf-8",
    )

    args = launch._prepare_cli_args(
        [
            "--api-bash-file",
            str(script),
            "--config-path",
            "env_configs/libero/franka_libero_goal_1.yaml",
            "--model",
            "override-model",
        ]
    )

    assert args == [
        "--server-url",
        "https://example.test/v1/chat/completions",
        "--api-key",
        "secret key",
        "--model",
        "azure/openai/gpt-5.1",
        "--config-path",
        "env_configs/libero/franka_libero_goal_1.yaml",
        "--model",
        "override-model",
    ]


def test_prepare_cli_args_rejects_empty_bash_output(tmp_path: Path) -> None:
    launch = _load_launch_module()
    script = tmp_path / "empty.sh"
    script.write_text("#!/usr/bin/env bash\n", encoding="utf-8")

    with pytest.raises(ValueError, match="produced no arguments"):
        launch._prepare_cli_args(["--api-bash-file", str(script), "--config-path", "dummy.yaml"])


def test_resolve_api_bash_script_uses_default_profile_dir(monkeypatch, tmp_path: Path) -> None:
    launch = _load_launch_module()
    profile = tmp_path / "custom_provider.sh"
    profile.write_text("#!/usr/bin/env bash\nprintf '%s\\n' --api-key secret\n", encoding="utf-8")

    monkeypatch.setattr(launch, "_default_api_bash_dir", lambda: tmp_path)

    resolved = launch._resolve_api_bash_script("custom_provider", None)

    assert resolved == profile.resolve()
