from __future__ import annotations

import importlib.util
from argparse import Namespace
from pathlib import Path


def _load_pipeline_module():
    path = (
        Path(__file__).resolve().parents[1]
        / "skills"
        / "capx-eap-data-collection"
        / "scripts"
        / "capx_eap_pipeline.py"
    )
    spec = importlib.util.spec_from_file_location("capx_eap_pipeline", path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_load_api_ports_from_config(tmp_path):
    module = _load_pipeline_module()
    repo_root = tmp_path
    cfg_dir = repo_root / "env_configs" / "libero"
    cfg_dir.mkdir(parents=True)
    cfg_path = cfg_dir / "test.yaml"
    cfg_path.write_text(
        """
api_servers:
  - port: 8116
  - port: 8115
  - host: 127.0.0.1
"""
    )

    ports = module._load_api_ports(repo_root, str(cfg_path.relative_to(repo_root)))

    assert ports == [8115, 8116]


def test_strip_tmux_flags():
    module = _load_pipeline_module()
    argv = [
        "collect",
        "--config-path",
        "env_configs/libero/franka_libero_goal_1.yaml",
        "--tmux-session",
        "goal1",
        "--tmux-log-file=outputs/tmux.log",
        "--tmux-replace-existing",
        "--total-trials",
        "10",
    ]

    stripped = module._strip_tmux_flags(argv)

    assert stripped == [
        "collect",
        "--config-path",
        "env_configs/libero/franka_libero_goal_1.yaml",
        "--total-trials",
        "10",
    ]


def test_build_collect_env_defaults():
    module = _load_pipeline_module()
    env = module._build_collect_env(
        Namespace(
            request_timeout_s=None,
            request_max_attempts=None,
            request_max_retry_walltime_s=None,
            request_retry_initial_s=None,
            request_retry_max_sleep_s=None,
        )
    )

    assert env["CAPX_MODEL_REQUEST_TIMEOUT_S"] == str(module.DEFAULT_MODEL_REQUEST_TIMEOUT_S)
    assert env["CAPX_MODEL_RETRY_MAX_ATTEMPTS"] == str(module.DEFAULT_MODEL_RETRY_MAX_ATTEMPTS)


def test_build_validate_env_uses_writable_tmp_cache(tmp_path):
    module = _load_pipeline_module()
    env = module._build_validate_env(tmp_path / "hf_cache")

    assert env["HF_HOME"].startswith(str(tmp_path))
    assert env["HF_DATASETS_CACHE"].startswith(str(tmp_path))
    assert env["HUGGINGFACE_HUB_CACHE"].startswith(str(tmp_path))
    assert Path(env["HF_HOME"]).exists()
    assert Path(env["HF_DATASETS_CACHE"]).exists()
