from __future__ import annotations

import sys
import types

from capx.integrations import libero as lib_mod


class FakeEnv:
    def __init__(self) -> None:
        self._t = 0

    def reset(self, seed=None, options=None):  # noqa: D401, ARG002
        self._t = 0
        return {"image": None}

    def seed(self, s: int) -> None:  # noqa: D401, ARG002
        pass

    def set_init_state(self, state):  # noqa: D401, ARG002
        pass

    def step(self, action):  # noqa: D401, ARG002
        self._t += 1
        done = self._t >= 2
        return {"image": None}, 1.0 if done else 0.0, done, {"t": self._t}


def _make_fake_suite() -> object:
    class FakeSuite:
        def get_task(self, task_id: int):  # noqa: D401
            return types.SimpleNamespace(
                problem_folder="pf",
                bddl_file="file.bddl",
                init_states_file="init.pt",
                language="lang",
            )

        def get_task_init_states(self, task_id: int):  # noqa: D401
            return [None]

    return FakeSuite()


def test_load_libero_task_root_layout(monkeypatch: object, tmp_path) -> None:
    def get_benchmark_dict(help=False):  # noqa: D401, ARG001
        return {"libero_10": lambda: _make_fake_suite()}

    task_root = tmp_path / "pf"
    task_root.mkdir()
    (task_root / "file.bddl").write_text("(:language lang)", encoding="utf-8")

    class FakeOffEnv(FakeEnv):
        def __init__(self, **kwargs):  # noqa: D401, ARG002
            super().__init__()

    libero_pkg = types.ModuleType("libero")
    benchmark_mod = types.ModuleType("libero.benchmark")
    envs_mod = types.ModuleType("libero.envs")
    utils_mod = types.ModuleType("libero.utils")
    benchmark_mod.get_benchmark_dict = get_benchmark_dict  # type: ignore[attr-defined]
    envs_mod.OffScreenRenderEnv = FakeOffEnv  # type: ignore[attr-defined]
    utils_mod.get_libero_path = lambda name: str(tmp_path)  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "libero", libero_pkg)
    monkeypatch.setitem(sys.modules, "libero.benchmark", benchmark_mod)
    monkeypatch.setitem(sys.modules, "libero.envs", envs_mod)
    monkeypatch.setitem(sys.modules, "libero.utils", utils_mod)

    handle = lib_mod.load_libero_task("libero_10", task_id=0)
    obs, info = handle.reset(seed=0)
    assert isinstance(obs, dict)
    obs, rew, done, info = handle.step([0.0] * 7)
    assert done in (True, False)


def test_load_libero_task_nested_layout(monkeypatch: object, tmp_path) -> None:
    def get_benchmark_dict(help=False):  # noqa: D401, ARG001
        return {"libero_goal": lambda: _make_fake_suite()}

    task_root = tmp_path / "pf"
    task_root.mkdir()
    (task_root / "file.bddl").write_text("(:language nested layout)", encoding="utf-8")

    class FakeOffEnv(FakeEnv):
        def __init__(self, **kwargs):  # noqa: D401, ARG002
            super().__init__()

    libero_pkg = types.ModuleType("libero")
    nested_pkg = types.ModuleType("libero.libero")
    benchmark_mod = types.ModuleType("libero.libero.benchmark")
    envs_mod = types.ModuleType("libero.libero.envs")
    utils_mod = types.ModuleType("libero.libero.utils")
    benchmark_mod.get_benchmark_dict = get_benchmark_dict  # type: ignore[attr-defined]
    envs_mod.OffScreenRenderEnv = FakeOffEnv  # type: ignore[attr-defined]
    utils_mod.get_libero_path = lambda name: str(tmp_path)  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "libero", libero_pkg)
    monkeypatch.setitem(sys.modules, "libero.libero", nested_pkg)
    monkeypatch.setitem(sys.modules, "libero.libero.benchmark", benchmark_mod)
    monkeypatch.setitem(sys.modules, "libero.libero.envs", envs_mod)
    monkeypatch.setitem(sys.modules, "libero.libero.utils", utils_mod)

    handle = lib_mod.load_libero_task("libero_goal", task_id=0)
    obs, info = handle.reset(seed=0)
    assert isinstance(obs, dict)
