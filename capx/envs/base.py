from collections.abc import Callable
from functools import lru_cache
from typing import Any, SupportsFloat, TypeVar, abstractmethod

from gymnasium import Env

ObsType = TypeVar("ObsType")
ActType = TypeVar("ActType")


class BaseEnv(Env):
    """
    Base environment class for low level control environments.
    This is a generic environment class for low level mujoco / simulator control environments.
    It is a subclass of the Gymnasium Env class.
    """

    privileged: bool = False
    max_steps: int = 999999

    @abstractmethod
    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[ObsType, dict[str, Any]]:
        """
        Resets the environment to an initial internal state, returning an initial observation and info.
        Args:
            seed: The seed to reset the environment with.
            options: The options to reset the environment with.
        Returns:
            tuple: A tuple containing the observation and info.
        """
        raise NotImplementedError

    @abstractmethod
    def step(self, action: ActType) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        """
        Takes a step in the environment with the given action.
        Here we assume the action is the low level control actions (joint position, gripper position, etc.)
        Args:
            action: The action to take in the environment.
        Returns:
            tuple: A tuple containing the observation, reward, terminated, truncated, and info.
        """
        raise NotImplementedError

    @abstractmethod
    def get_observation(self) -> ObsType:
        """
        Gets the observation of the environment.
        Returns:
            ObsType: The observation of the environment.
        """
        raise NotImplementedError

    @abstractmethod
    def compute_reward(self) -> SupportsFloat:
        """
        Computes the reward of the environment.
        Returns:
            SupportsFloat: The reward of the environment.
        """
        raise NotImplementedError

    @abstractmethod
    def task_completed(self) -> bool:
        """
        Checks if the task is completed.
        Returns:
            bool: True if the task is completed, False otherwise.
        """
        raise NotImplementedError

    def capture_state(self) -> dict[str, Any]:
        """Capture a restorable simulator snapshot.

        Low-level environments that support deterministic rewind should override
        this. The default implementation makes the capability explicit at runtime
        without forcing every environment subclass to implement it immediately.
        """
        raise NotImplementedError(f"{type(self).__name__} does not support capture_state()")

    def restore_state(self, state: dict[str, Any]) -> None:
        """Restore a previously captured simulator snapshot."""
        raise NotImplementedError(f"{type(self).__name__} does not support restore_state()")

    def get_reset_state(self) -> dict[str, Any] | None:
        """Return the canonical post-reset state, if the environment tracks one."""
        return None


# Use user's BaseEnv for low-level envs

_ENV_FACTORIES: dict[str, Callable[[], BaseEnv]] = {}


def register_env(name: str, factory: Callable[[], BaseEnv]) -> None:
    _ENV_FACTORIES[name] = factory


@lru_cache(maxsize=256)
def get_env(name: str, privileged: bool = False, enable_render: bool = False, viser_debug: bool = False) -> BaseEnv:
    if name not in _ENV_FACTORIES:
        raise KeyError(f"Environment '{name}' not registered")
    return _ENV_FACTORIES[name](privileged=privileged, enable_render=enable_render, viser_debug=viser_debug)


def list_envs() -> list[str]:
    return list(_ENV_FACTORIES.keys())


__all__ = [
    "BaseEnv",
    "register_env",
    "get_env",
    "list_envs",
]
