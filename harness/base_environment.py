from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple


class BaseEnvironment(ABC):
    """Abstract base class for HAB evaluation environments."""

    @abstractmethod
    def reset(self) -> Dict[str, Any]:
        """Start a new episode; return the initial observation."""
        ...

    @abstractmethod
    def step(self, action: str) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        """Execute action; return (observation, reward, done, info)."""
        ...

    @abstractmethod
    def get_final_state(self) -> Dict[str, Any]:
        """
        Return the normalized state dict consumed by evaluate_episode().
        Must include: task_id, run_id, signals, actions.
        """
        ...

    @abstractmethod
    def close(self) -> None:
        """Release resources."""
        ...

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
