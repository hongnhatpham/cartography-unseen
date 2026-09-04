from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Any

import numpy as np

from app.types import ConditioningFrame


class DiffusionBackend(ABC):
    """Contract between the diffusion worker and one image generator.

    Live tuning travels through a single generic channel, ``apply_settings``, so
    a new knob costs one config key and one overlay binding rather than a
    request/applied/set trio on every layer.
    """

    @abstractmethod
    def load(self, config: dict[str, Any]) -> None:
        raise NotImplementedError

    @abstractmethod
    def warmup(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def set_prompt(self, prompt: str, negative_prompt: str = "") -> None:
        raise NotImplementedError

    @abstractmethod
    def generate(
        self,
        conditioning: ConditioningFrame,
        previous_frame: np.ndarray | None = None,
    ) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def stats(self) -> dict[str, float | int | str]:
        raise NotImplementedError

    @abstractmethod
    def unload(self) -> None:
        raise NotImplementedError

    def apply_settings(self, settings: Mapping[str, Any]) -> None:
        """Apply a batch of live settings. Keys the backend has no use for are ignored."""
        return

    def set_resolution(self, width: int, height: int | None = None) -> None:
        raise NotImplementedError("This backend does not support changing resolution")

    def reseed(self, seed: int) -> None:
        raise NotImplementedError("This backend does not support reseeding")
