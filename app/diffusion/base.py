from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from app.types import ConditioningFrame


class DiffusionBackend(ABC):
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
        temporal_state: dict[str, Any] | None = None,
    ) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def stats(self) -> dict[str, float | int | str]:
        raise NotImplementedError

    @abstractmethod
    def unload(self) -> None:
        raise NotImplementedError

    def set_resolution(self, width: int, height: int | None = None) -> None:
        raise NotImplementedError("This backend does not support changing resolution")

    def reseed(self, seed: int) -> None:
        raise NotImplementedError("This backend does not support reseeding")

    def set_seed_mode(self, mode: str) -> None:
        raise NotImplementedError("This backend does not support seed modes")

    def set_steps(self, steps: int) -> None:
        raise NotImplementedError("This backend does not support changing steps")

    def set_guidance_scale(self, guidance_scale: float) -> None:
        raise NotImplementedError("This backend does not support changing guidance scale")

    def set_one_step_timestep(self, timestep: int) -> None:
        raise NotImplementedError("This backend does not support changing the timestep")

    def set_edge_softness(self, softness: float) -> None:
        raise NotImplementedError("This backend does not support edge softening")

    def set_img2img_strength(self, strength: float) -> None:
        raise NotImplementedError("This backend does not support changing img2img strength")

    def set_edge_strength(self, strength: float) -> None:
        raise NotImplementedError("This backend does not support changing edge strength")

    def set_noise_persistence(self, persistence: float) -> None:
        raise NotImplementedError("This backend does not support temporal noise")
