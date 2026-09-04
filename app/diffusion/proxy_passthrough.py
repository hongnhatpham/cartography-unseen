from __future__ import annotations

from time import perf_counter
from typing import Any

import numpy as np

from app.diffusion.base import DiffusionBackend
from app.types import ConditioningFrame


class ProxyPassthroughBackend(DiffusionBackend):
    """Dependency-light diagnostic backend; never selected as an implicit fallback."""

    def __init__(self) -> None:
        self.width = 512
        self.height = 512
        self.last_ms = 0.0
        self.prompt = ""

    def load(self, config: dict[str, Any]) -> None:
        self.width = int(config.get("diffusion_width", 512))
        self.height = int(config.get("diffusion_height", self.width))

    def warmup(self) -> None:
        return

    def set_prompt(self, prompt: str, negative_prompt: str = "") -> None:
        self.prompt = prompt

    def generate(
        self,
        conditioning: ConditioningFrame,
        previous_frame: np.ndarray | None = None,
    ) -> np.ndarray:
        started = perf_counter()
        source = conditioning.rgb.astype(np.float32)
        edge = conditioning.edges.astype(np.float32)[:, :, None] / 255.0
        # A deliberately obvious diagnostic treatment, not a diffusion substitute.
        graded = source * np.array([1.08, 0.93, 1.12], dtype=np.float32)
        graded += edge * np.array([42.0, 18.0, 54.0], dtype=np.float32)
        if previous_frame is not None:
            graded = graded * 0.85 + previous_frame.astype(np.float32) * 0.15
        self.last_ms = (perf_counter() - started) * 1000.0
        return np.clip(graded, 0, 255).astype(np.uint8)

    def stats(self) -> dict[str, float | int | str]:
        return {
            "backend": "proxy_passthrough",
            "inference_ms": self.last_ms,
            "vram_allocated_gb": 0.0,
            "vram_reserved_gb": 0.0,
            "resolution": f"{self.width}x{self.height}",
            "steps": 0,
        }

    def unload(self) -> None:
        return

    def set_resolution(self, width: int, height: int | None = None) -> None:
        self.width = int(width)
        self.height = int(height if height is not None else width)

    def reseed(self, seed: int) -> None:
        return
