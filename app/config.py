from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any


VALID_RESOLUTIONS = (384, 448, 512)
RESOLUTION_MODES = (
    (640, 384),
    (384, 256),
    (512, 512),
    (768, 512),
    (1024, 768),
)


@dataclass(slots=True)
class AppConfig:
    prompt: str = "dreamlike surreal painted architectural city, monumental arches, endless colonnades, floating domes, expressive oil brushwork, atmospheric perspective, luminous color"
    default_prompt: str = "dreamlike surreal painted architectural city, monumental arches, endless colonnades, floating domes, expressive oil brushwork, atmospheric perspective, luminous color"
    negative_prompt: str = "simple 3d primitives, greybox, blockout, low-poly render, UI, text"
    backend: str = "sd_turbo_stream"
    diffusion_resolution: int | str = "640x384"
    steps: int = 1
    seed: int = 12345
    seed_mode: str = "fixed"
    random_seed_on_launch: bool = False
    movement_speed: float = 9.0
    sprint_multiplier: float = 3.0
    mouse_sensitivity: float = 0.15
    img2img_strength: float = 1.0
    one_step_timestep: int = 750
    fixed_noise: bool = True
    previous_frame_weight: float = 0.0
    noise_persistence: float = 0.95
    edge_strength: float = 0.7
    edge_softness: float = 1.5
    depth_strength: float = 0.0
    reprojection: bool = True
    reprojection_strength: float = 0.35
    reprojection_max_translation: float = 1.0
    reprojection_max_rotation: float = 10.0
    display_sharpen: float = 0.3
    target_display_fps: int = 60
    conditioning_fps: int = 15
    debug_overlay: bool = True
    fullscreen: bool = False
    display_monitor: int = 0
    world_seed: int = 12345
    auto_resolution_fallback: bool = True
    warmup_passes: int = 3
    model_path: str = "models/sd_turbo"
    torch_compile: bool = False

    @classmethod
    def load(cls, path: Path) -> "AppConfig":
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise RuntimeError(f"Configuration file not found: {path}") from exc
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Invalid JSON in {path}: {exc}") from exc
        known = {field.name for field in fields(cls)}
        unknown = sorted(set(raw) - known)
        if unknown:
            raise RuntimeError(f"Unknown config keys: {', '.join(unknown)}")
        config = cls(**raw)
        config.validate()
        if config.random_seed_on_launch:
            config.seed = random.SystemRandom().randrange(0, 2**31)
        return config

    def validate(self) -> None:
        width, height = self.diffusion_size
        if (width, height) not in RESOLUTION_MODES and not (
            width == height and width in VALID_RESOLUTIONS
        ):
            raise RuntimeError(
                "diffusion_resolution must be 640x384, 384x256, 512x512, "
                "768x512, or 1024x768"
            )
        if not 1 <= self.steps <= 4:
            raise RuntimeError("steps must be between 1 and 4")
        if self.seed_mode not in ("fixed", "random_each_frame"):
            raise RuntimeError("seed_mode must be fixed or random_each_frame")
        if not 0.0 < self.img2img_strength <= 1.0:
            raise RuntimeError("img2img_strength must be in (0, 1]")
        if not 1 <= self.one_step_timestep <= 999:
            raise RuntimeError("one_step_timestep must be between 1 and 999")
        if not 0.0 <= self.previous_frame_weight <= 1.0:
            raise RuntimeError("previous_frame_weight must be in [0, 1]")
        if not 0.0 <= self.edge_softness <= 4.0:
            raise RuntimeError("edge_softness must be in [0, 4]")
        if not 0.0 <= self.edge_strength <= 1.0:
            raise RuntimeError("edge_strength must be in [0, 1]")
        if not 0.0 <= self.reprojection_strength <= 1.0:
            raise RuntimeError("reprojection_strength must be in [0, 1]")
        if self.reprojection_max_translation <= 0.0:
            raise RuntimeError("reprojection_max_translation must be positive")
        if self.reprojection_max_rotation <= 0.0:
            raise RuntimeError("reprojection_max_rotation must be positive")
        if not 0.0 <= self.display_sharpen <= 2.0:
            raise RuntimeError("display_sharpen must be in [0, 2]")
        if self.target_display_fps < 30:
            raise RuntimeError("target_display_fps must be at least 30")
        if self.display_monitor < 0:
            raise RuntimeError("display_monitor must be zero or greater")
        if not 1 <= self.conditioning_fps <= self.target_display_fps:
            raise RuntimeError("conditioning_fps must be between 1 and target_display_fps")
        if self.warmup_passes < 0:
            raise RuntimeError("warmup_passes must be zero or greater")

    def backend_dict(self, project_root: Path) -> dict[str, Any]:
        data = {field.name: getattr(self, field.name) for field in fields(self)}
        data["diffusion_width"], data["diffusion_height"] = self.diffusion_size
        data["project_root"] = project_root
        data["model_path"] = (project_root / self.model_path).resolve()
        return data

    @property
    def diffusion_size(self) -> tuple[int, int]:
        value = self.diffusion_resolution
        if isinstance(value, int):
            return value, value
        try:
            width_text, height_text = value.lower().split("x", 1)
            return int(width_text), int(height_text)
        except (AttributeError, TypeError, ValueError) as exc:
            raise RuntimeError(
                f"Invalid diffusion_resolution: {self.diffusion_resolution!r}"
            ) from exc

    @property
    def resolution_label(self) -> str:
        width, height = self.diffusion_size
        return f"{width}x{height}"


def configure_local_environment(project_root: Path, offline: bool = True) -> None:
    cache_root = project_root / "cache"
    values = {
        "HF_HOME": cache_root / "huggingface",
        "HUGGINGFACE_HUB_CACHE": cache_root / "huggingface" / "hub",
        "TRANSFORMERS_CACHE": cache_root / "huggingface" / "transformers",
        "TORCH_HOME": cache_root / "torch",
        "XDG_CACHE_HOME": cache_root,
    }
    for key, value in values.items():
        value.mkdir(parents=True, exist_ok=True)
        os.environ[key] = str(value)
    if offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["DIFFUSERS_OFFLINE"] = "1"
    os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    # This application is PyTorch-only. Prevent unrelated, possibly globally
    # installed TensorFlow/JAX stacks from being imported by Transformers.
    os.environ.setdefault("USE_TF", "0")
    os.environ.setdefault("USE_FLAX", "0")


def structure_lock_percent(config: AppConfig) -> float:
    """Return the user-facing proxy-composition retention percentage."""
    if config.steps == 1:
        return float(np_clip((1000 - config.one_step_timestep) / 10.0, 10.0, 75.0))
    return float(np_clip((1.0 - config.img2img_strength) * 100.0, 0.0, 75.0))


def np_clip(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))
