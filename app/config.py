from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

# Generation modes cycled by F10. Every side is divisible by 8 so the latent
# grid is exact. 512x512 is the default: it is the square aspect the reference
# sequence uses, and at a 1-step UNet the cost is dominated by kernel launches
# rather than pixel count, so 384x256 measures only about 10% faster.
RESOLUTION_MODES = (
    (512, 512),
    (640, 384),
    (512, 384),
    (384, 256),
    (768, 512),
    (1024, 768),
)

# Live settings forwarded to the diffusion backend through request_settings.
BACKEND_SETTING_KEYS = (
    "steps",
    "guidance_scale",
    "timestep_min",
    "timestep_max",
    "instability",
    "guide_strength",
    "memory_match",
    "memory_match_std",
    "memory_leash",
    "depth_guide",
    "noise_walk_seconds",
    "noise_jitter",
    "prompt_walk_seconds",
    "feedback_reprojection",
)


@dataclass(slots=True)
class AppConfig:
    """Runtime settings for the app, the renderer and the latent-walk backend."""

    prompt: str = "corrupted 3D render, corrupted, eye level view inside a vast outdoor voxel landscape, topology unknown, shattered strata melting and regrowing, floating fragments, hard black shadows, high contrast, foggy, haunted"
    default_prompt: str = "corrupted 3D render, corrupted, eye level view inside a vast outdoor voxel landscape, topology unknown, shattered strata melting and regrowing, floating fragments, hard black shadows, high contrast, foggy, haunted"
    negative_prompt: str = "(worst quality, low quality: 1.4), text, lettering, logo, ui, hud, game controller, gamepad, joystick, toy, product photo, desk, table, monitor, keyboard, circuit board, interior, room, person, figure, cars, road markings, street lights"
    backend: str = "latent_walk"
    diffusion_resolution: str = "512x512"
    # Diffusion
    sampler: str = ""
    steps: int = 1
    guidance_scale: float = 2.0
    timestep_min: int = 640
    timestep_max: int = 720
    instability: float = 0.7
    guide_strength: float = 0.7
    # How hard the memory latent is pulled back to the guide's per-channel mean
    # and spread each frame. Below about 0.8 the feedback loop drifts.
    memory_match: float = 1.0
    # Share of that re-anchoring applied to the spread. 1.0 forces the memory to
    # the guide's contrast every frame, which flattens the darks into a waxy
    # mid-grey band; around 0.5 holds the mean while letting contrast breathe.
    memory_match_std: float = 0.5
    # Half-width, in latent standard deviations, of the band the memory is held
    # in around the guide. 0 removes the leash and the walk drifts off the proxy.
    memory_leash: float = 1.1
    # How much weaker the proxy pulls at the far plane than up close, 0..1.
    depth_guide: float = 0.5
    noise_walk_seconds: float = 5.0
    noise_jitter: float = 0.06
    prompt_walk_seconds: float = 6.0
    # Seconds between automatic library prompt changes. 0 disables auto-advance.
    prompt_auto_advance_seconds: float = 40.0
    feedback_reprojection: bool = False
    seed: int = 12345
    random_seed_on_launch: bool = False
    warmup_passes: int = 2
    model_path: str = "models/sd_turbo"
    taesd_path: str = "models/taesd"
    # Flight
    movement_speed: float = 8.0
    sprint_multiplier: float = 3.0
    mouse_sensitivity: float = 0.15
    world_seed: int = 12345
    # Display
    reprojection: bool = True
    reprojection_strength: float = 0.8
    reprojection_max_translation: float = 1.0
    reprojection_max_rotation: float = 10.0
    display_sharpen: float = 0.3
    target_display_fps: int = 60
    conditioning_fps: int = 15
    debug_overlay: bool = False
    prompt_caption: bool = False
    # Seconds of no input before the walker strolls on its own; 0 disables.
    autowalk_idle_seconds: float = 60.0
    fullscreen: bool = True
    display_monitor: int = 0
    auto_resolution_fallback: bool = True

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
        if self.diffusion_size not in RESOLUTION_MODES:
            modes = ", ".join(f"{w}x{h}" for w, h in RESOLUTION_MODES)
            raise RuntimeError(f"diffusion_resolution must be one of: {modes}")
        if self.backend not in ("latent_walk", "proxy_passthrough"):
            raise RuntimeError("backend must be latent_walk or proxy_passthrough")
        if self.sampler not in ("", "euler_x0", "lcm"):
            raise RuntimeError("sampler must be empty, euler_x0 or lcm")
        if not 1 <= self.steps <= 4:
            raise RuntimeError("steps must be between 1 and 4")
        if not 0.0 <= self.guidance_scale <= 8.0:
            raise RuntimeError("guidance_scale must be in [0, 8]")
        for key in ("timestep_min", "timestep_max"):
            if not 1 <= getattr(self, key) <= 999:
                raise RuntimeError(f"{key} must be between 1 and 999")
        if self.timestep_min > self.timestep_max:
            raise RuntimeError("timestep_min must not exceed timestep_max")
        for key in (
            "instability",
            "guide_strength",
            "memory_match",
            "memory_match_std",
            "noise_jitter",
        ):
            if not 0.0 <= getattr(self, key) <= 1.0:
                raise RuntimeError(f"{key} must be in [0, 1]")
        for key in (
            "noise_walk_seconds",
            "prompt_walk_seconds",
            "prompt_auto_advance_seconds",
        ):
            if getattr(self, key) < 0.0:
                raise RuntimeError(f"{key} must be zero or greater")
        if not 0.0 <= self.memory_leash <= 8.0:
            raise RuntimeError("memory_leash must be in [0, 8]")
        if not 0.0 <= self.depth_guide <= 1.0:
            raise RuntimeError("depth_guide must be in [0, 1]")
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
        if self.movement_speed <= 0.0:
            raise RuntimeError("movement_speed must be positive")
        if self.autowalk_idle_seconds < 0.0:
            raise RuntimeError("autowalk_idle_seconds must be zero or greater")

    def backend_dict(self, project_root: Path) -> dict[str, Any]:
        """Full settings dict handed to the backend, with model paths resolved."""
        data = {field.name: getattr(self, field.name) for field in fields(self)}
        data["diffusion_width"], data["diffusion_height"] = self.diffusion_size
        data["project_root"] = project_root
        data["model_path"] = (project_root / self.model_path).resolve()
        data["taesd_path"] = (project_root / self.taesd_path).resolve()
        return data

    def backend_settings(self) -> dict[str, Any]:
        """The live-tunable subset, for one request_settings call."""
        return {key: getattr(self, key) for key in BACKEND_SETTING_KEYS}

    @property
    def diffusion_size(self) -> tuple[int, int]:
        try:
            width_text, height_text = str(self.diffusion_resolution).lower().split("x", 1)
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
    """Point every model/library cache at the project folder, optionally offline."""
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
