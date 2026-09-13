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

# Fog fades fully before the edge of the streamed world.
DEFAULT_FOG_DISTANCE = 196.0
MIN_FOG_DISTANCE = 40.0
MAX_FOG_DISTANCE = 300.0


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
    "depth_shade",
    "guide_wobble",
    "noise_walk_seconds",
    "noise_jitter",
    "prompt_walk_seconds",
    "feedback_reprojection",
)


@dataclass(slots=True)
class AppConfig:
    """Runtime settings for the app, the renderer and the latent-walk backend."""

    prompt: str = "corrupted 3D render, corrupted, topology unknown, shattered strata melting and regrowing, floating fragments, eye level view inside a vast outdoor voxel landscape, hard black shadows, high contrast, foggy, haunted"
    default_prompt: str = "corrupted 3D render, corrupted, topology unknown, shattered strata melting and regrowing, floating fragments, eye level view inside a vast outdoor voxel landscape, hard black shadows, high contrast, foggy, haunted"
    negative_prompt: str = "(worst quality, low quality: 1.4), text, lettering, logo, ui, hud, game controller, gamepad, joystick, toy, product photo, desk, table, monitor, keyboard, interior, room, furniture, bedroom, bed, showroom, window, houseplant"
    backend: str = "latent_walk"
    diffusion_resolution: str = "512x512"
    # Diffusion
    sampler: str = ""
    steps: int = 1
    guidance_scale: float = 1.8
    # The timestep, not the guide strength, decides how much of the proxy's
    # layout survives: at 640-720 only about a sixth of the signal entering the
    # UNet is the guide, and the output's edges landed on the proxy's at an SSIM\n    # of 0.05 whatever the guide did. 460-600 gives the walk room to introduce\n    # saturated structure while the picture still follows the proxy.
    timestep_min: int = 460
    timestep_max: int = 600
    instability: float = 0.72
    guide_strength: float = 0.82
    # How hard the memory latent is pulled back to the guide's per-channel mean
    # and spread each frame. Below about 0.8 the feedback loop drifts.
    memory_match: float = 1.0
    # Share of that re-anchoring applied to the spread. 1.0 forces the memory to
    # the guide's contrast every frame, which flattens the darks into a waxy
    # mid-grey band; around 0.5 holds the mean while letting contrast breathe.
    memory_match_std: float = 0.5
    # Half-width, in latent standard deviations, of the band the memory is held
    # in around the guide. 0 removes the leash and the walk drifts off the proxy;
    # 0.7 measured a little tighter to the proxy than 1.1 without flattening it.
    memory_leash: float = 0.7
    # How the proxy's pull varies with distance, -1..1. Positive holds the near
    # field hardest and lets the horizon hallucinate; negative inverts that;
    # 0 weights the frame flat. It is worth about 0.03 of conformance on its own
    # and measures neutral once depth_shade is carrying the depth cue, so it
    # stays as a per-family look knob rather than a structural one.
    depth_guide: float = 0.5
    # Luminance ramp baked into the proxy before the encode, -1..1. Positive
    # makes near geometry bright and the far field dark; negative reverses it;
    # 0 leaves the render as the shader painted it. Negative wins because the
    # proxy already fogs distance toward a bright sky, so darkening the near
    # field deepens a cue the picture is reading rather than fighting it. Past
    # about -0.8 the foreground goes to silhouette and the look flattens.
    depth_shade: float = -0.6
    # One-sided guide-strength wobble amplitude, 0..1. 0 holds the anchor
    # steady, which is what keeps a wall solid frame to frame; the wobble
    # measured neutral for conformance and only ever raised the average anchor,
    # which guide_strength does more legibly.
    guide_wobble: float = 0.0
    # Short keyframes plus a lively jitter. A slow noise walk let SD-Turbo lock
    # onto whatever object it first read in a noise blob (reliably a gamepad)
    # and hold it for the whole keyframe; refreshing the field every couple of
    # seconds breaks that lock and is the main source of within-walk variety.
    noise_walk_seconds: float = 5.0
    noise_jitter: float = 0.16
    prompt_walk_seconds: float = 6.0
    # Seconds between automatic library prompt changes. 0 disables auto-advance.
    prompt_auto_advance_seconds: float = 20.0
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
    # World distance where structures fully fade into the atmosphere. Keep the
    # maximum inside the streamed window so distant chunk edges stay hidden.
    fog_distance: float = DEFAULT_FOG_DISTANCE
    player_trail: bool = True
    # Separate projector map. Older configurations keep their single window.
    journey_map: bool = False
    map_capture_distance: float = 12.0
    map_idle_seconds: float = 10.0
    map_image_format: str = "webp"
    map_export_svg: bool = False
    # Upload is opt-in; credentials are read from the environment, never this file.
    map_sync_enabled: bool = False
    map_cache_gib: float = 5.0
    map_min_free_gib: float = 1.0
    target_display_fps: int = 60
    conditioning_fps: int = 15
    debug_overlay: bool = False
    prompt_caption: bool = False
    # Seconds of no input before the walker strolls on its own; 0 disables.
    autowalk_idle_seconds: float = 60.0
    fullscreen: bool = True
    display_monitor: int = 0
    map_display_monitor: int | None = None
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
            "guide_wobble",
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
        for key in ("depth_guide", "depth_shade"):
            if not -1.0 <= getattr(self, key) <= 1.0:
                raise RuntimeError(f"{key} must be in [-1, 1]")
        if not 0.0 <= self.reprojection_strength <= 1.0:
            raise RuntimeError("reprojection_strength must be in [0, 1]")
        if self.reprojection_max_translation <= 0.0:
            raise RuntimeError("reprojection_max_translation must be positive")
        if self.reprojection_max_rotation <= 0.0:
            raise RuntimeError("reprojection_max_rotation must be positive")
        if not 0.0 <= self.display_sharpen <= 2.0:
            raise RuntimeError("display_sharpen must be in [0, 2]")
        if not MIN_FOG_DISTANCE <= self.fog_distance <= MAX_FOG_DISTANCE:
            raise RuntimeError(f"fog_distance must be in [{MIN_FOG_DISTANCE:g}, {MAX_FOG_DISTANCE:g}]")
        if self.target_display_fps < 30:
            raise RuntimeError("target_display_fps must be at least 30")
        if self.display_monitor < 0:
            raise RuntimeError("display_monitor must be zero or greater")
        if self.map_display_monitor is not None and (type(self.map_display_monitor) is not int or self.map_display_monitor < 0):
            raise RuntimeError("map_display_monitor must be null or a nonnegative integer")
        if not 1 <= self.conditioning_fps <= self.target_display_fps:
            raise RuntimeError("conditioning_fps must be between 1 and target_display_fps")
        if self.warmup_passes < 0:
            raise RuntimeError("warmup_passes must be zero or greater")
        if self.movement_speed <= 0.0:
            raise RuntimeError("movement_speed must be positive")
        if self.autowalk_idle_seconds < 0.0:
            raise RuntimeError("autowalk_idle_seconds must be zero or greater")
        if not isinstance(self.journey_map, bool):
            raise RuntimeError("journey_map must be true or false")
        if not 0.1 <= self.map_capture_distance <= 10000.0:
            raise RuntimeError("map_capture_distance must be between 0.1 and 10000")
        if not 0.0 <= self.map_idle_seconds <= 300.0:
            raise RuntimeError("map_idle_seconds must be between 0 and 300")
        if self.map_image_format not in ("webp", "png"):
            raise RuntimeError("map_image_format must be webp or png")
        for key in ("map_export_svg", "map_sync_enabled"):
            if not isinstance(getattr(self, key), bool):
                raise RuntimeError(f"{key} must be true or false")
        for key in ("map_cache_gib", "map_min_free_gib"):
            if not 0.0 < getattr(self, key) <= 100000.0:
                raise RuntimeError(f"{key} must be positive and at most 100000 GiB")

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

