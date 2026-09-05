"""Realtime latent-walk backend: raw UNet + TAESD, no diffusers pipeline.

The proxy renderer only suggests a composition. The picture itself comes from a
continuous walk through latent space: a slerp between seeded noise keyframes, a
slerp between CLIP embeddings, and a memory latent (``x0_prev``) that carries the
previous frame's prediction forward. Everything stays on the GPU as fp16; the
only host transfer per frame is the proxy upload and one uint8 readback.
"""

from __future__ import annotations

import json
import logging
import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

from app.diffusion.base import DiffusionBackend
from app.types import ConditioningFrame
from app.utils.timing import ExponentialAverage

# Wall-clock periods of the instability oscillators. All three are mutually
# prime-ish so the timestep and the guide strength never breathe in lockstep,
# and the slow timestep tone is what carries a long loop through several looks
# instead of shuttling it back and forth on one 13 s cycle.
_BREATH_SECONDS = 13.0
_SLOW_BREATH_SECONDS = 47.0
_WOBBLE_SECONDS = 29.0
# Peak one-sided guide-strength excursion at instability 1.0.
_GUIDE_WOBBLE = 0.28
# Extra breathing amplitude at instability 1.0, as a fraction of the half-range.
# The result is still clamped into [timestep_min, timestep_max]; the overshoot
# only makes the walk dwell at the hot and cool ends rather than sweep past them.
_BREATH_REACH = 0.6
# AR(1) coefficient of the jitter field, so the jitter itself drifts instead of
# flickering independently every frame.
_JITTER_PERSISTENCE = 0.9
# Consistency-model data scale used by the LCM scheduler boundary conditions.
_SIGMA_DATA = 0.5
# Frames discarded after the memory latent is dropped (a Space reset, a reseed,
# a prompt hard cut). Without them the first published frame is conditioned on
# the raw proxy alone, and a large flat pale plane in it is completed as a game
# controller on a desk however the negative prompt is worded. These passes run
# at timestep_max so the memory the walk resumes from is hallucination.
RESET_WARMUP_FRAMES = 2


def classifier_free_guidance_enabled(guidance_scale: float) -> bool:
    """CFG costs a second UNet batch, so it only switches on above one."""
    return guidance_scale > 1.0


def clamp01(value: float) -> float:
    """Clamp to the unit interval."""
    return min(max(float(value), 0.0), 1.0)


# ---------------------------------------------------------------------------
# Interpolation primitives (torch and numpy both work; no device syncs per frame)
# ---------------------------------------------------------------------------


def _to_float32(value: Any) -> Any:
    return value.float() if hasattr(value, "float") else value.astype("float32")


def _inner_product(left: Any, right: Any) -> float:
    """Reduce in float32; fp16 embedding norms overflow the half range."""
    same = left is right
    left = _to_float32(left)
    right = left if same else _to_float32(right)
    return float((left.reshape(-1) * right.reshape(-1)).sum())


# Slerp coefficients peak at 1/sin(theta). fp16 embeddings carry outlier
# dimensions near +-60, so an arc whose coefficients exceed this bound is walked
# linearly instead of overflowing the half range.
_MAX_SLERP_COEFFICIENT = 4.0


@dataclass(frozen=True, slots=True)
class SlerpArc:
    """Frozen geometry of one interpolation arc between two tensors.

    ``prepare_slerp`` pays the three reductions (two norms and one dot product)
    once; every later ``at`` call is two Python scalars and one fused blend, so a
    walk costs no device-to-host synchronisation per frame. ``sin_theta`` of 0.0
    marks an arc that must be blended linearly.
    """

    source: Any
    target: Any
    theta: float
    sin_theta: float

    def at(self, t: float) -> Any:
        t = min(max(float(t), 0.0), 1.0)
        if t <= 0.0:
            return self.source
        if t >= 1.0:
            return self.target
        if self.sin_theta <= 0.0:
            return self.source * (1.0 - t) + self.target * t
        return self.source * (math.sin((1.0 - t) * self.theta) / self.sin_theta) + self.target * (
            math.sin(t * self.theta) / self.sin_theta
        )


def prepare_slerp(source: Any, target: Any) -> SlerpArc | None:
    """Measure the shortest arc between two tensors once, ahead of the walk.

    Works on torch tensors and numpy arrays. Degenerate, near-parallel and
    near-antiparallel pairs, and any pair whose slerp coefficients would exceed
    ``_MAX_SLERP_COEFFICIENT``, are marked for a linear blend so the whole arc
    stays finite and continuous.
    """
    if source is None or target is None:
        return None
    source_norm = math.sqrt(max(_inner_product(source, source), 0.0))
    target_norm = math.sqrt(max(_inner_product(target, target), 0.0))
    if source_norm < 1e-6 or target_norm < 1e-6:
        return SlerpArc(source, target, 0.0, 0.0)
    cosine = min(max(_inner_product(source, target) / (source_norm * target_norm), -1.0), 1.0)
    theta = math.acos(cosine)
    sin_theta = math.sin(theta)
    # Below a right angle both coefficients stay under 1; past it the larger one
    # reaches 1/sin(theta) somewhere inside the walk.
    peak = 1.0 / sin_theta if theta > math.pi / 2 and sin_theta > 0.0 else 1.0
    if sin_theta < 1e-4 or peak > _MAX_SLERP_COEFFICIENT:
        return SlerpArc(source, target, 0.0, 0.0)
    return SlerpArc(source, target, theta, sin_theta)


def slerp_embeddings(source: Any, target: Any, t: float) -> Any:
    """Spherically interpolate two tensors along the shortest arc."""
    arc = prepare_slerp(source, target)
    return target if arc is None else arc.at(t)


def blend_correlated_noise(previous: Any, fresh: Any, persistence: float) -> Any:
    """Blend fresh noise into a persistent field without changing its variance."""
    if previous is None or previous.shape != fresh.shape:
        return fresh
    persistence = clamp01(persistence)
    fresh_weight = math.sqrt(max(0.0, 1.0 - persistence * persistence))
    return previous * persistence + fresh * fresh_weight


# Reduce a NCHW latent over batch and space, leaving one figure per channel.
_STAT_REDUCE = (0, 2, 3)
_STAT_EPS = 1e-4


def _channel_stats(value: Any) -> tuple[Any, Any]:
    """Per-channel mean and variance of an NCHW latent (torch or numpy)."""
    kwargs = (
        {"dim": _STAT_REDUCE, "keepdim": True}
        if hasattr(value, "dim")
        else {"axis": _STAT_REDUCE, "keepdims": True}
    )
    mean = value.mean(**kwargs)
    return mean, (value * value).mean(**kwargs) - mean * mean


def match_latent_statistics(
    memory: Any, reference: Any, strength: float = 1.0, std_strength: float = 1.0
) -> Any:
    """Pull ``memory``'s per-channel mean and spread toward ``reference``'s.

    The memory latent is re-fed into its own prediction every frame, so its DC
    term and contrast random-walk with nothing restoring them. That drift is the
    engine of the feedback collapse: hue saturates, contrast climbs, and the
    picture leaves the proxy for whatever flat high-contrast attractor the
    prompt names. Re-anchoring the four channel statistics to the guide costs
    two reductions and leaves all the spatial structure of the memory intact.

    ``strength`` 0 disables the whole correction, 1 moves the mean all the way
    to the guide's. ``std_strength`` scales how much of that correction reaches
    the *spread*: the guide is a flat-shaded proxy, so matching its standard
    deviation every frame clamps the picture into the proxy's narrow luminance
    band and the darks never form. Below 1 the mean stays anchored while
    contrast is free to breathe.
    """
    strength = clamp01(strength)
    if strength <= 0.0:
        return memory
    spread_strength = strength * clamp01(std_strength)
    memory32 = _to_float32(memory)
    memory_mean, memory_var = _channel_stats(memory32)
    reference_mean, reference_var = _channel_stats(_to_float32(reference))
    scale = (reference_var.clip(min=0.0) ** 0.5 + _STAT_EPS) / (
        memory_var.clip(min=0.0) ** 0.5 + _STAT_EPS
    )
    blended = (memory32 - memory_mean) * (1.0 + (scale - 1.0) * spread_strength) + (
        memory_mean + (reference_mean - memory_mean) * strength
    )
    if hasattr(blended, "to"):
        return blended.to(memory.dtype)
    return blended.astype(memory.dtype)


def _cast_like(value: Any, reference: Any) -> Any:
    """Cast to the reference's dtype for torch tensors and numpy arrays alike."""
    if hasattr(value, "to"):
        return value.to(reference.dtype)
    return value.astype(reference.dtype)


def leash_to_guide(memory: Any, guide: Any, spread: float) -> Any:
    """Clamp the memory latent into a band ``spread`` standard deviations wide
    around the guide.

    Matching the channel statistics stops the memory drifting globally but not
    locally: it can still grow isolated high-contrast blobs, and the sampler
    eventually resolves those into glyphs, logos and objects that have nothing
    to do with the proxy. A per-element leash bounds how far any one part of the
    picture may leave the proxy, so the world keeps melting and regrowing
    without becoming a different image. ``spread`` 0 disables the leash; small
    values approach a proxy passthrough.
    """
    if spread <= 0.0:
        return memory
    _, guide_variance = _channel_stats(_to_float32(guide))
    radius = _cast_like(guide_variance.clip(min=0.0) ** 0.5 * float(spread), memory)
    return memory.clip(min=guide - radius, max=guide + radius)


# ---------------------------------------------------------------------------
# Scheduler maths, read from the model's own scheduler config
# ---------------------------------------------------------------------------


def alphas_cumprod_from_scheduler(config: Mapping[str, Any]) -> np.ndarray:
    """Rebuild the training noise schedule from a diffusers scheduler config."""
    steps = int(config.get("num_train_timesteps", 1000))
    schedule = str(config.get("beta_schedule", "scaled_linear"))
    trained = config.get("trained_betas")
    if trained is not None:
        betas = np.asarray(trained, dtype=np.float64)
    elif schedule in ("scaled_linear", "linear"):
        start = float(config.get("beta_start", 0.00085))
        end = float(config.get("beta_end", 0.012))
        if schedule == "scaled_linear":
            betas = np.linspace(start**0.5, end**0.5, steps, dtype=np.float64) ** 2
        else:
            betas = np.linspace(start, end, steps, dtype=np.float64)
    else:
        raise RuntimeError(f"Unsupported beta_schedule: {schedule}")
    return np.cumprod(1.0 - betas)


def infer_sampler(config: Mapping[str, Any]) -> str:
    """Pick the step parameterisation from the scheduler class name."""
    name = str(config.get("_class_name", ""))
    return "lcm" if "LCM" in name or "Consistency" in name else "euler_x0"


def add_noise(sample: Any, noise: Any, alpha_prod: float) -> Any:
    """Forward-diffuse a clean latent to the given cumulative alpha."""
    return sample * math.sqrt(alpha_prod) + noise * math.sqrt(max(0.0, 1.0 - alpha_prod))


def predict_x0(
    sample: Any, model_output: Any, alpha_prod: float, prediction_type: str = "epsilon"
) -> Any:
    """Solve the UNet output back to the clean latent it implies."""
    root_alpha = math.sqrt(alpha_prod)
    root_beta = math.sqrt(max(0.0, 1.0 - alpha_prod))
    if prediction_type == "v_prediction":
        return sample * root_alpha - model_output * root_beta
    if prediction_type == "sample":
        return model_output
    return (sample - model_output * root_beta) / max(root_alpha, 1e-6)


def lcm_scalings(timestep: float, timestep_scaling: float = 10.0) -> tuple[float, float]:
    """LCM boundary-condition scalings ``(c_skip, c_out)`` for one timestep."""
    scaled = float(timestep) * float(timestep_scaling)
    denominator = scaled**2 + _SIGMA_DATA**2
    return _SIGMA_DATA**2 / denominator, scaled / math.sqrt(denominator)


def breathing_timestep(
    elapsed: float, minimum: int, maximum: int, instability: float
) -> int:
    """Breathe the sampling timestep across the configured range.

    Two tones, a fast one and a slow one, so the walk wanders between the hot
    and cool ends of the range over a minute rather than oscillating on a fixed
    13 s beat. Instability scales both the amplitude and how long the walk
    dwells at each end; the value never leaves ``[minimum, maximum]``.
    """
    low, high = (minimum, maximum) if minimum <= maximum else (maximum, minimum)
    middle = (low + high) * 0.5
    reach = clamp01(instability)
    amplitude = (high - low) * 0.5 * reach * (1.0 + _BREATH_REACH * reach)
    phase = 0.65 * math.sin(2.0 * math.pi * elapsed / _BREATH_SECONDS) + 0.35 * math.sin(
        2.0 * math.pi * elapsed / _SLOW_BREATH_SECONDS
    )
    value = min(max(middle + amplitude * phase, float(low)), float(high))
    return int(round(min(max(value, 1.0), 999.0)))


def wobbled_guide_strength(
    elapsed: float, guide_strength: float, instability: float
) -> float:
    """Wobble how hard the proxy pulls the walk back toward itself.

    The excursion is one-sided: ``guide_strength`` is the weakest anchoring that
    ever happens and the wobble can only tighten it. A two-sided wobble spent
    part of every cycle below the configured anchor, and those troughs were
    where the walk let go of the proxy and drifted into interiors and glyphs.
    Instability still buys chaos through the breathing timestep.

    The period is long (29 s) on purpose: each anchoring level has to hold long
    enough for a look to settle, so a several-hundred-frame loop drifts through
    distinct states instead of shimmering between them.
    """
    phase = 0.5 + 0.5 * math.sin(2.0 * math.pi * elapsed / _WOBBLE_SECONDS)
    return clamp01(guide_strength + _GUIDE_WOBBLE * clamp01(instability) * phase)


# ---------------------------------------------------------------------------
# The two walks
# ---------------------------------------------------------------------------


class NoiseWalk:
    """Slerp between seeded noise keyframes on a wall clock, plus AR(1) jitter.

    ``sample`` returns a fresh unit-variance noise tensor. Screen-fixed noise
    that changes slowly is what makes the walk read as one continuous world
    rather than as per-frame static.
    """

    def __init__(
        self, sample: Callable[[], Any], clock: Callable[[], float] = perf_counter
    ) -> None:
        self._sample = sample
        self._clock = clock
        self.seconds = 20.0
        self.jitter = 0.0
        self._arc: SlerpArc | None = None
        self._started = 0.0
        self._t = 0.0
        self._current: Any = None
        self._jitter_state: Any = None

    @property
    def t(self) -> float:
        """Progress along the current keyframe arc, 0..1."""
        return self._t

    def reset(self) -> None:
        """Drop the walk state; the next frame starts from fresh keyframes."""
        self._arc = None
        self._current = None
        self._jitter_state = None
        self._t = 0.0

    def value(self) -> Any:
        """Noise for this frame, advancing the arc and the jitter field."""
        if self._arc is None:
            source = self._current if self._current is not None else self._sample()
            self._arc = prepare_slerp(source, self._sample())
            self._started = self._clock()
            self._t = 0.0
        arc = self._arc
        assert arc is not None
        if self.seconds <= 0.0:
            progress = 1.0
        else:
            progress = clamp01((self._clock() - self._started) / self.seconds)
        self._t = progress
        noise = arc.at(progress)
        if progress >= 1.0:
            self._current = arc.target
            self._arc = None
        if self.jitter > 0.0:
            self._jitter_state = blend_correlated_noise(
                self._jitter_state, self._sample(), _JITTER_PERSISTENCE
            )
            noise = blend_correlated_noise(self._jitter_state, noise, self.jitter)
        return noise


class PromptWalk:
    """Wall-clock slerp of CLIP embeddings toward the most recent prompt.

    A retarget mid-walk departs from the embedding currently on screen, so the
    image never jumps back toward the prompt before last.
    """

    def __init__(self, clock: Callable[[], float] = perf_counter) -> None:
        self._clock = clock
        self.seconds = 0.0
        self.positive: Any = None
        self.negative: Any = None
        self._arc: SlerpArc | None = None
        self._negative_arc: SlerpArc | None = None
        self._target: Any = None
        self._target_negative: Any = None
        self._started = 0.0
        self._t = 1.0

    @property
    def t(self) -> float:
        """Progress along the current prompt arc, 0..1."""
        return self._t

    def retarget(self, positive: Any, negative: Any) -> bool:
        """Aim at new embeddings; returns True when it was a hard cut."""
        self.advance()
        if self.seconds <= 0.0 or self.positive is None:
            self._clear()
            self.positive = positive
            self.negative = negative
            return True
        self._arc = prepare_slerp(self.positive, positive)
        self._negative_arc = prepare_slerp(self.negative, negative)
        self._target = positive
        self._target_negative = negative
        self._started = self._clock()
        self._t = 0.0
        return False

    def advance(self) -> tuple[Any, Any]:
        """Interpolate toward the target by wall clock and return the embeddings."""
        if self._target is None:
            return self.positive, self.negative
        if self.seconds <= 0.0:
            progress = 1.0
        else:
            progress = clamp01((self._clock() - self._started) / self.seconds)
        if progress >= 1.0:
            self.positive = self._target
            self.negative = self._target_negative
            self._clear()
            return self.positive, self.negative
        self._t = progress
        # A missing arc means there was no source to depart from, so the target
        # is already the only sensible value.
        self.positive = self._target if self._arc is None else self._arc.at(progress)
        self.negative = (
            self._target_negative
            if self._negative_arc is None
            else self._negative_arc.at(progress)
        )
        return self.positive, self.negative

    def _clear(self) -> None:
        self._arc = None
        self._negative_arc = None
        self._target = None
        self._target_negative = None
        self._t = 1.0


# ---------------------------------------------------------------------------
# Live settings
# ---------------------------------------------------------------------------

_FLOAT_SETTINGS: dict[str, tuple[float, float]] = {
    "guidance_scale": (0.0, 8.0),
    "instability": (0.0, 1.0),
    "guide_strength": (0.0, 1.0),
    "noise_walk_seconds": (0.0, 600.0),
    "noise_jitter": (0.0, 1.0),
    "prompt_walk_seconds": (0.0, 600.0),
    "memory_match": (0.0, 1.0),
    "memory_match_std": (0.0, 1.0),
    "memory_leash": (0.0, 8.0),
    "depth_guide": (0.0, 1.0),
}
_INT_SETTINGS: dict[str, tuple[int, int]] = {
    "steps": (1, 4),
    "timestep_min": (1, 999),
    "timestep_max": (1, 999),
}
_BOOL_SETTINGS = ("feedback_reprojection",)
SETTING_KEYS = frozenset({*_FLOAT_SETTINGS, *_INT_SETTINGS, *_BOOL_SETTINGS})


def normalise_settings(settings: Mapping[str, Any]) -> dict[str, Any]:
    """Clamp and type the settings this backend understands; ignore the rest."""
    clean: dict[str, Any] = {}
    for key, value in settings.items():
        if key in _FLOAT_SETTINGS:
            low, high = _FLOAT_SETTINGS[key]
            clean[key] = min(max(float(value), low), high)
        elif key in _INT_SETTINGS:
            low_int, high_int = _INT_SETTINGS[key]
            clean[key] = min(max(int(value), low_int), high_int)
        elif key in _BOOL_SETTINGS:
            clean[key] = bool(value)
    return clean


def _resolve_path(config: Mapping[str, Any], key: str, default: str) -> Path:
    """Resolve a model folder against ``project_root`` when it is relative."""
    value = Path(config.get(key, default))
    if value.is_absolute():
        return value
    return (Path(config.get("project_root", ".")) / value).resolve()


class LatentWalkBackend(DiffusionBackend):
    """Continuous latent-space walk steered by a proxy render.

    Generic over the diffusers layout: ``model_path`` supplies ``unet/``,
    ``text_encoder/``, ``tokenizer/`` and ``scheduler/``, ``taesd_path`` supplies
    an ``AutoencoderTiny``. The text encoder's hidden size and the step
    parameterisation both come from the loaded files, never from constants here.
    """

    def __init__(self) -> None:
        self.torch: Any = None
        self.unet: Any = None
        self.taesd: Any = None
        self.text_encoder: Any = None
        self.tokenizer: Any = None
        self.device = "cuda"
        self.dtype: Any = None
        self.width = 512
        self.height = 512
        self.steps = 1
        self.guidance_scale = 1.0
        self.sampler = "euler_x0"
        self.prediction_type = "epsilon"
        self.timestep_scaling = 10.0
        self.timestep_min = 650
        self.timestep_max = 900
        self.instability = 0.5
        self.guide_strength = 0.65
        self.memory_match = 1.0
        self.memory_match_std = 0.5
        self.memory_leash = 1.5
        self.depth_guide = 0.5
        self.feedback_reprojection = False
        self.seed = 12345
        self.warmup_passes = 2
        self.alphas_cumprod: np.ndarray = np.ones(1000, dtype=np.float64)
        self.prompt = ""
        self.negative_prompt = ""
        self.x0_prev: Any = None
        self._warm_frames = RESET_WARMUP_FRAMES
        self._noise_generator: Any = None
        # Wall-clock source for both walks and the instability oscillators, so
        # their speed is independent of diffusion FPS. Tests swap in a fake.
        self._clock = perf_counter
        self._origin = 0.0
        self.noise_walk = NoiseWalk(self._sample_noise, self._read_clock)
        self.prompt_walk = PromptWalk(self._read_clock)
        self._timestep_now = 0
        self._guide_now = 0.0
        self.inference_average = ExponentialAverage(alpha=0.2)
        self.load_ms = 0.0
        self.first_frame_ms = 0.0
        self.frames = 0

    # -- loading ------------------------------------------------------------

    def load(self, config: dict[str, Any]) -> None:
        """Load the UNet, CLIP text encoder and TAESD as raw fp16 modules."""
        started = perf_counter()
        try:
            import torch
            from diffusers import AutoencoderTiny, UNet2DConditionModel
            from transformers import CLIPTextModel, CLIPTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "Diffusion dependencies are missing. Prepare the bundled runtime first."
            ) from exc
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA-compatible NVIDIA GPU/driver not available")

        model_path = _resolve_path(config, "model_path", "models/sd_turbo")
        taesd_path = _resolve_path(config, "taesd_path", "models/taesd")
        missing = [
            str(path)
            for path in (
                model_path / "unet",
                model_path / "text_encoder",
                model_path / "tokenizer",
                model_path / "scheduler",
                taesd_path / "config.json",
            )
            if not path.exists()
        ]
        if missing:
            raise RuntimeError(
                f"Model files incomplete. Missing: {', '.join(missing)}. "
                "Run tools\\prepare_models.ps1 on the development PC."
            )

        self.torch = torch
        self.dtype = torch.float16
        self.width = int(config.get("diffusion_width", 512))
        self.height = int(config.get("diffusion_height", self.width))
        self.steps = int(config.get("steps", 1))
        self.guidance_scale = float(config.get("guidance_scale", 1.0))
        self.timestep_min = int(config.get("timestep_min", 650))
        self.timestep_max = int(config.get("timestep_max", 900))
        self.instability = float(config.get("instability", 0.5))
        self.guide_strength = float(config.get("guide_strength", 0.65))
        self.memory_match = clamp01(float(config.get("memory_match", 1.0)))
        self.memory_match_std = clamp01(float(config.get("memory_match_std", 0.5)))
        self.memory_leash = max(0.0, float(config.get("memory_leash", 1.5)))
        self.depth_guide = clamp01(float(config.get("depth_guide", 0.5)))
        self.feedback_reprojection = bool(config.get("feedback_reprojection", False))
        self.seed = int(config.get("seed", 12345))
        self.warmup_passes = int(config.get("warmup_passes", 2))
        self.noise_walk.seconds = max(0.0, float(config.get("noise_walk_seconds", 20.0)))
        self.noise_walk.jitter = clamp01(float(config.get("noise_jitter", 0.06)))
        self.prompt_walk.seconds = max(0.0, float(config.get("prompt_walk_seconds", 25.0)))

        scheduler_config = json.loads(
            (model_path / "scheduler" / "scheduler_config.json").read_text(encoding="utf-8")
        )
        self.alphas_cumprod = alphas_cumprod_from_scheduler(scheduler_config)
        self.prediction_type = str(scheduler_config.get("prediction_type", "epsilon"))
        self.timestep_scaling = float(scheduler_config.get("timestep_scaling", 10.0))
        sampler = str(config.get("sampler", "") or "").strip()
        if sampler and sampler not in ("euler_x0", "lcm"):
            raise RuntimeError(f"Unknown sampler: {sampler}")
        self.sampler = sampler or infer_sampler(scheduler_config)

        torch.backends.cuda.matmul.allow_tf32 = True
        # cudnn.benchmark stays off on purpose: measured here it buys about 5%
        # at 384x256 and nothing at 512x512, while pushing peak VRAM from
        # 2.4 GB to 4.7 GB - headroom an 8 GB exhibition machine needs.
        torch.set_grad_enabled(False)
        self.unet = _load_module(
            UNet2DConditionModel, model_path / "unet", torch.float16
        ).to(self.device, memory_format=torch.channels_last)
        self.unet.eval()
        self.taesd = _load_module(AutoencoderTiny, taesd_path, torch.float16).to(self.device)
        self.taesd.eval()
        self.text_encoder = _load_module(
            CLIPTextModel, model_path / "text_encoder", torch.float16
        ).to(self.device)
        self.text_encoder.eval()
        self.tokenizer = CLIPTokenizer.from_pretrained(
            str(model_path / "tokenizer"), local_files_only=True
        )
        self._noise_generator = torch.Generator(device=self.device).manual_seed(self.seed)
        self._origin = self._clock()
        self.set_prompt(str(config.get("prompt", "")), str(config.get("negative_prompt", "")))
        self.load_ms = (perf_counter() - started) * 1000.0

    def unload(self) -> None:
        self.unet = None
        self.taesd = None
        self.text_encoder = None
        self.tokenizer = None
        self.x0_prev = None
        self.noise_walk.reset()
        if self.torch is not None and self.torch.cuda.is_available():
            self.torch.cuda.empty_cache()

    # -- settings -----------------------------------------------------------

    def apply_settings(self, settings: Mapping[str, Any]) -> None:
        """Apply a batch of live settings; keys this backend has no use for are ignored."""
        clean = normalise_settings(settings)
        for key, value in clean.items():
            if key == "noise_walk_seconds":
                self.noise_walk.seconds = value
            elif key == "noise_jitter":
                self.noise_walk.jitter = value
            elif key == "prompt_walk_seconds":
                self.prompt_walk.seconds = value
            else:
                setattr(self, key, value)
        unknown = set(settings) - SETTING_KEYS
        if unknown:
            logging.debug("latent_walk ignored settings: %s", ", ".join(sorted(unknown)))

    def set_prompt(self, prompt: str, negative_prompt: str = "") -> None:
        """Aim the prompt walk at new text; ``prompt_walk_seconds`` 0 hard-cuts."""
        if self.text_encoder is None:
            self.prompt = prompt
            self.negative_prompt = negative_prompt
            return
        if (
            prompt == self.prompt
            and negative_prompt == self.negative_prompt
            and self.prompt_walk.positive is not None
        ):
            return
        self.prompt = prompt
        self.negative_prompt = negative_prompt
        # Both embeddings are cached so guidance can cross the CFG threshold
        # without a visible text-encoder pause.
        with self.torch.inference_mode():
            positive = self._encode_prompt(prompt)
            negative = self._encode_prompt(negative_prompt)
        if self.prompt_walk.retarget(positive, negative):
            # A hard cut restarts the look; stale memory would only carry the
            # previous prompt's structure into the new one.
            self.x0_prev = None
            self._warm_frames = RESET_WARMUP_FRAMES

    def set_clock(self, clock: Callable[[], float]) -> None:
        """Replace the wall clock driving both walks and the oscillators.

        Offline tooling and tests substitute a frame-driven clock so a run is
        reproducible instead of depending on GPU speed.
        """
        self._clock = clock
        self._origin = clock()

    def set_resolution(self, width: int, height: int | None = None) -> None:
        self.width = int(width)
        self.height = int(height if height is not None else width)
        self.x0_prev = None
        self._warm_frames = RESET_WARMUP_FRAMES
        self.noise_walk.reset()

    def reseed(self, seed: int) -> None:
        self.seed = int(seed)
        if self.torch is not None:
            self._noise_generator = self.torch.Generator(device=self.device).manual_seed(
                self.seed
            )
        self.x0_prev = None
        self._warm_frames = RESET_WARMUP_FRAMES
        self.noise_walk.reset()

    # -- hot path -----------------------------------------------------------

    def warmup(self) -> None:
        """Compile kernels and allocate the workspace before the first real frame."""
        yy, xx = np.mgrid[0 : self.height, 0 : self.width]
        rgb = np.empty((self.height, self.width, 3), dtype=np.uint8)
        rgb[:, :, 0] = np.clip(40 + xx * 150 / self.width, 0, 255)
        rgb[:, :, 1] = np.clip(60 + yy * 130 / self.height, 0, 255)
        rgb[:, :, 2] = 110
        for _ in range(max(0, self.warmup_passes)):
            self._render(rgb)
        self.x0_prev = None
        self._warm_frames = RESET_WARMUP_FRAMES
        self.noise_walk.reset()
        self.frames = 0
        self.inference_average = ExponentialAverage(alpha=0.2)

    def generate(
        self,
        conditioning: ConditioningFrame,
        previous_frame: np.ndarray | None = None,
    ) -> np.ndarray:
        if self.unet is None:
            raise RuntimeError("Latent-walk backend has not been loaded")
        # Reprojected feedback replaces the memory latent when it is enabled; it
        # is aligned to this camera, so it re-anchors the walk to the proxy.
        aligned = previous_frame if self.feedback_reprojection else None
        return self._render(conditioning.rgb, aligned, conditioning.depth)

    def _guide_weight_map(self, depth: np.ndarray | None, guide: Any) -> Any | None:
        """Per-latent-pixel guide strength from proxy depth, or None if unused.

        Near geometry keeps the configured strength; distant geometry and sky
        drop toward ``guide * (1 - depth_guide)``, so the horizon is where the
        walk hallucinates and the foreground stays anchored. That gradient is
        what gives a frame depth at eye level, where every form otherwise
        competes equally for the sampler.
        """
        if depth is None or self.depth_guide <= 0.0:
            return None
        torch = self.torch
        tensor = torch.from_numpy(np.ascontiguousarray(depth)).to(self.device, self.dtype)
        tensor = tensor[None, None]
        tensor = torch.nn.functional.interpolate(
            tensor, size=guide.shape[-2:], mode="bilinear", align_corners=False
        )
        # Depth-buffer values are non-linear; the far half of the frame sits in
        # the top few percent, so remap so the fade is visible over the walk.
        near = torch.clamp((tensor - 0.90) / 0.10, 0.0, 1.0)
        return 1.0 - self.depth_guide * near

    def _render(
        self, rgb: np.ndarray, aligned: np.ndarray | None = None, depth: np.ndarray | None = None
    ) -> np.ndarray:
        """Run one walk step; ``aligned`` replaces the memory latent when given."""
        torch = self.torch
        started = perf_counter()
        with torch.inference_mode():
            elapsed = self._clock() - self._origin
            self._timestep_now = breathing_timestep(
                elapsed, self.timestep_min, self.timestep_max, self.instability
            )
            self._guide_now = wobbled_guide_strength(
                elapsed, self.guide_strength, self.instability
            )
            positive, negative = self.prompt_walk.advance()
            guide = self._encode_image(rgb)
            if aligned is None and self.x0_prev is None and self._warm_frames > 0:
                self._seed_memory(guide, positive, negative)
            memory = self._encode_image(aligned) if aligned is not None else self.x0_prev
            if memory is None or memory.shape != guide.shape:
                base = guide
            else:
                memory = match_latent_statistics(
                    memory, guide, self.memory_match, self.memory_match_std
                )
                memory = leash_to_guide(memory, guide, self.memory_leash)
                weight_map = self._guide_weight_map(depth, guide)
                weight = self._guide_now if weight_map is None else self._guide_now * weight_map
                base = torch.lerp(memory, guide, weight)
            noise = self.noise_walk.value()
            x0 = self._denoise(base, noise, self._timestep_now, positive, negative)
            self.x0_prev = x0
            output = self._decode_image(x0)
        # The readback synchronises, so this records real end-to-end latency.
        inference_ms = (perf_counter() - started) * 1000.0
        self.inference_average.update(inference_ms)
        if self.frames == 0:
            self.first_frame_ms = inference_ms
        self.frames += 1
        return output

    def _seed_memory(self, guide: Any, positive: Any, negative: Any) -> None:
        """Fill the empty memory latent with hallucination, not with the proxy.

        Runs ``RESET_WARMUP_FRAMES`` discarded passes at ``timestep_max`` from
        the guide, so the first frame the walk publishes after a reset is
        conditioned on an already-hallucinated memory. See RESET_WARMUP_FRAMES.
        """
        base = guide
        for _ in range(self._warm_frames):
            base = self._denoise(
                base, self.noise_walk.value(), self.timestep_max, positive, negative
            )
        self.x0_prev = base
        self._warm_frames = 0

    def _denoise(
        self, base: Any, noise: Any, timestep: int, positive: Any, negative: Any
    ) -> Any:
        """One or more x0 steps down from ``timestep`` toward a clean latent."""
        torch = self.torch
        do_cfg = classifier_free_guidance_enabled(self.guidance_scale) and negative is not None
        embeds = torch.cat([negative, positive]) if do_cfg else positive
        sample = base
        for step in range(max(1, self.steps)):
            # Later steps re-noise the running prediction to a lower timestep,
            # which sharpens without discarding the walk's structure.
            t = max(1, int(round(timestep * (self.steps - step) / self.steps)))
            alpha = float(self.alphas_cumprod[min(t, len(self.alphas_cumprod) - 1)])
            x_t = add_noise(sample, noise, alpha).contiguous(
                memory_format=torch.channels_last
            )
            model_input = torch.cat([x_t, x_t]) if do_cfg else x_t
            timesteps = torch.full(
                (model_input.shape[0],), t, device=self.device, dtype=torch.long
            )
            output = self.unet(model_input, timesteps, encoder_hidden_states=embeds).sample
            if do_cfg:
                uncond, cond = output.chunk(2)
                output = uncond + self.guidance_scale * (cond - uncond)
            x0 = predict_x0(x_t, output, alpha, self.prediction_type)
            if self.sampler == "lcm":
                c_skip, c_out = lcm_scalings(t, self.timestep_scaling)
                x0 = x0 * c_out + x_t * c_skip
            sample = x0
        return sample

    def _encode_image(self, rgb: np.ndarray) -> Any:
        """Upload the proxy once and TAESD-encode it to a guide latent."""
        torch = self.torch
        tensor = torch.from_numpy(np.ascontiguousarray(rgb)).to(self.device)
        tensor = tensor.permute(2, 0, 1).unsqueeze(0).to(self.dtype).div_(127.5).sub_(1.0)
        if tensor.shape[-2:] != (self.height, self.width):
            tensor = torch.nn.functional.interpolate(
                tensor, size=(self.height, self.width), mode="bilinear", align_corners=False
            )
        return self.taesd.encode(tensor).latents

    def _decode_image(self, latents: Any) -> np.ndarray:
        """TAESD-decode and read back the only host copy of the frame."""
        image = self.taesd.decode(latents).sample
        image = image.add(1.0).mul_(127.5).clamp_(0.0, 255.0)
        return (
            image[0]
            .permute(1, 2, 0)
            .round()
            .to(self.torch.uint8)
            .contiguous()
            .cpu()
            .numpy()
        )

    def _encode_prompt(self, text: str) -> Any:
        tokens = self.tokenizer(
            text,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        ids = tokens.input_ids.to(self.device)
        return self.text_encoder(ids)[0].to(self.dtype)

    def _sample_noise(self) -> Any:
        """Fresh unit-variance latent noise for a walk keyframe."""
        return self.torch.randn(
            (1, self.unet.config.in_channels, self.height // 8, self.width // 8),
            generator=self._noise_generator,
            device=self.device,
            dtype=self.dtype,
        )

    def _read_clock(self) -> float:
        # Indirection so tests can replace ``_clock`` after construction.
        return self._clock()

    # -- reporting ----------------------------------------------------------

    def stats(self) -> dict[str, float | int | str]:
        allocated = reserved = peak = 0.0
        if self.torch is not None and self.torch.cuda.is_available():
            scale = 1024.0**3
            allocated = self.torch.cuda.memory_allocated() / scale
            reserved = self.torch.cuda.memory_reserved() / scale
            peak = self.torch.cuda.max_memory_allocated() / scale
        return {
            "backend": "latent_walk",
            "inference_ms": self.inference_average.value,
            "sampler": self.sampler,
            "resolution": f"{self.width}x{self.height}",
            "resolution_width": self.width,
            "resolution_height": self.height,
            "steps": self.steps,
            "guidance_scale": self.guidance_scale,
            "timestep_min": self.timestep_min,
            "timestep_max": self.timestep_max,
            "timestep_now": self._timestep_now,
            "instability": self.instability,
            "guide_strength": self.guide_strength,
            "guide_strength_now": self._guide_now,
            "memory_match": self.memory_match,
            "memory_match_std": self.memory_match_std,
            "memory_leash": self.memory_leash,
            "depth_guide": self.depth_guide,
            "noise_walk_seconds": self.noise_walk.seconds,
            "noise_jitter": self.noise_walk.jitter,
            "noise_walk_t": self.noise_walk.t,
            "prompt_walk_seconds": self.prompt_walk.seconds,
            "prompt_walk_t": self.prompt_walk.t,
            "feedback_reprojection": str(self.feedback_reprojection),
            "active_seed": self.seed,
            "vram_allocated_gb": allocated,
            "vram_reserved_gb": reserved,
            "peak_vram_gb": peak,
            "load_ms": self.load_ms,
            "first_frame_ms": self.first_frame_ms,
        }


def _load_module(cls: Any, path: Path, dtype: Any) -> Any:
    """Load a model folder, using the fp16 variant only when the repo ships one.

    Probing for the variant first keeps a dropped-in model (TAESD ships single
    precision weights) from logging a fetch error on every start.
    """
    kwargs: dict[str, Any] = {"torch_dtype": dtype, "local_files_only": True}
    if any(path.glob("*.fp16.safetensors")):
        kwargs["variant"] = "fp16"
    return cls.from_pretrained(str(path), **kwargs)
