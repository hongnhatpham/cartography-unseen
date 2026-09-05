from __future__ import annotations

import json
import math
import threading
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from app.diffusion.latent_walk import (
    RESET_WARMUP_FRAMES,
    LatentWalkBackend,
    NoiseWalk,
    PromptWalk,
    add_noise,
    alphas_cumprod_from_scheduler,
    blend_correlated_noise,
    breathing_timestep,
    classifier_free_guidance_enabled,
    infer_sampler,
    lcm_scalings,
    normalise_settings,
    predict_x0,
    prepare_slerp,
    slerp_embeddings,
    leash_to_guide,
    match_latent_statistics,
    wobbled_guide_strength,
)
from app.diffusion.worker import DiffusionWorker
from app.types import CameraSnapshot, ConditioningFrame

ROOT = Path(__file__).resolve().parents[1]


class ManualClock:
    """Deterministic wall clock so the walks are tested independently of FPS."""

    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def make_conditioning(width: int = 8, height: int = 8) -> ConditioningFrame:
    identity = np.eye(4, dtype=np.float32)
    camera = CameraSnapshot(
        view_matrix=identity,
        projection_matrix=identity,
        position=np.zeros(3, dtype=np.float32),
        rotation=np.zeros(3, dtype=np.float32),
    )
    yy, xx = np.mgrid[0:height, 0:width]
    rgb = np.stack(
        [
            np.clip(30 + xx * 200 / width, 0, 255),
            np.clip(40 + yy * 180 / height, 0, 255),
            np.full((height, width), 120.0),
        ],
        axis=-1,
    ).astype(np.uint8)
    return ConditioningFrame(
        rgb=rgb,
        depth=np.full((height, width), 0.5, dtype=np.float32),
        edges=np.zeros((height, width), dtype=np.uint8),
        camera=camera,
        timestamp=0.0,
        sequence=1,
    )


# -- scheduler maths --------------------------------------------------------


def test_alphas_cumprod_matches_the_shipped_scheduler_shape() -> None:
    config = json.loads(
        (ROOT / "models" / "sd_turbo" / "scheduler" / "scheduler_config.json").read_text(
            encoding="utf-8"
        )
    )

    alphas = alphas_cumprod_from_scheduler(config)

    assert alphas.shape == (1000,)
    assert 0.0 < alphas[-1] < alphas[0] < 1.0
    assert np.all(np.diff(alphas) < 0.0)
    assert infer_sampler(config) == "euler_x0"
    assert infer_sampler({"_class_name": "LCMScheduler"}) == "lcm"


def test_predict_x0_inverts_the_forward_diffusion() -> None:
    rng = np.random.default_rng(7)
    x0 = rng.normal(size=(1, 4, 4, 4))
    noise = rng.normal(size=(1, 4, 4, 4))
    alpha = 0.31

    x_t = add_noise(x0, noise, alpha)

    assert np.allclose(predict_x0(x_t, noise, alpha), x0)
    velocity = noise * math.sqrt(alpha) - x0 * math.sqrt(1.0 - alpha)
    assert np.allclose(predict_x0(x_t, velocity, alpha, "v_prediction"), x0)


def test_lcm_scalings_move_from_skip_to_output() -> None:
    skip_low, out_low = lcm_scalings(0.0)
    skip_high, out_high = lcm_scalings(900.0)

    assert (skip_low, out_low) == (1.0, 0.0)
    assert skip_high < 1e-4
    assert out_high == pytest.approx(1.0, abs=1e-4)


# -- instability ------------------------------------------------------------


def test_instability_scales_the_timestep_breathing_range() -> None:
    steady = {breathing_timestep(t * 0.5, 650, 900, 0.0) for t in range(40)}
    breathing = [breathing_timestep(t * 0.25, 650, 900, 1.0) for t in range(400)]
    mild = [breathing_timestep(t * 0.25, 650, 900, 0.35) for t in range(400)]

    assert steady == {775}
    assert min(breathing) == 650 and max(breathing) == 900
    # The band is a hard limit; the extra reach only makes the walk dwell at
    # the ends rather than push past them.
    assert all(650 <= value <= 900 for value in breathing)
    assert min(mild) > 650 and max(mild) < 900
    # Swapped bounds must not invert the range.
    assert breathing_timestep(0.0, 900, 650, 1.0) == 775


def test_the_slow_breath_tone_carries_a_long_loop_between_regimes() -> None:
    """One 13 s sine returned to the same timestep every cycle, so a 300-frame
    loop kept re-running the same look; the slow tone shifts each cycle."""
    first = [breathing_timestep(t * 0.5, 600, 800, 0.8) for t in range(26)]
    later = [breathing_timestep(60.0 + t * 0.5, 600, 800, 0.8) for t in range(26)]

    assert max(abs(a - b) for a, b in zip(first, later)) > 20


def test_guide_wobble_only_ever_tightens_the_anchor() -> None:
    steady = {wobbled_guide_strength(t * 0.5, 0.65, 0.0) for t in range(20)}
    wobbled = [wobbled_guide_strength(t * 0.5, 0.65, 1.0) for t in range(120)]

    assert steady == {0.65}
    # guide_strength is a floor: a trough below it is where the walk let go of
    # the proxy and drifted into interiors and lettering.
    assert min(wobbled) >= 0.65 - 1e-9
    assert max(wobbled) > 0.65
    assert all(0.0 <= value <= 1.0 for value in wobbled)
    assert wobbled_guide_strength(7.25, 0.95, 1.0) == 1.0
    # The wobble is slow on purpose: a look has to hold long enough to settle.
    assert max(wobbled) - min(wobbled) > 0.2


# -- memory statistics ------------------------------------------------------


def test_matching_latent_statistics_re_anchors_mean_and_spread() -> None:
    """The feedback loop has nothing else restoring the memory's DC and contrast."""
    rng = np.random.default_rng(7)
    memory = rng.normal(4.0, 6.0, (1, 4, 8, 8)).astype("float32")
    guide = rng.normal(-0.5, 1.5, (1, 4, 8, 8)).astype("float32")

    matched = match_latent_statistics(memory, guide, 1.0)
    axes = (0, 2, 3)
    assert np.allclose(matched.mean(axis=axes), guide.mean(axis=axes), atol=1e-3)
    assert np.allclose(matched.std(axis=axes), guide.std(axis=axes), atol=1e-3)
    # Only the statistics move; the spatial pattern is untouched.
    flat_memory = memory.reshape(4, -1)
    flat_matched = matched.reshape(4, -1)
    for channel in range(4):
        correlation = np.corrcoef(flat_memory[channel], flat_matched[channel])[0, 1]
        assert correlation > 0.999
    assert np.allclose(match_latent_statistics(memory, guide, 0.0), memory)
    half = match_latent_statistics(memory, guide, 0.5)
    assert abs(half.mean() - memory.mean()) < abs(matched.mean() - memory.mean())


def test_std_strength_holds_the_mean_while_letting_contrast_breathe() -> None:
    """Full std matching flattens the picture into the flat-shaded proxy's band."""
    rng = np.random.default_rng(13)
    memory = rng.normal(4.0, 6.0, (1, 4, 8, 8)).astype("float32")
    guide = rng.normal(-0.5, 1.5, (1, 4, 8, 8)).astype("float32")
    axes = (0, 2, 3)

    loose = match_latent_statistics(memory, guide, 1.0, 0.0)
    # The mean still lands on the guide, the spread is exactly the memory's.
    assert np.allclose(loose.mean(axis=axes), guide.mean(axis=axes), atol=1e-3)
    assert np.allclose(loose.std(axis=axes), memory.std(axis=axes), atol=1e-3)

    partial = match_latent_statistics(memory, guide, 1.0, 0.5)
    assert np.allclose(partial.mean(axis=axes), guide.mean(axis=axes), atol=1e-3)
    # The memory here is the wider of the two, so partial matching keeps more
    # contrast than a full match and less than none.
    assert np.all(partial.std(axis=axes) > guide.std(axis=axes))
    assert np.all(partial.std(axis=axes) < memory.std(axis=axes))
    # std_strength 1 reproduces the original mean-and-spread behaviour.
    assert np.allclose(
        match_latent_statistics(memory, guide, 1.0, 1.0),
        match_latent_statistics(memory, guide, 1.0),
    )


def test_leash_bounds_how_far_the_memory_may_leave_the_guide() -> None:
    rng = np.random.default_rng(11)
    guide = rng.normal(0.0, 2.0, (1, 4, 8, 8)).astype("float32")
    memory = guide + rng.normal(0.0, 20.0, (1, 4, 8, 8)).astype("float32")

    leashed = leash_to_guide(memory, guide, 1.5)
    radius = guide.std(axis=(0, 2, 3), keepdims=True) * 1.5
    assert np.all(np.abs(leashed - guide) <= radius + 1e-4)
    # Anything already inside the band is left exactly as it was.
    inside = guide + radius * 0.5
    assert np.allclose(leash_to_guide(inside, guide, 1.5), inside, atol=1e-4)
    assert np.allclose(leash_to_guide(memory, guide, 0.0), memory)


# -- noise walk -------------------------------------------------------------


def test_noise_walk_slerps_between_keyframes_on_the_wall_clock() -> None:
    keyframes = [
        np.array([1.0, 0.0], dtype=np.float32),
        np.array([0.0, 1.0], dtype=np.float32),
        np.array([-1.0, 0.0], dtype=np.float32),
    ]
    clock = ManualClock()
    walk = NoiseWalk(lambda: keyframes.pop(0), clock)
    walk.seconds = 4.0

    start = walk.value()
    clock.now = 2.0
    middle = walk.value()
    clock.now = 4.0
    end = walk.value()
    resumed = walk.value()

    assert np.allclose(start, [1.0, 0.0])
    assert np.allclose(middle, [math.sqrt(0.5), math.sqrt(0.5)], atol=1e-6)
    # Slerp keeps the norm, which is what stops the noise level from breathing.
    assert float(np.linalg.norm(middle)) == pytest.approx(1.0, abs=1e-6)
    assert np.allclose(end, [0.0, 1.0])
    # The next arc departs from the keyframe just reached, so there is no jump.
    assert np.allclose(resumed, end)
    assert walk.t == 0.0


def test_noise_walk_jitter_mixes_a_drifting_field_without_gaining_variance() -> None:
    rng = np.random.default_rng(3)
    clock = ManualClock()
    walk = NoiseWalk(lambda: rng.normal(size=4096).astype(np.float32), clock)
    walk.seconds = 30.0
    walk.jitter = 0.5

    clock.now = 5.0
    noisy = walk.value()

    assert float(noisy.std()) == pytest.approx(1.0, abs=0.05)
    walk.reset()
    assert walk.t == 0.0


def test_correlated_noise_blend_has_stable_endpoints() -> None:
    previous = np.full((1, 4, 2, 2), 3.0, dtype=np.float32)
    fresh = np.full((1, 4, 2, 2), 7.0, dtype=np.float32)

    assert np.array_equal(blend_correlated_noise(previous, fresh, 0.0), fresh)
    assert np.array_equal(blend_correlated_noise(previous, fresh, 1.0), previous)
    assert np.array_equal(blend_correlated_noise(None, fresh, 0.5), fresh)


# -- prompt walk ------------------------------------------------------------


def test_prompt_walk_reaches_the_target_over_wall_clock_seconds() -> None:
    source = np.array([1.0, 0.0], dtype=np.float32)
    target = np.array([0.0, 1.0], dtype=np.float32)
    clock = ManualClock()
    walk = PromptWalk(clock)
    walk.seconds = 4.0
    walk.positive = source
    walk.negative = source

    assert walk.retarget(target, target) is False
    distances = []
    for second in range(5):
        clock.now = float(second)
        positive, _negative = walk.advance()
        distances.append(float(np.linalg.norm(np.asarray(positive) - target)))

    assert distances[0] == pytest.approx(math.sqrt(2.0))
    assert all(later < earlier for earlier, later in zip(distances, distances[1:]))
    assert distances[-1] == 0.0
    assert walk.positive is target
    assert walk.t == 1.0


def test_retargeting_mid_walk_departs_from_the_current_embedding() -> None:
    source = np.array([1.0, 0.0], dtype=np.float32)
    first = np.array([0.0, 1.0], dtype=np.float32)
    second = np.array([-1.0, 0.0], dtype=np.float32)
    clock = ManualClock()
    walk = PromptWalk(clock)
    walk.seconds = 4.0
    walk.positive = source
    walk.negative = source

    walk.retarget(first, first)
    clock.now = 2.0
    midpoint = np.asarray(walk.advance()[0]).copy()
    walk.retarget(second, second)
    resumed = np.asarray(walk.advance()[0])

    assert np.allclose(resumed, midpoint)
    assert not np.allclose(resumed, source)


def test_zero_walk_seconds_hard_cuts_to_the_new_embeddings() -> None:
    walk = PromptWalk(ManualClock())
    walk.seconds = 0.0
    walk.positive = np.array([1.0, 0.0], dtype=np.float32)
    target = np.array([0.0, 1.0], dtype=np.float32)

    assert walk.retarget(target, target) is True
    assert walk.positive is target
    assert walk.t == 1.0


class ReductionCounter:
    def __init__(self) -> None:
        self.count = 0


class CountingArray(np.ndarray):
    """Array that records reductions, which are the device syncs on a GPU."""

    counter: ReductionCounter | None = None

    def __array_finalize__(self, obj: object) -> None:
        self.counter = getattr(obj, "counter", None)

    def sum(self, *args: object, **kwargs: object):  # type: ignore[override]
        if self.counter is not None:
            self.counter.count += 1
        return super().sum(*args, **kwargs)


def counting_array(values: list[float], counter: ReductionCounter) -> CountingArray:
    array = np.asarray(values, dtype=np.float32).view(CountingArray)
    array.counter = counter
    return array


def test_walking_a_prompt_reduces_the_embeddings_only_once() -> None:
    counter = ReductionCounter()
    clock = ManualClock()
    walk = PromptWalk(clock)
    walk.seconds = 4.0
    walk.positive = counting_array([1.0, 0.0], counter)
    walk.negative = walk.positive

    target = counting_array([0.0, 1.0], counter)
    walk.retarget(target, target)
    measured = counter.count
    assert measured > 0

    for second in (1.0, 2.0, 3.0):
        clock.now = second
        walk.advance()

    assert counter.count == measured


def test_slerp_arc_lerps_when_the_coefficients_would_overflow() -> None:
    source = np.array([1.0, 0.0], dtype=np.float32)
    # Just short of antiparallel: sin(theta) is small enough that the slerp
    # coefficients would leave the fp16-safe range.
    angle = math.pi - 0.05
    target = np.array([math.cos(angle), math.sin(angle)], dtype=np.float32)

    arc = prepare_slerp(source, target)

    assert arc is not None and arc.sin_theta == 0.0
    assert np.allclose(arc.at(0.5), (source + target) * 0.5)
    assert prepare_slerp(source, np.array([0.0, 1.0], dtype=np.float32)).sin_theta > 0.0


def test_slerp_of_parallel_embeddings_stays_finite() -> None:
    vector = np.array([0.3, -1.2, 4.0], dtype=np.float32)

    same = slerp_embeddings(vector, vector.copy(), 0.5)
    opposite = slerp_embeddings(vector, -vector, 0.5)

    assert np.isfinite(same).all() and np.allclose(same, vector)
    assert np.isfinite(opposite).all() and np.allclose(opposite, 0.0)


# -- settings ---------------------------------------------------------------


def test_settings_are_clamped_and_unknown_keys_dropped() -> None:
    clean = normalise_settings(
        {
            "steps": 9,
            "guidance_scale": -2.0,
            "instability": 1.4,
            "timestep_min": 0,
            "feedback_reprojection": 1,
            "img2img_strength": 0.4,
        }
    )

    assert clean == {
        "steps": 4,
        "guidance_scale": 0.0,
        "instability": 1.0,
        "timestep_min": 1,
        "feedback_reprojection": True,
    }
    assert classifier_free_guidance_enabled(1.0) is False
    assert classifier_free_guidance_enabled(1.5) is True


def test_apply_settings_routes_walk_values_into_the_walks() -> None:
    backend = LatentWalkBackend()

    backend.apply_settings(
        {
            "noise_walk_seconds": 12.5,
            "noise_jitter": 0.2,
            "prompt_walk_seconds": 30.0,
            "guide_strength": 0.4,
            "instability": 0.9,
            "seed_mode": "drift",
        }
    )

    assert backend.noise_walk.seconds == 12.5
    assert backend.noise_walk.jitter == 0.2
    assert backend.prompt_walk.seconds == 30.0
    assert backend.guide_strength == 0.4
    assert backend.instability == 0.9


# -- worker plumbing --------------------------------------------------------


class FakeBackend:
    """Records what the worker forwards, without loading any model."""

    def __init__(self) -> None:
        self.settings: dict[str, Any] = {}
        self.prompts: list[tuple[str, str]] = []
        self.seeds: list[int] = []
        self.frames = threading.Event()
        self.previous_frames: list[Any] = []

    def load(self, config: dict[str, Any]) -> None:
        return

    def warmup(self) -> None:
        return

    def apply_settings(self, settings: Any) -> None:
        self.settings.update(settings)

    def set_prompt(self, prompt: str, negative_prompt: str = "") -> None:
        self.prompts.append((prompt, negative_prompt))

    def reseed(self, seed: int) -> None:
        self.seeds.append(seed)

    def set_resolution(self, width: int, height: int | None = None) -> None:
        return

    def generate(self, conditioning, previous_frame=None):
        self.previous_frames.append(previous_frame)
        self.frames.set()
        return conditioning.rgb

    def stats(self) -> dict[str, float | int | str]:
        return {"backend": "fake"}

    def unload(self) -> None:
        return


def run_worker(monkeypatch: pytest.MonkeyPatch, config: dict[str, Any]) -> FakeBackend:
    backend = FakeBackend()
    monkeypatch.setattr("app.diffusion.worker.create_backend", lambda _name: backend)
    worker = DiffusionWorker("fake", config)
    worker.request_settings(instability=0.8, guide_strength=0.3)
    worker.start()
    try:
        for _ in range(2):
            backend.frames.clear()
            worker.publish(make_conditioning())
            assert backend.frames.wait(5.0)
    finally:
        worker.stop()
    return backend


def test_worker_forwards_settings_prompt_and_seed(monkeypatch: pytest.MonkeyPatch) -> None:
    config = {
        "prompt": "voxel ridge",
        "negative_prompt": "text",
        "seed": 4242,
        "steps": 2,
        "noise_walk_seconds": 18.0,
        "diffusion_width": 8,
        "diffusion_height": 8,
    }

    backend = run_worker(monkeypatch, config)

    # Config seeds the first batch; request_settings adds to it before start.
    assert backend.settings["steps"] == 2
    assert backend.settings["noise_walk_seconds"] == 18.0
    assert backend.settings["instability"] == 0.8
    assert backend.settings["guide_strength"] == 0.3
    assert backend.prompts == [("voxel ridge", "text")]
    assert backend.seeds == [4242]
    # Reprojection is off by default, so no aligned previous frame is produced.
    assert backend.previous_frames == [None, None]


def test_worker_reprojects_only_when_feedback_is_enabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = {
        "prompt": "voxel ridge",
        "seed": 1,
        "feedback_reprojection": True,
        "diffusion_width": 8,
        "diffusion_height": 8,
    }

    backend = run_worker(monkeypatch, config)

    assert backend.settings["feedback_reprojection"] is True
    assert backend.previous_frames[0] is None
    assert backend.previous_frames[1] is not None


# -- GPU smoke --------------------------------------------------------------


def cuda_models_available() -> bool:
    try:
        import torch
    except ImportError:
        return False
    return (
        torch.cuda.is_available()
        and (ROOT / "models" / "sd_turbo" / "unet").exists()
        and (ROOT / "models" / "taesd" / "config.json").exists()
    )


def test_dropping_the_memory_arms_a_discarded_warmup() -> None:
    """The first frame after a reset must not be conditioned on the raw proxy.

    A large flat pale plane in the proxy is completed as a game controller on a
    desk whatever the negative prompt says, so the memory is seeded from
    hallucination at ``timestep_max`` before anything is published.
    """
    backend = LatentWalkBackend()
    backend.timestep_max = 900
    backend._warm_frames = 0
    backend.reseed(7)
    assert backend._warm_frames == RESET_WARMUP_FRAMES

    timesteps: list[int] = []
    backend._denoise = lambda base, noise, t, positive, negative: timesteps.append(t) or base
    backend.noise_walk.value = lambda: None
    backend._seed_memory("guide", None, None)

    assert timesteps == [900] * RESET_WARMUP_FRAMES
    assert backend.x0_prev == "guide"
    assert backend._warm_frames == 0


@pytest.mark.skipif(not cuda_models_available(), reason="CUDA or local models unavailable")
def test_latent_walk_generates_on_the_gpu() -> None:
    backend = LatentWalkBackend()
    backend.load(
        {
            "project_root": ROOT,
            "model_path": "models/sd_turbo",
            "taesd_path": "models/taesd",
            "diffusion_width": 384,
            "diffusion_height": 256,
            "prompt": "screenshot of a video game neural network highway, corrupted, glitch art",
            "negative_prompt": "text, ui, interface",
            "warmup_passes": 1,
        }
    )
    try:
        backend.warmup()
        first = backend.generate(make_conditioning(384, 256))
        second = backend.generate(make_conditioning(384, 256))
    finally:
        backend.unload()

    assert first.shape == (256, 384, 3) and first.dtype == np.uint8
    assert first.std() > 5.0
    # The memory latent means consecutive frames differ without being unrelated.
    assert not np.array_equal(first, second)
    assert float(np.abs(first.astype(np.int16) - second.astype(np.int16)).mean()) < 60.0
