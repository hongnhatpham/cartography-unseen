from __future__ import annotations

import logging
import secrets
import threading
from dataclasses import dataclass, replace
from time import perf_counter
from typing import Any

import numpy as np

from app.diffusion.factory import create_backend
from app.types import ConditioningFrame, GeneratedFrame
from app.temporal.reprojection import reproject_previous_image
from app.utils.latest_value import LatestValue
from app.utils.timing import RateMeter


@dataclass(frozen=True, slots=True)
class WorkerStatus:
    state: str = "created"
    message: str = ""
    error: str = ""
    active_resolution: str = "512x512"


class DiffusionWorker:
    def __init__(self, backend_name: str, config: dict[str, Any]) -> None:
        self.backend_name = backend_name
        self.config = dict(config)
        self.conditioning: LatestValue[ConditioningFrame] = LatestValue()
        self.generated: LatestValue[GeneratedFrame] = LatestValue()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._status_lock = threading.Lock()
        initial_size = (
            int(config.get("diffusion_width", 512)),
            int(config.get("diffusion_height", 512)),
        )
        self._status = WorkerStatus(
            active_resolution=f"{initial_size[0]}x{initial_size[1]}"
        )
        self._request_lock = threading.Lock()
        self._requested_prompt = (str(config["prompt"]), str(config.get("negative_prompt", "")))
        self._prompt_revision = 0
        self._requested_seed = int(config.get("seed", 12345))
        self._requested_seed_mode = str(config.get("seed_mode", "fixed"))
        self._requested_steps = int(config.get("steps", 1))
        self._requested_guidance_scale = float(config.get("guidance_scale", 0.0))
        self._requested_one_step_timestep = int(config.get("one_step_timestep", 750))
        self._requested_edge_softness = float(config.get("edge_softness", 1.5))
        self._requested_img2img_strength = float(config.get("img2img_strength", 0.4))
        self._requested_edge_strength = float(config.get("edge_strength", 0.15))
        self._requested_noise_persistence = float(config.get("noise_persistence", 0.975))
        self._requested_resolution = initial_size
        self._freeze = False
        self._backend_stats: dict[str, float | int | str] = {}
        self._rate = RateMeter(window=60)

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, name="diffusion-worker", daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 8.0) -> None:
        self._stop.set()
        self.conditioning.wake()
        if self._thread is not None:
            self._thread.join(timeout)

    def publish(self, frame: ConditioningFrame) -> None:
        if not self._freeze:
            self.conditioning.publish(frame)

    def set_frozen(self, frozen: bool) -> None:
        self._freeze = frozen

    def request_prompt(self, prompt: str, negative_prompt: str = "") -> int:
        with self._request_lock:
            self._requested_prompt = (prompt, negative_prompt)
            self._prompt_revision += 1
            return self._prompt_revision

    def request_reseed(self, seed: int | None = None) -> int:
        value = seed if seed is not None else secrets.randbelow(2**31)
        with self._request_lock:
            self._requested_seed = value
        return value

    def request_seed_mode(self, mode: str) -> None:
        with self._request_lock:
            self._requested_seed_mode = mode

    def request_steps(self, steps: int) -> None:
        with self._request_lock:
            self._requested_steps = max(1, min(4, int(steps)))

    def request_guidance_scale(self, guidance_scale: float) -> None:
        with self._request_lock:
            self._requested_guidance_scale = max(0.0, min(4.0, float(guidance_scale)))

    def request_one_step_timestep(self, timestep: int) -> None:
        with self._request_lock:
            self._requested_one_step_timestep = max(250, min(900, int(timestep)))

    def request_edge_softness(self, softness: float) -> None:
        with self._request_lock:
            self._requested_edge_softness = max(0.0, min(4.0, float(softness)))

    def request_img2img_strength(self, strength: float) -> None:
        with self._request_lock:
            self._requested_img2img_strength = max(0.25, min(1.0, float(strength)))

    def request_edge_strength(self, strength: float) -> None:
        with self._request_lock:
            self._requested_edge_strength = max(0.0, min(1.0, float(strength)))

    def request_noise_persistence(self, persistence: float) -> None:
        with self._request_lock:
            self._requested_noise_persistence = max(0.0, min(1.0, float(persistence)))

    def request_resolution(self, resolution: tuple[int, int]) -> None:
        with self._request_lock:
            self._requested_resolution = (int(resolution[0]), int(resolution[1]))

    def status(self) -> WorkerStatus:
        with self._status_lock:
            return self._status

    def stats(self) -> dict[str, float | int | str]:
        with self._status_lock:
            return {**self._backend_stats, "diffusion_fps": self._rate.fps}

    def _set_status(self, **changes: Any) -> None:
        with self._status_lock:
            self._status = replace(self._status, **changes)

    def _run(self) -> None:
        backend = create_backend(self.backend_name)
        previous: np.ndarray | None = None
        previous_conditioning: ConditioningFrame | None = None
        try:
            self._set_status(state="loading", message=f"Loading {self.backend_name}")
            backend.load(self.config)
            self._set_status(state="warming", message="Warming GPU pipeline")
            backend.warmup()
            self._set_status(state="ready", message="Ready")
            logging.info("Diffusion backend ready: %s", self.backend_name)
            last_version = 0
            applied_prompt: tuple[str, str] | None = None
            applied_prompt_revision = -1
            applied_seed: int | None = None
            applied_seed_mode: str | None = None
            applied_steps: int | None = None
            applied_guidance_scale: float | None = None
            applied_one_step_timestep: int | None = None
            applied_edge_softness: float | None = None
            applied_img2img_strength: float | None = None
            applied_edge_strength: float | None = None
            applied_noise_persistence: float | None = None
            applied_resolution: tuple[int, int] | None = None
            logged_first_frame = False
            while not self._stop.is_set():
                version, conditioning = self.conditioning.wait_newer(last_version, self._stop)
                if conditioning is None:
                    continue
                last_version = version
                # Read interactive changes after waking so a prompt entered
                # while idle applies to this conditioning frame, not the next.
                with self._request_lock:
                    requested_prompt = self._requested_prompt
                    requested_prompt_revision = self._prompt_revision
                    requested_seed = self._requested_seed
                    requested_seed_mode = self._requested_seed_mode
                    requested_steps = self._requested_steps
                    requested_guidance_scale = self._requested_guidance_scale
                    requested_one_step_timestep = self._requested_one_step_timestep
                    requested_edge_softness = self._requested_edge_softness
                    requested_img2img_strength = self._requested_img2img_strength
                    requested_edge_strength = self._requested_edge_strength
                    requested_noise_persistence = self._requested_noise_persistence
                    requested_resolution = self._requested_resolution
                if requested_prompt != applied_prompt:
                    backend.set_prompt(*requested_prompt)
                    applied_prompt = requested_prompt
                    previous = None
                    previous_conditioning = None
                applied_prompt_revision = requested_prompt_revision
                if requested_seed != applied_seed:
                    backend.reseed(requested_seed)
                    applied_seed = requested_seed
                    previous = None
                    previous_conditioning = None
                if requested_seed_mode != applied_seed_mode:
                    backend.set_seed_mode(requested_seed_mode)
                    applied_seed_mode = requested_seed_mode
                if requested_steps != applied_steps:
                    backend.set_steps(requested_steps)
                    applied_steps = requested_steps
                if requested_guidance_scale != applied_guidance_scale:
                    backend.set_guidance_scale(requested_guidance_scale)
                    applied_guidance_scale = requested_guidance_scale
                if requested_one_step_timestep != applied_one_step_timestep:
                    backend.set_one_step_timestep(requested_one_step_timestep)
                    applied_one_step_timestep = requested_one_step_timestep
                if requested_edge_softness != applied_edge_softness:
                    backend.set_edge_softness(requested_edge_softness)
                    applied_edge_softness = requested_edge_softness
                if requested_img2img_strength != applied_img2img_strength:
                    backend.set_img2img_strength(requested_img2img_strength)
                    applied_img2img_strength = requested_img2img_strength
                if requested_edge_strength != applied_edge_strength:
                    backend.set_edge_strength(requested_edge_strength)
                    applied_edge_strength = requested_edge_strength
                if requested_noise_persistence != applied_noise_persistence:
                    backend.set_noise_persistence(requested_noise_persistence)
                    applied_noise_persistence = requested_noise_persistence
                if requested_resolution != applied_resolution:
                    backend.set_resolution(*requested_resolution)
                    applied_resolution = requested_resolution
                    previous = None
                    previous_conditioning = None
                    self._set_status(
                        active_resolution=(
                            f"{requested_resolution[0]}x{requested_resolution[1]}"
                        )
                    )
                temporal_state: dict[str, Any] | None = None
                aligned_previous = previous
                temporal_ms = 0.0
                if previous is not None and previous_conditioning is not None:
                    temporal_started = perf_counter()
                    try:
                        aligned_previous, confidence = reproject_previous_image(
                            previous, previous_conditioning, conditioning
                        )
                        temporal_state = {"confidence": confidence}
                    except ValueError:
                        aligned_previous = None
                    temporal_ms = (perf_counter() - temporal_started) * 1000.0
                try:
                    output = backend.generate(
                        conditioning,
                        previous_frame=aligned_previous,
                        temporal_state=temporal_state,
                    )
                except RuntimeError as exc:
                    if self._is_oom(exc) and self._try_lower_resolution(backend):
                        output = backend.generate(conditioning)
                    else:
                        raise
                previous = output
                previous_conditioning = conditioning
                now = perf_counter()
                stats = backend.stats()
                stats["temporal_ms"] = temporal_ms
                stats["prompt_revision"] = applied_prompt_revision
                generated = GeneratedFrame(
                    image=output,
                    depth=conditioning.depth,
                    view_matrix=conditioning.camera.view_matrix,
                    projection_matrix=conditioning.camera.projection_matrix,
                    camera_position=conditioning.camera.position,
                    camera_rotation=conditioning.camera.rotation,
                    generation_timestamp=now,
                    conditioning_timestamp=conditioning.timestamp,
                    sequence=conditioning.sequence,
                    stats=stats,
                )
                self.generated.publish(generated)
                if not logged_first_frame:
                    logging.info("Published first AI frame at sequence %s", conditioning.sequence)
                    logged_first_frame = True
                self._rate.tick(now)
                with self._status_lock:
                    self._backend_stats = stats
        except Exception as exc:  # the render loop must remain alive and show this error
            logging.exception("Diffusion worker failed")
            self._set_status(state="error", message="Diffusion unavailable", error=str(exc))
        finally:
            backend.unload()

    def _try_lower_resolution(self, backend: Any) -> bool:
        if not bool(self.config.get("auto_resolution_fallback", True)):
            return False
        active_label = self.status().active_resolution
        active = tuple(int(value) for value in active_label.split("x", 1))
        lower = {
            (1024, 768): (768, 512),
            (768, 512): (640, 384),
            (512, 512): (640, 384),
            (448, 448): (384, 384),
        }.get(active)
        if lower is None:
            return False
        logging.warning(
            "CUDA OOM at %s; retrying at %sx%s", active_label, lower[0], lower[1]
        )
        if backend.torch is not None:
            backend.torch.cuda.empty_cache()
        backend.set_resolution(*lower)
        self._set_status(
            state="ready",
            message=f"CUDA OOM: reduced resolution to {lower[0]}x{lower[1]}",
            active_resolution=f"{lower[0]}x{lower[1]}",
        )
        return True

    @staticmethod
    def _is_oom(exc: RuntimeError) -> bool:
        return "out of memory" in str(exc).lower()
