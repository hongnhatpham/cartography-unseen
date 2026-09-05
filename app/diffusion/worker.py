from __future__ import annotations

import logging
import secrets
import threading
from dataclasses import dataclass, replace
from time import perf_counter
from typing import Any

import numpy as np

from app.config import BACKEND_SETTING_KEYS
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
    # Bumped on every CUDA-OOM downgrade so the render loop can follow it.
    resolution_fallbacks: int = 0


class DiffusionWorker:
    """Runs a diffusion backend on its own thread against the latest proxy frame.

    Interactive changes arrive as a single generic settings dict
    (``request_settings``) plus the four requests that need side effects here:
    prompt, seed, resolution and freeze.
    """

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
        self._requested_prompt = (
            str(config.get("prompt", "")),
            str(config.get("negative_prompt", "")),
        )
        self._prompt_revision = 0
        self._requested_seed = int(config.get("seed", 12345))
        self._requested_resolution = initial_size
        self._pending_settings: dict[str, Any] = {
            key: config[key] for key in BACKEND_SETTING_KEYS if key in config
        }
        self._feedback_reprojection = bool(config.get("feedback_reprojection", False))
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

    def request_settings(self, **kwargs: Any) -> None:
        """Queue live backend settings; the backend ignores keys it has no use for."""
        with self._request_lock:
            self._pending_settings.update(kwargs)
        if "feedback_reprojection" in kwargs:
            # The reprojection itself runs here, not in the backend.
            self._feedback_reprojection = bool(kwargs["feedback_reprojection"])

    def request_prompt(
        self,
        prompt: str,
        negative_prompt: str = "",
        *,
        settings: dict[str, Any] | None = None,
    ) -> int:
        """Queue a prompt and its optional settings together for the next frame."""
        with self._request_lock:
            if settings:
                self._pending_settings.update(settings)
                if "feedback_reprojection" in settings:
                    self._feedback_reprojection = bool(settings["feedback_reprojection"])
            self._requested_prompt = (prompt, negative_prompt)
            self._prompt_revision += 1
            return self._prompt_revision

    def request_reseed(self, seed: int | None = None) -> int:
        value = seed if seed is not None else secrets.randbelow(2**31)
        with self._request_lock:
            self._requested_seed = value
        return value

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
            if self._stop.is_set():
                return
            self._set_status(state="warming", message="Warming GPU pipeline")
            backend.warmup()
            if self._stop.is_set():
                return
            self._set_status(state="ready", message="Ready")
            logging.info("Diffusion backend ready: %s", self.backend_name)
            last_version = 0
            applied_prompt: tuple[str, str] | None = None
            applied_prompt_revision = -1
            applied_seed: int | None = None
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
                    pending_settings = self._pending_settings
                    self._pending_settings = {}
                    requested_prompt = self._requested_prompt
                    requested_prompt_revision = self._prompt_revision
                    requested_seed = self._requested_seed
                    requested_resolution = self._requested_resolution
                # Settings reach the backend before the prompt they apply to, so
                # a new walk duration governs the prompt sent with it.
                if pending_settings:
                    backend.apply_settings(pending_settings)
                if requested_prompt != applied_prompt:
                    backend.set_prompt(*requested_prompt)
                    applied_prompt = requested_prompt
                applied_prompt_revision = requested_prompt_revision
                if requested_seed != applied_seed:
                    backend.reseed(requested_seed)
                    applied_seed = requested_seed
                    previous = None
                    previous_conditioning = None
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
                aligned_previous: np.ndarray | None = None
                temporal_ms = 0.0
                # Reprojection is CPU work in the frame budget, so it only runs
                # when a backend actually consumes the aligned previous frame.
                if (
                    self._feedback_reprojection
                    and previous is not None
                    and previous_conditioning is not None
                ):
                    temporal_started = perf_counter()
                    try:
                        aligned_previous, _ = reproject_previous_image(
                            previous, previous_conditioning, conditioning
                        )
                    except ValueError:
                        aligned_previous = None
                    temporal_ms = (perf_counter() - temporal_started) * 1000.0
                try:
                    output = backend.generate(
                        conditioning, previous_frame=aligned_previous
                    )
                except RuntimeError as exc:
                    lowered = (
                        self._try_lower_resolution(backend) if self._is_oom(exc) else None
                    )
                    if lowered is None:
                        raise
                    applied_resolution = lowered
                    output = backend.generate(conditioning)
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

    def _try_lower_resolution(self, backend: Any) -> tuple[int, int] | None:
        """Drop to the next smaller mode after a CUDA OOM; return it, or None."""
        if not bool(self.config.get("auto_resolution_fallback", True)):
            return None
        status = self.status()
        active_label = status.active_resolution
        active = tuple(int(value) for value in active_label.split("x", 1))
        lower = {
            (1024, 768): (768, 512),
            (768, 512): (640, 384),
            (640, 384): (512, 384),
            (512, 512): (512, 384),
            (512, 384): (384, 256),
        }.get(active)
        if lower is None:
            return None
        logging.warning(
            "CUDA OOM at %s; retrying at %sx%s", active_label, lower[0], lower[1]
        )
        torch = getattr(backend, "torch", None)
        if torch is not None:
            torch.cuda.empty_cache()
        backend.set_resolution(*lower)
        # Keep the request in step so a later resolution change is a real change,
        # and signal the render loop so the proxy follows the backend down.
        with self._request_lock:
            self._requested_resolution = lower
        self._set_status(
            state="ready",
            message=f"CUDA OOM: reduced resolution to {lower[0]}x{lower[1]}",
            active_resolution=f"{lower[0]}x{lower[1]}",
            resolution_fallbacks=status.resolution_fallbacks + 1,
        )
        return lower

    @staticmethod
    def _is_oom(exc: RuntimeError) -> bool:
        return "out of memory" in str(exc).lower()
