"""Measure normal play or compare a fixed, elapsed-time camera route.

runtime/python/python.exe tools/replay_performance.py --seconds 120 --max-stall-ms 100
The synthetic route deliberately overrides collision displacement, but executes
the production collision query. Startup does not advance the route. Real event
polling remains enabled; only Quit and Escape affect the automated run.
Add --normal to retain real controls, automatic prompts and saved image settings,
with idle flight after two seconds and only scalar display/publication/generation
timings. Both modes start measurement at the first AI frame.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import csv
import hashlib
import json
import math
import os
from pathlib import Path
from queue import Empty, Full, Queue
import sys
import threading
from time import perf_counter, strftime
import traceback
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.soak_test import Histogram


def route_pose(elapsed: float, origin, yaw: float, pitch: float, speed: float):
    """Smooth, unbounded three-axis route, independent of display cadence."""
    t = max(0.0, elapsed)
    heading = math.radians(yaw)
    forward = speed * t
    sideways = 30 * (1 - math.cos(t / 12))
    return ((origin[0] + math.sin(heading) * forward + math.cos(heading) * sideways,
             origin[1] + 35 * math.sin(t / 20),
             origin[2] - math.cos(heading) * forward + math.sin(heading) * sideways),
            yaw + 25 * math.sin(t / 15), pitch + 15 * math.sin(t / 20))


class Metrics:
    """Bounded timing queue; CSV serialization never runs on the render thread."""
    def __init__(self, output: Path, seconds: float):
        self.output, self.seconds = output, seconds
        self.started = self.ended = self.previous_display = None
        self.previous_publication = None
        self.histograms = {}
        self.queue = Queue(maxsize=8192)
        self.lock = threading.Lock()
        self.stop = threading.Event()
        self.dropped = 0
        self.writer_error = ""
        self.writer = threading.Thread(target=self._write, name="replay-metrics", daemon=True)

    def _write(self):
        try:
            with (self.output / "timings.csv").open("w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow(("stage", "elapsed_s", "duration_ms"))
                while not self.stop.is_set() or not self.queue.empty():
                    try:
                        writer.writerow(self.queue.get(timeout=.1))
                    except Empty:
                        handle.flush()
        except Exception:
            self.writer_error = traceback.format_exc()

    def record(self, stage: str, start: float, end: float):
        if self.started is None or start < self.started or self.ended is not None:
            return
        ms = (end - start) * 1000
        with self.lock:
            if stage not in self.histograms:
                self.histograms[stage] = Histogram()
            self.histograms[stage].add(ms)
        try:
            self.queue.put_nowait((stage, end - self.started, ms))
        except Full:
            self.dropped += 1

    def timed(self, function, stage):
        def wrapped(*args, **kwargs):
            start = perf_counter()
            try:
                return function(*args, **kwargs)
            finally:
                self.record(stage, start, perf_counter())
        return wrapped

    def displayed(self, now: float):
        previous, self.previous_display = self.previous_display, now
        if previous is not None:
            self.record("display_interval", previous, now)

    def published(self, now: float):
        """Measure new AI output arrival, independently of repeated presentation."""
        if self.ended is not None:
            return
        previous, self.previous_publication = self.previous_publication, now
        if self.started is None:
            self.started = now
        if previous is not None:
            self.record("ai_publication_interval", previous, now)

    def finish(self, exit_code: int, error: str, threshold: float, ai_threshold: float | None = None):
        self.stop.set()
        self.writer.join(timeout=5)
        duration = max(0, (self.ended or perf_counter()) - self.started) if self.started else 0
        completed = duration >= self.seconds and exit_code == 0 and not error
        timings = {key: value.summary() for key, value in self.histograms.items()}
        display = timings.get("display_interval", {})
        ai_gaps = dict(timings.get("ai_publication_interval", {}))
        terminal_gap = max(0, ((self.ended or perf_counter()) - self.previous_publication) * 1000) \
            if self.previous_publication is not None else 0
        ai_gaps["final_unpublished_gap_ms"] = terminal_gap
        ai_gaps["max_observed_gap_ms"] = max(ai_gaps.get("max_ms", 0), terminal_gap)
        ai_passed = ai_threshold is None or (ai_gaps.get("count", 0) > 0 and
                                            ai_gaps["max_observed_gap_ms"] <= ai_threshold)
        passed = (completed and bool(display) and display["max_ms"] <= threshold
                  and ai_passed and not self.dropped and not self.writer_error and not self.writer.is_alive())
        summary = dict(completed=completed, passed=passed, measured_seconds=duration,
                       max_stall_threshold_ms=threshold, max_ai_gap_threshold_ms=ai_threshold,
                       ai_gaps=ai_gaps, timings=timings, error=error,
                       dropped_timing_rows=self.dropped, writer_error=self.writer_error,
                       stalls_over_100ms_per_minute=display.get("over_100ms", 0) * 60 / duration if duration else 0)
        (self.output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        return summary


class NoKeys:
    def __getitem__(self, key):
        return False

    def __iter__(self):
        return iter(())


def measurement_config(config: dict, normal: bool = False):
    """Keep the source untouched; normal mode only shortens the idle-flight delay."""
    overrides = ({"autowalk_idle_seconds": 2.0} if normal else {
        "backend": "latent_walk", "diffusion_resolution": "384x256", "guidance_scale": 1.5,
        "prompt_auto_advance_seconds": 0, "random_seed_on_launch": False,
        "debug_overlay": False, "autowalk_idle_seconds": 0,
    })
    return {**config, **overrides}


def run(seconds: float, output: Path, source: Path, threshold: float, max_ai_gap_ms: float = 250,
        normal: bool = False):
    output.mkdir(parents=True, exist_ok=False)
    config = measurement_config(json.loads(source.read_text(encoding="utf-8")), normal)
    config_path = output / "config.json"
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    metadata = dict(pid=os.getpid(), requested_seconds=seconds, source_config=str(source),
                    mode="normal" if normal else "fixed_route",
                    route=("production controls and idle flight" if normal else
                           "elapsed-flight-v1; collision queries run, displacement overridden"),
                    collector=("minimal publication/presentation/generation timestamps" if normal else
                               "publication/presentation/generation and coarse stage timings"),
                    launched=strftime("%Y-%m-%dT%H:%M:%S%z"))
    metadata["source_sha256"] = {
        str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
        for folder in (ROOT / "app", ROOT / "shaders")
        for path in sorted(folder.rglob("*"))
        if path.suffix in (".py", ".vert", ".frag", ".geom")
    }
    metadata_path = output / "run.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    from app import main
    main.configure_local_environment(ROOT, offline=True)
    import pygame
    from app.diffusion import worker as worker_module
    from app.renderer.camera import Camera
    from app.renderer.proxy_renderer import ProxyRenderer

    metrics = Metrics(output, seconds)
    launched = perf_counter()
    origin = None
    worker = None
    original_worker_init = worker_module.DiffusionWorker.__init__
    original_backend = worker_module.create_backend
    original_poll = ProxyRenderer.poll_events
    original_present = ProxyRenderer._present_texture
    original_constrain = ProxyRenderer.constrain_camera
    original_hues = main.hue_words
    initial_hues = []

    def worker_init(self, *args, **kwargs):
        nonlocal worker
        original_worker_init(self, *args, **kwargs)
        worker = self
        original_publish = self.generated.publish
        def publish(frame):
            metrics.published(perf_counter())
            return original_publish(frame)
        self.generated.publish = publish

    def backend_factory(*args, **kwargs):
        backend = original_backend(*args, **kwargs)
        backend.generate = metrics.timed(backend.generate, "generate")
        return backend

    def poll():
        events = original_poll() if normal else metrics.timed(original_poll, "poll_events")()
        if not normal:
            events = [event for event in events if event.type == pygame.QUIT or
                      (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE)]
        now = perf_counter()
        if ((metrics.started is not None and now - metrics.started >= seconds)
                or now - launched > seconds + 300):
            metrics.ended = now
            events.append(pygame.event.Event(pygame.QUIT))
        return events

    def fly(camera, *args, **kwargs):
        nonlocal origin
        if origin is None:
            origin = (tuple(camera.position), camera.yaw, camera.pitch)
        elapsed = perf_counter() - metrics.started if metrics.started is not None else 0
        position, camera.yaw, camera.pitch = route_pose(elapsed, *origin, config["movement_speed"])
        camera.position[:] = position

    def constrain(renderer, camera, *args, **kwargs):
        position = camera.position.copy()
        result = metrics.timed(original_constrain, "constrain_camera")(renderer, camera, *args, **kwargs)
        camera.position[:] = position
        return result

    def present(renderer, *args, **kwargs):
        if "GL_RENDERER" not in metadata:
            metadata.update({key: renderer.ctx.info.get(key) for key in ("GL_RENDERER", "GL_VENDOR", "GL_VERSION")})
            metadata["window_size"] = renderer.window_size
            metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        result = original_present(renderer, *args, **kwargs)
        metrics.displayed(perf_counter())
        return result

    def hues(*args, **kwargs):
        if not initial_hues:
            initial_hues.append(original_hues(*args, **kwargs))
        return initial_hues[0]

    metrics.writer.start()
    code, error = 1, ""
    try:
        with ExitStack() as stack:
            for target, name, value in (
                (sys, "argv", ["app.main", "--config", str(config_path.resolve())]),
                (worker_module, "create_backend", backend_factory),
                (worker_module.DiffusionWorker, "__init__", worker_init),
                (ProxyRenderer, "poll_events", staticmethod(poll)),
                (ProxyRenderer, "_present_texture", present),
            ):
                stack.enter_context(patch.object(target, name, value))
            if not normal:
                for target, name, value in (
                    (ProxyRenderer, "constrain_camera", constrain),
                    (ProxyRenderer, "read_input", lambda self: ((0, 0), NoKeys(), (False, False, False))),
                    (Camera, "fly", fly), (Camera, "rotate", lambda *args: None),
                    (main, "hue_words", hues),
                ):
                    stack.enter_context(patch.object(target, name, value))
                for name in ("render_scene", "capture_conditioning", "_update_world"):
                    stack.enter_context(patch.object(ProxyRenderer, name, metrics.timed(getattr(ProxyRenderer, name), name)))
            code = main.run()
    except BaseException:
        error = traceback.format_exc()
    finally:
        metrics.ended = metrics.ended or perf_counter()
        if worker is not None:
            status = worker.status()
            metadata["worker_status"] = str(status)
            metadata["backend_stats"] = worker.stats()
            if status.error or status.resolution_fallbacks:
                error = error or str(status)
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        summary = metrics.finish(code, error, threshold, max_ai_gap_ms)
    print(json.dumps(summary, indent=2))
    return 0 if summary["passed"] else 1


def cli():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seconds", type=float, default=120)
    parser.add_argument("--normal", action="store_true",
                        help="measure normal play/settings with idle flight after two seconds; no synthetic route")
    parser.add_argument("--max-stall-ms", type=float, default=100)
    parser.add_argument("--max-ai-gap-ms", type=float, default=250,
                        help="fail if new AI image publications pause longer than this")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--config", type=Path, default=ROOT / "config.json")
    args = parser.parse_args()
    if any(not math.isfinite(value) or value <= 0 for value in (args.seconds, args.max_stall_ms, args.max_ai_gap_ms)):
        parser.error("duration and stall thresholds must be positive and finite")
    output = args.output or ROOT / "logs/performance" / f"replay-{strftime('%Y%m%d-%H%M%S')}"
    return run(args.seconds, output.resolve(), args.config.resolve(), args.max_stall_ms, args.max_ai_gap_ms,
               args.normal)


if __name__ == "__main__":
    raise SystemExit(cli())
