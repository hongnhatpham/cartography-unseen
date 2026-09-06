"""Measure normal play or compare a fixed, elapsed-time camera route.

runtime/python/python.exe tools/replay_performance.py --seconds 120 --max-stall-ms 100
The synthetic route deliberately overrides collision displacement, but executes
the production collision query. Startup does not advance the route. Real event
polling remains enabled; only Quit and Escape affect the automated run.
Add --normal to retain real controls, automatic prompts and saved image settings,
with idle flight after two seconds and only scalar display/publication/generation
timings. Both modes start measurement at the first AI frame.

--map-check off/active compares the same route with real W input and isolated
archives, requesting 1920x1080 windows with diagnostics closed. Verify actual
window_size in run.json and size in map-metrics.json: a tiling window manager
can override the request. The child timings measure CPU submission, not GPU
completion. --map-check cycle releases W from seconds 20 to 40 for autowalk.
--map-check soak/soak-off retains automatic prompt changes for long sessions.
"""
from __future__ import annotations

import argparse
from collections import deque
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


def measured_archive_task(task_name, args, output):
    """Importable worker probe; distinguish encoding CPU time from disk waits."""
    from app import journey
    from time import thread_time
    task = getattr(journey, task_name)
    original_json = journey._atomic_json
    rows = []

    def timed(function, name, *values):
        before, cpu_before = perf_counter(), thread_time()
        try:
            return function(*values)
        finally:
            rows.append((name, before, perf_counter(), (thread_time()-cpu_before)*1000))

    def atomic_json(*values):
        return timed(original_json, "archive_json", *values)

    try:
        with patch.object(journey, "_atomic_json", atomic_json):
            return timed(task, task_name, *args)
    finally:
        with Path(output).open("a", newline="", encoding="utf-8") as stream:
            csv.writer(stream).writerows(rows)


def measurement_config(config: dict, normal: bool = False):
    """Keep the source untouched; normal mode only shortens the idle-flight delay."""
    overrides = ({"autowalk_idle_seconds": 2.0} if normal else {
        "backend": "latent_walk", "diffusion_resolution": "384x256", "guidance_scale": 1.5,
        "prompt_auto_advance_seconds": 0, "random_seed_on_launch": False,
        "debug_overlay": False, "autowalk_idle_seconds": 0,
    })
    return {**config, **overrides}


def measured_map_run(root, poses, maps, notices, stop):
    """Spawn-safe instrumentation, with no framebuffer readback or recording."""
    import pygame
    from app.map_view import _run, _Scene
    from time import process_time
    original_mode, original_draw, original_update = pygame.display.set_mode, _Scene.draw, _Scene.update
    records, updates, details = [], [], {}
    failures = []
    class Notices:
        def put_nowait(self, value):
            if value.startswith("Map window failed:"):
                failures.append(value)
            notices.put_nowait(value)
    static_map = os.environ.get("CARTOGRAPHY_MAP_STATIC") == "1"
    started, cpu_started = perf_counter(), process_time()
    def mode(size, *args, **kwargs):
        return original_mode((1920, 1080), *args, **kwargs)
    def draw(self, pose, angle, size, operator, map_opacity=None):
        if not details:
            details.update(renderer=self.gl.info.get("GL_RENDERER"), size=size,
                           clock_started=started)
        before = perf_counter()
        original_draw(self, pose, angle, size, operator, 0.0 if static_map else map_opacity)
        records.append((before-started, (perf_counter()-before)*1000,
                        pose["active"], map_opacity, len(self.textures)))
    def update(self, snapshot):
        before = perf_counter()
        if static_map:
            self.snapshot = snapshot
        else:
            original_update(self, snapshot)
        updates.append((before-started, (perf_counter()-before)*1000,
                        len(snapshot.get("images", []))))
    with patch.object(pygame.display, "set_mode", mode), patch.object(_Scene, "draw", draw), patch.object(_Scene, "update", update):
        _run(root, poses, maps, Notices(), stop)
    Path(os.environ["CARTOGRAPHY_MAP_METRICS"]).write_text(json.dumps({
        "pid": os.getpid(), "cpu_seconds": process_time()-cpu_started,
        "wall_seconds": perf_counter()-started, "draws": records, "updates": updates,
        "static_map_probe": static_map, "failures": failures, **details,
    }), encoding="utf-8")


def run(seconds: float, output: Path, source: Path, threshold: float, max_ai_gap_ms: float = 250,
        normal: bool = False, map_check: str | None = None):
    output.mkdir(parents=True, exist_ok=False)
    config = measurement_config(json.loads(source.read_text(encoding="utf-8")), normal)
    if map_check:
        config.update(journey_map=map_check not in ("off", "soak-off"), fullscreen=False, debug_overlay=False)
        if map_check in ("soak", "soak-off"):
            config["prompt_auto_advance_seconds"] = json.loads(source.read_text(encoding="utf-8")).get(
                "prompt_auto_advance_seconds", 24.0)
        if map_check == "cycle":
            config["autowalk_idle_seconds"] = 2.0
    config_path = output / "config.json"
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    metadata = dict(pid=os.getpid(), requested_seconds=seconds, source_config=str(source),
                    mode="normal" if normal else "fixed_route",
                    route=("production controls and idle flight" if normal else
                           "elapsed-flight-v1; collision queries run, displacement overridden"),
                    collector=("minimal publication/presentation/generation timestamps" if normal else
                               "publication/presentation/generation and coarse stage timings"),
                    launched=strftime("%Y-%m-%dT%H:%M:%S%z"))
    metadata["archive_pickle_samples"] = []
    if map_check:
        metadata.update(map_check=map_check, main_window_size=[1920, 1080],
                        map_window_size=[1920, 1080],
                        map_input=("W for 20s, release for 20s, then W" if map_check == "cycle" else
                                   "no human input" if map_check == "idle" else "held W"))
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
    recorder = None
    original_worker_init = worker_module.DiffusionWorker.__init__
    original_backend = worker_module.create_backend
    original_poll = ProxyRenderer.poll_events
    original_present = ProxyRenderer._present_texture
    original_constrain = ProxyRenderer.constrain_camera
    original_hues = main.hue_words
    original_renderer_init = ProxyRenderer.__init__
    controls_closed = False
    stop_check_at = 0.0
    stop_requested = False
    initial_hues = []
    gc_events = deque(maxlen=4096)
    gc_started = None

    def gc_event(phase, info):
        nonlocal gc_started
        if phase == "start":
            gc_started = perf_counter()
        elif gc_started is not None:
            # A GC callback may run while Metrics.lock is held. Defer recording.
            gc_events.append((gc_started, perf_counter()))
            gc_started = None

    def worker_init(self, *args, **kwargs):
        nonlocal worker
        original_worker_init(self, *args, **kwargs)
        worker = self
        original_publish = self.generated.publish
        def publish(frame):
            metrics.published(perf_counter())
            metadata["measurement_started"] = metrics.started
            return original_publish(frame)
        self.generated.publish = publish

    def backend_factory(*args, **kwargs):
        backend = original_backend(*args, **kwargs)
        backend.generate = metrics.timed(backend.generate, "generate")
        backend.set_prompt = metrics.timed(backend.set_prompt, "set_prompt")
        return backend

    def poll():
        nonlocal controls_closed, stop_check_at, stop_requested
        events = original_poll() if normal else metrics.timed(original_poll, "poll_events")()
        if not normal:
            events = [event for event in events if event.type == pygame.QUIT or
                      (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE)]
        now = perf_counter()
        while gc_events:
            metrics.record("python_gc", *gc_events.popleft())
        if now >= stop_check_at:
            stop_requested = (output / "stop-requested").exists()
            stop_check_at = now+1
        if stop_requested:
            metadata["stop_reason"] = "requested through stop-requested file"
        if map_check and config["journey_map"] and not controls_closed:
            events.append(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_F1, mod=0))
            controls_closed = True
        if (stop_requested or (metrics.started is not None and now - metrics.started >= seconds)
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

    def renderer_init(self, *args, **kwargs):
        kwargs["window_size"] = (1920, 1080)
        kwargs["window_position"] = (0, 0)
        original_renderer_init(self, *args, **kwargs)

    class ReplayKeys(NoKeys):
        def __getitem__(self, key):
            elapsed = perf_counter()-metrics.started if metrics.started is not None else 0
            moving = map_check != "idle" and not (map_check == "cycle" and 20 <= elapsed < 40)
            return moving and key == pygame.K_w

    def map_inputs(self):
        return (0, 0), ReplayKeys(), (False, False, False)

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
            if map_check:
                import gc
                from multiprocessing.reduction import ForkingPickler
                from app import journey, map_view
                from app.journey import JourneyRecorder
                from app.journey_session import JourneySession
                gc.callbacks.append(gc_event)
                stack.callback(gc.callbacks.remove, gc_event)
                original_dumps = ForkingPickler.dumps
                def pickle_dump(cls, value, protocol=None):
                    if type(value).__name__ != "_CallItem":
                        return original_dumps(value, protocol)
                    before = perf_counter()
                    result = original_dumps(value, protocol)
                    end = perf_counter()
                    metrics.record("archive_pickle", before, end)
                    if metrics.started is not None:
                        metadata["archive_pickle_samples"].append((end-metrics.started, len(result), (end-before)*1000))
                    return result
                stack.enter_context(patch.object(ForkingPickler, "dumps", classmethod(pickle_dump)))
                original_recorder_init = JourneyRecorder.__init__
                def recorder_init(self, root, *args, **kwargs):
                    nonlocal recorder
                    original_recorder_init(self, output / "journeys", *args, **kwargs)
                    recorder = self
                stack.enter_context(patch.object(ProxyRenderer, "__init__", renderer_init))
                stack.enter_context(patch.object(ProxyRenderer, "read_input", map_inputs))
                stack.enter_context(patch.object(JourneyRecorder, "__init__", recorder_init))
                stack.enter_context(patch.dict(os.environ,
                    {"CARTOGRAPHY_MAP_METRICS": str(output / "map-metrics.json"),
                     "CARTOGRAPHY_MAP_STATIC": "1" if map_check == "recorder" else "0"}))
                stack.enter_context(patch.object(map_view, "_run", measured_map_run))
                stack.enter_context(patch.object(JourneySession, "observe",
                    metrics.timed(JourneySession.observe, "journey_observe")))
                stack.enter_context(patch.object(JourneyRecorder, "snapshot",
                    metrics.timed(JourneyRecorder.snapshot, "journey_snapshot")))
                stack.enter_context(patch.object(JourneySession, "offer_frame",
                    metrics.timed(JourneySession.offer_frame, "journey_offer_frame")))
                if hasattr(JourneyRecorder, "_new_executor"):
                    original_executor = JourneyRecorder._new_executor
                    def archive_executor():
                        executor = original_executor()
                        submit = executor.submit
                        def measured_submit(task, *args):
                            return submit(measured_archive_task, task.__name__, args,
                                          str(output / "archive-timings.csv"))
                        executor.submit = measured_submit
                        return executor
                    stack.enter_context(patch.object(JourneyRecorder, "_new_executor", staticmethod(archive_executor)))
                else:
                    stack.enter_context(patch.object(journey, "_atomic_json",
                        metrics.timed(journey._atomic_json, "archive_json")))
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
        if map_check and config["journey_map"]:
            if recorder is not None and getattr(recorder, "nonempty", False):
                manifest_path = recorder.archive_dir / "manifest.json"
                if not manifest_path.exists():
                    error = error or "Journey manifest was not saved"
                else:
                    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                    images = manifest.get("images", [])
                    metadata["archive_images"] = len(images)
                    if (manifest.get("completion_reason") != "quit" or
                            len(images) != len(recorder._data["images"]) or
                            not (recorder.archive_dir / "map.svg").is_file() or
                            not (recorder.archive_dir / "index.html").is_file() or
                            any(not (recorder.archive_dir / item["path"]).is_file() for item in images)):
                        error = error or "Journey final export is incomplete"
            map_metrics = output / "map-metrics.json"
            if not map_metrics.exists():
                error = error or "Map child did not return rendering measurements"
            else:
                observed = json.loads(map_metrics.read_text(encoding="utf-8"))
                metadata["map_actual_size"] = observed.get("size")
                if observed.get("failures"):
                    error = error or "; ".join(observed["failures"])
                elif observed.get("clock_started", 0) + observed["wall_seconds"] < metrics.ended - .25:
                    error = error or "Map child stopped before measurement ended"
                elif not observed["draws"] or (map_check != "idle" and not observed["updates"]):
                    error = error or "Map workload was not exercised"
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        summary = metrics.finish(code, error, threshold, max_ai_gap_ms)
    print(json.dumps(summary, indent=2))
    return 0 if summary["passed"] else 1


def cli():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seconds", type=float, default=120)
    parser.add_argument("--normal", action="store_true",
                        help="measure normal play/settings with idle flight after two seconds; no synthetic route")
    parser.add_argument("--map-check", choices=("off", "active", "idle", "cycle", "recorder", "soak", "soak-off"),
                        help="matched map check; recorder keeps archives/IPC but draws only the cached title")
    parser.add_argument("--max-stall-ms", type=float, default=100)
    parser.add_argument("--max-ai-gap-ms", type=float, default=250,
                        help="fail if new AI image publications pause longer than this")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--config", type=Path, default=ROOT / "config.json")
    args = parser.parse_args()
    if args.normal and args.map_check:
        parser.error("--map-check uses the fixed route and cannot be combined with --normal")
    if any(not math.isfinite(value) or value <= 0 for value in (args.seconds, args.max_stall_ms, args.max_ai_gap_ms)):
        parser.error("duration and stall thresholds must be positive and finite")
    output = args.output or ROOT / "logs/performance" / f"replay-{strftime('%Y%m%d-%H%M%S')}"
    return run(args.seconds, output.resolve(), args.config.resolve(), args.max_stall_ms, args.max_ai_gap_ms,
               args.normal, args.map_check)


if __name__ == "__main__":
    raise SystemExit(cli())
