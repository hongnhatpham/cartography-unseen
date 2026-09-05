"""Monitor the real interactive app, starting the clock at its first AI frame.

runtime/python/python.exe tools/soak_test.py --minutes 30
Uses a private config copy, enables idle traversal after two seconds, and exits
normally after the requested duration. No synthetic movement or settings input.
Timing quantiles use bounded 1 ms histogram bins, with an exact maximum.
"""
from __future__ import annotations

import argparse
from collections import Counter, deque
from dataclasses import dataclass
import csv
import ctypes
import gc
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import threading
from time import perf_counter, strftime, thread_time_ns
from types import MappingProxyType
import traceback
from unittest.mock import patch
from contextlib import ExitStack

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SPLIT_POLL_WINDOWS_ERROR = (
    "--split-poll is unavailable on Windows: SDL input must stay on its window-owning thread. "
    "Run without --split-poll to monitor production input handling."
)


class Histogram:
    """Fixed storage even during a week-long run. Values are milliseconds."""

    def __init__(self):
        self.bins = [0] * 2002
        self.count = 0
        self.total = self.maximum = 0.0
        self.over = {50: 0, 100: 0, 250: 0}

    def add(self, value: float):
        self.bins[min(2001, max(0, int(value)))] += 1
        self.count += 1
        self.total += value
        self.maximum = max(self.maximum, value)
        for threshold in self.over:
            self.over[threshold] += value > threshold

    def summary(self):
        def percentile(fraction):
            target = math.ceil(self.count * fraction)
            accumulated = 0
            for index, count in enumerate(self.bins):
                accumulated += count
                if accumulated >= target:
                    return index if index < 2001 else self.maximum
            return 0
        return {
            "count": self.count,
            "mean_ms": self.total / self.count if self.count else 0,
            "p50_ms": percentile(.5), "p95_ms": percentile(.95),
            "p99_ms": percentile(.99), "max_ms": self.maximum,
            **{f"over_{key}ms": value for key, value in self.over.items()},
        }


class TimedClock:
    """Delegate to Pygame's real clock, timing only its frame limiter."""

    def __init__(self, clock, timed):
        self._clock = clock
        self.tick = timed(clock.tick, "clock_tick")

    def __getattr__(self, name):
        return getattr(self._clock, name)


def trace_presentation(stack, timed, renderer_type, trail_type, display):
    """Split presentation work without changing its order or GL/input ownership."""
    for target, method, stage in (
        (renderer_type, "_upload_display_image", "_upload_display_image"),
        (renderer_type, "_draw_overlay", "_draw_overlay"),
        (trail_type, "draw", "trail_draw"),
        (display, "flip", "display_flip"),
    ):
        stack.enter_context(patch.object(target, method, timed(getattr(target, method), stage)))


def poll_pygame_events(pygame, monitor):
    """Split the existing single pump from queue conversion for stall diagnosis."""
    timings = []
    events = None
    for name, function in (("event_pump", pygame.event.pump),
                           ("event_get", lambda: pygame.event.get(pump=False))):
        start = perf_counter()
        cpu_start = thread_time_ns()
        identity = monitor.stage_started(name, start)
        try:
            result = function()
            if name == "event_get":
                events = result
        finally:
            cpu_ms = (thread_time_ns() - cpu_start) / 1_000_000
            end = perf_counter()
            details = {"thread_cpu_ms": cpu_ms}
            timings.append((start, end, details))
            if name == "event_get" and events is not None and end - timings[0][0] > .05:
                details.update(
                    event_count=len(events),
                    event_types=dict(Counter(event.type for event in events)),
                    joystick_initialized=pygame.joystick.get_init(),
                    mixer_initialized=pygame.mixer.get_init() is not None,
                )
            monitor.stage_finished(name, start, end, identity, details)
    return events


def process_memory():
    if os.name != "nt":
        return {}
    from ctypes import wintypes

    class Counters(ctypes.Structure):
        _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD)] + [
            (name, ctypes.c_size_t) for name in (
                "PeakWorkingSetSize", "WorkingSetSize", "QuotaPeakPagedPoolUsage",
                "QuotaPagedPoolUsage", "QuotaPeakNonPagedPoolUsage", "QuotaNonPagedPoolUsage",
                "PagefileUsage", "PeakPagefileUsage", "PrivateUsage",
            )
        ]

    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.GetCurrentProcess.restype = wintypes.HANDLE
    query = ctypes.WinDLL("psapi", use_last_error=True).GetProcessMemoryInfo
    query.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
    query.restype = wintypes.BOOL
    counters = Counters()
    counters.cb = ctypes.sizeof(counters)
    if not query(kernel.GetCurrentProcess(), ctypes.byref(counters), counters.cb):
        return {"memory_error": ctypes.get_last_error()}
    return {"rss_mb": counters.WorkingSetSize / 1048576,
            "private_mb": counters.PrivateUsage / 1048576}


def gpu_sample():
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,utilization.gpu,memory.used,temperature.gpu,power.draw",
             "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=5,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        if result.returncode:
            return {"error": result.stderr.strip()}
        keys = ("index", "utilization_pct", "memory_mb", "temperature_c", "power_w")
        return {"devices": [dict(zip(keys, [float(value.strip()) if value.strip().replace('.', '', 1).isdigit()
                                          else value.strip() for value in line.split(',')]))
                            for line in result.stdout.splitlines() if line.strip()]}
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"error": str(exc)}


@dataclass(frozen=True, slots=True)
class StallSnapshot:
    started: float
    previous: float
    now: float
    recent: tuple
    active: tuple
    thread_names: tuple
    counts: tuple
    gc_counts: tuple


class Monitor:
    def __init__(self, output: Path, seconds: float):
        self.output, self.seconds = output, seconds
        self.started = self.ended = None
        self.launched = perf_counter()
        # GC callbacks can run while a metrics allocation already holds this.
        self.lock = threading.RLock()
        self.stop = threading.Event()
        self.minute = {}
        self.total = {}
        self.counts = {}
        self.latest = {}
        self.previous_display = None
        self.previous_publication = None
        self.previous_camera = None
        self.distance = 0.0
        self.altitude_min = math.inf
        self.altitude_max = -math.inf
        self.worker = None
        self.rows = []
        self.resource_first = []
        self.resource_last = []
        self.resource_peak = {}
        self.last_flush = None
        self.error = ""
        self.quit_sent = False
        # Keep only recent stage boundaries; serialize stalls on the sampler.
        self.recent_stages = deque(maxlen=256)
        self.active_stages = {}
        self.thread_names = {}
        self.pending_stalls = deque(maxlen=64)
        self.stalls_dropped = 0

    def timed(self, function, name):
        def wrapped(*args, **kwargs):
            start = perf_counter()
            identity = self.stage_started(name, start)
            try:
                return function(*args, **kwargs)
            finally:
                self.stage_finished(name, start, perf_counter(), identity)
        return wrapped

    def stage_started(self, name, now):
        identity = threading.get_ident()
        with self.lock:
            if identity not in self.thread_names:
                self.thread_names[identity] = threading.current_thread().name
            self.active_stages[identity, name] = now
        return identity

    def stage_finished(self, name, start, end, identity, details=None):
        if details:
            # Metadata is complete at this boundary and read-only thereafter.
            details = MappingProxyType(dict(details))
        with self.lock:
            self.active_stages.pop((identity, name), None)
            self.record(name, (end - start) * 1000)
            self.recent_stages.append((name, start, end, identity, details))

    def displayed(self, now):
        previous, self.previous_display = self.previous_display, now
        if previous is None or self.started is None or self.ended is not None:
            return
        # Exclude a first interval that began before the timed AI run.
        if previous < self.started:
            return
        milliseconds = (now - previous) * 1000
        self.record("display_interval", milliseconds)
        if milliseconds <= 50:
            return
        with self.lock:
            # Copy bounded raw records only. Constructing every stage dictionary
            # here would add work to the next display interval while holding the
            # same lock used by worker/input timing callbacks.
            event = StallSnapshot(self.started, previous, now, tuple(self.recent_stages),
                                  tuple(self.active_stages.items()), tuple(self.thread_names.items()),
                                  tuple(self.counts.items()), gc.get_count())
            if len(self.pending_stalls) == self.pending_stalls.maxlen:
                self.stalls_dropped += 1
            self.pending_stalls.append(event)

    @staticmethod
    def format_stall(snapshot):
        """Expand one immutable snapshot on the sampler, outside the timing lock."""
        started, previous, now = snapshot.started, snapshot.previous, snapshot.now
        thread_names = dict(snapshot.thread_names)

        def describe(name, start, end, identity, details=None):
            finish = end if end is not None else now
            return {"stage": name, "start_s": start - started,
                    "end_s": end - started if end is not None else None,
                    "duration_ms": (finish - start) * 1000,
                    "overlap_ms": max(0, min(finish, now) - max(start, previous)) * 1000,
                    "thread_id": identity, "thread": thread_names.get(identity, "unknown"),
                    **({"details": dict(details)} if details else {})}

        return {"elapsed_s": now - started, "display_interval_ms": (now - previous) * 1000,
                "interval_start_s": previous - started,
                "stages": [describe(*stage) for stage in snapshot.recent if stage[2] >= previous - .25],
                "active_stages": [describe(name, start, None, identity)
                                  for (identity, name), start in snapshot.active],
                "gc_counts": snapshot.gc_counts, "counts": dict(snapshot.counts)}

    def write_stalls(self):
        """Run off the render thread, including the final drain after exit."""
        with self.lock:
            events = list(self.pending_stalls)
            self.pending_stalls.clear()
        if events:
            with (self.output / "stalls.jsonl").open("a", encoding="utf-8") as stream:
                for event in events:
                    stream.write(json.dumps(self.format_stall(event)) + "\n")

    def record(self, name, milliseconds):
        if self.started is None or self.ended is not None:
            return
        with self.lock:
            for group in (self.minute, self.total):
                if name not in group:
                    group[name] = Histogram()
                group[name].add(milliseconds)

    def count(self, name):
        with self.lock:
            self.counts[name] = self.counts.get(name, 0) + 1

    def generated(self, frame):
        now = perf_counter()
        if self.ended is not None:
            return
        if self.started is None:
            self.started = self.last_flush = now
            self.latest["first_frame_wall_time"] = strftime("%Y-%m-%dT%H:%M:%S%z")
        previous, self.previous_publication = self.previous_publication, now
        if previous is not None:
            self.record("ai_publication_interval", (now - previous) * 1000)
        self.count("generated_frames")
        self.latest["backend_stats"] = dict(frame.stats)
        self.latest["last_generated_elapsed_s"] = now - self.started

    def camera(self, position, renderer):
        if self.started is None or self.ended is not None:
            return
        point = tuple(float(value) for value in position)
        if self.previous_camera is not None:
            self.distance += math.dist(point, self.previous_camera)
        self.previous_camera = point
        self.altitude_min = min(self.altitude_min, point[1])
        self.altitude_max = max(self.altitude_max, point[1])
        self.latest["world_chunks"] = len(renderer._chunk_instances)
        self.latest["camera_position"] = point

    def flush(self, now):
        if self.started is None or self.last_flush is None or now <= self.last_flush:
            return
        with self.lock:
            metrics, self.minute = self.minute, {}
        elapsed = now - self.started
        duration = now - self.last_flush
        row = {"elapsed_s": elapsed, "interval_s": duration,
               "metrics": {key: value.summary() for key, value in metrics.items()},
               "counts": dict(self.counts), "latest": dict(self.latest),
               "distance": self.distance}
        row["display_fps"] = metrics.get("display_interval", Histogram()).count / duration
        row["ai_fps"] = metrics.get("generate", Histogram()).count / duration
        row["ai_publication_fps"] = metrics.get("ai_publication_interval", Histogram()).count / duration
        with (self.output / "minutes.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row) + "\n")
        with (self.output / "minutes.csv").open("a", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            if not self.rows:
                writer.writerow(("elapsed_s", "interval_s", "display_fps", "ai_fps", "display_p95_ms",
                                 "display_max_ms", "generate_p95_ms", "private_mb", "rss_mb", "distance",
                                 "ai_publication_fps", "ai_gap_p95_ms", "ai_gap_max_ms"))
            writer.writerow((elapsed, duration, row["display_fps"], row["ai_fps"],
                             row["metrics"].get("display_interval", {}).get("p95_ms"),
                             row["metrics"].get("display_interval", {}).get("max_ms"),
                             row["metrics"].get("generate", {}).get("p95_ms"),
                             self.latest.get("private_mb"), self.latest.get("rss_mb"), self.distance,
                             row["ai_publication_fps"],
                             row["metrics"].get("ai_publication_interval", {}).get("p95_ms"),
                             row["metrics"].get("ai_publication_interval", {}).get("max_ms")))
        # Only the first and most recent five minute rows are needed for trend comparison.
        if duration >= 30:
            self.rows.append({key: row[key] for key in ("elapsed_s", "display_fps", "ai_fps")})
            if len(self.rows) > 10:
                del self.rows[5]
        self.last_flush = now

    def sample(self):
        next_gpu = 0.0
        with (self.output / "resources.jsonl").open("a", encoding="utf-8") as stream:
            while not self.stop.wait(1):
                self.write_stalls()
                now = perf_counter()
                memory = process_memory()
                self.latest.update(memory)
                if self.worker is not None:
                    status = self.worker.status()
                    self.latest["worker_state"] = status.state
                    self.latest["resolution_fallbacks"] = status.resolution_fallbacks
                    if status.error:
                        self.error = status.error
                elapsed = now - self.started if self.started else None
                if elapsed is not None:
                    sample = {"elapsed_s": elapsed, **memory}
                    if now >= next_gpu:
                        sample["gpu"] = gpu_sample()
                        next_gpu = now + 10
                    stream.write(json.dumps(sample) + "\n")
                    stream.flush()
                    if elapsed <= 300:
                        self.resource_first.append(memory)
                    self.resource_last.append((elapsed, memory))
                    self.resource_last = [(at, item) for at, item in self.resource_last if elapsed - at <= 300]
                    for key, value in memory.items():
                        self.resource_peak[key] = max(self.resource_peak.get(key, 0), value)
                    if now - self.last_flush >= 60:
                        self.flush(now)

    def finish(self, exit_code):
        self.ended = self.ended or perf_counter()
        self.stop.set()
        elapsed = self.ended - self.started if self.started else 0
        self.flush(self.ended)
        self.write_stalls()
        ai_gaps = self.total.get("ai_publication_interval", Histogram()).summary()
        terminal_gap = max(0, (self.ended - self.previous_publication) * 1000) \
            if self.previous_publication is not None else 0
        ai_gaps["final_unpublished_gap_ms"] = terminal_gap
        ai_gaps["max_observed_gap_ms"] = max(ai_gaps["max_ms"], terminal_gap)
        def trend(rows):
            return {key: statistics.median(row[key] for row in rows if key in row)
                    for key in ("display_fps", "ai_fps") if rows}
        def memory_median(rows):
            return {key: statistics.median(row[key] for row in rows if key in row)
                    for key in ("rss_mb", "private_mb") if any(key in row for row in rows)}
        summary = {"requested_seconds": self.seconds, "measured_seconds": elapsed,
                   "completed": elapsed >= self.seconds and exit_code == 0 and not self.error,
                   "exit_code": exit_code, "error": self.error,
                   "startup_seconds": self.started - self.launched if self.started else None,
                   "metrics": {key: value.summary() for key, value in self.total.items()},
                   "ai_gaps": ai_gaps,
                   "counts": self.counts, "latest": self.latest, "distance": self.distance,
                   "altitude_min": self.altitude_min if math.isfinite(self.altitude_min) else None,
                   "altitude_max": self.altitude_max if math.isfinite(self.altitude_max) else None,
                   "first_five_minutes": trend([row for row in self.rows if row["elapsed_s"] <= 305]),
                   "last_five_minutes": trend([row for row in self.rows if row["elapsed_s"] > elapsed - 300]),
                   "memory_first_five_minutes": memory_median(self.resource_first),
                   "memory_last_five_minutes": memory_median([item for _, item in self.resource_last]),
                   "memory_peak": self.resource_peak,
                   "quantile_resolution_ms": 1,
                   "quantile_overflow_ms": 2001,
                   "stall_trace": {"file": "stalls.jsonl", "threshold_ms": 50,
                                   "dropped_events": self.stalls_dropped,
                                   "recent_stage_capacity": self.recent_stages.maxlen,
                                   "queue_capacity": self.pending_stalls.maxlen}}
        (self.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        return summary


def run(seconds: float, output: Path, config_path: Path, fixed_settings: bool = False,
        split_poll: bool = False):
    if split_poll and sys.platform == "win32":
        raise ValueError(SPLIT_POLL_WINDOWS_ERROR)
    output.mkdir(parents=True, exist_ok=False)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    overrides = {"autowalk_idle_seconds": 2.0}
    if fixed_settings:
        overrides.update(prompt_auto_advance_seconds=0.0, random_seed_on_launch=False)
    config.update(overrides)
    copied_config = output / "config.json"
    copied_config.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    (output / "run.json").write_text(json.dumps({"source_config": str(config_path), "pid": os.getpid(),
        "requested_seconds": seconds, "launched": strftime("%Y-%m-%dT%H:%M:%S%z"),
        "config_overrides": overrides, "fixed_settings": fixed_settings,
        "split_poll": split_poll}, indent=2), encoding="utf-8")
    from app import main
    from app.config import configure_local_environment
    configure_local_environment(ROOT, offline=True)
    import pygame
    from app.diffusion import worker as worker_module
    from app.renderer.proxy_renderer import ProxyRenderer
    from app.renderer.trail_renderer import TrailRenderer
    from app.renderer.world import Autowalk

    monitor = Monitor(output, seconds)
    original_factory = worker_module.create_backend
    original_init = worker_module.DiffusionWorker.__init__
    original_poll = ProxyRenderer.poll_events
    original_clock = pygame.time.Clock
    timed = monitor.timed
    timed_poll = timed((lambda: poll_pygame_events(pygame, monitor)) if split_poll else original_poll,
                       "poll_events")

    def create_backend(*args, **kwargs):
        backend = original_factory(*args, **kwargs)
        for name in ("generate", "set_prompt"):
            setattr(backend, name, timed(getattr(backend, name), name))
        original_stats = backend.stats
        def stats():
            result = original_stats()
            result["prompt_embedding_cache_entries"] = len(getattr(backend, "_prompt_embeddings", {}))
            return result
        backend.stats = stats
        return backend

    def worker_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        monitor.worker = self
        original_publish = self.generated.publish
        def publish(frame):
            result = original_publish(frame)
            monitor.generated(frame)
            return result
        self.generated.publish = publish

    def poll():
        events = timed_poll()
        for event in events:
            if event.type == pygame.KEYDOWN and event.key in (pygame.K_RETURN, pygame.K_KP_ENTER):
                monitor.count("enter_key_events")
        now = perf_counter()
        if not monitor.quit_sent and ((monitor.started is not None and now - monitor.started >= seconds)
                                     or monitor.error or now - monitor.launched > seconds + 300):
            monitor.quit_sent = True
            monitor.ended = now
            events.append(pygame.event.Event(pygame.QUIT))
        return events

    original_present = ProxyRenderer._present_texture
    def present(self, *args, **kwargs):
        result = original_present(self, *args, **kwargs)
        monitor.displayed(perf_counter())
        return result

    original_update = ProxyRenderer._update_world
    def update(self, position):
        result = original_update(self, position)
        monitor.camera(position, self)
        return result

    original_prompt = worker_module.DiffusionWorker.request_prompt
    def request_prompt(self, *args, **kwargs):
        monitor.count("prompt_requests")
        return original_prompt(self, *args, **kwargs)

    gc_started = {}
    def collection(phase, info):
        identity = threading.get_ident()
        name = f"gc_generation_{info['generation']}"
        if phase == "start":
            gc_started[identity] = perf_counter()
            monitor.stage_started(name, gc_started[identity])
        elif identity in gc_started:
            monitor.stage_finished(name, gc_started.pop(identity), perf_counter(), identity,
                                   {key: info[key] for key in ("collected", "uncollectable")})

    sampler = threading.Thread(target=monitor.sample, name="soak-monitor", daemon=True)
    exit_code = 1
    sampler.start()
    gc.callbacks.append(collection)
    try:
        with ExitStack() as stack:
            trace_presentation(stack, timed, ProxyRenderer, TrailRenderer, pygame.display)
            if fixed_settings:
                # Freeze the initial AI palette as well as sampler settings.
                # World geometry and rendered regional colors still vary normally.
                original_hues = main.hue_words
                initial_hues = []
                def fixed_hues(*args, **kwargs):
                    if not initial_hues:
                        initial_hues.append(original_hues(*args, **kwargs))
                    return initial_hues[0]
                stack.enter_context(patch.object(main, "hue_words", fixed_hues))
                stack.enter_context(patch.object(main, "random_prompt_settings", lambda: {}))
            stack.enter_context(patch.object(sys, "argv", ["app.main", "--config", str(copied_config.resolve())]))
            stack.enter_context(patch.object(worker_module, "create_backend", create_backend))
            stack.enter_context(patch.object(worker_module.DiffusionWorker, "__init__", worker_init))
            stack.enter_context(patch.object(worker_module.DiffusionWorker, "request_prompt", request_prompt))
            stack.enter_context(patch.object(ProxyRenderer, "poll_events", staticmethod(poll)))
            stack.enter_context(patch.object(pygame.event, "wait", timed(pygame.event.wait, "event_wait")))
            stack.enter_context(patch.object(ProxyRenderer, "_present_texture", present))
            stack.enter_context(patch.object(ProxyRenderer, "_update_world", timed(update, "world_update")))
            stack.enter_context(patch.object(Autowalk, "step", timed(Autowalk.step, "autowalk_step")))
            stack.enter_context(patch.object(pygame.time, "Clock",
                                            lambda *args, **kwargs: TimedClock(original_clock(*args, **kwargs), timed)))
            for name in ("capture_conditioning", "render_scene", "display", "display_reprojected",
                         "constrain_camera", "nearby_colliders"):
                stack.enter_context(patch.object(ProxyRenderer, name, timed(getattr(ProxyRenderer, name), name)))
            exit_code = main.run()
    except BaseException:
        monitor.error = traceback.format_exc()
        (output / "failure.txt").write_text(monitor.error, encoding="utf-8")
    finally:
        gc.callbacks.remove(collection)
        monitor.ended = monitor.ended or perf_counter()
        monitor.stop.set()
        sampler.join(timeout=7)
        summary = monitor.finish(exit_code)
    print(json.dumps({key: summary[key] for key in ("completed", "measured_seconds", "error")}))
    return 0 if summary["completed"] else 1


def cli():
    parser = argparse.ArgumentParser(description=__doc__)
    duration = parser.add_mutually_exclusive_group()
    duration.add_argument("--minutes", type=float, default=30)
    duration.add_argument("--seconds", type=float)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--config", type=Path, default=ROOT / "config.json")
    parser.add_argument("--fixed-settings", action="store_true",
                        help="hold initial AI prompt palette, sampler settings and seed; disable auto-advance")
    parser.add_argument("--split-poll", action="store_true",
                        help="non-Windows diagnostic override: replace polling with one pump and queue read")
    args = parser.parse_args()
    if args.split_poll and sys.platform == "win32":
        parser.error(SPLIT_POLL_WINDOWS_ERROR)
    seconds = args.seconds if args.seconds is not None else args.minutes * 60
    if not math.isfinite(seconds) or seconds <= 0:
        parser.error("duration must be positive and finite")
    output = args.output or ROOT / "logs" / "performance" / f"soak-{strftime('%Y%m%d-%H%M%S')}"
    return run(seconds, output.resolve(), args.config.resolve(), args.fixed_settings, args.split_poll)


if __name__ == "__main__":
    raise SystemExit(cli())
