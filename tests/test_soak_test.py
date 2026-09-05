import json
from contextlib import ExitStack
import threading
from types import SimpleNamespace

import pytest

from tools.soak_test import Histogram, Monitor, TimedClock


def test_histogram_tracks_stalls_and_bounds_storage():
    histogram = Histogram()
    for value in (16.7, 17.1, 50.1, 100.1, 250.1, 9000):
        histogram.add(value)
    result = histogram.summary()
    assert result["count"] == 6
    assert result["p50_ms"] == 50
    assert result["over_50ms"] == 4
    assert result["over_100ms"] == 3
    assert result["over_250ms"] == 2
    assert result["max_ms"] == 9000
    assert len(histogram.bins) == 2002


def test_monitor_excludes_startup_and_requires_full_duration(tmp_path):
    monitor = Monitor(tmp_path, 1800)
    monitor.record("display_interval", 5000)
    assert not monitor.total
    monitor.started = monitor.last_flush = monitor.launched
    monitor.record("display_interval", 16)
    monitor.ended = monitor.started + 10
    summary = monitor.finish(0)
    assert not summary["completed"]
    assert summary["measured_seconds"] == 10
    assert summary["metrics"]["display_interval"]["count"] == 1
    assert json.loads((tmp_path / "summary.json").read_text())["completed"] is False


def test_monitor_counts_success_only_without_worker_failure(tmp_path):
    monitor = Monitor(tmp_path, 5)
    monitor.started = monitor.last_flush = monitor.launched
    monitor.ended = monitor.started + 5
    assert monitor.finish(0)["completed"]
    monitor.error = "CUDA out of memory"
    assert not monitor.finish(0)["completed"]


def test_publication_gaps_exclude_startup_and_report_final_frozen_image(monkeypatch, tmp_path):
    from tools import soak_test
    monitor = Monitor(tmp_path, 1)
    frame = SimpleNamespace(stats={"inference_ms": 50})
    times = iter((100.0, 100.1, 100.45, 102.0))
    monkeypatch.setattr(soak_test, "perf_counter", lambda: next(times))
    monitor.generated(frame)
    assert monitor.started == 100
    assert "ai_publication_interval" not in monitor.total
    monitor.generated(frame)
    monitor.generated(frame)
    monitor.ended = 101
    # Output during cleanup must not erase the gap visible at the run's end.
    monitor.generated(frame)
    summary = monitor.finish(0)
    assert summary["completed"]
    assert summary["counts"]["generated_frames"] == 3
    assert summary["ai_gaps"]["count"] == 2
    assert summary["ai_gaps"]["max_ms"] == pytest.approx(350)
    assert summary["ai_gaps"]["over_250ms"] == 1
    assert summary["ai_gaps"]["final_unpublished_gap_ms"] == pytest.approx(550)
    assert summary["ai_gaps"]["max_observed_gap_ms"] == pytest.approx(550)
    minute = json.loads((tmp_path / "minutes.jsonl").read_text())
    assert minute["metrics"]["ai_publication_interval"]["count"] == 2
    assert minute["ai_publication_fps"] == 2
    assert "ai_gap_max_ms" in (tmp_path / "minutes.csv").read_text().splitlines()[0]


def test_single_published_image_reports_entire_unpublished_tail(monkeypatch, tmp_path):
    from tools import soak_test
    monitor = Monitor(tmp_path, 1)
    monkeypatch.setattr(soak_test, "perf_counter", lambda: 100)
    monitor.generated(SimpleNamespace(stats={}))
    monitor.ended = 101
    gaps = monitor.finish(0)["ai_gaps"]
    assert gaps["count"] == 0
    assert gaps["max_observed_gap_ms"] == 1000


def test_inner_display_probes_preserve_calls_results_and_restore_methods(monkeypatch, tmp_path):
    from tools import soak_test
    monitor = Monitor(tmp_path, 1)
    monitor.started = 100
    times = iter((100, 100.06, 100.07, 100.072, 100.08, 100.085, 100.09, 100.12))
    monkeypatch.setattr(soak_test, "perf_counter", lambda: next(times))
    calls = []

    class Renderer:
        def _upload_display_image(self, image, *, texture):
            calls.append(("upload", image, texture))
            return "uploaded"

        def _draw_overlay(self, lines):
            calls.append(("overlay", lines))

    class Trail:
        def draw(self, points):
            calls.append(("trail", points))

    display = SimpleNamespace(flip=lambda: calls.append(("flip",)))
    original_upload = Renderer._upload_display_image
    with ExitStack() as stack:
        soak_test.trace_presentation(stack, monitor.timed, Renderer, Trail, display)
        renderer = Renderer()
        assert renderer._upload_display_image("pixels", texture="target") == "uploaded"
        renderer._draw_overlay(["title"])
        Trail().draw("route")
        display.flip()
    assert Renderer._upload_display_image is original_upload
    assert calls == [("upload", "pixels", "target"), ("overlay", ["title"]),
                     ("trail", "route"), ("flip",)]
    assert set(monitor.total) == {"_upload_display_image", "_draw_overlay", "trail_draw", "display_flip"}
    assert monitor.total["_upload_display_image"].maximum == pytest.approx(60)
    assert monitor.total["display_flip"].maximum == pytest.approx(30)
    assert not monitor.active_stages


def test_stall_trace_correlates_overlapping_gc_and_active_worker_without_disk_io(tmp_path):
    monitor = Monitor(tmp_path, 5)
    monitor.started = monitor.last_flush = 100.0
    monitor.displayed(100.1)
    monitor.stage_started("generate", 100.09)
    identity = monitor.stage_started("gc_generation_2", 100.11)
    monitor.stage_finished("gc_generation_2", 100.11, 100.17, identity,
                           {"collected": 12, "uncollectable": 0})
    monitor.displayed(100.18)
    assert not (tmp_path / "stalls.jsonl").exists()
    monitor.write_stalls()
    rows = [json.loads(line) for line in (tmp_path / "stalls.jsonl").read_text().splitlines()]
    assert len(rows) == 1
    event = rows[0]
    assert event["display_interval_ms"] == pytest.approx(80)
    assert event["elapsed_s"] == pytest.approx(.18)
    collection = event["stages"][0]
    assert collection["stage"] == "gc_generation_2"
    assert collection["overlap_ms"] == pytest.approx(60)
    assert collection["thread"] == threading.current_thread().name
    assert collection["details"]["collected"] == 12
    assert event["active_stages"][0]["stage"] == "generate"
    assert event["active_stages"][0]["end_s"] is None
    assert event["active_stages"][0]["overlap_ms"] == pytest.approx(80)
    assert not monitor.pending_stalls


def test_stall_trace_bounds_storage_and_reports_dropped_events(tmp_path):
    monitor = Monitor(tmp_path, 5)
    monitor.started = monitor.last_flush = 100.0
    monitor.displayed(100.0)
    for index in range(300):
        now = 100.0 + index / 10
        identity = monitor.stage_started("world_update", now)
        monitor.stage_finished("world_update", now, now + .08, identity)
        monitor.displayed(now + .1)
    assert len(monitor.recent_stages) == 256
    assert len(monitor.pending_stalls) == 64
    assert monitor.stalls_dropped == 236
    assert not monitor.active_stages
    monitor.ended = 130.0
    summary = monitor.finish(0)
    assert summary["stall_trace"]["dropped_events"] == 236
    assert len((tmp_path / "stalls.jsonl").read_text().splitlines()) == 64


def test_stall_trace_excludes_startup_and_normal_intervals(tmp_path):
    monitor = Monitor(tmp_path, 5)
    monitor.displayed(90.0)
    monitor.started = monitor.last_flush = 100.0
    monitor.displayed(100.1)
    monitor.displayed(100.116)
    assert not monitor.pending_stalls
    assert monitor.total["display_interval"].count == 1


def test_stall_formatting_is_deferred_and_uses_capture_time_state(monkeypatch, tmp_path):
    monitor = Monitor(tmp_path, 5)
    monitor.started = monitor.last_flush = 100
    monitor.displayed(100.1)
    identity = monitor.stage_started("generate", 100.09)
    monitor.count("generated_frames")
    formatted = []
    original = monitor.format_stall

    def format_stall(snapshot):
        # A worker can still record timing while the sampler expands its event.
        assert not monitor.lock._is_owned()
        formatted.append(snapshot)
        return original(snapshot)

    monkeypatch.setattr(monitor, "format_stall", format_stall)
    monitor.displayed(100.18)
    assert len(monitor.pending_stalls) == 1
    assert formatted == []
    # Mutating live state after the stall must not rewrite that past snapshot.
    monitor.stage_finished("generate", 100.09, 100.2, identity)
    monitor.count("generated_frames")
    monitor.thread_names[identity] = "later name"
    monitor.write_stalls()
    assert len(formatted) == 1
    event = json.loads((tmp_path / "stalls.jsonl").read_text())
    assert event["counts"] == {"generated_frames": 1}
    assert event["stages"] == []
    active = event["active_stages"][0]
    assert active["thread"] == threading.current_thread().name
    assert active["end_s"] is None
    assert active["duration_ms"] == pytest.approx(90)
    assert active["overlap_ms"] == pytest.approx(80)


def test_cli_forwards_fixed_settings_without_loading_app(monkeypatch, tmp_path):
    from tools import soak_test
    calls = []
    monkeypatch.setattr(soak_test.sys, "argv", ["soak_test", "--seconds", "3", "--fixed-settings",
                                               "--output", str(tmp_path)])
    monkeypatch.setattr(soak_test, "run", lambda *args: calls.append(args) or 0)
    assert soak_test.cli() == 0
    assert calls[0][0] == 3
    assert calls[0][-2] is True
    assert calls[0][-1] is False


def test_split_poll_override_requires_explicit_cli_flag(monkeypatch, tmp_path):
    from tools import soak_test
    calls = []
    monkeypatch.setattr(soak_test.sys, "platform", "linux")
    monkeypatch.setattr(soak_test.sys, "argv", ["soak_test", "--seconds", "3", "--split-poll",
                                               "--output", str(tmp_path)])
    monkeypatch.setattr(soak_test, "run", lambda *args: calls.append(args) or 0)
    assert soak_test.cli() == 0
    assert calls[0][-2:] == (False, True)


def test_split_poll_rejected_before_starting_windows_app(monkeypatch, tmp_path, capsys):
    from tools import soak_test
    monkeypatch.setattr(soak_test.sys, "platform", "win32")
    monkeypatch.setattr(soak_test.sys, "argv", ["soak_test", "--split-poll"])
    with pytest.raises(SystemExit) as caught:
        soak_test.cli()
    assert caught.value.code == 2
    assert "window-owning thread" in capsys.readouterr().err
    output = tmp_path / "unused"
    with pytest.raises(ValueError, match="window-owning thread"):
        soak_test.run(120, output, tmp_path / "missing-config.json", split_poll=True)
    assert not output.exists()


def test_clock_proxy_times_limiter_but_delegates_other_clock_behavior(monkeypatch, tmp_path):
    from tools import soak_test
    monitor = Monitor(tmp_path, 5)
    monitor.started = monitor.last_flush = 100.0
    times = iter((100.0, 100.097))
    monkeypatch.setattr(soak_test, "perf_counter", lambda: next(times))

    class Clock:
        def tick(self, framerate=0):
            self.framerate = framerate
            return 97

        def get_time(self):
            return 97

    original = Clock()
    proxy = TimedClock(original, monitor.timed)
    assert proxy.tick(framerate=60) == 97
    assert original.framerate == 60
    assert proxy.get_time() == 97
    assert proxy.framerate == 60
    assert monitor.total["clock_tick"].maximum == pytest.approx(97)
    assert set(monitor.total) == {"clock_tick"}
    assert not monitor.active_stages


def test_timing_records_failure_without_swallowing_it(monkeypatch, tmp_path):
    from tools import soak_test
    monitor = Monitor(tmp_path, 5)
    monitor.started = monitor.last_flush = 100.0
    times = iter((100.0, 100.08))
    monkeypatch.setattr(soak_test, "perf_counter", lambda: next(times))

    def failing_navigation():
        raise ValueError("navigation failure")

    with pytest.raises(ValueError, match="navigation failure"):
        monitor.timed(failing_navigation, "autowalk_step")()
    assert monitor.total["autowalk_step"].maximum == pytest.approx(80)
    assert not monitor.active_stages


def test_poll_probe_pumps_once_and_records_cpu_and_slow_event_details(monkeypatch, tmp_path):
    from tools import soak_test
    monitor = Monitor(tmp_path, 5)
    monitor.started = monitor.last_flush = 100.0
    wall_times = iter((100.0, 100.11, 100.111, 100.112))
    cpu_times = iter((0, 2_000_000, 2_000_000, 3_000_000))
    monkeypatch.setattr(soak_test, "perf_counter", lambda: next(wall_times))
    monkeypatch.setattr(soak_test, "thread_time_ns", lambda: next(cpu_times))
    calls = []
    events = [SimpleNamespace(type=1), SimpleNamespace(type=1), SimpleNamespace(type=2)]

    def get(*, pump):
        calls.append(("get", pump))
        return events

    pygame = SimpleNamespace(
        event=SimpleNamespace(pump=lambda: calls.append("pump"), get=get),
        joystick=SimpleNamespace(get_init=lambda: False),
        mixer=SimpleNamespace(get_init=lambda: None),
    )
    assert soak_test.poll_pygame_events(pygame, monitor) is events
    assert calls == ["pump", ("get", False)]
    pump_stage, get_stage = monitor.recent_stages
    assert pump_stage[0] == "event_pump"
    assert pump_stage[-1] == {"thread_cpu_ms": 2}
    assert get_stage[0] == "event_get"
    assert get_stage[-1] == {"thread_cpu_ms": 1, "event_count": 3,
                             "event_types": {1: 2, 2: 1}, "joystick_initialized": False,
                             "mixer_initialized": False}
    assert monitor.total["event_pump"].maximum == pytest.approx(110)
    assert not monitor.active_stages
