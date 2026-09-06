"""Run real AI/map traversal with periodic telemetry and scoped desktop setup.

Uses the replay's synthetic camera route and human input, keeps automatic prompt
changes, and accumulates one archive for the entire requested duration.
"""
from __future__ import annotations

import argparse
import ctypes
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import replay_performance as replay


def cli():
    import psutil
    from app.journey import JourneyRecorder

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seconds", type=float, default=7200)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not math.isfinite(args.seconds) or args.seconds <= 0:
        parser.error("--seconds must be positive and finite")
    output = args.output.resolve()
    state, finished = {}, threading.Event()
    original_metrics_init = replay.Metrics.__init__
    original_recorder_init = JourneyRecorder.__init__

    def metrics_init(self, *values, **kwargs):
        original_metrics_init(self, *values, **kwargs)
        state["metrics"] = self

    def recorder_init(self, *values, **kwargs):
        original_recorder_init(self, *values, **kwargs)
        state["recorder"] = self

    def monitor():
        parent = psutil.Process()
        while not finished.wait(10):
            metrics = state.get("metrics")
            if metrics is None:
                continue
            try:
                with metrics.lock:
                    timings = {name: histogram.summary() for name, histogram in metrics.histograms.items()}
                processes = []
                for process in [parent, *parent.children(recursive=True)]:
                    try:
                        cpu = process.cpu_times()
                        processes.append(dict(pid=process.pid, name=process.name(), rss=process.memory_info().rss,
                                              cpu_seconds=cpu.user+cpu.system))
                    except psutil.Error:
                        pass
                recorder = state.get("recorder")
                data = recorder._data if recorder else {}
                gpu = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,temperature.gpu,power.draw",
                                      "--format=csv,noheader,nounits"], capture_output=True, text=True,
                                     timeout=5, creationflags=subprocess.CREATE_NO_WINDOW)
                report = dict(time=time.time(), elapsed_s=time.perf_counter()-metrics.started if metrics.started else 0,
                              processes=processes, timings=timings, gpu=gpu.stdout.strip(),
                              disk_free=shutil.disk_usage(output).free, images=len(data.get("images", [])),
                              points=sum(len(segment["points"]) for segment in data.get("segments", [])),
                              prompts=len(data.get("prompts", [])), archive_error=recorder._error if recorder else None,
                              dropped_timing_rows=metrics.dropped, writer_error=metrics.writer_error)
                with (output / "resources.jsonl").open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(report)+"\n")
                temporary = output / "progress.json.tmp"
                temporary.write_text(json.dumps(report, indent=2), encoding="utf-8")
                temporary.replace(output / "progress.json")
            except Exception as error:
                with (output / "monitor-errors.txt").open("a", encoding="utf-8") as stream:
                    stream.write(f"{type(error).__name__}: {error}\n")

    # Keep the display and system awake only for this test's lifetime.
    ctypes.windll.kernel32.SetThreadExecutionState(0x80000003)
    thread = threading.Thread(target=monitor, name="soak-resources", daemon=True)
    thread.start()
    try:
        with patch.object(replay.Metrics, "__init__", metrics_init), patch.object(
                JourneyRecorder, "__init__", recorder_init):
            return replay.run(args.seconds, output, ROOT / "config.json", 100, map_check="soak")
    finally:
        finished.set()
        thread.join(timeout=6)
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)


if __name__ == "__main__":
    raise SystemExit(cli())
