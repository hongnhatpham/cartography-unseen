"""Summarize a finished replay without loading its entire timing CSV into memory.

Usage: runtime/python/python.exe tools/summarize_long_map.py RUN_DIRECTORY
The final summary is required unless --interrupted explicitly permits recovery
from periodic telemetry. Recovery never labels an interrupted replay complete.
"""
from __future__ import annotations

import argparse
from array import array
import csv
import json
import math
from pathlib import Path

import numpy as np


INTERVAL_STAGES = ("display_interval", "ai_publication_interval")
BIN_SECONDS = 600


def read_json(path: Path) -> dict:
    with path.open(encoding="utf-8-sig") as stream:
        return json.load(stream)


def statistics(values, *, cadence: bool = False) -> dict:
    samples = np.asarray(values, dtype=float)
    if not samples.size:
        return {"count": 0}
    mean = float(samples.mean())
    result = {
        "count": int(samples.size), "mean_ms": mean,
        "p95_ms": float(np.quantile(samples, .95)),
        "p99_ms": float(np.quantile(samples, .99)), "max_ms": float(samples.max()),
        "over_100ms": int(np.count_nonzero(samples > 100)),
        "over_250ms": int(np.count_nonzero(samples > 250)),
    }
    if cadence:
        result["cadence_hz"] = 1000 / mean if mean > 0 else None
    return result


def read_timings(path: Path, measured_seconds: float | None):
    """Retain only primary intervals and archive windows; stream other stages."""
    # The replay writer terminates every row. An interrupted partial number can
    # otherwise parse as a valid float and silently change the final duration.
    with path.open("rb") as stream:
        if stream.seek(0, 2):
            stream.seek(-1, 2)
            if stream.read(1) != b"\n":
                raise ValueError(f"Unterminated final timing row in {path}; possible interrupted CSV write")
    samples = {stage: {} for stage in INTERVAL_STAGES}
    display_stalls, archive_intervals = [], []
    row_counts = {}
    maximum_elapsed = 0.0
    with path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        if not {"stage", "elapsed_s", "duration_ms"}.issubset(reader.fieldnames or []):
            raise ValueError(f"Unexpected timing columns in {path}")
        for line_number, row in enumerate(reader, 2):
            if None in row or any(value is None for value in row.values()):
                raise ValueError(f"Partial or malformed timing row at {path}:{line_number}")
            stage = row["stage"]
            row_counts[stage] = row_counts.get(stage, 0) + 1
            try:
                elapsed, duration = float(row["elapsed_s"]), float(row["duration_ms"])
            except ValueError as error:
                raise ValueError(f"Invalid timing at {path}:{line_number}: {error}") from error
            if not math.isfinite(elapsed) or not math.isfinite(duration) or min(elapsed, duration) < 0:
                raise ValueError(f"Invalid timing at {path}:{line_number}")
            maximum_elapsed = max(maximum_elapsed, elapsed)
            if stage not in INTERVAL_STAGES and stage != "archive_json":
                continue
            start = elapsed-duration/1000
            if stage == "archive_json":
                archive_intervals.append((start, elapsed))
                continue
            bucket = int(elapsed // BIN_SECONDS)
            samples[stage].setdefault(bucket, array("d")).append(duration)
            if stage == "display_interval" and duration > 100:
                display_stalls.append((start, elapsed))
    full = {}
    for stage, buckets in samples.items():
        combined = array("d")
        for bucket in buckets.values():
            combined.extend(bucket)
        full[stage] = statistics(combined, cadence=True)
    if measured_seconds is None:
        measured_seconds = maximum_elapsed
    last_bucket = max([int(max(0, math.ceil(measured_seconds/BIN_SECONDS)-1)),
                       *(key for buckets in samples.values() for key in buckets)])
    bins = []
    for bucket in range(last_bucket+1):
        bins.append({
            "start_s": bucket*BIN_SECONDS,
            "end_s": min((bucket+1)*BIN_SECONDS, measured_seconds),
            **{stage: statistics(buckets.get(bucket, ()), cadence=True)
               for stage, buckets in samples.items()},
        })
    return full, bins, display_stalls, archive_intervals, row_counts, maximum_elapsed


def archive_worker_timings(path: Path, measurement_started: float | None):
    """Worker rows have no header: stage, absolute start/end, thread CPU ms."""
    if not path.exists():
        return None, []
    stages, intervals = {}, []
    with path.open(newline="", encoding="utf-8-sig") as stream:
        for line_number, row in enumerate(csv.reader(stream), 1):
            if not row:
                continue
            if len(row) != 4:
                raise ValueError(f"Unexpected worker timing columns at {path}:{line_number}")
            stage, start, end, cpu = row
            start, end, cpu = float(start), float(end), float(cpu)
            if not all(map(math.isfinite, (start, end, cpu))) or end < start or cpu < 0:
                raise ValueError(f"Invalid worker timing at {path}:{line_number}")
            wall_values, cpu_values = stages.setdefault(stage, (array("d"), array("d")))
            wall_values.append((end-start)*1000)
            cpu_values.append(cpu)
            if stage in ("archive_json", "_atomic_json") and measurement_started is not None:
                intervals.append((start-measurement_started, end-measurement_started))
    return {
        "scope": "All recorded worker tasks, including startup and final archive export.",
        "alignment_available": measurement_started is not None,
        "stages": {stage: {"wall": statistics(wall), "thread_cpu": statistics(cpu),
                           "total_wall_ms": sum(wall), "total_thread_cpu_ms": sum(cpu)}
                   for stage, (wall, cpu) in stages.items()},
    }, intervals


def overlap_summary(stalls, intervals, measured_seconds: float, source: str) -> dict:
    """Merge archive windows, then count intersecting stalls in one sorted sweep."""
    merged = []
    for start, end in sorted(intervals):
        start, end = max(0, start), min(measured_seconds, end)
        if end <= start:
            continue
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    index, overlapping = 0, 0
    for start, end in sorted(stalls):
        while index < len(merged) and merged[index][1] <= start:
            index += 1
        if index < len(merged) and merged[index][0] < end:
            overlapping += 1
    return {
        "source": source, "display_stalls_over_100ms": len(stalls),
        "stalls_overlapping_archive_json": overlapping,
        "fraction_of_stalls_overlapping": overlapping/len(stalls) if stalls else None,
        "archive_json_intervals": len(intervals), "merged_archive_windows": len(merged),
        "archive_json_active_seconds": sum(end-start for start, end in merged),
        "interpretation": "Temporal overlap is correlation, not evidence of causality.",
    }


def resource_summary(path: Path, parent_pid) -> dict | None:
    if not path.exists():
        return None
    first = last = peak = None
    gpu_samples, errors = [], []
    count = 0
    with path.open(encoding="utf-8-sig") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            processes = record.get("processes", [])
            total = sum(process.get("rss", 0) for process in processes)
            parent = next((process.get("rss", 0) for process in processes
                           if process.get("pid") == parent_pid), None)
            sample = {"elapsed_s": record.get("elapsed_s"), "parent_plus_children_rss_bytes": total,
                      "parent_rss_bytes": parent, "observed_process_count": len(processes)}
            first = first or sample
            last = sample
            if peak is None or total > peak["parent_plus_children_rss_bytes"]:
                peak = sample
            count += 1
            if record.get("gpu"):
                gpu_samples.append([record.get("elapsed_s"), record["gpu"]])
            for key in ("archive_error", "writer_error"):
                if record.get(key):
                    errors.append({"line": line_number, "elapsed_s": record.get("elapsed_s"),
                                   "kind": key, "error": record[key]})
    return {"samples": count, "first": first, "last": last, "peak": peak,
            "scope": "Sampled process-tree working sets; GPU memory is separate.",
            "errors": errors,
            "gpu_telemetry": {
                "source": "nvidia-smi; does not measure Intel GPU memory",
                "raw_columns": ["utilization_percent", "memory_used_MiB", "temperature_C", "power_W"],
                "samples_elapsed_s_and_raw_csv": gpu_samples,
            }}


def map_summary(path: Path, measurement_started: float | None) -> dict | None:
    if not path.exists():
        return None
    data = read_json(path)
    draws, updates = data.get("draws", []), data.get("updates", [])
    last_draw = None
    if draws:
        last = draws[-1]
        last_draw = {"child_elapsed_s": last[0], "duration_ms": last[1],
                     "active": last[2], "map_opacity": last[3], "textures": last[4]}
        if measurement_started is not None and data.get("clock_started") is not None:
            last_draw["measurement_elapsed_s"] = data["clock_started"]+last[0]-measurement_started
    return {"renderer": data.get("renderer"), "size": data.get("size"),
            "wall_seconds": data.get("wall_seconds"), "cpu_seconds": data.get("cpu_seconds"),
            "last_draw": last_draw, "max_textures": max((row[4] for row in draws), default=0),
            "draw": statistics([row[1] for row in draws]),
            "update": statistics([row[1] for row in updates]),
            "last_update": ({"child_elapsed_s": updates[-1][0], "duration_ms": updates[-1][1],
                             "images": updates[-1][2]} if updates else None),
            "static_map_probe": data.get("static_map_probe", False),
            "failures": data.get("failures", []),
            "scope": "Child lifetime; draw durations measure CPU submission, not GPU completion."}


def archive_summary(directory: Path) -> dict:
    archives = []
    for path in sorted((directory / "journeys").glob("*/manifest.json")):
        data = read_json(path)
        archives.append({"manifest": str(path.relative_to(directory)), "id": data.get("id"),
                         "images": len(data.get("images", [])),
                         "points": sum(len(segment.get("points", []))
                                       for segment in data.get("segments", [])),
                         "segments": len(data.get("segments", [])),
                         "prompts": len(data.get("prompts", [])),
                         "completion_reason": data.get("completion_reason"),
                         "finalized": bool(data.get("ended_at") and data.get("completion_reason"))})
    return {"manifests": archives,
            "totals": {key: sum(item[key] for item in archives)
                       for key in ("images", "points", "segments", "prompts")},
            "note": "Counts come from manifests; image file integrity is not checked here."}


def summarize(directory: Path, *, interrupted: bool = False) -> dict:
    summary_path = directory / "summary.json"
    recovered = False
    warnings = []
    if not summary_path.exists():
        if not interrupted:
            raise ValueError(f"Missing final summary: {summary_path}. The replay may still be running. "
                             "Use --interrupted only after it has stopped.")
        final = read_json(directory / "progress.json")
        final.update(completed=False, passed=False, measured_seconds=None,
                     error="Interrupted run: no final summary was written.")
        recovered = True
        warnings.extend([
            "Periodic telemetry and flushed CSV may omit events immediately before termination.",
            "Archive counts describe durable checkpoints, not guaranteed final session counts.",
            "The unfinished final AI publication gap is unknown.",
        ])
    else:
        final = read_json(summary_path)
    if "completed" not in final or "passed" not in final:
        raise ValueError(f"Final summary lacks completed/passed status: {summary_path}")
    run = read_json(directory / "run.json")
    measured_seconds = None if recovered else float(final["measured_seconds"])
    if measured_seconds is not None and (not math.isfinite(measured_seconds) or measured_seconds < 0):
        raise ValueError("Final measured_seconds must be nonnegative and finite")
    full, bins, stalls, archive_intervals, row_counts, csv_elapsed = read_timings(
        directory / "timings.csv", measured_seconds)
    if recovered:
        final["measured_seconds"] = measured_seconds = csv_elapsed
    if not (directory / "map-metrics.json").exists() and run.get("map_check") not in (None, "off", "soak-off"):
        warnings.append("Map child metrics are unavailable; draw lifetime, texture limits and prompt-card rendering are unverified.")
    worker, worker_intervals = archive_worker_timings(
        directory / "archive-timings.csv", run.get("measurement_started"))
    # Prefer independently timed worker JSON windows when available; do not double count.
    if worker_intervals:
        overlap = overlap_summary(stalls, worker_intervals, measured_seconds, "archive-timings.csv")
    elif archive_intervals:
        overlap = overlap_summary(stalls, archive_intervals, measured_seconds, "timings.csv")
    else:
        overlap = {"available": False, "display_stalls_over_100ms": len(stalls),
                   "reason": "No aligned archive_json intervals recorded."}
    return {
        "run_directory": str(directory.resolve()),
        "summary_source": "periodic telemetry (progress.json)" if recovered else "summary.json",
        "interrupted_recovery": recovered, "warnings": warnings,
        "elapsed_source": "maximum elapsed_s in flushed timings.csv" if recovered else "summary.json measured_seconds",
        "maximum_flushed_csv_elapsed_s": csv_elapsed,
        "last_periodic_counts": ({key: final.get(key) for key in ("elapsed_s", "images", "points", "prompts")}
                                 if recovered else None),
        "status": {key: final.get(key) for key in (
            "completed", "passed", "measured_seconds", "error", "writer_error", "dropped_timing_rows",
            "max_stall_threshold_ms", "max_ai_gap_threshold_ms")},
        "requested_seconds": run.get("requested_seconds"), "map_check": run.get("map_check"),
        "actual_main_window_size": run.get("window_size"),
        "ai_final_gap": {key: final.get("ai_gaps", {}).get(key)
                         for key in ("final_unpublished_gap_ms", "max_observed_gap_ms")},
        "full_run_intervals": full, "ten_minute_bins": bins,
        "interval_method": "Exact CSV quantiles. Bins use interval completion time; cadence is 1000/mean interval ms.",
        "csv_stage_counts": row_counts,
        "csv_count_mismatches": {
            stage: {"summary_count": values.get("count"), "csv_count": row_counts.get(stage, 0)}
            for stage, values in final.get("timings", {}).items()
            if values.get("count") != row_counts.get(stage, 0)},
        "count_reference": "Periodic telemetry may precede the flushed CSV tail." if recovered else "Final summary.",
        "summary_stage_timings": final.get("timings", {}),
        "display_archive_overlap": overlap, "archive_worker": worker,
        "resources": resource_summary(directory / "resources.jsonl", run.get("pid")),
        "map_child": map_summary(directory / "map-metrics.json", run.get("measurement_started")),
        "archives": {**archive_summary(directory), "interrupted_session_counts": recovered},
    }


def cli():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_directory", type=Path)
    parser.add_argument("--output", type=Path, help="Defaults to RUN_DIRECTORY/analysis.json")
    parser.add_argument("--interrupted", action="store_true",
                        help="Allow progress.json fallback after a run has stopped without its final summary")
    args = parser.parse_args()
    try:
        result = summarize(args.run_directory, interrupted=args.interrupted)
    except (OSError, ValueError, KeyError) as error:
        parser.error(str(error))
    destination = args.output or args.run_directory / "analysis.json"
    destination.write_text(json.dumps(result, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    print(destination.resolve())


if __name__ == "__main__":
    cli()
