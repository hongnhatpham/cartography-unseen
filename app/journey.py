"""Record human journeys and save portable originals without encoding on the UI thread."""
from __future__ import annotations

import base64
from bisect import bisect_right
from collections import deque
from concurrent.futures import Future, ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from copy import deepcopy
from datetime import datetime, timezone
from html import escape
import hashlib
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import textwrap
from typing import Callable
from uuid import uuid4

import numpy as np
from PIL import Image

from app.types import GeneratedFrame
from app.journey_updates import _merge_snapshot, _snapshot_delta


# Each recorder owns one persistent process, which applies its live jobs in order.
_worker_snapshot: dict | None = None


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    with temporary.open("w", encoding="utf-8") as output:
        portable = {key: value for key, value in data.items() if key != "archive_dir"}
        # One-shot encoding avoids streaming Python chunks while gameplay is running.
        output.write(json.dumps(portable, ensure_ascii=False, separators=(",", ":"), allow_nan=False))
        output.flush()
        os.fsync(output.fileno())
    temporary.replace(path)


def _apply_update(delta: dict) -> dict:
    global _worker_snapshot
    _worker_snapshot = _merge_snapshot(_worker_snapshot, delta)
    return _worker_snapshot


def _save_checkpoint(directory: Path, delta: dict) -> None:
    _atomic_json(directory / "manifest.json", _apply_update(delta))


def _save_frame(directory: Path, descriptor: dict, pixels: np.ndarray, delta: dict) -> None:
    """Publish lossless image pixels before a manifest that references them."""
    data = _apply_update(delta)
    path = directory / descriptor["path"]
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as output:
        if path.suffix == ".webp":
            Image.fromarray(pixels).save(output, format="WEBP", lossless=True, exact=True, method=3)
        else:
            Image.fromarray(pixels).save(output, format="PNG")
        output.flush()
        os.fsync(output.fileno())
    temporary.replace(path)
    _atomic_json(directory / "manifest.json", data)


def _save_archive(directory: Path, data: dict, export_svg: bool = False) -> None:
    """Finish portable exports in the same worker after all image writes."""
    marker = directory / "complete.json"
    marker.unlink(missing_ok=True)
    _atomic_json(directory / "manifest.json", data)
    if export_svg:
        _export_svg(directory / "map.svg", data)
    viewer = Path(__file__).resolve().parents[1] / "assets" / "journey-viewer.html"
    manifest = json.dumps(data, ensure_ascii=False).replace("</", "<\\/")
    html = viewer.read_text(encoding="utf-8").replace(
        "/* JOURNEY_MANIFEST */", f"window.JOURNEY_MANIFEST = {manifest};")
    temporary = directory / "index.html.tmp"
    with temporary.open("w", encoding="utf-8") as output:
        output.write(html)
        output.flush()
        os.fsync(output.fileno())
    temporary.replace(directory / "index.html")
    _atomic_json(marker, {"schema_version": 1,
                          "manifest_sha256": hashlib.sha256((directory / "manifest.json").read_bytes()).hexdigest()})


class JourneyRecorder:
    """One mutable journey, one bounded disk task, and immutable archive checkpoints.

    All public methods belong to the main thread. Image encoding and file writes
    run in one worker process. Only reset/close wait for durable storage. A failed task
    retains its original image and can be retried by reset/close.
    """

    def __init__(self, root: Path, world_seed: int, capture_distance: float = 12.0, *,
                 image_format: str = "webp", export_svg: bool = False) -> None:
        if not math.isfinite(capture_distance) or capture_distance <= 0:
            raise ValueError("capture_distance must be positive and finite")
        if image_format not in {"webp", "png"}:
            raise ValueError("image_format must be webp or png")
        self.root = Path(root).resolve()
        self.world_seed = int(world_seed)
        self.capture_distance = float(capture_distance)
        self.image_format = image_format
        self.export_svg = export_svg
        self._executor = self._new_executor()
        self._worker_broken = False
        self._future: Future | None = None
        self._task: tuple[Callable, tuple, dict] | None = None
        self._error: str | None = None
        self._reported_error: str | None = None
        self._closed = False
        self._new_journey(None, None)

    @staticmethod
    def _new_executor() -> ProcessPoolExecutor:
        return ProcessPoolExecutor(max_workers=1, mp_context=mp.get_context("spawn"))

    def _new_journey(self, timestamp: float | None, preceding: str | None) -> None:
        self.id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "-" + uuid4().hex[:12]
        self.archive_dir = self.root / self.id
        self._data = {
            "schema_version": 1, "id": self.id, "started_at": _utc(),
            "started_monotonic": timestamp, "world_seed": self.world_seed,
            "coordinate_system": {"up": "Y", "forward": "-Z", "rotation": "pitch,yaw,roll degrees", "units": "world units"},
            "overview": {"direction": "A", "plane_width": 8.0, "capture_distance": self.capture_distance},
            "preceding_archive": preceding, "segments": [], "images": [], "prompts": [],
            "current_pose": None,
        }
        self._segment: dict | None = None
        self._last_position: np.ndarray | None = None
        self._last_timestamp = timestamp
        self._distance = 0.0
        # Source frames are normally milliseconds old. Bound pose lookup even
        # if generation stalls, rejecting frames older than this history.
        self._history: deque[tuple[float, str, float]] = deque(maxlen=8192)
        self._capture_distances: dict[str, float] = {}
        self._seen_sequences: set[int] = set()
        self._last_checkpoint = timestamp
        self._last_sent_snapshot: dict | None = None

    @property
    def nonempty(self) -> bool:
        return bool(self._data["segments"] or self._data["images"])

    def _start_clock(self, timestamp: float) -> None:
        if self._data["started_monotonic"] is None:
            self._data["started_monotonic"] = float(timestamp)
            self._last_checkpoint = float(timestamp)

    def update_pose(self, position, rotation, timestamp: float, active: bool) -> None:
        self._start_clock(timestamp)
        point = np.asarray(position, dtype=np.float64)
        self._data["current_pose"] = {"position": point.tolist(), "rotation": list(map(float, rotation)),
                                      "timestamp": float(timestamp), "active": bool(active)}
        if active:
            if self._segment is None:
                self._segment = {"id": f"segment-{len(self._data['segments']) + 1}",
                                 "started": float(timestamp), "ended": None, "points": []}
                self._data["segments"].append(self._segment)
                self._distance = 0.0
                self._last_position = None
            if self._last_position is not None:
                self._distance += float(np.linalg.norm(point - self._last_position))
            points = self._segment["points"]
            if not points or np.linalg.norm(point - np.asarray(points[-1]["position"])) >= 1.0:
                points.append({"position": point.tolist(), "timestamp": float(timestamp)})
            self._history.append((float(timestamp), self._segment["id"], self._distance))
            self._last_position = point.copy()
        elif self._segment is not None:
            # End at the last human pose, never at the first automatic pose.
            if self._last_position is not None:
                endpoint = {"position": self._last_position.tolist(), "timestamp": self._last_timestamp}
                if self._segment["points"][-1]["position"] != endpoint["position"]:
                    self._segment["points"].append(endpoint)
            self._segment["ended"] = float(timestamp)
            self._segment = None
            self._last_position = None
        self._last_timestamp = float(timestamp)
        self._check_pending()
        if self.nonempty and self._future is None and timestamp - self._last_checkpoint >= 5.0:
            data = self.snapshot()
            self._submit(_save_checkpoint, self.archive_dir, snapshot=data)
            self._last_checkpoint = float(timestamp)

    def record_prompt(self, prompt: str, revision: int, trigger: str, position, rotation,
                      timestamp: float, active: bool, settings: dict | None = None) -> None:
        self._start_clock(timestamp)
        prompts = self._data["prompts"]
        prompts.append({"id": f"prompt-{len(prompts) + 1}", "prompt": prompt,
                        "previous_prompt": prompts[-1]["prompt"] if prompts else None,
                        "revision": int(revision), "trigger": trigger, "timestamp": float(timestamp),
                        "elapsed": float(timestamp - self._data["started_monotonic"]),
                        "position": list(map(float, position)), "rotation": list(map(float, rotation)),
                        "active": bool(active), "segment_id": self._segment["id"] if active and self._segment else None,
                        "settings": deepcopy(settings or {}), "first_frame": None})

    def offer_frame(self, frame: GeneratedFrame) -> bool:
        self._check_pending()
        source_time = float(frame.conditioning_timestamp)
        start = self._data["started_monotonic"]
        if start is None or source_time < start:
            return False
        revision = frame.stats.get("prompt_revision")
        prompt = next((event for event in reversed(self._data["prompts"])
                       if event["revision"] == revision), None)
        if prompt is None:
            return False
        if prompt["first_frame"] is None:
            prompt["first_frame"] = {"timestamp": source_time, "generation_timestamp": float(frame.generation_timestamp),
                                     "position": frame.camera_position.tolist(), "rotation": frame.camera_rotation.tolist(),
                                     "sequence": int(frame.sequence)}
        if self._future is not None or frame.sequence in self._seen_sequences or not self._history:
            return False
        history = list(self._history)
        index = bisect_right([entry[0] for entry in history], source_time) - 1
        if index < 0:
            return False
        timestamp, segment_id, distance = history[index]
        segment = next(segment for segment in reversed(self._data["segments"]) if segment["id"] == segment_id)
        if source_time < segment["started"] or (segment["ended"] is not None and source_time >= segment["ended"]):
            return False
        # Interpolate distance at the source time, without bridging idle gaps.
        if index + 1 < len(history) and history[index + 1][1] == segment_id:
            next_time, _, next_distance = history[index + 1]
            if next_time > timestamp:
                distance += (next_distance - distance) * (source_time - timestamp) / (next_time - timestamp)
        elif source_time > timestamp + 0.25:
            return False
        last = self._capture_distances.get(segment_id)
        if last is not None and distance - last < self.capture_distance:
            return False
        number = len(self._data["images"]) + 1
        height, width = frame.image.shape[:2]
        descriptor = {"id": f"image-{number}", "path": f"images/{number:06d}.{self.image_format}",
                      "segment_id": segment_id, "position": frame.camera_position.tolist(),
                      "rotation": frame.camera_rotation.tolist(), "timestamp": source_time,
                      "generation_timestamp": float(frame.generation_timestamp), "sequence": int(frame.sequence),
                      "prompt_revision": int(revision), "generation_stats": dict(frame.stats),
                      "width": width, "height": height, "plane_width": 8.0}
        self._data["images"].append(descriptor)
        self._capture_distances[segment_id] = distance
        self._seen_sequences.add(frame.sequence)
        pixels = np.array(frame.image, copy=True)
        data = self.snapshot()
        self._submit(_save_frame, self.archive_dir, descriptor, pixels, snapshot=data)
        return True

    def _submit(self, task: Callable, *args, snapshot: dict) -> None:
        # Retain the full immutable view for recovery, but only transfer new records.
        delta = _snapshot_delta(self._last_sent_snapshot, snapshot)
        self._task = task, args, snapshot
        self._last_sent_snapshot = snapshot
        try:
            self._future = self._executor.submit(task, *args, delta)
        except BrokenProcessPool as error:
            # Keep the accepted image for an explicit retry even if submission fails.
            self._future = Future()
            self._future.set_exception(error)

    def _check_pending(self) -> None:
        if self._future is not None and self._future.done():
            try:
                self._future.result()
            except Exception as exc:
                self._error = f"MAP SAVE FAILED: {exc}"
            else:
                self._future = None
                self._task = None
                self._error = None
                self._reported_error = None

    def poll(self) -> str | None:
        self._check_pending()
        if self._error and self._error != self._reported_error:
            self._reported_error = self._error
            return self._error
        return None

    def snapshot(self) -> dict:
        """Return a read-only view whose contents stay stable as recording proceeds.

        Point records, image records, poses and settings are never edited after
        insertion. Copy growing lists and mutable segment/prompt dictionaries,
        while sharing their immutable records. This avoids walking every XYZ
        value in a long exhibition route on the main thread.
        """
        data = self._data.copy()
        data["segments"] = [dict(segment, points=segment["points"].copy())
                            for segment in self._data["segments"]]
        data["images"] = self._data["images"].copy()
        # first_frame changes from None to a newly created immutable record.
        data["prompts"] = [event.copy() for event in self._data["prompts"]]
        data["archive_dir"] = str(self.archive_dir)
        # Include the unsampled tail so saved paths reach the actual last pose.
        if self._segment is not None and self._last_position is not None:
            points = data["segments"][-1]["points"]
            if points[-1]["position"] != self._last_position.tolist():
                points.append({"position": self._last_position.tolist(), "timestamp": self._last_timestamp})
        return data

    def flush(self) -> None:
        """Finish writes, retrying a previously failed task once per explicit call."""
        failed = self._future.exception() if self._future is not None and self._future.done() else None
        if self._worker_broken or isinstance(failed, BrokenProcessPool):
            self._executor.shutdown(wait=True)
            self._executor = self._new_executor()
            self._last_sent_snapshot = None
            self._worker_broken = False
        if self._future is not None:
            if failed is not None:
                task, args, snapshot = self._task
                # A failed job may already have merged its delta. Replace that state
                # on every retry, including retries in a newly spawned worker.
                try:
                    self._future = self._executor.submit(task, *args, _snapshot_delta(None, snapshot))
                except BrokenProcessPool:
                    self._worker_broken = True
                    raise
                self._last_sent_snapshot = snapshot
            self._future.result()
            self._future = None
            self._task = None
            self._error = self._reported_error = None

    def _archive(self, reason: str, timestamp: float | None = None) -> Path | None:
        self.flush()
        if not self.nonempty:
            return None
        data = self.snapshot()
        data.update(ended_at=_utc(), ended_monotonic=timestamp or self._last_timestamp, completion_reason=reason)
        data.pop("archive_dir", None)
        directory = self.archive_dir

        # Keep the old journey intact until every requested artifact is saved.
        try:
            self._executor.submit(_save_archive, directory, data, self.export_svg).result()
        except BrokenProcessPool:
            self._worker_broken = True
            raise
        return directory

    def reset(self, timestamp: float, *, reason: str = "space") -> Path | None:
        previous = self._archive(reason, timestamp)
        self._new_journey(float(timestamp), previous.name if previous else None)
        return previous

    def close(self, reason: str = "quit") -> Path | None:
        if self._closed:
            return None
        path = self._archive(reason)
        self._executor.shutdown(wait=True)
        self._closed = True
        return path


def _project(point) -> np.ndarray:
    x, y, z = point
    return np.array([0.8 * x - 0.6 * z, 0.36 * x - 0.8 * y + 0.48 * z])


def _plane_corners(image: dict) -> list[np.ndarray]:
    pitch, yaw, roll = np.radians(image["rotation"])
    right = np.array([math.cos(yaw), 0., math.sin(yaw)])
    forward = np.array([math.sin(yaw) * math.cos(pitch), math.sin(pitch), -math.cos(yaw) * math.cos(pitch)])
    up = np.cross(right, forward)
    rotated_right = right * math.cos(roll) + up * math.sin(roll)
    rotated_up = up * math.cos(roll) - right * math.sin(roll)
    half_width = image["plane_width"] / 2
    half_height = half_width * image["height"] / image["width"]
    center = np.asarray(image["position"])
    return [_project(center + sx * half_width * rotated_right + sy * half_height * rotated_up)
            for sx, sy in ((-1, 1), (1, 1), (-1, -1), (1, -1))]


def _export_svg(path: Path, data: dict, image_directory: Path | None = None) -> None:
    """Export vector routes and lossless images, including legacy PNG archives."""
    image_directory = image_directory or path.parent
    planes = [(sample, _plane_corners(sample)) for sample in data["images"]]
    geometry = [_project(point["position"]) for segment in data["segments"] for point in segment["points"]]
    geometry.extend(corner for _, corners in planes for corner in corners)
    geometry.extend(_project(event["position"]) for event in data["prompts"] if event["active"])
    coordinates = np.asarray(geometry or [[0., 0.]])
    minimum, maximum = coordinates.min(axis=0), coordinates.max(axis=0)
    extent = np.maximum(maximum - minimum, 1.)
    scale = min(2240 / extent[0], 1400 / extent[1])
    offset = np.array([1200., 790.]) - (minimum + maximum) * 0.5 * scale
    project = lambda position: _project(position) * scale + offset
    lines = []
    for event in data["prompts"]:
        text = f"{event['id']} | +{event['elapsed']:.1f}s | {event['trigger']} | {'human' if event['active'] else 'idle'} | {event['prompt']}"
        lines.extend(textwrap.wrap(text, width=175, replace_whitespace=False) or [""])
    height = 1600 + max(80, len(lines) * 22 + 50)
    temporary = path.with_suffix(".svg.tmp")
    with temporary.open("w", encoding="utf-8") as output:
        output.write(f'<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" width="2400" height="{height}" viewBox="0 0 2400 {height}">\n')
        output.write(f'<rect width="2400" height="{height}" fill="#050607"/><g fill="#e4e7e8" font-family="sans-serif"><text x="70" y="55" font-size="24">Mapping the Blackbox</text>')
        output.write(f'<text x="70" y="82" font-size="13">{escape(data["id"])} | {escape(data["completion_reason"])}</text></g>\n')
        for segment in data["segments"]:
            points = " ".join(f"{x:.4f},{y:.4f}" for x, y in (project(point["position"]) for point in segment["points"]))
            output.write(f'<polyline points="{points}" fill="none" stroke="#858f93" stroke-width="1.2"/>\n')
        # Painter order is consistent with the fixed orthographic export view.
        planes.sort(key=lambda item: float(np.dot(item[0]["position"], [0.48, 0.6, 0.64])))
        for sample, corners in planes:
            top_left, top_right, bottom_left, _ = [point * scale + offset for point in corners]
            across, down = (top_right - top_left) / sample["width"], (bottom_left - top_left) / sample["height"]
            matrix = f"{across[0]:.8f} {across[1]:.8f} {down[0]:.8f} {down[1]:.8f} {top_left[0]:.5f} {top_left[1]:.5f}"
            image_path = image_directory / sample["path"]
            mime = "image/webp" if image_path.suffix.lower() == ".webp" else "image/png"
            output.write(f'<image width="{sample["width"]}" height="{sample["height"]}" transform="matrix({matrix})" xlink:href="data:{mime};base64,')
            output.write(base64.b64encode(image_path.read_bytes()).decode("ascii"))
            output.write(f'"><title>{escape(sample["id"])} | prompt revision {sample["prompt_revision"]}</title></image>\n')
        for event in data["prompts"]:
            if event["active"]:
                x, y = project(event["position"])
                output.write(f'<circle cx="{x:.4f}" cy="{y:.4f}" r="4" fill="#c6e3dd"><title>{escape(event["prompt"])}</title></circle>')
                output.write(f'<text x="{x + 8:.4f}" y="{y - 8:.4f}" fill="#c6e3dd" font-family="sans-serif" font-size="12">{event["id"]}</text>\n')
        output.write('<g font-family="sans-serif" font-size="16" fill="#c4cbce">')
        for index, line in enumerate(lines):
            output.write(f'<text x="70" y="{1635 + index * 22}">{escape(line)}</text>\n')
        output.write('</g></svg>')
        output.flush()
        os.fsync(output.fileno())
    temporary.replace(path)
