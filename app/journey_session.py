"""Connect exhibition input to the archive and the independent map window."""
from __future__ import annotations

from pathlib import Path
from time import perf_counter
from typing import Any
import logging

from app.journey_storage import DiskSpaceMonitor, start_sync


class JourneySession:
    def __init__(self, root: Path, config: Any):
        from app.journey import JourneyRecorder
        from app.map_view import MapWindow

        self.recorder = JourneyRecorder(
            root / "journeys", config.world_seed,
            capture_distance=config.map_capture_distance,
            image_format=getattr(config, "map_image_format", "webp"),
            export_svg=getattr(config, "map_export_svg", False),
        )
        self.window = MapWindow(root)
        self.idle_seconds = config.map_idle_seconds
        self.last_activity: float | None = None
        self.active = False
        self._last_snapshot = -float("inf")
        self._last_view_update = -float("inf")
        self._snapshot: dict = {}
        self._frame_sequence = -1
        self._disk = DiskSpaceMonitor(root, int(getattr(config, "map_min_free_gib", 1.0) * 1024**3))
        self._disk_available: bool | None = None
        self._notices: list[str] = []
        if getattr(config, "map_sync_enabled", False):
            try:
                start_sync(root, self.recorder.root, getattr(config, "map_cache_gib", 5.0))
            except (RuntimeError, OSError, ValueError, KeyError) as error:
                logging.warning("Journey upload did not start: %s", error)
                self._notices.append(f"MAP UPLOAD NOT STARTED: {error}")

    def observe(self, position, rotation, timestamp: float, *, interacting: bool,
                autowalking: bool, suppressed: bool = False) -> None:
        """Keep human distance separate from idle flight and operator mouse motion."""
        if interacting and not suppressed:
            self.last_activity = timestamp
        disk_available = self._disk.available
        if disk_available != self._disk_available:
            if not disk_available:
                self._notices.append("MAP RECORDING PAUSED: low disk space. Save with Space or free disk space.")
            elif self._disk_available is False:
                self._notices.append("MAP RECORDING RESUMED: disk space is available.")
            self._disk_available = disk_available
        active = (
            disk_available and not autowalking and not suppressed and self.last_activity is not None
            and timestamp - self.last_activity <= self.idle_seconds
        )
        changed = active != self.active
        self.active = active
        self.recorder.update_pose(position, rotation, timestamp, active)
        if active and (timestamp - self._last_snapshot >= .5 or changed):
            self._snapshot = self.recorder.snapshot()
            self._last_snapshot = timestamp
        if timestamp - self._last_view_update >= .05 or changed:
            self.window.update(self._snapshot, list(position), list(rotation), active)
            self._last_view_update = timestamp

    def offer_frame(self, frame) -> None:
        if self._disk.available and frame is not None and frame.sequence != self._frame_sequence:
            self.recorder.offer_frame(frame)
            self._frame_sequence = frame.sequence

    def prompt(self, prompt: str, revision: int, trigger: str, camera,
               settings: dict, *, timestamp: float | None = None) -> None:
        self.recorder.record_prompt(
            prompt, revision, trigger, camera.position,
            [camera.pitch, camera.yaw, 0.0],
            perf_counter() if timestamp is None else timestamp,
            self.active, settings,
        )
        self._last_snapshot = -float("inf")

    def reset(self, timestamp: float) -> Path | None:
        saved = self.recorder.reset(timestamp)
        self._last_snapshot = -float("inf")
        return saved

    def poll(self) -> list[str]:
        notices = self.window.poll()
        notices.extend(self._notices)
        self._notices.clear()
        notice = self.recorder.poll()
        if notice:
            notices.append(notice)
        return notices

    def close(self) -> None:
        try:
            self.recorder.close()
            self._disk.close()
        finally:
            self.window.close()
