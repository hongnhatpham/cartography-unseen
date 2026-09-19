"""Connect exhibition input to the archive and the independent map window."""
from __future__ import annotations

from copy import deepcopy
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
        monitor = getattr(config, 'map_display_monitor', None)
        self.window = (MapWindow(root) if monitor is None else
                       MapWindow(root, display_monitor=monitor, fullscreen=config.fullscreen))
        self.idle_seconds = config.map_idle_seconds
        self.last_activity: float | None = None
        self.active = False
        self._last_snapshot = -float("inf")
        self._last_view_update = -float("inf")
        self._snapshot: dict = {}
        self._frame_sequence = -1
        self._last_position: list[float] | None = None
        self._last_rotation: list[float] | None = None
        self._current_prompt: tuple[str, int, dict] | None = None
        self._disk = DiskSpaceMonitor(root, int(getattr(config, "map_min_free_gib", 1.0) * 1024**3))
        self._disk_available: bool | None = None
        self._title_idle = False
        self._idle_archive_complete = False
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
        self._last_position = list(map(float, position))
        self._last_rotation = list(map(float, rotation))
        if interacting and not suppressed:
            self.last_activity = timestamp
            self._title_idle = False
            self._idle_archive_complete = False
        disk_available = self._disk.available
        if disk_available != self._disk_available:
            if not disk_available:
                self._notices.append("MAP RECORDING PAUSED: low disk space. Save with Space or free disk space.")
            elif self._disk_available is False:
                self._notices.append("MAP RECORDING RESUMED: disk space is available.")
            self._disk_available = disk_available
        active = (
            disk_available and not self._title_idle and not autowalking and not suppressed
            and self.last_activity is not None
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
        self._current_prompt = (prompt, int(revision), deepcopy(settings))
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

    def set_title_idle(self, idle: bool, timestamp: float, *, resumed: bool = False) -> Path | None:
        """Complete one journey when the visitor title enters its idle state."""
        if not idle:
            # Notices and operator overlays also suppress the title. Keep the
            # capture gate closed unless physical visitor input actually returns.
            if resumed:
                self._title_idle = False
                self._idle_archive_complete = False
            return None
        self._title_idle = True
        if self._idle_archive_complete:
            return None
        if not self.recorder.nonempty:
            self._idle_archive_complete = True
            return None
        saved = self.recorder.reset(timestamp, reason="idle")
        self._idle_archive_complete = True
        if self._current_prompt is not None and self._last_position is not None:
            prompt, revision, settings = self._current_prompt
            self.recorder.record_prompt(
                prompt, revision, "idle", self._last_position,
                self._last_rotation or [0.0, 0.0, 0.0], timestamp, False, settings,
            )
        self._snapshot = self.recorder.snapshot()
        self._last_snapshot = -float("inf")
        if saved is not None:
            logging.info("Journey completed after visitor session became idle: %s", saved.name)
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
