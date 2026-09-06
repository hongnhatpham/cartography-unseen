"""Connect exhibition input to the archive and the independent map window."""
from __future__ import annotations

from pathlib import Path
from time import perf_counter
from typing import Any


class JourneySession:
    def __init__(self, root: Path, config: Any):
        from app.journey import JourneyRecorder
        from app.map_view import MapWindow

        self.recorder = JourneyRecorder(
            root / "journeys", config.world_seed,
            capture_distance=config.map_capture_distance,
        )
        self.window = MapWindow(root)
        self.idle_seconds = config.map_idle_seconds
        self.last_activity: float | None = None
        self.active = False
        self._last_snapshot = -float("inf")
        self._last_view_update = -float("inf")
        self._snapshot: dict = {}
        self._frame_sequence = -1

    def observe(self, position, rotation, timestamp: float, *, interacting: bool,
                autowalking: bool, suppressed: bool = False) -> None:
        """Keep human distance separate from idle flight and operator mouse motion."""
        if interacting and not suppressed:
            self.last_activity = timestamp
        active = (
            not autowalking and not suppressed and self.last_activity is not None
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
        if frame is not None and frame.sequence != self._frame_sequence:
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
        notice = self.recorder.poll()
        if notice:
            notices.append(notice)
        return notices

    def close(self) -> None:
        try:
            self.recorder.close()
        finally:
            self.window.close()
