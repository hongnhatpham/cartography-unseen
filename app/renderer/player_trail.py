"""A short route history, independent of the AI frame rate and world origin."""
from collections import deque

import numpy as np


TRAIL_SECONDS = 10.0
TRAIL_SAMPLE_SECONDS = 1.0 / 30.0
MAX_TRAIL_SAMPLES = 320
TRAIL_HEAD_GAP = 4.0


class PlayerTrail:
    """Record movement only; standing still lets the whole route disappear."""

    def __init__(self) -> None:
        self._samples: deque[tuple[float, np.ndarray]] = deque(maxlen=MAX_TRAIL_SAMPLES)

    def update(self, position: np.ndarray, now: float, enabled: bool = True) -> None:
        if not enabled:
            self._samples.clear()
            return
        if self._samples and now < self._samples[-1][0]:
            self._samples.clear()
        while self._samples and self._samples[0][0] <= now - TRAIL_SECONDS:
            self._samples.popleft()
        position = np.asarray(position, dtype=np.float64)
        if self._samples:
            at, previous = self._samples[-1]
            distance = float(np.linalg.norm(position - previous))
            if distance > 64.0:
                # A relocation must not draw a line across an untravelled gap.
                self._samples.clear()
            elif distance < .05 or now - at < TRAIL_SAMPLE_SECONDS:
                return
        self._samples.append((now, position.copy()))

    def vertices(self, origin: np.ndarray, now: float, until: float | None = None) -> np.ndarray:
        """Camera-relative segment pairs, XYZ plus opacity, for a small GL draw."""
        until = now if until is None else min(now, until)
        samples = [(at, point) for at, point in self._samples
                   if now - TRAIL_SECONDS < at <= until]
        # Withhold the newest few units along the actual route, in any flight
        # direction. Never move old marks when the player turns their head.
        gap = TRAIL_HEAD_GAP
        while len(samples) >= 2 and gap > 0.0:
            at, point = samples.pop()
            previous_at, previous = samples[-1]
            length = float(np.linalg.norm(point - previous))
            if length > gap:
                share = (length - gap) / length
                samples.append((previous_at + share * (at - previous_at),
                                previous + share * (point - previous)))
                break
            gap -= length
        if len(samples) < 2:
            return np.empty((0, 4), dtype="f4")
        at, points = zip(*samples)
        packed = np.empty((len(samples), 4), dtype="f4")
        # Subtract in double precision before upload, including at remote Y.
        packed[:, :3] = np.asarray(points) - np.asarray(origin, dtype=np.float64)
        remaining = np.clip(1.0 - (now - np.asarray(at)) / TRAIL_SECONDS, 0., 1.)
        packed[:, 3] = .85 * remaining ** 1.4
        return np.stack((packed[:-1], packed[1:]), axis=1).reshape(-1, 4)
