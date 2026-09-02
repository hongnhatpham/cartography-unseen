from __future__ import annotations

from collections import deque
from time import perf_counter


class RateMeter:
    def __init__(self, window: int = 120) -> None:
        self._times: deque[float] = deque(maxlen=window)

    def tick(self, timestamp: float | None = None) -> None:
        self._times.append(timestamp if timestamp is not None else perf_counter())

    @property
    def fps(self) -> float:
        if len(self._times) < 2:
            return 0.0
        elapsed = self._times[-1] - self._times[0]
        return (len(self._times) - 1) / elapsed if elapsed > 0.0 else 0.0


class ExponentialAverage:
    def __init__(self, alpha: float = 0.15) -> None:
        self.alpha = alpha
        self.value = 0.0
        self.initialized = False

    def update(self, sample: float) -> float:
        if not self.initialized:
            self.value = sample
            self.initialized = True
        else:
            self.value += self.alpha * (sample - self.value)
        return self.value
