from __future__ import annotations

import threading
from typing import Generic, TypeVar


T = TypeVar("T")


class LatestValue(Generic[T]):
    """A single atomic slot: publishing a new value discards the old one."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._event = threading.Event()
        self._version = 0
        self._value: T | None = None

    def publish(self, value: T) -> int:
        with self._lock:
            self._value = value
            self._version += 1
            version = self._version
            self._event.set()
            return version

    def get(self) -> tuple[int, T | None]:
        with self._lock:
            return self._version, self._value

    def wait_newer(
        self, last_version: int, stop_event: threading.Event, timeout: float = 0.1
    ) -> tuple[int, T | None]:
        while not stop_event.is_set():
            with self._lock:
                if self._version > last_version:
                    return self._version, self._value
                self._event.clear()
            self._event.wait(timeout)
        return last_version, None

    def wake(self) -> None:
        self._event.set()
