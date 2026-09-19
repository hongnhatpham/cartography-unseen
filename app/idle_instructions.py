"""Timing for the visitor title and controls, independent of automatic flight."""

from dataclasses import dataclass, field


@dataclass
class IdleInstructions:
    last_input: float
    _fade_start: float = field(init=False)
    _fade_from: float = 0.0
    _target: float = 0.0

    def __post_init__(self) -> None:
        self._fade_start = self.last_input

    def _opacity(self, now: float) -> float:
        duration = 1.2 if self._target else .25
        progress = min(1.0, max(0.0, (now - self._fade_start) / duration))
        return self._fade_from + (self._target - self._fade_from) * progress

    @property
    def showing(self) -> bool:
        """Whether the visitor title has entered its idle presentation state."""
        return bool(self._target)

    def update(self, now: float, *, active: bool, suppressed: bool) -> float:
        """Wait ten idle seconds, then fade in; physical input fades back out."""
        if active:
            self.last_input = now
        if suppressed:
            self._fade_start = now
            self._fade_from = self._target = 0.0
            return 0.0
        target = float(now - self.last_input >= 10.0)
        if target != self._target:
            self._fade_from = self._opacity(now)
            self._fade_start = max(self.last_input + 10.0, self._fade_start) if target else now
            self._target = target
        return self._opacity(now)
