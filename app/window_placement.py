"""Fullscreen placement without replacing a window or its OpenGL context."""
from __future__ import annotations

import pygame
from pygame._sdl2 import Window


class WindowPlacement:
    """Call on the SDL window's owning thread, including construction.

    SDL desktop fullscreen uses the display containing the window, so an
    operator can drag windows to projectors before entering fullscreen.
    """

    def __init__(self):
        self.window = Window.from_display_module()
        self.is_fullscreen = bool(pygame.display.is_fullscreen())
        self._windowed_bounds = None

    def set_fullscreen(self, enabled: bool) -> bool:
        enabled = bool(enabled)
        if enabled == self.is_fullscreen:
            return enabled
        if enabled:
            bounds = (self.window.position, self.window.size)
            self.window.set_fullscreen(desktop=True)
            self._windowed_bounds = bounds
        else:
            self.window.set_windowed()
            if self._windowed_bounds is not None:
                position, size = self._windowed_bounds
                self.window.size = size
                self.window.position = position
        self.is_fullscreen = enabled
        return enabled
