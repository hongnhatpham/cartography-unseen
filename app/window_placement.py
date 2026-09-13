"""Fullscreen placement without replacing a window or its OpenGL context."""
from __future__ import annotations

import pygame
import ctypes
import os
from pathlib import Path
from pygame._sdl2 import Window


class WindowPlacement:
    """Call on the SDL window's owning thread, including construction.

    Fullscreen fills the display containing the window, so an operator can
    drag windows to projectors before entering fullscreen.
    """

    def __init__(self):
        # Pygame 2.6 stores a borrowed pointer to this wrapper in SDL event data.
        # Keep it alive until pygame.quit(), and reuse it for focus/resize too.
        # A temporary from_display_module() replaces that pointer and leaves
        # events accessing freed Python memory after the temporary is released.
        self.window = Window.from_display_module()
        self.is_fullscreen = bool(pygame.display.is_fullscreen())
        self._windowed_bounds = None
        self._windowed_borderless = False

    def snapshot(self):
        """Read actual SDL window/display state on the window's owning thread."""
        class Rect(ctypes.Structure):
            _fields_ = [('x', ctypes.c_int), ('y', ctypes.c_int),
                        ('w', ctypes.c_int), ('h', ctypes.c_int)]
        sdl = ctypes.CDLL(str(Path(pygame.__file__).parent / 'SDL2.dll'))
        get_window = sdl.SDL_GetWindowFromID
        get_window.argtypes, get_window.restype = [ctypes.c_uint32], ctypes.c_void_p
        pointer = get_window(self.window.id)
        get_flags = sdl.SDL_GetWindowFlags
        get_flags.argtypes, get_flags.restype = [ctypes.c_void_p], ctypes.c_uint32
        get_display = sdl.SDL_GetWindowDisplayIndex
        get_display.argtypes, get_display.restype = [ctypes.c_void_p], ctypes.c_int
        display = get_display(pointer)
        bounds = Rect()
        sdl.SDL_GetDisplayBounds.argtypes = [ctypes.c_int, ctypes.POINTER(Rect)]
        if display < 0 or sdl.SDL_GetDisplayBounds(display, ctypes.byref(bounds)):
            raise RuntimeError('Window display is unavailable')
        flags = get_flags(pointer)
        is_visible = ctypes.windll.user32.IsWindowVisible
        is_visible.argtypes, is_visible.restype = [ctypes.c_void_p], ctypes.c_int
        native_visible = bool(is_visible(pygame.display.get_wm_info()['window']))
        position, size = list(self.window.position), list(self.window.size)
        borderless_full = bool(flags & 16) and position == [bounds.x, bounds.y] and size == [bounds.w, bounds.h]
        return dict(display=display, position=position, size=size,
                    bounds=[bounds.x, bounds.y, bounds.w, bounds.h],
                    fullscreen=bool(flags & 1) or borderless_full,
                    presentation='borderless' if borderless_full else 'sdl_fullscreen' if flags & 1 else 'windowed',
                    minimized=bool(flags & 64), shown=bool(flags & 4) and native_visible,
                    native_visible=native_visible,
                    input_focus=bool(flags & 512))

    def set_fullscreen(self, enabled: bool) -> bool:
        enabled = bool(enabled)
        if enabled == self.is_fullscreen:
            return enabled
        if enabled:
            bounds = (self.window.position, self.window.size)
            self._windowed_borderless = self.window.borderless
            if os.name == 'nt':
                # Keep the ordinary composited window/GL drawable. Cover the
                # display without requesting an SDL fullscreen transition.
                desktop = self.snapshot()['bounds']
                self.window.restore()
                self.window.borderless = True
                self.window.size = tuple(desktop[2:])
                self.window.position = tuple(desktop[:2])
            else:
                self.window.set_fullscreen(desktop=True)
            self._windowed_bounds = bounds
        else:
            if pygame.display.is_fullscreen():
                self.window.set_windowed()
            self.window.borderless = self._windowed_borderless
            if self._windowed_bounds is not None:
                position, size = self._windowed_bounds
                self.window.size = size
                self.window.position = position
        self.is_fullscreen = enabled
        return enabled
