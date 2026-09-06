"""Windows SDL ownership, isolated from simulation and OpenGL presentation.

Windows can block inside PeekMessage even in an empty window. The video thread
therefore owns input and window operations; the caller owns the transferred GL
context. No lock is held while calling SDL or waiting for either thread.
"""
from __future__ import annotations

from collections import deque
from concurrent.futures import Future, TimeoutError
import ctypes
from pathlib import Path
from queue import Empty, Full, Queue
import threading
from time import perf_counter
from typing import Callable, Any

import pygame

from app.window_placement import WindowPlacement


class WindowLoop:
    active: WindowLoop | None = None

    def __init__(self, create_window: Callable[[], None]):
        self._commands: Queue = Queue(maxsize=16)
        self._stop = threading.Event()
        self._render_released = threading.Event()
        self._ready: Future = Future()
        self._lock = threading.Lock()
        self._events = deque()
        self._window_events = {}
        self._motion_event = None
        self._quit_event = None
        self._motion = (0, 0)
        self._keys = ()
        self._buttons = (False, False, False)
        self.size = (0, 0)
        self.error: BaseException | None = None
        self._attached = False
        self._thread = threading.Thread(target=self._run, args=(create_window,),
                                        name="window-input", daemon=True)
        self._thread.start()
        try:
            self._ready.result(timeout=30)
            self._attach(self.context)
            self._attached = True
        except BaseException:
            self.close()
            raise
        WindowLoop.active = self

    def _api(self, name, result, *arguments):
        function = getattr(self._sdl, name)
        function.restype, function.argtypes = result, arguments
        return function

    def _attach(self, context):
        if self._make_current(self.window, context) != 0:
            raise RuntimeError(self._get_error().decode("utf-8", errors="replace"))

    def _run(self, create_window):
        try:
            create_window()
            if self._stop.is_set():
                raise RuntimeError("Window initialization cancelled")
            # Pygame's automatic resize watcher steals the GL context to call
            # glViewport. ProxyRenderer already handles its own viewport.
            pygame.display._set_autoresize(False)
            self._sdl = ctypes.CDLL(str(Path(pygame.__file__).parent / "SDL2.dll"))
            pointer = ctypes.c_void_p
            self._make_current = self._api("SDL_GL_MakeCurrent", ctypes.c_int, pointer, pointer)
            self._get_error = self._api("SDL_GetError", ctypes.c_char_p)
            self._placement = WindowPlacement()
            self.window = self._api("SDL_GL_GetCurrentWindow", pointer)()
            self.context = self._api("SDL_GL_GetCurrentContext", pointer)()
            if not self.window or not self.context:
                raise RuntimeError("SDL did not create an OpenGL window/context")
            self.size = pygame.display.get_window_size()
            self._keys = pygame.key.get_pressed()
            self._attach(None)
            self._ready.set_result(None)
            while not self._stop.is_set():
                started = perf_counter()
                self._dispatch()
                if self._stop.is_set():
                    break
                first = pygame.event.wait(1)
                events = pygame.event.get(pump=False)
                if first.type != pygame.NOEVENT:
                    events.insert(0, first)
                # These are copies of SDL state, read only on its owning thread.
                keys = pygame.key.get_pressed()
                motion = pygame.mouse.get_rel()
                buttons = pygame.mouse.get_pressed()
                size = pygame.display.get_window_size()
                with self._lock:
                    self._record_events(events)
                    self._keys, self._buttons, self.size = keys, buttons, size
                    self._motion = (self._motion[0] + motion[0], self._motion[1] + motion[1])
                self._stop.wait(max(0, 1 / 60 - (perf_counter() - started)))
        except BaseException as exc:
            with self._lock:
                self.error = exc
            if not self._ready.done():
                self._ready.set_exception(exc)
        finally:
            with self._lock:
                self._stop.set()
            self._dispatch(cancel=True)
            # A native input failure must not destroy a window whose GL context
            # is still in use. Constructor failure/timeout also releases this.
            self._render_released.wait()
            # create_window may have initialized SDL before raising.
            pygame.quit()

    def _record_events(self, events):
        """Called under the snapshot lock; retain control edges through window floods."""
        for event in events:
            if event.type == pygame.MOUSEMOTION:
                self._motion_event = event
            elif event.type == pygame.QUIT:
                self._quit_event = event
            elif (pygame.WINDOWSHOWN <= event.type <= pygame.WINDOWDISPLAYCHANGED
                  or event.type in (pygame.VIDEOEXPOSE, pygame.VIDEORESIZE)):
                self._window_events[event.type] = event
            else:
                if len(self._events) >= 512:
                    raise RuntimeError("Window input overflow: 512 unconsumed control events")
                self._events.append(event)

    def _dispatch(self, cancel=False):
        while True:
            try:
                function, future = self._commands.get_nowait()
            except Empty:
                return
            if not future.set_running_or_notify_cancel():
                continue
            if cancel:
                future.set_exception(RuntimeError("Window input loop stopped"))
                continue
            try:
                future.set_result(function())
            except BaseException as exc:
                future.set_exception(exc)

    def call(self, function: Callable[[], Any]):
        if threading.current_thread() is self._thread:
            raise RuntimeError("Cannot wait for a window command on the window thread")
        future: Future = Future()
        with self._lock:
            if self.error:
                raise RuntimeError("Window input loop failed") from self.error
            if self._stop.is_set() or not self._thread.is_alive():
                raise RuntimeError("Window input loop stopped")
            try:
                self._commands.put_nowait((function, future))
            except Full:
                raise RuntimeError("Window command queue is full") from None
        try:
            return future.result(timeout=10)
        except TimeoutError:
            # Prevent a command that never started from running unexpectedly
            # after the caller has already handled its timeout.
            future.cancel()
            raise

    def poll_events(self):
        if self.error:
            raise RuntimeError("Window input loop failed") from self.error
        with self._lock:
            events = list(self._events)
            self._events.clear()
            events.extend(self._window_events.values())
            self._window_events.clear()
            if self._motion_event is not None:
                events.append(self._motion_event)
                self._motion_event = None
            if self._quit_event is not None:
                events.append(self._quit_event)
                self._quit_event = None
        return events

    def read_input(self):
        with self._lock:
            result = self._motion, self._keys, self._buttons
            self._motion = (0, 0)
        return result

    @property
    def is_fullscreen(self):
        return self._placement.is_fullscreen

    def set_fullscreen(self, enabled):
        def change():
            fullscreen = self._placement.set_fullscreen(enabled)
            self.size = pygame.display.get_window_size()
            return fullscreen
        return self.call(change)

    def toggle_fullscreen(self):
        return self.set_fullscreen(not self.is_fullscreen)

    def close(self):
        with self._lock:
            self._stop.set()
        if self._attached:
            # If detaching fails, report it without authorizing another thread
            # to destroy a context that may still be current here.
            self._attach(None)
            self._attached = False
        if WindowLoop.active is self:
            WindowLoop.active = None
        self._render_released.set()
        self._thread.join(timeout=10)
        if self._thread.is_alive():
            raise RuntimeError("Window input thread did not stop")
