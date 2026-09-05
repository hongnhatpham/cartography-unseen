"""Real GL rendering must keep progressing while the SDL owner is blocked."""
from pathlib import Path
import sys
import threading

import numpy as np
import pygame
import pytest

from app.renderer.proxy_renderer import ProxyRenderer


@pytest.mark.skipif(sys.platform != "win32", reason="Windows SDL context transfer")
def test_rendering_continues_during_native_input_pause_and_mode_changes(monkeypatch):
    renderer = ProxyRenderer(Path(__file__).resolve().parents[1], (384, 256), False)
    entered, release, finished = threading.Event(), threading.Event(), threading.Event()
    original_wait = pygame.event.wait
    owner_id = renderer._window_loop._thread.ident

    def paused_wait(timeout):
        assert threading.get_ident() == owner_id
        if not entered.is_set():
            entered.set()
            release.wait(3)
            finished.set()
        return original_wait(timeout)

    try:
        monkeypatch.setattr(pygame.event, "wait", paused_wait)
        assert entered.wait(2)
        # Each image uploads and presents through the real GL context while the
        # window owner remains inside the injected input wait.
        for level in (32, 96, 160):
            renderer.poll_events()
            renderer.display(np.full((256, 384, 3), level, dtype=np.uint8), [])
        assert not finished.is_set(), "Presentation waited for native input"
        release.set()
        assert finished.wait(2)
        monkeypatch.setattr(pygame.event, "wait", original_wait)

        # SDL mode changes must not steal the current context from rendering.
        assert renderer.toggle_fullscreen()
        renderer.display(np.full((256, 384, 3), 192, dtype=np.uint8), [])
        assert not renderer.toggle_fullscreen()
        renderer.set_prompt_editing(True)
        renderer.set_prompt_editing(False)
        renderer.display(np.full((256, 384, 3), 224, dtype=np.uint8), [])
        assert renderer.ctx.error == "GL_NO_ERROR"
    finally:
        release.set()
        renderer.close()
    assert not renderer._window_loop._thread.is_alive()
