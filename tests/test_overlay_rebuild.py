"""Overlay caching and partial uploads must preserve the original RGBA pixels."""
from collections import defaultdict
from types import SimpleNamespace

import numpy as np
import pygame
import pytest

from app.renderer import proxy_renderer


class Texture:
    def __init__(self, size):
        self.pixels = np.zeros((size[1], size[0], 4), dtype=np.uint8)
        self.writes = []

    def write(self, data, viewport=None):
        x, y, width, height = viewport or (0, 0, self.pixels.shape[1], self.pixels.shape[0])
        self.pixels[y:y + height, x:x + width] = np.frombuffer(data, np.uint8).reshape(height, width, 4)
        self.writes.append(len(data))

    def use(self, *args): pass
    def release(self): pass


class Font:
    def __init__(self):
        pygame.font.init()
        self.font = pygame.font.Font(None, 20)
        self.calls = []

    def render(self, *args):
        self.calls.append(args)
        return self.font.render(*args)


def make_renderer(size):
    renderer = proxy_renderer.ProxyRenderer.__new__(proxy_renderer.ProxyRenderer)
    renderer.window_size = renderer._overlay_size = size
    renderer._last_overlay_update = 0
    renderer._last_overlay_key = ""
    renderer._overlay_surface = pygame.Surface(size, pygame.SRCALPHA)
    renderer._overlay_regions = [None, None]
    renderer._overlay_text_cache = {}
    renderer._overlay_texture_initialized = False
    renderer.small_font = Font()
    renderer.overlay_texture = Texture(size)
    renderer.ctx = SimpleNamespace(enable=lambda *args: None, disable=lambda *args: None,
                                   texture=lambda size, *args, **kwargs: Texture(size))
    renderer.screen_program = defaultdict(lambda: SimpleNamespace(value=None))
    renderer.quad_vao = SimpleNamespace(render=lambda: None)
    return renderer


def original_pixels(renderer, lines, caption):
    """Original full-window rasterization, without caching or partial uploads."""
    size = renderer.window_size
    surface = pygame.Surface(size, pygame.SRCALPHA)
    font = getattr(renderer.small_font, "font", renderer.small_font)
    if lines:
        panel_height = min(size[1] - 12, 18 + len(lines) * 23)
        panel_width = min(size[0] - 24, max(490, min(840, 42 + max(map(len, lines)) * 9)))
        pygame.draw.rect(surface, (4, 8, 12, 51), (12, 12, panel_width, panel_height), border_radius=7)
        for index, line in enumerate(lines):
            surface.blit(font.render(line, True, (230, 240, 245)), (24, 21 + index * 23))
    if caption:
        rows = renderer._caption_lines(caption)
        height = 18 + len(rows) * 23
        top = size[1] - height - 12
        pygame.draw.rect(surface, (0, 0, 0, 77), (12, top, size[0] - 24, height), border_radius=7)
        for index, line in enumerate(rows):
            text = font.render(line, True, (240, 245, 248))
            text.set_alpha(77)
            surface.blit(text, (24, top + 9 + index * 23))
    return pygame.image.tostring(surface, "RGBA", True)


@pytest.mark.parametrize("size", [(1920, 1200), (800, 600), (320, 220)])
def test_exact_pixels_through_caption_toggle_shrink_long_text_and_resize(monkeypatch, size):
    renderer = make_renderer(size)
    now = [100.]
    monkeypatch.setattr(proxy_renderer, "perf_counter", lambda: now[0])
    long_lines = ["DIAGNOSTICS", "long text beyond the panel " * 12] + [f"setting {i}" for i in range(26)]
    scenes = [(long_lines, None), (long_lines, "organic distant shapes " * 30),
              (["DIAGNOSTICS", "short"], "other prompt " * 25),
              ([], "caption only"), (["DIAGNOSTICS"], None), ([], None), (long_lines, None)]
    for lines, caption in scenes:
        now[0] += .11
        renderer._draw_overlay(lines, caption)
        assert renderer.overlay_texture.pixels.tobytes() == original_pixels(renderer, lines, caption)
        assert len(renderer._overlay_text_cache) <= len(lines) + (3 if caption else 0)
    # A resize must rebuild even when all text is unchanged.
    renderer.window_size = (640, 360) if size != (640, 360) else (800, 600)
    now[0] += .11
    renderer._draw_overlay(long_lines)
    assert renderer.overlay_texture.pixels.tobytes() == original_pixels(renderer, long_lines, None)


def test_rebuild_reuses_surface_and_unchanged_text_without_changing_ten_hz(monkeypatch):
    renderer = make_renderer((1920, 1200))
    now = [100.]
    monkeypatch.setattr(proxy_renderer, "perf_counter", lambda: now[0])
    renderer._draw_overlay(["STABLE TITLE", "FPS 60"])
    surface = renderer._overlay_surface
    title = renderer._overlay_text_cache["STABLE TITLE", False]
    assert len(renderer.small_font.calls) == 2
    first_writes = len(renderer.overlay_texture.writes)
    now[0] += .05
    renderer._draw_overlay(["STABLE TITLE", "FPS 59"])
    assert len(renderer.overlay_texture.writes) == first_writes
    assert len(renderer.small_font.calls) == 2
    now[0] += .051
    renderer._draw_overlay(["STABLE TITLE", "FPS 59"])
    assert renderer._overlay_surface is surface
    assert renderer._overlay_text_cache["STABLE TITLE", False] is title
    assert len(renderer.small_font.calls) == 3
    assert len(renderer._overlay_text_cache) == 2
    assert renderer.overlay_texture.writes[-1] < renderer.overlay_texture.writes[0] / 10
    writes = len(renderer.overlay_texture.writes)
    now[0] += 1
    renderer._draw_overlay(["STABLE TITLE", "FPS 59"])
    assert len(renderer.overlay_texture.writes) == writes


def test_partial_uploads_match_reference_on_real_gl_texture(monkeypatch):
    from app.main import project_root
    renderer = proxy_renderer.ProxyRenderer(project_root(), (384, 256), False,
                                             window_size=(800, 600))
    now = [100.]
    monkeypatch.setattr(proxy_renderer, "perf_counter", lambda: now[0])
    try:
        for lines, caption in ((["HEADER", "diagnostic value 60"], "caption " * 50),
                               (["HEADER", "diagnostic value 59"], "changed caption"),
                               ([], "caption alone"), (["HEADER"], None), ([], None)):
            now[0] += .11
            renderer._draw_overlay(lines, caption)
            assert renderer.overlay_texture.read() == original_pixels(renderer, lines, caption)
    finally:
        renderer.close()
