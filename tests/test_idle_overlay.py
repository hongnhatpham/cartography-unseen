"""Corner labels fade without uploading text again or covering the centre."""
from pathlib import Path

import moderngl
import numpy as np
import pygame

from app.renderer.idle_overlay import IdleOverlay


ROOT = Path(__file__).resolve().parents[1]


def test_corner_labels_fade_on_gpu_and_clear_the_caption():
    pygame.font.init()
    ctx = moderngl.create_standalone_context(require=330)
    size = (800, 500)
    color = ctx.texture(size, 3)
    target = ctx.framebuffer(color)
    overlay = IdleOverlay(ctx, ROOT, size)
    try:
        target.use()
        ctx.viewport = (0, 0, *size)
        textures = tuple(overlay.textures)

        def render(alpha, caption_height=0):
            target.clear()
            overlay.draw(size, alpha, caption_height)
            return np.flipud(np.frombuffer(target.read(components=3), dtype=np.uint8).reshape(500, 800, 3))

        full = render(1.)
        half = render(.5)
        assert full[:100, :350].max() > 200
        assert full[300:, 500:].max() > 200
        title, controls = overlay.textures
        assert full[23 + title.height:500 - 24 - controls.height].max() == 0
        assert .48 < half.sum() / full.sum() < .52
        assert render(0.).max() == 0
        lifted = render(1., 87)
        assert lifted[500 - 87 - 24:].max() == 0
        assert tuple(overlay.textures) == textures
    finally:
        overlay.close()
        target.release()
        color.release()
        ctx.release()
