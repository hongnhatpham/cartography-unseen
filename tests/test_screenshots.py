from pathlib import Path

import numpy as np
from PIL import Image
import pygame
import pytest

from app.renderer.camera import Camera
from app.renderer.player_trail import PlayerTrail
from app.renderer.proxy_renderer import ProxyRenderer
from app.screenshots import ScreenshotWriter
from app.types import GeneratedFrame


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("presentation", ["raw", "reprojected", "fallback"])
def test_saved_png_matches_displayed_ai_without_text_or_trail(monkeypatch, tmp_path, presentation):
    renderer = ProxyRenderer(ROOT, (128, 128), False, window_size=(800, 500))
    screenshots = ScreenshotWriter(tmp_path / "screenshot")
    try:
        monkeypatch.setattr(pygame.display, "flip", lambda: None)
        camera = Camera.create_default()
        renderer.spawn_camera(camera)
        frame = renderer.render_proxy(camera, 1.)
        generated = GeneratedFrame(frame.rgb, frame.depth, frame.camera.view_matrix,
            frame.camera.projection_matrix, frame.camera.position, frame.camera.rotation, 1., 1., frame.sequence)
        trail = PlayerTrail()
        for index in range(4):
            trail.update(camera.position - camera.forward * (12. - index * 4), index / 3.)
        if presentation == "fallback":
            monkeypatch.setattr(renderer, "_world_contains", lambda position: False)

        def draw(**kwargs):
            if presentation == "raw":
                renderer.display(generated.image, sharpen=1.7, trail_frame=generated, **kwargs)
            else:
                renderer.display_reprojected(generated, camera, sharpen=1.7, **kwargs)

        draw()
        size = renderer.window_size
        expected = renderer.ctx.screen.read(viewport=(0, 0, *size), components=3, alignment=1)
        draw(overlay_lines=["DIAGNOSTIC TEXT MUST NOT BE SAVED"],
             prompt_caption="PROMPT TEXT MUST NOT BE SAVED", idle_opacity=1.,
             trail=trail, trail_time=1., screenshot=screenshots)
        screenshots.close()
        files = list((tmp_path / "screenshot").glob("*.png"))
        assert len(files) == 1
        with Image.open(files[0]) as saved:
            assert saved.size == size
            expected_pixels = np.flipud(np.frombuffer(expected, dtype=np.uint8).reshape(size[1], size[0], 3))
            np.testing.assert_array_equal(np.asarray(saved), expected_pixels)
        decorated = renderer.ctx.screen.read(viewport=(0, 0, *size), components=3, alignment=1)
        assert decorated != expected
        np.testing.assert_array_equal(renderer.capture_conditioning(frame.camera, 2.).rgb, frame.rgb)
    finally:
        screenshots.close()
        renderer.close()


def test_screenshot_write_errors_are_reported_without_crashing(tmp_path):
    folder = tmp_path / "not-a-folder"
    folder.write_text("existing file")
    writer = ScreenshotWriter(folder)
    try:
        writer.submit(bytes(12), (2, 2))
        writer._executor.shutdown(wait=True)
        assert writer.poll().startswith("SCREENSHOT FAILED:")
        assert folder.read_text() == "existing file"
    finally:
        writer.close()
