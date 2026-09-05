"""Small, cached corner labels, composited after generation with a GPU fade."""
from pathlib import Path

import moderngl
import numpy as np
import pygame


CONTROLS = (
    "W A S D  MOVE",
    "MOUSE  LOOK",
    "Q / E  DOWN / UP",
    "SHIFT  MOVE FASTER",
    "SPACE  CHANGE THE SCENE",
    "ENTER  SAVE IMAGE",
)


def instruction_surfaces(font_root: Path, window_size: tuple[int, int]) -> tuple[pygame.Surface, pygame.Surface]:
    """Rasterize only on launch or resize; font files also work fully offline."""
    scale = 2.25 * max(.95, min(1.4, window_size[0] / 1600, window_size[1] / 1000))
    title_font = pygame.font.Font(str(font_root / "SpaceGrotesk-SemiBold.woff"), round(26 * scale))
    controls_font = pygame.font.Font(str(font_root / "IBMPlexMono-Regular.woff"), round(14 * scale))

    def text(font: pygame.font.Font, label: str) -> pygame.Surface:
        ink = font.render(label, True, (234, 231, 221))
        shadow = font.render(label, True, (8, 12, 14))
        shadow.set_alpha(150)
        result = pygame.Surface((ink.get_width() + 4, ink.get_height() + 4), pygame.SRCALPHA)
        # A slight shadow keeps lettering readable without a backing panel.
        for offset in ((1, 2), (2, 2)):
            result.blit(shadow, offset)
        result.blit(ink, (1, 1))
        return result

    title = text(title_font, "Cartography Unseen")
    rows = [text(controls_font, label) for label in CONTROLS]
    line_height = round(24 * scale)
    gutter = round(12 * scale)
    width = max(row.get_width() for row in rows) + gutter
    height = line_height * (len(rows) - 1) + rows[-1].get_height()
    controls = pygame.Surface((width, height), pygame.SRCALPHA)
    for index, row in enumerate(rows):
        controls.blit(row, (width - gutter - row.get_width(), index * line_height))
    pygame.draw.line(controls, (234, 231, 221, 115), (width - 1, 3), (width - 1, height - 4))
    # Retain the requested size on exhibition displays, but fit small windows.
    available_width = max(1, round(window_size[0] * .95))
    fit = min(1., available_width / max(title.get_width(), controls.get_width()),
              max(1, window_size[1] * .6) / (title.get_height() + controls.get_height()))
    if fit < 1.:
        title, controls = (pygame.transform.smoothscale(surface,
                           (max(1, round(surface.get_width() * fit)), max(1, round(surface.get_height() * fit))))
                           for surface in (title, controls))
    return title, controls


class IdleOverlay:
    def __init__(self, ctx: moderngl.Context, project_root: Path, window_size: tuple[int, int]) -> None:
        self.ctx = ctx
        self.font_root = project_root / "assets/fonts"
        self.program = ctx.program(
            vertex_shader=(project_root / "shaders/idle.vert").read_text(encoding="utf-8"),
            fragment_shader=(project_root / "shaders/idle.frag").read_text(encoding="utf-8"),
        )
        self.buffer = ctx.buffer(np.array([(0, 0), (1, 0), (1, 1), (0, 0), (1, 1), (0, 1)], dtype="f4").tobytes())
        self.vao = ctx.vertex_array(self.program, [(self.buffer, "2f", "in_position")])
        self.textures: list[moderngl.Texture] = []
        self.window_size = (0, 0)
        self.prepare(window_size)

    def prepare(self, window_size: tuple[int, int]) -> None:
        if self.window_size == window_size:
            return
        for texture in self.textures:
            texture.release()
        self.textures = []
        for surface in instruction_surfaces(self.font_root, window_size):
            texture = self.ctx.texture(surface.get_size(), 4, pygame.image.tostring(surface, "RGBA"))
            texture.filter = (moderngl.LINEAR, moderngl.LINEAR)
            self.textures.append(texture)
        self.window_size = window_size

    def draw(self, window_size: tuple[int, int], opacity: float, caption_height: int = 0) -> None:
        if opacity <= 0.0:
            return
        self.prepare(window_size)
        width, height = window_size
        margin_x, margin_y = round(width * .025), round(height * .045)
        title, controls = self.textures
        positions = ((margin_x, margin_y),
                     (width - margin_x - controls.width,
                      height - max(margin_y, caption_height + 24) - controls.height))
        self.ctx.disable(moderngl.DEPTH_TEST | moderngl.CULL_FACE)
        self.ctx.enable(moderngl.BLEND)
        self.ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
        self.program["window_size"].value = window_size
        self.program["opacity"].value = min(1.0, opacity)
        self.program["image_texture"].value = 0
        for texture, position in zip(self.textures, positions):
            self.program["rect"].value = (*position, *texture.size)
            texture.use(0)
            self.vao.render()
        self.ctx.disable(moderngl.BLEND)

    def close(self) -> None:
        for resource in (*self.textures, self.vao, self.buffer, self.program):
            resource.release()
