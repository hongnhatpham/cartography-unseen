"""Present the short route filament without sending it through diffusion."""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import moderngl
import numpy as np

from app.types import CameraSnapshot

if TYPE_CHECKING:
    from app.renderer.player_trail import PlayerTrail


class TrailRenderer:
    def __init__(self, ctx: moderngl.Context, shader_root: Path) -> None:
        self.ctx = ctx
        self.program = ctx.program(
            vertex_shader=(shader_root / "trail.vert").read_text(encoding="utf-8"),
            geometry_shader=(shader_root / "trail.geom").read_text(encoding="utf-8"),
            fragment_shader=(shader_root / "trail.frag").read_text(encoding="utf-8"),
        )
        self.buffer = ctx.buffer(reserve=640 * 4 * 4)
        self.vao = ctx.vertex_array(
            self.program, [(self.buffer, "3f 1f", "in_position", "in_opacity")]
        )
        self._depth_texture: moderngl.Texture | None = None
        # Retain the array itself: Python can reuse the id of a released frame.
        self._depth_source: np.ndarray | None = None

    def _depth(self, data: np.ndarray | moderngl.Texture) -> moderngl.Texture:
        if not isinstance(data, np.ndarray):
            return data
        if data is self._depth_source:
            assert self._depth_texture is not None
            return self._depth_texture
        size = (data.shape[1], data.shape[0])
        if self._depth_texture is None or self._depth_texture.size != size:
            if self._depth_texture is not None:
                self._depth_texture.release()
            self._depth_texture = self.ctx.texture(size, 1, dtype="f4")
            self._depth_texture.filter = (moderngl.NEAREST, moderngl.NEAREST)
            self._depth_texture.repeat_x = False
            self._depth_texture.repeat_y = False
        self._depth_texture.write(np.ascontiguousarray(np.flipud(data), dtype="f4").tobytes())
        self._depth_source = data
        return self._depth_texture

    def draw(
        self,
        trail: PlayerTrail,
        now: float,
        camera: CameraSnapshot,
        depth: np.ndarray | moderngl.Texture,
        window_size: tuple[int, int],
        uv_scale: tuple[float, float],
        uv_offset: tuple[float, float],
        until: float | None = None,
    ) -> None:
        vertices = trail.vertices(camera.position, now, until)
        if not len(vertices):
            return
        if vertices.nbytes > self.buffer.size:
            self.buffer.orphan(vertices.nbytes)
        self.buffer.write(vertices.tobytes())
        # Vertices already use the camera as origin, preserving precision at
        # remote world coordinates. Only its view rotation remains to apply.
        view = np.array(camera.view_matrix, dtype="f4", copy=True)
        view[:3, 3] = 0.0
        self.program["view_projection"].write(
            np.asarray(camera.projection_matrix @ view, dtype="f4").T.tobytes()
        )
        self.program["uv_scale"].value = uv_scale
        self.program["uv_offset"].value = uv_offset
        self.program["window_size"].value = window_size
        self._depth(depth).use(1)
        self.program["scene_depth"].value = 1
        self.ctx.disable(moderngl.DEPTH_TEST | moderngl.CULL_FACE)
        self.ctx.enable(moderngl.BLEND)
        self.ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
        self.vao.render(mode=moderngl.LINES, vertices=len(vertices))
        self.ctx.disable(moderngl.BLEND)

    def close(self) -> None:
        self.vao.release()
        self.buffer.release()
        self.program.release()
        if self._depth_texture is not None:
            self._depth_texture.release()
        self._depth_source = None
