from __future__ import annotations

from dataclasses import dataclass
from math import cos, pi, radians, sin
from pathlib import Path
from time import perf_counter
import textwrap

import numpy as np

try:
    import moderngl
    import pygame
except ImportError as exc:  # pragma: no cover - rendered as a startup error by main
    raise RuntimeError(
        "Renderer dependencies are missing. Run setup_dev.ps1 or tools/prepare_runtime.ps1."
    ) from exc

from app.renderer.camera import Camera
from app.types import ConditioningFrame, GeneratedFrame


def _cube_vertices() -> np.ndarray:
    faces = (
        ((0, 0, 1), ((-1, -1, 1), (1, -1, 1), (1, 1, 1), (-1, 1, 1))),
        ((0, 0, -1), ((1, -1, -1), (-1, -1, -1), (-1, 1, -1), (1, 1, -1))),
        ((1, 0, 0), ((1, -1, 1), (1, -1, -1), (1, 1, -1), (1, 1, 1))),
        ((-1, 0, 0), ((-1, -1, -1), (-1, -1, 1), (-1, 1, 1), (-1, 1, -1))),
        ((0, 1, 0), ((-1, 1, 1), (1, 1, 1), (1, 1, -1), (-1, 1, -1))),
        ((0, -1, 0), ((-1, -1, -1), (1, -1, -1), (1, -1, 1), (-1, -1, 1))),
    )
    rows: list[tuple[float, ...]] = []
    for normal, corners in faces:
        for index in (0, 1, 2, 0, 2, 3):
            rows.append((*corners[index], *normal))
    return np.asarray(rows, dtype="f4")


def _sphere_vertices(segments: int = 20, rings: int = 12) -> np.ndarray:
    rows: list[tuple[float, ...]] = []
    for ring in range(rings):
        latitude_0 = -pi * 0.5 + pi * ring / rings
        latitude_1 = -pi * 0.5 + pi * (ring + 1) / rings
        for segment in range(segments):
            longitude_0 = 2.0 * pi * segment / segments
            longitude_1 = 2.0 * pi * (segment + 1) / segments

            def point(latitude: float, longitude: float) -> tuple[float, float, float]:
                return (
                    cos(latitude) * cos(longitude),
                    sin(latitude),
                    cos(latitude) * sin(longitude),
                )

            points = (
                point(latitude_0, longitude_0),
                point(latitude_0, longitude_1),
                point(latitude_1, longitude_1),
                point(latitude_1, longitude_0),
            )
            for index in (0, 1, 2, 0, 2, 3):
                vertex = points[index]
                rows.append((*vertex, *vertex))
    return np.asarray(rows, dtype="f4")


def _cylinder_vertices(segments: int = 24) -> np.ndarray:
    rows: list[tuple[float, ...]] = []
    for segment in range(segments):
        angle_0 = 2.0 * pi * segment / segments
        angle_1 = 2.0 * pi * (segment + 1) / segments
        x0, z0 = cos(angle_0), sin(angle_0)
        x1, z1 = cos(angle_1), sin(angle_1)
        side = (
            (x0, -1.0, z0, x0, 0.0, z0),
            (x1, -1.0, z1, x1, 0.0, z1),
            (x1, 1.0, z1, x1, 0.0, z1),
            (x0, 1.0, z0, x0, 0.0, z0),
        )
        for index in (0, 1, 2, 0, 2, 3):
            rows.append(side[index])
        rows.extend(
            [
                (0.0, 1.0, 0.0, 0.0, 1.0, 0.0),
                (x0, 1.0, z0, 0.0, 1.0, 0.0),
                (x1, 1.0, z1, 0.0, 1.0, 0.0),
                (0.0, -1.0, 0.0, 0.0, -1.0, 0.0),
                (x1, -1.0, z1, 0.0, -1.0, 0.0),
                (x0, -1.0, z0, 0.0, -1.0, 0.0),
            ]
        )
    return np.asarray(rows, dtype="f4")


def _reprojection_grid(subdivisions: int = 192) -> np.ndarray:
    """Dense triangle grid used to forward-warp source depth on the GPU."""
    coordinates = np.linspace(0.0, 1.0, subdivisions + 1, dtype=np.float32)
    rows: list[tuple[float, float]] = []
    for y_index in range(subdivisions):
        y0, y1 = coordinates[y_index], coordinates[y_index + 1]
        for x_index in range(subdivisions):
            x0, x1 = coordinates[x_index], coordinates[x_index + 1]
            rows.extend(((x0, y0), (x1, y0), (x1, y1), (x0, y0), (x1, y1), (x0, y1)))
    return np.asarray(rows, dtype="f4")


def _transform(
    position: tuple[float, float, float],
    scale: tuple[float, float, float],
    rotation: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> np.ndarray:
    rx, ry, rz = (radians(value) for value in rotation)
    scale_matrix = np.diag((*scale, 1.0)).astype(np.float32)
    rotate_x = np.array(
        [[1, 0, 0, 0], [0, cos(rx), -sin(rx), 0], [0, sin(rx), cos(rx), 0], [0, 0, 0, 1]],
        dtype=np.float32,
    )
    rotate_y = np.array(
        [[cos(ry), 0, sin(ry), 0], [0, 1, 0, 0], [-sin(ry), 0, cos(ry), 0], [0, 0, 0, 1]],
        dtype=np.float32,
    )
    rotate_z = np.array(
        [[cos(rz), -sin(rz), 0, 0], [sin(rz), cos(rz), 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
        dtype=np.float32,
    )
    translation = np.identity(4, dtype=np.float32)
    translation[:3, 3] = position
    return translation @ rotate_z @ rotate_y @ rotate_x @ scale_matrix


@dataclass(frozen=True, slots=True)
class SceneObject:
    mesh: str
    model: np.ndarray
    color: tuple[float, float, float]


class ProxyRenderer:
    def __init__(
        self,
        project_root: Path,
        resolution: int | tuple[int, int],
        fullscreen: bool,
        window_size: tuple[int, int] = (640, 384),
        display_monitor: int = 0,
        world_seed: int = 12345,
    ) -> None:
        pygame.init()
        pygame.font.init()
        pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MAJOR_VERSION, 3)
        pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MINOR_VERSION, 3)
        pygame.display.gl_set_attribute(pygame.GL_CONTEXT_PROFILE_MASK, pygame.GL_CONTEXT_PROFILE_CORE)
        pygame.display.gl_set_attribute(pygame.GL_DOUBLEBUFFER, 1)
        flags = pygame.OPENGL | pygame.DOUBLEBUF | pygame.RESIZABLE
        desktops = pygame.display.get_desktop_sizes()
        if display_monitor >= len(desktops):
            detected = ", ".join(f"{index}:{size[0]}x{size[1]}" for index, size in enumerate(desktops))
            raise RuntimeError(
                f"display_monitor {display_monitor} is unavailable. Detected monitors: {detected or 'none'}"
            )
        # Create a real windowed mode first so SDL remembers a useful restore
        # size when an app launched fullscreen later receives F11.
        pygame.display.set_mode(window_size, flags, display=display_monitor, vsync=0)
        if fullscreen:
            result = pygame.display.toggle_fullscreen()
            if result < 0:
                raise RuntimeError(f"SDL could not enter fullscreen: {pygame.get_error()}")
        window_size = pygame.display.get_window_size()
        pygame.display.set_caption("Latent Space")
        pygame.event.set_grab(True)
        pygame.mouse.set_visible(False)
        pygame.mouse.get_rel()

        self.ctx = moderngl.create_context(require=330)
        self.ctx.enable(moderngl.DEPTH_TEST | moderngl.CULL_FACE)
        self.project_root = project_root
        if isinstance(resolution, int):
            resolution = (resolution, resolution)
        self.render_width, self.render_height = resolution
        self.window_size = window_size
        self.font = pygame.font.Font(None, 24)
        self.small_font = pygame.font.Font(None, 20)

        shader_root = project_root / "shaders"
        self.proxy_program = self.ctx.program(
            vertex_shader=(shader_root / "proxy.vert").read_text(encoding="utf-8"),
            fragment_shader=(shader_root / "proxy.frag").read_text(encoding="utf-8"),
        )
        self.screen_program = self.ctx.program(
            vertex_shader=(shader_root / "screen.vert").read_text(encoding="utf-8"),
            fragment_shader=(shader_root / "screen.frag").read_text(encoding="utf-8"),
        )
        self.reproject_program = self.ctx.program(
            vertex_shader=(shader_root / "reproject.vert").read_text(encoding="utf-8"),
            fragment_shader=(shader_root / "reproject.frag").read_text(encoding="utf-8"),
        )

        meshes = {
            "cube": _cube_vertices(),
            "sphere": _sphere_vertices(),
            "cylinder": _cylinder_vertices(),
        }
        self.mesh_buffers = {name: self.ctx.buffer(vertices.tobytes()) for name, vertices in meshes.items()}
        self.mesh_vaos = {
            name: self.ctx.vertex_array(
                self.proxy_program,
                [(buffer, "3f 3f", "in_position", "in_normal")],
            )
            for name, buffer in self.mesh_buffers.items()
        }
        quad = np.asarray(
            [
                (-1, -1, 0, 0), (1, -1, 1, 0), (1, 1, 1, 1),
                (-1, -1, 0, 0), (1, 1, 1, 1), (-1, 1, 0, 1),
            ],
            dtype="f4",
        )
        self.quad_buffer = self.ctx.buffer(quad.tobytes())
        self.quad_vao = self.ctx.vertex_array(
            self.screen_program,
            [(self.quad_buffer, "2f 2f", "in_position", "in_uv")],
        )
        self.reproject_vao = self.ctx.vertex_array(
            self.reproject_program,
            [(self.quad_buffer, "2f 2f", "in_position", "in_uv")],
        )
        render_size = (self.render_width, self.render_height)
        self.color_texture = self.ctx.texture(render_size, 3, dtype="f1")
        self.color_texture.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self.depth_texture = self.ctx.depth_texture(render_size)
        self.depth_texture.filter = (moderngl.NEAREST, moderngl.NEAREST)
        self.depth_texture.compare_func = ""
        self.proxy_fbo = self.ctx.framebuffer(self.color_texture, self.depth_texture)
        self.display_texture = self.ctx.texture(render_size, 3, dtype="f1")
        self.display_texture.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self.display_texture.repeat_x = False
        self.display_texture.repeat_y = False
        self.reproject_source_depth = self.ctx.texture(render_size, 1, dtype="f4")
        self.reproject_source_depth.filter = (moderngl.NEAREST, moderngl.NEAREST)
        self.reproject_source_depth.repeat_x = False
        self.reproject_source_depth.repeat_y = False
        self.reproject_texture = self.ctx.texture(render_size, 3, dtype="f1")
        self.reproject_texture.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self.reproject_texture.repeat_x = False
        self.reproject_texture.repeat_y = False
        self.reproject_depth = self.ctx.depth_texture(render_size)
        self.reproject_fbo = self.ctx.framebuffer(self.reproject_texture, self.reproject_depth)
        self.overlay_texture = self.ctx.texture(window_size, 4, dtype="f1")
        self.overlay_texture.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self._overlay_size = window_size
        self._last_overlay_update = 0.0
        self._last_overlay_key = ""
        self.loading_image = self._make_loading_image(self.render_width, self.render_height)
        self._reproject_sequence = -1
        self.reproject_ms = 0.0

        self.world_seed = world_seed
        self.scene = self._make_scene(world_seed)
        self.sequence = 0

    def toggle_fullscreen(self) -> bool:
        """Switch display mode without rebuilding the active OpenGL context."""
        result = pygame.display.toggle_fullscreen()
        if result < 0:
            raise RuntimeError(f"SDL could not toggle fullscreen: {pygame.get_error()}")
        is_fullscreen = bool(pygame.display.is_fullscreen())
        if not is_fullscreen:
            self.resize_window_to_render()
        self.window_size = pygame.display.get_window_size()
        self._last_overlay_update = 0.0
        pygame.event.set_grab(True)
        pygame.mouse.set_visible(False)
        pygame.mouse.get_rel()
        return is_fullscreen

    def resize_window_to_render(self) -> None:
        """Match a windowed SDL window to the active generation dimensions."""
        if pygame.display.is_fullscreen():
            return
        from pygame._sdl2 import Window

        window = Window.from_display_module()
        window.size = (self.render_width, self.render_height)
        self.window_size = pygame.display.get_window_size()
        self._last_overlay_update = 0.0

    def set_resolution(self, resolution: tuple[int, int]) -> None:
        """Rebuild offscreen targets for a new diffusion aspect mode."""
        width, height = (int(resolution[0]), int(resolution[1]))
        if (width, height) == (self.render_width, self.render_height):
            return
        for resource in (
            self.proxy_fbo,
            self.reproject_fbo,
            self.color_texture,
            self.depth_texture,
            self.display_texture,
            self.reproject_source_depth,
            self.reproject_texture,
            self.reproject_depth,
        ):
            resource.release()
        self.render_width, self.render_height = width, height
        render_size = (width, height)
        self.color_texture = self.ctx.texture(render_size, 3, dtype="f1")
        self.color_texture.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self.depth_texture = self.ctx.depth_texture(render_size)
        self.depth_texture.filter = (moderngl.NEAREST, moderngl.NEAREST)
        self.depth_texture.compare_func = ""
        self.proxy_fbo = self.ctx.framebuffer(self.color_texture, self.depth_texture)
        self.display_texture = self.ctx.texture(render_size, 3, dtype="f1")
        self.display_texture.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self.display_texture.repeat_x = False
        self.display_texture.repeat_y = False
        self.reproject_source_depth = self.ctx.texture(render_size, 1, dtype="f4")
        self.reproject_source_depth.filter = (moderngl.NEAREST, moderngl.NEAREST)
        self.reproject_source_depth.repeat_x = False
        self.reproject_source_depth.repeat_y = False
        self.reproject_texture = self.ctx.texture(render_size, 3, dtype="f1")
        self.reproject_texture.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self.reproject_texture.repeat_x = False
        self.reproject_texture.repeat_y = False
        self.reproject_depth = self.ctx.depth_texture(render_size)
        self.reproject_fbo = self.ctx.framebuffer(
            self.reproject_texture, self.reproject_depth
        )
        self.loading_image = self._make_loading_image(width, height)
        self._reproject_sequence = -1
        self.resize_window_to_render()

    @staticmethod
    def _make_scene(seed: int) -> list[SceneObject]:
        rng = np.random.default_rng(seed)
        palette = np.asarray(
            [
                (0.12, 0.55, 0.46),
                (0.22, 0.68, 0.72),
                (0.52, 0.25, 0.66),
                (0.77, 0.36, 0.19),
                (0.72, 0.66, 0.22),
                (0.25, 0.38, 0.73),
                (0.68, 0.22, 0.48),
            ],
            dtype=np.float32,
        )

        def color(offset: int = 0) -> tuple[float, float, float]:
            index = (int(rng.integers(0, len(palette))) + offset) % len(palette)
            return tuple(float(value) for value in palette[index])

        road_center_z = -200.0
        road_half_length = 210.0
        objects = [
            # Continuous ground, asphalt and raised sidewalks keep the central
            # path readable to both the viewer and the diffusion model.
            SceneObject(
                "cube",
                _transform((0.0, -0.42, road_center_z), (28.0, 0.34, road_half_length)),
                (0.10, 0.17, 0.16),
            ),
            SceneObject(
                "cube",
                _transform((0.0, -0.04, road_center_z), (5.2, 0.06, road_half_length)),
                (0.13, 0.15, 0.18),
            ),
            SceneObject(
                "cube",
                _transform((-6.15, 0.06, road_center_z), (0.85, 0.12, road_half_length)),
                color(1),
            ),
            SceneObject(
                "cube",
                _transform((6.15, 0.06, road_center_z), (0.85, 0.12, road_half_length)),
                color(1),
            ),
        ]

        # Broken center lines make forward movement legible without placing
        # geometry across the road. The route reaches past -400 Z, three times
        # the previous approximately -130 Z corridor.
        for marker in range(34):
            z = -4.0 - marker * 12.0
            objects.append(
                SceneObject(
                    "cube",
                    _transform((0.0, 0.035, z), (0.10, 0.035, 2.6)),
                    (0.86, 0.78, 0.30),
                )
            )

        # A varied but clean-sided city wall runs along both sides of the road.
        # Box architecture avoids the severe depth discontinuities caused by
        # the former torus portals while preserving a strong vanishing point.
        for row in range(50):
            z = -5.0 - row * 8.0 + float(rng.uniform(-0.7, 0.7))
            for side_index, side in enumerate((-1.0, 1.0)):
                half_width = float(rng.uniform(1.6, 3.2))
                half_height = float(rng.uniform(2.8, 8.5))
                half_depth = float(rng.uniform(2.3, 3.7))
                setback = float(rng.uniform(0.3, 2.4))
                x = side * (7.4 + setback + half_width)
                building_color = color(row + side_index)
                objects.append(
                    SceneObject(
                        "cube",
                        _transform((x, half_height, z), (half_width, half_height, half_depth)),
                        building_color,
                    )
                )

                # Stepped rooftops and occasional narrow towers give each seed
                # a distinct skyline while keeping all geometry road-side.
                if rng.random() < 0.72:
                    crown_height = float(rng.uniform(0.45, 1.8))
                    crown_width = half_width * float(rng.uniform(0.35, 0.72))
                    crown_depth = half_depth * float(rng.uniform(0.35, 0.72))
                    objects.append(
                        SceneObject(
                            "cube",
                            _transform(
                                (x, half_height * 2.0 + crown_height, z),
                                (crown_width, crown_height, crown_depth),
                            ),
                            color(row + side_index + 2),
                        )
                    )
                if rng.random() < 0.22:
                    mast_height = float(rng.uniform(1.2, 3.5))
                    objects.append(
                        SceneObject(
                            "cylinder",
                            _transform(
                                (x, half_height * 2.0 + mast_height, z),
                                (0.12, mast_height, 0.12),
                            ),
                            color(row + side_index + 4),
                        )
                    )
        return objects

    def randomize_world(self) -> int:
        self.world_seed = int(np.random.SeedSequence().generate_state(1, dtype=np.uint32)[0])
        self.scene = self._make_scene(self.world_seed)
        self._reproject_sequence = -1
        return self.world_seed

    @staticmethod
    def _make_loading_image(width: int, height: int) -> np.ndarray:
        yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
        x = (xx / max(width - 1, 1)) * 2.0 - 1.0
        y = (yy / max(height - 1, 1)) * 2.0 - 1.0
        glow = np.exp(-((x * 0.75) ** 2 + (y * 1.15) ** 2) * 2.3)
        image = np.empty((height, width, 3), dtype=np.float32)
        image[:, :, 0] = 12 + glow * 48 + (1.0 - y) * 8
        image[:, :, 1] = 16 + glow * 28
        image[:, :, 2] = 31 + glow * 62 + (x + 1.0) * 5
        return np.clip(image, 0, 255).astype(np.uint8)

    def render_scene(self, camera: Camera):
        snapshot = camera.snapshot(aspect=self.render_width / self.render_height)
        self.proxy_fbo.use()
        self.ctx.viewport = (0, 0, self.render_width, self.render_height)
        self.ctx.enable(moderngl.DEPTH_TEST)
        self.ctx.disable(moderngl.CULL_FACE)
        self.proxy_fbo.clear(0.34, 0.50, 0.66, 1.0, depth=1.0)
        self.proxy_program["view"].write(snapshot.view_matrix.T.astype("f4").tobytes())
        self.proxy_program["projection"].write(snapshot.projection_matrix.T.astype("f4").tobytes())
        for item in self.scene:
            self.proxy_program["model"].write(item.model.T.astype("f4").tobytes())
            self.proxy_program["material_color"].value = item.color
            self.mesh_vaos[item.mesh].render()
        return snapshot

    def capture_conditioning(self, snapshot, timestamp: float) -> ConditioningFrame:
        rgb = np.frombuffer(self.color_texture.read(alignment=1), dtype=np.uint8)
        rgb = np.flipud(rgb.reshape(self.render_height, self.render_width, 3)).copy()
        depth = np.frombuffer(self.depth_texture.read(alignment=1), dtype=np.float32)
        depth = np.flipud(depth.reshape(self.render_height, self.render_width)).copy()
        edges = self._edges(rgb, depth)
        self.sequence += 1
        return ConditioningFrame(rgb, depth, edges, snapshot, timestamp, self.sequence)

    def render_proxy(self, camera: Camera, timestamp: float) -> ConditioningFrame:
        return self.capture_conditioning(self.render_scene(camera), timestamp)

    @staticmethod
    def _edges(rgb: np.ndarray, depth: np.ndarray) -> np.ndarray:
        luminance = (
            rgb[:, :, 0].astype(np.float32) * 0.2126
            + rgb[:, :, 1].astype(np.float32) * 0.7152
            + rgb[:, :, 2].astype(np.float32) * 0.0722
        ) / 255.0
        gx = np.abs(np.diff(luminance, axis=1, append=luminance[:, -1:]))
        gy = np.abs(np.diff(luminance, axis=0, append=luminance[-1:, :]))
        dx = np.abs(np.diff(depth, axis=1, append=depth[:, -1:]))
        dy = np.abs(np.diff(depth, axis=0, append=depth[-1:, :]))
        edge = np.maximum((gx + gy) * 2.2, np.clip((dx + dy) * 80.0, 0.0, 1.0))
        return (np.clip(edge, 0.0, 1.0) * 255.0).astype(np.uint8)

    @staticmethod
    def diagnostic_image(frame: ConditioningFrame, mode: str) -> np.ndarray:
        if mode == "depth":
            depth = frame.depth
            near, far = np.percentile(depth, (2, 92))
            normalized = 1.0 - np.clip((depth - near) / max(far - near, 1e-6), 0.0, 1.0)
            return np.stack(
                [normalized * 80, normalized * 190, normalized * 255], axis=2
            ).astype(np.uint8)
        if mode == "edges":
            return np.repeat(frame.edges[:, :, None], 3, axis=2)
        return frame.rgb

    def display(
        self,
        image: np.ndarray,
        overlay_lines: list[str] | None = None,
        sharpen: float = 0.3,
        prompt_caption: str | None = None,
    ) -> None:
        target_shape = (self.render_height, self.render_width)
        if image.shape[:2] != target_shape:
            surface = pygame.surfarray.make_surface(np.swapaxes(image, 0, 1))
            surface = pygame.transform.smoothscale(
                surface, (self.render_width, self.render_height)
            )
            image = np.swapaxes(pygame.surfarray.array3d(surface), 0, 1)
        # Always upload the selected presentation image. Python can recycle an
        # ndarray object's identity, so identity-based caching occasionally left
        # the proxy texture visually frozen even though the camera was moving.
        self.display_texture.write(np.ascontiguousarray(np.flipud(image)).tobytes())

        self._present_texture(
            self.display_texture, overlay_lines, sharpen, prompt_caption
        )

    def display_reprojected(
        self,
        frame: GeneratedFrame,
        camera: Camera,
        overlay_lines: list[str] | None = None,
        strength: float = 1.0,
        max_translation: float = 1.25,
        max_rotation: float = 12.0,
        sharpen: float = 0.3,
        prompt_caption: str | None = None,
    ) -> None:
        started = perf_counter()
        if frame.sequence != self._reproject_sequence:
            image = frame.image
            target_shape = (self.render_height, self.render_width)
            if image.shape[:2] != target_shape:
                surface = pygame.surfarray.make_surface(np.swapaxes(image, 0, 1))
                surface = pygame.transform.smoothscale(
                    surface, (self.render_width, self.render_height)
                )
                image = np.swapaxes(pygame.surfarray.array3d(surface), 0, 1)
            self.display_texture.write(np.ascontiguousarray(np.flipud(image)).tobytes())
            depth = frame.depth
            if depth.shape != target_shape:
                depth_surface = pygame.surfarray.make_surface(
                    np.repeat((depth * 255.0).astype(np.uint8)[:, :, None], 3, axis=2).swapaxes(0, 1)
                )
                depth_surface = pygame.transform.smoothscale(
                    depth_surface, (self.render_width, self.render_height)
                )
                depth = pygame.surfarray.array3d(depth_surface)[:, :, 0].T.astype(np.float32) / 255.0
            self.reproject_source_depth.write(
                np.ascontiguousarray(np.flipud(depth).astype("f4")).tobytes()
            )
            self._reproject_sequence = frame.sequence

        live = camera.snapshot(aspect=self.render_width / self.render_height)
        raw_delta = live.position - frame.camera_position
        raw_distance = float(np.linalg.norm(raw_delta))
        translation_scale = min(1.0, max_translation / max(raw_distance, 1e-6))
        source_pitch = float(frame.camera_rotation[0])
        source_yaw_degrees = float(frame.camera_rotation[1])
        yaw_delta = (float(live.rotation[1]) - source_yaw_degrees + 180.0) % 360.0 - 180.0
        pitch_delta = float(live.rotation[0]) - source_pitch
        rotation_scale = min(
            1.0,
            max_rotation / max(abs(yaw_delta), abs(pitch_delta), 1e-6),
        )
        effective_strength = float(np.clip(strength * min(translation_scale, rotation_scale), 0.0, 1.0))
        delta = raw_delta * effective_strength
        yaw_delta *= effective_strength
        pitch_delta *= effective_strength
        warped_camera = Camera(
            position=(frame.camera_position + delta).astype(np.float32),
            yaw=source_yaw_degrees + yaw_delta,
            pitch=source_pitch + pitch_delta,
            fov=camera.fov,
            near=camera.near,
            far=camera.far,
        )
        current = self.render_scene(warped_camera)
        source_yaw = radians(float(frame.camera_rotation[1]))
        source_forward = np.array([sin(source_yaw), 0.0, -cos(source_yaw)], dtype=np.float32)
        source_right = np.array([cos(source_yaw), 0.0, sin(source_yaw)], dtype=np.float32)
        forward_motion = float(np.dot(delta, source_forward))
        lateral_motion = float(np.dot(delta, source_right))
        zoom = float(np.clip(np.exp(forward_motion * 0.14), 0.65, 1.8))
        base_scale = 1.0 / zoom
        base_offset = (
            (1.0 - base_scale) * 0.5 + yaw_delta / 105.0 + lateral_motion * 0.018,
            (1.0 - base_scale) * 0.5 + pitch_delta / 105.0,
        )

        source_vp = frame.projection_matrix @ frame.view_matrix
        current_vp = current.projection_matrix @ current.view_matrix
        inverse_current_vp = np.linalg.inv(current_vp).astype("f4")

        # render_scene above produces a depth texture from the exact same
        # interpolated camera used by the reprojection matrices. Keeping those
        # paired prevents the apparent scale/registration drift seen at partial
        # warp strengths.
        self.reproject_fbo.use()
        self.ctx.viewport = (0, 0, self.render_width, self.render_height)
        self.ctx.disable(moderngl.DEPTH_TEST | moderngl.CULL_FACE)
        self.reproject_fbo.clear(0.02, 0.025, 0.04, 1.0, depth=1.0)
        self.reproject_program["current_inverse_view_projection"].write(
            inverse_current_vp.T.tobytes()
        )
        self.reproject_program["source_view_projection"].write(
            source_vp.astype("f4").T.tobytes()
        )
        self.reproject_program["fallback_uv_scale"].value = (base_scale, base_scale)
        self.reproject_program["fallback_uv_offset"].value = base_offset
        self.reproject_program["warp_strength"].value = 1.0
        self.reproject_program["occlusion_tolerance"].value = 0.0015
        self.display_texture.use(0)
        self.reproject_source_depth.use(1)
        self.depth_texture.use(2)
        self.reproject_program["source_image"].value = 0
        self.reproject_program["source_depth"].value = 1
        self.reproject_program["current_depth"].value = 2
        self.ctx.disable(moderngl.DEPTH_TEST)
        self.ctx.disable(moderngl.CULL_FACE)
        self.reproject_vao.render(mode=moderngl.TRIANGLES)
        elapsed_ms = (perf_counter() - started) * 1000.0
        self.reproject_ms += 0.15 * (elapsed_ms - self.reproject_ms)
        self._present_texture(
            self.reproject_texture, overlay_lines, sharpen, prompt_caption
        )

    def _present_texture(
        self,
        texture: moderngl.Texture,
        overlay_lines: list[str] | None = None,
        sharpen: float = 0.3,
        prompt_caption: str | None = None,
    ) -> None:
        current_size = pygame.display.get_window_size()
        if current_size != self.window_size:
            self.window_size = current_size

        self.ctx.screen.use()
        self.ctx.viewport = (0, 0, *self.window_size)
        self.ctx.disable(moderngl.DEPTH_TEST | moderngl.CULL_FACE)
        self.ctx.clear(0.0, 0.0, 0.0, 1.0)
        width, height = self.window_size
        window_aspect = width / max(height, 1)
        image_aspect = self.render_width / max(self.render_height, 1)
        if window_aspect >= image_aspect:
            uv_scale = (1.0, image_aspect / window_aspect)
        else:
            uv_scale = (window_aspect / image_aspect, 1.0)
        uv_offset = ((1.0 - uv_scale[0]) * 0.5, (1.0 - uv_scale[1]) * 0.5)
        self.screen_program["uv_scale"].value = uv_scale
        self.screen_program["uv_offset"].value = uv_offset
        self.screen_program["sharpen"].value = float(sharpen)
        texture.use(0)
        self.screen_program["image_texture"].value = 0
        self.quad_vao.render()

        if overlay_lines or prompt_caption:
            self._draw_overlay(overlay_lines or [], prompt_caption)
        pygame.display.flip()

    def _draw_overlay(self, lines: list[str], prompt_caption: str | None = None) -> None:
        now = perf_counter()
        key = "\n".join(lines) + f"\nPROMPT_CAPTION:{prompt_caption or ''}"
        if self.window_size != self._overlay_size:
            self.overlay_texture.release()
            self.overlay_texture = self.ctx.texture(self.window_size, 4, dtype="f1")
            self.overlay_texture.filter = (moderngl.LINEAR, moderngl.LINEAR)
            self._overlay_size = self.window_size
            self._last_overlay_update = 0.0
        if key != self._last_overlay_key and now - self._last_overlay_update >= 0.1:
            surface = pygame.Surface(self.window_size, pygame.SRCALPHA)
            if lines:
                panel_height = min(self.window_size[1] - 12, 18 + len(lines) * 23)
                longest = max((len(line) for line in lines), default=0)
                panel_width = min(
                    self.window_size[0] - 24,
                    max(490, min(840, 42 + longest * 9)),
                )
                pygame.draw.rect(
                    surface,
                    (4, 8, 12, 51),
                    (12, 12, panel_width, panel_height),
                    border_radius=7,
                )
                y = 21
                for line in lines:
                    text_surface = self.small_font.render(line, True, (230, 240, 245))
                    surface.blit(text_surface, (24, y))
                    y += 23
            if prompt_caption:
                caption_width = max(24, self.window_size[0] - 48)
                characters = max(24, caption_width // 10)
                caption_lines = textwrap.wrap(
                    f"PROMPT  {prompt_caption}", width=characters
                )[:3]
                caption_height = 18 + len(caption_lines) * 23
                caption_y = self.window_size[1] - caption_height - 12
                pygame.draw.rect(
                    surface,
                    (0, 0, 0, 77),
                    (12, caption_y, self.window_size[0] - 24, caption_height),
                    border_radius=7,
                )
                text_y = caption_y + 9
                for line in caption_lines:
                    text_surface = self.small_font.render(
                        line, True, (240, 245, 248)
                    )
                    text_surface.set_alpha(77)
                    surface.blit(text_surface, (24, text_y))
                    text_y += 23
            data = pygame.image.tostring(surface, "RGBA", True)
            self.overlay_texture.write(data)
            self._last_overlay_key = key
            self._last_overlay_update = now
        self.ctx.enable(moderngl.BLEND)
        self.ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
        self.screen_program["uv_scale"].value = (1.0, 1.0)
        self.screen_program["uv_offset"].value = (0.0, 0.0)
        self.screen_program["sharpen"].value = 0.0
        self.overlay_texture.use(0)
        self.quad_vao.render()
        self.ctx.disable(moderngl.BLEND)

    def loading_screen(self, message: str, detail: str = "") -> None:
        image = np.zeros((self.render_height, self.render_width, 3), dtype=np.uint8)
        image[:, :, 0] = 12
        image[:, :, 1] = 19
        image[:, :, 2] = 26
        lines = ["REALTIME DIFFUSION ART", "", message]
        if detail:
            lines.append(detail)
        self.display(image, lines)

    @staticmethod
    def poll_events() -> list[pygame.event.Event]:
        return pygame.event.get()

    def close(self) -> None:
        pygame.event.set_grab(False)
        pygame.mouse.set_visible(True)
        pygame.quit()
