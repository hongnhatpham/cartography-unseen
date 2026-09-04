from __future__ import annotations

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
from app.renderer.world import (
    ACTIVE_CHUNK_RADIUS,
    CHUNK_SIZE,
    WorldChunk,
    WorldCube,
    chunk_colliders,
    open_heading,
    resolve_collisions,
    settle_height,
    step_blocked,
    generate_chunk,
    plan_chunk_cache,
    sky_color,
    spawn_pose,
    world_label,
    world_to_chunk,
)
from app.types import ConditioningFrame, GeneratedFrame

# Chunks generated per frame once the streaming window is warm. Eleven chunks
# arrive per boundary crossing and a crossing takes about twenty frames at
# cruise speed, so one per frame keeps the window warm inside the 16.7 ms budget.
CHUNK_LOAD_BUDGET = 1


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
        self.ctx.enable(moderngl.DEPTH_TEST)
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

        meshes = {"cube": _cube_vertices()}
        self.mesh_buffers = {name: self.ctx.buffer(vertices.tobytes()) for name, vertices in meshes.items()}
        self.instance_buffers = {
            name: self.ctx.buffer(reserve=19 * np.dtype("f4").itemsize)
            for name in meshes
        }
        self.mesh_vaos = {
            name: self.ctx.vertex_array(
                self.proxy_program,
                [
                    (buffer, "3f 3f", "in_position", "in_normal"),
                    (
                        self.instance_buffers[name],
                        "4f 4f 4f 4f 3f /i",
                        "instance_model_0",
                        "instance_model_1",
                        "instance_model_2",
                        "instance_model_3",
                        "instance_color",
                    ),
                ],
            )
            for name, buffer in self.mesh_buffers.items()
        }
        self._instance_counts = {name: 0 for name in meshes}
        # Observability counter: how many times the instance buffer was rewritten.
        self._instance_buffer_revision = 0
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
        self.sky_color = sky_color(world_seed)
        self._chunks: dict[tuple[int, int], WorldChunk] = {}
        self._chunk_instances: dict[tuple[int, int], np.ndarray] = {}
        self._chunk_colliders: dict[tuple[int, int], np.ndarray] = {}
        self._last_footfall: tuple[float, float] | None = None
        self._stream_center: tuple[int, int] | None = None
        self.sequence = 0
        self._closed = False

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
    def _pack_instances(objects: tuple[WorldCube, ...]) -> np.ndarray:
        """Pack column-major transforms and colors for one instanced draw.

        Vectorised over the whole chunk: a per-cube Python loop building 4x4
        matrices costs milliseconds per chunk and shows up as a frame spike.
        """
        count = len(objects)
        packed = np.zeros((count, 19), dtype="f4")
        if count == 0:
            return packed
        position = np.array([item.position for item in objects], dtype="f4")
        extents = np.array([item.half_extents for item in objects], dtype="f4")
        angles = np.radians(np.array([item.rotation for item in objects], dtype="f4"))
        cos_x, cos_y, cos_z = np.cos(angles).T
        sin_x, sin_y, sin_z = np.sin(angles).T
        zero = np.zeros(count, dtype="f4")
        one = np.ones(count, dtype="f4")
        rotate_x = np.stack(
            [one, zero, zero, zero, cos_x, -sin_x, zero, sin_x, cos_x], axis=1
        ).reshape(count, 3, 3)
        rotate_y = np.stack(
            [cos_y, zero, sin_y, zero, one, zero, -sin_y, zero, cos_y], axis=1
        ).reshape(count, 3, 3)
        rotate_z = np.stack(
            [cos_z, -sin_z, zero, sin_z, cos_z, zero, zero, zero, one], axis=1
        ).reshape(count, 3, 3)
        # Rz @ Ry @ Rx @ diag(half_extents), i.e. each column scaled by its axis.
        linear = (rotate_z @ rotate_y @ rotate_x) * extents[:, None, :]
        packed[:, 0:3] = linear[:, :, 0]
        packed[:, 4:7] = linear[:, :, 1]
        packed[:, 8:11] = linear[:, :, 2]
        packed[:, 12:15] = position
        packed[:, 15] = 1.0
        packed[:, 16:] = np.array([item.color for item in objects], dtype="f4")
        return packed

    def _update_world(self, position: np.ndarray) -> None:
        """Move the fixed chunk window, loading a bounded number of chunks.

        A cold cache (startup, or a new world seed) is filled in one go because
        a half-built landscape is worse than one hitch. Once the window is
        populated, boundary crossings load at most CHUNK_LOAD_BUDGET chunks per
        frame; plan.load is centre-out, so the nearest arrive first.
        """
        plan = plan_chunk_cache(
            self._chunks,
            float(position[0]),
            float(position[2]),
            ACTIVE_CHUNK_RADIUS,
        )
        self._stream_center = plan.center
        if not plan.load and not plan.evict:
            return

        for coord in plan.evict:
            del self._chunks[coord]
            del self._chunk_instances[coord]
            self._chunk_colliders.pop(coord, None)
        budget = len(plan.load) if not self._chunks else CHUNK_LOAD_BUDGET
        for coord in plan.load[:budget]:
            chunk = generate_chunk(coord, self.world_seed)
            self._chunks[coord] = chunk
            self._chunk_instances[coord] = self._pack_instances(chunk.objects)
            self._chunk_colliders[coord] = chunk_colliders(chunk)

        cube_data = np.concatenate(
            tuple(
                self._chunk_instances[coord]
                for coord in plan.desired
                if coord in self._chunk_instances
            ),
            axis=0,
        )
        cube_buffer = self.instance_buffers["cube"]
        if cube_data.nbytes > cube_buffer.size:
            cube_buffer.orphan(cube_data.nbytes)
        # np.concatenate is C-contiguous, so moderngl can upload it directly;
        # tobytes() here costs a second copy of a couple of megabytes per load.
        cube_buffer.write(cube_data)
        self._instance_counts["cube"] = len(cube_data)
        self._instance_buffer_revision += 1

    def _world_contains(self, position: np.ndarray) -> bool:
        return world_to_chunk(float(position[0]), float(position[2])) in self._chunks

    def spawn_camera(self, camera: Camera) -> None:
        """Stand the walker on open ground inside this seed's relief.

        Resets the footfall memory so the riser check does not compare the new
        spawn against wherever the walker last stood.

        The landform picks the spot; the forms around it are only known once
        the chunk exists, so the walker is then pushed out of anything it
        spawned inside and turned to face the longest open walk.
        """
        position, yaw, pitch = spawn_pose(self.world_seed)
        camera.position[:] = position
        camera.yaw = yaw
        camera.pitch = pitch
        self._last_footfall = None
        self._update_world(camera.position)
        colliders = self._nearby_colliders(float(position[0]), float(position[2]))
        self.constrain_camera(camera, None)
        x, y, z = (float(value) for value in camera.position)
        open_yaw, distance = open_heading(x, y, z, colliders, world_seed=self.world_seed)
        if distance > 0.0:
            camera.yaw = open_yaw

    def nearby_colliders(self, camera: Camera) -> np.ndarray:
        """Standing-form footprints around the camera, for autowalk steering."""
        return self._nearby_colliders(float(camera.position[0]), float(camera.position[2]))

    def open_heading(self, camera: Camera) -> tuple[float, float]:
        """Yaw of the longest unobstructed walk from the camera, and its length."""
        x, y, z = (float(value) for value in camera.position)
        return open_heading(x, y, z, self._nearby_colliders(x, z), world_seed=self.world_seed)

    def _nearby_colliders(self, x: float, z: float) -> np.ndarray:
        """Footprints from the 3x3 chunks around (x, z), stacked for one query."""
        home = world_to_chunk(x, z)
        nearby = [
            self._chunk_colliders[coord]
            for coord in (
                (home[0] + dx, home[1] + dz) for dx in (-1, 0, 1) for dz in (-1, 0, 1)
            )
            if coord in self._chunk_colliders
        ]
        return np.concatenate(nearby, axis=0) if nearby else np.empty((0, 8))

    def constrain_camera(self, camera: Camera, dt: float | None = None) -> None:
        """Slide the walker out of standing forms, then settle it onto the terrain.

        ``dt`` rate-limits climbs; None snaps straight to the walk height.
        """
        x, y, z = (float(value) for value in camera.position)
        x, z = resolve_collisions(x, y, z, self._nearby_colliders(x, z))
        # A riser taller than a step is a wall: keep whichever axis of the
        # move stays on walkable ground, so the walker slides along the edge.
        # This runs after form collision so a push out of a block can never
        # land the walker on the far side of a riser; the last footfall is
        # always a legal spot to fall back to.
        if self._last_footfall is not None:
            last_x, last_z = self._last_footfall
            if step_blocked(last_x, last_z, x, z, self.world_seed):
                if not step_blocked(last_x, last_z, x, last_z, self.world_seed):
                    z = last_z
                elif not step_blocked(last_x, last_z, last_x, z, self.world_seed):
                    x = last_x
                else:
                    x, z = last_x, last_z
        self._last_footfall = (x, z)
        camera.position[0] = x
        camera.position[2] = z
        camera.position[1] = settle_height(x, y, z, self.world_seed, dt)

    def world_label(self) -> str:
        """Short biome and palette label for the overlay, at the streamed center."""
        center = self._stream_center or (0, 0)
        return world_label(
            self.world_seed,
            (center[0] + 0.5) * CHUNK_SIZE,
            (center[1] + 0.5) * CHUNK_SIZE,
        )

    def randomize_world(self, world_seed: int | None = None) -> int:
        """Switch to a new (or given) world seed and drop every cached chunk."""
        self.world_seed = (
            int(np.random.SeedSequence().generate_state(1, dtype=np.uint32)[0])
            if world_seed is None
            else int(world_seed)
        )
        self.sky_color = sky_color(self.world_seed)
        self._chunks.clear()
        self._chunk_instances.clear()
        self._chunk_colliders.clear()
        self._stream_center = None
        self._instance_counts = {name: 0 for name in self.mesh_buffers}
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

    def render_scene(self, camera: Camera, *, manage_chunks: bool = True):
        if manage_chunks:
            self._update_world(camera.position)
        snapshot = camera.snapshot(aspect=self.render_width / self.render_height)
        self.proxy_fbo.use()
        self.ctx.viewport = (0, 0, self.render_width, self.render_height)
        self.ctx.enable(moderngl.DEPTH_TEST)
        # Culling stays off: the walker can brush through a slab or overhang,
        # and back faces are what makes that readable instead of empty.
        self.ctx.disable(moderngl.CULL_FACE)
        self.proxy_fbo.clear(*self.sky_color, 1.0, depth=1.0)
        self.proxy_program["view"].write(snapshot.view_matrix.T.astype("f4").tobytes())
        self.proxy_program["projection"].write(snapshot.projection_matrix.T.astype("f4").tobytes())
        self.proxy_program["camera_position"].value = tuple(float(value) for value in camera.position)
        self.proxy_program["fog_color"].value = self.sky_color
        for mesh, count in self._instance_counts.items():
            if count:
                self.mesh_vaos[mesh].render(instances=count)
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
            position=(frame.camera_position + delta).astype(np.float64),
            yaw=source_yaw_degrees + yaw_delta,
            pitch=source_pitch + pitch_delta,
            fov=camera.fov,
            near=camera.near,
            far=camera.far,
        )
        if not self._world_contains(warped_camera.position):
            self.reproject_ms += 0.15 * ((perf_counter() - started) * 1000.0 - self.reproject_ms)
            self._present_texture(
                self.display_texture, overlay_lines, sharpen, prompt_caption
            )
            return
        current = self.render_scene(warped_camera, manage_chunks=False)
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
        if self._closed:
            return
        self._closed = True
        pygame.event.set_grab(False)
        pygame.mouse.set_visible(True)
        resources = (
            *self.mesh_vaos.values(),
            *self.instance_buffers.values(),
            *self.mesh_buffers.values(),
            self.quad_vao,
            self.reproject_vao,
            self.quad_buffer,
            self.proxy_fbo,
            self.reproject_fbo,
            self.color_texture,
            self.depth_texture,
            self.display_texture,
            self.reproject_source_depth,
            self.reproject_texture,
            self.reproject_depth,
            self.overlay_texture,
            self.proxy_program,
            self.screen_program,
            self.reproject_program,
        )
        for resource in resources:
            resource.release()
        self.ctx.release()
        pygame.quit()
