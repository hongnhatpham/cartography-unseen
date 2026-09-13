from __future__ import annotations

from math import cos, radians, sin, tan
from pathlib import Path
from time import perf_counter
import textwrap
import sys

import numpy as np

try:
    import moderngl
    import pygame
except ImportError as exc:  # pragma: no cover - rendered as a startup error by main
    raise RuntimeError(
        "Renderer dependencies are missing. Run setup_dev.ps1 or tools/prepare_runtime.ps1."
    ) from exc

from app.renderer.camera import Camera
from app.config import DEFAULT_FOG_DISTANCE
from app.renderer.form_meshes import form_meshes
from app.renderer.idle_overlay import IdleOverlay
from app.screenshots import ScreenshotWriter
from app.window_loop import WindowLoop
from app.window_placement import WindowPlacement
from app.renderer.player_trail import PlayerTrail
from app.renderer.trail_renderer import TrailRenderer
from app.renderer.world import (
    ACTIVE_CHUNK_RADIUS,
    CHUNK_SIZE,
    MAX_ACTIVE_CHUNKS,
    ChunkCoord,
    WorldCube,
    WorldForm,
    chunk_colliders,
    open_heading,
    resolve_collisions,
    generate_chunk,
    atmosphere_colors,
    plan_chunk_cache,
    SPAWN_PITCHES,
    spawn_pose,
    world_label,
    world_to_chunk,
)
from app.types import CameraSnapshot, ConditioningFrame, GeneratedFrame

# A boundary crossing adds a plane of chunks. Load nearest first, yielding
# after a short construction budget so expensive chunks do not stack up.
CHUNK_LOAD_BUDGET = 3
CHUNK_LOAD_SECONDS = .003


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
        fog_distance: float = DEFAULT_FOG_DISTANCE,
        window_position: tuple[int, int] | None = None,
    ) -> None:
        pygame.font.init()
        self._operator_mode = False
        self._prompt_editing = False
        create_window = lambda: self._create_window(window_size, fullscreen, display_monitor, window_position)
        self._window_loop = WindowLoop(create_window) if sys.platform == "win32" else None
        if self._window_loop is None:
            self._placement = create_window()
        else:
            self._placement = self._window_loop.placement
        window_size = self._window_size()

        self.ctx = None
        try:
            self.ctx = moderngl.create_context(require=330)
            self._initialize_renderer(project_root, resolution, window_size, world_seed, fog_distance)
        except BaseException:
            # main cannot close an object whose constructor did not return.
            # Destroy the context/window even if shader or asset setup failed.
            try:
                if self.ctx is not None:
                    self.ctx.release()
            finally:
                if self._window_loop:
                    self._window_loop.close()
                else:
                    pygame.quit()
            raise

    @staticmethod
    def _create_window(window_size, fullscreen, display_monitor, window_position=None):
        pygame.display.init()
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
        placement = WindowPlacement()
        if window_position is not None:
            placement.window.position = window_position
        if fullscreen:
            placement.set_fullscreen(True)
        pygame.display.set_caption("Cartography Unseen")
        pygame.event.set_grab(True)
        pygame.mouse.set_visible(False)
        pygame.mouse.get_rel()
        # The prompt editor explicitly enables composition when opened.
        pygame.key.stop_text_input()
        return placement

    def _initialize_renderer(self, project_root, resolution, window_size, world_seed, fog_distance):
        self.ctx.enable(moderngl.DEPTH_TEST)
        self.project_root = project_root
        self.fog_distance = fog_distance
        if isinstance(resolution, int):
            resolution = (resolution, resolution)
        self.render_width, self.render_height = resolution
        self.window_size = window_size
        self.font = pygame.font.Font(None, 24)
        self.small_font = pygame.font.Font(None, 20)

        shader_root = project_root / "shaders"
        self.trail_renderer = TrailRenderer(self.ctx, shader_root)
        self.idle_overlay = IdleOverlay(self.ctx, project_root, window_size)
        self.proxy_program = self.ctx.program(
            vertex_shader=(shader_root / "proxy.vert").read_text(encoding="utf-8"),
            fragment_shader=(shader_root / "proxy.frag").read_text(encoding="utf-8"),
        )
        self.screen_program = self.ctx.program(
            vertex_shader=(shader_root / "screen.vert").read_text(encoding="utf-8"),
            fragment_shader=(shader_root / "screen.frag").read_text(encoding="utf-8"),
        )
        self.sky_program = self.ctx.program(
            vertex_shader=(shader_root / "sky.vert").read_text(encoding="utf-8"),
            fragment_shader=(shader_root / "sky.frag").read_text(encoding="utf-8"),
        )
        self.reproject_program = self.ctx.program(
            vertex_shader=(shader_root / "reproject.vert").read_text(encoding="utf-8"),
            fragment_shader=(shader_root / "reproject.frag").read_text(encoding="utf-8"),
        )

        meshes = {"cube": _cube_vertices(), **form_meshes()}
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
        # Observability counter: how many times geometry was uploaded.
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
        # The sky reconstructs its ray from the clip-space corner, so the quad's
        # uv pair is skipped rather than bound to an attribute GL would optimise
        # away.
        self.sky_vao = self.ctx.vertex_array(
            self.sky_program,
            [(self.quad_buffer, "2f 2x4", "in_position")],
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
        self._overlay_surface = pygame.Surface(window_size, pygame.SRCALPHA)
        self._overlay_regions = [None, None]
        self._overlay_text_cache = {}
        self._overlay_texture_initialized = False
        self.loading_image = self._make_loading_image(self.render_width, self.render_height)
        self._display_frame: ConditioningFrame | GeneratedFrame | None = None
        self._reproject_frame: GeneratedFrame | None = None
        self.reproject_ms = 0.0

        self.world_seed = world_seed
        self.sky_color, self.zenith_color, self.nadir_color = atmosphere_colors(world_seed)
        # Packed arrays own the active world; do not also retain source objects.
        self._chunk_instances: dict[ChunkCoord, np.ndarray] = {}
        self._chunk_form_instances: dict[ChunkCoord, dict[str, np.ndarray]] = {}
        self._form_instances: dict[str, np.ndarray] = {}
        self._form_bounds: dict[str, np.ndarray] = {}
        self._form_view_key: tuple | None = None
        self._chunk_colliders: dict[ChunkCoord, np.ndarray] = {}
        self._stream_center: ChunkCoord | None = None
        self._render_origin = np.zeros(3, dtype=np.float64)
        # Where the overlay label is sampled from; the biome field is 3D now.
        self._focus: tuple[float, float, float] = (0.0, 0.0, 0.0)
        self.sequence = 0
        self._closed = False

    def _window_size(self):
        return self._window_loop.size if self._window_loop else pygame.display.get_window_size()

    def _window_call(self, function):
        return self._window_loop.call(function) if self._window_loop else function()

    def set_prompt_editing(self, editing: bool):
        self._prompt_editing = editing
        def change():
            if editing:
                pygame.key.start_text_input()
            else:
                pygame.key.stop_text_input()
            released = editing or self._operator_mode
            pygame.event.set_grab(not released)
            pygame.mouse.set_visible(released)
            pygame.mouse.get_rel()
        self._window_call(change)
        if self._window_loop:
            self._window_loop.read_input()

    def set_operator_mode(self, enabled: bool) -> None:
        """Release the pointer while arranging windows or using the F1 panel."""
        self._operator_mode = bool(enabled)
        self.set_prompt_editing(self._prompt_editing)

    def read_input(self):
        if self._window_loop:
            return self._window_loop.read_input()
        return pygame.mouse.get_rel(), pygame.key.get_pressed(), pygame.mouse.get_pressed()

    def focus(self) -> None:
        """Return input to traversal after closing controls on the map projector."""
        self._window_call(self._placement.window.focus)

    @property
    def is_fullscreen(self) -> bool:
        placement = self._window_loop if self._window_loop else self._placement
        return placement.is_fullscreen

    def set_fullscreen(self, enabled: bool) -> bool:
        """Fill the current monitor, preserving the window and its restore bounds."""
        if self._window_loop:
            is_fullscreen = self._window_loop.set_fullscreen(enabled)
        else:
            is_fullscreen = self._placement.set_fullscreen(enabled)
        self.window_size = self._window_size()
        self._last_overlay_update = 0.0
        return is_fullscreen

    def toggle_fullscreen(self) -> bool:
        return self.set_fullscreen(not self.is_fullscreen)

    def resize_window_to_render(self) -> None:
        """Match a windowed SDL window to the active generation dimensions."""
        def resize():
            if not self._placement.is_fullscreen:
                self._placement.window.size = (self.render_width, self.render_height)
            if self._window_loop:
                self._window_loop.size = pygame.display.get_window_size()
        self._window_call(resize)
        self.window_size = self._window_size()
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
        self._display_frame = None
        self._reproject_frame = None
        self.resize_window_to_render()

    @staticmethod
    def _pack_instances(
        objects: tuple[WorldCube, ...] | tuple[WorldForm, ...], origin: np.ndarray | None = None
    ) -> np.ndarray:
        """Pack column-major transforms and colors for one instanced draw.

        Vectorised over the whole chunk: a per-cube Python loop building 4x4
        matrices costs milliseconds per chunk and shows up as a frame spike.
        """
        count = len(objects)
        packed = np.zeros((count, 19), dtype="f4")
        if count == 0:
            return packed
        position = np.array([item.position for item in objects], dtype=np.float64)
        if origin is not None:
            position -= origin
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

    def update_world(self, position: np.ndarray) -> None:
        """Keep streaming and the location label current even without a draw."""
        self._update_world(position)

    def _update_world(self, position: np.ndarray) -> None:
        """Move the fixed chunk window, loading a bounded number of chunks.

        A cold cache (startup, or a new world seed) is filled in one go because
        a half-built landscape is worse than one hitch. Once the window is
        populated, boundary crossings yield after three milliseconds of work or
        CHUNK_LOAD_BUDGET chunks. The nearest chunks still arrive first.
        """
        self._focus = (float(position[0]), float(position[1]), float(position[2]))
        center = world_to_chunk(*self._focus)
        if center == self._stream_center and len(self._chunk_instances) == MAX_ACTIVE_CHUNKS:
            return
        plan = plan_chunk_cache(self._chunk_instances, *self._focus, ACTIVE_CHUNK_RADIUS)
        rebuild = self._stream_center != plan.center or bool(plan.evict)
        self._stream_center = plan.center
        self._render_origin = np.asarray(plan.center, dtype=np.float64) * CHUNK_SIZE
        if not plan.load and not plan.evict:
            return

        for coord in plan.evict:
            del self._chunk_instances[coord]
            self._chunk_form_instances.pop(coord, None)
            self._chunk_colliders.pop(coord, None)
        cold = not self._chunk_instances
        budget = len(plan.load) if cold else CHUNK_LOAD_BUDGET
        started = perf_counter()
        loaded = []
        for coord in plan.load[:budget]:
            chunk = generate_chunk(coord, self.world_seed)
            self._chunk_instances[coord] = self._pack_instances(
                chunk.objects, origin=np.asarray(coord, dtype=np.float64) * CHUNK_SIZE
            )
            self._chunk_colliders[coord] = chunk_colliders(chunk)
            by_mesh: dict[str, list[WorldForm]] = {}
            for form in chunk.forms:
                by_mesh.setdefault(form.mesh, []).append(form)
            self._chunk_form_instances[coord] = {
                mesh: self._pack_instances(tuple(forms), origin=np.asarray(coord, dtype=np.float64) * CHUNK_SIZE)
                for mesh, forms in by_mesh.items()
            }
            loaded.append(coord)
            if not cold and perf_counter() - started >= CHUNK_LOAD_SECONDS:
                break
        loaded = tuple(loaded)

        cube_buffer = self.instance_buffers["cube"]
        count = sum(len(part) for part in self._chunk_instances.values())
        required_bytes = count * 19 * np.dtype("f4").itemsize
        if required_bytes > cube_buffer.size:
            # Growing a GL buffer discards its contents. Leave room for the rest
            # of this window so its following batches can append in place.
            cube_buffer.orphan(max(required_bytes, cube_buffer.size * 2))
            rebuild = True
        coords = (
            tuple(coord for coord in plan.desired if coord in self._chunk_instances)
            if rebuild else loaded
        )
        if coords:
            parts = [self._chunk_instances[coord] for coord in coords]
            cube_data = np.concatenate(parts, axis=0)
            # Translate all instances together. Integer subtraction before the
            # float conversion preserves detail during long vertical flights.
            offsets = (np.asarray(coords, dtype=np.int64) - plan.center) * CHUNK_SIZE
            cube_data[:, 12:15] += np.repeat(offsets, [len(part) for part in parts], axis=0)
            offset = 0 if rebuild else self._instance_counts["cube"] * 19 * 4
            cube_buffer.write(cube_data, offset=offset)
        self._instance_counts["cube"] = count
        self._update_form_instances(coords, plan.center, rebuild)
        self._instance_buffer_revision += 1

    def _update_form_instances(
        self, coords: tuple[ChunkCoord, ...], center: ChunkCoord, rebuild: bool
    ) -> None:
        """Keep additive transforms in the same local origin as the cube buffer.

        Chunk batches append during streaming; eviction and rebasing rebuild the
        bounded CPU cache. Only the forms visible to a draw are sent to the GPU.
        """
        if rebuild:
            self._form_instances.clear()
            self._form_bounds.clear()
        batches: dict[str, list[np.ndarray]] = {}
        offsets: dict[str, list[tuple[int, int, int]]] = {}
        for coord in coords:
            for mesh, packed in self._chunk_form_instances[coord].items():
                batches.setdefault(mesh, []).append(packed)
                offsets.setdefault(mesh, []).append(tuple(coord[i] - center[i] for i in range(3)))
        for mesh, parts in batches.items():
            packed = np.concatenate(parts, axis=0)
            translation = np.asarray(offsets[mesh], dtype=np.float64) * CHUNK_SIZE
            packed[:, 12:15] += np.repeat(translation, [len(part) for part in parts], axis=0)
            if mesh in self._form_instances:
                packed = np.concatenate((self._form_instances[mesh], packed), axis=0)
            self._form_instances[mesh] = packed
            # Rotation preserves the norm of the scaled unit-box corner.
            linear = packed[:, :12].reshape(-1, 3, 4)[:, :, :3]
            radius = np.sqrt(np.sum(linear * linear, axis=(1, 2)))
            self._form_bounds[mesh] = np.column_stack((packed[:, 12:15], radius))
        self._form_view_key = None

    @staticmethod
    def _visible_forms(bounds: np.ndarray, clip: np.ndarray) -> np.ndarray:
        """Conservative sphere/frustum test, including near-plane crossings."""
        planes = np.asarray([clip[3] + sign * clip[axis] for axis in range(3) for sign in (-1, 1)])
        planes /= np.linalg.norm(planes[:, :3], axis=1)[:, None]
        return ((bounds[:, :3] @ planes[:, :3].T + planes[:, 3]) >= -bounds[:, 3, None]).all(axis=1)

    def _prepare_form_draw(self, local_view: np.ndarray, projection: np.ndarray) -> None:
        """Upload visible forms when the camera or streamed geometry changes."""
        key = (local_view.tobytes(), projection.tobytes())
        if key == self._form_view_key:
            return
        clip = projection.astype(np.float64) @ local_view
        for mesh in self.mesh_vaos:
            if mesh == "cube":
                continue
            packed = self._form_instances.get(mesh)
            count = 0
            if packed is not None:
                selected = packed[self._visible_forms(self._form_bounds[mesh], clip)]
                count = len(selected)
                if count:
                    buffer = self.instance_buffers[mesh]
                    if selected.nbytes > buffer.size:
                        buffer.orphan(max(selected.nbytes, buffer.size * 2))
                    buffer.write(selected)
            self._instance_counts[mesh] = count
        self._form_view_key = key

    def _world_contains(self, position: np.ndarray) -> bool:
        return world_to_chunk(*position) in self._chunk_instances

    def spawn_camera(self, camera: Camera) -> None:
        """Float the flier in an open pocket of this seed's volume.

        The pocket is picked from the field alone; the forms around it are only
        known once the chunk exists, so the flier is then pushed out of anything
        it spawned inside and turned to face the longest clear line in 3D.
        """
        position, yaw, pitch = spawn_pose(self.world_seed)
        camera.position[:] = position
        camera.yaw = yaw
        camera.pitch = pitch
        self._update_world(camera.position)
        self.constrain_camera(camera, None)
        x, y, z = (float(value) for value in camera.position)
        open_yaw, open_pitch, distance = open_heading(
            x, y, z, self._nearby_colliders(x, y, z), pitches=SPAWN_PITCHES
        )
        if distance > 0.0:
            camera.yaw = open_yaw
            camera.pitch = open_pitch

    def nearby_colliders(self, camera: Camera) -> np.ndarray:
        """Form boxes around the camera, for autopilot steering."""
        return self._nearby_colliders(*camera.position)

    def open_heading(self, camera: Camera) -> tuple[float, float, float]:
        """(yaw, pitch) of the longest clear line from the camera, and its length."""
        x, y, z = (float(value) for value in camera.position)
        return open_heading(x, y, z, self._nearby_colliders(x, y, z))

    def _nearby_colliders(self, x: float, y: float, z: float) -> np.ndarray:
        """Boxes from the 3x3x3 chunk neighbourhood, including above and below."""
        home = world_to_chunk(x, y, z)
        nearby = [
            self._chunk_colliders[coord]
            for coord in (
                (home[0] + dx, home[1] + dy, home[2] + dz)
                for dx in (-1, 0, 1) for dy in (-1, 0, 1) for dz in (-1, 0, 1)
            )
            if coord in self._chunk_colliders
        ]
        return np.concatenate(nearby, axis=0) if nearby else np.empty((0, 8))

    def constrain_camera(self, camera: Camera, dt: float | None = None) -> None:
        """Slide out of solid forms without restricting altitude.

        ``dt`` is unused: with no gravity there is nothing to settle toward, but
        the signature stays so the caller's per-frame loop is unchanged.
        """
        if not self._world_contains(camera.position):
            self._update_world(camera.position)
        x, y, z = (float(value) for value in camera.position)
        x, y, z = resolve_collisions(x, y, z, self._nearby_colliders(x, y, z))
        camera.position[0] = x
        camera.position[1] = y
        camera.position[2] = z

    def world_label(self) -> str:
        """Short biome and palette label for the overlay, where the flier is."""
        return world_label(self.world_seed, *self._focus)

    def randomize_world(self, world_seed: int | None = None) -> int:
        """Switch to a new (or given) world seed and drop every cached chunk."""
        self.world_seed = (
            int(np.random.SeedSequence().generate_state(1, dtype=np.uint32)[0])
            if world_seed is None
            else int(world_seed)
        )
        self.sky_color, self.zenith_color, self.nadir_color = atmosphere_colors(self.world_seed)
        self._chunk_instances.clear()
        self._chunk_form_instances.clear()
        self._form_instances.clear()
        self._form_bounds.clear()
        self._form_view_key = None
        self._chunk_colliders.clear()
        self._stream_center = None
        self._instance_counts = {name: 0 for name in self.mesh_buffers}
        self._display_frame = None
        self._reproject_frame = None
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
        self.sky_color, self.zenith_color, self.nadir_color = atmosphere_colors(
            self.world_seed, *camera.position
        )
        self.proxy_fbo.use()
        self.ctx.viewport = (0, 0, self.render_width, self.render_height)
        self.ctx.enable(moderngl.DEPTH_TEST)
        # Culling stays off: the walker can brush through a slab or overhang,
        # and back faces are what makes that readable instead of empty.
        self.ctx.disable(moderngl.CULL_FACE)
        self.proxy_fbo.clear(*self.sky_color, 1.0, depth=1.0)
        self._draw_sky(camera)
        local_view = self._local_view(snapshot.view_matrix)
        self._prepare_form_draw(local_view, snapshot.projection_matrix)
        self.proxy_program["view"].write(local_view.T.astype("f4").tobytes())
        self.proxy_program["projection"].write(snapshot.projection_matrix.T.astype("f4").tobytes())
        self.proxy_program["camera_position"].value = tuple(camera.position - self._render_origin)
        # Keep both noise scales attached to the world through origin rebases,
        # without uploading huge coordinates to the float32 shader. Match the
        # scales and lattice period in proxy.frag.
        self.proxy_program["surface_origin_coarse"].value = tuple(np.remainder(self._render_origin * 0.20, 256.0))
        self.proxy_program["surface_origin_fine"].value = tuple(np.remainder(self._render_origin * 0.53, 256.0))
        self.proxy_program["fog_color"].value = self.sky_color
        self.proxy_program["fog_distance"].value = self.fog_distance
        self.proxy_program["zenith_color"].value = self.zenith_color
        self.proxy_program["nadir_color"].value = self.nadir_color
        for mesh, count in self._instance_counts.items():
            if count:
                self.mesh_vaos[mesh].render(instances=count)
        return snapshot

    def _local_view(self, view: np.ndarray) -> np.ndarray:
        """Express a world camera in the origin used by this GPU instance buffer."""
        result = view.astype(np.float64).copy()
        result[:3, 3] += result[:3, :3] @ self._render_origin
        return result

    def _draw_sky(self, camera: Camera) -> None:
        """Graded background behind everything: sky above, dark below, fog level.

        Drawn at the far depth so geometry always wins. The gradient is a
        function of the world-space view ray, not of screen height, so it stays
        anchored however far the flier pitches up or down.
        """
        self.ctx.disable(moderngl.DEPTH_TEST)
        forward = camera.forward
        right = camera.right
        up = np.cross(right, forward)
        self.sky_program["camera_right"].value = tuple(float(value) for value in right)
        self.sky_program["camera_up"].value = tuple(float(value) for value in up)
        self.sky_program["camera_forward"].value = tuple(float(value) for value in forward)
        self.sky_program["tan_half_fov"].value = float(tan(radians(camera.fov) * 0.5))
        self.sky_program["aspect"].value = self.render_width / self.render_height
        self.sky_program["fog_color"].value = self.sky_color
        self.sky_program["zenith_color"].value = self.zenith_color
        self.sky_program["nadir_color"].value = self.nadir_color
        self.sky_vao.render()
        self.ctx.enable(moderngl.DEPTH_TEST)

    def capture_conditioning(
        self, snapshot, timestamp: float, *, include_edges: bool = True
    ) -> ConditioningFrame:
        rgb = np.frombuffer(self.color_texture.read(alignment=1), dtype=np.uint8)
        rgb = np.flipud(rgb.reshape(self.render_height, self.render_width, 3)).copy()
        depth = np.frombuffer(self.depth_texture.read(alignment=1), dtype=np.float32)
        depth = np.flipud(depth.reshape(self.render_height, self.render_width)).copy()
        edges = self._edges(rgb, depth) if include_edges else None
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
            edges = frame.edges if frame.edges is not None else ProxyRenderer._edges(frame.rgb, frame.depth)
            return np.repeat(edges[:, :, None], 3, axis=2)
        return frame.rgb

    def _upload_display_image(
        self, image: np.ndarray, frame: ConditioningFrame | GeneratedFrame | None = None,
    ) -> None:
        """Retain the published frame, so recycled IDs cannot freeze the image.

        Unversioned or diagnostic arrays still upload every time because callers
        may mutate them. Raw and reprojected views share this texture and cache.
        """
        if frame is not None:
            source = frame.rgb if isinstance(frame, ConditioningFrame) else frame.image
            if image is not source:
                frame = None
        if frame is not None and self._display_frame is frame:
            return
        target_shape = (self.render_height, self.render_width)
        if image.shape[:2] != target_shape:
            surface = pygame.surfarray.make_surface(np.swapaxes(image, 0, 1))
            surface = pygame.transform.smoothscale(surface, (self.render_width, self.render_height))
            image = np.swapaxes(pygame.surfarray.array3d(surface), 0, 1)
        self.display_texture.write(np.ascontiguousarray(np.flipud(image)).tobytes())
        self._display_frame = frame

    def display(
        self,
        image: np.ndarray,
        overlay_lines: list[str] | None = None,
        sharpen: float = 0.3,
        prompt_caption: str | None = None,
        *,
        trail: PlayerTrail | None = None,
        trail_frame: ConditioningFrame | GeneratedFrame | None = None,
        trail_time: float = 0.0,
        idle_opacity: float = 0.0,
        screenshot: ScreenshotWriter | None = None,
    ) -> None:
        self._upload_display_image(image, trail_frame)

        self._present_texture(
            self.display_texture, overlay_lines, sharpen, prompt_caption,
            trail=trail,
            trail_camera=self._frame_camera(trail_frame) if trail_frame is not None else None,
            trail_depth=trail_frame.depth if trail_frame is not None else None,
            trail_time=trail_time,
            trail_until=self._frame_time(trail_frame) if trail_frame is not None else None,
            idle_opacity=idle_opacity,
            screenshot=screenshot,
        )

    @staticmethod
    def _frame_time(frame: ConditioningFrame | GeneratedFrame) -> float:
        return frame.timestamp if isinstance(frame, ConditioningFrame) else frame.conditioning_timestamp

    @staticmethod
    def _frame_camera(frame: ConditioningFrame | GeneratedFrame) -> CameraSnapshot:
        if isinstance(frame, ConditioningFrame):
            return frame.camera
        return CameraSnapshot(
            frame.view_matrix, frame.projection_matrix,
            frame.camera_position, frame.camera_rotation,
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
        *,
        trail: PlayerTrail | None = None,
        trail_time: float = 0.0,
        idle_opacity: float = 0.0,
        screenshot: ScreenshotWriter | None = None,
    ) -> None:
        started = perf_counter()
        self._upload_display_image(frame.image, frame)
        if self._reproject_frame is not frame:
            target_shape = (self.render_height, self.render_width)
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
            self._reproject_frame = frame

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
                self.display_texture, overlay_lines, sharpen, prompt_caption,
                trail=trail, trail_camera=self._frame_camera(frame),
                trail_depth=frame.depth, trail_time=trail_time,
                trail_until=frame.conditioning_timestamp,
                idle_opacity=idle_opacity,
                screenshot=screenshot,
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

        source_vp = frame.projection_matrix @ self._local_view(frame.view_matrix)
        current_vp = current.projection_matrix @ self._local_view(current.view_matrix)
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
            self.reproject_texture, overlay_lines, sharpen, prompt_caption,
            trail=trail, trail_camera=current, trail_depth=self.depth_texture,
            trail_time=trail_time,
            trail_until=frame.conditioning_timestamp,
            idle_opacity=idle_opacity,
            screenshot=screenshot,
        )

    def _present_texture(
        self,
        texture: moderngl.Texture,
        overlay_lines: list[str] | None = None,
        sharpen: float = 0.3,
        prompt_caption: str | None = None,
        *,
        trail: PlayerTrail | None = None,
        trail_camera: CameraSnapshot | None = None,
        trail_depth: np.ndarray | moderngl.Texture | None = None,
        trail_time: float = 0.0,
        trail_until: float | None = None,
        idle_opacity: float = 0.0,
        screenshot: ScreenshotWriter | None = None,
    ) -> None:
        current_size = self._window_size()
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

        if screenshot is not None:
            # ModernGL 5.12 selects GL_COLOR_ATTACHMENT0 even when reading the
            # default backbuffer, which raises GL_INVALID_OPERATION. Draw the
            # same crop/sharpening into a real color attachment for capture.
            capture_texture = self.ctx.texture(self.window_size, 3)
            capture = self.ctx.framebuffer(color_attachments=[capture_texture])
            try:
                capture.use()
                self.quad_vao.render()
                screenshot.submit(capture.read(components=3, alignment=1), self.window_size)
            finally:
                self.ctx.screen.use()
                self.ctx.viewport = (0, 0, *self.window_size)
                capture.release()
                capture_texture.release()

        if trail is not None and trail_camera is not None and trail_depth is not None:
            self.trail_renderer.draw(
                trail, trail_time, trail_camera, trail_depth,
                self.window_size, uv_scale, uv_offset,
                trail_until,
            )
        if overlay_lines or prompt_caption:
            self._draw_overlay(overlay_lines or [], prompt_caption)
        caption_height = 18 + 23 * len(self._caption_lines(prompt_caption)) if prompt_caption else 0
        self.idle_overlay.draw(self.window_size, idle_opacity, caption_height)
        pygame.display.flip()

    def _caption_lines(self, prompt: str) -> list[str]:
        characters = max(24, max(24, self.window_size[0] - 48) // 10)
        return textwrap.wrap(f"PROMPT  {prompt}", width=characters)[:3]

    def _draw_overlay(self, lines: list[str], prompt_caption: str | None = None) -> None:
        now = perf_counter()
        key = "\n".join(lines) + f"\nPROMPT_CAPTION:{prompt_caption or ''}"
        if self.window_size != self._overlay_size:
            self.overlay_texture.release()
            self.overlay_texture = self.ctx.texture(self.window_size, 4, dtype="f1")
            self.overlay_texture.filter = (moderngl.LINEAR, moderngl.LINEAR)
            self._overlay_size = self.window_size
            self._last_overlay_update = 0.0
            self._last_overlay_key = ""
            self._overlay_surface = pygame.Surface(self.window_size, pygame.SRCALPHA)
            self._overlay_regions = [None, None]
            self._overlay_texture_initialized = False
        if key != self._last_overlay_key and now - self._last_overlay_update >= 0.1:
            surface = self._overlay_surface
            previous_regions = self._overlay_regions
            for region in previous_regions:
                if region is not None:
                    surface.fill((0, 0, 0, 0), region)
            regions = [None, None]
            text_cache = {}

            def text_surface(line, caption=False):
                text_key = (line, caption)
                rendered = self._overlay_text_cache.get(text_key)
                if rendered is None:
                    color = (240, 245, 248) if caption else (230, 240, 245)
                    rendered = self.small_font.render(line, True, color)
                    if caption:
                        rendered.set_alpha(77)
                text_cache[text_key] = rendered
                return rendered

            if lines:
                panel_height = min(self.window_size[1] - 12, 18 + len(lines) * 23)
                longest = max((len(line) for line in lines), default=0)
                panel_width = min(
                    self.window_size[0] - 24,
                    max(490, min(840, 42 + longest * 9)),
                )
                region = pygame.draw.rect(
                    surface,
                    (4, 8, 12, 51),
                    (12, 12, panel_width, panel_height),
                    border_radius=7,
                )
                y = 21
                for line in lines:
                    written = surface.blit(text_surface(line), (24, y))
                    if written.width and written.height:
                        region = region.union(written)
                    y += 23
                regions[0] = region
            if prompt_caption:
                caption_lines = self._caption_lines(prompt_caption)
                caption_height = 18 + len(caption_lines) * 23
                caption_y = self.window_size[1] - caption_height - 12
                region = pygame.draw.rect(
                    surface,
                    (0, 0, 0, 77),
                    (12, caption_y, self.window_size[0] - 24, caption_height),
                    border_radius=7,
                )
                text_y = caption_y + 9
                for line in caption_lines:
                    written = surface.blit(text_surface(line, caption=True), (24, text_y))
                    if written.width and written.height:
                        region = region.union(written)
                    text_y += 23
                regions[1] = region
            if not self._overlay_texture_initialized:
                # Initialize transparent pixels outside the panels once per size.
                self.overlay_texture.write(pygame.image.tostring(surface, "RGBA", True))
                self._overlay_texture_initialized = True
            else:
                for previous, current in zip(previous_regions, regions):
                    dirty = previous or current
                    if previous is not None and current is not None:
                        dirty = previous.union(current)
                    if dirty is not None and dirty.width and dirty.height:
                        data = pygame.image.tostring(surface.subsurface(dirty), "RGBA", True)
                        self.overlay_texture.write(data, viewport=(dirty.x, self.window_size[1] - dirty.bottom,
                                                                   dirty.width, dirty.height))
            self._overlay_regions = regions
            # Keep only the current panel/caption text; changing statistics never
            # grow a session-long cache of old values.
            self._overlay_text_cache = text_cache
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
        if WindowLoop.active:
            return WindowLoop.active.poll_events()
        # Unlike get()/pump(), wait() releases the GIL during Windows event
        # processing, so a native input wait cannot also starve AI submissions.
        first = pygame.event.wait(1)
        events = pygame.event.get(pump=False)
        if first.type != pygame.NOEVENT:
            events.insert(0, first)
        return events

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self.trail_renderer.close()
        self.idle_overlay.close()
        resources = (
            *self.mesh_vaos.values(),
            *self.instance_buffers.values(),
            *self.mesh_buffers.values(),
            self.quad_vao,
            self.reproject_vao,
            self.sky_vao,
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
            self.sky_program,
        )
        for resource in resources:
            resource.release()
        self.ctx.release()
        if self._window_loop:
            self._window_loop.close()
        else:
            pygame.quit()
