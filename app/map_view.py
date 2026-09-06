"""An independent, capped-rate exhibition map. The first-person loop never waits on it."""
from __future__ import annotations

import math
import multiprocessing as mp
from pathlib import Path
from queue import Empty, Full
import time

from app.map_idle import MapFade, title_surface
from app.journey_updates import _merge_snapshot, _snapshot_delta


class MapWindow:
    """Send small live poses separately from infrequent immutable recorder snapshots."""

    def __init__(self, root: Path):
        context = mp.get_context("spawn")
        self._poses = context.Queue(maxsize=2)
        self._maps = context.Queue(maxsize=1)
        self._notices = context.Queue(maxsize=16)
        self._stop = context.Event()
        self._fullscreen = False
        self._operator = False
        self._snapshot = None
        self._reported_exit = False
        self._process = context.Process(target=_run, args=(str(root), self._poses,
            self._maps, self._notices, self._stop), name="journey-map", daemon=True)
        self._process.start()

    def update(self, snapshot: dict, position: list, rotation: list, active: bool):
        if self._stop.is_set():
            return
        if snapshot is not self._snapshot:
            try:
                self._maps.put_nowait(_snapshot_delta(self._snapshot, snapshot))
                self._snapshot = snapshot
            except Full:
                pass
        try:
            self._poses.put_nowait({"position": list(position), "rotation": list(rotation),
                "active": bool(active), "fullscreen": self._fullscreen,
                "operator": self._operator})
        except Full:
            pass

    def set_fullscreen(self, enabled: bool):
        self._fullscreen = bool(enabled)

    def set_operator_mode(self, enabled: bool):
        self._operator = bool(enabled)

    def poll(self) -> list:
        notices = []
        while True:
            try:
                notices.append(self._notices.get_nowait())
            except Empty:
                break
        if not self._process.is_alive() and not self._reported_exit:
            self._reported_exit = True
            notices.append("Map window closed. Journey recording continues.")
        return notices

    def close(self):
        self._stop.set()
        self._process.join(timeout=2)
        if self._process.is_alive():
            self._process.terminate()
            self._process.join(timeout=1)
        for queue in (self._poses, self._maps, self._notices):
            queue.cancel_join_thread()
            queue.close()


def image_corners(image: dict):
    """Plane centered at the capture pose, matching Camera's pitch/yaw convention."""
    import numpy as np
    pitch, yaw, roll = map(math.radians, image.get("rotation", [0, 0, 0]))
    right = np.array([math.cos(yaw), 0, math.sin(yaw)])
    forward = np.array([math.sin(yaw)*math.cos(pitch), math.sin(pitch),
                        -math.cos(yaw)*math.cos(pitch)])
    up = np.cross(right, forward)
    right, up = right*math.cos(roll)+up*math.sin(roll), up*math.cos(roll)-right*math.sin(roll)
    half = float(image.get("plane_width", 8)) / 2
    height = half * float(image.get("height", 1)) / max(float(image.get("width", 1)), 1)
    center = np.asarray(image["position"], dtype=float)
    return np.array([center-right*half-up*height, center+right*half-up*height,
                     center+right*half+up*height, center-right*half+up*height])


def representative_images(images, position, limit=512, nearby=256):
    """Keep nearby detail and evenly spaced older views with a fixed GPU budget."""
    if len(images) <= limit:
        return list(images)
    nearest = sorted(images, key=lambda item: sum((a-b)**2 for a, b in
                     zip(item["position"], position)))[:nearby]
    seen = {str(item["id"]) for item in nearest}
    remainder = [item for item in images if str(item["id"]) not in seen]
    count = min(limit-len(nearest), len(remainder))
    return nearest+[remainder[min(len(remainder)-1, int(i*len(remainder)/count))]
                    for i in range(count)]


class _Scene:
    def __init__(self, context, root=None):
        import moderngl
        self.gl = context
        self.root = Path(root) if root else Path(__file__).resolve().parents[1]
        self.title_texture = None
        self.title_size = None
        self.title_quad = None
        self.shader = context.program(vertex_shader='''#version 330
            uniform mat4 mvp; in vec3 position; in vec2 uv; out vec2 texcoord;
            void main(){gl_Position=mvp*vec4(position,1); texcoord=uv;}''',
            fragment_shader='''#version 330
            uniform sampler2D picture; uniform bool textured; uniform vec4 tint;
            in vec2 texcoord; out vec4 color;
            void main(){color=textured ? texture(picture,texcoord)*tint : tint;}''')
        self.shader["picture"] = 0
        self.snapshot = {}
        self.anchor = None
        self.lines = []
        self.outlines = None
        self.planes = []
        self._journey_id = None
        self._image_count = 0
        self._image_corners = {}
        self._plane_ids = set()
        self.textures = {}
        self.retry = {}
        self.caption = None
        self.caption_key = None
        self.caption_vao = None
        self.bounds = None
        self.prompt_key = None
        self.caption_prompt = None
        self.prompt_until = 0
        self.nearest = []
        self.nearest_at = 0
        self.player_ring = None
        self.player_ring_size = None
        self.gl.enable(moderngl.BLEND | moderngl.DEPTH_TEST)
        self.gl.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA

    def _geometry(self, points, uvs=None):
        import numpy as np
        if uvs is None:
            uvs = np.zeros((len(points), 2))
        data = np.column_stack((points, uvs)).astype("f4")
        buffer = self.gl.buffer(data.tobytes())
        return buffer, self.gl.vertex_array(self.shader, [(buffer, "3f 2f", "position", "uv")])

    def update(self, snapshot):
        import numpy as np
        reset = snapshot.get("id") != self._journey_id
        if reset:
            self._journey_id = snapshot.get("id")
            for texture in self.textures.values():
                texture.release()
            self.textures.clear()
            self.retry.clear()
            self.anchor = None
            for buffer, vao, _ in self.planes:
                vao.release()
                buffer.release()
            self.planes = []
            self._image_count = 0
            self._image_corners.clear()
            self._plane_ids.clear()
            self.nearest = []
            self.nearest_at = 0
        for buffer, vao, *_ in self.lines:
            vao.release()
            buffer.release()
        self.lines = []
        self.snapshot = snapshot
        self.caption_prompt = next((event for event in reversed(snapshot.get("prompts", []))
                                    if event.get("prompt") != event.get("previous_prompt")), None)
        key = (snapshot.get("id"), self.caption_prompt.get("id")) if self.caption_prompt else None
        if key != self.prompt_key:
            self.prompt_key = key
            self.prompt_until = time.monotonic()+14
        if self.anchor is None:
            pose = snapshot.get("current_pose") or {}
            self.anchor = np.asarray(pose.get("position", [0, 0, 0]), dtype=float)
        for segment in snapshot.get("segments", []):
            points = [p["position"] for p in segment.get("points", [])]
            if len(points) > 1:
                self.lines.append((*self._geometry(np.asarray(points)-self.anchor), len(points)))
        positions = [p["position"] for s in snapshot.get("segments", []) for p in s.get("points", [])]
        if positions:
            all_points = np.asarray(positions)
            self.bounds = (all_points.min(axis=0), all_points.max(axis=0))
        else:
            self.bounds = None
        all_images = snapshot.get("images", [])
        # Image descriptors are immutable and append-only within a journey.
        # Keep independent counts: the receiver merges deltas into this snapshot.
        added_images = len(all_images) != self._image_count
        for item in all_images[self._image_count:]:
            self._image_corners[str(item["id"])] = image_corners(item)-self.anchor
        self._image_count = len(all_images)
        selected = representative_images(all_images,
            (snapshot.get("current_pose") or {}).get("position", self.anchor))
        selected_ids = {str(item["id"]) for item in selected}
        selection_changed = selected_ids != self._plane_ids
        planes = {str(plane[2]["id"]): plane for plane in self.planes}
        for key in planes.keys()-selected_ids:
            buffer, vao, _ = planes.pop(key)
            vao.release()
            buffer.release()
        self.planes = []
        for item in all_images:
            key = str(item["id"])
            if key in selected_ids:
                plane = planes.get(key)
                if plane is None:
                    plane = (*self._geometry(self._image_corners[key],
                              [[0,0],[1,0],[1,1],[0,1]]), item)
                self.planes.append(plane)
        if reset or added_images or selection_changed:
            if self.outlines:
                for resource in self.outlines:
                    resource.release()
                self.outlines = None
            outlines = [self._image_corners[str(item["id"])] for item in all_images
                        if str(item["id"]) not in selected_ids]
            if outlines:
                vertices = np.asarray(outlines)[:, [0,1,1,2,2,3,3,0]].reshape(-1, 3)
                self.outlines = self._geometry(vertices)
        if selection_changed:
            self.nearest_at = 0
        self._plane_ids = selected_ids

    def _load_textures(self, position):
        from PIL import Image
        import numpy as np
        # A 256-image, 512-pixel GPU working set. Distant frames retain outlines.
        if time.monotonic() >= self.nearest_at:
            selected = representative_images([plane[2] for plane in self.planes],
                                               position, limit=256, nearby=192)
            selected_ids = {str(item["id"]) for item in selected}
            self.nearest = [plane for plane in self.planes if str(plane[2]["id"]) in selected_ids]
            self.nearest_at = time.monotonic()+1
        nearest = self.nearest
        wanted = {str(p[2]["id"]) for p in nearest}
        for key in set(self.textures)-wanted:
            self.textures.pop(key).release()
        loaded = 0
        for _, _, item in nearest:
            key = str(item["id"])
            if key in self.textures or self.retry.get(key, 0) > time.monotonic():
                continue
            try:
                path = Path(self.snapshot.get("archive_dir", ".")) / item["path"]
                with Image.open(path) as source:
                    source.thumbnail((512, 512))
                    rgb = source.convert("RGB").transpose(Image.Transpose.FLIP_TOP_BOTTOM)
                    texture = self.gl.texture(rgb.size, 3, rgb.tobytes())
                    texture.build_mipmaps()
                    self.textures[key] = texture
            except (OSError, ValueError):
                self.retry[key] = time.monotonic()+1
            loaded += 1
            if loaded >= 2:
                break

    def _draw_caption(self, text, size):
        import pygame
        import moderngl
        import numpy as np
        key = (text, size)
        if key != self.caption_key:
            self.caption_key = key
            if self.caption:
                self.caption.release()
                for resource in self.caption_vao:
                    resource.release()
            font = pygame.font.SysFont("Segoe UI", max(18, min(30, size[1]//34)))
            max_width = max(240, size[0]-96)
            lines = []
            for paragraph in text.split("\n"):
                line = ""
                for word in paragraph.split():
                    candidate = (line+" "+word).strip()
                    if line and font.size(candidate)[0] > max_width:
                        lines.append(line)
                        line = word
                    else:
                        line = candidate
                lines.append(line)
            # Full prompt stays in the archive; the live caption can use most of the screen.
            width = min(max_width, max((font.size(line)[0] for line in lines), default=1))+24
            height = min(size[1]-48, len(lines)*font.get_linesize()+24)
            surface = pygame.Surface((width, height), pygame.SRCALPHA)
            surface.fill((0, 0, 0, 205))
            for index, line in enumerate(lines):
                surface.blit(font.render(line, True, (224, 225, 224)), (12, 12+index*font.get_linesize()))
            self.caption = self.gl.texture(surface.get_size(), 4, pygame.image.tobytes(surface, "RGBA", True))
            x0, x1 = -1+48/size[0], -1+(48+2*width)/size[0]
            y0, y1 = 1-(48+2*height)/size[1], 1-48/size[1]
            self.caption_vao = self._geometry([[x0,y0,0],[x1,y0,0],[x1,y1,0],[x0,y1,0]],
                                              [[0,0],[1,0],[1,1],[0,1]])
        self.gl.disable(moderngl.DEPTH_TEST)
        self.shader["mvp"].write(np.eye(4, dtype="f4").tobytes())
        self.shader["textured"] = True
        self.shader["tint"] = (1,1,1,1)
        self.caption.use(0)
        self.caption_vao[1].render(moderngl.TRIANGLE_FAN)
        self.gl.enable(moderngl.DEPTH_TEST)

    def _draw_title(self, size, opacity):
        import moderngl
        import numpy as np
        import pygame
        if self.title_size != size:
            if self.title_texture:
                self.title_texture.release()
            surface = title_surface(self.root, size)
            self.title_texture = self.gl.texture(surface.get_size(), 3,
                pygame.image.tobytes(surface, "RGB", True))
            self.title_texture.filter = (moderngl.LINEAR, moderngl.LINEAR)
            self.title_size = size
        if self.title_quad is None:
            self.title_quad = self._geometry([[-1,-1,0],[1,-1,0],[1,1,0],[-1,1,0]],
                                              [[0,0],[1,0],[1,1],[0,1]])
        self.gl.disable(moderngl.DEPTH_TEST)
        self.shader["mvp"].write(np.eye(4, dtype="f4").tobytes())
        self.shader["textured"] = True
        self.shader["tint"] = (1,1,1,opacity)
        self.title_texture.use(0)
        self.title_quad[1].render(moderngl.TRIANGLE_FAN)
        self.gl.enable(moderngl.DEPTH_TEST)

    def draw(self, pose, angle, size, operator, map_opacity=None):
        import numpy as np
        import moderngl
        from app.renderer.camera import look_at, perspective
        self.gl.viewport = (0, 0, *size)
        self.gl.clear(0, 0, 0)
        if map_opacity is None:
            map_opacity = float(pose["active"])
        if map_opacity > 0:
            self._load_textures(np.asarray(pose["position"]))
            target = np.asarray(pose["position"])-(self.anchor if self.anchor is not None else 0)
            radius = 48.0
            # Grow the view gently with the route; always orbit the actual live player.
            if self.bounds is not None:
                extent = np.maximum(np.abs(self.bounds[0]-pose["position"]),
                                    np.abs(self.bounds[1]-pose["position"]))
                radius = max(radius, float(np.linalg.norm(extent))*0.8)
            eye = target+np.array([math.cos(angle)*radius, radius*0.95, math.sin(angle)*radius])
            mvp = perspective(48, size[0]/max(size[1],1), 0.1, radius*8) @ look_at(eye, target, np.array([0.,1.,0.]))
            self.shader["mvp"].write(mvp.T.astype("f4").tobytes())
            self.shader["textured"] = False
            self.shader["tint"] = (0.57,0.6,0.59,0.8)
            for _, vao, count in self.lines:
                vao.render(moderngl.LINE_STRIP, vertices=count)
            if self.outlines:
                self.shader["tint"] = (0.25,0.29,0.27,0.5)
                self.outlines[1].render(moderngl.LINES)
            for _, vao, item in self.planes:
                texture = self.textures.get(str(item["id"]))
                self.shader["textured"] = texture is not None
                self.shader["tint"] = (1,1,1,1) if texture else (0.32,0.35,0.34,0.5)
                if texture:
                    texture.use(0)
                vao.render(moderngl.TRIANGLE_FAN if texture else moderngl.LINE_LOOP)
            self.shader["textured"] = False
            self.shader["tint"] = (0.95,0.94,0.86,1)
            self.gl.point_size = 7
            points = [target]
            points.extend(np.asarray(p["position"])-self.anchor for p in self.snapshot.get("prompts", []) if p.get("active"))
            buffer, vao = self._geometry(points)
            self.gl.disable(moderngl.DEPTH_TEST)
            vao.render(moderngl.POINTS)
            self.gl.enable(moderngl.DEPTH_TEST)
            vao.release()
            buffer.release()
            # The orbit target stays at screen center. Distinguish the live
            # player from historical prompt markers with a fixed-size ring.
            if self.player_ring_size != size:
                if self.player_ring:
                    for resource in self.player_ring:
                        resource.release()
                self.player_ring = self._geometry([
                    [26 * math.cos(a) / size[0], 26 * math.sin(a) / size[1], 0]
                    for a in np.linspace(0, math.tau, 32, endpoint=False)
                ])
                self.player_ring_size = size
            self.gl.disable(moderngl.DEPTH_TEST)
            self.shader["mvp"].write(np.eye(4, dtype="f4").tobytes())
            self.shader["tint"] = (1, 1, 1, 1)
            self.player_ring[1].render(moderngl.LINE_LOOP)
            self.gl.enable(moderngl.DEPTH_TEST)
        caption = ""
        if operator:
            caption = "JOURNEY MAP · Drag this window to its projector\nF · Fullscreen both displays     F1 · Close controls"
            if not pose["active"]:
                caption += "\nWaiting for a visitor. Automatic flight is not recorded."
        elif map_opacity > 0:
            if self.caption_prompt and time.monotonic() < self.prompt_until:
                latest = self.caption_prompt
                elapsed = latest.get("elapsed", latest.get("timestamp", 0)-
                                     (self.snapshot.get("started_monotonic") or 0))
                caption = f"{latest.get('id', 'PROMPT')} · {elapsed:.1f}s\n{latest.get('prompt', '')}"
        # Prompt captions fade with the map; operator controls stay above both views.
        if caption and not operator:
            self._draw_caption(caption, size)
        if map_opacity < 1:
            self._draw_title(size, 1-map_opacity)
        if caption and operator:
            self._draw_caption(caption, size)


def _run(root, poses, maps, notices, stop):
    """The child owns SDL and GL. Failure only closes this process."""
    def notify(value):
        try:
            notices.put_nowait(value)
        except Full:
            pass
    try:
        import pygame
        import moderngl
        from app.window_placement import WindowPlacement
        pygame.display.init()
        pygame.font.init()
        pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MAJOR_VERSION, 3)
        pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MINOR_VERSION, 3)
        pygame.display.gl_set_attribute(pygame.GL_CONTEXT_PROFILE_MASK, pygame.GL_CONTEXT_PROFILE_CORE)
        desktop = pygame.display.get_desktop_sizes()[0]
        size = (min(1100, max(320, desktop[0]-180)), min(720, max(240, desktop[1]-220)))
        pygame.display.set_mode(size, pygame.OPENGL | pygame.DOUBLEBUF | pygame.RESIZABLE, vsync=0)
        pygame.display.set_caption("Cartography · Journey map")
        placement = WindowPlacement()
        placement.window.position = (140, 140)
        scene = _Scene(moderngl.create_context(), root)
        fade = MapFade()
        pose = {"position": [0,0,0], "rotation": [0,0,0], "active": False,
                "fullscreen": False, "operator": False}
        clock = pygame.time.Clock()
        dirty, angle = True, 0.0
        received_snapshot = None
        snapshot_dirty = False
        while not stop.is_set():
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    stop.set()
                elif event.type in (pygame.WINDOWEXPOSED, pygame.WINDOWSIZECHANGED):
                    dirty = True
                elif event.type == pygame.KEYDOWN and not getattr(event, "repeat", False):
                    if event.key == pygame.K_F1:
                        notify("__toggle_overlay__")
                    elif event.key == pygame.K_f and pose["operator"]:
                        notify("__toggle_fullscreen__")
            while True:
                try:
                    newest = poses.get_nowait()
                    dirty |= newest["active"] != pose["active"] or newest["operator"] != pose["operator"]
                    pose = newest
                except Empty:
                    break
            try:
                received_snapshot = _merge_snapshot(received_snapshot, maps.get_nowait())
                snapshot_dirty = True
            except Empty:
                pass
            if pose["active"] and snapshot_dirty:
                scene.update(received_snapshot)
                snapshot_dirty = False
                dirty = True
            if placement.is_fullscreen != pose["fullscreen"]:
                placement.set_fullscreen(pose["fullscreen"])
                dirty = True
            now = time.monotonic()
            fade.set_active(pose["active"], now)
            transitioning = fade.running(now)
            elapsed = clock.tick(30 if pose["active"] or transitioning else 10)/1000
            map_opacity = fade.value(time.monotonic())
            if pose["active"] or transitioning:
                angle += min(elapsed, 0.1)*math.tau/180
            if pose["active"] or transitioning or dirty:
                size = pygame.display.get_window_size()
                if min(size) > 0:
                    scene.draw(pose, angle, size, pose["operator"], map_opacity)
                    pygame.display.flip()
                # One final draw is required after the last intermediate fade frame.
                dirty = transitioning
        pygame.quit()
    except Exception as error:
        notify(f"Map window failed: {type(error).__name__}: {error}. Journey recording continues.")
