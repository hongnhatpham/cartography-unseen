"""Render isolated proxy shader treatments and build a static comparison report."""
from __future__ import annotations

import base64
from collections import defaultdict
import io
from itertools import product
import json
from math import radians, tan
from pathlib import Path
import sys

import moderngl
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from app.renderer.camera import Camera
from app.renderer.form_meshes import form_meshes
from app.renderer.proxy_renderer import ProxyRenderer, _cube_vertices
from app.renderer.world import WorldCube, atmosphere_colors, generate_chunk, world_to_chunk

OUT = ROOT / "logs/reference/surface-study"
HTML = ROOT / "docs/prototypes/broad-surface-variation.html"
SIZE = (768, 512)
NOISE = """
float hash3(vec3 p) {
    p = fract(p * 0.1031);
    p += dot(p, p.yzx + 33.33);
    return fract((p.x + p.y) * p.z);
}
float field(vec3 p) {
    vec3 cell = floor(p), f = fract(p);
    f = f * f * (3.0 - 2.0 * f);
    return mix(mix(mix(hash3(cell), hash3(cell+vec3(1,0,0)), f.x),
                   mix(hash3(cell+vec3(0,1,0)), hash3(cell+vec3(1,1,0)), f.x), f.y),
               mix(mix(hash3(cell+vec3(0,0,1)), hash3(cell+vec3(1,0,1)), f.x),
                   mix(hash3(cell+vec3(0,1,1)), hash3(cell+vec3(1,1,1)), f.x), f.y), f.z);
}
"""
TREATMENTS = {
    "current": "",
    "a": """
    float broad = field(world_position * 0.20);
    float secondary = field(world_position * 0.53 + vec3(17.0));
    float tone = smoothstep(0.20, 0.80, broad * 0.78 + secondary * 0.22);
    albedo *= mix(0.48, 1.20, tone);
    """,
    "b": """
    float warp = field(world_position * 0.13);
    float patches = field(world_position * 0.36 + vec3(warp * 2.4));
    float erosion = smoothstep(0.37, 0.53, patches);
    albedo *= mix(0.36, 1.13, erosion);
    """,
    "c": """
    float warp = field(world_position * 0.16) * 5.0;
    float layers = sin(world_position.y * 1.35 + warp);
    float seams = smoothstep(-0.35, 0.45, layers);
    float broad = field(world_position * 0.28 + vec3(7.0));
    albedo *= mix(0.42, 1.14, seams) * mix(0.80, 1.08, broad);
    """,
}
LABELS = [
    ("current", "Now", "Uniform faces", "Current shader. Color changes between objects; broad faces remain mostly uniform."),
    ("a", "A", "Soft mottling", "Broad clouds of light and dark with a smaller secondary scale. My recommended starting point."),
    ("b", "B", "Erosion patches", "Stronger, uneven boundaries between dark and light areas. More visible, but may read as camouflage."),
    ("c", "C", "Warped strata", "Broad layers bend through the forms. A stronger direction that may push prompts toward rock or sediment."),
]


def render(ctx, groups, camera, atmosphere, key):
    """Use the live mesh and lighting code with only the candidate albedo changed."""
    source = (ROOT / "shaders/proxy.frag").read_text()
    # Preserve the study's flat baseline after A ships in the live shader.
    marker = "    // A: broad soft mottling"
    if marker in source:
        start = source.index(marker)
        end = source.index("    vec3 lit =", start)
        source = source[:start] + source[end:]
    if key != "current":
        source = source.replace("void main() {", NOISE + "\nvoid main() {")
        source = source.replace("vec3 albedo = material_color;", "vec3 albedo = material_color;" + TREATMENTS[key])
    program = ctx.program(vertex_shader=(ROOT / "shaders/proxy.vert").read_text(), fragment_shader=source)
    sky = ctx.program(vertex_shader=(ROOT / "shaders/sky.vert").read_text(), fragment_shader=(ROOT / "shaders/sky.frag").read_text())
    fbo = ctx.simple_framebuffer(SIZE)
    fbo.use()
    ctx.viewport = (0, 0, *SIZE)
    fbo.clear(*atmosphere[0], 1.0, depth=1.0)
    quad = ctx.buffer(np.array([[-1,-1],[1,-1],[1,1],[-1,-1],[1,1],[-1,1]], dtype="f4"))
    sky_vao = ctx.simple_vertex_array(sky, quad, "in_position")
    for target in (sky, program):
        for name, color in zip(("fog_color", "zenith_color", "nadir_color"), atmosphere):
            target[name].value = color
    sky["camera_forward"].value = tuple(camera.forward)
    sky["camera_right"].value = tuple(camera.right)
    sky["camera_up"].value = tuple(np.cross(camera.right, camera.forward))
    sky["tan_half_fov"].value = tan(radians(camera.fov) / 2)
    sky["aspect"].value = SIZE[0] / SIZE[1]
    ctx.disable(moderngl.DEPTH_TEST)
    sky_vao.render()
    ctx.enable(moderngl.DEPTH_TEST)
    ctx.disable(moderngl.CULL_FACE)
    snapshot = camera.snapshot(SIZE[0] / SIZE[1])
    program["view"].write(snapshot.view_matrix.T.astype("f4").tobytes())
    program["projection"].write(snapshot.projection_matrix.T.astype("f4").tobytes())
    program["camera_position"].value = tuple(camera.position)
    program["fog_distance"].value = 150.0
    meshes = {"cube": _cube_vertices(), **form_meshes()}
    for name, objects in groups.items():
        vertices = ctx.buffer(meshes[name].tobytes())
        instances = ctx.buffer(ProxyRenderer._pack_instances(tuple(objects)).tobytes())
        vao = ctx.vertex_array(program, [
            (vertices, "3f 3f", "in_position", "in_normal"),
            (instances, "4f 4f 4f 4f 3f /i", "instance_model_0", "instance_model_1", "instance_model_2", "instance_model_3", "instance_color"),
        ])
        vao.render(instances=len(objects))
        vao.release(); vertices.release(); instances.release()
    result = Image.frombytes("RGB", SIZE, fbo.read(components=3, alignment=1)).transpose(Image.Transpose.FLIP_TOP_BOTTOM)
    for resource in (sky_vao, quad, sky, program, fbo):
        resource.release()
    return result


def image_url(path):
    image = Image.open(path).convert("RGB")
    buffer = io.BytesIO()
    image.save(buffer, "WEBP", quality=83, method=6)
    return "data:image/webp;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    config = json.loads((ROOT / "config.json").read_text())
    pose = json.loads((ROOT / "logs/reference/form-study/poses.json").read_text())[0]
    camera = Camera(position=np.array(pose["position"]), yaw=pose["yaw"], pitch=pose["pitch"])
    center = world_to_chunk(*camera.position)
    groups = defaultdict(list)
    for offset in product(range(-3, 4), repeat=3):
        coord = tuple(a+b for a,b in zip(center, offset))
        chunk = generate_chunk(coord, config["world_seed"])
        groups["cube"].extend(chunk.objects)
        for form in chunk.forms:
            groups[form.mesh].append(form)
    print("World prepared:", sum(map(len, groups.values())), "forms", flush=True)
    close_camera = Camera(position=np.array([0., 1., 14.]), yaw=0., pitch=-3., fov=62.)
    close_groups = {"cube": [
        WorldCube("panel", (-2.5,0.,0.), (3.9,4.2,1.1), (0.,-24.,0.), (.38,.68,.60)),
        WorldCube("panel", (4.,-1.2,-2.), (2.2,2.6,1.9), (0.,12.,0.), (.75,.59,.40)),
    ]}
    ctx = moderngl.create_standalone_context(require=330)
    for key, *_ in LABELS:
        for name, objects, view, atmosphere in (
            ("scene", groups, camera, atmosphere_colors(config["world_seed"], *camera.position)),
            ("close", close_groups, close_camera, ((.16,.22,.25),(.09,.16,.20),(.025,.03,.04))),
        ):
            render(ctx, objects, view, atmosphere, key).save(OUT / f"{key}-{name}.png")
        print("Rendered", key, flush=True)
    ctx.release()
    columns = []
    for key, letter, title, description in LABELS:
        figures = "".join(f'<figure><img src="{image_url(OUT / f"{key}-{kind}.png")}" width="768" height="512" alt="{title}, {caption}"><figcaption>{caption}</figcaption></figure>' for kind, caption in (("close", "Close surface study"), ("scene", "Existing world, fixed viewpoint")))
        columns.append(f'<article><header><span>{letter}</span><h2>{title}</h2><p>{description}</p></header>{figures}</article>')
    doc = '''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Cartography Unseen / Broad surface variation</title><style>
    :root{color-scheme:dark;--bg:#000;--surface:#141414;--line:#343434;--text:#fff;--muted:#b8b8b8;--accent:#a8c5bd}*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font:15px/1.5 "Segoe UI",sans-serif}main{max-width:1800px;margin:auto;padding:28px}h1,h2{font-family:Bahnschrift,"Segoe UI",sans-serif;font-weight:500;line-height:1.2}h1{font-size:30px;margin:6px 0 12px}h2{font-size:21px;margin:5px 0 9px}p{margin:0 0 12px;max-width:100ch}.eyebrow,figcaption{color:var(--muted);font-size:12px}.intro{max-width:105ch;color:#ddd}.compare{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:18px;margin:28px 0}article header{min-height:164px;border-top:1px solid var(--line);padding-top:12px}article header span{font:14px Consolas,monospace;color:var(--accent)}article p{font-size:14px;color:var(--muted)}figure{margin:0 0 18px}img{display:block;width:100%;height:auto;background:var(--surface)}figcaption{margin-top:7px}.notes{display:grid;grid-template-columns:1fr 1fr;gap:28px;border-top:1px solid var(--line);padding-top:20px}.notes p{color:var(--muted);font-size:14px}.notes h2{font-size:19px}.choice{padding:14px 18px;background:var(--surface);margin-bottom:24px}.method{font-size:13px;color:var(--muted);margin-top:18px}@media(max-width:1100px){.compare{grid-template-columns:repeat(2,minmax(0,1fr))}article header{min-height:145px}}@media(max-width:550px){main{padding:18px 12px}.compare{gap:12px}h1{font-size:25px}h2{font-size:18px}article header{min-height:202px}article p{font-size:13px}.notes{grid-template-columns:1fr;gap:10px}}@media(max-width:350px){.compare{grid-template-columns:1fr}article header{min-height:0}}
    </style></head><body><main><header><p class="eyebrow">CARTOGRAPHY UNSEEN / SURFACE STUDY / 5 SEPTEMBER 2026</p><h1>Give the broad faces something to work with</h1><p class="intro">Three procedural treatments for the proxy. Compare each against the current flat material on the same surfaces and in the existing world. These are isolated proxy renders, not AI outputs.</p></header><section class="compare" aria-label="Current shader and three candidate treatments">''' + "".join(columns) + '''</section><p class="choice">My pick is A. It breaks up the large faces while leaving room for different prompts to interpret the material. B is more forceful; C makes the strongest commitment to a layered material.</p><section class="notes"><div><h2>What stays consistent</h2><p>Each row keeps its camera, geometry, object colors, four-step lighting and fog fixed. Only the surface color modulation changes. The lower row uses the existing generated world and meshes.</p><p>The close study uses two large slabs to make the differences readable. The variation changes color, not geometry or surface normals. It does not add relief or cast shadows.</p></div><div><h2>What implementation needs</h2><p>Anchor the pattern to stable world coordinates so it stays attached during flight and coordinate rebasing. Keep the variation broad enough to survive the low-resolution proxy, and check the result through diffusion before tuning its strength.</p><p>These stills establish appearance only. Live GPU cost, distant aliasing, and generated-image behavior remain to be checked after choosing a direction.</p></div></section><p class="method">Rendered offscreen with the current proxy and sky shaders plus isolated candidate material code. No animated noise, image textures or additional geometry. Production shaders have not changed. Choose A, B or C before implementation.</p></main></body></html>'''
    assert len(doc.encode()) <= 512 * 1024
    HTML.write_text(doc, encoding="utf-8")
    sheet = Image.new("RGB", (4*384, 2*256+28), "#000")
    draw = ImageDraw.Draw(sheet)
    for i, (key, letter, *_rest) in enumerate(LABELS):
        draw.text((i*384+8,6), letter, fill="white")
        for j, kind in enumerate(("close", "scene")):
            sheet.paste(Image.open(OUT/f"{key}-{kind}.png").resize((384,256)), (i*384,j*256+28))
    sheet.save(OUT/"comparison.png")
    print("Built", HTML.name, HTML.stat().st_size, "bytes", flush=True)


if __name__ == "__main__":
    main()
