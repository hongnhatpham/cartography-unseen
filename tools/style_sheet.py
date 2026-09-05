"""Offline style sheet for the latent-walk look on the new procedural world.

Grid mode renders one aerial proxy view per world seed and pushes it through the
configured backend for every prompt x timestep combination, so biome variety and
timestep behaviour can be judged side by side. Walk mode flies the camera and
keeps the real backend state (x0 memory, noise walk, prompt walk) so temporal
continuity is visible.

Family mode walks every style family through one world and lays them out one
row per family, which is how the library's range is judged. Space-reset mode
simulates consecutive Space presses (new world, new family, new noise seed),
one row each, which is how "a reset changes everything" is judged.

Usage:
    runtime/python/python.exe tools/style_sheet.py               # grid + walk + html
    runtime/python/python.exe tools/style_sheet.py --grid        # grid only
    runtime/python/python.exe tools/style_sheet.py --walk        # walk only
    runtime/python/python.exe tools/style_sheet.py --families    # look_families.jpg
    runtime/python/python.exe tools/style_sheet.py --space-resets 8  # space_resets.jpg
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import textwrap
from dataclasses import fields
from pathlib import Path
from time import perf_counter

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.config import AppConfig, configure_local_environment
from app.diffusion.factory import create_backend
from app.main import (
    RECENT_FAMILY_MEMORY,
    PromptEntry,
    apply_master_prefix,
    choose_family_prompt,
    compose_prompt,
    hue_words,
    load_master_prefix,
    load_prompt_library,
    world_hues,
)
from app.renderer.camera import Camera
from app.renderer.proxy_renderer import ProxyRenderer

FONT = ImageFont.load_default()
TILE_WIDTH = 200
CAPTION_HEIGHT = 34


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "logs" / "reference" / "v4")
    parser.add_argument("--seeds", type=int, default=6, help="world seeds in the grid")
    parser.add_argument("--prompts", type=int, default=4, help="library prompts in the grid")
    parser.add_argument("--timesteps", nargs="+", type=int, default=[700, 800, 900])
    parser.add_argument("--resolution", help="WIDTHxHEIGHT, defaults to config.json")
    parser.add_argument("--walk-frames", type=int, default=24)
    parser.add_argument("--walk-speed", type=float, default=6.0, help="world units per frame")
    parser.add_argument("--walk-clock-fps", type=float, default=10.0,
                        help="synthetic frame clock so a walk is reproducible; 0 uses the wall clock")
    parser.add_argument("--strip-stride", type=int, default=4,
                        help="every Nth frame in walk_every<N>.jpg; use 12 for a 300-frame walk")
    parser.add_argument("--seed", type=int, help="world seed override for the walk")
    parser.add_argument("--prefix", help="master prefix override ('' or 'none' drops it)")
    parser.add_argument("--negative", help="negative prompt override")
    parser.add_argument("--prompt-text", help="prompt body override (before the master prefix)")
    parser.add_argument("--prompt-index", type=int, default=0, help="library entry used by the walk")
    parser.add_argument("--family", help="style family name used by the walk")
    parser.add_argument("--family-frames", type=int, default=48,
                        help="frames walked per family in --families mode")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                        help="override a config key (repeatable), e.g. --set guide_strength=0.5")
    parser.add_argument("--grid", action="store_true", help="grid only")
    parser.add_argument("--walk", action="store_true", help="walk only")
    parser.add_argument("--families", action="store_true",
                        help="one labelled row per style family into look_families.jpg")
    parser.add_argument("--space-resets", type=int, metavar="N",
                        help="N simulated Space presses, one row each, into space_resets.jpg")
    parser.add_argument("--reset-frames", type=int, default=64,
                        help="frames walked after each simulated Space press")
    return parser.parse_args()


class FrameClock:
    """Deterministic clock advancing a fixed step per rendered frame.

    The backend's walks and instability oscillators run on wall time, so an
    offline experiment would otherwise depend on GPU speed. Driving them from
    the frame counter makes two runs of the same settings comparable.
    """

    def __init__(self, fps: float) -> None:
        self.step = 1.0 / max(fps, 1e-6)
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def tick(self) -> None:
        self.now += self.step


def apply_overrides(config: AppConfig, assignments: list[str]) -> dict[str, object]:
    """Apply ``KEY=VALUE`` overrides to the config, coerced to each field's type."""
    types = {field.name: field.type for field in fields(config)}
    applied: dict[str, object] = {}
    for assignment in assignments:
        key, _, raw = assignment.partition("=")
        key = key.strip()
        if key not in types:
            raise SystemExit(f"Unknown config key: {key}")
        declared = str(types[key])
        if "bool" in declared:
            value: object = raw.strip().lower() in ("1", "true", "yes", "on")
        elif "int" in declared:
            value = int(raw)
        elif "float" in declared:
            value = float(raw)
        else:
            value = raw
        setattr(config, key, value)
        applied[key] = value
    config.validate()
    return applied


def sanitize(text: str) -> str:
    return "".join(char if char.isalnum() else "_" for char in text).strip("_")


def grid_seeds(base_seed: int, count: int) -> list[int]:
    """Deterministic spread of world seeds so the sheet is reproducible."""
    generator = np.random.default_rng(base_seed)
    return [int(value) for value in generator.integers(0, 2**31 - 1, size=count)]


def caption_tile(image: Image.Image, lines: list[str]) -> Image.Image:
    aspect = image.height / image.width
    thumb = image.resize((TILE_WIDTH, max(1, round(TILE_WIDTH * aspect))))
    canvas = Image.new("RGB", (TILE_WIDTH, thumb.height + CAPTION_HEIGHT), (14, 14, 16))
    canvas.paste(thumb, (0, 0))
    draw = ImageDraw.Draw(canvas)
    y = thumb.height + 3
    for line in lines:
        for wrapped in textwrap.wrap(line, width=32)[:1]:
            draw.text((4, y), wrapped, fill=(225, 225, 230), font=FONT)
            y += 11
    return canvas


def compose_sheet(
    rows: list[tuple[str, list[Image.Image]]], title: str, out_path: Path
) -> None:
    """Label column on the left, one row of captioned tiles per entry."""
    if not rows:
        return
    label_width = 150
    cell_w = max(cell.width for _, cells in rows for cell in cells)
    cell_h = max(cell.height for _, cells in rows for cell in cells)
    columns = max(len(cells) for _, cells in rows)
    header = 28
    sheet = Image.new(
        "RGB",
        (label_width + columns * cell_w, header + len(rows) * cell_h),
        (18, 18, 20),
    )
    draw = ImageDraw.Draw(sheet)
    draw.text((8, 8), title, fill=(230, 230, 230), font=FONT)
    y = header
    for label, cells in rows:
        for offset, line in enumerate(textwrap.wrap(label, width=22)[:4]):
            draw.text((6, y + cell_h // 2 + offset * 11), line, fill=(235, 235, 240), font=FONT)
        x = label_width
        for cell in cells:
            sheet.paste(cell, (x, y))
            x += cell_w
        y += cell_h
    sheet.save(out_path, quality=88)


def pick_prompts(count: int) -> list[PromptEntry]:
    """Evenly spaced entries from prompts.json, master prefix already applied."""
    entries = load_prompt_library(ROOT / "prompts.json")
    if count >= len(entries):
        return entries
    stride = len(entries) / count
    return [entries[int(index * stride)] for index in range(count)]


def pin_timestep(backend: object, timestep: int) -> None:
    """Freeze the breathing sampler at one timestep for a comparable tile."""
    backend.apply_settings(
        {"timestep_min": timestep, "timestep_max": timestep, "instability": 0.0}
    )


def run_grid(args: argparse.Namespace, config: AppConfig, out_dir: Path) -> dict[str, object]:
    resolution = config.diffusion_size
    prompts = pick_prompts(args.prompts)
    seeds = grid_seeds(config.world_seed, args.seeds)
    tiles_dir = out_dir / "grid"
    tiles_dir.mkdir(parents=True, exist_ok=True)

    renderer = ProxyRenderer(
        ROOT, resolution, fullscreen=False, window_size=resolution, world_seed=config.world_seed
    )
    manifest: list[dict[str, object]] = []
    rows: list[tuple[str, list[Image.Image]]] = []
    started = perf_counter()
    try:
        backend = create_backend(config.backend)
        backend.load(config.backend_dict(ROOT))
        # Hard-cut prompts: a 25 s wall-clock slerp would leave every tile
        # showing the previous prompt's embedding under the new label.
        backend.apply_settings({"prompt_walk_seconds": 0.0})
        try:
            camera = Camera.create_default()
            index = 0
            for seed in seeds:
                renderer.randomize_world(seed)
                renderer.spawn_camera(camera)
                renderer.constrain_camera(camera, None)
                conditioning = renderer.render_proxy(camera, perf_counter())
                label = renderer.world_label()
                Image.fromarray(conditioning.rgb, mode="RGB").save(
                    tiles_dir / f"proxy_{seed}.jpg", quality=88
                )
                cells: list[Image.Image] = []
                for entry in prompts:
                    backend.set_prompt(
                        compose_prompt(entry.prompt, label), config.negative_prompt
                    )
                    for timestep in args.timesteps:
                        pin_timestep(backend, timestep)
                        # Each tile is judged on its own, so drop the walk memory.
                        backend.reseed(config.seed)
                        image = Image.fromarray(
                            backend.generate(conditioning), mode="RGB"
                        )
                        file_name = f"{index:03d}_{seed}_{sanitize(entry.family)}_t{timestep}.jpg"
                        image.save(tiles_dir / file_name, quality=90)
                        cells.append(caption_tile(image, [f"{entry.family} t{timestep}"]))
                        manifest.append(
                            {
                                "file": f"grid/{file_name}",
                                "world_seed": seed,
                                "world_label": label,
                                "prompt_name": entry.family,
                                "prompt_text": compose_prompt(entry.prompt, label),
                                "timestep": timestep,
                                "steps": config.steps,
                                "guidance_scale": config.guidance_scale,
                                "guide_strength": config.guide_strength,
                                "resolution": config.resolution_label,
                            }
                        )
                        index += 1
                rows.append((f"{seed}\n{label}", cells))
                print(f"  seed {seed} ({label}): {len(cells)} tiles")
        finally:
            backend.unload()
    finally:
        renderer.close()
    compose_sheet(
        rows,
        f"Latent walk style sheet - {config.resolution_label}, "
        f"steps {config.steps}, cfg {config.guidance_scale:g}, guide {config.guide_strength:g}",
        out_dir / "style_sheet.jpg",
    )
    (out_dir / "style_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    elapsed = perf_counter() - started
    print(f"Grid: {len(manifest)} tiles in {elapsed:.1f} s")
    return {"tiles": len(manifest), "seconds": elapsed, "seeds": seeds}


def contact_strip(images: list[Image.Image], stride: int, columns: int) -> Image.Image:
    """Every ``stride``-th frame, ending on the last one, laid out in a grid."""
    picked = images[stride - 1 :: stride] if stride > 1 else list(images)
    if picked and picked[-1] is not images[-1]:
        picked.append(images[-1])
    rows = (len(picked) + columns - 1) // columns
    width = TILE_WIDTH
    height = round(TILE_WIDTH * picked[0].height / picked[0].width)
    sheet = Image.new("RGB", (columns * width, rows * height), (12, 12, 14))
    for index, image in enumerate(picked):
        sheet.paste(
            image.resize((width, height)),
            ((index % columns) * width, (index // columns) * height),
        )
    return sheet


def walk_prompt(args: argparse.Namespace) -> PromptEntry:
    """The library entry the walk uses, with --family / --prompt-index / --prompt-text applied.

    A --prompt-text override ships no family settings, so an experiment is
    governed entirely by --set and the shipped config.
    """
    entries = load_prompt_library(ROOT / "prompts.json")
    if args.family:
        matches = [entry for entry in entries if entry.family.lower() == args.family.lower()]
        if not matches:
            names = ", ".join(sorted({entry.family for entry in entries}))
            raise SystemExit(f"Unknown family {args.family!r}; known: {names}")
        entries = matches
    entry = entries[args.prompt_index % len(entries)]
    if args.prefix is None and args.prompt_text is None:
        return entry
    prefix = load_master_prefix(ROOT / "prompts.json") if args.prefix is None else args.prefix
    if prefix.strip().lower() in ("", "none"):
        prefix = ""
    body = entry.prompt if args.prompt_text is None else args.prompt_text
    return PromptEntry(
        family="Override" if args.prompt_text is not None else entry.family,
        prompt=apply_master_prefix(prefix, body.strip()),
        settings={} if args.prompt_text is not None else dict(entry.settings),
    )


def run_walk(args: argparse.Namespace, config: AppConfig, out_dir: Path) -> dict[str, object]:
    """Walk forward through one world with the backend's walk state intact."""
    resolution = config.diffusion_size
    entry = walk_prompt(args)
    negative = config.negative_prompt if args.negative is None else args.negative
    world_seed = config.world_seed if args.seed is None else args.seed
    frames_dir = out_dir / "walk"
    frames_dir.mkdir(parents=True, exist_ok=True)

    renderer = ProxyRenderer(
        ROOT, resolution, fullscreen=False, window_size=resolution, world_seed=world_seed
    )
    images: list[Image.Image] = []
    started = perf_counter()
    clock = FrameClock(args.walk_clock_fps) if args.walk_clock_fps > 0.0 else None
    try:
        backend = create_backend(config.backend)
        backend.load(config.backend_dict(ROOT))
        try:
            backend.apply_settings(config.backend_settings())
            backend.reseed(config.seed)
            if clock is not None and hasattr(backend, "set_clock"):
                backend.set_clock(clock)
            camera = Camera.create_default()
            renderer.spawn_camera(camera)
            label = renderer.world_label()
            effective_prompt = compose_prompt(entry.prompt, label)
            backend.set_prompt(effective_prompt, negative)
            for step in range(args.walk_frames):
                before = camera.position.copy()
                camera.walk(0.0, 1.0, args.walk_speed)
                renderer.constrain_camera(camera, 1.0 / max(args.walk_clock_fps, 1.0))
                # A person turns when a block stops them; the walk does the same.
                if float(np.linalg.norm((camera.position - before)[[0, 2]])) < args.walk_speed * 0.4:
                    open_yaw, distance = renderer.open_heading(camera)
                    if distance > 0.0:
                        camera.yaw = open_yaw
                # The renderer animates on the clock it is handed, so the frame
                # clock has to drive it too or the walk is not reproducible.
                now = clock() if clock is not None else perf_counter()
                conditioning = renderer.render_proxy(camera, now)
                output = backend.generate(conditioning)
                image = Image.fromarray(output, mode="RGB")
                image.save(frames_dir / f"frame_{step:02d}.jpg", quality=90)
                images.append(image)
                if clock is not None:
                    clock.tick()
        finally:
            backend.unload()
    finally:
        renderer.close()

    stride = max(1, args.strip_stride)
    contact_strip(images, 1, 6).save(out_dir / "walk.jpg", quality=90)
    contact_strip(images, stride, 8).save(out_dir / f"walk_every{stride}.jpg", quality=90)

    small = [np.asarray(image.resize((60, 48)), dtype=np.float32) for image in images]
    diffs = [float(np.abs(small[i] - small[i - 1]).mean()) for i in range(1, len(small))]
    report = {
        "prompt_name": entry.family,
        "prompt_text": effective_prompt,
        "hue_words": hue_words(label),
        "negative_prompt": negative,
        "world_seed": world_seed,
        "world_label": label,
        "frames": len(images),
        "walk_speed": args.walk_speed,
        "walk_clock_fps": args.walk_clock_fps,
        "mean_abs_diff": float(np.mean(diffs)) if diffs else 0.0,
        "seconds": perf_counter() - started,
        **config.backend_settings(),
    }
    (out_dir / "walk_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(
        f"Walk: {len(images)} frames, mean |delta| {report['mean_abs_diff']:.1f}/255, "
        f"{report['seconds']:.1f} s"
    )
    return report


def run_families(args: argparse.Namespace, config: AppConfig, out_dir: Path) -> dict[str, object]:
    """One short walk per style family through one world, eight frames per row.

    Every family gets the same world, spawn and world seed, so the sheet shows
    what the style library alone changes. Each family's own settings are applied
    before its walk, and the hard prompt cut drops the memory latent so no look
    bleeds into the next.
    """
    resolution = config.diffusion_size
    world_seed = config.world_seed if args.seed is None else args.seed
    negative = config.negative_prompt if args.negative is None else args.negative
    families: list[PromptEntry] = []
    for entry in load_prompt_library(ROOT / "prompts.json"):
        if entry.family not in {seen.family for seen in families}:
            families.append(entry)
    columns = 8
    stride = max(1, args.family_frames // columns)

    renderer = ProxyRenderer(
        ROOT, resolution, fullscreen=False, window_size=resolution, world_seed=world_seed
    )
    rows: list[tuple[str, list[Image.Image]]] = []
    report: list[dict[str, object]] = []
    started = perf_counter()
    try:
        backend = create_backend(config.backend)
        backend.load(config.backend_dict(ROOT))
        try:
            label = renderer.world_label()
            for entry in families:
                clock = FrameClock(args.walk_clock_fps if args.walk_clock_fps > 0 else 10.0)
                if hasattr(backend, "set_clock"):
                    backend.set_clock(clock)
                backend.apply_settings(
                    {**config.backend_settings(), **entry.settings, "prompt_walk_seconds": 0.0}
                )
                backend.reseed(config.seed)
                backend.set_prompt(compose_prompt(entry.prompt, label), negative)
                camera = Camera.create_default()
                renderer.spawn_camera(camera)
                frames: list[Image.Image] = []
                for _ in range(args.family_frames):
                    camera.walk(0.0, 1.0, args.walk_speed)
                    renderer.constrain_camera(camera, 1.0 / max(args.walk_clock_fps, 1.0))
                    frames.append(
                        Image.fromarray(
                            backend.generate(renderer.render_proxy(camera, clock())), mode="RGB"
                        )
                    )
                    clock.tick()
                picked = frames[stride - 1 :: stride][:columns] or frames[:columns]
                rows.append((entry.family, [caption_tile(image, []) for image in picked]))
                report.append({"family": entry.family, "prompt": entry.prompt, **entry.settings})
                print(f"  {entry.family}: {len(frames)} frames")
        finally:
            backend.unload()
    finally:
        renderer.close()
    compose_sheet(
        rows,
        f"Style families - world {world_seed} ({label}), {config.resolution_label}, "
        f"{args.family_frames} frames each",
        out_dir / "look_families.jpg",
    )
    (out_dir / "look_families.json").write_text(
        json.dumps({"world_seed": world_seed, "world_label": label, "families": report}, indent=2),
        encoding="utf-8",
    )
    elapsed = perf_counter() - started
    print(f"Families: {len(rows)} rows in {elapsed:.1f} s")
    return {"families": len(rows), "seconds": elapsed}


def step_walker(
    renderer: ProxyRenderer, camera: Camera, speed: float, dt: float
) -> None:
    """One walk step with the app's stuck-turn: blocked walkers turn, they do not grind."""
    before = camera.position.copy()
    camera.walk(0.0, 1.0, speed)
    renderer.constrain_camera(camera, dt)
    if float(np.linalg.norm((camera.position - before)[[0, 2]])) < speed * 0.4:
        open_yaw, distance = renderer.open_heading(camera)
        if distance > 0.0:
            camera.yaw = open_yaw


def run_space_resets(
    args: argparse.Namespace, config: AppConfig, out_dir: Path
) -> dict[str, object]:
    """One row per simulated Space press, to judge whether a reset changes everything.

    Mirrors the Space handler in ``app.main``: a new world seed, a new diffusion
    noise seed (which drops the memory latent and the noise walk), and a family
    from some other family than the current one, with that family's settings
    applied live. The prompt walk is left at its configured length, so the first
    tiles of a row still carry the previous prompt's embedding exactly as they
    do in the app; the world, noise and memory change on the press itself.
    """
    resolution = config.diffusion_size
    presses = max(1, args.space_resets)
    columns = 8
    stride = max(1, args.reset_frames // columns)
    clock_fps = args.walk_clock_fps if args.walk_clock_fps > 0 else 10.0

    renderer = ProxyRenderer(
        ROOT, resolution, fullscreen=False, window_size=resolution, world_seed=config.world_seed
    )
    rows: list[tuple[str, list[Image.Image]]] = []
    report: list[dict[str, object]] = []
    started = perf_counter()
    clock = FrameClock(clock_fps)
    try:
        backend = create_backend(config.backend)
        backend.load(config.backend_dict(ROOT))
        try:
            backend.apply_settings(config.backend_settings())
            if hasattr(backend, "set_clock"):
                backend.set_clock(clock)
            entry = PromptEntry(family="Custom", prompt=config.prompt, settings={})
            recent_families: list[str] = [entry.family]
            camera = Camera.create_default()
            for press in range(presses):
                # Mirrors the app: one redraw when the palette repeats.
                previous_hues = world_hues(renderer.world_label())
                world_seed = renderer.randomize_world()
                if world_hues(renderer.world_label()) == previous_hues:
                    world_seed = renderer.randomize_world()
                noise_seed = int(np.random.SeedSequence().generate_state(1, dtype=np.uint32)[0])
                backend.reseed(noise_seed)
                renderer.spawn_camera(camera)
                label = renderer.world_label()
                entry = choose_family_prompt(
                    load_prompt_library(ROOT / "prompts.json"), recent_families
                )
                recent_families.append(entry.family)
                del recent_families[:-RECENT_FAMILY_MEMORY]
                backend.apply_settings({**config.backend_settings(), **entry.settings})
                effective = compose_prompt(entry.prompt, label)
                backend.set_prompt(effective, config.negative_prompt)
                frames: list[Image.Image] = []
                for _ in range(args.reset_frames):
                    step_walker(renderer, camera, args.walk_speed, 1.0 / clock_fps)
                    frames.append(
                        Image.fromarray(
                            backend.generate(renderer.render_proxy(camera, clock())), mode="RGB"
                        )
                    )
                    clock.tick()
                picked = frames[stride - 1 :: stride][:columns] or frames[:columns]
                # The seed is long enough to push the palette off the label, so
                # it lives in space_resets.json instead.
                rows.append(
                    (f"{entry.family}\n{label}", [caption_tile(i, []) for i in picked])
                )
                report.append(
                    {
                        "press": press + 1,
                        "family": entry.family,
                        "world_seed": world_seed,
                        "world_label": label,
                        "noise_seed": noise_seed,
                        "prompt": effective,
                        **entry.settings,
                    }
                )
                print(f"  press {press + 1}: {entry.family} / {world_seed} ({label})")
        finally:
            backend.unload()
    finally:
        renderer.close()
    compose_sheet(
        rows,
        f"Space resets - {presses} presses, {config.resolution_label}, "
        f"{args.reset_frames} frames each, every {stride}th shown",
        out_dir / "space_resets.jpg",
    )
    (out_dir / "space_resets.json").write_text(
        json.dumps({"presses": report}, indent=2), encoding="utf-8"
    )
    elapsed = perf_counter() - started
    print(f"Space resets: {len(rows)} rows in {elapsed:.1f} s")
    return {"presses": len(rows), "seconds": elapsed}


def build_html(out_dir: Path) -> None:
    """Rebuild style_sheet.html from whatever grid/walk output exists in --out."""
    manifest_path = out_dir / "style_manifest.json"
    manifest = (
        json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else []
    )
    walk_path = out_dir / "walk_report.json"
    walk = json.loads(walk_path.read_text(encoding="utf-8")) if walk_path.exists() else None
    reference = os.path.relpath(
        ROOT / "logs" / "reference" / "kosma3_contact.jpg", out_dir
    ).replace("\\", "/")

    by_seed: dict[str, list[dict[str, object]]] = {}
    for entry in manifest:
        by_seed.setdefault(f"{entry['world_seed']} — {entry['world_label']}", []).append(entry)

    parts = [
        "<!doctype html><html><head><meta charset='utf-8'>",
        "<title>Latent walk style sheet v2</title>",
        "<style>body{background:#111;color:#eee;font-family:sans-serif;padding:16px}",
        "h1,h2{margin-top:32px} .row{display:flex;flex-wrap:wrap;gap:8px;align-items:flex-start}",
        ".tile{width:200px;font-size:11px} .tile img{width:200px;display:block}",
        ".cap{margin-top:2px;color:#bbb} img.wide{max-width:100%}",
        "table{border-collapse:collapse;margin-top:8px} td,th{border:1px solid #444;padding:5px 9px;text-align:left}",
        "</style></head><body>",
        "<h1>Reference: KOSMA 2025 latent-space walk</h1>",
        f"<img class='wide' src='{reference}'>",
    ]
    if (out_dir / "style_sheet.jpg").exists():
        parts += ["<h1>Grid: world seed x prompt x timestep</h1>", "<img class='wide' src='style_sheet.jpg'>"]
    for label, entries in by_seed.items():
        parts.append(f"<h2>{label}</h2><div class='row'>")
        for entry in entries:
            parts.append(
                f"<div class='tile'><img src='{entry['file']}'>"
                f"<div class='cap'>{entry['prompt_name']} · t{entry['timestep']}</div></div>"
            )
        parts.append("</div>")
    if walk is not None:
        parts += [
            f"<h1>Walk: {walk['frames']} frames of flight with live walk state</h1>",
            f"<p>{walk['prompt_name']} · seed {walk['world_seed']} · {walk['world_label']} · "
            f"mean |&Delta;| (60x48) = {walk['mean_abs_diff']:.1f}/255</p>",
            "<img class='wide' src='walk.jpg'>",
            "<table><tr><th>setting</th><th>value</th></tr>",
        ]
        for key in (
            "steps",
            "guidance_scale",
            "timestep_min",
            "timestep_max",
            "instability",
            "guide_strength",
            "memory_match",
            "memory_match_std",
            "memory_leash",
            "noise_walk_seconds",
            "noise_jitter",
            "prompt_walk_seconds",
            "feedback_reprojection",
        ):
            parts.append(f"<tr><td>{key}</td><td>{walk.get(key)}</td></tr>")
        parts.append("</table>")
    parts.append("</body></html>")
    (out_dir / "style_sheet.html").write_text("\n".join(parts), encoding="utf-8")


def main() -> int:
    args = parse_args()
    configure_local_environment(ROOT, offline=True)
    config = AppConfig.load(ROOT / "config.json")
    if args.resolution:
        config.diffusion_resolution = args.resolution
    # The family's own settings come first so an explicit --set still wins.
    # Space-reset mode applies each row's family settings itself, so the base
    # config must stay as shipped.
    if args.walk or not (args.grid or args.families or args.space_resets):
        for key, value in walk_prompt(args).settings.items():
            setattr(config, key, value)
    overrides = apply_overrides(config, args.set)
    if overrides:
        print(f"Overrides: {overrides}")
    out_dir = args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    run_both = not (args.grid or args.walk or args.families or args.space_resets)
    if args.space_resets:
        run_space_resets(args, config, out_dir)
        return 0
    if args.families:
        run_families(args, config, out_dir)
        return 0
    if args.grid or run_both:
        run_grid(args, config, out_dir)
    if args.walk or run_both:
        run_walk(args, config, out_dir)
    build_html(out_dir)
    strip_name = f"walk_every{max(1, args.strip_stride)}.jpg"
    for name in ("style_sheet.jpg", "walk.jpg", strip_name, "style_sheet.html"):
        print(f"  {out_dir / name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
