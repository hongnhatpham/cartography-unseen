"""Isolated spacing studies. Mutations live only in this process, never the app.

Run with the bundled Python from the repository root. Geometry is filtered after
the existing chunk cap, so removing panels cannot introduce formerly capped mass.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from functools import lru_cache
import json
from math import floor
from pathlib import Path
import sys

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from app.config import AppConfig, configure_local_environment
from app.renderer import world

BASE_GENERATE = world.generate_chunk
VARIANTS = {
    "current": (0.0, 0.0, 160.0),
    "a": (0.22, 0.22, 160.0),
    "b": (0.05, 0.60, 160.0),
    "c": (0.05, 0.65, 240.0),
}


def removal_at(position, seed, variant):
    """Neighbouring cells share a broad density field, preserving denser areas."""
    low, high, scale = VARIANTS[variant]
    x, y, z = position
    if variant == "c":
        return low + (high - low) * world.channel_weight(x, y, z, seed)
    noise = world.value_noise(x, y * 0.5, z, scale, seed, 0x5FACE)
    amount = world._smoothstep(float(np.clip((noise - 0.25) / 0.5, 0.0, 1.0)))
    return low + (high - low) * amount


def install_variant(variant):
    """Replace chunk access in the probe process, with consistent collision data."""
    @lru_cache(maxsize=192)
    def generate(coord, seed):
        chunk = BASE_GENERATE(coord, seed)
        if variant == "current":
            return chunk
        cubes = []
        for cube in chunk.objects:
            if variant == "c" and cube.role == "panel":
                cubes.append(cube)
                continue
            # Stable spatial groups keep adjacent pieces from becoming a field
            # of unrelated specks; every variant uses the same removal order.
            group = tuple(floor(value / 16.0) for value in cube.position)
            draw = world._unit_float(seed, 0x5AACE, *group)
            if draw >= removal_at(cube.position, seed, variant):
                cubes.append(cube)
        return replace(chunk, objects=tuple(cubes))

    world.generate_chunk = generate
    world._chunk_collider_rows.cache_clear()
    world._neighbourhood_colliders.cache_clear()
    # The live renderer imported the function by name. Patch that binding only
    # when rendering, leaving geometry-only runs free of SDL initialization.
    module = sys.modules.get("app.renderer.proxy_renderer")
    if module is not None:
        module.generate_chunk = generate


def read_poses(path):
    """Read the shared, baseline-valid positions used by all four treatments."""
    return json.loads(path.read_text(encoding="utf-8"))


def evaluate(poses):
    results = {}
    directions = tuple(world.direction_of(yaw, pitch) for yaw in range(0, 360, 30)
                       for pitch in (-25.0, 0.0, 25.0))
    for variant in VARIANTS:
        install_variant(variant)
        rows = []
        for pose in poses:
            position = pose["position"]
            seed = pose["seed"]
            colliders = world._neighbourhood_colliders(world.world_to_chunk(position[0], position[2]), seed)
            distances = world.clear_distances(position, directions, colliders, reach=60.0, step=2.0)
            rows.append({
                **pose,
                "mean_clearance": float(distances.mean()),
                "near_share": float((distances < 8.0).mean()),
                "open_share": float((distances >= 30.0).mean()),
                "empty_share": float((distances >= 60.0).mean()),
                "removal": removal_at(position, seed, variant),
            })
        results[variant] = rows
        print(variant, "near", round(np.mean([r["near_share"] for r in rows]), 3),
              "open", round(np.mean([r["open_share"] for r in rows]), 3), flush=True)
    return results


def render(poses, out, config, variants):
    from app.diffusion.factory import create_backend
    from app.main import compose_prompt
    from app.renderer.camera import Camera
    from app.renderer.proxy_renderer import ProxyRenderer
    from tools.style_sheet import FrameClock

    backend = create_backend(config.backend)
    backend.load(config.backend_dict(ROOT))
    try:
        for variant in variants:
            install_variant(variant)
            renderer = ProxyRenderer(ROOT, config.diffusion_size, fullscreen=False,
                                     window_size=config.diffusion_size, world_seed=poses[0]["seed"])
            try:
                for index, pose in enumerate(poses):
                    if renderer.world_seed != pose["seed"]:
                        renderer.randomize_world(pose["seed"])
                    camera = Camera.create_default()
                    camera.position[:] = pose["position"]
                    camera.yaw, camera.pitch = pose["yaw"], pose["pitch"]
                    renderer._update_world(camera.position)
                    clock = FrameClock(10.0)
                    backend.set_clock(clock)
                    backend.reseed(config.seed)
                    backend.apply_settings(config.backend_settings())
                    backend.set_prompt(compose_prompt(config.prompt, renderer.world_label()), config.negative_prompt)
                    # Finish any prompt transition before comparing geometry.
                    # Otherwise the first treatment has different embeddings
                    # from later treatments despite identical prompt text.
                    clock.now = config.prompt_walk_seconds + 1.0
                    backend.prompt_walk.advance()
                    assert backend.prompt_walk.t == 1.0
                    clock.now = 0.0
                    for _ in range(12):
                        frame = renderer.render_proxy(camera, clock())
                        output = backend.generate(frame)
                        clock.tick()
                    Image.fromarray(frame.rgb).save(out / f"{variant}-{index}-proxy.jpg", quality=86)
                    Image.fromarray(output).save(out / f"{variant}-{index}-ai.jpg", quality=88)
                    print("rendered", variant, index, flush=True)
            finally:
                renderer.close()
    finally:
        backend.unload()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--poses", type=Path, required=True)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--variants", nargs="+", choices=VARIANTS, default=list(VARIANTS))
    parser.add_argument("--out", type=Path, default=ROOT / "logs/reference/spacing-review/variants")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    poses = read_poses(args.poses)
    if args.render:
        configure_local_environment(ROOT, offline=True)
        config = AppConfig.load(ROOT / "config.json")
        # Snapshot the current live tuning for this offline run. Nothing is
        # written to config.json, and no family override is silently applied.
        (args.out / "render-settings.json").write_text(json.dumps(asdict(config), indent=2), encoding="utf-8")
        render(poses, args.out, config, args.variants)
    else:
        results = evaluate(poses)
        (args.out / "comparison.json").write_text(json.dumps(results, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
