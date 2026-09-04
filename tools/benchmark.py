"""Measure diffusion frame cost across resolution modes and step counts.

Loads the configured backend once, then sweeps every requested mode at each
requested step count and writes benchmark_results.json / .txt next to the repo.

Usage:
    runtime/python/python.exe tools/benchmark.py
    runtime/python/python.exe tools/benchmark.py --modes 512x512 --steps 1 --frames 30
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from dataclasses import asdict
from pathlib import Path
from time import perf_counter

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.config import RESOLUTION_MODES, AppConfig, configure_local_environment
from app.diffusion.factory import create_backend
from app.renderer.camera import Camera
from app.types import ConditioningFrame


def make_conditioning(
    size: int | tuple[int, int], sequence: int = 0
) -> ConditioningFrame:
    """Synthetic proxy frame: blocky slabs with a depth ramp, no GL required."""
    width, height = (size, size) if isinstance(size, int) else size
    yy, xx = np.mgrid[0:height, 0:width]
    checker = ((xx // 48 + yy // 48) % 2).astype(np.float32)
    rgb = np.empty((height, width, 3), dtype=np.uint8)
    rgb[:, :, 0] = np.clip(48 + xx * 145 / width + checker * 28, 0, 255)
    rgb[:, :, 1] = np.clip(55 + yy * 135 / height, 0, 255)
    rgb[:, :, 2] = np.clip(155 - checker * 52, 0, 255)
    depth = np.clip(0.15 + yy / height * 0.8, 0, 1).astype(np.float32)
    edges = np.zeros((height, width), dtype=np.uint8)
    edges[:, 47::48] = 255
    edges[47::48, :] = 255
    return ConditioningFrame(
        rgb, depth, edges, Camera.create_default().snapshot(), perf_counter(), sequence
    )


def parse_modes(values: list[str], parser: argparse.ArgumentParser) -> list[tuple[int, int]]:
    modes: list[tuple[int, int]] = []
    for value in values:
        try:
            width_text, height_text = value.lower().split("x", 1)
            mode = (int(width_text), int(height_text))
        except ValueError:
            parser.error(f"unsupported resolution: {value}")
        if mode not in RESOLUTION_MODES:
            parser.error(f"{value} is not one of the six RESOLUTION_MODES")
        modes.append(mode)
    return modes


def measure(
    backend: object,
    mode: tuple[int, int],
    steps: int,
    guidance_scale: float,
    warmup: int,
    frames: int,
) -> dict[str, object]:
    """Time one mode/steps combination on an already-loaded backend."""
    width, height = mode
    result: dict[str, object] = {
        "resolution": f"{width}x{height}",
        "steps": steps,
        "guidance_scale": guidance_scale,
        "status": "failed",
    }
    torch = getattr(backend, "torch", None)
    try:
        backend.set_resolution(width, height)
        backend.apply_settings({"steps": steps, "guidance_scale": guidance_scale})
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        # Built up front: synthesising a frame costs 4-37 ms of numpy and must
        # not be charged to diffusion latency.
        prepared = [make_conditioning(mode, index) for index in range(frames)]
        for _ in range(max(0, warmup)):
            backend.generate(prepared[0])
        if torch is not None and torch.cuda.is_available():
            torch.cuda.synchronize()
        timings: list[float] = []
        for conditioning in prepared:
            started = perf_counter()
            backend.generate(conditioning)
            if torch is not None and torch.cuda.is_available():
                torch.cuda.synchronize()
            timings.append((perf_counter() - started) * 1000.0)
        stats = backend.stats()
        average_ms = statistics.fmean(timings)
        result.update(
            {
                "status": "ok",
                "mean_ms": average_ms,
                "median_ms": statistics.median(timings),
                "min_ms": min(timings),
                "max_ms": max(timings),
                "fps": 1000.0 / average_ms,
                "peak_vram_gb": float(stats.get("peak_vram_gb", 0.0)),
                "allocated_vram_gb": float(stats.get("vram_allocated_gb", 0.0)),
                "samples": frames,
            }
        )
    except Exception as exc:
        result["error"] = str(exc)
    return result


def write_text(payload: dict[str, object], path: Path) -> None:
    results: list[dict[str, object]] = payload["results"]  # type: ignore[assignment]
    lines = [
        "LATENT SPACE - DIFFUSION BENCHMARK",
        f"GPU:     {payload.get('gpu', 'unknown')}",
        f"Backend: {payload['backend']}",
        f"Load:    {float(payload.get('load_ms', 0.0)):.0f} ms",
        "",
        f"{'mode':>10}  {'steps':>5}  {'cfg':>4}  {'ms':>8}  {'fps':>6}  {'peak VRAM':>10}",
    ]
    for item in results:
        if item["status"] == "ok":
            lines.append(
                f"{item['resolution']:>10}  {item['steps']:>5}  {item['guidance_scale']:>4.2g}  "
                f"{item['mean_ms']:>8.1f}  {item['fps']:>6.2f}  {item['peak_vram_gb']:>7.2f} GB"
            )
        else:
            lines.append(
                f"{item['resolution']:>10}  {item['steps']:>5}  FAILED  {item.get('error', '')}"
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("latent_walk", "proxy_passthrough"))
    parser.add_argument(
        "--modes",
        nargs="+",
        default=[f"{width}x{height}" for width, height in RESOLUTION_MODES],
        help="WIDTHxHEIGHT modes; defaults to every RESOLUTION_MODES entry",
    )
    parser.add_argument("--steps", nargs="+", type=int, default=[1, 2])
    parser.add_argument("--guidance-scale", type=float)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--frames", type=int, default=20)
    parser.add_argument("--output-dir", type=Path, default=ROOT)
    args = parser.parse_args()
    modes = parse_modes(args.modes, parser)
    if any(not 1 <= value <= 4 for value in args.steps):
        parser.error("steps must be between 1 and 4")

    configure_local_environment(ROOT, offline=True)
    config = AppConfig.load(ROOT / "config.json")
    if args.backend:
        config.backend = args.backend
    guidance_scale = (
        args.guidance_scale if args.guidance_scale is not None else config.guidance_scale
    )
    gpu = "unavailable"
    try:
        import torch

        if torch.cuda.is_available():
            gpu = torch.cuda.get_device_name(0)
    except ImportError:
        pass

    payload: dict[str, object] = {
        "backend": config.backend,
        "gpu": gpu,
        "config": asdict(config),
        "results": [],
    }
    backend = create_backend(config.backend)
    try:
        load_started = perf_counter()
        backend.load(config.backend_dict(ROOT))
        payload["load_ms"] = (perf_counter() - load_started) * 1000.0
        for mode in modes:
            for steps in args.steps:
                item = measure(
                    backend, mode, steps, guidance_scale, args.warmup, args.frames
                )
                payload["results"].append(item)  # type: ignore[union-attr]
                print(
                    f"{item['resolution']} steps={steps}: "
                    + (
                        f"{item['mean_ms']:.1f} ms, {item['fps']:.2f} FPS"
                        if item["status"] == "ok"
                        else f"FAILED {item.get('error', '')}"
                    )
                )
    finally:
        backend.unload()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "benchmark_results.json"
    text_path = args.output_dir / "benchmark_results.txt"
    json_path.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")
    write_text(payload, text_path)
    print(text_path.read_text(encoding="utf-8"))
    results: list[dict[str, object]] = payload["results"]  # type: ignore[assignment]
    return 0 if all(item["status"] == "ok" for item in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
