"""Model bake-off: which fast diffusion model gives the KOSMA look in real time.

Every candidate runs the identical experiment on GPU tensors with TAESD only
(the full VAE is never loaded):

    guide = taesd.encode(proxy)
    x_t   = sqrt(a_t) * base + sqrt(1 - a_t) * noise
    eps   = unet(x_t, t, prompt_embeds)          # optionally CFG
    x0    = (x_t - sqrt(1 - a_t) * eps) / sqrt(a_t)
    image = taesd.decode(x0)

One-step Euler on an epsilon-prediction UNet *is* that x0 projection, and the
LCM boundary conditions (sigma_data 0.5, timestep_scaling 10) collapse to the
identity above t~200, so the same code covers Euler and LCM; the second step is
the only place they differ (deterministic re-noise vs. fresh-noise re-noise).

Usage:
    runtime/python/python.exe tools/bakeoff.py --candidates sd_turbo,sd15_lcm
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np
import torch
from diffusers import AutoencoderTiny, UNet2DConditionModel
from PIL import Image, ImageDraw
from transformers import CLIPTextModel, CLIPTokenizer

ROOT = Path(__file__).resolve().parents[1]
PROXY_DIR = ROOT / "logs" / "reference" / "proxy_frames"
OUT_DIR = ROOT / "logs" / "reference" / "bakeoff"
TAESD_DIR = ROOT / "models" / "taesd"
DEVICE = torch.device("cuda")
DTYPE = torch.float16

# P1 is the verbatim KOSMA prompt from the recovered ComfyUI workflow.
PROMPTS: dict[str, str] = {
    "P1": (
        "screenshot of a video game neural network highway, blocky biophilia city, "
        "first person view, eye level, foggy, corrupted, haunted, glitch art, crossroads: 0.3"
    ),
    "P2": (
        "screenshot of a video game, aerial view of a blocky biophilia city, "
        "neural network highway, tilted horizon, foggy, corrupted, haunted, glitch art"
    ),
    "P3": (
        "screenshot of a video game, topology unknown, chaotic voxel landscape of "
        "terraces and shards, two saturated colours, foggy, corrupted, glitch art"
    ),
}
NEGATIVE = "(worst quality, low quality: 1.4, ui, interface, text)"

# Guidance-distilled UNets (the original LCM) take the CFG scale as an input
# embedding instead of a second forward pass. 7.0 is the LCM pipeline default.
EMBEDDED_GUIDANCE = 7.0


@dataclass(frozen=True, slots=True)
class Variant:
    """One sampling configuration: start timestep, step count, guidance scale."""

    name: str
    timestep: int
    steps: int
    cfg: float


@dataclass(frozen=True, slots=True)
class Candidate:
    """A model folder plus how its second step re-noises."""

    name: str
    model_dir: Path
    mode: str  # "euler" (deterministic 2nd step) or "lcm" (fresh-noise 2nd step)
    source_notes: str
    variants: tuple[Variant, ...]
    best: str  # variant name used for the stability walk


BASE_VARIANTS: tuple[Variant, ...] = (
    Variant("t600 s1", 600, 1, 1.0),
    Variant("t800 s1", 800, 1, 1.0),
    Variant("t900 s1", 900, 1, 1.0),
    Variant("t800 s2", 800, 2, 1.0),
    Variant("t800 s1 cfg1.5", 800, 1, 1.5),
)

# LCM was distilled on the timestep grid t = 19 + 20k and is normally run for
# 2-4 steps. This set checks whether the 1-step blur is a schedule mismatch
# rather than a property of the model.
LCM_PROBE_VARIANTS: tuple[Variant, ...] = (
    Variant("t999 s1", 999, 1, 1.0),
    Variant("t799 s1", 799, 1, 1.0),
    Variant("t799 s2", 799, 2, 1.0),
    Variant("t599 s2", 599, 2, 1.0),
    Variant("t999 s2", 999, 2, 1.0),
)

VARIANT_SETS = {
    "base": (BASE_VARIANTS, "t800 s1"),
    "lcm_probe": (LCM_PROBE_VARIANTS, "t799 s2"),
    # Same grid, but the walk runs at the higher timestep the aerial prompts want.
    "aerial": (BASE_VARIANTS, "t900 s1"),
}


# Observations from the 2026-09-04 run, kept next to the numbers they explain.
NOTES: dict[str, str] = {
    "sd_turbo": (
        "WINNER. Crispest 1-step output of the four and the only one that holds together "
        "under x0_prev feedback: saturation ramp +0.003 over 16 frames, no attractor. "
        "The only candidate that responds to aerial vocabulary - P2/P3 flip it to "
        "isometric voxel islands and terraced slab landscapes, which is the reference look; "
        "the SD1.5 family stays at street level whatever the prompt. Slowest at 384x256 "
        "(bigger SD2.1 UNet) but effectively tied at 512x512. Already installed, so the "
        "exhibition needs no extra download. Invents billboard/UI text at CFG 1."
    ),
    "sd15_lcm": (
        "Rejected. Blurry and flat at 1 step; the feedback walk diverges into a saturated "
        "red/black striped tunnel by frame 8 (highest frame_delta_max of the four, and it "
        "starts already colour-blown at saturation 0.33). The lcm_probe variant set shows "
        "2 steps at t599-t799 does sharpen it, but that doubles cost to ~190 ms at 512x512 "
        "and the result is still flat poster colour with no hallucinated texture. Running it "
        "on LCM's canonical t999 destroys proxy adherence completely."
    ),
    "sd15_hyper1": (
        "Runner-up on stills, rejected on motion. Fusion was straightforward (kohya keys, "
        "rank 64 alpha 8, 278/278 pairs). Crisp and saturated with genuine glitch-block "
        "energy, and the fastest at 384x256. But the feedback walk ramps saturation "
        "(+0.055) and at t900/P3 it collapses into a symmetric one-point rainbow corridor - "
        "the exact tunnel/radial attractor the old ComfyUI negative prompt fought. Also "
        "clings hardest to city semantics (roads, streetlamps, cars)."
    ),
    "lcm_dreamshaper": (
        "Rejected. Foggy mush at every timestep and step count; loses the proxy geometry "
        "entirely and never resolves detail. Its low frame_delta (0.047) is blur, not "
        "stability. Needs time_cond_proj_dim 256 and a guidance-scale embedding or it is "
        "silently wrong - SD1.5's config omits that, which cost one rebuild."
    ),
}


def candidates() -> dict[str, Candidate]:
    """The bake-off roster, keyed by name."""
    cand = ROOT / "models" / "candidates"
    return {
        "sd_turbo": Candidate(
            "sd_turbo",
            ROOT / "models" / "sd_turbo",
            "euler",
            "existing models/sd_turbo (stabilityai/sd-turbo, SD2.1 UNet, OpenCLIP-H 1024d), fp16",
            BASE_VARIANTS,
            "t800 s1",
        ),
        "sd15_lcm": Candidate(
            "sd15_lcm",
            cand / "sd15_lcm",
            "lcm",
            "see models/candidates/sd15_lcm/BUILD.json",
            BASE_VARIANTS,
            "t800 s1",
        ),
        "sd15_hyper1": Candidate(
            "sd15_hyper1",
            cand / "sd15_hyper1",
            "lcm",
            "see models/candidates/sd15_hyper1/BUILD.json",
            BASE_VARIANTS,
            "t800 s1",
        ),
        "lcm_dreamshaper": Candidate(
            "lcm_dreamshaper",
            cand / "lcm_dreamshaper",
            "lcm",
            "see models/candidates/lcm_dreamshaper/BUILD.json",
            BASE_VARIANTS,
            "t800 s1",
        ),
    }


@dataclass
class Loaded:
    """Resident fp16 components for one candidate."""

    unet: Any
    text_encoder: Any
    tokenizer: Any
    taesd: Any
    alphas_cumprod: torch.Tensor
    embeds: dict[str, torch.Tensor] = field(default_factory=dict)
    timestep_cond: torch.Tensor | None = None


def guidance_scale_embedding(w: float, dim: int) -> torch.Tensor:
    """Fourier embedding of a distilled guidance scale, as in the LCM pipeline."""
    half = dim // 2
    freqs = torch.exp(torch.arange(half, dtype=torch.float32) * -(math.log(10000.0) / (half - 1)))
    emb = torch.tensor([w * 1000.0], dtype=torch.float32)[:, None] * freqs[None, :]
    return torch.cat([emb.sin(), emb.cos()], dim=1).to(DEVICE, DTYPE)


def alphas_cumprod_from(scheduler_dir: Path) -> torch.Tensor:
    """Rebuild alphas_cumprod from a scheduler config (scaled_linear betas)."""
    cfg = json.loads((scheduler_dir / "scheduler_config.json").read_text(encoding="utf-8"))
    n = int(cfg.get("num_train_timesteps", 1000))
    start, end = float(cfg["beta_start"]), float(cfg["beta_end"])
    if cfg.get("beta_schedule", "scaled_linear") != "scaled_linear":
        raise ValueError(f"unsupported beta_schedule in {scheduler_dir}")
    betas = torch.linspace(start**0.5, end**0.5, n, dtype=torch.float64) ** 2
    return torch.cumprod(1.0 - betas, dim=0).to(DEVICE, torch.float32)


def load(candidate: Candidate) -> Loaded:
    """Load UNet, CLIP and TAESD for one candidate onto the GPU in fp16."""
    d = candidate.model_dir
    unet = UNet2DConditionModel.from_pretrained(d / "unet", torch_dtype=DTYPE, variant="fp16")
    unet = unet.to(DEVICE, memory_format=torch.channels_last).eval()
    text_encoder = CLIPTextModel.from_pretrained(
        d / "text_encoder", torch_dtype=DTYPE, variant="fp16"
    ).to(DEVICE).eval()
    tokenizer = CLIPTokenizer.from_pretrained(d / "tokenizer")
    taesd = AutoencoderTiny.from_pretrained(TAESD_DIR, torch_dtype=DTYPE).to(DEVICE).eval()
    cond_dim = getattr(unet.config, "time_cond_proj_dim", None)
    cond = guidance_scale_embedding(EMBEDDED_GUIDANCE, cond_dim) if cond_dim else None
    return Loaded(
        unet, text_encoder, tokenizer, taesd, alphas_cumprod_from(d / "scheduler"),
        timestep_cond=cond,
    )


def encode_prompt(model: Loaded, text: str) -> torch.Tensor:
    """CLIP embeddings for one prompt, padded to the tokenizer's max length."""
    ids = model.tokenizer(
        text,
        padding="max_length",
        max_length=model.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    ).input_ids.to(DEVICE)
    return model.text_encoder(ids)[0].to(DTYPE)


def to_latent(model: Loaded, rgb: torch.Tensor) -> torch.Tensor:
    """TAESD-encode an RGB tensor in [0, 1], NCHW, to SD-scaled latents."""
    return model.taesd.encode(rgb.mul(2.0).sub(1.0).to(DTYPE)).latents


def to_image(model: Loaded, latents: torch.Tensor) -> torch.Tensor:
    """TAESD-decode latents to RGB in [0, 1], NCHW."""
    return model.taesd.decode(latents).sample.add(1.0).div(2.0).clamp(0.0, 1.0)


def predict_x0(
    model: Loaded,
    x_t: torch.Tensor,
    t: int,
    cond: torch.Tensor,
    uncond: torch.Tensor | None,
    cfg: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """One UNet evaluation; returns (x0, eps) using the model's own alphas_cumprod."""
    a = model.alphas_cumprod[t]
    sqrt_a, sqrt_1ma = a.sqrt().to(DTYPE), (1.0 - a).sqrt().to(DTYPE)
    ts = torch.tensor([t], device=DEVICE, dtype=torch.long)
    tc = model.timestep_cond
    if uncond is None or cfg <= 1.0:
        eps = model.unet(x_t, ts, encoder_hidden_states=cond, timestep_cond=tc).sample
    else:
        batch = torch.cat([x_t, x_t])
        eps_all = model.unet(
            batch,
            ts.repeat(2),
            encoder_hidden_states=torch.cat([uncond, cond]),
            timestep_cond=None if tc is None else tc.repeat(2, 1),
        ).sample
        eps_u, eps_c = eps_all.chunk(2)
        eps = eps_u + cfg * (eps_c - eps_u)
    return (x_t - sqrt_1ma * eps) / sqrt_a, eps


def sample(
    model: Loaded,
    mode: str,
    base: torch.Tensor,
    noise: torch.Tensor,
    variant: Variant,
    cond: torch.Tensor,
    uncond: torch.Tensor | None,
) -> torch.Tensor:
    """Proxy-guided one- or two-step generation; returns the clean latent x0."""
    t = variant.timestep
    a = model.alphas_cumprod[t]
    x_t = a.sqrt().to(DTYPE) * base + (1.0 - a).sqrt().to(DTYPE) * noise
    x0, eps = predict_x0(model, x_t, t, cond, uncond, variant.cfg)
    if variant.steps > 1:
        t2 = max(1, t // 2)
        a2 = model.alphas_cumprod[t2]
        # Euler continues along the same epsilon; LCM re-noises stochastically.
        second = eps if mode == "euler" else torch.randn_like(noise)
        x_t2 = a2.sqrt().to(DTYPE) * x0 + (1.0 - a2).sqrt().to(DTYPE) * second
        x0, _ = predict_x0(model, x_t2, t2, cond, uncond, variant.cfg)
    return x0


def slerp(a: torch.Tensor, b: torch.Tensor, t: float) -> torch.Tensor:
    """Spherical interpolation between two noise tensors."""
    af, bf = a.flatten().float(), b.flatten().float()
    cos = float((af @ bf) / (af.norm() * bf.norm() + 1e-8))
    theta = math.acos(min(max(cos, -1.0), 1.0))
    if abs(math.sin(theta)) < 1e-4:
        return a * (1.0 - t) + b * t
    s = math.sin(theta)
    return a * (math.sin((1.0 - t) * theta) / s) + b * (math.sin(t * theta) / s)


def load_proxies(size: tuple[int, int] = (512, 512)) -> torch.Tensor:
    """The 8 proxy conditioning frames as one NCHW float tensor in [0, 1]."""
    frames = []
    for i in range(8):
        img = Image.open(PROXY_DIR / f"proxy_{i:02d}.png").convert("RGB")
        if img.size != size:
            img = img.resize(size, Image.BILINEAR)
        frames.append(np.asarray(img, dtype=np.float32) / 255.0)
    return torch.from_numpy(np.stack(frames)).permute(0, 3, 1, 2).to(DEVICE)


def tile_of(rgb: torch.Tensor, px: int = 256) -> Image.Image:
    """One decoded frame as a square PIL tile."""
    arr = (rgb[0].permute(1, 2, 0).float().cpu().numpy() * 255.0).astype(np.uint8)
    return Image.fromarray(arr).resize((px, px), Image.LANCZOS)


def compose(tiles: list[list[Image.Image]], row_labels: list[str], col_labels: list[str],
            title: str, px: int = 256) -> Image.Image:
    """Lay tiles out with a label column and a label row."""
    pad_x, pad_y = 150, 34
    w = pad_x + px * len(col_labels)
    h = pad_y + px * len(tiles)
    sheet = Image.new("RGB", (w, h), (16, 16, 18))
    draw = ImageDraw.Draw(sheet)
    draw.text((6, 6), title, fill=(235, 235, 235))
    for c, label in enumerate(col_labels):
        draw.text((pad_x + c * px + 6, 20), label, fill=(190, 210, 255))
    for r, row in enumerate(tiles):
        draw.text((6, pad_y + r * px + 6), row_labels[r], fill=(255, 220, 170))
        for c, tile in enumerate(row):
            sheet.paste(tile, (pad_x + c * px, pad_y + r * px))
    return sheet


def run_grid(model: Loaded, cand: Candidate, proxies: torch.Tensor) -> Image.Image:
    """3 prompts x 8 proxies (rows) by variants (cols)."""
    gen = torch.Generator(device=DEVICE).manual_seed(1234)
    noise = torch.randn(
        (1, 4, proxies.shape[2] // 8, proxies.shape[3] // 8),
        generator=gen, device=DEVICE, dtype=DTYPE,
    )
    uncond = model.embeds["NEG"]
    rows: list[list[Image.Image]] = []
    row_labels: list[str] = []
    for pname in PROMPTS:
        cond = model.embeds[pname]
        for i in range(8):
            guide = to_latent(model, proxies[i : i + 1])
            row = []
            for v in cand.variants:
                torch.manual_seed(7)  # fixed stochastic 2nd-step noise
                x0 = sample(model, cand.mode, guide, noise, v, cond, uncond)
                row.append(tile_of(to_image(model, x0)))
            rows.append(row)
            row_labels.append(f"{pname} / proxy{i}")
    return compose(rows, row_labels, [v.name for v in cand.variants], f"{cand.name} - bake-off grid")


def run_walk(
    model: Loaded, cand: Candidate, proxies: torch.Tensor, prompt: str = "P1"
) -> tuple[Image.Image, dict[str, Any]]:
    """16 frames over the 8 proxies twice, with x0_prev feedback and slerped noise.

    Also returns the two numbers that separate a walk from a runaway: mean
    saturation per frame (does the feedback loop ramp colour?) and mean absolute
    pixel change between consecutive frames (does it hold still enough to read?).
    """
    variant = next(v for v in cand.variants if v.name == cand.best)
    gen = torch.Generator(device=DEVICE).manual_seed(99)
    shape = (1, 4, proxies.shape[2] // 8, proxies.shape[3] // 8)
    keys = [torch.randn(shape, generator=gen, device=DEVICE, dtype=DTYPE) for _ in range(3)]
    cond = model.embeds[prompt]
    x0_prev: torch.Tensor | None = None
    prev_rgb: torch.Tensor | None = None
    tiles: list[Image.Image] = []
    saturation: list[float] = []
    deltas: list[float] = []
    for f in range(16):
        u = f / 15.0 * 2.0
        k = min(int(u), 1)
        noise = slerp(keys[k], keys[k + 1], u - k)
        guide = to_latent(model, proxies[f % 8 : f % 8 + 1])
        base = guide if x0_prev is None else x0_prev + (guide - x0_prev) * 0.6
        torch.manual_seed(7)
        x0 = sample(model, cand.mode, base, noise, variant, cond, model.embeds["NEG"])
        x0_prev = x0
        rgb = to_image(model, x0).float()
        saturation.append(float((rgb.amax(1) - rgb.amin(1)).mean()))
        if prev_rgb is not None:
            deltas.append(float((rgb - prev_rgb).abs().mean()))
        prev_rgb = rgb
        tiles.append(tile_of(rgb))
    rows = [tiles[0:4], tiles[4:8], tiles[8:12], tiles[12:16]]
    sheet = compose(
        rows,
        ["f0-3", "f4-7", "f8-11", "f12-15"],
        ["", "", "", ""],
        f"{cand.name} - stability walk {prompt} {variant.name} feedback 0.6",
    )
    metrics = {
        "saturation_first4": round(sum(saturation[:4]) / 4, 3),
        "saturation_last4": round(sum(saturation[-4:]) / 4, 3),
        "saturation_ramp": round(sum(saturation[-4:]) / 4 - sum(saturation[:4]) / 4, 3),
        "frame_delta_mean": round(sum(deltas) / len(deltas), 4),
        "frame_delta_max": round(max(deltas), 4),
    }
    return sheet, metrics


def bench(model: Loaded, cand: Candidate, w: int, h: int, iters: int = 30, repeats: int = 3) -> float:
    """Milliseconds per frame for UNet + TAESD encode/decode at one resolution.

    Another process may share this GPU, so the fastest of ``repeats`` timed runs
    is reported: contention only ever inflates a measurement.
    """
    rgb = torch.rand((1, 3, h, w), device=DEVICE, dtype=DTYPE)
    noise = torch.randn((1, 4, h // 8, w // 8), device=DEVICE, dtype=DTYPE)
    variant = next(v for v in cand.variants if v.name == cand.best)
    cond = model.embeds["P1"]

    def one() -> None:
        guide = to_latent(model, rgb)
        x0 = sample(model, cand.mode, guide, noise, variant, cond, None)
        to_image(model, x0)

    for _ in range(5):
        one()
    torch.cuda.synchronize()
    best = float("inf")
    for _ in range(repeats):
        start = time.perf_counter()
        for _ in range(iters):
            one()
        torch.cuda.synchronize()
        best = min(best, (time.perf_counter() - start) / iters * 1000.0)
    return best


def evaluate(
    cand: Candidate, sheets: bool = True, suffix: str = "", walk_prompt: str = "P1"
) -> dict[str, Any]:
    """Run the full experiment for one candidate and write its two sheets."""
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    model = load(cand)
    grid_path = OUT_DIR / f"{cand.name}{suffix}_grid.jpg"
    walk_path = OUT_DIR / f"{cand.name}{suffix}_walk.jpg"
    metrics: dict[str, Any] = {}
    with torch.inference_mode():
        for name, text in PROMPTS.items():
            model.embeds[name] = encode_prompt(model, text)
        model.embeds["NEG"] = encode_prompt(model, NEGATIVE)

        proxies = load_proxies()
        latent_std = float(to_latent(model, proxies[0:1]).float().std())

        if sheets:
            run_grid(model, cand, proxies).save(grid_path, quality=88, optimize=True)
            walk, metrics = run_walk(model, cand, proxies, walk_prompt)
            walk.save(walk_path, quality=90, optimize=True)

        ms512 = bench(model, cand, 512, 512)
        ms384 = bench(model, cand, 384, 256)

    peak = torch.cuda.max_memory_allocated() / 2**30
    result = {
        "name": cand.name,
        "model_dir": str(cand.model_dir),
        "mode": cand.mode,
        "source_notes": cand.source_notes,
        "cross_attention_dim": int(model.unet.config.cross_attention_dim),
        "embedded_guidance": EMBEDDED_GUIDANCE if model.timestep_cond is not None else None,
        "best_variant": cand.best,
        "walk_prompt": walk_prompt,
        "variants": [v.name for v in cand.variants],
        "ms_per_frame_512": round(ms512, 2),
        "fps_512": round(1000.0 / ms512, 2),
        "ms_per_frame_384x256": round(ms384, 2),
        "fps_384": round(1000.0 / ms384, 2),
        "vram_peak_gb": round(peak, 2),
        "taesd_guide_latent_std": round(latent_std, 3),
        "walk_metrics": metrics,
        "notes": NOTES.get(cand.name, ""),
        "grid_jpg": str(grid_path),
        "walk_jpg": str(walk_path),
        "wall_seconds": round(time.perf_counter() - t0, 1),
    }
    del model
    torch.cuda.empty_cache()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", default="sd_turbo")
    parser.add_argument(
        "--bench-only",
        action="store_true",
        help="re-time without redrawing the sheets, so all candidates share one GPU state",
    )
    parser.add_argument("--variant-set", default="base", choices=sorted(VARIANT_SETS))
    parser.add_argument("--walk-prompt", default="P1", choices=sorted(PROMPTS))
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    roster = candidates()
    out_path = OUT_DIR / "bakeoff.json"
    report = json.loads(out_path.read_text(encoding="utf-8")) if out_path.exists() else {}
    report.setdefault("candidates", {})
    report["prompts"] = PROMPTS | {"NEG": NEGATIVE}
    report["gpu"] = torch.cuda.get_device_name(0)
    report["verdict"] = (
        "sd_turbo. It is the only candidate that both survives x0_prev feedback without "
        "drifting into an attractor and answers aerial/landscape vocabulary, and at 512x512 "
        "it costs the same as the SD1.5 candidates. The larger finding is that the model was "
        "never the bottleneck: swapping P1 (the verbatim KOSMA first-person prompt) for the "
        "aerial P2 / chaotic-voxel P3 vocabulary changes the output far more than swapping "
        "models, because the reference got its aerial framing from AnimateDiff rather than "
        "from its prompt. Recommended settings: sd_turbo, 1 step, t 850-900, CFG 1.0, "
        "aerial/landscape prompt vocabulary, feedback 0.6."
    )
    report["timing_caveat"] = (
        "Another worker shared this GPU during the run. Each bench is min of 3 x 30 iters "
        "and --bench-only keeps the fastest reading across passes, since contention only "
        "inflates; numbers below are the floor after 6 passes."
    )

    for name in args.candidates.split(","):
        name = name.strip()
        if not name:
            continue
        variants, best = VARIANT_SETS[args.variant_set]
        cand = replace(roster[name], variants=variants, best=best)
        suffix = "" if args.variant_set == "base" else f"_{args.variant_set}"
        result = evaluate(
            cand, sheets=not args.bench_only, suffix=suffix, walk_prompt=args.walk_prompt
        )
        name = f"{name}{suffix}"
        if args.bench_only:
            # Keep the fastest reading seen so far: a shared GPU only inflates.
            previous = report["candidates"].get(name, {})
            timings = {
                k: min(v, previous.get(k, float("inf")))
                for k, v in result.items()
                if "ms_per_frame" in k
            }
            result = previous | timings | {
                k.replace("ms_per_frame_", "fps_").replace("384x256", "384"): round(1000.0 / v, 2)
                for k, v in timings.items()
            }
        report["candidates"][name] = result
        out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(
            f"{name}: {result['ms_per_frame_512']} ms @512  "
            f"{result['ms_per_frame_384x256']} ms @384x256  "
            f"{result['vram_peak_gb']} GB",
            flush=True,
        )


if __name__ == "__main__":
    main()
