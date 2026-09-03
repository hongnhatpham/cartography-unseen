from __future__ import annotations

import math
import secrets
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
from PIL import Image, ImageFilter

from app.diffusion.base import DiffusionBackend
from app.types import ConditioningFrame
from app.utils.timing import ExponentialAverage


def classifier_free_guidance_enabled(guidance_scale: float) -> bool:
    """Diffusers enables CFG only when guidance is greater than one."""
    return guidance_scale > 1.0


def blend_correlated_noise(previous: Any, fresh: Any, persistence: float) -> Any:
    """Blend fresh noise into a persistent field without changing its variance."""
    if previous is None or previous.shape != fresh.shape:
        return fresh
    persistence = min(max(float(persistence), 0.0), 1.0)
    fresh_weight = math.sqrt(max(0.0, 1.0 - persistence * persistence))
    return previous * persistence + fresh * fresh_weight


class SDTurboStreamBackend(DiffusionBackend):
    """Resident SD-Turbo img2img backend for a latest-frame streaming worker.

    ``steps`` means effective denoising/UNet evaluations. Diffusers removes early
    timesteps according to img2img strength, so a larger scheduler step count is
    requested when needed to retain one effective step at strengths below 1.0.
    """

    def __init__(self) -> None:
        self.pipe: Any = None
        self.torch: Any = None
        self.device = "cuda"
        self.width = 512
        self.height = 512
        self.steps = 1
        self.guidance_scale = 0.0
        self.schedule_steps = 1
        self.strength = 0.45
        self.one_step_timestep = 0
        self.fixed_noise = True
        self.previous_frame_weight = 0.0
        self.noise_persistence = 0.975
        self.temporal_noise: Any = None
        self.noise_generator: Any = None
        self._default_prepare_latents: Any = None
        self.edge_strength = 0.0
        self.edge_softness = 1.5
        self.seed = 12345
        self.seed_mode = "fixed"
        self.active_seed = 12345
        self.generator: Any = None
        self.prompt = ""
        self.negative_prompt = ""
        self.prompt_embeds: Any = None
        self.negative_prompt_embeds: Any = None
        self.inference_average = ExponentialAverage(alpha=0.2)
        self.copy_in_average = ExponentialAverage(alpha=0.2)
        self.copy_out_average = ExponentialAverage(alpha=0.2)
        self.load_ms = 0.0
        self.first_frame_ms = 0.0
        self.frames = 0
        self.warmup_passes = 3

    def load(self, config: dict[str, Any]) -> None:
        started = perf_counter()
        try:
            import torch
            from diffusers import StableDiffusionImg2ImgPipeline
        except ImportError as exc:
            raise RuntimeError(
                "SD-Turbo dependencies are missing. Prepare the bundled runtime first."
            ) from exc
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA-compatible NVIDIA GPU/driver not available")

        model_path = Path(config["model_path"])
        required = ("model_index.json", "unet", "vae", "tokenizer", "text_encoder")
        missing = [name for name in required if not (model_path / name).exists()]
        if missing:
            raise RuntimeError(
                f"SD-Turbo model incomplete at {model_path}. Missing: {', '.join(missing)}. "
                "Run tools\\prepare_models.ps1 on the development PC."
            )

        self.torch = torch
        self.width = int(config.get("diffusion_width", config["diffusion_resolution"]))
        self.height = int(config.get("diffusion_height", self.width))
        self.steps = int(config.get("steps", 1))
        self.guidance_scale = float(config.get("guidance_scale", 0.0))
        self.strength = float(config.get("img2img_strength", 0.45))
        self.one_step_timestep = int(config.get("one_step_timestep", 0))
        self.fixed_noise = bool(config.get("fixed_noise", True))
        self.previous_frame_weight = float(config.get("previous_frame_weight", 0.0))
        self.noise_persistence = float(config.get("noise_persistence", 0.975))
        self.edge_strength = float(config.get("edge_strength", 0.0))
        self.edge_softness = float(config.get("edge_softness", 1.5))
        self.seed = int(config.get("seed", 12345))
        self.seed_mode = str(config.get("seed_mode", "fixed"))
        self.active_seed = self.seed
        self.warmup_passes = int(config.get("warmup_passes", 3))
        self.schedule_steps = (
            1
            if self.steps == 1 and self.one_step_timestep > 0
            else max(self.steps, math.ceil(self.steps / self.strength))
        )

        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_grad_enabled(False)
        self.pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
            str(model_path),
            torch_dtype=torch.float16,
            variant="fp16",
            local_files_only=True,
            safety_checker=None,
            requires_safety_checker=False,
        )
        self.pipe.set_progress_bar_config(disable=True)
        self.pipe.to(self.device)
        self.pipe.unet.eval()
        self.pipe.vae.eval()
        if hasattr(self.pipe, "enable_vae_slicing"):
            self.pipe.enable_vae_slicing()
        if bool(config.get("torch_compile", False)) and hasattr(torch, "compile"):
            self.pipe.unet = torch.compile(
                self.pipe.unet, mode="reduce-overhead", fullgraph=True
            )
        self.generator = torch.Generator(device=self.device).manual_seed(self.seed)
        self.noise_generator = torch.Generator(device=self.device).manual_seed(self.seed + 1)
        self._default_prepare_latents = self.pipe.prepare_latents
        self.pipe.prepare_latents = self._prepare_latents
        self.set_prompt(str(config["prompt"]), str(config.get("negative_prompt", "")))
        self.load_ms = (perf_counter() - started) * 1000.0

    def set_prompt(self, prompt: str, negative_prompt: str = "") -> None:
        if self.pipe is None:
            self.prompt = prompt
            self.negative_prompt = negative_prompt
            return
        if (
            prompt == self.prompt
            and negative_prompt == self.negative_prompt
            and self.prompt_embeds is not None
            and self.negative_prompt_embeds is not None
        ):
            return
        self.temporal_noise = None
        # Cache both embeddings so C can cross the CFG threshold without a
        # visible text-encoder pause. Diffusers ignores the negative embedding
        # on its guidance <= 1 fast path.
        do_cfg = True
        with self.torch.inference_mode():
            encoded = self.pipe.encode_prompt(
                prompt=prompt,
                device=self.device,
                num_images_per_prompt=1,
                do_classifier_free_guidance=do_cfg,
                negative_prompt=negative_prompt or None,
            )
        if isinstance(encoded, tuple):
            self.prompt_embeds = encoded[0]
            self.negative_prompt_embeds = encoded[1] if len(encoded) > 1 else None
        else:
            self.prompt_embeds = encoded
            self.negative_prompt_embeds = None
        self.prompt = prompt
        self.negative_prompt = negative_prompt

    def _prepare_latents(
        self,
        image: Any,
        timestep: Any,
        batch_size: int,
        num_images_per_prompt: int,
        dtype: Any,
        device: Any,
        generator: Any = None,
    ) -> Any:
        """Use slowly changing diffusion noise for the latent-walk mode."""
        if self.seed_mode != "drift":
            return self._default_prepare_latents(
                image,
                timestep,
                batch_size,
                num_images_per_prompt,
                dtype,
                device,
                generator,
            )

        from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_img2img import (
            retrieve_latents,
        )
        from diffusers.utils.torch_utils import randn_tensor

        image = image.to(device=device, dtype=dtype)
        effective_batch = batch_size * num_images_per_prompt
        if image.shape[1] == 4:
            init_latents = image
        else:
            # The posterior mode avoids unrelated VAE sampling noise. Variation
            # comes only from the correlated diffusion-noise walk below.
            init_latents = retrieve_latents(self.pipe.vae.encode(image), sample_mode="argmax")
            init_latents = self.pipe.vae.config.scaling_factor * init_latents
        if effective_batch > init_latents.shape[0]:
            if effective_batch % init_latents.shape[0] != 0:
                raise ValueError("Image batch cannot be expanded to the requested prompt batch")
            init_latents = self.torch.cat(
                [init_latents] * (effective_batch // init_latents.shape[0]), dim=0
            )

        fresh_noise = randn_tensor(
            init_latents.shape,
            generator=self.noise_generator,
            device=device,
            dtype=dtype,
        )
        noise = blend_correlated_noise(
            self.temporal_noise, fresh_noise, self.noise_persistence
        )
        self.temporal_noise = noise.detach()
        return self.pipe.scheduler.add_noise(init_latents, noise, timestep)

    def warmup(self) -> None:
        from app.renderer.camera import Camera

        width, height = self.width, self.height
        yy, xx = np.mgrid[0:height, 0:width]
        rgb = np.empty((height, width, 3), dtype=np.uint8)
        rgb[:, :, 0] = np.clip(40 + xx * 150 / width, 0, 255)
        rgb[:, :, 1] = np.clip(60 + yy * 130 / height, 0, 255)
        rgb[:, :, 2] = 110
        depth = np.full((height, width), 0.5, dtype=np.float32)
        edges = np.zeros((height, width), dtype=np.uint8)
        frame = ConditioningFrame(rgb, depth, edges, Camera.create_default().snapshot(), 0.0, 0)
        for _ in range(max(0, self.warmup_passes)):
            self.generate(frame)
        self.frames = 0
        self.inference_average = ExponentialAverage(alpha=0.2)

    def generate(
        self,
        conditioning: ConditioningFrame,
        previous_frame: np.ndarray | None = None,
        temporal_state: dict[str, Any] | None = None,
    ) -> np.ndarray:
        if self.pipe is None:
            raise RuntimeError("SD-Turbo backend has not been loaded")
        copy_started = perf_counter()
        source = conditioning.rgb
        if self.edge_strength > 0.0:
            edge_image = Image.fromarray(conditioning.edges, mode="L")
            if self.edge_softness > 0.0:
                edge_image = edge_image.filter(ImageFilter.GaussianBlur(self.edge_softness))
            edge = np.asarray(edge_image, dtype=np.float32)[:, :, None] / 255.0
            edge_darkening = 1.0 - edge * min(self.edge_strength, 1.0) * 0.72
            source = np.clip(source.astype(np.float32) * edge_darkening, 0, 255).astype(np.uint8)
        image = Image.fromarray(source, mode="RGB")
        if image.size != (self.width, self.height):
            image = image.resize((self.width, self.height), Image.Resampling.BILINEAR)
        if previous_frame is not None and self.previous_frame_weight > 0.0:
            previous = Image.fromarray(previous_frame, mode="RGB")
            if previous.size != image.size:
                previous = previous.resize(image.size, Image.Resampling.BILINEAR)
            current_array = np.asarray(image, dtype=np.float32)
            previous_array = np.asarray(previous, dtype=np.float32)
            confidence = None if temporal_state is None else temporal_state.get("confidence")
            if confidence is None:
                alpha: float | np.ndarray = self.previous_frame_weight
            else:
                alpha = np.asarray(confidence, dtype=np.float32)[:, :, None]
                alpha = np.clip(alpha * self.previous_frame_weight, 0.0, 1.0)
            blended = current_array * (1.0 - alpha) + previous_array * alpha
            image = Image.fromarray(np.clip(blended, 0, 255).astype(np.uint8), mode="RGB")
        self.copy_in_average.update((perf_counter() - copy_started) * 1000.0)

        started = perf_counter()
        custom_one_step = self.steps == 1 and self.one_step_timestep > 0
        kwargs: dict[str, Any] = {
            "image": image,
            "prompt_embeds": self.prompt_embeds,
            # A custom one-item schedule must retain strength=1 or Diffusers
            # truncates its only timestep. Multi-step mode uses the adjustable
            # img2img strength to preserve the proxy composition.
            "strength": 1.0 if custom_one_step else self.strength,
            "guidance_scale": self.guidance_scale,
            "generator": self.generator,
            "output_type": "pil",
        }
        if classifier_free_guidance_enabled(self.guidance_scale):
            kwargs["negative_prompt_embeds"] = self.negative_prompt_embeds
        if custom_one_step:
            kwargs["timesteps"] = [self.one_step_timestep]
        else:
            kwargs["num_inference_steps"] = self.schedule_steps
        if self.seed_mode == "random_each_frame":
            self.active_seed = secrets.randbelow(2**31)
            self.generator.manual_seed(self.active_seed)
        elif self.fixed_noise:
            # Reusing the same noise field makes adjacent camera views respond
            # coherently instead of receiving unrelated random texture each frame.
            self.active_seed = self.seed
            self.generator.manual_seed(self.seed)
        with self.torch.inference_mode(), self.torch.autocast("cuda", dtype=self.torch.float16):
            result = self.pipe(**kwargs).images[0]
        # CUDA work must complete before the PIL result exists, so this records actual latency.
        inference_ms = (perf_counter() - started) * 1000.0
        self.inference_average.update(inference_ms)
        if self.frames == 0:
            self.first_frame_ms = inference_ms
        output_started = perf_counter()
        output = np.asarray(result.convert("RGB"), dtype=np.uint8).copy()
        self.copy_out_average.update((perf_counter() - output_started) * 1000.0)
        self.frames += 1
        return output

    def stats(self) -> dict[str, float | int | str]:
        allocated = reserved = peak = 0.0
        if self.torch is not None and self.torch.cuda.is_available():
            scale = 1024.0**3
            allocated = self.torch.cuda.memory_allocated() / scale
            reserved = self.torch.cuda.memory_reserved() / scale
            peak = self.torch.cuda.max_memory_allocated() / scale
        return {
            "backend": "sd_turbo_stream",
            "inference_ms": self.inference_average.value,
            "copy_in_ms": self.copy_in_average.value,
            "copy_out_ms": self.copy_out_average.value,
            "vram_allocated_gb": allocated,
            "vram_reserved_gb": reserved,
            "peak_vram_gb": peak,
            "resolution": f"{self.width}x{self.height}",
            "resolution_width": self.width,
            "resolution_height": self.height,
            "steps": self.steps,
            "guidance_scale": self.guidance_scale,
            "schedule_steps": self.schedule_steps,
            "one_step_timestep": self.one_step_timestep,
            "img2img_strength": self.strength,
            "fixed_noise": str(self.fixed_noise),
            "noise_persistence": self.noise_persistence,
            "seed_mode": self.seed_mode,
            "active_seed": self.active_seed,
            "load_ms": self.load_ms,
            "first_frame_ms": self.first_frame_ms,
        }

    def set_resolution(self, width: int, height: int | None = None) -> None:
        self.width = int(width)
        self.height = int(height if height is not None else width)
        self.temporal_noise = None

    def reseed(self, seed: int) -> None:
        self.seed = seed
        if self.torch is not None:
            self.generator = self.torch.Generator(device=self.device).manual_seed(seed)
            self.noise_generator = self.torch.Generator(device=self.device).manual_seed(seed + 1)
        self.temporal_noise = None

    def set_seed_mode(self, mode: str) -> None:
        if mode not in ("fixed", "drift", "random_each_frame"):
            raise RuntimeError(f"Unsupported seed mode: {mode}")
        if mode != self.seed_mode:
            self.temporal_noise = None
        self.seed_mode = mode

    def set_steps(self, steps: int) -> None:
        self.steps = max(1, min(4, int(steps)))
        self.schedule_steps = (
            1
            if self.steps == 1 and self.one_step_timestep > 0
            else max(self.steps, math.ceil(self.steps / self.strength))
        )

    def set_guidance_scale(self, guidance_scale: float) -> None:
        self.guidance_scale = min(max(float(guidance_scale), 0.0), 4.0)

    def set_one_step_timestep(self, timestep: int) -> None:
        self.one_step_timestep = max(250, min(900, int(timestep)))

    def set_edge_softness(self, softness: float) -> None:
        self.edge_softness = max(0.0, min(4.0, float(softness)))

    def set_img2img_strength(self, strength: float) -> None:
        self.strength = max(0.25, min(1.0, float(strength)))
        self.schedule_steps = (
            1
            if self.steps == 1 and self.one_step_timestep > 0
            else max(self.steps, math.ceil(self.steps / self.strength))
        )

    def set_edge_strength(self, strength: float) -> None:
        self.edge_strength = max(0.0, min(1.0, float(strength)))

    def set_noise_persistence(self, persistence: float) -> None:
        self.noise_persistence = max(0.0, min(1.0, float(persistence)))

    def unload(self) -> None:
        if self.pipe is not None:
            del self.pipe
            self.pipe = None
        self.prompt_embeds = None
        self.negative_prompt_embeds = None
        if self.torch is not None and self.torch.cuda.is_available():
            self.torch.cuda.empty_cache()
