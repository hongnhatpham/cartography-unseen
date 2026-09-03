from __future__ import annotations

from contextlib import nullcontext
from types import SimpleNamespace

import numpy as np

from app.diffusion.sd_turbo_stream import SDTurboStreamBackend, blend_correlated_noise
from app.renderer.camera import Camera
from app.types import ConditioningFrame


class FakeTorch:
    float16 = "float16"

    @staticmethod
    def inference_mode():
        return nullcontext()

    @staticmethod
    def autocast(*_args, **_kwargs):
        return nullcontext()


class FakePipeline:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []
        self.encode_calls = 0

    def encode_prompt(self, **_kwargs):
        self.encode_calls += 1
        return "positive", "negative"

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        image = kwargs["image"]
        return SimpleNamespace(images=[image])


def make_frame() -> ConditioningFrame:
    image = np.full((8, 8, 3), 96, dtype=np.uint8)
    return ConditioningFrame(
        rgb=image,
        depth=np.full((8, 8), 0.5, dtype=np.float32),
        edges=np.zeros((8, 8), dtype=np.uint8),
        camera=Camera.create_default().snapshot(),
        timestamp=0.0,
        sequence=1,
    )


def make_backend() -> tuple[SDTurboStreamBackend, FakePipeline]:
    backend = SDTurboStreamBackend()
    pipeline = FakePipeline()
    backend.pipe = pipeline
    backend.torch = FakeTorch()
    backend.prompt_embeds = "positive"
    backend.negative_prompt_embeds = "negative"
    backend.width = 8
    backend.height = 8
    backend.one_step_timestep = 600
    backend.fixed_noise = False
    return backend, pipeline


def test_cfg_uses_negative_embedding_only_above_one() -> None:
    backend, pipeline = make_backend()

    backend.set_guidance_scale(0.0)
    backend.generate(make_frame())
    assert pipeline.calls[-1]["guidance_scale"] == 0.0
    assert "negative_prompt_embeds" not in pipeline.calls[-1]

    backend.set_guidance_scale(1.25)
    backend.generate(make_frame())
    assert pipeline.calls[-1]["guidance_scale"] == 1.25
    assert pipeline.calls[-1]["negative_prompt_embeds"] == "negative"


def test_prompt_change_caches_both_cfg_embeddings_once() -> None:
    backend, pipeline = make_backend()
    backend.temporal_noise = np.ones((1, 4, 2, 2), dtype=np.float32)

    backend.set_prompt("signal city", "photorealistic")
    backend.set_prompt("signal city", "photorealistic")

    assert pipeline.encode_calls == 1
    assert backend.prompt_embeds == "positive"
    assert backend.negative_prompt_embeds == "negative"
    assert backend.temporal_noise is None


def test_temporal_noise_persistence_has_stable_endpoints() -> None:
    previous = np.full((1, 4, 2, 2), 3.0, dtype=np.float32)
    fresh = np.full((1, 4, 2, 2), 7.0, dtype=np.float32)

    assert np.array_equal(blend_correlated_noise(previous, fresh, 0.0), fresh)
    assert np.array_equal(blend_correlated_noise(previous, fresh, 1.0), previous)
