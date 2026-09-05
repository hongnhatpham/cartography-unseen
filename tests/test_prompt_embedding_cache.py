"""Prompt changes reuse exact embeddings without retaining an endless library."""
from contextlib import nullcontext
from types import SimpleNamespace

import numpy as np

from app.diffusion.latent_walk import LatentWalkBackend


class Tensor(np.ndarray):
    def to(self, *_args):
        return self


def backend_with_encoder():
    backend = LatentWalkBackend()
    calls = []

    class Tokenizer:
        model_max_length = 77

        def __call__(self, text, **kwargs):
            calls.append(text)
            values = np.array([len(text), sum(map(ord, text))], dtype=np.float32)
            return SimpleNamespace(input_ids=values.view(Tensor))

    backend.tokenizer = Tokenizer()
    backend.text_encoder = lambda ids: (ids,)
    backend.torch = SimpleNamespace(
        inference_mode=nullcontext,
        cuda=SimpleNamespace(is_available=lambda: False),
    )
    return backend, calls


def test_prompt_rotation_reuses_negative_and_returning_positive():
    backend, calls = backend_with_encoder()
    backend.set_prompt("coral", "unchanged negative")
    original = backend.prompt_walk.positive
    backend.set_prompt("ribbons", "unchanged negative")
    backend.set_prompt("coral", "unchanged negative")

    assert calls == ["coral", "unchanged negative", "ribbons"]
    assert backend.prompt_walk.positive is original


def test_prompt_cache_is_bounded_and_unload_releases_embeddings():
    backend, calls = backend_with_encoder()
    for index in range(150):
        backend.set_prompt(f"landscape {index}", "negative")

    assert len(backend._prompt_embeddings) <= 64
    assert calls.count("negative") == 1
    backend.set_prompt("landscape 0", "negative")
    assert calls.count("landscape 0") == 2
    backend.unload()
    assert not backend._prompt_embeddings
