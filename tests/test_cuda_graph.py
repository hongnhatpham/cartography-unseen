from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from app.diffusion.cuda_graph import UNetGraphs
from app.diffusion.latent_walk import LatentWalkBackend, PromptWalk


def test_uncaptured_inputs_use_eager_without_preparing_graphs():
    calls = []
    module = lambda *args, **kwargs: calls.append((args, kwargs)) or SimpleNamespace(sample="eager")
    graphs = UNetGraphs(module, None)
    value = SimpleNamespace(shape=(1,), stride=lambda: (1,), dtype="float16", device="cuda")
    assert graphs.predict(value, value, value) == "eager"
    assert len(calls) == 1
    assert graphs.count == graphs.replay_calls == 0


def test_resolution_releases_graphs_only_when_dimensions_change():
    backend = LatentWalkBackend()
    closed = []
    graphs = SimpleNamespace(close=lambda: closed.append(True))
    backend._unet_graphs = graphs
    backend.set_resolution(512, 512)
    assert backend._unet_graphs is graphs
    assert not closed
    backend.set_resolution(384, 256)
    assert backend._unet_graphs is None
    assert closed == [True]


@pytest.mark.parametrize("failure", [RuntimeError("capture unsupported"), RuntimeError("CUDA out of memory")])
def test_capture_failure_keeps_eager_backend_and_releases_partial_graphs(monkeypatch, failure):
    backend = LatentWalkBackend()
    tensor = SimpleNamespace(contiguous=lambda **kwargs: "sample")
    freed, emptied = [], []
    backend.torch = SimpleNamespace(inference_mode=nullcontext, zeros=lambda *a, **k: tensor,
                                    channels_last="channels_last", full=lambda *a, **k: "time",
                                    long="long", cuda=SimpleNamespace(empty_cache=lambda: emptied.append(True)))
    backend.unet = SimpleNamespace(config=SimpleNamespace(in_channels=4))
    backend.prompt_walk.positive = "embedding"

    class FailingGraphs:
        def __init__(self, *args):
            pass

        def prepare(self, *args):
            raise failure

        def close(self):
            freed.append(True)

    monkeypatch.setattr("app.diffusion.latent_walk.UNetGraphs", FailingGraphs)
    backend._prepare_unet_graphs()
    assert backend._unet_graphs is None
    assert freed == emptied == [True]


def test_graph_preserves_full_image_sequence_across_live_settings():
    torch = pytest.importorskip("torch")
    root = Path(__file__).resolve().parents[1]
    if not torch.cuda.is_available() or not (root / "models/sd_turbo/unet").exists():
        pytest.skip("CUDA and local models required")
    backend = LatentWalkBackend()
    backend.load(dict(project_root=root, diffusion_width=384, diffusion_height=256,
                      prompt="organic lattice in a foggy courtyard", negative_prompt="text, ui",
                      guidance_scale=1.5, warmup_passes=1, seed=713, noise_jitter=.12))
    graphs = None
    try:
        backend.warmup()
        graphs = backend._unet_graphs
        assert graphs is not None and graphs.count == 2
        now = [0.0]
        pixels = np.random.default_rng(713).integers(0, 256, size=(256, 384, 3), dtype=np.uint8)

        def sequence(use_graph):
            backend._unet_graphs = graphs if use_graph else None
            now[0] = 0.0
            backend.set_clock(lambda: now[0])
            backend.prompt_walk = PromptWalk(backend._read_clock)
            backend.prompt_walk.seconds = 6
            backend.reseed(713)
            backend.set_prompt("organic lattice in a foggy courtyard", "text, ui")
            output = []
            for index, cfg in enumerate((1.0, 1.5, 1.5, .5, 1.5)):
                now[0] = index * 2.5
                backend.apply_settings(dict(guidance_scale=cfg, timestep_min=80 + index * 7,
                                            timestep_max=280 + index * 11, guide_strength=.8 + index * .03,
                                            instability=index * .05, steps=2 if index == 3 else 1))
                if index == 2:
                    backend.set_prompt("a distant figure among curved roots", "text, ui")
                output.append(backend._render(np.roll(pixels, index * 3, axis=1)).copy())
            return output

        before = torch.cuda.get_rng_state().clone()
        eager = sequence(False)
        captured = sequence(True)
        assert torch.equal(before, torch.cuda.get_rng_state())
        assert graphs.replay_calls >= 8
        for original, replayed in zip(eager, captured):
            assert np.array_equal(original, replayed)
        backend.set_resolution(384, 256)
        assert backend._unet_graphs is graphs
    finally:
        backend._unet_graphs = graphs
        backend.unload()
    assert graphs is None or graphs.count == 0
