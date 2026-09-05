from queue import Queue
from typing import Any

import numpy as np

from app.diffusion.worker import DiffusionWorker
from app.types import CameraSnapshot, ConditioningFrame, GeneratedFrame


def test_prompt_settings_apply_together_without_resetting_history(monkeypatch) -> None:
    events: list[tuple[str, Any]] = []
    outputs: Queue[GeneratedFrame] = Queue()

    class Backend:
        def __init__(self) -> None:
            self.settings: dict[str, Any] = {}
            self.prompt = ""
            self.previous_frames: list[np.ndarray | None] = []

        def load(self, config):
            pass

        def warmup(self):
            pass

        def apply_settings(self, settings):
            self.settings.update(settings)
            events.append(("settings", dict(settings)))

        def set_prompt(self, prompt, negative_prompt):
            self.prompt = prompt
            events.append(("prompt", (prompt, negative_prompt, dict(self.settings))))

        def reseed(self, seed):
            events.append(("seed", seed))

        def set_resolution(self, width, height):
            events.append(("resolution", (width, height)))

        def generate(self, conditioning, previous_frame=None):
            self.previous_frames.append(previous_frame)
            events.append(("frame", (self.prompt, dict(self.settings))))
            return conditioning.rgb

        def stats(self):
            return {}

        def unload(self):
            pass

    backend = Backend()
    monkeypatch.setattr("app.diffusion.worker.create_backend", lambda _: backend)
    monkeypatch.setattr(
        "app.diffusion.worker.reproject_previous_image",
        lambda previous, *_: (previous, None),
    )
    worker = DiffusionWorker(
        "fake",
        {
            "prompt": "old landscape",
            "seed": 42,
            "feedback_reprojection": True,
            "diffusion_width": 8,
            "diffusion_height": 8,
        },
    )
    monkeypatch.setattr(worker.generated, "publish", outputs.put)
    identity = np.eye(4, dtype=np.float32)
    camera = CameraSnapshot(identity, identity, np.zeros(3), np.zeros(3))

    def publish(sequence):
        worker.publish(
            ConditioningFrame(
                np.full((8, 8, 3), sequence, dtype=np.uint8),
                np.ones((8, 8), dtype=np.float32),
                None,
                camera,
                float(sequence),
                sequence,
            )
        )
        return outputs.get(timeout=5)

    worker.start()
    try:
        first = publish(1)
        events.clear()
        settings = {
            "timestep_min": 120,
            "timestep_max": 260,
            "guidance_scale": 1.5,
            "instability": 0.2,
            "guide_strength": 0.9,
        }
        revision = worker.request_prompt("organic landscape", "text", settings=settings)
        second = publish(2)

        assert events == [
            ("settings", settings),
            ("prompt", ("organic landscape", "text", dict(backend.settings))),
            ("frame", ("organic landscape", dict(backend.settings))),
        ]
        assert second.stats["prompt_revision"] == revision == 1
        assert backend.previous_frames[1] is first.image

        # Existing prompt callers retain the current tuning and temporal history.
        events.clear()
        worker.request_prompt("next landscape")
        third = publish(3)
        assert [kind for kind, _ in events] == ["prompt", "frame"]
        assert third.stats["prompt_revision"] == 2
        assert backend.previous_frames[2] is second.image
    finally:
        worker.stop()
