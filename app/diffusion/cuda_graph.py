"""Replay warmed UNet kernels without changing their operations or precision."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class _Capture:
    graph: Any
    inputs: tuple[Any, Any, Any]
    output: Any


class UNetGraphs:
    """At most two startup captures: the current resolution with/without CFG.

    Unprepared shapes run eagerly. Capture is never attempted from a live frame,
    and callers receive independent predictions just as they do from the UNet.
    """

    def __init__(self, module: Any, torch: Any):
        self.module, self.torch = module, torch
        self._captures: dict[tuple, _Capture] = {}
        self.replay_calls = 0

    @staticmethod
    def _key(*inputs: Any) -> tuple:
        return tuple((tuple(value.shape), tuple(value.stride()), value.dtype, value.device)
                     for value in inputs)

    @property
    def count(self) -> int:
        return len(self._captures)

    def prepare(self, sample: Any, timestep: Any, embedding: Any) -> None:
        torch = self.torch
        key = self._key(sample, timestep, embedding)
        if key in self._captures:
            return
        if len(self._captures) >= 2:
            raise RuntimeError("UNet graph startup cache is full")
        with torch.inference_mode():
            inputs = tuple(value.detach().clone(memory_format=torch.preserve_format)
                           for value in (sample, timestep, embedding))
            stream = torch.cuda.Stream(device=sample.device)
            current = torch.cuda.current_stream(sample.device)
            stream.wait_stream(current)
            with torch.cuda.stream(stream):
                for _ in range(3):
                    warm_output = self.module(inputs[0], inputs[1], encoder_hidden_states=inputs[2]).sample
            current.wait_stream(stream)
            torch.cuda.synchronize(sample.device)
            del warm_output
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                output = self.module(inputs[0], inputs[1], encoder_hidden_states=inputs[2]).sample
            current.wait_stream(stream)
            torch.cuda.synchronize(sample.device)
        self._captures[key] = _Capture(graph, inputs, output)

    def predict(self, sample: Any, timestep: Any, embedding: Any) -> Any:
        capture = self._captures.get(self._key(sample, timestep, embedding))
        if capture is None:
            return self.module(sample, timestep, encoder_hidden_states=embedding).sample
        with self.torch.inference_mode():
            for destination, source in zip(capture.inputs, (sample, timestep, embedding)):
                destination.copy_(source)
            capture.graph.replay()
            self.replay_calls += 1
            # A later replay overwrites graph output storage; the sampler may
            # retain this prediction as memory or across another denoising step.
            return capture.output.clone()

    def close(self) -> None:
        if self._captures:
            self.torch.cuda.synchronize()
            self._captures.clear()
