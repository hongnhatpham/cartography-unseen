# Stutter investigation and fixes, 5 September 2026

The later [Windows input investigation](window-input-stutter-20260905.md)
identified native `PeekMessage` waits and separated input from rendering.
The results below document the earlier pass before that change.

This pass reduces retained world objects, spreads expensive chunk construction
across frames, and lets AI work continue during Windows input waits. It preserves
generation resolution, model precision, prompt settings, geometry and collision
data. It does **not** eliminate every display pause on this computer.

## Changes supported by measurements

### World construction and garbage collection

The renderer retained each source `WorldChunk` after packing its shapes into
numeric render and collision arrays. A second cache retained up to 2,662 source
chunks. Both kept Python objects alive that rendering no longer needed.

The renderer now uses its packed arrays to track the active window. The source
cache retains 128 recent chunks for nearby generation queries. A CPU probe of
the same 2,662 chunks reduced retained cubes/forms from 38,281 to 1,964, about
95 percent. Median explicit full-GC time fell from 34.0 to 20.4 ms in that probe.
These timings are component measurements, not a guarantee about live GC pauses.

Chunk construction previously always attempted three chunks per update. A
replay reached 48.6 ms for one update. It now yields after three milliseconds of
construction or three chunks, whichever comes first. One chunk always completes,
so unusually expensive individual chunks can exceed that budget. Planning,
rebasing and uploads also lie outside this construction budget.

The full 11x11x11 window remains. Startup fills it before showing the world.
Revisiting an evicted region regenerates identical geometry instead of retaining
all its source objects. The probe measured about 1.22 ms per regenerated chunk;
the frame budget limits how many are rebuilt together.

Actual proxy RGB, depth, packed geometry and collision hashes matched exactly
before and after at three sampled positions across horizontal/vertical chunk
boundaries. Existing streaming tests also cover remote positive/negative
altitudes, partial loading and revisits.

### Input processing

Pygame's default initialization enabled unused joystick and audio support.
Starting only display and font support reduced an isolated input probe's worst
poll from 29.3 to 0.47 ms. Longer AI runs still exhibited native input waits,
so subsystem initialization was not the entire cause.

Splitting polling into an OS pump and queue read located residual waits inside
`SDL_PumpEvents`, with near-zero measured thread CPU time. Queue reads remained
below 0.5 ms. The pinned Pygame 2.6.1 C source also shows that `event.get()` and
`event.pump()` retain Python's GIL during pumping, while `event.wait()` releases
it. A blocked pump can therefore prevent the AI worker from submitting work.

Production polling now calls `event.wait(1)` and drains remaining events using
`event.get(pump=False)`. The consumed first event is restored to the front of
the result, preserving input order. The one-millisecond timeout bounds the
requested idle wait; it cannot cap a delay inside the native Windows pump.
Prompt text composition is enabled only when the editor opens.

The final trace recorded five AI generations completing during native input
waits of 57 to 75 ms. This confirms that the worker can progress during those
waits. Display pauses remain and should not be described as fixed completely.

Source reference: [Pygame 2.6.1 event.c](https://github.com/pygame/pygame/blob/2.6.1/src_c/event.c),
functions `_pg_event_pump`, `pg_event_pump`, `pg_event_wait`.

### Trail compatibility

Full-suite GPU checks exposed a native line-width cap of ten pixels on one
graphics context. The requested fifteen-pixel trail now uses screen-space
triangles with near-plane clipping. Width, fading, depth occlusion and the
AI-only screenshot checks pass, including both endpoint orders for a segment
crossing behind the camera. The trail still does not enter AI conditioning.

## Live results and limits

| Run | Display FPS | AI FPS | Longest display interval | Intervals above 100 ms |
|---|---:|---:|---:|---:|
| Before fixes, fixed settings, 120 seconds | 58.85 | 8.37 | 131 ms | 5 |
| Intermediate normal operation, 300 seconds | 59.46 | 10.90 | 333 ms | 9 |
| Final input path, fixed settings, 180 seconds | 58.83 | 10.59 | 707 ms | 8 |

All three runs completed without worker errors or resolution fallback. The
intermediate run included three screenshot key events; the final run included
one. Input was preserved throughout monitoring. These are live observations,
not perfectly matched performance benchmarks: routes, input and machine load
varied. They do not establish improved worst-case display latency. The 707 ms
interval was outside the measured render/input stages and remains unexplained.

A separate short moving replay improved from a 119 ms maximum display interval
to 21 ms after the streaming/retention/initialization changes. Its event driver
differs from normal gameplay, so it cannot establish normal input performance.

The final run's display p95/p99 were 26/36 ms. Slow-frame tracing should stay
available for further investigation, particularly Windows/native waits and
work outside the currently instrumented stages. A full week on the exhibition
computer remains necessary to assess unattended reliability.

## Reproduction and evidence

```powershell
runtime/python/python.exe tools/soak_test.py --minutes 30
```

Add `--fixed-settings` to hold sampler settings and the initial AI prompt
constant. Normal monitoring preserves production input handling. `--split-poll`
explicitly replaces it with separate pump/queue probes for diagnosis only.
The monitor now ships in exhibition packages.

Raw local artifacts under `logs/performance/` are ignored by Git:

- `gc_retention_probe.py` and `.json`: retention, GC timings and exact geometry hash.
- `stutter-geometry-comparison.json`: real GPU pixel/depth and collision comparison.
- `stutter-baseline-short.json`, `stutter-fixed-short.json`: short moving replay.
- `stutter-trace-before/`: fixed-settings baseline.
- `stutter-confirm-five-minutes/`: normal operation, including three screenshot requests.
- `stutter-split-poll/`, `stutter-text-input-off/`: native input isolation probes.
- `stutter-yielding-input/`: final production input path.

The focused streaming and input tests were run failing before their fixes.
Trail GPU tests reproduced the ten-pixel cap and pass with triangle rendering.
All 211 tests passed across the full-suite run and focused reruns after updating
the navigation tests' event-provider fixtures. The tests include real GPU
generation, screenshots, prompt editing, manual/idle flight and trail clipping.
