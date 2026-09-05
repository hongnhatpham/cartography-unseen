# Windows input stutter, 5 September 2026

Windows input processing was stopping presentation even while AI generation
continued. The app now keeps window creation and input processing on a separate
Windows thread. Simulation and OpenGL presentation read buffered input without
waiting for that thread. Image resolution, model settings, shaders, geometry,
fog, trail and prompt behavior are unchanged.

## Evidence for the cause

A native stack sampler captured 44 samples across six input waits lasting
58–109 ms. Every sample was inside `win32u!NtUserPeekMessage`, reached through
`USER32!PeekMessageW`, SDL2 and pygame's event wait. The AI worker continued
through inference and decoding during these waits. These samples did not show
the display thread waiting for Python's execution lock or an OpenGL fence.

A minimal fullscreen window reproduced the problem without AI or scene geometry.
Its worst display interval was 102.42 ms, with 102.03 ms spent in input polling.
Drawing and swapping took at most 2.88 ms. A newer isolated pygame-ce 2.5.8 runtime
with SDL 2.32.10 still paused in input processing, so the pinned pygame 2.6.1
dependency was retained.

This identifies the blocked syscall, not the Windows component or external
process responsible for its delay. Wait Chain Traversal did not capture one of
the long waits. No other applications, drivers or system settings were changed.

## Implementation

`app/window_loop.py` creates the SDL window and pumps events on its owner thread.
It transfers the OpenGL context to the existing rendering thread. Mouse movement
accumulates until consumed, held-key state uses the latest snapshot, and noisy
window/mouse events coalesce. Control events have a bounded queue with explicit
overflow reporting. Quit remains available independently of that queue.

Pygame's automatic resize watcher is disabled because it would make the context
current on the input thread. The renderer already sets its own viewport.
Fullscreen changes use `SDL_SetWindowFullscreen`, avoiding pygame's wrapper
that also moves the context. Window, cursor and text-entry commands run on the
owner thread. Shutdown releases GL before that thread destroys the window.
The owner samples input at 60 Hz, matching the original exhibition cadence.
Other platforms retain the previous synchronous input path.

## Reproduction and comparison

The fixed-route comparison runs real AI generation at 384×256, CFG 1.5, one step,
with identical seeds, prompt and settings. Timing and movement start at the first
AI publication. Collision queries execute, but their displacement is overridden
to keep the synthetic route independent of timing. No native stack sampler or
GPU telemetry subprocess runs during these comparisons.

```powershell
runtime/python/python.exe tools/replay_performance.py --seconds 120 --max-stall-ms 100
```

The command fails when a display interval exceeds 100 ms. Raw CSV timings,
configuration and actual GL renderer metadata are saved under `logs/performance`.
All compared runs used Intel Arc Pro OpenGL and NVIDIA AI generation.

| Test | Worst input wait | Worst display interval | Display intervals above 50 ms |
|---|---:|---:|---:|
| Empty window, original thread arrangement, 120 s | 102.03 ms | 102.42 ms | 5 |
| Separate threads, three injected 100 ms pauses, 20 s | 100.4 ms | 21.96 ms | 0 |
| Separate threads, natural input waits, 120 s | 65.25 ms | 22.81 ms | 0 |

During the two natural 60 and 65 ms input waits in the separated test, overlapping
display intervals remained below 23 ms. This establishes that the change removes
the dependency, rather than relying on a run in which Windows happened not to stall.

### Real AI comparisons

| 120-second run | Display >100 ms | Display >50 ms | Worst display interval | AI FPS |
|---|---:|---:|---:|---:|
| Direct input, second baseline | 4 | 20 | 194.56 ms | 11.26 |
| Separated input at 60 Hz | 1 | 11 | 113.71 ms | 10.58 |
| Final direct input | 6 | 16 | 157.41 ms | 8.79 |
| Final separated input at 60 Hz | 3 | 15 | 127.41 ms | 8.06 |

The final pair has identical recorded SHA-256 hashes for every production Python
and shader file, and identical configurations. The baseline disables only the
new owner-thread factory, exercising the retained direct input path. In that
pair, the display-side input poll maximum fell from 109.38 to 1.04 ms. Long pauses
were halved, but the strict 100 ms maximum threshold still fails because pauses
remain in other work. The 99th-percentile display interval was 39 ms in both runs.

AI throughput was 6–8% lower in these separated runs. The tests establish the
display/input improvement, not an increase in AI throughput. A temporary 120 Hz
input cadence performed worse and was reduced to the original 60 Hz cadence.
No model, precision, generation resolution or image settings were reduced.

An unrelated surface-material edit arrived during the earlier experiments.
The first baseline and first threaded run straddle that edit and are excluded
from this comparison. The material edit is preserved.

Raw artifacts are in `logs/performance/goal-replay-direct-2`,
`goal-replay-threaded-60hz`, `goal-final-direct` and `goal-final-threaded`.
The final source-matched pair can be reproduced with the diagnostic
`logs/performance/input_mode_replay.py`, passing `direct` or `threaded`, the
duration, and a new output directory.

## Validation and limits

The full existing suite passed 225 tests after integration, including GPU
generation and navigation. Focused lifecycle and mailbox tests cover input
accumulation, event floods, initialization failure, readiness timeout, native
input failure, pending commands and shutdown. A real GL regression test also
blocks the input owner while presenting multiple images, then exercises
fullscreen, windowed mode and text-entry transitions.

Exact proxy RGB, depth, packed geometry and collider hashes matched at three
positions across horizontal and vertical chunk boundaries with direct and
separated window ownership. The results are stored in
`logs/performance/window-pixel-comparison.json`.

### Five-minute normal-operation check

The final build completed 300 seconds using a copy of the user's current settings,
including the open diagnostics overlay, automatic prompt variation and regional
palette updates. It processed 27 prompt requests and published 1,919 AI frames
without worker errors or resolution fallback. Average rates over the whole run
were 53.5 display FPS and 6.4 AI FPS.

There were still 16 display intervals above 100 ms, with a maximum of 337.36 ms.
The display-side input poll remained below 1.01 ms even though the owner thread
encountered 30 native waits above 50 ms. This is a functional stress check with
different settings, route and monitoring overhead, not a matched performance
comparison. It does not establish that all normal-operation stutter is fixed.

Process RSS went from 1,677 to 1,701 MB as new prompts populated the embedding
cache, with a 1,718 MB peak. Five minutes is insufficient to establish a memory
plateau or week-long stability. Raw results are in
`logs/performance/goal-normal-confirmation`.

Windows can still delay delivery of new input during a native wait. Rendering
and autorun continue using the latest state. GPU work, world construction and
system scheduling can still cause other pauses. Short comparison runs do not
prove a week of uninterrupted operation.

Relevant sources: [pygame 2.6.1 display.c](https://github.com/pygame/pygame/blob/2.6.1/src_c/display.c),
[pygame 2.6.1 event.c](https://github.com/pygame/pygame/blob/2.6.1/src_c/event.c),
[SDL 2.28.4 Windows event loop](https://github.com/libsdl-org/SDL/blob/release-2.28.4/src/video/windows/SDL_windowsevents.c).
