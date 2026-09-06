# Growing journey map performance, 6 September 2026

The archive and geometry fixes passed the controlled large-map test. The growth
run was stopped cleanly after 52.59 minutes at the user's request, once enough
data was available. It was not a completed two-hour test.

[Published performance report](https://reports.bynhat.com/r/f9995ee389c4a2672c7a/).

## Problem and fix

The original archive thread encoded the entire journey as JSON after each
accepted image and checkpoint. Python held the gameplay process's interpreter
lock during that work. As the history grew, saving caused longer display gaps.

Moving encoding into a process removed most of the interference, but transferring
a full snapshot still required the parent to serialize several megabytes. A
148.69 ms display gap overlapped 98.96 ms of parent serialization in a controlled
large-map trace. Shallow snapshot creation stayed below 1 ms in that trace.

The archive now uses one persistent worker process. Live jobs send incremental
updates through the same protocol as the map window. The worker keeps the full
history and writes the original PNG before replacing its manifest. One pending
job limits the queue. The parent retains its immutable snapshot and pixels for
recovery; every retry sends a full reset, including after worker death.

Space and quit still wait for complete exports. These explicit saves may send
the full history. They do not change the worker's live state, so a failed final
export cannot add completion metadata to later live checkpoints.

The map caches immutable image corners and reuses surviving plane buffers.
Selection, image order, original pixel quality, 512 selected planes and the
256-texture budget remain unchanged. Route buffers still update with new paths.

## Interrupted baseline

The requested baseline was stopped after 4,705.83 measured seconds, or 78.43
minutes. It was not a completed two-hour test. Its flushed timing CSV and
periodic telemetry were recovered without fabricating a final summary.

| Measurement | Interrupted baseline |
|---|---:|
| First ten minutes display cadence | 58.74 Hz |
| Last 8.43 minutes display cadence | 46.91 Hz |
| Overall display cadence | 53.17 Hz |
| Display gaps over 100 ms | 2,677 |
| Longest display gap | 711.31 ms |
| Display gaps overlapping archive JSON writes | 2,639 / 2,677, or 98.58% |
| AI publication cadence | 11.52 Hz |
| AI gaps over 250 ms | 957 |
| Longest AI gap | 1,392.05 ms |
| Last durable checkpoint | 3,143 images, 33,710 points, 197 prompts |

Overlap alone is correlation. The subsequent controlled changes and transfer
trace provide the stronger evidence for the archive bottleneck. This interrupted
run has no final map-child metrics or normal final export, and its tail cannot
be certified complete.

## Controlled large-map tests

These probes start with 4,800 image descriptors and 60,000 route points. They
reuse one real PNG for historical planes, then record new real AI images while
following the same elapsed-time route. They test history scaling, not diverse
historical image loading. Prompts are held fixed.

| Build | Duration | Display Hz | Display gaps >100 ms | AI gaps >250 ms |
|---|---:|---:|---:|---:|
| Original archive thread | 90 s | 49.95 | 79 | 46 |
| Archive process, full transfers | 90 s | 57.77 | 2 | 8 |
| Archive process plus geometry cache, full transfers | 180 s | 57.74 | 1 | 13 |
| Archive process, geometry cache and incremental transfers | 180 s | 59.89 | 0 | 0 |
| Map disabled control | 180 s | approximately 60 | 0 | 0 |

The final large-map test's longest display gap was 43.84 ms; its longest AI
publication gap was 149.36 ms. Parent serialization averaged 0.216 ms and peaked
at 1.152 ms during measurement. Live payloads ranged from 2,744 to 299,150 bytes,
including the raw pixels of newly captured images. The preceding full-transfer
trace sent 3.68 to 3.76 MB per live job and averaged 37.42 ms of serialization.

The geometry-only probe at 4,800 images and 60,000 points reduced mean CPU update
time from 403.56 to 32.93 ms. Final geometry hashes matched exactly. This isolated
probe uses fake GL resources; the live measurements do not establish a separate
first-person FPS gain from geometry caching alone.

## Growth validation

This run started with an empty map and retained automatic prompt changes. It
recorded 3,155.41 seconds before the graceful stop request. Both workers exited
normally, the final archive completed, and no timing rows were lost. The test's
strict `passed` flag is false because it ended before its requested duration and
because occasional display/AI gaps still crossed its thresholds.

| Measurement | Fixed build |
|---|---:|
| Display cadence | 59.02 Hz |
| Display gaps over 100 ms | 20 |
| Longest display gap | 165.56 ms |
| AI publication cadence | 12.65 Hz |
| AI gaps over 250 ms | 109 |
| Longest AI publication gap | 738.24 ms |
| Parent archive serialization mean / maximum | 0.83 / 6.16 ms |
| Map window cadence | 29.45 Hz |
| Map cadence, first / last ten minutes | 29.73 / 28.70 Hz |
| Maximum resident map textures | 256 |
| Saved history | 2,114 images, 23,908 points, 133 prompts |

The same-duration portion of the original run averaged 55.15 display Hz with
1,326 display gaps over 100 ms. The fixed run had 98.49% fewer such gaps. These
growth runs share the route and prompt-change policy; their randomly selected
prompts and runtime conditions differ. The fixed-prompt large-map comparison
above isolates the archive change more directly.

Ten-minute display cadence bins stayed between 58.43 and 59.55 Hz, followed by
a 2.59-minute tail at 59.34 Hz. This removes the observed progressive display
collapse, without claiming that all occasional pauses are eliminated.

Sampled process-tree working set rose from a 2,161.72 MiB median during minutes
1-10 to 2,315.19 MiB during the last ten minutes. It peaked at 2,324.91 MiB.
This includes the growing history, archive worker, map worker and benchmark
instrumentation. It is not a bounded-memory or week-long reliability claim.

Every saved PNG passed its format and dimension checks. All 2,114 images embedded
in the SVG matched the original PNG bytes by SHA-256, and the interactive viewer's
embedded manifest matched the final JSON exactly. The map worker covered the
entire measured interval and reported no failures.

## Short comparison without the map

The map-disabled run retained automatic prompt changes and was stopped after
218.31 seconds once it reproduced the remaining class of pauses. It averaged
59.93 display Hz, with one 110.11 ms display gap and five AI gaps over 250 ms.
The longest AI gap was 513.00 ms. No map or archive worker was present.

This shows that occasional pauses can occur independently of the map. It does
not identify the cause of every remaining gap or establish zero map overhead.
Prompt choices were random, so this was not an exact replay of the growth run's
prompts. No additional production change was made on the strength of these
isolated pauses. The archive bottleneck has a reproduced cause, a measured fix
and focused recovery tests; broader prompt/render changes need their own loop.

## Setup and evidence

Intel Core Ultra 7 155H, RTX 4070 Laptop GPU with 8 GB for AI, and Intel Arc Pro
Graphics for both OpenGL windows. Both benchmark windows are 1920 by 1080.
Generation is 384 by 256, the display target is 60 Hz and conditioning target is
15 Hz. GlazeWM is stopped during tests. GPU workloads run one at a time.

Focused checks cover real spawned archive workers, failure after delta merge,
worker death, failed retry submission, final-export failure followed by resumed
recording, immutable accepted images, Space reset, snapshot merging and cached
geometry. All 52 focused tests passed before the long validation.
The final full test suite passed all 293 tests.

A final 60-second input/autowalk/input replay verified two separate path
segments, no captures during the 18-second autowalk interval, a fully visible
idle title without repeated redraws, and a fully restored map after input.
The final 29-image archive saved normally. This functional check averaged 59.29
display Hz but did not pass strict timing thresholds: it recorded one display
gap over 100 ms and three AI gaps over 250 ms. GlazeWM was restored after tests.

Local evidence, ignored by Git:

- `logs/performance/map-soak-2h-20260906/analysis.json`
- `logs/performance/map-fix-before-4800/`
- `logs/performance/map-fix-process-4800/`
- `logs/performance/map-fix-combined-4800/`
- `logs/performance/map-fix-off-control/`
- `logs/performance/map-fix-trace-4800/`
- `logs/performance/map-fix-incremental-4800/`
- `logs/performance/map-fixed-soak-2h-20260906/`
- `logs/performance/map-fixed-soak-2h-20260906/validation.json`
- `logs/performance/map-fixed-soak-off-10m-20260906/`
- `logs/performance/map-fixed-cycle-20260906/validation.json`
- `logs/performance/geometry-before.json` and `geometry-after.json`

Use `tools/long_map_session.py` to repeat the growth run and
`tools/summarize_long_map.py` to produce exact timing statistics and ten-minute
bins. Create `stop-requested` inside the run folder for a graceful early exit.
An early exit is still incomplete, even if the final archive saves successfully.
