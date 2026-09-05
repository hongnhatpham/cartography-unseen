# AI image freezes

This investigation measures gaps between newly published AI images as well as
display intervals. A display can redraw an old image at 60 FPS while inference
has stopped producing images. The earlier input-thread report measured the
display improvement and did not establish that AI image freezes were fixed.

Reprojection remains disabled. Tests retain the model, fp16 precision, diffusion
resolution, sampler, guidance, prompts and proxy materials.

## Diagnosis

The previous Windows input change isolates native event waits from rendering.
Remaining traces showed world construction competing with the inference thread
for Python execution. A 267 ms frame included 156 ms of world updates and 47 ms
of form preparation/rendering. Native samples repeatedly found Python thread
reacquisition waits while the other thread progressed through inference.

Running inference in a separate process improved throughput but retained four
AI publication gaps above 250 ms in 120 seconds, with a maximum of 413 ms.
Those gaps matched inference durations inside the child process. Image transfer
between processes did not explain the remaining tail. This experiment remains
outside production.

A process-local OpenBLAS thread limit also improved one live run, but a focused
visibility calculation test did not reproduce that benefit. No BLAS limit or
system power/driver setting was changed in production.

## CUDA Graph experiment

The UNet repeatedly submits the same GPU operations with different latent,
timestep and prompt tensors. A CUDA Graph records those operations once during
startup and replays them after copying the current inputs into persistent
buffers. Prompt interpolation, noise, temporal memory and settings calculations
continue to run normally outside the graph.

The standalone probe compared six eager/captured predictions across batches
one and two, different input latents, embeddings and timesteps 202, 81 and 299.
All outputs were bitwise equal. CUDA RNG state and retained previous outputs
were unchanged. Mean UNet wall time fell from 45.00 to 23.84 ms for batch one
and from 47.93 to 38.79 ms for batch two over 24 alternating pairs.
Both cached graphs together reserved 2,544 MiB of GPU memory. Capture took
about 0.3 seconds per batch during startup.

Raw standalone results: `logs/performance/cuda-graph-unet-check/summary.json`.
Process experiment: `logs/performance/freeze-process-default/summary.json`.

## Matched 120-second comparison

Both runs used the same saved configuration, route and production source hashes.
The graph experiment patched only startup to prepare the UNet graphs. Both ran
at 384x256, CFG 1.5, with the default thread environment. Graph replay ran first,
followed by eager inference.

| Measurement | Eager inference | UNet graph |
|---|---:|---:|
| Longest gap between new AI images | 1,684.86 ms | 122.55 ms |
| AI gaps above 250 ms | 26 | 0 |
| Longest display interval | 213.97 ms | 38.33 ms |
| Display intervals above 100 ms | 4 | 0 |
| New AI images per second | 9.33 | 12.79 |

The graph served 1,539 UNet calls with no eager fallbacks. No measurements were
dropped and neither run changed resolution. This supports reducing repeated
Python submission work as the fix for the observed inference stalls. It does
not establish a universal maximum latency on Windows or week-long stability.

Raw results: `logs/performance/goal-unet-eager-matched` and
`logs/performance/goal-unet-graph-replay`.

## Production behavior and output checks

`app/diffusion/cuda_graph.py` stores at most two startup captures for the current
resolution, covering CFG on and off. Live calls copy all three inputs and return
an independent output tensor. Unknown shapes use eager inference. A resolution
change releases the captures and uses eager inference until the next launch;
reapplying the same resolution keeps the captures. Unload releases them.
Unsupported capture or capture OOM falls back to the same eager computation
without lowering resolution.

The GPU test in `tests/test_cuda_graph.py` compares five complete output images
against eager execution with a fixed clock and seed. It changes prompts, CFG
across its batch threshold, timesteps, guide strength, instability and step
count. All five images were bitwise identical. CUDA RNG state remained unchanged.
Focused tests also cover eager fallback, capture failure and cache lifecycle.

The integrated production version then repeated the same 120-second route
without the experimental adapter. It passed again: maximum AI publication gap
115.38 ms, no gaps above 250 ms, maximum display interval 42.11 ms, and no display
intervals above 50 ms. It published about 12.98 new AI images per second, with
no errors or lost measurements. Results are in
`logs/performance/goal-unet-graph-production`.

## Diagnostics and normal-operation checks

The first five-minute normal run with UNet graphs still had 24 new-image gaps
above 250 ms, with a maximum around 635 ms. The diagnostics overlay was enabled
in this run; the fixed replay disables it. The worst display pauses spent
173 to 225 ms rebuilding the overlay, overlapping longer inference and input
waits. Many long inference calls were not adjacent to a prompt change.

A 180-second normal run then changed only diagnostics visibility in a private
copy of the same configuration. Automatic prompt and settings changes remained
active. It processed 16 prompt requests and produced 2,337 AI images, with no
AI gaps above 250 ms and a maximum gap of 138.68 ms. No display interval exceeded
100 ms; the maximum was 56.24 ms. Additional inference timings found no live
generation above 100 ms and did not justify changing TAESD computation.

These results implicated diagnostics rebuilding as another source of contention,
but normal routes and setting schedules are time-dependent. They do not by
themselves prove diagnostics caused the inference stalls.
Raw results: `logs/performance/graph-normal-confirmation` and
`logs/performance/graph-normal-overlay-off-inner`.

Diagnostics now reuse the RGBA surface and unchanged text rasters. Subsequent
texture writes cover the old and new panel/caption bounds, including text that
extends beyond its panel. Removing or shrinking text clears its old pixels.
The cache retains only the current text, and resize invalidates the texture.
Layout, colors, opacity and the 10 Hz update limit stay the same.

A same-process CPU comparison at 1920x1200 with five changing lines among 26
reduced mean rebuild time from 12.53 to 1.53 ms and the 95th percentile from
17.44 to 2.52 ms in the saved final benchmark. Uploaded bytes fell 86.5%.
Exact RGBA comparisons cover three window sizes, long text, overlapping captions,
removal, shrinking and resize. A real GL texture readback also matched the original
rasterization exactly. Results: `logs/performance/overlay-rebuild-benchmark.txt`.

The optimized-overlay run was stopped after 234.8 seconds because it still
reproduced severe stalls. Lower overlay CPU cost alone did not fix the remaining
normal-operation freezes. A further process-plus-graphs experiment completed
180 seconds with four AI gaps above 250 ms, maximum 574 ms, and three display
intervals above 100 ms, maximum 182 ms. The child independently measured a
622 ms generation call. Process isolation remains outside production because
it does not yet justify the added lifecycle complexity.

Review also found that the detailed soak collector can amplify pauses: it builds
stage dictionaries under a shared lock after a slow display interval. Its
`generate` timer starts before acquiring that lock. This affects interpretation
of detailed timings, although it cannot explain the separately timed child
generation. A minimal-collector normal run is needed before attributing the
remaining pauses to one component.

The minimal normal recorder completed 180 seconds with diagnostics open,
maximum display interval 45.66 ms and maximum AI gap 176.36 ms. A second minimal
run adding GPU queries had three AI gaps above 250 ms, maximum 391.47 ms. Those
stalls did not overlap the queries; this does not establish telemetry as the
cause. Automatic settings use nondeterministic draws, so these normal runs are
functional checks rather than identical-workload comparisons.

The detailed collector now queues bounded raw snapshots and formats them on
the sampler outside its timing lock. The reusable lightweight command is
`tools/replay_performance.py --normal --seconds 600`. It preserves normal
controls, collisions, prompts, palettes, resolution and diagnostics, and changes
only idle-flight activation in a private configuration copy.

The 600-second lightweight normal run completed with 7,527 AI publications,
nine AI gaps above 250 ms and one display interval above 100 ms. Maximum gaps
were 460.05 ms for new images and 180.36 ms for display. The AI-gap 99th
percentile was 113 ms. No records were dropped and no worker error occurred.
This run fails the strict maximum-gap thresholds: the changes reduce measured
stalls but do not establish a freeze-free experience. Results:
`logs/performance/final-normal-600`.

A following 300-second normal run added only worker-local inference phase
timings, without GPU synchronization. It completed with 3,898 measured live
frames, maximum AI gap 148.85 ms and maximum display interval 45.06 ms. No raw
inference exceeded 100 ms, so no slow-phase records were emitted. This clean
run cannot locate or invalidate the pauses seen during the longer run. Results:
`logs/performance/minimal-normal-inner-300`.

The shipped changes are UNet graph replay and cheaper, pixel-identical overlay
rebuilding. They preserve the image computation and improve the controlled
comparison. Rare normal-operation inference pauses remain unlocalized; no
process worker, TAESD change, thread-pool limit or driver/power setting was added.
Reprojection remains disabled.

The final full test suite passed 253 tests, including actual GPU image equality,
GL overlay pixel equivalence, screenshots, input/window lifecycle, navigation
and the timing collectors. The saved user configuration retained its SHA-256
`53b7540401ee0ffa38f1384f9d8de4aa91164dd1904bb57219a209b15b027229`.

## Verification method

`tools/replay_performance.py` fixes the prompt, seed, settings and an elapsed-time
flight route. It records source hashes and every timing to CSV. A run passes
only when it completes, loses no timing records, has no display interval above
100 ms, and has no new-AI-image gap above 250 ms. The final period without a
publication is included, so silent worker stalls cannot pass.

`tools/soak_test.py` also records new-image gaps during normal automatic flight,
prompt changes and diagnostics. Separate upload, overlay, trail and display-swap
timings help locate any remaining display stalls. All timing storage is bounded.
