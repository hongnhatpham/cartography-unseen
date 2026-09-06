# Journey map performance check, 6 September 2026

The fresh map adds little average overhead on this laptop. Accumulated history
still causes intermittent stalls, so this is **not an all-day exhibition sign-off**.

## Matched fresh-map comparison

GlazeWM was disabled for all final runs and restored afterward. Both actual
windows were 1920×1080. OpenGL used Intel Arc Pro Graphics; AI generation used
the RTX 4070 Laptop GPU. Inference was 384×256, with a 60 Hz display cap and
15 Hz conditioning cap. No video capture ran during measurement.

Two runs per condition used the same elapsed-time route, seed, fixed prompts
and held-W input. Run order was off/on/on/off. Runs lasted 90/120/120/120 seconds
after the first generated frame. Pooled statistics use the common 10–90 second
window; full-run maxima below include the later samples.

| Metric | Map off | Map on |
| --- | ---: | ---: |
| Display cadence | 60.46 FPS | 60.24 FPS |
| AI publication cadence | 12.92/s | 12.87/s |
| Display interval p95 | 24.94 ms | 26.65 ms |
| Display interval p99 | 27.53 ms | 31.43 ms |
| AI publication interval p95 | 87.71 ms | 91.59 ms |
| AI publication interval p99 | 94.74 ms | 106.50 ms |

Display cadence fell 0.37%, AI cadence 0.41%; display p95 increased 1.71 ms.
Each map-on run saved 81 images. Peak sampled combined parent/child RAM rose
about 275 MiB, from roughly 1,758 to 2,030 MiB. CUDA reserved memory was unchanged
at 2.56 GiB. This does not measure Intel GPU memory.

| Full run | Maximum display interval | Maximum AI gap | Verdict |
| --- | ---: | ---: | --- |
| Off 1 | 44.67 ms | 106.45 ms | Pass |
| On 1 | 52.77 ms | 134.54 ms | Pass |
| On 2 | 46.32 ms | 127.70 ms | Pass |
| Off 2 | 41.91 ms | 115.73 ms | Pass |

The preset limits were 100 ms between displays and 250 ms between AI outputs.
Passing these limits does not mean every frame meets a 16.7 ms budget. Earlier
GlazeWM-on runs used different actual viewports and are excluded; their results
do not establish that GlazeWM caused stalls.

## Accumulated history exposes a remaining regression

A synthetic archive started with 1,200 image planes and 12,000 route points,
then continued real AI generation and recording for 90 seconds. Historical
planes reused one actual generated PNG. This is a scaling probe, not a genuine
long play session or a test of diverse image files.

| Probe | Display maximum | AI gap maximum | Verdict |
| --- | ---: | ---: | --- |
| Original archive writer, full map | 147.24 ms | 481.42 ms | Fail |
| Original writer, geometry disabled | 175.78 ms | 496.46 ms | Fail |
| Faster serialization experiment, full map | 97.55 ms | 170.16 ms | Pass |
| Production serialization change, full map repeat | 190.07 ms | 736.98 ms | Fail |

Disabling geometry retained recording, IPC and a cached title drawn at 30 Hz.
The map child's CPU time fell sharply, but the stalls remained. Rendering alone
does not explain them. Archive encoding was one measurable source of overhead:
12 alternating microbenchmark pairs on a 1.62-million-character manifest took
106.85 ms on average with streamed `json.dump`, versus 27.84 ms with one-shot
`json.dumps` followed by `write`. Their output was identical.

The one-shot encoder is now used in the atomic archive writer. It preserves
flush, fsync, temporary-file replacement, Unicode and invalid-number rejection.
It allocates the complete JSON string in memory. The repeat test proves this
optimization does **not** establish that the large-history stall is fixed.
That repeat had eight display intervals above 100 ms and twelve AI gaps above
250 ms. The remaining cause needs further isolation before unattended long runs
can receive a performance sign-off. No archive truncation or capture limit was added.

## Interaction and verification

A 60-second cycle held W for 20 seconds, released it for 20 seconds, then resumed.
Autowalk began after two idle seconds. The map stopped issuing draws for 16.81
seconds after its fade and resumed with two disconnected human segments. All 29
image generation timestamps fell within human segments. Maximum display interval
was 43.15 ms; maximum AI gap was 151.07 ms. The child reported no failures.
This checks the real autowalk state and recording boundaries on a synthetic
camera route; it is not a natural-navigation or projector test.

42 focused tests passed across replay measurement, archive integrity, map view
and journey integration. The archive test verifies that a failed invalid-data
replacement preserves the prior manifest and a valid retry succeeds. The replay
tool rejects absent or early-ended map measurements and reported child failures.

## Reproduce

Run sequentially, with GlazeWM disabled and no other GPU workload:

```powershell
runtime/python/python.exe tools/replay_performance.py --map-check off --seconds 120 --output logs/performance/check-off
runtime/python/python.exe tools/replay_performance.py --map-check active --seconds 120 --output logs/performance/check-on
runtime/python/python.exe tools/replay_performance.py --map-check cycle --seconds 60 --output logs/performance/check-cycle
runtime/python/python.exe tools/replay_map_history.py --image PATH_TO_GENERATED_PNG --output logs/performance/check-history
```

Output folders must be new. Add `--recorder-only` to the history probe to retain
archives and IPC while replacing geometry work with cached-title draws. Child
draw timings measure CPU submission, not GPU completion. Check actual window
sizes and GPU names in each run's metadata before comparing.

Evidence remains locally under `logs/performance/map-final-plain-*`,
`map-final-history*`, `map-final-cycle`, `map-final-comparison.json` and
`map-final-resources.jsonl`. The fresh-map comparison was measured at commit
`990291b`; the production repeat and cycle include the serialization change.
These short runs cannot establish behavior at arbitrary map size, projector
resolution, graphics routing, or all-day operating temperature.
