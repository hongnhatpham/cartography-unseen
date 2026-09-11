# Latent Space

Latent Space is a real-time Windows installation. You move freely through a dense procedural landscape of blocks, curved forms and open wire structures, and a diffusion model continuously rewrites what you see. The 3D render is never shown in the normal AI view: it is a steering signal for a continuous walk through latent space. Input and display run independently of AI generation. Depth reprojection is optional and remains disabled in the current exhibition settings.

## Two-projector journey map

The normal launch opens two movable windows: the first-person experience and
the journey map. The F1 operator overlay starts open so the mouse is free.
Drag each window by its title bar onto the intended screen, then press **F**
to make both fullscreen on their respective screens. Press F again with the
overlay open to restore their windowed positions. Close F1 to return mouse
control to traversal. F works from either window while the overlay is open;
F11 still toggles only the first-person window.

The map follows the selected Image field design. It orbits the current player
position, records human movement in three dimensions, and places generated
images at their original camera positions and viewing angles. An initial view
starts each human segment, then captures occur every `map_capture_distance`
world units travelled, 12 by default. Turning in place does not capture more
images. These clean generated frames retain their source resolution and omit
display overlays, sharpening and optional display reprojection.

After `map_idle_seconds` without movement or mouse look, 10 by default, the
map crossfades over 1.2 seconds to a centered Cartography Unseen title in the
same Space Grotesk SemiBold font as the first-person title. The idle screen
credits Nhat (Hong) Pham, Agnieszka Kiejziewicz, Ricardo Arce and Kok Yoong Lim,
with a contributor credit for Tom Nguyen and a QR code linking to
[emergentplay.bynhat.com](https://emergentplay.bynhat.com).
Automatic flight also starts this fade and stops path/image capture immediately.
Returning fades the map back in over 0.6 seconds, smoothly reversing an unfinished
transition. The player marker uses the actual current position, and recording
starts a disconnected stroke. After the fade out, map scene rendering stops; the cached
title screen has no continuous animation. Operator placement instructions can
still appear when F1 is open. Prompt events retain exact text, timing, pose,
trigger and the revision associated with each image, including changes while
the map is idle.

Space saves the current nonempty map before starting a new prompt and an empty
map. Procedural geometry, player position and diffusion seed keep their existing
behavior. Escape or closing the main window saves the final map. A failed save
keeps the application open for retry. Each archive lives in `journeys/<id>/`:

- `manifest.json` contains the route, image poses, prompt history and settings.
- `images/` contains lossless WebP captures at their original resolution and pixel
  quality. `map_image_format: "png"` selects PNG for new journeys instead.
- `index.html` is an interactive viewer. Open it locally to orbit, pan, zoom and
  inspect prompts and original images. Keep it with its manifest and images.
- `complete.json` marks a successfully finalized archive and checksums its manifest.
  Active checkpoints have no completion marker.

SVG export is now optional because it duplicates every image. To create one later,
run `runtime/python/python.exe tools/export_journey_svg.py journeys/<id>`.
Set `map_export_svg` to true to include it automatically on each completed save.
The SVG preserves vector paths and embeds the images at their source resolution.

Optional private R2 backup uploads completed journeys in a separate process,
verifies remote checksums, then removes the oldest verified local copies above
`map_cache_gib`, 5 GiB by default. Active and unsynced data stays local. Below
`map_min_free_gib`, 1 GiB by default, path and image capture pause and resume with
a disconnected segment when space returns. Prompt metadata continues recording.
Uploads are disabled until configured. See [storage setup](docs/journey-storage.md)
for credentials, recovery, and the distinction between backup and publication.
An existing Wrangler login can back up portable ZIPs without new access keys;
large ZIPs stream as verified parts. Scoped S3 credentials remain an option for
storing individual viewer files directly in R2.

If the browser blocks local image loading, use **Open archive** in the viewer
and select that saved journey folder. The files stay on your computer.

The recorder sends incremental checkpoints to a separate archive process during
play. Existing archives are never overwritten by a new journey. The map viewer
runs separately, reuses image geometry, caps drawing at 30 FPS and bounds loaded
textures. Closing only the map window leaves recording running in the main
application. The [growing-map investigation](docs/performance/map-history-20260906.md)
records the fixes, timing comparisons and archive-integrity checks.

Use `run.bat --no-map`, or set `journey_map` to false in `config.json`, for the
single-window experience. Map enablement and capture/inactivity settings apply
at launch. The [journey map brief](docs/journey-map.md) records the decisions.
Publishing selected archives on Emergent Play remains a separate step.

## One-click installation

Requirements:

- Windows 10/11 x64
- NVIDIA GPU with a current display driver
- 6 GB VRAM minimum; 2.5 GB is used at the default resolution
- Internet for the first launch
- Approximately 12 GB free disk space during setup

Download or clone the project, then double-click `run.bat`. On its first launch it automatically downloads and installs:

- Portable Python 3.11.9
- PyTorch 2.6 with its bundled CUDA 12.4 runtime
- Pinned application dependencies
- A pinned fp16 SD-Turbo snapshot (`stabilityai/sd-turbo`)
- The tiny autoencoder TAESD (`madebyollin/taesd`), used for every latent encode and decode

The NVIDIA display driver must already be installed. A separate CUDA Toolkit is not required. Later launches can operate offline; optional archive uploads need a connection.

## Controls

Open the F1 diagnostics overlay to unlock settings hotkeys. Closing it locks
them immediately. Movement, mouse look, Space, Enter, Escape and F1 always work.
Temporary loading or status panels do not unlock settings.

In the first-person window, after ten seconds without player input, the title fades
in at the upper left and movement instructions appear at the lower right.
They use Emergent Play's Space Grotesk and IBM Plex Mono typography. The fade
takes 1.2 seconds; keyboard or mouse input fades them out in 0.25 seconds.
Holding a key or mouse button counts as activity. Automatic flight and prompt
changes leave them visible. Diagnostics, prompt editing and status panels hide
them, and the instructions sit above the prompt caption when it is enabled.
The text is cached locally and faded on the GPU; it does not enter AI generation.
Title and control text are 2.25 times their original size, with proportional
fitting on smaller windows to preserve the gap between them.

| Key | Action |
|---|---|
| W / A / S / D | Fly along your gaze and strafe |
| Q / E | Descend / rise, independent of gaze |
| Mouse | Look |
| Shift | Fly faster |
| Escape | Exit |
| Space | Save/reset the journey map, choose a new prompt and random tuning; keep world, viewpoint and diffusion seed |
| Enter | Save the displayed AI image as a PNG in `screenshot/` |
| Shift + R | New diffusion seed |
| P | Edit prompt |
| F1 | Diagnostics overlay |
| F | Fullscreen both windows on their current screens, or restore placement; requires F1 open |
| F2 | Proxy / generated AI view |
| F3 | Normal / depth / edge diagnostics |
| F4 | Freeze diffusion |
| F5 | Toggle depth reprojection |
| F6 | Return to generated AI view |
| F7 | Prompt caption at the bottom |
| F8 | Toggle feedback reprojection |
| F9 | Cycle prompt auto-advance: off, 12, 24, 48, 96 s. Each advance chooses a different family and samples new tuning |
| F12 | Toggle idle flight (starts after a minute without input) |
| F10 | Cycle resolution modes |
| F11 | Fullscreen / windowed |
| [ / ] | Reprojection strength |
| - / = | Diffusion steps, 1–4 |
| , / . | Display sharpening |
| H / J | Fog distance nearer / farther in steps of 10. Nearer means more fog |
| V | Toggle the ten-second player trail |
| I / O | Instability down / up |
| K / L | Guide strength down / up |
| T / Y | Shift the timestep window down / up by 25 |
| N | Cycle noise-walk seconds: off, 2.5, 5, 10, 20, 40 |
| M | Cycle prompt-walk seconds: off, 3, 6, 12, 25, 50 |
| C | Cycle CFG: 1, 1.25, 1.5, 2, 3 |

Screenshots preserve the displayed crop, sharpening and reprojection, excluding
all text, diagnostics and the player trail. Timestamped filenames avoid replacing
earlier captures. PNG compression and writing run in the background, with one save
pending at a time. Capture requires the generated AI view, not proxy/depth/edge mode.

While editing a prompt, Enter applies it instead of saving a screenshot, Escape cancels, and Ctrl+D restores the default text. Each new subject independently samples these settings:

- Timestep window: independently sample the lower bound from 80 to 200 and the upper bound from 280 to 400. The window width varies with each draw.
- Sharpness: 1.00–2.00.
- Instability: 0–40%.
- Guide strength: 65–100%.

Space, timed advance, changed text accepted in the prompt editor, and external prompt edits all trigger this variation. CFG, fog, position, world and diffusion seeds, diagnostic state, and other settings stay unchanged. Automatic variation works with F1 closed; manual settings hotkeys still require F1. Regional color changes during travel do not sample new tuning.

Manual settings changes are written back to `config.json` immediately. Space and accepted prompt edits save the prompt and its sampled settings; timed prompt choices and their tuning remain session-only. Startup uses the saved settings until a new subject is chosen. Freeze and diagnostic views stay session-only. Editing `config.json` in a text editor while the app runs also works: changes are picked up within half a second.

`fog_distance` controls how far away structures fully fade into the atmosphere.
The default, 196 world units, matches the original fog. Its range is 40–300.
Press H to soften distant color sooner, or J to see farther. F1 shows the current
distance. The setting affects the proxy sent to the AI and survives prompt changes.

Movement has no gravity, fixed eye height or altitude limit. Look up and hold W to climb, look down to descend, or use Q/E to change height without changing your gaze. The world continues above and below you as well as horizontally. Release the keys to stop. Solid forms still block movement and you slide along them; you can pass above or below them wherever there is space. Idle flight also steers in three dimensions.

The generated image carries the environment. An optional pale, 15-pixel-wide trail marks the
last ten seconds of travel, fading oldest-first and disappearing when you stop.
With F1 open, press V to toggle it; `player_trail` saves the choice. Turning it off clears the
route. It sits slightly below the travelled path so you can see it when looking
back, and structures hide it where the route passes behind them. The trail is
drawn after the AI image so its fade stays exact and does not alter generation.
Its visible end leaves a four-unit gap along your route, measured from the
displayed frame's capture time so newer marks cannot appear ahead of an older AI view.

## Continuous-run monitoring

Run `runtime/python/python.exe tools/soak_test.py --minutes 30` to monitor the real
app for thirty minutes after its first AI frame. It uses a copy of your config,
starts idle flight after two seconds, and exits normally at the end. Escape or
closing the window ends the run early and marks it incomplete.

Results go in a timestamped folder under `logs/performance/`: `summary.json`,
per-minute CSV/JSON timings and per-second process memory samples, with NVIDIA
telemetry every ten seconds. The summary includes frame stalls, generation and
prompt timing, garbage collection pauses, memory trends, worker errors and
resolution fallbacks. Timing percentiles have one-millisecond precision.
`stalls.jsonl` records display intervals above 50 ms with overlapping render,
navigation, event-polling, frame-wait and worker stages, including garbage
collections. Its queues are bounded, and disk writes run on the sampler thread.
Add `--fixed-settings` for comparisons with automatic prompt advance and random
tuning disabled and the initial AI palette held constant. Rendered world colours
still vary with location; your saved configuration stays unchanged.
Normal monitoring preserves the app's input path. On Windows, a separate thread
owns the window and SDL input; drawing reads buffered input and can continue
during a native event wait. The monitor still times those native waits
on the owning thread. `--split-poll` is rejected on Windows because it would pump
input from the drawing thread. It remains a diagnostic override on other platforms.

For repeatable comparisons, run `runtime/python/python.exe tools/replay_performance.py
--seconds 120 --max-stall-ms 100`. This uses a saved config copy, fixes generation
at 384×256 and CFG 1.5, holds the prompt and seed, and follows the same camera route
from the first AI frame. Collision queries run, but their displacement is overridden
to keep the route independent of frame timing. Escape and Quit remain available.
Every display interval, generation duration and new AI image publication interval
is saved to `timings.csv`. The command also checks `--max-ai-gap-ms 250` by default,
so repeated presentation of an old image cannot hide an AI freeze. It fails if
either threshold is exceeded, the run ends early, measurements are lost, or
generation falls back to lower resolution.
See [the input-stutter investigation](docs/performance/window-input-stutter-20260905.md).

Use `--minutes 10080` for a full week on the exhibition computer. A thirty-minute
run can expose early slowdown or memory growth; it cannot establish week-long
reliability. Launch `run.bat` afterward to return to normal operation.

The dense slabs and layered spaces retain the KOSMA reference direction. Interior clutter thins along existing passages, giving them more room while retaining surrounding panels and denser areas. Free flight changes how you traverse them; it does not add sparse floating islands or an open-sky setting.

## Resolution modes

F10 cycles `512x512`, `640x384`, `512x384` (default), `384x256`, `768x512`, `1024x768`. Measured on an RTX 4070 Laptop, one step, CFG 1, 6 warmup frames and 20 timed frames on an otherwise idle GPU:

| mode | ms/frame | diffusion FPS | peak VRAM |
|---|---|---|---|
| 512x512 | 98.8 | 10.1 | 2.43 GB |
| 640x384 | 97.6 | 10.3 | 2.42 GB |
| 512x384 | 96.0 | 10.4 | 2.39 GB |
| 384x256 | 88.4 | 11.3 | 2.33 GB |
| 768x512 | 128.6 | 7.8 | 2.51 GB |
| 1024x768 | 285.5 | 3.5 | 2.75 GB |

The four smaller modes cluster within about 10% of each other: at one step the frame is dominated by kernel launches and by TAESD, not by pixel count, so dropping resolution buys VRAM headroom and framing rather than speed. A repeat sweep in the same session stayed within 11% on every mode. Two steps costs about 1.6x, and CFG above 1 runs the negative prompt as a second batch item for roughly another 1.4x: at the shipped `guidance_scale` of 2.2, 512x512 measures 133 ms / 7.5 FPS.

Reproduce with `runtime\python\python.exe tools\benchmark.py --steps 1 --guidance-scale 1 --warmup 6 --frames 20`.

The generated view draws its proxy only when a conditioning capture is due.
Movement, collision checks, world streaming and presentation still run at the
display rate. Proxy and diagnostic views keep their display-rate updates. Edge
maps are calculated only for consumers that use them; the AI still receives the
same RGB and depth images.

Streaming appends newly loaded geometry to the GPU buffer. It rebuilds the active
buffer when chunks are evicted, the render origin moves, or capacity grows.
A replay across three chunk boundaries reduced average CPU streaming time from
14.0 to 4.6 ms and geometry uploads from 146.8 to 5.0 MB. These are streaming
measurements, not diffusion FPS. The GPU comparison matched RGB and depth exactly
across 42 sampled views, including partial loads and distant positive and negative
altitudes. Resolution, CFG, sampler settings, model precision and world density
remain unchanged.

Streaming yields after three milliseconds of chunk construction or three chunks,
whichever comes first. Startup still fills the complete window. The renderer
retains packed arrays instead of duplicate source objects, and the source-chunk
cache holds only 128 recent chunks. Revisiting an evicted region regenerates the
same geometry within that frame budget. The 11x11x11 active window is unchanged.

Only Pygame's display and font subsystems start. Unused joystick and audio
initialization previously introduced work into Windows event polling; the app
uses keyboard and mouse input.
Input polling uses a one-millisecond event wait, which releases Python's shared
lock during native Windows processing, then drains the queue without pumping
again. Text composition is enabled only while the prompt editor is open.
The trail uses screen-space triangles to retain its 15-pixel width even on
graphics drivers that cap native line width at ten pixels.

Published raw frames now reuse their GPU image until the frame changes. The
renderer retains the frame itself, avoiding recycled-object-ID errors, and
restores the correct texture when switching views. In a fixed-image comparison,
180 repeated refreshes avoided 53 MB of duplicate uploads and reduced presentation
time from 0.65 to 0.45 ms per refresh, with identical screen pixels.

World streaming caches repeated channel-field samples and the current ordered
chunk window. Both caches are bounded for endless exploration. A six-boundary
CPU profile fell from 3.10 to 2.43 seconds, with identical geometry, colors,
transforms and colliders across 1,331 chunks. Reprojection also skips live-camera
renders between captures; its interpolated-camera depth still updates every
display frame. These are reductions in component work, not a measured AI FPS
increase. Live inference timing remained variable. Same-frame buffered readback
was slower in testing and was not adopted.

Prompt changes reuse a bounded cache of 64 text embeddings, including the shared
negative prompt. After model warmup, long-lived startup objects are excluded from
full garbage-collection scans. New objects still undergo normal collection, and
unloading the backend restores the previous collection behavior. These changes
reduce prompt work and periodic pauses without changing the model or its settings.
In a same-backend comparison, full collection fell from 170 ms to under 0.1 ms;
revisiting a prompt fell from 26–46 ms to 1–3 ms. A previously unseen prompt still
needs text encoding. These measurements do not imply constant AI frame times.
The interactive app also uses a 1 ms Python thread interval so world updates and
AI kernel submission share execution more frequently, restoring the host's value
on exit. With the cache and collection changes active, a moving replay improved
AI p95 from 203 to 118 ms and the longest display interval from 99 to 65 ms.

## Prompts

`prompts.json` holds **style families**, not flat prompts. Each family has a `name`, a
`base` tail shared by its variants, four subject `variants`, and a `settings` block
of sampler presets for offline style comparisons. Live subject changes use the
random tuning ranges above. One library entry is built per variant as
`"<variant>, <base>"`, and `master_prefix` ("corrupted 3D render") is prepended to all of
them.

The twelve shipped families are Topology Unknown, Biophilic City, Mangled Data, Wire
Field, Washed Strata, Warped Spacetime, Corrupted Bloom, Datamosh Ravines, Cellular
Karst, Root Networks, Membrane Folds and Distant Presences. Distant Presences adds
four views with small human silhouettes and distant figures, partly hidden by
fog or structures. The three organic families add porous
formations, branching strands and curved sheets without the voxel wording used
by the original subjects. There are 44 variants in total. They are
deliberately abstract: wires, corruption, data, mangled geometry, warped space and time,
lattice, and the blocky biophilic cityscape of the original ComfyUI sequence. Every entry
is anchored as an eye-level view inside a vast outdoor landscape, and material nouns that
resolve into product shots (chrome, glass, foil) are kept out, so the sampler has room to
guess without landing on something concrete. Occasional human residue is accepted; the
negative prompt names only gamepads, desks, interiors and furniture.

Space and automatic advance choose a prompt from another family, sharing a history
that avoids the last three families when possible. Neither action applies the family's sampler presets. A family's
`settings` may only name live-tunable backend keys; anything else is
rejected when the library loads. F7 shows the effective prompt at 30% opacity, and the F1
overlay names the current family.

The effective prompt keeps the original text and appends the local landscape's hue
pair. The proxy supplies the new curved and wire forms; there is no shared shape
phrase appended to every subject. The saved prompt, default prompt and library
text stay unchanged. Travelling into a different color region updates the hue cue
through the existing prompt interpolation, without changing the subject or restarting
the automatic prompt timer or sampling new tuning.

A flat `"prompts": ["...", ...]` list still loads, with each entry treated as its own
family carrying no settings.

## Config keys

| Key | Default | Meaning |
|---|---|---|
| `backend` | `latent_walk` | `latent_walk` or `proxy_passthrough` |
| `model_path` | `models/sd_turbo` | Folder with `unet/`, `text_encoder/`, `tokenizer/`, `scheduler/` |
| `taesd_path` | `models/taesd` | `AutoencoderTiny` folder used for encode and decode |
| `sampler` | `""` | Empty infers from the scheduler; `euler_x0` or `lcm` force it |
| `diffusion_resolution` | `512x384` | One of the six F10 modes |
| `steps` | `1` | UNet evaluations per frame, 1–4 |
| `guidance_scale` | `2.0` | CFG. Above 1 activates the negative prompt and costs about 1.4x |
| `timestep_min` / `timestep_max` | `480` / `580` | Sampling timestep window. Higher hallucinates more, lower redraws the proxy |
| `instability` | `0.7` | Scales the timestep breathing (two tones, 13 s and 47 s) and the guide wobble (29 s) |
| `guide_strength` | `0.85` | How far each frame is pulled from its own memory toward the proxy. A floor: guide wobble can only raise it |
| `memory_match` | `1.0` | Re-anchor the memory latent's per-channel mean to the proxy latent each frame, so hue cannot random-walk |
| `memory_match_std` | `0.5` | Share of that re-anchoring applied to the spread; below 1 lets contrast breathe |
| `memory_leash` | `0.7` | Clamp the memory to a band this many proxy standard deviations wide. 0 disables |
| `depth_guide` | `0.5` | How much weaker the proxy pulls at the far plane than up close. Near forms stay anchored, the horizon hallucinates, which is what gives an eye-level frame depth |
| `depth_shade` | `-0.6` | Depth-based proxy luminance before encoding. Negative darkens nearby geometry; positive brightens it. 0 disables |
| `guide_wobble` | `0.0` | Amplitude of the slow increase above guide strength. 0 keeps the anchor steady |
| `noise_walk_seconds` | `5.0` | Seconds to slerp between seeded noise keyframes. 0 means fresh noise each frame |
| `noise_jitter` | `0.12` | Small variance-preserving jitter mixed into the walk noise |
| `prompt_walk_seconds` | `6.0` | Seconds to slerp CLIP embeddings to a new prompt. 0 hard-cuts |
| `prompt_auto_advance_seconds` | `24.0` | Seconds between changes to a different library prompt family. 0 disables |
| per-family `settings` | — | Sampler presets retained for offline style comparisons. Live subject changes use the random tuning ranges above |
| `feedback_reprojection` | `false` | Use the camera-aligned previous frame as the walk memory in place of `x0_prev`. The proxy still guides every frame. Costs about 39 ms of CPU per frame |
| `seed` | `12345` | Noise keyframe seed |
| `world_seed` | — | Landscape seed; preserved when the prompt changes |
| `movement_speed` / `sprint_multiplier` | `8.0` / `3.0` | Flight speed in world units per second |
| `autowalk_idle_seconds` | `60.0` | Seconds without input before idle flight starts. 0 disables |
| `reprojection` | `false` | Display-side depth warp between diffusion frames |
| `conditioning_fps` | `15` | How often the proxy is captured for the worker |

## How it works

The renderer streams deterministic geometry in 64-unit chunks, in an 11x11x11 window around the camera. Thick panels define walls and openings, and a connected channel field opens routes horizontally and vertically. Reefs, lattice bars, shards, columns and overhangs remain, with curved wire sheets, ribbed arches, open cages and rounded solids added among them. Along the channel field, 5% to 65% of interior object groups are omitted to clear passages while retaining the panels.

The seeded palette varies by region, with stronger color across surfaces and the surrounding atmosphere. Broad, soft mottling varies surface brightness at two scales, giving the AI detail to interpret across large faces without a checker pattern. The mottling stays attached to the world through travel and coordinate rebasing, including at remote coordinates. It changes surface color only; geometry, normals and four-step lighting stay the same. The forms, palette and surface variation preserve the current resolution, CFG, guidance and other sampler settings.

Six local palettes combine teal, cobalt, violet, rose, yellow, lime and pale accents
in 128-unit regions across all three axes. Materials belong to their world positions.
The atmosphere blends across region borders while the lower view stays dark.
Returning to a region restores its colors; standing still does not cycle the palette.

Wire collisions follow individual bars, leaving their openings available when the player fits. Shared instanced meshes and camera culling limit the cost of the additions. The original cubes retain their incremental upload path.

There is no ceiling or bottom boundary. The streaming window follows the camera in all three axes, generating nearby chunks and releasing distant ones. Returning to a location recreates the same geometry from its coordinates and world seed. Fog hides the streaming edges. The collision body and spawn search use all three axes; movement is not attached to a terrain height. The flood-fill tests in `tests/test_world.py` check connected open space, and `tests/test_endless_world.py` checks vertical streaming and passage thinning.

That render is hidden in the normal AI view. Every conditioning frame is uploaded to the GPU and encoded by TAESD into a latent. The backend then runs a continuous walk:

- `x0_prev` is the previous predicted clean latent, the memory of the walk.
- Each frame first re-anchors the memory to the proxy latent (`memory_match`, `memory_match_std`, `memory_leash`) so hue and contrast cannot random-walk, then blends it toward the encoded proxy by `guide_strength`, adds noise at a timestep that breathes between `timestep_min` and `timestep_max`, runs one UNet step, and solves back to a clean latent.
- The noise is not independent per frame. It slerps between seeded keyframes over `noise_walk_seconds`, which is what makes forms melt and regrow instead of flickering.
- Prompt changes slerp the CLIP embeddings over `prompt_walk_seconds` rather than cutting. A pair of the local landscape's hues follows the original prompt and changes as the player enters another color region. Space and timed changes avoid recent subject families. New subjects sample the tuning ranges above while preserving the world, camera, noise seed and latent memory. Regional hue updates preserve tuning.
- TAESD decodes the result. The full VAE is never loaded in the realtime path.

`instability` scales both the timestep breathing and the guide wobble; at 0 both are pinned to their configured centres. Everything stays on the GPU as fp16, `channels_last` on the UNet, one uint8 readback per frame.

During startup, the backend prepares CUDA Graphs for the UNet at the current
resolution, with and without CFG. Each frame supplies the current latent,
timestep and prompt embeddings to the recorded GPU operations. This reduces
Python submission delays while preserving the computation and precision.
Unsupported capture uses ordinary inference at the same quality. Changing
resolution releases the graphs and uses ordinary inference until the next launch.
See [the freeze investigation](docs/performance/inference-freezes-20260905.md)
for measured gaps between new AI images and validation results.

With F5 reprojection enabled, the GPU warps the latest generated frame through the
live camera between diffusion updates. With it disabled, the display holds the
latest AI image until the next one arrives. Display FPS and fresh AI image FPS
are separate measurements in the diagnostics overlay.

## Style sheet

```powershell
runtime\python\python.exe tools\style_sheet.py
```

Writes, under `--out` (default `logs/reference/v4`):

- `style_sheet.jpg` -- world seeds x prompts x timesteps, plus `style_sheet.html`
- `walk.jpg` and `walk_every4.jpg` -- a flight with the backend's walk state
  live, on a deterministic frame clock

Useful modes:

| Flags | Output |
|---|---|
| `--walk --walk-frames 96 --seed N --set key=value` | One long walk on one seed with config overrides. `--set backend=proxy_passthrough` shows the proxy alone |
| `--families` | `look_families.jpg`: one labelled row per style family through a single world and spawn, which is how the library's range is judged |
| `--space-resets 8` | Legacy world/style sweep into `space_resets.jpg` and `space_resets.json`: each row changes the world, family presets and diffusion seed. This does not reproduce the current Space control |

## Packaging

Build the lightweight self-installing archive with:

```powershell
powershell -ExecutionPolicy Bypass -File tools\pack_exhibition.ps1 -Zip
```

Output: `dist\RealtimeDiffusionArt.zip`

The normal archive excludes Python, wheels, caches, and model weights. `run.bat` downloads them on the target PC's first launch. It is not an offline deployment.

To build a prepared offline package from a trusted checkout whose runtime and
pinned models have already been installed with `setup_first_run.bat`:

```powershell
powershell -ExecutionPolicy Bypass -File tools\pack_exhibition.ps1 -PreparedOffline -Zip
```

This copies `runtime/python` and the required SD-Turbo/TAESD files, then checks
CUDA imports and generates an image with the copied offline verifier before
writing a new `cache/INSTALL_COMPLETE.txt`. It needs a working NVIDIA GPU on
the packaging machine; `-SkipVerify` cannot be combined with `-PreparedOffline`.
Configured model paths must remain `models/sd_turbo` and `models/taesd`.
Both modes replace only `dist/RealtimeDiffusionArt` and, with `-Zip`, its ZIP.
Prepared packages are several gigabytes. Neither mode copies source caches,
journeys, logs, screenshots, dashboard credentials, private keys, or `.git`.
Use a clean, trusted runtime; packaging cannot identify secrets hidden in
arbitrary application or dependency files. Optional monitoring and storage
dependencies must be installed before packaging if they are needed offline:

```powershell
runtime\python\python.exe -m pip install -r requirements-monitor.txt -r requirements-storage.txt
```

Both packages include the Windows setup, supervisor, status and telemetry tools,
journey upload/restore/export tools, and operator runbooks. Dashboard server code
and credentials are not deployed; its README is included for commissioning reference.
Follow [the Windows host runbook](docs/exhibition-windows.md) for the standard
exhibition account, private Tailscale inbound SSH, optional telemetry and startup.
Privileged `-Apply` must run on the actual target machine. Recheck offline
generation and the interactive display there before opening the exhibition.

See [README_EXHIBITION.txt](README_EXHIBITION.txt) for complete deployment, monitor-selection, offline-operation, and troubleshooting instructions.
