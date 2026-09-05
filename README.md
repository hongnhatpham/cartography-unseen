# Latent Space

Latent Space is a real-time Windows installation. You move freely through a dense procedural blocky landscape and a diffusion model continuously rewrites what you see. The 3D render is never shown: it is a loose steering signal for a continuous walk through latent space. WASD and mouse movement stay responsive between diffusion updates through GPU depth reprojection.

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

The NVIDIA display driver must already be installed. A separate CUDA Toolkit is not required. Later launches operate offline.

## Controls

| Key | Action |
|---|---|
| W / A / S / D | Fly along your gaze and strafe |
| Q / E | Descend / rise, independent of gaze |
| Mouse | Look |
| Shift | Fly faster |
| Escape | Exit |
| Space | New prompt; preserve current tuning, world, viewpoint and diffusion seed |
| Shift + R | New diffusion seed |
| P | Edit prompt |
| F1 | Diagnostics overlay |
| F2 | Proxy / generated AI view |
| F3 | Normal / depth / edge diagnostics |
| F4 | Freeze diffusion |
| F5 | Toggle depth reprojection |
| F6 | Return to generated AI view |
| F7 | Prompt caption at the bottom |
| F8 | Toggle feedback reprojection |
| F9 | Cycle prompt auto-advance: off, 12, 24, 48, 96 s. Auto-advance usually stays inside the current family and jumps to another about one time in three; current tuning and palette stay fixed |
| F12 | Toggle idle flight (starts after a minute without input) |
| F10 | Cycle resolution modes |
| F11 | Fullscreen / windowed |
| [ / ] | Reprojection strength |
| - / = | Diffusion steps, 1–4 |
| , / . | Display sharpening |
| I / O | Instability down / up |
| K / L | Guide strength down / up |
| T / Y | Shift the timestep window down / up by 25 |
| N | Cycle noise-walk seconds: off, 2.5, 5, 10, 20, 40 |
| M | Cycle prompt-walk seconds: off, 3, 6, 12, 25, 50 |
| C | Cycle CFG: 1, 1.25, 1.5, 2, 3 |

While editing a prompt, Enter applies it, Escape cancels, and Ctrl+D restores the default text. Space changes only the prompt and preserves all current settings, including CFG and guide strength. It also keeps your position, world seed, diffusion seed and diagnostic state.

Every accepted setting is written back to `config.json` immediately, so the next launch restores it. Freeze and diagnostic views stay session-only. Editing `config.json` in a text editor while the app runs also works: changes are picked up within half a second.

Movement has no gravity, fixed eye height or altitude limit. Look up and hold W to climb, look down to descend, or use Q/E to change height without changing your gaze. The world continues above and below you as well as horizontally. Release the keys to stop. Solid forms still block movement and you slide along them; you can pass above or below them wherever there is space. Idle flight also steers in three dimensions.

The generated image carries the environment, with no navigation contours or proximity reveals. The dense slabs and layered spaces retain the KOSMA reference direction. Interior clutter thins along existing passages, giving them more room while retaining surrounding panels and denser areas. Free flight changes how you traverse them; it does not add sparse floating islands or an open-sky setting.

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

## Prompts

`prompts.json` holds **style families**, not flat prompts. Each family has a `name`, a
`base` tail shared by its variants, four subject `variants`, and a `settings` block
of sampler presets for offline style comparisons. Live prompt changes preserve
your current settings. One library entry is built per variant as
`"<variant>, <base>"`, and `master_prefix` ("corrupted 3D render") is prepended to all of
them.

The eight shipped families are Topology Unknown, Biophilic City, Mangled Data, Wire
Field, Washed Strata, Warped Spacetime, Corrupted Bloom and Datamosh Ravines. They are
deliberately abstract: wires, corruption, data, mangled geometry, warped space and time,
lattice, and the blocky biophilic cityscape of the original ComfyUI sequence. Every entry
is anchored as an eye-level view inside a vast outdoor landscape, and material nouns that
resolve into product shots (chrome, glass, foil) are kept out, so the sampler has room to
guess without landing on something concrete. Occasional human residue is accepted; the
negative prompt names only gamepads, desks, interiors and furniture.

Space chooses a prompt from another family, avoiding recently used families when possible.
Automatic advance usually picks another variant of the same family and jumps families
about one time in three. Neither action applies the family's sampler presets. A family's
`settings` may only name live-tunable backend keys; anything else is
rejected when the library loads. F7 shows the effective prompt at 30% opacity, and the F1
overlay names the current family.

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
| `prompt_auto_advance_seconds` | `24.0` | Seconds between automatic library prompt changes. 0 disables. Short enough that a few hundred frames cross a family boundary, which is how a long walk changes subject and colour the way the reference does |
| per-family `settings` | — | Sampler presets retained for offline style comparisons. Space and auto-advance preserve the current live settings |
| `feedback_reprojection` | `false` | Use the camera-aligned previous frame as the walk memory in place of `x0_prev`. The proxy still guides every frame. Costs about 39 ms of CPU per frame |
| `seed` | `12345` | Noise keyframe seed |
| `world_seed` | — | Landscape seed; preserved when the prompt changes |
| `movement_speed` / `sprint_multiplier` | `8.0` / `3.0` | Flight speed in world units per second |
| `autowalk_idle_seconds` | `60.0` | Seconds without input before idle flight starts. 0 disables |
| `reprojection` | `false` | Display-side depth warp between diffusion frames |
| `conditioning_fps` | `15` | How often the proxy is captured for the worker |

## How it works

The renderer streams a deterministic field of instanced cubes in 64-unit chunks, in an 11x11x11 window around the camera. Thick panels define walls and openings, and a connected channel field opens routes horizontally and vertically. Reefs, lattice bars, shards, columns and overhangs vary across space. Along the channel field, 5% to 65% of interior object groups are omitted to clear passages while retaining the panels. World-space panel patterns and a seeded palette give the model large value breaks and depth cues.

There is no ceiling or bottom boundary. The streaming window follows the camera in all three axes, generating nearby chunks and releasing distant ones. Returning to a location recreates the same geometry from its coordinates and world seed. Fog hides the streaming edges. The collision body and spawn search use all three axes; movement is not attached to a terrain height. The flood-fill tests in `tests/test_world.py` check connected open space, and `tests/test_endless_world.py` checks vertical streaming and passage thinning.

That render never reaches the screen. Every conditioning frame is uploaded to the GPU and encoded by TAESD into a latent. The backend then runs a continuous walk:

- `x0_prev` is the previous predicted clean latent, the memory of the walk.
- Each frame first re-anchors the memory to the proxy latent (`memory_match`, `memory_match_std`, `memory_leash`) so hue and contrast cannot random-walk, then blends it toward the encoded proxy by `guide_strength`, adds noise at a timestep that breathes between `timestep_min` and `timestep_max`, runs one UNet step, and solves back to a clean latent.
- The noise is not independent per frame. It slerps between seeded keyframes over `noise_walk_seconds`, which is what makes forms melt and regrow instead of flickering.
- Prompt changes slerp the CLIP embeddings over `prompt_walk_seconds` rather than cutting. A pair of the world's palette hues is spliced into every prompt between its subject clause and its material clause, so each seed reads in its own colour; the pair stays fixed during prompt changes. Space draws a family that none of the last few presses used. Prompt changes preserve the world, camera, noise seed, latent memory and sampler settings.
- TAESD decodes the result. The full VAE is never loaded in the realtime path.

`instability` scales both the timestep breathing and the guide wobble; at 0 both are pinned to their configured centres. Everything stays on the GPU as fp16, `channels_last` on the UNet, one uint8 readback per frame.

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

The archive intentionally excludes Python, wheels, caches, and model weights. `run.bat` downloads them on the target PC's first launch.

See [README_EXHIBITION.txt](README_EXHIBITION.txt) for complete deployment, monitor-selection, offline-operation, and troubleshooting instructions.
