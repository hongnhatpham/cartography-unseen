# Latent Space

Latent Space is a real-time Windows installation. You walk through a procedural blocky landscape and a diffusion model continuously rewrites what you see. The 3D render is never shown: it is a loose steering signal for a continuous walk through latent space. WASD and mouse movement stay responsive between diffusion updates through GPU depth reprojection.

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
| W / A / S / D | Walk and strafe |
| Mouse | Look |
| Shift | Run |
| Escape | Exit |
| Space | New world: new landscape seed and spawn, a new style family with its own sampler settings, a new prompt from it, and a new diffusion seed |
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
| F9 | Cycle prompt auto-advance: off, 12, 24, 48, 96 s. Auto-advance usually stays inside the current family and jumps to another about one time in three; each advance also rotates the world's hue pair |
| F12 | Toggle autowalk (strolls on its own after a minute idle) |
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

While editing a prompt, Enter applies it, Escape cancels, and Ctrl+D restores the default text. Space preserves the current F1 overlay visibility.

Every accepted setting is written back to `config.json` immediately, so the next launch restores it. Freeze and diagnostic views stay session-only. Editing `config.json` in a text editor while the app runs also works: changes are picked up within half a second.

The walker stands at eye height on the terrain. Terrace risers are walls; pass corridors, where the terraces give way to a smooth slope, are how you change level. Standing forms are solid and you slide along them. Overhangs and tilted giants hang above head height, so you walk under them.

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

## Prompts

`prompts.json` holds **style families**, not flat prompts. Each family has a `name`, a
`base` tail shared by its variants, four subject `variants`, and a `settings` block
overriding the sampler for that family. One library entry is built per variant as
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

Space always leaves the current family, so a reset changes the look completely. Automatic
advance usually picks another variant of the same family and jumps families about one time
in four. A family's `settings` may only name live-tunable backend keys; anything else is
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
| `guidance_scale` | `2.6` | CFG. Above 1 activates the negative prompt and costs about 1.4x |
| `timestep_min` / `timestep_max` | `560` / `780` | Sampling timestep window. Higher hallucinates more, lower redraws the proxy |
| `instability` | `0.7` | Scales the timestep breathing (two tones, 13 s and 47 s) and the guide wobble (29 s) |
| `guide_strength` | `0.7` | How far each frame is pulled from its own memory toward the proxy. A floor: instability only raises it |
| `memory_match` | `1.0` | Re-anchor the memory latent's per-channel mean to the proxy latent each frame, so hue cannot random-walk |
| `memory_match_std` | `0.75` | Share of that re-anchoring applied to the spread; below 1 lets contrast breathe |
| `memory_leash` | `0.8` | Clamp the memory to a band this many proxy standard deviations wide. 0 disables |
| `depth_guide` | `0.6` | How much weaker the proxy pulls at the far plane than up close. Near forms stay anchored, the horizon hallucinates, which is what gives an eye-level frame depth |
| `noise_walk_seconds` | `2.0` | Seconds to slerp between seeded noise keyframes. 0 means fresh noise each frame |
| `noise_jitter` | `0.28` | Small variance-preserving jitter mixed into the walk noise |
| `prompt_walk_seconds` | `6.0` | Seconds to slerp CLIP embeddings to a new prompt. 0 hard-cuts |
| `prompt_auto_advance_seconds` | `12.0` | Seconds between automatic library prompt changes. 0 disables. Short enough that a few hundred frames cross a family boundary, which is how a long walk changes subject and colour the way the reference does |
| per-family `settings` | — | Each style family in `prompts.json` overrides the tunable keys above for as long as it is selected. Space persists them; auto-advance applies them live without writing `config.json` |
| `feedback_reprojection` | `false` | Use the camera-aligned previous frame as the walk memory in place of `x0_prev`. The proxy still guides every frame. Costs about 39 ms of CPU per frame |
| `seed` | `12345` | Noise keyframe seed |
| `world_seed` | — | Landscape seed; Space replaces it |
| `movement_speed` / `sprint_multiplier` | `8.0` / `3.0` | Walking speed in world units per second |
| `autowalk_idle_seconds` | `60.0` | Seconds without input before the walker strolls on its own. 0 disables |
| `reprojection` | `false` | Display-side depth warp between diffusion frames |
| `conditioning_fps` | `15` | How often the proxy is captured for the worker |

## How it works

The renderer streams a deterministic landscape of instanced cubes in 64-unit chunks around the camera, in an 11x11 window with a hard cap per chunk. Terrain is a terraced height field with cliff faces and a ravine network. Pass corridors -- a connected ridge network at two scales, where the smooth field shows through the terracing -- are the only way to change level, and a ravine floor is itself a corridor, with its mouth widened where a pass crosses it. Standing forms are thinned so they can never seal a walkable gap. A flood-fill probe verifies every shipped seed: a walker can reach at least 90% of the ground within a 640-unit window without backtracking, and fewer than 3% of cells are dead ends. A graded sky behind the landscape and two skyline monoliths per chunk keep the horizon reading as outdoors. The ground is laid out as a hard light/dark checker on global cell parity: a flat pale floor plane is the sampler's strongest eye-level attractor, and the checker replaces it with converging grid lines and true blacks. Vertical faces carry the same break as two world-space panel grids in the shader, coarse and fine, swinging symmetrically about each face's own tone so a large wall or slab never fills the frame with one flat value. Ravine floors darken further, one contiguous accent vein per world carries the seed's hue on the forms, and slabs, monoliths and fragments come in several sizes with nothing at rooftop scale. Five biome generators (strata, shards, canyons, voxels, monoliths) are blended across space by a low-frequency field, so walking far enough crosses into a different landscape. Each world derives a saturated palette from its seed. There are no roads and no buildings.

That render never reaches the screen. Every conditioning frame is uploaded to the GPU and encoded by TAESD into a latent. The backend then runs a continuous walk:

- `x0_prev` is the previous predicted clean latent, the memory of the walk.
- Each frame first re-anchors the memory to the proxy latent (`memory_match`, `memory_match_std`, `memory_leash`) so hue and contrast cannot random-walk, then blends it toward the encoded proxy by `guide_strength`, adds noise at a timestep that breathes between `timestep_min` and `timestep_max`, runs one UNet step, and solves back to a clean latent.
- The noise is not independent per frame. It slerps between seeded keyframes over `noise_walk_seconds`, which is what makes forms melt and regrow instead of flickering.
- Prompt changes slerp the CLIP embeddings over `prompt_walk_seconds` rather than cutting. A pair of the world's palette hues is spliced into every prompt between its subject clause and its material clause, so each seed reads in its own colour; the pair rotates through the world's accents on each auto-advance. Space draws a family that none of the last few presses used, and redraws the world once if the palette repeats. A new style family also swaps the sampler settings underneath the prompt, so a reset changes regime as well as subject.
- TAESD decodes the result. The full VAE is never loaded in the realtime path.

`instability` scales both the timestep breathing and the guide wobble; at 0 both are pinned to their configured centres. Everything stays on the GPU as fp16, `channels_last` on the UNet, one uint8 readback per frame.

Between diffusion updates the GPU reprojects the latest generated frame through the live camera using the proxy depth that produced it, so movement stays smooth at display rate while diffusion runs at about 10 FPS.

## Style sheet

```powershell
runtime\python\python.exe tools\style_sheet.py
```

Writes, under `--out` (default `logs/reference/v4`):

- `style_sheet.jpg` -- world seeds x prompts x timesteps, plus `style_sheet.html`
- `walk.jpg` and `walk_every4.jpg` -- a ground-level walk with the backend's walk state
  live, on a deterministic frame clock

Useful modes:

| Flags | Output |
|---|---|
| `--walk --walk-frames 96 --seed N --set key=value` | One long walk on one seed with config overrides. `--set backend=proxy_passthrough` shows the proxy alone |
| `--families` | `look_families.jpg`: one labelled row per style family through a single world and spawn, which is how the library's range is judged |
| `--space-resets 8` | `space_resets.jpg` and `space_resets.json`: one row per simulated Space press -- new world seed, new family, new diffusion seed -- which is how "a reset changes everything" is judged |

## Packaging

Build the lightweight self-installing archive with:

```powershell
powershell -ExecutionPolicy Bypass -File tools\pack_exhibition.ps1 -Zip
```

Output: `dist\RealtimeDiffusionArt.zip`

The archive intentionally excludes Python, wheels, caches, and model weights. `run.bat` downloads them on the target PC's first launch.

See [README_EXHIBITION.txt](README_EXHIBITION.txt) for complete deployment, monitor-selection, offline-operation, and troubleshooting instructions.
