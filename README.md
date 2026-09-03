# Latent Space

Latent Space is a real-time Windows installation that transforms a navigable procedural 3D city into continuously generated SD-Turbo imagery. WASD and mouse movement remain responsive between diffusion updates through GPU depth reprojection.

## One-click installation

Requirements:

- Windows 10/11 x64
- NVIDIA GPU with a current display driver
- 6 GB VRAM minimum for the recommended lower-resolution modes
- Internet for the first launch
- Approximately 12 GB free disk space during setup

Download or clone the project, then double-click `run.bat`. On its first launch it automatically downloads and installs:

- Portable Python 3.11.9
- PyTorch 2.6 with its bundled CUDA 12.4 runtime
- Pinned application dependencies
- A pinned fp16 SD-Turbo model snapshot

The NVIDIA display driver must already be installed. A separate CUDA Toolkit is not required. Later launches operate offline.

## Controls

| Key | Action |
|---|---|
| W / A / S / D | Walk along the roads |
| Mouse | Look |
| Shift | Move faster |
| Escape | Exit |
| Space | New randomized world and prompt |
| Shift + R | New diffusion seed |
| P | Edit prompt |
| F1 | Diagnostics overlay |
| F2 | Proxy / generated AI view |
| F3 | Normal / depth / edge diagnostics |
| F4 | Freeze diffusion |
| F5 | Toggle depth reprojection |
| F6 | Return to generated AI view |
| F7 | Prompt caption at the bottom |
| F10 | Cycle resolution modes |
| F11 | Fullscreen / windowed |
| [ / ] | Reprojection strength |
| - / = | Diffusion steps, 1–4 |
| , / . | Display sharpening |
| K / L | Proxy structure lock |
| B | Geometry-edge softness |
| G | Geometry-guide strength |
| C | Cycle CFG: 0, 1.25, 1.5, 2, 3 |
| N | Fixed / drift / random-each-frame seed mode |

While editing a prompt, Enter applies it, Escape cancels, and Ctrl+D restores the default text. Space preserves the current F1 overlay visibility.

The app saves prompt, world and diffusion seeds, resolution, CFG, seed mode, structure and edge controls, sharpening, reprojection, fullscreen, diagnostics visibility, and the F7 prompt caption as soon as they change. The next normal launch restores them. Freeze and diagnostic views remain session-only so the app does not reopen paused or in a troubleshooting view.

## Resolution modes

F10 cycles through `384×216` (default), `640×384`, `384×256`, `512×512`, `768×512`, and `1024×768`. The default renders only 82,944 pixels, then the GPU scales the result to fullscreen. In windowed mode the application window follows the generation resolution.

## Prompts

Edit `prompts.json` to change the randomized prompt library. The top-level `master_prefix` is automatically prepended to every library, startup, and manually entered prompt. F7 displays the effective prompt at 30% opacity.

## How it works

The procedural road-and-city proxy supplies RGB composition and softened geometry edges to SD-Turbo img2img. This is not a ControlNet build. Proxy depth and camera matrices align the prior generated image with the next camera view before feedback, then GPU reprojection keeps movement responsive between diffusion updates.

The proxy city streams deterministic 64-unit chunks around the camera. Curving roads follow the terrain, climb through dense terraces, and occasionally cross ravines on raised decks. Buildings rotate into irregular leftover parcels, vary in height, and carry cable runs across selected roofs. Every proxy element receives an independent seeded color, and each world randomizes the sky and matching fog. The camera stays at walking height and cannot leave the road surface. Only the nearest 49 chunks remain in RAM and one bounded GPU instance buffer; walking back regenerates the same geometry from the world seed.

One diffusion step gives maximum speed. Drift mode correlates each frame's diffusion noise with the last, producing gradual latent motion instead of unrelated redraws. Fixed mode reuses one noise field; random-each-frame intentionally flickers.

CFG values at or below `1.0` use SD-Turbo's fastest no-CFG path and produce the same guidance behavior. Values above `1.0` enable classifier-free guidance and make the negative prompt active. The default `1.25` is deliberately low. Press C to compare it with `0.0` while the app is running. The display preserves SD-Turbo's native output, so prompts and CFG determine the rendering style.

## Packaging

Build the lightweight self-installing archive with:

```powershell
powershell -ExecutionPolicy Bypass -File tools\pack_exhibition.ps1 -Zip
```

Output: `dist\RealtimeDiffusionArt.zip`

The archive intentionally excludes Python, wheels, caches, and model weights. `run.bat` downloads them on the target PC's first launch.

See [README_EXHIBITION.txt](README_EXHIBITION.txt) for complete deployment, monitor-selection, offline-operation, and troubleshooting instructions.
