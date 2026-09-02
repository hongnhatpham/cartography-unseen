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
| W / A / S / D | Move |
| Mouse | Look |
| Shift | Move faster |
| Q / E | Move down / up |
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
| N | Fixed / random-each-frame seed mode |

While editing a prompt, Enter applies it, Escape cancels, and Ctrl+D restores the default text. Space preserves the current F1 overlay visibility.

## Resolution modes

F10 cycles through `640×384` (default), `384×256`, `512×512`, `768×512`, and `1024×768`. In windowed mode the application window follows the generation resolution. Use `384×256` or `640×384` on a 6 GB exhibition GPU.

## Prompts

Edit `prompts.json` to change the randomized prompt library. The top-level `master_prefix` is automatically prepended to every library, startup, and manually entered prompt. F7 displays the effective prompt at 30% opacity.

## How it works

The procedural road-and-city proxy supplies RGB composition and softened geometry edges to SD-Turbo img2img. This is not a ControlNet build. Proxy depth and camera matrices are stored with each generated frame and used for GPU reprojection between diffusion updates.

One diffusion step gives maximum speed. Additional steps improve detail at lower generation FPS. Fixed seed mode is more temporally coherent; random-each-frame mode intentionally creates more variation and flicker.

## Packaging

Build the lightweight self-installing archive with:

```powershell
powershell -ExecutionPolicy Bypass -File tools\pack_exhibition.ps1 -Zip
```

Output: `dist\RealtimeDiffusionArt.zip`

The archive intentionally excludes Python, wheels, caches, and model weights. `run.bat` downloads them on the target PC's first launch.

See [README_EXHIBITION.txt](README_EXHIBITION.txt) for complete deployment, monitor-selection, offline-operation, and troubleshooting instructions.
