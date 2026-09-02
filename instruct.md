# INSTRUCT.md — Realtime Diffusion WASD Art Renderer

## 0. Mission

Build a **Windows-first, self-contained realtime diffusion art application** for an exhibition PC.

The application is a WASD-navigable 3D/proxy world whose camera view is continuously transformed by a very fast diffusion model. **Speed, responsiveness, temporal continuity, and structural coherence matter more than image quality.**

Target hardware:

- NVIDIA RTX 4060-class GPU
- Assume as little as **8 GB VRAM**
- Windows 10/11 x64
- NVIDIA GPU driver may be the only external runtime prerequisite
- Exhibition display is likely 1080p, but diffusion should run much lower resolution

Primary goal:

> Make movement feel realtime even if diffusion itself only creates ~10–25 genuinely new frames per second.

Do this with:
- low-resolution 1-step / few-step diffusion,
- a crude geometry/proxy render,
- depth / edge structural information,
- persistent temporal state,
- camera-motion reprojection / frame warping,
- asynchronous rendering and inference.

The final exhibition build must run by copying one project folder to the exhibition PC and launching a `.bat` or `.exe`. It must not depend on the developer machine, user profile paths, Hugging Face cache, Git, Conda, a globally installed Python, or Internet access.

---

# 1. Non-negotiable priorities

Optimize in this exact order:

1. Input latency
2. Stable frame presentation
3. Diffusion throughput
4. Temporal continuity
5. Geometry / camera-motion adherence
6. Portability / offline reliability
7. Image quality

Do NOT sacrifice a large amount of speed for modest image-quality improvements.

Do NOT begin with SDXL, FLUX, SD3, video diffusion, or large multi-ControlNet pipelines.

Start at **512×512 or smaller**.

The application must remain usable while diffusion is late. WASD camera control and screen refresh must never block on model inference.

---

# 2. Core design

Use a decoupled architecture:

```text
                INPUT / CAMERA LOOP
                       60–144 Hz
                           |
                           v
                  +----------------+
                  |  Proxy 3D World |
                  +----------------+
                    |     |      |
                    |     |      +----> motion / camera transform
                    |     +-----------> depth
                    +-----------------> RGB / material IDs / edges
                           |
                           v
                 latest conditioning frame
                           |
                ASYNC DIFFUSION WORKER
                    ~10–25+ FPS target
                           |
                           v
                   latest AI frame
                           |
                           v
            TEMPORAL REPROJECTION / WARP
                           |
                           v
                  DISPLAY LOOP 60+ Hz
```

The display must always use the newest available AI frame, warped/reprojected according to camera motion since that AI frame was generated.

Never wait for the diffusion worker in the render loop.

Use a **latest-frame-wins** design. If the diffusion worker is busy, discard stale conditioning frames instead of queueing them.

---

# 3. Model/backend strategy

Implement the diffusion layer behind a backend interface so multiple fast pipelines can be benchmarked without rewriting the app.

Required backend interface:

```python
class DiffusionBackend:
    def load(self, config): ...
    def warmup(self): ...
    def set_prompt(self, prompt, negative_prompt=""): ...
    def generate(self, conditioning, previous_frame=None, temporal_state=None): ...
    def stats(self) -> dict: ...
    def unload(self): ...
```

Backends should be selectable in `config.json`.

## Backend A — FIRST implementation

**StreamDiffusion + SD-Turbo img2img**

This is the default first path.

Target:
- 512×512
- 1 diffusion step
- CFG off / effectively zero
- fp16
- batch size 1
- model permanently resident on GPU
- prompt embedding cached
- no safety checker
- no model reload between prompt changes
- inference mode enabled
- avoid CPU/GPU transfers wherever possible

The proxy RGB render itself should carry as much geometry information as possible so the first implementation can work **without a full ControlNet**.

Example proxy encoding:
- sky = simple flat value/color
- ground = another value/color
- architecture = clear high-contrast materials
- objects = unique value/color ranges
- depth can modulate brightness or fog
- black/bright edge overlays can expose silhouettes
- normals may modulate simple directional lighting

This intentionally gives img2img a strongly structured source image.

## Backend B — lightweight structural control

Implement only after Backend A is working and benchmarked.

Preferred experiment:
- SD 1.5-class base
- Hyper-SD 1-step or 2-step acceleration
- T2I-Adapter Canny / sketch, if integration is practical

Use this when pure img2img drifts too much from geometry.

## Backend C — full structural control

Optional:
- SD1.5-class model
- Hyper-SD 1-step / 2-step
- Canny ControlNet

Only use full ControlNet if the structural benefit justifies the throughput loss.

Start with **one** control signal only.

Prefer:
1. Canny / edge
2. depth
3. canny + depth only if the first two are insufficient

## Backend D — experimental

Optional benchmark:
- SDXS-512 / available sketch or ControlNet-compatible SDXS implementation

Do not make the whole project dependent on this backend. Treat it as experimental because ecosystem/tooling stability may be lower.

---

# 4. Do not estimate depth with a neural model

The world already exists in 3D.

Do NOT run:
- MiDaS
- ZoeDepth
- Depth Anything
- any AI depth estimator

for the normal realtime path.

Render actual camera depth from the 3D scene.

Expose:
- RGB proxy
- depth buffer
- object/material ID if useful
- edge map
- previous and current camera matrices

Generate edges cheaply from:
- geometry edges, and/or
- Sobel gradient of depth, and/or
- Sobel gradient of luminance/material IDs

Prefer a GPU shader.

---

# 5. Renderer

For the first implementation, use the simplest Windows-compatible renderer that Codex can make reliable.

Preferred stack:
- Python 3.11
- pygame or glfw for window/input
- ModernGL / OpenGL for proxy rendering

Do not introduce Electron, Unreal, Unity, or a browser unless there is a concrete reason.

The renderer must support:
- WASD
- mouse look
- configurable movement speed
- configurable mouse sensitivity
- Shift = faster movement
- Escape = menu / exit
- F1 = debug overlay
- F2 = toggle raw proxy vs AI output
- F3 = toggle edge/depth diagnostics
- F4 = freeze/unfreeze diffusion for debugging

Scene does not need sophisticated graphics.

Start with:
- floor plane
- cubes / walls
- several primitives
- simple directional light
- a few different material colors
- optional OBJ/GLTF loading if straightforward

The world is merely a structural scaffold for diffusion.

---

# 6. Conditioning resolution

Support these fixed internal diffusion resolutions:

- 384×384
- 448×448
- 512×512

Default: **512×512**

Do not dynamically resize every frame.

Use a fixed square internal render target.

The exhibition display may be 1920×1080 or other resolution. Upscale the final AI frame with a cheap GPU method.

Do NOT use an AI super-resolution network in the realtime loop.

Preferred upscale:
- bilinear or bicubic first
- optional simple sharpening pass
- optional lightweight spatial upscaler later

---

# 7. Temporal design

Temporal coherence is critical.

Implement all of the following independently so each can be enabled/disabled and benchmarked.

## 7.1 Previous-frame img2img

When supported by the backend, use a blend of:
- newest proxy conditioning
- previous AI output
- low denoise strength

Expose:
- `img2img_strength`
- `previous_frame_weight`

Avoid resetting to unrelated noise on every frame.

## 7.2 Persistent noise

Maintain slowly evolving noise rather than generating a completely unrelated random noise tensor every frame.

Conceptually:

```python
noise = old_noise * persistence + new_noise * (1.0 - persistence)
```

Default test values:
- `noise_persistence = 0.90`
- also benchmark 0.95, 0.98, and 0.80

This must be configurable.

## 7.3 Seed

Support:
- fixed seed
- random seed on launch
- manual reseed key

Default exhibition behavior should be stable and deterministic enough to avoid violent unrelated flicker.

## 7.4 Reprojection / warping

This is important.

Each generated AI frame must store metadata representing the camera pose and projection used to create it:

```python
GeneratedFrame:
    image
    depth
    view_matrix
    projection_matrix
    camera_position
    camera_rotation
    generation_timestamp
```

As the user continues moving, warp/reproject that generated frame toward the current camera pose.

First implementation may be approximate.

Preferred order:

### Phase 1
Implement simple screen-space affine/perspective warping from camera yaw/pitch/translation.

### Phase 2
Implement depth-aware reprojection:
- reconstruct approximate world position from the source depth
- transform by current camera
- project into current screen
- resolve holes with nearest/bilinear fill
- optional simple dilation/inpainting of tiny holes

The output display loop should run at monitor rate even when no new diffusion frame has arrived.

The reprojection should therefore create intermediate presentation frames like:

```text
0 ms    NEW AI FRAME
16 ms   warped frame
32 ms   warped frame
48 ms   warped frame
64 ms   NEW AI FRAME
```

This is how ~15 FPS diffusion can feel much closer to realtime navigation.

---

# 8. Threading / process model

Do not run everything sequentially.

Use three conceptual loops:

## Main/render loop
- input
- camera update
- proxy render
- update latest conditioning buffers
- display newest AI frame or warped AI frame
- never block on diffusion

## Diffusion worker
- read the newest available conditioning snapshot
- ignore/discard older snapshots
- generate one AI frame
- atomically publish it
- immediately consume the newest conditioning snapshot

## Optional preprocessing worker
Only if CPU preprocessing is actually needed.

Prefer GPU preprocessing.

Use thread-safe latest-value containers, not an ever-growing queue.

There must be no latency accumulation over time.

If diffusion falls behind, frame age should remain approximately one inference cycle, not seconds.

---

# 9. GPU memory rules

Target VRAM budget must work on an 8 GB RTX 4060 where possible.

At runtime:
- keep only one diffusion model loaded
- use fp16
- no gradients
- no optimizer
- no training
- inference mode
- no unnecessary duplicate UNets/VAEs
- aggressively release unused test backends before loading another
- do not keep multiple ControlNets resident unless explicitly required
- preallocate reusable tensors where practical

Add VRAM stats to debug overlay if obtainable.

If OOM occurs:
1. reduce resolution
2. disable structural adapter/control
3. use smaller VAE/decoder option if available
4. unload optional assets
5. never silently fall back to CPU diffusion

CPU diffusion is unacceptable for exhibition runtime.

---

# 10. Performance optimization path

Do not prematurely optimize before measuring.

Implement instrumentation first.

Measure:
- proxy render FPS
- diffusion FPS
- diffusion ms/frame
- displayed FPS
- frame age
- conditioning preprocessing time
- GPU VRAM allocated/reserved
- CPU→GPU copy time if present
- GPU→CPU copy time if present
- reprojection time

Debug overlay example:

```text
DISPLAY       60.0 FPS
DIFFUSION     18.7 FPS
INFERENCE     51.2 ms
REPROJECT      1.8 ms
COND AGE      12.4 ms
AI AGE        34.1 ms
VRAM           6.2 GB
RES           512x512
BACKEND       sd_turbo_stream
STEP          1
```

After baseline works, optimize in this order:

1. Remove avoidable CPU/GPU copies
2. Cache text embeddings
3. Preallocate tensors
4. Keep conditioning on GPU
5. Use CUDA streams if beneficial
6. Use torch compile only if stable and benchmarked
7. Evaluate TensorRT after behavior is correct
8. Optional CUDA/OpenGL interop if readback/upload is a significant bottleneck

Do not use an optimization merely because it sounds faster. Keep it only if benchmarked faster and stable.

---

# 11. TensorRT

TensorRT is an optimization stage, not a requirement for the first working build.

Create the project so the model backend can later have a TensorRT implementation.

If TensorRT is added:
- use static 512×512 engine first
- cache/build the engine during project preparation, not at exhibition launch
- store the engine inside the project
- never require Internet at runtime
- if the engine is GPU-architecture-specific, document that clearly
- preserve PyTorch backend as fallback

The exhibition launcher must select a working backend automatically or from config.

---

# 12. Prompt system

The prompt must be changeable without restarting.

`config.json` should contain at least:

```json
{
  "prompt": "dreamlike surreal painted architectural world, expressive brushwork",
  "negative_prompt": "",
  "backend": "sd_turbo_stream",
  "diffusion_resolution": 512,
  "steps": 1,
  "seed": 12345,
  "movement_speed": 3.0,
  "sprint_multiplier": 3.0,
  "mouse_sensitivity": 0.15,
  "img2img_strength": 0.45,
  "noise_persistence": 0.95,
  "edge_strength": 0.7,
  "depth_strength": 0.0,
  "reprojection": true,
  "target_display_fps": 60,
  "debug_overlay": true
}
```

Hot reload the prompt/config if easy, or provide a tiny overlay UI.

Prompt embeddings must be cached and recomputed only when prompt text changes.

---

# 13. Exhibition UI

Normal exhibition mode must be visually clean.

Required states:

### Exhibition mode
- fullscreen
- AI output only
- mouse captured
- no console window if packaged
- auto-start after initialization
- graceful fallback message if model cannot load

### Debug mode
Allow:
- windowed mode
- proxy RGB
- depth view
- edge view
- AI raw
- AI reprojected
- FPS/latency overlay
- backend information
- seed
- prompt
- VRAM

Provide a command-line flag:

```text
run.bat --debug
```

or equivalent.

---

# 14. Portability / self-contained build

This requirement is critical.

The final exhibition project must look approximately like:

```text
RealtimeDiffusionArt/
│
├── run.bat
├── run_debug.bat
├── config.json
├── README_EXHIBITION.txt
│
├── app/
│   ├── main.py
│   ├── renderer/
│   ├── diffusion/
│   ├── temporal/
│   ├── ui/
│   └── utils/
│
├── models/
│   ├── sd_turbo/
│   ├── optional_hyper_sd/
│   ├── optional_adapter/
│   └── optional_sdxs/
│
├── runtime/
│   └── python/
│       ├── python.exe
│       └── local site-packages / DLLs
│
├── shaders/
├── assets/
├── cache/
├── logs/
├── tools/
│   ├── prepare_models.ps1
│   ├── prepare_runtime.ps1
│   ├── benchmark.py
│   ├── verify_offline.py
│   └── pack_exhibition.ps1
│
└── wheelhouse/
```

Do NOT rely on:
- `%USERPROFILE%\.cache`
- Hugging Face global cache
- system Python
- Anaconda / Conda on exhibition machine
- Git on exhibition machine
- environment variables set manually on exhibition machine
- Internet at exhibition runtime

Set all caches to project-local paths at startup.

Examples:
- `HF_HOME=<project>/cache/huggingface`
- `TRANSFORMERS_CACHE=<project>/cache/huggingface`
- `TORCH_HOME=<project>/cache/torch`
- `XDG_CACHE_HOME=<project>/cache`

Force Hugging Face offline mode in exhibition runtime after models have been prepared:
- `HF_HUB_OFFLINE=1`
- `TRANSFORMERS_OFFLINE=1`

Use local filesystem paths for models.

---

# 15. Portable Python strategy

Do not create a normal developer venv and assume it can be copied blindly.

Preferred approach:

1. Bundle a Windows x64 Python 3.11 runtime under:
   `runtime/python/`

2. Install required packages into that local runtime / local site-packages.

3. Keep a `wheelhouse/` containing the exact wheels needed to rebuild the runtime offline if necessary.

4. `run.bat` must invoke:
   `runtime\python\python.exe`

5. Use only relative/project-root-derived paths.

The final folder must run when moved from:

```text
D:\Development\RealtimeDiffusionArt
```

to something completely different such as:

```text
C:\EXHIBITION\ART01
```

No hard-coded absolute paths.

The NVIDIA driver is allowed to be external.

If a Visual C++ runtime is required, either:
- package the redistributable installer in `tools/`, or
- document it as a one-time prerequisite,
but try to minimize external prerequisites.

---

# 16. Model preparation

During development, create:

```text
tools\prepare_models.ps1
```

It should download/copy all required model files into `models/` using the exact revisions needed.

After preparation, runtime code must load only from local paths.

Also create:

```text
tools\verify_offline.py
```

This script must:
1. enable offline environment flags,
2. verify required model/config/tokenizer files exist,
3. load the selected backend,
4. perform a warmup,
5. generate one test image,
6. exit nonzero with a useful error if anything is missing.

No surprise network access.

---

# 17. Packaging command

Create:

```text
tools\pack_exhibition.ps1
```

It should generate a clean folder such as:

```text
dist\RealtimeDiffusionArt\
```

containing only what the exhibition PC needs.

It should:
- copy app code
- copy shaders/assets
- copy config
- copy bundled runtime
- copy selected model(s)
- copy necessary DLLs
- omit developer caches/logs/tests
- include run scripts
- include README_EXHIBITION.txt
- run the offline verification against the packed folder if practical

Optionally create a `.zip`, but the unpacked folder is the authoritative build.

---

# 18. Startup behavior

`run.bat` must:

1. resolve project root from its own location
2. set local cache environment variables
3. enable offline model loading
4. launch bundled Python
5. launch app fullscreen by default
6. write logs to `<project>/logs/`
7. never open an installer
8. never use pip
9. never download anything

A separate `setup_dev.bat` or PowerShell script may be used on the development computer, but `run.bat` must be exhibition-safe.

---

# 19. Warmup

On startup:
- initialize renderer
- load model
- cache prompt embeddings
- allocate tensors
- perform 3–10 warmup inference passes
- then enter exhibition mode

Do not report benchmark numbers from cold startup.

Show a simple loading screen while warming up.

---

# 20. Benchmark harness

Create:

```text
tools\benchmark.py
```

It must benchmark available backends and resolutions.

At minimum test:

### Backend A
SD-Turbo / StreamDiffusion:
- 384
- 448
- 512
- 1 step

### Optional
Hyper-SD:
- 384
- 448
- 512
- 1 step
- 2 steps

### Optional structural
Canny adapter / ControlNet if installed.

Record:
- warm inference ms
- FPS
- peak VRAM
- average VRAM
- first-frame startup time
- backend load time

Write:
```text
benchmark_results.json
benchmark_results.txt
```

The application may optionally choose the fastest backend that meets a minimum structural score, but simple config selection is acceptable.

---

# 21. Acceptance targets

On RTX 4060-class hardware, aim for:

### Required
- display loop >= 60 FPS when monitor supports it
- WASD feels immediate
- no growing input latency
- no diffusion queue buildup
- application remains interactive if diffusion temporarily stalls
- fully offline runtime
- movable/copyable project directory

### Desired
- >= 15 actual AI frames/sec at 512² with fastest backend
- or >= 20 FPS at 448² / 384²
- <= ~70 ms age for newest displayed AI content during normal movement
- reprojection < 4 ms
- proxy renderer comfortably > 120 FPS by itself

Do not treat these as guaranteed hardware numbers. Measure and adapt.

If 512² misses latency goals, automatically or manually drop to 448² or 384².

---

# 22. Visual quality strategy

Because speed matters more than fidelity:

Prefer:
- painterly
- impressionistic
- abstract
- dreamlike
- surreal
- noisy
- brush-stroke-heavy
- low-detail styles

These naturally hide:
- low resolution
- warping
- temporal errors
- weak geometry
- diffusion artifacts

Avoid visual targets requiring:
- readable text
- exact faces
- photoreal consistency
- precise object identity
- fine architecture
- small mechanical detail

---

# 23. Failure handling

The exhibition application must fail gracefully.

If the model cannot load:
- show a fullscreen readable error
- write detailed log
- mention missing backend/model
- do not attempt Internet download

If CUDA is unavailable:
- show:
  `CUDA-compatible NVIDIA GPU/driver not available`
- do not silently run diffusion on CPU

If selected resolution OOMs:
- catch the error
- clean GPU state
- attempt one lower configured resolution if `auto_resolution_fallback=true`
- log the fallback visibly in debug mode

---

# 24. Coding rules for Codex

- Keep modules small and clear.
- Add type hints where useful.
- Favor boring, reliable code over clever abstractions.
- No network calls from normal app runtime.
- No cloud APIs.
- No telemetry.
- No accounts.
- No database.
- No Docker requirement.
- No services that must be installed.
- No launcher that depends on PATH.
- Use relative paths.
- Use project-local caches.
- Add useful comments around GPU synchronization and temporal logic.
- Avoid unnecessary copies.
- Avoid `torch.cuda.synchronize()` inside the hot path except when benchmarking.
- Do not allocate large tensors every frame if reusable buffers are possible.
- Use atomic/latest-frame state instead of accumulating queues.

---

# 25. Development phases

Implement in this order.

## PHASE 1 — Skeleton
- project structure
- local config
- window
- WASD
- mouse look
- primitive 3D scene
- fixed 512×512 proxy render
- 60+ FPS display
- debug overlay

Do not integrate diffusion until this is stable.

## PHASE 2 — Basic diffusion
- SD-Turbo backend
- one-step img2img
- prompt cache
- asynchronous worker
- latest-frame-wins
- show AI output fullscreen
- benchmark

## PHASE 3 — Temporal continuity
- previous AI frame feedback
- persistent noise
- fixed seed mode
- simple frame warp

## PHASE 4 — Depth-aware reprojection
- save source depth with AI frame
- camera matrix metadata
- depth-aware screen reprojection
- hole fill
- benchmark

## PHASE 5 — Structural conditioning
Only if needed:
- GPU edge extraction
- T2I-Adapter or equivalent lightweight structural conditioning
- compare against pure img2img

## PHASE 6 — Full ControlNet experiment
Only if structural drift remains unacceptable:
- one Canny ControlNet
- benchmark
- keep only if artistic benefit is worth FPS cost

## PHASE 7 — Alternative backend
- SDXS experiment
- Hyper-SD experiment
- benchmark against baseline

## PHASE 8 — Optimization
- remove copies
- preallocate
- optional CUDA interop
- optional TensorRT
- benchmark every change

## PHASE 9 — Exhibition package
- bundled Python
- local models
- local caches
- offline flags
- pack script
- offline verification
- copy packed folder to a different path and test
- test with network disabled

---

# 26. Definition of done

The project is DONE when:

1. I can copy `dist/RealtimeDiffusionArt/` to another Windows PC with a compatible NVIDIA GPU.
2. The PC does not need Python, Git, Conda, Hugging Face, or project dependencies installed globally.
3. Internet can be disabled.
4. I double-click `run.bat`.
5. The app initializes and warms up.
6. It enters fullscreen.
7. WASD + mouse navigate the proxy world.
8. The display continuously shows a diffusion-transformed version of the world.
9. Camera motion feels immediate even when actual diffusion FPS is below display FPS.
10. The app does not accumulate latency.
11. The app can run for hours without increasing VRAM/RAM usage.
12. Debug mode exposes enough timing information to diagnose bottlenecks.
13. All model files and caches required at runtime are inside the project.
14. Moving the project folder to another path does not break it.

---

# 27. First task for Codex

Begin with PHASE 1 and PHASE 2.

Before adding ControlNet or advanced temporal techniques, produce a working application with:

- portable project-relative paths
- 512×512 proxy world
- WASD + mouse navigation
- SD-Turbo / StreamDiffusion-style one-step img2img backend
- asynchronous latest-frame inference
- fullscreen AI output
- debug toggle showing raw proxy and timing
- `config.json`
- `run.bat`
- development setup script
- benchmark script

After it runs, benchmark 384/448/512 and record the results.

Do not guess performance. Measure it.

Only then proceed to temporal reprojection and structural conditioning.

---

# 28. Guiding principle

This is not a conventional high-quality image generator.

Treat diffusion as a **low-resolution neural shader** sitting on top of a realtime navigable geometric scaffold.

Whenever there is a tradeoff:

> choose the solution that makes the user feel more directly connected to WASD movement, even if individual frames look worse.
