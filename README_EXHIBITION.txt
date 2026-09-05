LATENT SPACE — INSTALLATION AND OPERATOR GUIDE
==============================================

QUICK START
-----------
1. Copy this entire folder to a local drive on the new Windows PC.
2. Connect the PC to the Internet.
3. Double-click run.bat.
4. Leave the first-run setup window open until verification completes.
5. The artwork starts automatically after setup.

The first run downloads and installs a private, portable Python environment,
CUDA-enabled PyTorch, the application libraries, the pinned SD-Turbo model and
the pinned TAESD tiny autoencoder. Nothing is installed into the system Python.
Later launches are offline and do not reinstall or redownload anything.

Allow several gigabytes of download traffic and at least 12 GB of free disk
space. Downloads can resume after interruption. Installation time depends on
the Internet connection and disk speed.


SYSTEM REQUIREMENTS
-------------------
- Windows 10 or Windows 11, 64-bit
- NVIDIA GPU with a current NVIDIA display driver
- 6 GB VRAM minimum; 2.5 GB is used at the default 512x512 mode
- Internet connection for the first run only
- At least 12 GB free disk space during setup

The setup installs PyTorch 2.6 with its bundled CUDA 12.4 runtime. A separate
CUDA Toolkit installation is NOT required. The NVIDIA display driver itself is
hardware-specific and is not installed automatically. If setup reports that no
NVIDIA driver is available, install the current driver from NVIDIA or the laptop
manufacturer, reboot if requested, and run run.bat again.


LAUNCH FILES
------------
run.bat
  Normal one-click launch. It performs first-run setup when necessary, then
  starts the artwork fullscreen at the 512x512 generation resolution.

run_debug.bat
  Windowed launch with a visible console and diagnostics. Use this when
  troubleshooting. It also performs first-run setup when necessary.

setup_first_run.bat
  Runs or repairs the installation without launching the artwork afterward.
  It is normally called automatically by run.bat.


CONTROLS
--------
W / A / S / D    Fly along your gaze and strafe
Q / E            Descend / rise, independent of gaze
Mouse            Look around
Shift            Fly faster while held
Escape           Exit

Space            New prompt; keep current tuning, world, viewpoint and
                 diffusion seed
Shift + R        Choose a new diffusion seed
P                Edit the prompt
Enter            Apply a prompt while editing
Escape           Cancel prompt editing
Ctrl + D         Restore the default prompt text while editing

F1               Toggle the diagnostics overlay
F2               Toggle raw proxy / generated AI view
F3               Cycle normal / depth / edge diagnostic views
F4               Freeze / unfreeze diffusion
F5               Toggle depth reprojection
F6               Return directly to generated AI view
F7               Toggle the current prompt at the bottom at 30% opacity
F8               Toggle feedback reprojection: the previous frame, aligned
                 to the camera, becomes the walk memory (slower, stickier)
F9               Cycle prompt auto-advance: off, 12, 24, 48, 96 s. Advancing
                 changes family and avoids the last three families when possible,
                 sharing history with Space. Current tuning stays fixed
H / J            Fog distance nearer / farther, 10 units per press. Nearer
                 means more fog; farther means less. F1 shows the distance
V                Toggle the player trail. The last five seconds of travel
                 fade oldest-first; turning it off clears the route
F12              Toggle idle flight. With no input for a minute the camera
                 flies on its own; any key or mouse movement takes over
F10              Cycle generation resolution/aspect modes
F11              Toggle fullscreen / windowed mode

[ / ]            Reduce / increase reprojection strength
- / =            Reduce / increase diffusion steps (1-4)
, / .            Reduce / increase display sharpening
I / O            Reduce / increase instability
K / L            Reduce / increase guide strength (how much proxy shows through)
T / Y            Shift the timestep window down / up by 25
N                Cycle noise-walk seconds: off, 2.5, 5, 10, 20, 40
M                Cycle prompt-walk seconds: off, 3, 6, 12, 25, 50
C                Cycle CFG: 1, 1.25, 1.5, 2, 3

Look up and hold W to climb, or look down to descend. Q/E changes height
without changing your gaze. Release the keys to stop: there is no gravity
or fixed eye height. There is no altitude limit: the world continues above
and below you as well as horizontally. Forms are solid and you slide along
them; pass above or below wherever there is space. Idle flight also steers
up and down.

The generated image carries the dense slab and terrace-like environment.
There are no navigation contours or proximity reveals. Interior clutter
thins along existing passages, giving them more room while retaining
surrounding panels and denser areas. New geometry streams in as you fly
up or down, keeping the environment around you at every height.

Space changes only the prompt. CFG, guide strength and all other current
settings stay as you set them. Your position, world seed, diffusion seed and
diagnostic state also stay unchanged. The picture transitions using the current
prompt-walk duration. Automatic prompt changes also preserve your tuning.

Every accepted setting is written to config.json immediately and restored on the
next launch. Freeze and diagnostic views remain session-only so the artwork does
not reopen paused or in a troubleshooting view.


RESOLUTION MODES
----------------
F10 cycles through, with measured one-step speed on an RTX 4070 Laptop
(CFG 1, 6 warmup frames, 20 timed frames, idle GPU):

  512x512    default; 99 ms/frame, 10.1 diffusion FPS
  640x384    98 ms/frame, 10.3 FPS
  512x384    96 ms/frame, 10.4 FPS
  384x256    88 ms/frame, 11.3 FPS
  768x512    129 ms/frame, 7.8 FPS
  1024x768   286 ms/frame, 3.5 FPS

At one step the frame is bound by kernel launches and TAESD rather than by
pixel count, so the four smaller modes are within about 10% of each other.
Dropping resolution to recover frame rate buys almost nothing; use F10 for
VRAM headroom and for framing. To go faster, lower CFG (C) toward 1: the
shipped guidance_scale of 2.2 costs about 1.4x, so 512x512 runs at 133 ms /
7.5 FPS as configured.

Peak VRAM ranges from 2.33 GB to 2.75 GB across those modes. In windowed mode,
F10 changes both the AI generation size and the window size. In fullscreen, F10
changes the generation size while the display stays fullscreen; F11 then returns
to a window matching the selected mode. The chosen mode is saved in config.json.


PROMPTS
-------
prompts.json is editable in Notepad. It holds STYLE FAMILIES rather than a flat
list of prompts. Each family has:

  name       what the F1 overlay shows, e.g. "Warped Spacetime"
  base       the tail shared by the family's variants
  variants   four subjects; one library prompt is built per variant as
             "<variant>, <base>"
  settings   sampler presets for offline style comparisons; live prompt
             changes preserve your current tuning

master_prefix ("corrupted 3D render") is prepended to every prompt.

The eleven shipped families are Topology Unknown, Biophilic City, Mangled Data,
Wire Field, Washed Strata, Warped Spacetime, Corrupted Bloom and Datamosh
Ravines, plus Cellular Karst, Root Networks and Membrane Folds. The three
organic families add porous formations, branching strands and curved sheets
without the voxel wording of the original subjects. There are 44 variants.
They are abstract on purpose: wires, corruption, data, warped space
and time, lattice, and the blocky biophilic cityscape of the original piece.
Material nouns such as chrome or glass are kept out because they resolve into
product shots and furnished rooms.

Space and automatic advance choose a prompt from another family, avoiding
the last three families when possible. Neither action applies the family's
sampler presets.

Families and variants can be added, removed or rewritten. Changes are read on the
next Space press or automatic prompt change. A family's settings block may only
name tunable sampler keys; anything else makes the library fail to load, and the
app logs the reason and keeps the prompt it already has.


WALKING THE LATENT SPACE
------------------------
Four keys control how unstable the picture is. All of them are live and saved.

  guide_strength (K / L)    How far each frame is pulled back toward the 3D
                            render. Low means the model ignores your geometry
                            and hallucinates freely; high means it traces it.

  timestep window (T / Y)   How much noise is added before each denoise step.
                            High (780-900) melts the geometry into terraces,
                            ravines and floating fragments. Low simply redraws
                            the render.

  instability (I / O)       Scales the slow breathing of the timestep and the
                            guide strength. At 0 both are pinned.

  noise walk (N)            Seconds to travel between noise keyframes. This is
                            what makes shapes melt and regrow instead of
                            flickering. Off means unrelated noise every frame.

prompt_walk_seconds (M) is how long the picture takes to morph from the current
prompt to a new one; F9 sets how often a new library prompt is chosen. Space
and automatic prompt changes keep the current sampler settings. Set
prompt_walk_seconds to 0 for an immediate transition, or the auto-advance
interval to 0 to stop automatic changes. F7 provides a subtle prompt caption
without the full F1 diagnostics.


HOW THE IMAGE IS MADE
---------------------
An optional pale filament marks your recent route. Look back to see it just
below the path you travelled. Older sections fade first; the route disappears
after five seconds, including when you stand still. Structures hide the trail
where it passes behind them. V toggles it and saves your choice. It is drawn
after the AI image, so it fades on time without changing the generation.

Fog distance controls where structures fully fade into the atmosphere. The
default is 196 world units, matching the original fog, with a range of 40–300.
H brings the fog closer to soften distant colors; J pushes it farther away.
The value saves automatically and survives prompt changes and restarts.

The application renders a dense procedural field of thick panels, reefs,
lattice bars, shards, columns and overhangs. Curved wire sheets, ribbed arches,
open cages and rounded solids add variety among the existing blocks. Openings
connect in three dimensions, with less interior clutter along passages. Each
world derives its palette and sky from a seed, with stronger regional color
across surfaces and the atmosphere. The surface checker texture is removed;
light and dark panels still give the AI large value breaks and depth cues.
The renderer keeps a finite window of nearby chunks and streams new ones
as you move in any direction. There is no ceiling or bottom boundary.
Returning to a location recreates the same geometry and colors.
Fog hides the streaming edges.

The effective prompt keeps your original wording and adds the local landscape's
hue pair. Curved and wire forms come from the proxy, without appending the same
shape phrase to every prompt. As you travel, colors change with the landscape
in any direction, including upward and downward. Color cues use the existing
prompt interpolation and do not restart the automatic prompt timer.
Your saved prompt, default prompt and library text stay unchanged. Resolution,
CFG, guide strength and the other sampler settings remain as you set them.
Space still changes only the prompt.

That render is hidden in the normal AI view. TAESD encodes it on the GPU into
a latent, which steers a continuous walk: the
previous clean latent is re-anchored to it (so colour and contrast cannot
drift away), blended toward it, noised at a breathing timestep,
pushed through one SD-Turbo UNet step, and decoded by TAESD. The noise itself
travels between seeded keyframes instead of being redrawn each frame, and prompt
changes interpolate rather than cut. The full VAE is never loaded.

Between diffusion frames the GPU reprojects the latest generated image through
the live camera, using the depth of the render that produced it, so movement
stays smooth at display rate while diffusion runs at about 10 frames per second.

CFG at or below 1 uses the fastest no-CFG path. Above 1 the negative prompt
becomes active and each frame costs roughly 1.4x. The default is 2, which is
what keeps text, gamepads and interiors out of the picture.


DISPLAY AND MONITORS
--------------------
The default launch is fullscreen. Press F11 for a window.

Set display_monitor in config.json to a zero-based monitor index, or launch from
a Command Prompt with:

  run.bat --monitor 1

List detected monitors with:

  run_debug.bat --list-monitors

Launch directly in a particular generation mode with:

  run.bat --resolution 768x512


TROUBLESHOOTING
---------------
The normal AI view draws the proxy at the conditioning rate, while movement and
world streaming keep updating at the display rate. Chunk loading uploads new
geometry incrementally. These optimizations preserve the resolution, image
settings and world density. F1 reports display and diffusion FPS separately;
60 display FPS does not mean 60 newly generated AI images per second.

First-run setup logs:
  logs\first_run_setup_YYYYMMDD_HHMMSS.log

Application logs:
  logs\realtime_diffusion_YYYYMMDD_HHMMSS.log

If setup is interrupted, run run.bat again. Existing downloads are reused or
resumed. To repair the setup manually, double-click setup_first_run.bat.

If CUDA is unavailable:
1. Confirm the computer has an NVIDIA GPU.
2. Install/update the NVIDIA display driver.
3. Reboot if the driver installer requests it.
4. Run setup_first_run.bat again.

If 1024x768 runs out of VRAM, use F10 to select 640x384 or 384x256. The worker
also attempts an automatic lower-resolution fallback after a CUDA OOM.

To confirm an offline install without launching the artwork:

  runtime\python\python.exe tools\verify_offline.py


OFFLINE USE AFTER INSTALLATION
------------------------------
After the first-run marker cache\INSTALL_COMPLETE.txt exists, run.bat launches
without Internet access. Model and package loading are forced offline. Keep the
runtime, models, cache, app, shaders, config.json, and prompts.json folders
together; all paths are derived from this package folder.


PINNED COMPONENTS
-----------------
Python:    3.11.9 portable runtime
PyTorch:   2.6.0 + CUDA 12.4
SD-Turbo:  stabilityai/sd-turbo revision
           b261bac6fd2cf515557d5d0707481eafa0485ec2
TAESD:     madebyollin/taesd revision
           614f76814bbe30edbe2e627ace1c2234c81a2c0e

The pinned versions make installations repeatable instead of silently changing
when upstream packages or models are updated.
