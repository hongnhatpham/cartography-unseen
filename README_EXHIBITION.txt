LATENT SPACE — INSTALLATION AND OPERATOR GUIDE
==============================================

QUICK START, NORMAL DOWNLOAD-ON-FIRST-RUN PACKAGE
-----------
1. Copy this entire folder to a local drive on the new Windows PC.
2. Connect the PC to the Internet.
3. Double-click run.bat.
4. Leave the first-run setup window open until verification completes.
5. The artwork starts automatically after setup.

The first run downloads and installs a private, portable Python environment,
CUDA-enabled PyTorch, the application libraries, the pinned SD-Turbo model and
the pinned TAESD tiny autoencoder. Nothing is installed into the system Python.
Later launches can operate offline and do not reinstall or redownload anything.
Optional archive uploads need an Internet connection.

For a prepared offline package or a managed exhibition account, use the
MANAGED WINDOWS EXHIBITION section below and docs/exhibition-windows.md.

Allow several gigabytes of download traffic and at least 12 GB of free disk
space. Downloads can resume after interruption. Installation time depends on
the Internet connection and disk speed.


SYSTEM REQUIREMENTS
-------------------
- Windows 10 or Windows 11, 64-bit
- NVIDIA GPU with a current NVIDIA display driver
- 6 GB VRAM minimum; 2.5 GB is used at the default 512x512 mode
- Internet connection for first-run downloads unless using a prepared offline package
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
  opens the artwork and journey map in movable windows for projector placement.
  Generation uses the resolution configured in config.json.

run_debug.bat
  Windowed launch with a visible console and diagnostics. Use this when
  troubleshooting. It also performs first-run setup when necessary.

setup_first_run.bat
  Runs or repairs the installation without launching the artwork afterward.
  It is normally called automatically by run.bat.


MANAGED WINDOWS EXHIBITION
--------------------------
Read docs/exhibition-windows.md before provisioning a dedicated host. It covers
a standard local exhibition account, supervised startup after console logon,
status checks, recovery, and inbound Windows OpenSSH restricted to exact approved
Tailscale peer addresses and public keys. Fleet private keys stay on fleet PCs.
Setup prints a plan by default. Privileged -Apply requires the actual target
Windows machine and its local administrator console; packaging does not apply it.

The normal package downloads Python and models on first run. A prepared offline
package already contains a verified runtime, model weights and a fresh install
marker. Build it on a prepared NVIDIA Windows machine with:
powershell -ExecutionPolicy Bypass -File tools\pack_exhibition.ps1 -PreparedOffline -Zip
This packaging command runs from the source checkout, not the deployed release.
Prepared packages still need the target's NVIDIA driver and commissioning checks:
runtime\python\python.exe tools\verify_offline.py --root C:\Exhibition\Cartography

Use the runbook's full parameter example for tools\setup_exhibition_windows.ps1.
It registers tools\run_exhibition.ps1 as the supervised interactive logon task.
From an approved fleet machine, verify the server fingerprint, then connect:
ssh -i <private-key-on-fleet-PC> exhibition@<host-private-Tailscale-address>
In that remote shell, enter powershell -NoProfile and check:
& C:\Exhibition\Cartography\tools\exhibition_status.ps1 -DeploymentPath C:\Exhibition\Cartography -UserName exhibition -Json
Restart through the scheduled task as documented; the artwork needs the logged-on
console session to appear on the public displays.

Telemetry is optional and never grants remote commands. Before provisioning it:
runtime\python\python.exe -m pip install -r requirements-monitor.txt
Use the runbook's MonitorCredentialPath and MonitorEndpoint setup parameters
to install the protected collector task. Obtain the credential from the trusted
dashboard administrator and keep it outside this folder. dashboard/README.md
describes server-side credential provisioning from the dashboard source checkout.
For offline deployment, install optional dependencies before building the prepared
package. Neither package includes credentials, personal archives or source caches.


TWO-PROJECTOR PLACEMENT AND ARCHIVES
----------------------------------
For the supervised exhibition installation, config.exhibition.json supplies
the game-left/map-right borderless layout and F1 starts closed. Setup preserves
the active copy in cache/exhibition/config.json. Startup reapplies the Windows
high-performance GPU preference. See docs/exhibition-commissioning.md for the
saved setup, verification results and next-exhibition checklist.

With `display_monitor` and `map_display_monitor` configured, both windows open
fullscreen on those displays and F1 starts closed, ready for visitors. To change
placement, press F1, then F to restore both windows. Drag them to the desired
screens and press F again. The selected monitor indices are saved automatically;
the next launch restores both fullscreen windows there with F1 closed.

To rearrange later, open F1, press F to restore the window bounds, drag the
windows, and press F again. F works in either window while F1 is open.
F11 remains a main-window-only fullscreen control.

The map captures one initial image and another every 12 world units of human
travel. It preserves XYZ and the source viewing angle. Ten seconds without
movement or mouse look crossfades the map to the centered Cartography Unseen
title over 1.2 seconds. Automatic flight also starts this fade and stops path
and image capture immediately. The title uses the same Space Grotesk SemiBold
font as the main experience. Underneath are the authors Nhat (Hong) Pham,
Agnieszka Kiejziewicz, Ricardo Arce and Kok Yoong Lim, contributor Tom Nguyen,
and a QR code for https://emergentplay.bynhat.com.

Returning fades the map back in over 0.6 seconds. Input during a fade reverses
it smoothly. The player position stays current during automatic travel, so
returning starts a new stroke at the actual position and leaves a gap for the
automatic movement. Map scene rendering stops once the title has fully appeared.
The title screen is cached and has no continuous animation. F1 placement
instructions are hidden on the idle title screen, including while F1 is open.

When the idle title screen begins to appear, the current nonempty map is
completed once and becomes immediately eligible for upload. Returning interaction
starts a fresh map for the next visitor. Space still archives manually before
clearing the map and selecting a new prompt. Quitting archives any remaining
nonempty map. Failed saves keep the main
window open with an error; resolve the disk problem and press Escape to retry.
Closing only the map window leaves recording running until the main app exits.

Live archive encoding runs in a separate process. Normal saves transfer new
records to that process, keeping accumulated history off the gameplay thread.
The map reuses image geometry as the journey grows. Original pixel quality,
capture spacing and the saved route are preserved. A failed save retains its
pending image and retries when you save/reset or quit again.

Look in journeys/<id>/ for manifest.json, lossless WebP images, complete.json
and index.html. SVG is optional; generate it later with:
runtime\python\python.exe tools\export_journey_svg.py journeys\<id>

Private R2 backup is optional. See docs/journey-storage.md for setup. It uploads
completed maps, verifies them, then retains a 5 GiB local cache. Active or
unverified maps stay local. With less than 1 GiB free, path and image capture
pause; prompts continue recording. Free space or press Space to finish the map
and make it eligible for upload. Capture resumes as a new stroke when space returns.

Open index.html for an interactive map with prompt history and original-image
inspection. If the browser blocks local images, choose Open archive and select
the saved journey folder. The files stay local. The SVG embeds bitmap images; those
images cannot gain detail beyond their original resolution. Keep complete
archive folders when copying or backing up the exhibition records.

journey_map, map_capture_distance and map_idle_seconds in config.json control
map enablement, capture spacing and inactivity timeout at launch. Use
run.bat --no-map to launch only the first-person window. Website publication
is separate from local saving.


CONTROLS
--------
W / A / S / D    Fly along your gaze and strafe
Q / E            Descend / rise, independent of gaze
Mouse            Look around
Shift            Fly faster while held
Escape           Exit

Space            New prompt and random tuning within the ranges below;
                 save/reset the map; keep world, viewpoint and diffusion seed
Enter            Save the displayed AI image as PNG in the project screenshot
                 folder. Excludes all text, diagnostics and the player trail
Settings hotkeys below require the F1 diagnostics overlay to be open.
Closing F1 locks them immediately. Movement, mouse look, Space, Enter, Escape and F1
remain available. Loading and temporary status panels do not unlock settings.

In the first-person window, after ten seconds without player input, the title fades
in at the upper left and controls appear at the lower right. They use Space
Grotesk and IBM Plex Mono, matching Emergent Play. The fade takes 1.2 seconds;
keyboard or mouse input fades them out in 0.25 seconds. Held keys and mouse
buttons count as activity. Automatic flight and prompt changes do not hide
them. Diagnostics, prompt editing and status panels hide them. The controls
sit above the prompt caption. Fonts are bundled for offline use; cached text
fades on the GPU without changing the generated image.
Title and control text are 2.25 times their original size, fitted proportionally
on smaller windows. Enter appears in the visitor instructions.

Screenshots retain the current crop, sharpening and reprojection. Each gets a
timestamped filename. PNG compression and disk writing run in the background,
with one save pending at a time. Switch to the generated AI view before saving.
Inside the prompt editor, Enter applies text instead of taking a screenshot.

Shift + R        Choose a new diffusion seed
P                Edit the prompt
Enter            Apply a prompt while editing
Escape           Cancel prompt editing
Ctrl + D         Restore the default prompt text while editing

F1               Toggle the diagnostics overlay
F                Fullscreen both windows / restore placement, with F1 open
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
                 sharing history with Space. Each advance samples new tuning
H / J            Fog distance nearer / farther, 10 units per press. Nearer
                 means more fog; farther means less. F1 shows the distance
V                Toggle the player trail. The last ten seconds of travel
                 fade oldest-first; turning it off clears the route.
                 The trail is 15 pixels wide, ten times its original width
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

Each new subject independently samples these settings:

  Timestep       Lower bound 80-200; upper bound 280-400.
                 Both bounds are sampled independently; the window width varies.
  Sharpness      1.00-2.00
  Instability    0-40%
  Guide strength 65-100%

Space, timed advance, changed text accepted in the prompt editor, and external
prompt edits all trigger this variation. CFG, fog, position, world and diffusion
seeds, diagnostic state, and other settings stay unchanged. Automatic variation
works with F1 closed; manual settings hotkeys still require F1. Regional color
changes during travel do not sample new tuning. The picture transitions using
the current prompt-walk duration.

Manual settings changes save to config.json immediately. Space and accepted
prompt edits save the prompt and its sampled settings. Timed prompt choices
and their tuning remain session-only. Startup uses the saved settings until
a new subject is chosen. Freeze and diagnostic views remain session-only so
the artwork does not reopen paused or in a troubleshooting view.


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
  settings   sampler presets for offline style comparisons; live subject
             changes use the random tuning ranges above

master_prefix ("corrupted 3D render") is prepended to every prompt.

The twelve shipped families are Topology Unknown, Biophilic City, Mangled Data,
Wire Field, Washed Strata, Warped Spacetime, Corrupted Bloom and Datamosh
Ravines, plus Cellular Karst, Root Networks, Membrane Folds and Distant Presences.
Distant Presences adds four views with small human silhouettes and distant
figures, partly hidden by fog or structures. The three
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
prompt to a new one; F9 sets how often a new library prompt is chosen. New
subjects sample the tuning ranges above. Set
prompt_walk_seconds to 0 for an immediate transition, or the auto-advance
interval to 0 to stop automatic changes. F7 provides a subtle prompt caption
without the full F1 diagnostics.


HOW THE IMAGE IS MADE
---------------------
An optional pale filament marks your recent route. Look back to see it just
below the path you travelled. Older sections fade first; the route disappears
after ten seconds, including when you stand still. Structures hide the trail
where it passes behind them. With F1 open, V toggles it and saves your choice. It is drawn
after the AI image, so it fades on time without changing the generation.
The visible end stays four units back along your route, measured from the
displayed frame's capture time so it cannot extend ahead of an older AI view.

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
prompt interpolation and do not restart the automatic prompt timer or sample
new tuning.
Your saved prompt, default prompt and library text stay unchanged. Regional hue
updates keep the current resolution, CFG, guide strength and other sampler settings.
Space changes the subject and samples the tuning ranges above.

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
The configured two-window launch starts both windows fullscreen on their saved
display indices with the F1 overlay closed. Use F1 then F to enter placement
mode; pressing F again saves both current displays for the next launch. Legacy
configs without map_display_monitor still start in placement mode. With
--no-map, the first-person window follows the fullscreen setting in config.json.

Set display_monitor in config.json to a zero-based monitor index, or launch from
a Command Prompt with:

  run.bat --monitor 1

List detected monitors with:

  run_debug.bat --list-monitors

Launch directly in a particular generation mode with:

  run.bat --resolution 768x512


TROUBLESHOOTING
---------------
Remote operators should begin with docs\exhibition-remote-operations.md. It
documents the hp-zbook SSH boundary, authoritative health snapshots, supervisor,
live watcher, power baseline, storage checks and two-projector verification.

Raw AI frames reuse their GPU image between updates, avoiding repeated uploads.
Reprojection skips unused live-camera renders while retaining its display-rate
interpolated depth. Bounded caches reuse exact world-field samples and chunk
ordering. Geometry, colors, collision data and rendering settings are unchanged.
Fixed-image presentation measured 0.65 to 0.45 ms per refresh with identical
pixels; a six-boundary CPU streaming profile fell from 3.10 to 2.43 seconds.
These component savings do not establish a higher AI frame rate.

Chunk construction now yields after three milliseconds or three chunks. Startup
still fills the whole 11x11x11 window. Packed geometry and collision arrays own
the loaded world; redundant source objects are released, with only 128 recent
source chunks cached. Revisits regenerate identical geometry. Pygame starts
only display and font support, avoiding unused joystick and audio device work
inside Windows event polling. Keyboard and mouse controls are unchanged.
Input polling releases Python's shared lock while waiting on Windows, so the
AI worker can continue. Text composition starts only in the prompt editor.
The 15-pixel trail uses triangles, avoiding driver-specific line-width caps.

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


CONTINUOUS-RUN PERFORMANCE CHECK
--------------------------------
From the project folder, run:

  runtime\python\python.exe tools\soak_test.py --minutes 30

The thirty-minute timer starts at the first generated image. A private copy of
your config enables idle flight after two seconds; your saved settings remain
unchanged. The app exits normally when the timer finishes. Escape or closing
the window ends the run early and records an incomplete result.

Find summary.json, minutes.csv, minutes.jsonl and resources.jsonl in the new
timestamped folder under logs\performance. These record display and generation
timings, stalls, prompt work, garbage collection, process memory, NVIDIA GPU
telemetry, worker errors and automatic resolution fallbacks.
AI publication gaps measure how long the same generated image remains current,
including the final gap if generation stops. Inner display timings separate
image uploads, text overlays, trail drawing and the display swap.
Diagnostics reuse unchanged text and update only the affected texture regions;
their appearance and ten-updates-per-second limit remain the same.
stalls.jsonl adds timestamped events for display pauses above 50 ms, including
overlapping render, navigation, event-polling, frame-wait, AI and GC activity.
Add --fixed-settings to disable automatic prompt changes and tuning variation
and hold the initial AI palette constant for a controlled comparison. The
rendered world still changes colour with location. Your saved config is kept.
Normal monitoring preserves production input handling. On Windows, a separate
thread owns the window and SDL input. Drawing reads buffered input and can
continue during a native event wait. Monitoring still times native waits on
their owning thread. --split-poll is rejected on Windows because it would pump
input from the drawing thread; it remains available on other platforms.

For repeatable comparisons, run:

  runtime\python\python.exe tools\replay_performance.py --seconds 120 --max-stall-ms 100

This keeps a private config copy at 384x256, CFG 1.5, with the prompt and seed
held constant. The same camera route starts at the first AI image. Collision
queries run, but their displacement is overridden to keep the route independent
of frame timing. Escape and closing the window still end the run.
timings.csv records every display interval, generation duration and new AI image
publication interval. --max-ai-gap-ms 250 is checked by default, so repeating an
old image cannot hide an AI freeze. The command fails on either threshold, an
incomplete run, lost measurements or a generation resolution fallback.

For a two-hour growing-map test with real AI images and automatic prompt
changes, use a new output folder:

  runtime\python\python.exe tools\long_map_session.py --seconds 7200 --output logs\performance\map-soak-new

This simulates human traversal on the repeatable route. It records process
memory, GPU telemetry, archive progress and both window timings. Keep other
GPU workloads stopped. Disable your window manager separately if comparing
against runs made without it. To stop early while preserving final exports,
create an empty stop-requested file inside the run's output folder. An early
stop is recorded as incomplete. Analyze a finished run with:

  runtime\python\python.exe tools\summarize_long_map.py logs\performance\map-soak-new

Use --interrupted only to recover available measurements after an abnormal
termination. It does not certify the missing tail or final archive.

For normal play with automatic prompts and your current diagnostics settings,
use the lightweight recorder:

  runtime\python\python.exe tools\replay_performance.py --normal --seconds 600

This changes only idle-flight activation to two seconds in a private config.
It records display, new-image publication and generation intervals without the
detailed stage, garbage-collection or GPU telemetry probes of soak_test.py.
Use it to check visible pauses; use the detailed soak tool to investigate them.
Detailed stall formatting runs on the sampler rather than the display thread.
Investigation and results:

  docs\performance\window-input-stutter-20260905.md
  docs\performance\inference-freezes-20260905.md

Thirty minutes can reveal early slowdown or memory growth. It does not prove
a week of unattended operation. Use --minutes 10080 for a full week on the
exhibition computer. After monitoring, run run.bat for normal operation.


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
