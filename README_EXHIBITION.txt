REALTIME DIFFUSION ART — INSTALLATION AND OPERATOR GUIDE
========================================================

QUICK START
-----------
1. Copy this entire folder to a local drive on the new Windows PC.
2. Connect the PC to the Internet.
3. Double-click run.bat.
4. Leave the first-run setup window open until verification completes.
5. The artwork starts automatically after setup.

The first run downloads and installs a private, portable Python environment,
CUDA-enabled PyTorch, the application libraries, and the pinned SD-Turbo model.
Nothing is installed into the system Python. Later launches are offline and do
not reinstall or redownload anything.

Allow several gigabytes of download traffic and at least 12 GB of free disk
space. Downloads can resume after interruption. Installation time depends on
the Internet connection and disk speed.


SYSTEM REQUIREMENTS
-------------------
- Windows 10 or Windows 11, 64-bit
- NVIDIA GPU with a current NVIDIA display driver
- 6 GB VRAM minimum for the recommended lower-resolution exhibition modes
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
  starts the artwork. The default is a 640x384 window.

run_debug.bat
  Windowed launch with a visible console and diagnostics. Use this when
  troubleshooting. It also performs first-run setup when necessary.

setup_first_run.bat
  Runs or repairs the installation without launching the artwork afterward.
  It is normally called automatically by run.bat.


CONTROLS
--------
W / A / S / D    Move through the world
Mouse            Look around
Shift            Move faster while held
Q / E            Move down / up
Escape           Exit

Space            Generate a new randomized world and prompt
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
F10              Cycle generation resolution/aspect modes
F11              Toggle fullscreen / windowed mode

[ / ]            Reduce / increase reprojection strength
- / =            Reduce / increase diffusion steps (1-4)
, / .            Reduce / increase display sharpening
K / L            Reduce / increase proxy structure lock
B                Cycle geometry-edge softness
G                Cycle geometry-guide strength
N                Toggle fixed seed / new seed for every AI frame

Space preserves the current F1 state. If the F1 overlay is visible, it remains
visible during a world change. If it is hidden, it remains hidden. Space never
flashes the proxy geometry while the replacement generation is pending.


RESOLUTION MODES
----------------
F10 cycles through:

  640x384    default; recommended balance for exhibition use
  384x256    fastest and safest for a 6 GB GPU
  512x512    square composition
  768x512    wider and more detailed, but slower
  1024x768   highest detail and heaviest GPU load

In windowed mode, F10 changes both the AI generation size and the actual window
size. In fullscreen, F10 changes the generation size while the display remains
fullscreen; F11 then returns to a window matching the selected mode. The chosen
mode is saved in config.json.


PROMPTS
-------
prompts.json is editable in Notepad. Its top-level master_prefix is prepended to
every library prompt, including randomized prompts:

  Photoreal abstract rendering, highly detailed

Edit that one value to change the common visual language. Individual entries
under prompts can be added, removed, or rewritten. Changes are read on the next
Space press. The app avoids selecting the exact prompt already in use.

F7 provides a subtle prompt caption without enabling the full F1 diagnostics.


HOW THE IMAGE IS MADE
---------------------
The application renders a fast procedural 3D proxy: a long road, buildings,
lane markings, sidewalks, and randomized architectural details. That proxy RGB
image and softened geometry edges become the img2img input to resident
SD-Turbo. The prompt controls its visual interpretation.

This build does not load ControlNet. Depth does not directly condition the
diffusion model. Instead, every generated frame stores the matching proxy depth
and camera matrices. Between slower diffusion updates, the GPU reprojects the
latest AI frame through the live camera so WASD and mouse movement remain
responsive at display speed.

One diffusion step is fastest but can vary in detail. Two or more steps usually
produce cleaner images at lower generation FPS. The K/L structure control
changes how closely results retain the proxy composition. The fixed seed mode
is more temporally consistent; random-each-frame mode intentionally flickers.


DISPLAY AND MONITORS
--------------------
The default launch is windowed. Press F11 for fullscreen.

Set display_monitor in config.json to a zero-based monitor index, or launch from
a Command Prompt with:

  run.bat --monitor 1

List detected monitors with:

  run_debug.bat --list-monitors

Launch directly in a particular generation mode with:

  run.bat --resolution 768x512


TROUBLESHOOTING
---------------
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

The pinned versions make installations repeatable instead of silently changing
when upstream packages or models are updated.
