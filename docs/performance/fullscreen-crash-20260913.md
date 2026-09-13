# Fullscreen crash repair — 13 September 2026

**Final commissioning outcome:** the pointer repair below fixed a native crash,
but preliminary process-survival tests did not establish that both physical screens
remained visible. Later testing reproduced a black output with the Intel OpenGL
renderer. The final installation uses NVIDIA OpenGL and borderless fullscreen on
both outputs, with a persistent startup GPU preference and an explicit left/right
preset. The operator confirmed visibility and the subsequent
[two-hour operational test](exhibition-soak-20260913.md) passed. The earlier live
fullscreen measurements below are preliminary evidence, not the final display test.

The exhibition app contained a use-after-free bug in its Pygame window handling. Closing F1 with the journey map enabled calls `ProxyRenderer.focus()`. That method created a temporary `Window.from_display_module()` wrapper. Pygame 2.6.1 stores that wrapper as a borrowed native `pg_window` pointer, and its borrowed-wrapper destructor neither retains it nor clears the pointer. Subsequent keyboard, mouse and window events can access freed Python memory. Window resizing and initial placement also created temporary wrappers.

## Evidence

- Windows Application Error recorded access violations (`0xc0000005`) in `python311.dll` at 11:29:28, 11:32:52 and 11:42:12. The supervisor recorded the corresponding exit code `-1073741819`.
- The last crash dump's stack contains Python and NumPy allocation/conversion frames, consistent with heap corruption rather than a Python exception. Existing dumps were preserved in the user's local CrashDumps directory.
- An isolated test on the installed runtime confirmed that the retained window wrapper and SDL's `pg_window` pointer initially matched. Calling the old temporary-wrapper focus expression changed the native pointer to a different, released wrapper. The test restored the valid pointer before event processing.
- The installed Pygame source explains this behavior: [wrapper creation and borrowed-wrapper destruction](https://github.com/pygame/pygame/blob/2.6.1/src_c/cython/pygame/_sdl2/video.pyx) and [event conversion using pg_window](https://github.com/pygame/pygame/blob/2.6.1/src_c/event.c).

## Repair

A single retained `WindowPlacement` now owns the wrapper from initial window creation through SDL shutdown. The Windows input thread and renderer share that placement. Focus and resizing reuse its wrapper. Initial fullscreen uses that placement as well. This preserves the existing threading model, GL context and projector controls.

Native Python crash tracing now writes `logs/realtime_diffusion_<timestamp>.fault.log`, including all thread stacks, even without a visible console.

Changed files: `app/main.py`, `app/window_loop.py`, `app/window_placement.py`, `app/renderer/proxy_renderer.py`. Added `tools/check_fullscreen.py` for live regression testing using a private configuration and separate test archives under its log folder.

## Verification

- Compilation succeeded for all four modified application modules.
- Renderer regression: 30 fullscreen/restore cycles, 60 focus calls, and 30 resizes passed. Native pointer identity and event-window identity stayed correct through garbage collection and rendering.
- Live AI regression: 180.012 seconds after the first AI frame; 41 fullscreen transitions through F/F11; 21 focus calls through F1; 62 native-pointer checks; 2,261 generated AI frames and 11,506 display frames including startup. Both main and map windows survived; the main window ended fullscreen. No errors or native crash traces.
- [Machine-readable live test result](../../logs/fullscreen-check-20260913_120340/result.json).
- Explicit direct-fullscreen startup with the proxy backend and 30 smoke frames also exited successfully.

The live regression can be repeated with `runtime/python/python.exe tools/check_fullscreen.py` while normal supervised operation is stopped. It intentionally exercises real fullscreen windows and takes three minutes after AI startup.

## GPU findings

During the initial crash investigation this machine rendered OpenGL on Intel Arc
Pro Graphics and ran AI on the NVIDIA GeForce RTX 4070 Laptop GPU (8 GiB, driver
581.80). No NVIDIA driver-reset event coincided with the three recorded application
crashes. Older NVIDIA/WHEA events existed on the preceding day. Subsequent display
testing required selecting the high-performance NVIDIA GPU for both bundled Python
executables. `tools/configure_exhibition_gpu.ps1` now reapplies that preference from
the exhibition user's supervisor before window creation. No driver was changed.

## Recovery and rollback

Original source files and `changes.diff` are preserved in `C:\Exhibition\Commissioning\fullscreen-fix-20260913`. Each original is named after its source file with `.original` appended. The original `proxy_renderer.py` belongs under `app/renderer`; the other three belong under `app`.

Normal operation was restored through `CartographyExhibition-exhibition` after removing the temporary maintenance flag. Production configuration and visitor archives were preserved. The final operational validation lasted two hours, not multiple days.
