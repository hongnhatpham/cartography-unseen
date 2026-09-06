# Journey map

Status: direction A approved by Nhat on 5 September 2026. Implemented on
6 September 2026: live map, distance capture, prompt history, archive/SVG export,
local interactive archive viewer and coordinated projector fullscreen.
Publishing archives on the Emergent Play website remains future work.
The idle title treatment uses the approved A crossfade from the
[transition preview](https://reports.bynhat.com/r/31fe3a0b08f5cb892e69/).

## Implemented controls and defaults

Both windows start windowed with the F1 operator overlay open. Drag their title
bars to the desired screens. F toggles both fullscreen on their current monitors
and restores their previous bounds when pressed again. It requires the overlay
open and works from either window. Closing F1 returns input to the main window.
F11 remains a main-window-only control.

The defaults are one image at the start of each human segment, then 12 world
units between captures, a ten-second inactivity grace period and a three-minute
orbit at up to 30 FPS. Inactivity crossfades the map to the project title;
placement instructions can appear while the operator overlay is open.

Archives are written under `journeys/<id>/` with original PNGs, `manifest.json`,
an embedded-image `map.svg` and an interactive `index.html`. Five-second
checkpoints and image saves preserve partial journeys during play. Normal quit
and Space finish the archive; failure retains it for retry. Checkpoints are
recovery records, not an automatic resume of a crashed exhibition session.

Space continues to change the prompt while keeping geometry and position.
Map-related configuration applies at launch. See the [operator instructions](../README.md#two-projector-journey-map).

This document is the current brief. It supersedes the prototype's suggestions
where they differ, especially its gaze-triggered capture and archive reset rules.

Design reference: [Mapping the Blackbox prototypes](https://reports.bynhat.com/r/14db5e2b4d0744361b05/).
The selected treatment is **A, Image field**. The local comparison is
[journey-map.html](prototypes/journey-map.html).

## Approved experience

The first-person traversal remains the main experience. A separate window shows
the journey from above, with each window independently fullscreen on its own
screen or projector.

The map follows direction A: black space, a fine movement path, and generated
image planes placed along the journey. Keep the composition sparse. Preserve
actual world XYZ coordinates, including climbs and descents. Place each image
at the position that produced it, with its recorded viewing orientation,
including pitch. The image planes stay fixed in the world while the overview
camera slowly orbits the player's current position.

Human interaction reveals the map. Inactivity or automatic movement crossfades
it over 1.2 seconds to a centered Cartography Unseen title on black. The title
uses Space Grotesk SemiBold, matching the first-person experience. Underneath,
credit the authors Nhat (Hong) Pham, Agnieszka Kiejziewicz, Ricardo Arce and
Kok Yoong Lim, followed by contributor Tom Nguyen and a QR code linking to
https://emergentplay.bynhat.com. The second window starts with this title screen
before the first visitor interacts.

Returning interaction fades the map back in over 0.6 seconds. If input changes
during a transition, reverse from the current opacity without a flash or jump.
Automatic movement contributes no path points or image captures, including
during the fade. Map scene rendering stops after the fade out. Cache the idle
title screen and leave it static without continuous animation. Operator
placement instructions remain available when F1 is open.

## Capture by distance

Image capture is driven by distance travelled during human play. It is not
driven by elapsed time, display frames, or changes in gaze while standing still.
The exact spacing in world units remains to be tuned.

Measure accumulated path length in three dimensions. Using only straight-line
displacement from the last image would miss a visitor walking a loop. Count
actual movement after collision handling, not attempted movement into a wall.
Keep path sampling separate from image sampling so curves remain readable
without filling the map with near-identical images.

Use clean generated image samples without titles, diagnostics or the player
trail. Preserve the full available source resolution in the archive. Low-resolution
thumbnails may be used by the live overview, but must not replace the originals.

Each sample must retain the camera pose and timestamp that produced its image.
AI generation can finish after the player has moved; attaching the image to the
live camera position would misplace it. Distance makes a capture eligible; the
next suitable generated frame supplies the image and its own source pose. Do
not duplicate an old image at several invented positions to fill a capture gap.

The implementation captures one initial view when a human segment begins,
then captures by distance. The 12-unit default remains adjustable for the
exhibition; looking around while stationary does not add captures.

## Returning after automatic movement

Preserve the existing map while nobody is playing. Stop map capture immediately
and stop scene rendering after the fade to the title. The game may continue
moving automatically in the meantime. Keep receiving the current player pose
so returning interaction uses the actual location.

When a visitor moves the mouse or uses movement controls again:

1. Read the actual current player pose from the game.
2. Show the existing map with the player marker and orbit centered on that pose.
3. Begin a new path segment at that location. Do not draw a line across the
   unrecorded automatic journey.
4. Reset the segment's capture-distance accumulator. Automatic travel must not
   cause an immediate burst of distance-triggered images.

For example, if human play stops at P and automatic flight reaches Q, the map
returns with its old path ending at P and the player at Q. New human movement
starts a separate stroke from Q. Never teleport the player marker back to P,
rebase the old map onto Q, or invent a recorded connection between them.

Discard delayed images whose source pose belongs to automatic flight. A delayed
image captured during human play may remain in its original human segment, even
if it finishes after that segment ends. Never assign it to a later segment.

The implementation uses a ten-second inactivity grace period to avoid flicker
between gestures. It can be adjusted in config.json at launch. Automatic flight always
starts the fade to the title and stops capture, regardless of the grace period.

## Space and quit

When the visitor activates the Space command, archive the current map if it
contains a path or an image, then start a fresh empty map for the next journey.
Space therefore defines a map archive boundary. The archive includes every
human segment since the previous reset, including segments separated by idle
flight. It must not reset merely because a different visitor starts playing.

Normal application exit, including Escape and window close, also saves the
current nonempty map. Do not create empty archives. Give each archive a unique
identity so repeated Space presses or exits cannot overwrite an earlier map.
Handle a held Space key as one command rather than a stream of resets. Space
typed inside the prompt editor remains text input.

Saving must precede discarding the old journey. On a save failure, retain the
map and offer an operator-visible error and retry path. Do not silently clear
the only copy. Finish pending archive writes on normal exit. A durable local
checkpoint during play is recommended so a crash or power loss does not erase
an entire exhibition session; a quit handler alone cannot guarantee that.

Associate pending generated frames with their original journey. A frame from
before Space must never appear in the fresh map. If archival work continues in
the background, it must own an immutable snapshot of the old journey.

### What Space currently does

The user describes Space as starting a new world. In the current application,
Space changes the prompt and samples tuning, while preserving procedural world
geometry, camera position and diffusion seed. See the Space handler in
[app/main.py](../app/main.py) and the controls in [README.md](../README.md).

The approved map behavior is clear: save and clear the map on Space. Whether
Space should also regenerate the procedural geometry, change seeds or reset
the player position needs a separate decision. Do not silently bundle those
changes into the map feature.

Automatic prompt changes are not Space commands and should not save or clear
the map. Preserve and highlight their prompt history as described below.

## Prompt changes on both maps

Record prompt changes as timestamped events in the journey archive. Highlight
them on both the live exhibition map and the interactive website map, so someone
can connect a change in the generated world with the words that prompted it.
This is an approved requirement, added after selecting direction A.

Record the initial prompt as well as subsequent changes. Each change event
should retain:

- A stable event ID and prompt revision, linked to the journey and human segment
  when one is active.
- The exact previous and new prompt text. Preserve the composed text submitted
  to generation if it differs from the visible subject prompt.
- The trigger, such as Space, automatic prompt advance, an accepted editor change
  or an external configuration change. Unsubmitted editing is not an event.
- The request time, elapsed journey time, player XYZ and viewing orientation,
  and whether the player was active or the game was moving automatically.
- The first generated frame using that revision, with its source pose and time,
  when available. Keep this separate from the request position and time.
- Relevant generation settings and any configured prompt transition duration.

Associate captured images with their actual prompt revision. Generation is
delayed, and prompt transitions may blend gradually. A marker identifies a
prompt change; it must not imply that the whole image switched instantly at
that location. If another change supersedes a prompt before it produces a frame,
retain the event without inventing an associated image.

For changes during human play, anchor the event to its recorded request position
on the route. Keep the exact text available even when no distance-triggered image
is captured nearby. A prompt event does not trigger an extra image capture or
reset the distance accumulator.

The direction A implementation uses a small persistent route marker, with
the newest subject change's time and prompt readable for fourteen seconds on the live projection.
Keep older full prompt text from filling the map. In the website viewer, selecting
a marker should reveal the exact prompt, time, trigger and associated images.
The website also provides prompt history buttons for keyboard and touch access.
Regional palette updates retain their submitted prompt text in the archive
without repeatedly showing an unchanged subject caption on the live projection.

Continue recording lightweight prompt-change metadata during inactivity, while
the title is visible and route/image recording and map scene rendering remain stopped.
Mark those events as occurring during idle movement. On human return, show the
current prompt with the new segment; do not place an idle event on the previous
human path or draw an automatic-travel connection. The website can expose idle
events in the history without presenting them as visitor exploration.

Space still saves and resets the map. The closing archive keeps its previous
prompt history and the reason it ended. The Space-triggered new prompt becomes
the opening event of the new journey, with a reference to the preceding archive
when one exists. Prompt metadata alone does not make an otherwise empty map
eligible for export.

The portable archive must retain full event text and image associations. Include
prompt event markers and a readable prompt index in the SVG export so the still
remains interpretable without the interactive viewer.

## Save a reusable archive and a detailed export

The goal is to keep each completed exhibition map, inspect its details later,
and reuse the same map in an interactive website viewer.

Use a portable archive of the three-dimensional journey as the source of truth.
A practical initial format is a versioned JSON manifest plus original image
files. The archive should retain:

- Journey identity, start and end times, and completion reason such as Space or quit.
- World coordinate convention, units, world seed and relevant generation settings.
- Separate human path segments with timestamped XYZ points.
- Each image's file reference, dimensions, source timestamp, frame sequence,
  position, orientation and displayed plane dimensions.
- Initial prompt, prompt-change events, exact text, settings and associations
  with generated images, including changes during idle movement.
- Enough overview settings to reproduce the selected A treatment and a saved view.

An archive should be readable without the diffusion model, original machine,
or a running exhibition app. Saving a map does not promise that rerunning the
model will reproduce its images; the saved image files preserve what was seen.

For a high-quality still, export **SVG with embedded original bitmap images**.
Paths, markers and text can remain vectors and stay sharp when zoomed. The
image planes retain their bitmap resolution. Embedding avoids missing linked
images when someone moves or shares the SVG file.

SVG records one projected view of the map. It does not preserve an orbitable 3D
scene by itself. A vector file also cannot recover detail absent from a generated
bitmap. Keep the 3D archive and original images alongside the SVG so the same
map can be viewed from other angles and individual images can be opened at
their native resolution. A PNG may be added as a convenient preview; it is not
the master archive.

Recommended export view: frame the full accumulated journey, including separated
segments, rather than saving only the live camera's crop around the player.
The exact export camera and optional alternate views remain to be chosen.

## Website continuation

Future destination: [emergentplay.bynhat.com](https://emergentplay.bynhat.com).
The website should be able to host completed exhibition maps as interactive
works. Visitors should be able to orbit, pan and zoom through a saved journey,
and inspect a view image at its original available resolution.

Use the same archive as the exhibition app. The website viewer needs geometry,
metadata and image files, not live AI generation. Preserve human-segment gaps,
capture positions, viewing angles and altitude in the web version. Highlight
the same prompt-change events and let visitors inspect their exact text and
associated images.

For large maps, load nearby thumbnails first and original images on demand.
Keep the live exhibition's graphics memory bounded as well; an expanding disk
archive does not require every original image to stay loaded on the GPU.

Website design, deployment and publication are future work. Saving locally
must not automatically publish a map. Choose the website presentation and
which archives to publish in a later task.

## Implementation direction

Reuse the existing GeneratedFrame image and source camera metadata in
[app/types.py](../app/types.py) and [app/diffusion/worker.py](../app/diffusion/worker.py).
Use live game pose updates for the player marker, independently of those
delayed image samples.

The current ten-second first-person trail expires and can include automatic
movement. It cannot serve as the persistent map archive. Record the journey
separately.

A lightweight separate map process with bounded local communication is the
current recommendation. It would own its window and graphics resources without
adding a second AI model or blocking the main input/display loop. Keep disk
encoding and export work away from traversal rendering. Measure the actual
two-projector workload before choosing final update rates and texture budgets.

## Acceptance checks for implementation

- Both windows can be placed and made fullscreen on different displays.
- Direction A shows a spatial path and fixed image planes with correct XYZ,
  yaw and pitch while the overview orbits the live player position.
- Human travel triggers image capture by accumulated 3D distance. Standing
  still, turning in place and automatic travel do not accumulate capture distance.
- Startup shows the centered title, author and contributor credits, and website
  QR code. Inactivity and automatic flight crossfade to it over 1.2 seconds;
  returning interaction fades back over 0.6 seconds without jumping if interrupted.
- Automatic flight stops path recording and image capture immediately. After
  the fade out, map scene rendering stops and the cached title stays static.
  Current pose and lightweight prompt-change metadata continue updating.
- The title matches the first-person font, the QR code opens Emergent Play,
  and F1 placement instructions remain readable over the idle screen.
- Resuming at another location updates the player marker immediately and
  starts a disconnected segment without counting automatic travel.
- Space saves a nonempty archive before clearing the map. Normal quit saves
  the current nonempty archive. Repeated saves do not overwrite prior journeys.
- Save failures preserve recoverable journey data, and delayed frames cannot
  cross a journey reset or enter the wrong human segment.
- The archive retains original image resolution and enough data to rebuild
  the 3D map without running generation.
- Prompt changes retain their exact text, trigger, time and position. Human-play
  changes are highlighted on both maps, independently of distance-based captures.
- Prompt events distinguish request time from the first affected generated frame.
  Idle changes do not create human route markers, and Space assigns the new prompt
  to the new journey without losing the closing archive's history.
- The SVG opens with embedded images and sharp vector paths. Its limitations
  as one viewpoint with finite-resolution images are stated accurately.
- A later website viewer can read the archive and reproduce the map's
  coordinates, orientations, segment gaps and selected visual treatment.

## Remaining tuning and future work

- Tune the 12-unit capture spacing and ten-second grace period on the projectors.
- Whether Space should also change procedural geometry, seeds or player position.
- Review the full-map SVG viewpoint and prompt caption legibility at exhibition scale.
- Website presentation and the selection of archives for publication.
