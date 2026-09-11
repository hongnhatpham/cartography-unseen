---
name: Cartography Unseen exhibition monitor
colors:
  background: '#101a25'
  surface: '#14202d'
  surface-container: '#20303e'
  surface-container-lowest: '#111b25'
  on-surface: '#edf3f6'
  on-surface-variant: '#a6b8c9'
  outline: '#354553'
  primary: '#bcb1ff'
  success: '#9cddd0'
  warning: '#f1be75'
  error: '#f7aeb3'
typography:
  headline-md:
    fontFamily: Segoe UI
    fontSize: 29px
    fontWeight: '500'
    lineHeight: '1.25'
    letterSpacing: -0.65px
  body-base:
    fontFamily: Segoe UI
    fontSize: 13px
    fontWeight: '400'
    lineHeight: '1.6'
  label-caps:
    fontFamily: Segoe UI
    fontSize: 10px
    fontWeight: '600'
    letterSpacing: 1.2px
  stat-lg:
    fontFamily: Consolas
    fontSize: 35px
    fontWeight: '400'
    lineHeight: '1.1'
    letterSpacing: -1.2px
rounded:
  sm: 4px
  DEFAULT: 5px
  lg: 8px
spacing:
  unit: 4px
  sm: 8px
  md: 16px
  lg: 24px
  gutter: 28px
  margin-mobile: 18px
  margin-desktop: 28px
---

# Cartography Unseen design system

## Visual Theme & Atmosphere

The approved direction is A, Projection desk, with C's direct incident wording from
the exhibition monitor study, revision 5. A slate desk holds the installation's
two projection readouts. The independent host, artwork and map-backup strip is the
first thing an operator reads. The layout suits Nhat's remote checks during a
week-long exhibition. No imagery, decorative diagrams or animated status lights
compete with those answers.

## Color Palette & Roles

Deep slate `#14202d` is the main surface, inside the darker `#101a25` page.
Upload evidence sits on the lighter `#20303e` panel. Paired projection readouts use
`#111b25`. Lilac `#bcb1ff` identifies the projections, generation trace and keyboard
focus. Text uses `#edf3f6`, with secondary information in `#a6b8c9`.

Sea green `#9cddd0`, amber `#f1be75` and rose `#f7aeb3` indicate observed activity,
attention and failure. Every status also has explicit text. Thin `#354553` borders
divide related readings without adding shadows.

## Typography Rules

Segoe UI carries the brand, incident sentence and interface labels, with Arial and
sans-serif fallbacks. The restrained 29px incident heading reduces to 24px on
phones. Consolas carries measured values, with SFMono-Regular and Courier New
fallbacks. No web fonts are fetched. Eyebrows use small uppercase labels only for
signal and projection identities. Body copy stays in sentence case.

## Component Stylings

The health strip always separates machine contact, artwork evidence and map
verification. The machine picker and Refresh now button are native, labelled,
keyboard-accessible controls. Thin outlines and 4px corners keep them quiet.
The 3px lilac focus ring stays visible. There are no remote command controls.

Paired 5px projection panels distinguish generation FPS from display FPS and keep
the map projection visible even when recording telemetry is unavailable. A 24-hour
SVG trace uses five-minute averages and leaves gaps for absent data. Resource
meters complement numeric readings; unavailable values never become zero.

The upload panel separates transfer from verification. It never turns session byte
counters into a transfer percentage. Stale uploader phases receive a last-known
label and no current-stage highlight. The last verified receipt confirms only a
past backup. Native disclosures expose timestamps and raw status evidence.

Loading, no-machine, no-heartbeat, expired-session, request-error and recovery
states explain what is known and what to do next. Last-known readings remain
visible after contact or monitoring failures. Alert resolution means the server
condition cleared, not that the exhibition recovered.

## Layout Principles

The desktop desk is at most 1360px wide. A 28px gutter separates a 1.45-fraction
readout column from a one-fraction upload column. Below 780px, the columns stack.
Below 480px, health answers become three compact rows so all remain legible.
Projection readouts stay paired on a 390px phone. Details expand in normal document
flow. Open disclosures and their keyboard focus survive automatic refreshes.
There is no continuous animation, and reduced-motion preferences are respected.

## Voice & Tone

Lead with the exhibition consequence, then the supported next action. For example,
"The machine is unreachable. The artwork may still be running." Follow with a
request to check the venue connection and power. Say "Recording not measured"
when telemetry cannot prove activity. Use "Latest report" and "Last-known report"
to distinguish receipt freshness, with observation ages in evidence. Never claim
that process activity confirms a physical projector or that transfer means backup.
