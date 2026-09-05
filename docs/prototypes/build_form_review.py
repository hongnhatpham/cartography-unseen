"""Build the standalone form and color comparison from isolated study renders."""

from __future__ import annotations

import base64
import html
import io
import json
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[2]
STUDY = ROOT / "logs/reference/form-study"
OUT = ROOT / "docs/prototypes/fable-form-color.html"
REFERENCE = Path(
    "C:/Users/v12485/01 Projects/RMIT/Emergent Play/Exhibitions/VFCD 2026/"
    "viewer_20260625_204805"
)
REFERENCES = (
    ("viewer_01846_frame_001870_mode_sd15_ts_1782398327.jpg", "Fine wire and open frames"),
    ("viewer_02006_frame_002030_mode_sd15_ts_1782398592.jpg", "Curved layers and color"),
    ("viewer_02519_frame_002543_mode_sd15_ts_1782399444.jpg", "Color across the space"),
)
VARIANTS = (
    ("current", "Current", "Block structure", "Existing cubes, checker texture and mostly neutral color."),
    ("a", "A", "Rounded additions", "Keep the cubes and add softer solid forms, with quieter color."),
    ("b", "B", "Wire and ribs", "Keep the cubes and add curved wire sheets, ribs and cages."),
    ("c", "C", "Mixed forms and color", "Keep the cubes and add mixed curves, broader color regions and colored atmosphere."),
)


def image_url(path: Path, quality: int, max_width: int = 480) -> str:
    with Image.open(path) as source:
        image = source.convert("RGB")
        if image.width > max_width:
            image = image.resize(
                (max_width, round(image.height * max_width / image.width)),
                Image.Resampling.LANCZOS,
            )
        buffer = io.BytesIO()
        image.save(buffer, format="WEBP", quality=quality, method=6)
    return "data:image/webp;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def build(quality: int) -> str:
    settings = json.loads((STUDY / "settings.json").read_text(encoding="utf-8"))
    references = "".join(
        '<figure><img src="' + image_url(REFERENCE / name, quality, 420)
        + '" alt="' + html.escape(caption) + ' in the newer reference batch" '
        + 'width="420" height="336"><figcaption>' + caption + ' · ' + name.split('_frame')[0] + '</figcaption></figure>'
        for name, caption in REFERENCES
    )
    columns = []
    for key, letter, title, description in VARIANTS:
        views = []
        for pose in range(2):
            images = "".join(
                '<img class="' + kind + '" src="'
                + image_url(STUDY / f"{key}-{pose}-{kind}.png", quality)
                + '" alt="' + html.escape(f"{letter}, view {pose + 1}, {kind.upper()}")
                + '" width="384" height="256">'
                for kind in ("ai", "proxy")
            )
            views.append(f'<figure>{images}<figcaption>View {pose + 1}</figcaption></figure>')
        columns.append(
            f'<article class="variant {key}"><header><span class="letter">{letter}</span>'
            f'<h3>{title}</h3><p>{description}</p></header>' + "".join(views) + '</article>'
        )
    tuning = (
        f'{html.escape(settings["diffusion_resolution"])} · CFG {settings["guidance_scale"]:g}'
        f' · Guidance {settings["guide_strength"]:g}'
        f' · Timestep {settings["timestep_min"]} to {settings["timestep_max"]}'
    )
    wording_trial = ""
    wording_file = STUDY / "wire-wording.json"
    wording_image = STUDY / "b-0-wire-ai.png"
    if wording_file.exists() and wording_image.exists():
        wording = json.loads(wording_file.read_text(encoding="utf-8"))
        wording_trial = '''<section class="wording" aria-labelledby="wording-title">
<h2 id="wording-title">Extra wire wording in the prompt</h2>
<p>The current prompt asks for biophilia, so the AI can turn B's wire forms into vegetation. This trial keeps that full prompt and appends a description of wire and ribs. B's geometry, camera, noise and sampler settings stay fixed.</p>
<div class="wording-pair"><figure><img src="''' + image_url(STUDY / "b-0-ai.png", quality) + '''" alt="B, first viewpoint, current biophilia prompt" width="384" height="256"><figcaption>B / Current prompt</figcaption></figure>
<figure><img src="''' + image_url(wording_image, quality) + '''" alt="B, first viewpoint, added wire wording" width="384" height="256"><figcaption>B / Extra wire wording</figcaption></figure></div>
<details><summary>Full prompt with the added wording</summary><p>''' + html.escape(wording["prompt"]) + '''</p></details></section>'''
    return '''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Fable · Form and color</title>
<style>
.wording{margin:24px 0 27px;border-top:1px solid #343434;padding-top:20px}.wording>p{font-size:14px;color:#ccc;margin-top:8px;max-width:92ch}.wording-pair{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:16px;max-width:950px;margin-top:15px}
:root{color-scheme:dark;--bg:#000;--surface:#141414;--line:#343434;--text:#f5f5f5;--muted:#b1b1b1;--accent:#d9e2e7}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font:15px/1.5 "Segoe UI",sans-serif}
main{max-width:1740px;margin:auto;padding:26px 32px 38px}h1,h2,h3,p,figure{margin:0}h1,h2,h3,.letter{font-family:Bahnschrift,"Segoe UI",sans-serif;font-weight:500}h1{font-size:29px;letter-spacing:-.6px}h2{font-size:20px}h3{font-size:18px;line-height:1.25}p{max-width:80ch}p+p{margin-top:9px}.masthead{display:flex;align-items:baseline;gap:22px;flex-wrap:wrap;margin-bottom:18px}.masthead p{font-size:13px;color:var(--muted)}.intro{max-width:95ch;color:#ddd;margin-bottom:22px}.refs{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:12px;margin:12px 0 26px;max-width:1180px}img{display:block;width:100%;height:auto;background:var(--surface)}.refs img{aspect-ratio:5/4;object-fit:contain}figcaption{font-size:12px;color:var(--muted);margin-top:5px}.compare-heading{display:flex;gap:16px;align-items:baseline;flex-wrap:wrap;margin-bottom:12px}.compare-heading p{color:var(--muted);font-size:13px}.view-switch{position:absolute;opacity:0;width:1px;height:1px}.switch-label{display:inline-block;padding:7px 15px;border:1px solid var(--line);margin:0 5px 15px 0;cursor:pointer;font-size:13px}.view-switch:focus-visible+.switch-label{outline:2px solid white;outline-offset:4px}.view-switch:checked+.switch-label{color:#000;background:var(--accent);border-color:var(--accent)}.comparison{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:16px}.variant header{min-height:124px;border-top:1px solid var(--line);padding-top:12px}.letter{font-size:13px;display:block;margin-bottom:6px;color:var(--muted)}.variant header p{font-size:13px;color:var(--muted);margin:7px 0 14px;max-width:37ch}.variant figure{margin-bottom:17px}.proxy{display:none}#show-proxy:checked~.comparison .ai{display:none}#show-proxy:checked~.comparison .proxy{display:block}.note{border-left:2px solid #666;padding-left:13px;margin:15px 0 25px;font-size:15px}.explanation{display:grid;grid-template-columns:1fr 1fr;gap:32px;border-top:1px solid var(--line);padding-top:21px;margin-top:7px}.explanation h2{font-size:18px;margin-bottom:9px}.explanation p{color:#ccc;font-size:14px}.method{margin-top:25px;background:var(--surface);padding:16px 19px}.method h2{font-size:16px;margin-bottom:7px}.method p{max-width:110ch;font-size:13px;color:var(--muted)}details{margin-top:12px;font-size:13px;color:var(--muted)}summary{cursor:pointer;width:fit-content;color:#ddd}summary:focus-visible{outline:2px solid white;outline-offset:3px}details p{margin-top:7px}.settings{font-variant-numeric:tabular-nums;color:#ddd!important}.footer{color:var(--muted);font-size:12px;margin-top:19px}@media(min-width:1450px){.variant header{min-height:112px}}@media(max-width:1000px){main{padding:22px}.comparison{grid-template-columns:repeat(2,minmax(0,1fr));gap:18px}.variant header{min-height:110px}}@media(max-width:600px){main{padding:18px 14px}.masthead{gap:5px}.masthead h1{font-size:25px}.refs{gap:7px}.refs figcaption{font-size:10px}.comparison{grid-template-columns:1fr;gap:12px}.variant header{min-height:0}.variant header p{max-width:none}.variant figure{margin-bottom:13px}.explanation{grid-template-columns:1fr;gap:23px}.method{padding:14px}.intro{font-size:14px}}
</style></head><body><main>
<header class="masthead"><h1>Fable / Form and color</h1><p>Isolated visual study · 5 September 2026</p></header>
<p class="intro">Keep the cubes and slabs, then add curves, wire and a wider range of forms alongside them. These three directions preserve the existing geometry and use the current AI settings. Color and checker treatments can be chosen separately.</p>
<section aria-labelledby="reference-title"><h2 id="reference-title">From the newer references</h2><div class="refs">''' + references + '''</div></section>
<section aria-labelledby="compare-title"><div class="compare-heading"><h2 id="compare-title">Current + three ways to add variety</h2><p>Two matched viewpoints in every column.</p></div>
<input class="view-switch" type="radio" name="view" id="show-ai" checked><label class="switch-label" for="show-ai">Generated image</label>
<input class="view-switch" type="radio" name="view" id="show-proxy"><label class="switch-label" for="show-proxy">Proxy geometry</label>
<div class="comparison">''' + "".join(columns) + '''</div></section>
<p class="note">My pick is B's wire and rib additions with C's broader color. Every direction keeps the existing cubes and slabs. Some fine wires still become abstract shapes in the AI image; use the proxy switch to see the geometry itself.</p>''' + wording_trial + '''
<section class="explanation" aria-label="Why the current image looks this way"><div><h2>Why the cubes have texture</h2><p>The checker was added as a perspective cue on broad, nearby walls, to help reduce furniture-like interpretations. It travels into the AI as part of the color image, so the AI can also treat it as a material pattern. It is not required by the model.</p><p>These studies reduce almost all of that checker while adding curves, ribs and open frames. The checker treatment can be chosen independently of which shapes we add.</p></div><div><h2>Why so much reads as gray</h2><p>The current palette keeps most surfaces nearly neutral. Saturation sits around 0.04 to 0.13, with occasional stronger accents at 0.62 to 0.88. A 0.94 selection threshold makes those accents sparse.</p><p>The alternatives spread color across individual forms and larger regions. C also colors the atmosphere, so distant space can carry color as well as nearby objects. We can use its color treatment with another set of shapes.</p></div></section>
<section class="method" aria-labelledby="method-title"><h2 id="method-title">What this comparison establishes</h2><p>The main comparison uses the same two cameras, world and noise seeds, prompt, and sampler settings in every direction. Each image follows 20 stationary generated updates. These are actual proxy and AI renders from an isolated study. The images are compressed for this report.</p><p class="settings">''' + tuning + '''</p><p>Every direction preserves the existing cube and slab geometry and adds forms alongside it. The additions use approximate, sampled meshes for this still-image study. Passage clearance, collision behavior and live performance still need validation before implementation. The running experience has not changed.</p>
<details><summary>Prompt used for the main comparison</summary><p>''' + html.escape(settings["prompt"]) + '''</p></details></section>
<p class="footer">Choose A, B or C, or combine the form of one with the color of another.</p>
</main></body></html>'''


def main() -> None:
    for quality in (82, 76, 70, 64, 58):
        document = build(quality)
        size = len(document.encode("utf-8"))
        if size <= 512 * 1024:
            OUT.write_text(document, encoding="utf-8")
            print(f"Wrote {OUT.name}: {size:,} bytes; WebP quality {quality}")
            return
    raise RuntimeError("Report exceeds 512 KiB even at minimum image quality")


if __name__ == "__main__":
    main()
