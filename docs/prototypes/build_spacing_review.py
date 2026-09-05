"""Publishable comparison built from real, matched offline renders."""

from base64 import b64encode
from io import BytesIO
import json
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "logs/reference/spacing-review/variants"
OUT = Path(__file__).with_name("fable-spacing.html")


def embedded(path):
    image = Image.open(path).resize((360, 270), Image.Resampling.LANCZOS)
    stream = BytesIO()
    image.save(stream, format="WEBP", quality=58, method=6)
    return "data:image/webp;base64," + b64encode(stream.getvalue()).decode()


names = {"current": "Current", "a": "A / Slightly roomier", "b": "B / Broad clearings", "c": "C / Clearer passages"}
descriptions = {
    "current": "The world you just tried.",
    "a": "Reduce structure coverage evenly. More room, with much the same rhythm everywhere.",
    "b": "Keep dense areas, then thin panels and interior forms together across broad regions.",
    "c": "Keep the panels. Thin interior clutter most where the existing channel network opens up.",
}
poses = json.loads((DATA.parent / "render-poses.json").read_text())
metrics = json.loads((DATA / "comparison.json").read_text())
settings = json.loads((DATA / "render-settings.json").read_text())
rows = []
for location, pose in enumerate(poses):
    cards = []
    for variant in names:
        ai = embedded(DATA / f"{variant}-{location}-ai.jpg")
        proxy = embedded(DATA / f"{variant}-{location}-proxy.jpg")
        cards.append(f'''<figure><figcaption>{names[variant]}</figcaption><div class="pair">
        <img class="ai" src="{ai}" alt="{names[variant]} generated view at location {location + 1}" width="360" height="270">
        <img class="proxy" src="{proxy}" alt="Matching proxy geometry at location {location + 1}, {names[variant]}" width="360" height="270"></div></figure>''')
    title = ("A dense area stays dense", "Between the dense and open areas", "An opening beyond the foreground")[location]
    explanation = (
        "The broad-clearing field is weak here. B preserves the near column and much of the original enclosure.",
        "This camera looks between nearby structures. Compare the gap and the ceiling, then switch to the proxy to separate geometry from interpretation.",
        "The broad-clearing field is strong here. This is the useful test of whether removing a foreground barrier reveals a space you can read.",
    )[location]
    rows.append(f'''<section class="sample"><h2>{title}</h2><p>{explanation}</p><div class="grid">{"".join(cards)}</div></section>''')

table = []
for variant in names:
    samples = metrics[variant]
    near = np.mean([p["near_share"] for p in samples])
    open_share = np.mean([p["open_share"] for p in samples])
    table.append(f'<tr><td>{names[variant]}</td><td>{near:.1%}</td><td>{open_share:.1%}</td></tr>')

page = '''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Fable: more room between structures</title><style>
:root{color-scheme:dark;--text:#f2f5f6;--muted:#aab5bd;--line:#34414a;--accent:#a3c9c5;--panel:#141a20}*{box-sizing:border-box}body{margin:0;background:#000;color:var(--text);font:16px/1.6 'Segoe UI',sans-serif}main{max-width:1520px;margin:auto;padding:36px 28px 64px}h1,h2,h3{font-family:Bahnschrift,'Arial Narrow',sans-serif;font-weight:500;line-height:1.15}h1{font-size:clamp(28px,4vw,46px);margin:8px 0 16px}h2{font-size:25px;margin:0 0 8px}h3{font-size:20px;margin:0 0 8px}p{margin:8px 0 16px}.intro{max-width:900px;font-size:18px;color:#d0d9de}.eyebrow,code{font-family:Consolas,monospace}.eyebrow{font-size:12px;color:var(--muted);letter-spacing:.08em}.choices,.grid{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:16px}.choices{margin:26px 0}.choice{padding:16px;background:var(--panel)}.choice p{font-size:14px;color:var(--muted);margin:0}.choice:first-child{background:transparent;padding-left:0}.toolbar{display:flex;gap:12px;align-items:center;flex-wrap:wrap;padding:12px 0;border-top:1px solid var(--line);border-bottom:1px solid var(--line)}button{font:inherit;padding:8px 14px;border:1px solid var(--line);border-radius:4px;background:var(--panel);color:var(--text);cursor:pointer}button[aria-pressed=true]{background:var(--accent);color:#0d2424}button:focus-visible{outline:3px solid #dca18a;outline-offset:3px}.toolbar span{color:var(--muted);font-size:14px}.sample{margin-top:34px}.sample>p{font-size:14px;color:var(--muted);max-width:900px}figure{margin:0;min-width:0}figcaption{font-size:14px;padding:6px 0}.pair{position:relative;background:var(--panel);aspect-ratio:4/3}img{display:block;width:100%;height:auto}.proxy{display:none}body.show-proxy .ai{display:none}body.show-proxy .proxy{display:block}.evidence{margin-top:38px;display:grid;grid-template-columns:1fr 1fr;gap:42px;border-top:1px solid var(--line);padding-top:24px}.evidence p{color:#c3ced5}table{border-collapse:collapse;width:100%;font-size:14px}td,th{text-align:left;padding:10px 8px;border-bottom:1px solid var(--line)}th{font-weight:400;color:var(--muted)}.note,footer{font-size:13px;color:var(--muted)}details{border-top:1px solid var(--line);margin-top:26px;padding-top:16px}summary{cursor:pointer}code{font-size:12px;overflow-wrap:anywhere}footer{margin-top:26px}noscript p{padding:12px;background:var(--panel)}
@media(max-width:1000px){.grid,.choices{grid-template-columns:repeat(2,minmax(0,1fr))}.evidence{gap:24px}}@media(max-width:600px){main{padding:24px 16px 44px}.choices{gap:10px}.choice{padding:12px}.choice:first-child{padding-left:12px}.grid,.evidence{grid-template-columns:1fr}h3{font-size:18px}.sample{margin-top:28px}.grid{gap:18px}}
</style></head><body><main>
<div class="eyebrow">CARTOGRAPHY UNSEEN / SPACING STUDY / 5 SEPTEMBER 2026</div>
<h1>Let the space open up.</h1>
<p class="intro">Keep the dense, layered world. Give it places where nearby forms recede and you can see farther into the scene. These are spacing choices; all keep your selected image-only presentation and the current AI tuning.</p>
<div class="choices">__CHOICES__</div>
<div class="toolbar" aria-label="Comparison view"><button type="button" id="ai-view" aria-pressed="true">Generated images</button><button type="button" id="proxy-view" aria-pressed="false">Compare proxy geometry</button><span>Same camera, prompt, noise seed and sampler within every row.</span></div>
<noscript><p>The generated comparisons remain visible. Enable JavaScript to switch the entire comparison to proxy geometry.</p></noscript>
__ROWS__
<div class="evidence"><section><h2>What the geometry says</h2><table><thead><tr><th>Spacing</th><th>Close obstacles</th><th>Longer clear views</th></tr></thead><tbody>__TABLE__</tbody></table><p class="note">28 matched free positions in world seed __WORLD__. 36 headings per position. Close means blocked within 8 world units; longer means at least 30 clear units, measured against the collision body. These are geometry samples, not player success rates or perceived AI depth.</p></section>
<section><h2>Recommendation</h2><p>__RECOMMENDATION__</p><p>The current geometry already has connected exits and some regional variation. Across three seeds, 33–42% of sampled headings meet a form within 16 units. In your current seed, interior masses account for about two-thirds of those nearby blockages. Opening only the panel shell would leave much of that clutter.</p><p>These are isolated prototypes. The running app, world source, prompts and config have not been changed.</p></section></div>
<details><summary>Method and limits</summary><p>The baseline survey sampled 432 free points across three seeds, with 36 headings per point. The smaller comparison reuses 28 baseline-valid positions in the current seed. Each candidate only removes existing geometry, so it cannot add a barrier at a previously clear position.</p><p>The three pictured positions represent weak, intermediate and strong regions of B's spacing field. They are selected examples rather than a random sample of final image quality. All four treatments use exactly the same camera within a row.</p><p>A removes about 22% of stable spatial object groups everywhere. B varies removal from 5% to 60% with a smooth field spanning roughly 160 horizontal units; vertical variation is slower. C preserves every panel and removes up to 65% of interior groups in the existing open-channel regions. These percentages are prototype generation rules, not percentages of visible screen area.</p><p>Each image follows 12 stationary diffusion updates from a reset noise seed. Current prompt and tuning were captured from the local config. CFG __CFG__, timestep __TIMESTEP__, guide __GUIDE__, one step, 512×384. No model, prompt or sampler adjustment was made between treatments. A long moving comparison is still needed before treating any option as a proven navigation improvement.</p><p>Reproduce the geometry comparison with <code>runtime/python/python.exe docs/prototypes/spacing_study.py --poses logs/reference/spacing-review/matched-poses.json</code>. Use <code>--poses logs/reference/spacing-review/render-poses.json --render</code> for the image comparison. The isolated script patches chunk generation only inside its own process.</p></details>
<footer>Choose a spacing direction before integration, following the prototype-first workflow in AGENTS.md. Prepared by GPT-6 in Codex.</footer>
</main><script>
const ai=document.getElementById('ai-view'),proxy=document.getElementById('proxy-view');
function showProxy(value){document.body.classList.toggle('show-proxy',value);ai.setAttribute('aria-pressed',String(!value));proxy.setAttribute('aria-pressed',String(value));}
ai.addEventListener('click',()=>showProxy(false));proxy.addEventListener('click',()=>showProxy(true));
</script></body></html>'''

choices = "".join(f'<section class="choice"><h3>{names[v]}</h3><p>{descriptions[v]}</p></section>' for v in names)
recommendation = "C is my pick for finding your way. It clears deeper views while retaining the surrounding panels, so the space still feels substantial. B creates stronger contrast between dense areas and broad clearings. A loosens the whole scene without much change in pacing."
for key, value in {
    "__CHOICES__": choices, "__ROWS__": "".join(rows), "__TABLE__": "".join(table),
    "__WORLD__": str(settings["world_seed"]), "__RECOMMENDATION__": recommendation,
    "__CFG__": str(settings["guidance_scale"]),
    "__TIMESTEP__": f'{settings["timestep_min"]}–{settings["timestep_max"]}',
    "__GUIDE__": str(settings["guide_strength"]),
}.items():
    page = page.replace(key, value)
OUT.write_text(page, encoding="utf-8")
assert OUT.stat().st_size <= 512 * 1024, OUT.stat().st_size
print(f"Wrote {OUT.name}, {OUT.stat().st_size} bytes")
