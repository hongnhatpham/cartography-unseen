"""Build the standalone Fable interaction study from the captured baseline."""

from base64 import b64encode
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).with_name("fable-free-flight.html")


def scene(choice: str) -> str:
    """Keep the composition identical so only the navigation treatment changes."""
    return f'''<svg viewBox="0 0 640 440" role="img" aria-label="Floating abstract slabs with open space above and below, option {choice}">
    <defs>
      <linearGradient id="space{choice}" x2="0" y2="1"><stop stop-color="#283e53"/><stop offset=".52" stop-color="#a9c5d5"/><stop offset="1" stop-color="#38405a"/></linearGradient>
      <pattern id="grain{choice}" width="31" height="23" patternUnits="userSpaceOnUse" patternTransform="rotate(18)"><path d="M0 4h22M9 10h19M2 17h13" stroke="#a9c5d5" stroke-opacity=".18" stroke-width="3"/></pattern>
    </defs>
    <rect width="640" height="440" fill="url(#space{choice})"/>
    <g class="scene">
      <g fill="#6e8492"><path d="M264 105l32-7 0 39-32 4zM358 284l29-9 0 50-29 14zM396 156l13-3 0 38-13 4zM246 264l15-2 0 16-15 4z"/></g>
      <g class="solid">
        <path fill="#14232d" d="M-50 32L177 91 227 167 15 149z"/>
        <path fill="#7399a5" d="M-50 32L30-14 205 36 177 91z"/>
        <path fill="#43606d" d="M177 91L205 36 245 121 227 167z"/>
        <path fill="#172732" d="M439 32L560-20 562 195 435 170z"/>
        <path fill="#d0a476" d="M562-20L651 2 642 162 562 195z"/>
        <path fill="#516c7a" d="M435 170L562 195 642 162 512 145z"/>
        <path fill="#233744" d="M60 287L209 257 210 473 27 479z"/>
        <path fill="#af927c" d="M60 287L123 226 264 226 209 257z"/>
        <path fill="#648894" d="M209 257L264 226 277 461 210 473z"/>
        <path fill="#253343" d="M404 342L535 271 694 328 548 413z"/>
        <path fill="#b6a19c" d="M404 342L548 413 550 455 407 383z"/>
        <path fill="#3e5667" d="M326 180L349 173 350 213 326 222z"/>
        <path fill="#c9a17f" d="M326 180L337 169 360 165 349 173z"/>
      </g>
      <path fill="url(#grain{choice})" d="M-50 32L177 91 227 167 15 149zM439 32L560-20 562 195 435 170zM60 287L209 257 210 473 27 479zM404 342L535 271 694 328 548 413z"/>
      <g class="contours" fill="none" stroke="#d6edf2" stroke-width="1.6" stroke-linejoin="round"><path d="M0 49L177 91 227 167 15 149M177 91L205 36M439 32L435 170 562 195 642 162M562 0V195M27 439L60 287 123 226 264 226 277 439M60 287L209 257 264 226M209 257L210 439M404 342L535 271 640 309M404 342L548 413 640 360M404 342L407 383 550 455"/></g>
      <g class="reveal" fill="#a9c5d5" fill-opacity=".24" stroke="#d6edf2" stroke-width="1.2"><path d="M60 287L209 257 210 473 27 479zM209 257L264 226 277 461 210 473z"/><path fill="none" stroke-opacity=".5" d="M53 321L209 294 266 265M45 361L209 335 270 309M37 401L209 377 273 352M110 277L91 446M161 267L156 449"/></g>
    </g></svg>'''


cards = []
for choice, title, desc, trade in [
    ("A", "Let the image carry it", "Free flight through an uninterrupted generated world. Preserve openings and large silhouettes through stronger spatial conditioning.", "Cleanest image. Generation can still invent a passage unless structural adherence improves."),
    ("B", "Draw the boundaries", "Fine contours follow nearby solid surfaces. They stay fixed to the world while the material inside them changes.", "Easiest boundaries to read. The persistent linework becomes part of the art direction."),
    ("C", "Reveal on approach", "The image fills the view at a distance. A faint surface and its boundary appear as you approach a collision.", "My pick. It gives contact a visible cause, with less linework during open flight."),
]:
    cards.append(f'''<article class="option {choice}"><header><span class="letter">{choice}</span><h2>{title}</h2></header>
    <div class="viewport">{scene(choice)}<span class="viewlabel">Level flight</span></div>
    <p>{desc}</p><p class="trade">{trade}</p></article>''')

baseline = b64encode((ROOT / "logs/reference/fable-review-baseline/proxy_vs_out.jpg").read_bytes()).decode()
page = '''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>Fable: free flight and readable space</title>
<style>
:root{color-scheme:dark;--paper:#000;--surface:#141920;--text:#f1f4f7;--muted:#a3adb7;--accent:#a9c5d5;--line:#303840}
*{box-sizing:border-box}body{margin:0;background:var(--paper);color:var(--text);font:16px/1.6 'Segoe UI',sans-serif}main{max-width:1480px;margin:auto;padding:40px 32px 70px}h1,h2,h3{font-family:Bahnschrift,'Arial Narrow',sans-serif;font-weight:500;line-height:1.12}h1{font-size:clamp(30px,4vw,50px);margin:8px 0 20px}h2{font-size:23px;margin:0}h3{font-size:22px}p{margin:12px 0}.kicker,code,.letter,.viewlabel{font-family:Consolas,monospace}.kicker{color:var(--muted);font-size:13px;letter-spacing:.08em}.intro{max-width:820px;color:#c8d0d8;font-size:18px}.notice{padding:12px 16px;border-left:2px solid var(--accent);background:var(--surface);margin:22px 0}.controls{display:flex;align-items:center;gap:12px;flex-wrap:wrap;margin:24px 0 16px}button{font:inherit;color:var(--text);background:var(--surface);border:1px solid var(--line);border-radius:4px;padding:9px 16px;cursor:pointer}button[aria-pressed=true]{background:var(--accent);color:#101b24}button:focus-visible,input:focus-visible{outline:3px solid #e2ae65;outline-offset:4px}.distance{margin-left:auto;display:flex;align-items:center;gap:10px;flex-wrap:wrap}input{accent-color:var(--accent);width:130px}.options{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:24px}.option header{display:flex;gap:12px;align-items:center;min-height:62px}.letter{font-size:25px;color:var(--accent)}.viewport{position:relative;overflow:hidden;background:var(--surface);border:1px solid var(--line)}svg{display:block;width:100%;height:auto}.viewlabel{position:absolute;bottom:10px;left:12px;background:#14232de8;padding:3px 8px;font-size:12px;color:#d6edf2}.contours,.reveal{opacity:0}.B .contours{opacity:.82}.C .reveal{opacity:.75}.trade{color:var(--muted);font-size:14px}.columns{display:grid;grid-template-columns:1fr 1fr;gap:48px;margin-top:36px;border-top:1px solid var(--line);padding-top:8px}ul{padding-left:22px}li{margin:8px 0}figure{margin:20px 0}figure img{display:block;width:100%;height:auto}figcaption{font-size:13px;color:var(--muted);margin-top:8px}table{border-collapse:collapse;width:100%;font-size:14px}th,td{padding:9px 10px;border-bottom:1px solid var(--line);text-align:left}th{color:var(--muted);font-weight:400}details{margin-top:28px;border-top:1px solid var(--line);padding-top:18px}summary{cursor:pointer}code{font-size:12px;overflow-wrap:anywhere}footer{margin-top:36px;color:var(--muted);font-size:13px}.hint{font-size:13px;color:var(--muted)}
@media(max-width:960px){.options{grid-template-columns:1fr}.option{display:grid;grid-template-columns:1fr 1fr;gap:0 24px}.option header{grid-column:1/-1}.viewport{grid-row:2/4}.option p{margin-top:0}.columns{gap:24px}.distance{margin-left:0}}
@media(max-width:620px){main{padding:24px 18px 40px}.option,.columns{display:block}.option p{margin-top:12px}.options{gap:28px}.controls{gap:8px}.distance{width:100%}h2{font-size:22px}}
</style></head><body><main>
<div class="kicker">CARTOGRAPHY UNSEEN / FABLE / 5 SEPTEMBER 2026</div>
<h1>Fly anywhere the space is open.</h1>
<p class="intro">No gravity, no terrace climbing. Look up and move forward to ascend. Look down to descend. Keep the world abstract, and make every solid boundary readable.</p>
<div class="notice">Three illustrative interaction mocks, shown in the same space. These drawings are proposed treatments, not new diffusion results. The actual generated baseline is below.</div>
<div class="controls"><span>Preview the view</span><button type="button" data-view="level" aria-pressed="true">Level</button><button type="button" data-view="up" aria-pressed="false">Look up</button><button type="button" data-view="down" aria-pressed="false">Look down</button><label class="distance">Distance to a surface <input id="distance" type="range" min="0" max="100" value="75"><output id="distance-label">Near</output></label></div>
<p class="hint">The view buttons shift the composition to illustrate pitch. The distance slider previews C's surface reveal. This is not a collision or flight simulation.</p>
<section class="options" aria-label="Design choices">__CARDS__</section>
<div class="columns"><section><h3>Movement shared by all three</h3><ul><li>W/S follows the full look direction. A/D strafes.</li><li>E rises and Q descends, independent of gaze. Shift moves faster.</li><li>Release a key to stop. No gravity or automatic descent.</li><li>Solid forms still block movement. Slide along a surface, and fly over or under it.</li><li>Space keeps its current new-world action. Idle flight must climb, descend and steer around forms too.</li></ul></section>
<section><h3>What I would build after your pick</h3><p>Complete the existing free-flight migration and compare proxy and generated boundaries along the same recorded route. Preserve openings through generation in every option.</p><p>I recommend C. It keeps open flight visually quiet and makes a nearby collision understandable. It does not solve a false opening in the distance, so closer image adherence remains required.</p><p>Start with the current model and measure the result. If conditioning adjustments cannot preserve openings, benchmark one structural adapter against the current frame time.</p></section></div>
<section><h3>The current image still changes the layout</h3><figure><img src="data:image/jpeg;base64,__BASELINE__" alt="Four fresh matched frames, each with proxy on the left and AI output on the right. The output adds architectural detail and changes the silhouettes."><figcaption>Fresh local capture. Each pair is proxy left, generated image right. World seed 12345, four poses, current one-step sampler. The image introduces surfaces and structures absent from the proxy.</figcaption></figure>
<table><thead><tr><th>Check</th><th>Observed</th><th>What it tells us</th></tr></thead><tbody><tr><td>Proxy/output edge SSIM</td><td>0.100</td><td>Weak edge agreement in this small sample. This is not a navigation success rate.</td></tr><tr><td>Wall/open image separation</td><td>Pass, 2 poses each</td><td>Brightness and detail differ. Correct opening boundaries are not established.</td></tr><tr><td>Camera and world tests</td><td>52 passed</td><td>The lower-level tests pass on the inherited work.</td></tr><tr><td>Application startup</td><td>Fails</td><td>The input loop calls Camera.walk, but the camera now exposes fly.</td></tr></tbody></table></section>
<details><summary>Evidence and implementation review notes</summary><p>Branch: <code>fable-free-flight-proxy</code>. The 16 pre-existing modified files were preserved. This review adds only prototype files.</p><p>Observed in <code>app/renderer/camera.py</code> and <code>app/main.py</code>: the free-flight camera exists, but manual and idle input still call the removed walking method. Application smoke launch reproduced <code>'Camera' object has no attribute 'walk'</code>. This is an integration failure in the unfinished work, not a diagnosis of earlier image drift.</p><p>Baseline command: <code>runtime/python/python.exe tools/style_sheet.py --pairs 4 --wall-poses 2 --seed 12345 --out logs/reference/fable-review-baseline</code>.</p><p>Test command: <code>runtime/python/python.exe -m pytest tests/test_core.py tests/test_world.py -q</code>.</p><p>The image experiment is a baseline, not a completed fix. Next verification must include upward and downward movement through actual input, collision from above and below, idle flight, and matched generated views across multiple styles and seeds. Edge SSIM supports visual review but cannot prove traversability. Compare latency with the same route before accepting any added conditioning or reveal pass.</p></details>
<footer>Choose A, B or C before implementation. Prepared by GPT-6 in Codex. No production components were edited in this review.</footer>
</main><script>
const offsets={level:0,up:72,down:-72};
document.querySelectorAll('[data-view]').forEach(button=>button.addEventListener('click',()=>{
document.querySelectorAll('[data-view]').forEach(item=>item.setAttribute('aria-pressed',String(item===button)));
document.querySelectorAll('.scene').forEach(scene=>scene.setAttribute('transform','translate(0 '+offsets[button.dataset.view]+')'));
document.querySelectorAll('.viewlabel').forEach(label=>label.textContent=button.dataset.view==='level'?'Level flight':button.dataset.view==='up'?'Looking upward':'Looking downward');
}));
const slider=document.getElementById('distance');
slider.addEventListener('input',()=>{const value=Number(slider.value)/100;document.querySelector('.C .reveal').style.opacity=String(value);document.getElementById('distance-label').textContent=value<.25?'Far':value<.7?'Approaching':'Near';});
</script></body></html>'''

OUT.write_text(page.replace("__CARDS__", "".join(cards)).replace("__BASELINE__", baseline), encoding="utf-8")
assert OUT.stat().st_size <= 512 * 1024
print(f"Wrote {OUT.name}: {OUT.stat().st_size} bytes")
