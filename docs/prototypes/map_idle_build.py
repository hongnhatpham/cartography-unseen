"""Build the approved A preview. Requires Pillow and qrcode; no game changes."""
from pathlib import Path
from io import BytesIO
from base64 import b64encode
from PIL import Image
import qrcode

ROOT = Path(__file__).resolve().parents[2]
WEBSITE = 'https://emergentplay.bynhat.com/'

def data_url(content, mime):
    return f'data:{mime};base64,' + b64encode(content).decode()

still = Image.open(ROOT / 'logs/map-demo-20260906-take3/poster.jpg').crop((960,48,1920,688))
buffer = BytesIO()
still.save(buffer, format='JPEG', quality=85, optimize=True)
map_image = data_url(buffer.getvalue(), 'image/jpeg')
fonts = '\n'.join(
    f"@font-face{{font-family:'{family}';font-style:normal;font-weight:{weight};src:url({data_url((ROOT / 'assets/fonts' / name).read_bytes(), 'font/woff')}) format('woff');font-display:block}}"
    for family, weight, name in [('Space Grotesk',600,'SpaceGrotesk-SemiBold.woff'),('IBM Plex Mono',400,'IBMPlexMono-Regular.woff')]
)
# Keep a four-module quiet zone and conventional black ink for projection.
qr = qrcode.QRCode(error_correction=qrcode.constants.ERROR_CORRECT_M, border=4, box_size=8)
qr.add_data(WEBSITE)
qr.make(fit=True)
matrix = qr.get_matrix()
size = len(matrix)
path = ' '.join(f'M{x} {y}h1v1h-1z' for y,row in enumerate(matrix) for x,dark in enumerate(row) if dark)
qr_svg = f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {size} {size}" role="img" aria-label="QR code to emergentplay.bynhat.com" shape-rendering="crispEdges"><rect width="{size}" height="{size}" fill="white"/><path d="{path}" fill="black"/></svg>'
qr.make_image().save(ROOT / 'logs/map-idle-qr.png')
html = '''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><meta name="color-scheme" content="dark"><title>Cartography Unseen · Title and credits preview</title><style>
__FONTS__
:root{color-scheme:dark;--black:#000;--text:#eae7dd;--muted:#aaa;--line:#343434}*{box-sizing:border-box}body{margin:0;background:var(--black);color:var(--text);font:16px/1.5 Arial,Helvetica,sans-serif}main{max-width:1456px;margin:auto;padding:28px 28px 42px}h1{font:600 clamp(25px,3vw,36px)/1.15 'Space Grotesk',sans-serif;letter-spacing:-.035em;margin:8px 0 12px}p{color:var(--muted);margin:8px 0}.eyebrow,.status,figcaption{font:400 12px/1.5 'IBM Plex Mono',monospace}.eyebrow{color:var(--muted)}header p{max-width:850px}.toolbar{display:flex;align-items:center;gap:10px;flex-wrap:wrap;margin:22px 0 14px}button{font:inherit;border:1px solid #555;padding:9px 15px;border-radius:3px;background:#161616;color:var(--text);cursor:pointer}button:hover{background:#292929}button[aria-pressed=true]{background:var(--text);color:#111}button:focus-visible,a:focus-visible{outline:2px solid white;outline-offset:4px}.status{color:var(--muted);margin-left:5px}figure{margin:0}figcaption{display:flex;justify-content:space-between;gap:12px;color:var(--muted);padding:9px 0}.stage{position:relative;container-type:inline-size;aspect-ratio:16/9;background:#000;overflow:hidden;border:1px solid #242424}.map{position:absolute;inset:0;width:100%;height:100%;object-fit:contain;opacity:0}.idle-screen{position:absolute;inset:0;opacity:1;text-align:center}.identity{position:absolute;top:19%;left:5%;right:5%}.project-title{font:600 7.6cqw/1.02 'Space Grotesk',sans-serif;letter-spacing:-.045em;color:var(--text)}.project-title span{display:block}.authors{margin-top:2.3cqw;color:var(--text);font:400 1.3cqw/1.7 'IBM Plex Mono',monospace}.authors span{display:block}.contributor{margin-top:.55cqw;font:400 1.05cqw/1.5 'IBM Plex Mono',monospace;color:#aaa}.visit{position:absolute;bottom:4.5%;left:0;right:0;display:flex;align-items:center;flex-direction:column;gap:.65cqw;color:var(--text);text-decoration:none;font:400 1.05cqw/1.5 'IBM Plex Mono',monospace}.visit svg{display:block;width:8.8cqw;height:8.8cqw}.visit:hover span{text-decoration:underline}.notes{display:grid;grid-template-columns:1fr 1fr;gap:32px;margin-top:20px;border-top:1px solid var(--line);padding-top:16px}.notes h2{font:600 19px/1.25 'Space Grotesk',sans-serif;margin:0 0 9px}.notes p{font-size:14px}.notes a{color:var(--text)}footer{font-size:12px;color:var(--muted);margin-top:24px}noscript p{border:1px solid #555;padding:12px}@media(max-width:650px){main{padding:22px 14px 30px}.notes{grid-template-columns:1fr;gap:18px}.status{width:100%;margin:0}figcaption{font-size:10px}button{font-size:14px;padding:9px 12px}}
</style></head><body><main>
<header><div class="eyebrow">CARTOGRAPHY UNSEEN / MAP WINDOW / APPROVED DIRECTION A</div><h1>The title screen between journeys.</h1><p>The chosen crossfade, now with the project's own typefaces, author credits and a scannable website link. The preview opens on the idle title so you can inspect the layout.</p></header>
<div class="toolbar" aria-label="Preview controls"><button id="idle" type="button" aria-pressed="true">Replay autowalk fade</button><button id="active" type="button" aria-pressed="false">Resume interaction</button><button id="qr-toggle" type="button" aria-pressed="true">QR code on</button><span class="status" role="status" aria-live="polite">Autowalk · title and credits visible</span></div>
<figure><figcaption><span>PROJECTOR 2 / MAP WINDOW</span><span>16:9 exhibition preview</span></figcaption><div class="stage"><img class="map" src="__IMAGE__" alt="Recorded journey map with its route and captured views"><div class="idle-screen"><div class="identity"><div class="project-title"><span>Cartography</span><span>Unseen</span></div><div class="authors"><span>Nhat (Hong) Pham · Agnieszka Kiejziewicz</span><span>Ricardo Arce · Kok Yoong Lim</span></div><div class="contributor">Contributor · Tom Nguyen</div></div><a class="visit" href="https://emergentplay.bynhat.com/" target="_blank" rel="noopener noreferrer">__QR__<span>emergentplay.bynhat.com</span></a></div></div></figure>
<noscript><p>The idle title, credits and QR code are visible above. Enable JavaScript to replay the crossfade.</p></noscript>
<section class="notes" aria-label="Preview details"><div><h2>The same typography</h2><p>Space Grotesk SemiBold for the title. IBM Plex Mono for credits and the website address. Both are the bundled fonts used by the main experience.</p><p>Credits follow the <a href="https://emergentplay.bynhat.com/#team" target="_blank" rel="noopener noreferrer">Emergent Play website</a>: Nhat (Hong) Pham, Agnieszka Kiejziewicz, Ricardo Arce and Kok Yoong Lim. Tom Nguyen remains credited separately as contributor.</p></div><div><h2>One fade, one quiet screen</h2><p>Autowalk crossfades the map into the complete title screen over 1.2 seconds. Interaction returns to the map in 0.6 seconds. Try interrupting a fade; it reverses from its current opacity.</p><p>The QR code and address fade with the credits. Toggle them to compare the layout. The code links directly to the website, with a white border for scanning. Reduced motion switches immediately.</p></div></section>
<footer>6 September 2026 · Updated design preview using an actual map still. The new title layout and transition are not yet applied to the game. QR scanning distance still needs a check on the exhibition projector.</footer>
</main><script>
(function(){
'use strict';
const reduced=window.matchMedia('(prefers-reduced-motion: reduce)');
const idle=document.getElementById('idle'),active=document.getElementById('active'),qr=document.getElementById('qr-toggle');
const map=document.querySelector('.map'),title=document.querySelector('.idle-screen'),visit=document.querySelector('.visit'),status=document.querySelector('.status');
let timers=[],isIdle=true;
function clearTimers(){timers.forEach(clearTimeout);timers=[];}
function fade(el,to,duration){const from=getComputedStyle(el).opacity;el.getAnimations().forEach(function(a){a.cancel();});el.style.opacity=to;if(!reduced.matches&&duration)el.animate([{opacity:from},{opacity:to}],{duration:duration,easing:'cubic-bezier(.4,0,.2,1)'});}
function change(next,duration){
 clearTimers();isIdle=next;idle.setAttribute('aria-pressed',String(next));active.setAttribute('aria-pressed',String(!next));
 title.inert=!next;title.setAttribute('aria-hidden',String(!next));map.setAttribute('aria-hidden',String(next));
 fade(map,next?0:1,duration);fade(title,next?1:0,duration);
 status.textContent=next?'Autowalk · fading to title and credits':'Interaction active · map returning';
 timers.push(setTimeout(function(){status.textContent=next?'Autowalk · title and credits visible':'Interaction active · map visible';},reduced.matches?0:duration));
}
idle.addEventListener('click',function(){if(isIdle){change(false,0);timers.push(setTimeout(function(){change(true,1200);},reduced.matches?0:400));}else change(true,1200);});
active.addEventListener('click',function(){change(false,600);});
qr.addEventListener('click',function(){const show=qr.getAttribute('aria-pressed')!=='true';qr.setAttribute('aria-pressed',String(show));qr.textContent=show?'QR code on':'QR code off';visit.style.display=show?'flex':'none';});
reduced.addEventListener('change',function(){change(isIdle,0);});
map.setAttribute('aria-hidden','true');
})();
</script></body></html>'''
html = html.replace('__FONTS__',fonts).replace('__IMAGE__',map_image).replace('__QR__',qr_svg)
html += '\n<!-- Embedded font licenses\n' + '\n'.join((ROOT/'assets/fonts'/name).read_text(encoding='utf-8') for name in ['SpaceGrotesk-OFL.txt','IBMPlexMono-OFL.txt']) + '\n-->\n'
out = ROOT / 'docs/prototypes/map-idle-transitions.html'
out.write_text(html,encoding='utf-8')
assert out.stat().st_size < 512*1024
print(f'{out}: {out.stat().st_size} bytes; QR {size} modules including quiet zone')
