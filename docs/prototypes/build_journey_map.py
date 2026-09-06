"""Build a standalone design study. No application code or live game connection."""

from base64 import b64encode
from io import BytesIO
from math import sin, cos, atan2, sqrt
from pathlib import Path
import json

from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).with_name('journey-map.html')
textures = []
for source in sorted((ROOT / 'screenshot').glob('*.png'))[-12:][::2]:
    stream = BytesIO()
    ImageOps.fit(Image.open(source).convert('RGB'), (256, 160)).save(stream, 'JPEG', quality=66)
    textures.append('data:image/jpeg;base64,' + b64encode(stream.getvalue()).decode())


def point(t):
    return [160*sin(t*6.283185)+35*sin(t*18.849556), 38*sin(t*9.424778), 115*sin(t*12.566371)+20*cos(t*18.849556)]


route = [point(i/260) for i in range(261)]


def project(p, target, angle=.16, scale=1.7):
    x,y,z = [p[i]-target[i] for i in range(3)]
    depth = sin(angle)*x+cos(angle)*z
    return [480+scale*(cos(angle)*x-sin(angle)*z), 290+scale*(.7*depth-.714*y), .714*depth+.7*y]


def plane(i, width):
    p = route[i]
    nxt = point((i+1)/260)
    dx,dy,dz = [nxt[k]-p[k] for k in range(3)]
    yaw = atan2(dx,-dz)+.55*sin(i*.12)
    pitch = atan2(dy,sqrt(dx*dx+dz*dz))
    right = [cos(yaw),0,sin(yaw)]
    up = [-sin(yaw)*sin(pitch),cos(pitch),cos(yaw)*sin(pitch)]
    return [[p[k]+sx*width*right[k]+sy*width*.625*up[k] for k in range(3)] for sx,sy in [(-.5,.5),(.5,.5),(-.5,-.5)]]


def static_scene(mode):
    target = route[260]
    step, width = [(10,45),(3,34),(15,30)][mode]
    bg = ['#000000','#08080b','#091119'][mode]
    parts = [f'<svg viewBox="0 0 960 540" role="img" aria-label="Option {chr(65+mode)}, a three dimensional route with generated image planes"><rect width="960" height="540" fill="{bg}"/>']
    if mode == 2:
        for n in range(-240,241,60):
            for a,b in [([n,-55,-200],[n,-55,240]),([-240,-55,n],[240,-55,n])]:
                p,q=project(a,target),project(b,target)
                parts.append(f'<path d="M{p[0]:.2f},{p[1]:.2f}L{q[0]:.2f},{q[1]:.2f}" stroke="#26313b" fill="none"/>')
    pts=[project(p,target) for p in route[:261]]
    parts.append('<polyline points="'+' '.join(f'{p[0]:.2f},{p[1]:.2f}' for p in pts)+'" fill="none" stroke="#c8d4db" stroke-width="1.2"/>')
    for i in sorted(range(0,261,step),key=lambda i:project(route[i],target)[2],reverse=True):
        p,q,r=[project(v,target) for v in plane(i,width)]
        if mode == 2:
            bottom=project([route[i][0],-55,route[i][2]],target)
            center=project(route[i],target)
            parts.append(f'<path d="M{center[0]},{center[1]}L{bottom[0]},{bottom[1]}" stroke="#657e8c" stroke-width=".7"/>')
        matrix=f'{(q[0]-p[0])/256} {(q[1]-p[1])/256} {(r[0]-p[0])/160} {(r[1]-p[1])/160} {p[0]} {p[1]}'
        parts.append(f'<use href="#texture{(i//step)%len(textures)}" transform="matrix({matrix})"/>')
    parts.append('<circle cx="480" cy="290" r="5" fill="#fff"/><circle cx="480" cy="290" r="13" fill="none" stroke="#fff" stroke-width="1"/><path d="M497 282L528 260H625" fill="none" stroke="#fff" stroke-width=".8"/><text x="533" y="253" fill="#fff" font-size="14" font-family="monospace">YOU ARE HERE</text></svg>')
    return ''.join(parts)


directions = [
    ('A','Image field','Closest to your sketch. A fine route hangs in black space; individual views keep their own viewing angle.','My pick. The empty space gives the images room, even at exhibition scale.'),
    ('B','Memory ribbon','More frequent views gather into a continuous, folded strip. Turns and revisits become dense knots of images.','Stronger as a moving sculpture. Overlap makes individual views harder to read.'),
    ('C','Spatial survey','Smaller views sit above a faint reference plane. Vertical threads reveal climbs and descents.','Clearest geography. The grid gives it a more technical character.'),
]
cards=''.join(f'<article class="choice"><div class="option-title"><span>{letter}</span><h2>{title}</h2></div><div class="thumb">{static_scene(i)}</div><p>{desc}</p><p class="trade">{trade}</p><button class="choose" data-mode="{i}" aria-pressed="{str(i==0).lower()}">Preview {letter}</button></article>' for i,(letter,title,desc,trade) in enumerate(directions))
defs='<svg class="texture-library" aria-hidden="true"><defs>'+''.join(f'<image id="texture{i}" width="256" height="160" href="{src}"/>' for i,src in enumerate(textures))+'</defs></svg>'
fonts=''.join(f"@font-face{{font-family:'{family}';src:url(data:font/woff;base64,{b64encode((ROOT/'assets/fonts'/file).read_bytes()).decode()}) format('woff');font-weight:{weight};font-display:swap}}" for family,file,weight in [('Space','SpaceGrotesk-SemiBold.woff',600),('Plex','IBMPlexMono-Regular.woff',400)])
page='''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Mapping the Blackbox | Journey map study</title><style>__FONTS__
:root{color-scheme:dark;--black:#000;--surface:#111317;--line:#353a40;--white:#f0f2f4;--muted:#a9b0b8;--ice:#b7d1df;--ink:#091119}*{box-sizing:border-box}body{margin:0;background:var(--black);color:var(--white);font:16px/1.55 'Segoe UI',sans-serif}main{max-width:1600px;margin:auto;padding:32px 36px 60px}h1,h2,h3{font-family:Space,sans-serif;font-weight:600;line-height:1.13}h1{font-size:clamp(30px,4vw,48px);margin:7px 0 14px}h2{font-size:24px;margin:0}h3{font-size:22px}.eyebrow,.small,button,output,label,figcaption{font-family:Plex,monospace;font-size:12px}.eyebrow{color:var(--muted)}.intro{max-width:950px;font-size:18px;color:#cdd2d6;margin:0 0 28px}.options{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:24px}.option-title{display:flex;align-items:center;gap:13px;margin:0 0 16px}.option-title>span{font:26px Plex,monospace;color:var(--ice)}.thumb{border:1px solid var(--line);overflow:hidden;aspect-ratio:16/9}.thumb svg{display:block;width:100%;height:100%}.choice p{margin:14px 0 9px}.trade{color:var(--muted);font-size:14px;min-height:44px}button{background:var(--surface);border:1px solid var(--line);color:var(--white);padding:10px 16px;cursor:pointer;border-radius:2px}button[aria-pressed=true]{color:var(--black);background:var(--ice);border-color:var(--ice)}button:hover{border-color:var(--ice)}button:focus-visible,input:focus-visible,summary:focus-visible{outline:2px solid var(--ice);outline-offset:5px}.choose{margin-top:4px}.preview{margin-top:36px;padding-top:25px;border-top:1px solid var(--line)}.preview-head{display:flex;align-items:baseline;gap:18px;flex-wrap:wrap}.preview-head p{margin:0;color:var(--muted)}.controls{display:flex;gap:12px;align-items:center;flex-wrap:wrap;margin:22px 0}.scrub{display:flex;gap:12px;align-items:center;margin-left:auto}input{accent-color:var(--ice);width:160px}.screens{display:grid;grid-template-columns:1fr 2.45fr;gap:22px;align-items:center}figure{margin:0}figcaption{display:flex;justify-content:space-between;color:var(--muted);margin-bottom:10px;gap:8px}.screen{position:relative;aspect-ratio:16/9;overflow:hidden;background:#000;border:1px solid var(--line)}.main-image{width:100%;height:100%;object-fit:cover}.map-screen svg,.map-screen canvas{position:absolute;width:100%;height:100%;inset:0}.map-screen canvas{display:none}.map-screen.ready canvas{display:block}.map-screen.ready svg{display:none}.projector-label{position:absolute;top:18px;left:20px;font:14px Space,sans-serif;letter-spacing:-.02em}.map-meta{position:absolute;bottom:16px;left:20px;font:10px Plex,monospace;color:var(--muted)}.map-screen.auto>*{visibility:hidden}.hint{color:var(--muted);font-size:13px;margin-top:12px}.notes{display:grid;grid-template-columns:1fr 1fr;gap:50px;margin-top:28px;border-top:1px solid var(--line)}.notes p{color:#cdd2d6}.notes h3{margin-bottom:12px}table{border-collapse:collapse;width:100%;font-size:14px}th,td{text-align:left;padding:10px 0;border-bottom:1px solid var(--line);vertical-align:top}th{font-weight:400;color:var(--muted);padding-right:20px;width:29%}details{margin-top:30px;padding-top:20px;border-top:1px solid var(--line);max-width:1040px}summary{cursor:pointer;font-family:Space,sans-serif;font-size:18px}details p,details li{color:var(--muted);font-size:14px}li{margin:10px 0}.texture-library{position:absolute;width:0;height:0;overflow:hidden}footer{margin-top:30px;color:var(--muted);font-size:12px}.js-only{display:none}.has-js .js-only{display:flex}.has-js .choose{display:inline-block}body:not(.has-js) .choose{display:none}
@media(max-width:900px){main{padding:24px 20px 40px}.options{gap:16px}.option-title{align-items:flex-start}h2{font-size:20px}.screens{grid-template-columns:1fr}.experience{max-width:380px;width:100%}.scrub{margin-left:0}.notes{gap:26px}}@media(max-width:650px){.options,.notes{grid-template-columns:1fr}.choice{padding-bottom:22px;border-bottom:1px solid var(--line)}.thumb{aspect-ratio:16/9}.trade{min-height:0}.preview-head{gap:9px}.projector-label{font-size:11px;left:10px;top:10px}.map-meta{font-size:8px;left:10px;bottom:8px}.controls{gap:8px}.scrub{width:100%;flex-wrap:wrap}input{flex:1}.notes{gap:0}h1{font-size:32px}}@media(prefers-reduced-motion:reduce){*{scroll-behavior:auto}}
</style></head><body>__DEFS__<main><header><div class="eyebrow">CARTOGRAPHY UNSEEN / TWO-PROJECTOR STUDY / 5 SEPTEMBER 2026</div><h1>Mapping the Blackbox</h1><p class="intro">One screen is the journey. The other collects what the player has seen, at the places and angles where they saw it. Three directions for the second projection.</p></header><section class="options" aria-label="Compare map directions">__CARDS__</section>
<section class="preview" aria-label="Two projector preview"><div class="preview-head"><h2 id="preview-title">A / Image field</h2><p>Two independent windows, each fullscreen on its own projector.</p></div><div class="controls js-only"><button id="orbit" aria-pressed="false">Start slow orbit</button><button id="walk" aria-pressed="false">Play sample journey</button><button id="auto" aria-pressed="false">Preview automatic flight</button><label class="scrub">Journey <input id="progress" type="range" min="8" max="100" value="100" aria-label="Journey progress"><output id="elapsed">03:00</output></label></div><div class="screens"><figure class="experience"><figcaption><span>PROJECTOR 1</span><span>First-person view</span></figcaption><div class="screen"><img class="main-image" id="main-image" src="__MAIN__" alt="An actual generated view from Cartography Unseen"></div></figure><figure><figcaption><span>PROJECTOR 2</span><span id="state-label" aria-live="polite">Human interaction / map visible</span></figcaption><div id="map-screen" class="screen map-screen">__STATIC__<canvas id="map" aria-label="Interactive 3D journey map, showing the route and image planes"></canvas><div class="projector-label">Mapping the Blackbox</div><div class="map-meta" id="map-meta">03:00 / WORLD POSITION +000, +000, +020</div></div></figure></div><p class="hint" id="preview-note">Illustrative route using actual generated images from this project. The two views are simulated, not connected to a running game. Use the controls to inspect the proposal.</p><noscript><p>The static previews show all three designs. Enable JavaScript to inspect the orbit, route growth and automatic-flight state.</p></noscript></section>
<section class="notes"><div><h3>The map follows the person</h3><table><tbody><tr><th>Human play</th><td>The map is black at startup and appears on the first movement or look input. A thin line records movement in all three dimensions. Looking around can add a view even without moving.</td></tr><tr><th>A brief pause</th><td>The map stays visible during a proposed 10-second grace period, so it does not blink between gestures.</td></tr><tr><th>No interaction</th><td>The second projector becomes black after that grace period. Capture and orbit stop. Automatic flight never records a route.</td></tr><tr><th>Returning</th><td>The accumulated map returns. A new stroke starts at the current position, leaving a gap for any automatic travel.</td></tr></tbody></table></div><div><h3>Let the route stay legible</h3><p>I recommend A. Keep the field open and let the generated images supply the color. The map camera circles the player's position slowly, at roughly one revolution per three minutes. The view planes keep their original orientation as the map camera moves.</p><p>Sample views after meaningful movement or a change in gaze, with a capture-rate limit. A new image every display frame would turn the map into a wall almost immediately.</p><p>Keep the route across visits during the exhibition. Store the image history, then show older views more sparsely as the map grows. A deliberate operator action starts a new archive.</p></div></section>
<details><summary>Implementation notes and decisions behind the mock</summary><p>This is a design proposal. Production code is unchanged. The preview uses a synthetic route with real saved AI images; image order does not claim to reproduce the recorded camera path.</p><ul><li>Use clean generated frames as the view planes. The existing GeneratedFrame already carries its source world position, pitch, yaw, timestamp and sequence. Pair each image with that source pose, since generation finishes after the player has moved. Exact displayed crops and reprojection would need a separate capture path.</li><li>Center each plane at its recorded XYZ position and align its normal with the recorded gaze, including pitch. Preserve those poses permanently. The orbit moves only the overview camera. The back of a view plane may show a dimmed, mirrored image; it must not swivel to face the map viewer.</li><li>A light map viewer in a separate process is the proposed starting point. Pass sampled images and route events through bounded local communication, with history retained outside graphics memory. It must not block the main input, display or diffusion loops.</li><li>Give both windows independent display selection and fullscreen control. Keep the map window open but black while inactive so the second projector does not reveal the Windows desktop. The interactive prototype simulates this arrangement inside one document.</li><li>Starting values to tune in the app: one map update per second at most for image capture, captures triggered by distance or gaze change, a 30 FPS orbit cap, and a 10-second human-activity grace period. Limit texture residency and simplify distant history as the archive grows. These are proposals, not measured performance.</li><li>This is a spatial record of generated views. The procedural world provides the coordinates; the model's latent space is not a directly measured three-dimensional terrain. Revisiting one coordinate may produce a different image, and both observations can belong in the archive.</li></ul><p>Code reviewed for feasibility: app/types.py, app/renderer/camera.py, app/diffusion/worker.py and the window/input loop. The existing ten-second first-person trail is separate from this persistent overview archive.</p></details><footer>Design decision pending. Choose A, B or C before implementation. Sample journey is three minutes long. Motion starts only when requested; reduced-motion mode uses manual orbit steps.</footer></main>
<script>__SCRIPT__</script></body></html>'''

script = r'''
'use strict';
document.body.classList.add('has-js');
const route=__ROUTE__, titles=['Image field','Memory ribbon','Spatial survey'];
const canvas=document.getElementById('map'),ctx=canvas.getContext('2d'),screen=document.getElementById('map-screen');
const slider=document.getElementById('progress'),orbitBtn=document.getElementById('orbit'),walkBtn=document.getElementById('walk'),autoBtn=document.getElementById('auto');
const reduced=matchMedia('(prefers-reduced-motion: reduce)').matches;
let zoom=1.7;
let mode=0,angle=.16,progress=1,orbit=false,walking=false,automatic=false,frame=0,last=0;
const textures=[...document.querySelectorAll('.texture-library image')].map(el=>{const img=new Image();img.src=el.getAttribute('href');return img;});
function position(t){return [160*Math.sin(t*6.283185)+35*Math.sin(t*18.849556),38*Math.sin(t*9.424778),115*Math.sin(t*12.566371)+20*Math.cos(t*18.849556)];}
function proj(p,target){const x=p[0]-target[0],y=p[1]-target[1],z=p[2]-target[2],d=Math.sin(angle)*x+Math.cos(angle)*z;return [480+zoom*(Math.cos(angle)*x-Math.sin(angle)*z),290+zoom*(.7*d-.714*y),.714*d+.7*y];}
function plane(i,width){const p=route[i],n=position((i+1)/260),dx=n[0]-p[0],dy=n[1]-p[1],dz=n[2]-p[2],yaw=Math.atan2(dx,-dz)+.55*Math.sin(i*.12),pitch=Math.atan2(dy,Math.hypot(dx,dz)),right=[Math.cos(yaw),0,Math.sin(yaw)],up=[-Math.sin(yaw)*Math.sin(pitch),Math.cos(pitch),Math.cos(yaw)*Math.sin(pitch)];return [[-.5,.5],[.5,.5],[-.5,-.5]].map(([sx,sy])=>p.map((v,k)=>v+sx*width*right[k]+sy*width*.625*up[k]));}
function line(points,color,width=1){ctx.beginPath();points.forEach((p,i)=>i?ctx.lineTo(p[0],p[1]):ctx.moveTo(p[0],p[1]));ctx.strokeStyle=color;ctx.lineWidth=width;ctx.stroke();}
function draw(){
 if(!ctx)return;
 const ratio=Math.min(devicePixelRatio||1,1.5),w=Math.max(320,Math.round(canvas.clientWidth*ratio)),h=Math.round(w*540/960);
 if(canvas.width!==w||canvas.height!==h){canvas.width=w;canvas.height=h;}
 ctx.setTransform(w/960,0,0,h/540,0,0);ctx.fillStyle=automatic?'#000':['#000','#08080b','#091119'][mode];ctx.fillRect(0,0,960,540);if(automatic)return;
 const end=Math.floor(progress*260),target=route[end],step=[10,3,15][mode],width=[45,34,30][mode];
 zoom=1.7;let fit=1;for(const p of route.slice(0,end+1)){const q=proj(p,target);fit=Math.min(fit,415/(Math.abs(q[0]-480)+42),215/(Math.abs(q[1]-290)+35));}zoom*=fit;
 if(mode===2)for(let n=-240;n<=240;n+=60){line([proj([n,-55,-200],target),proj([n,-55,240],target)],'#26313b');line([proj([-240,-55,n],target),proj([240,-55,n],target)],'#26313b');}
 line(route.slice(0,end+1).map(p=>proj(p,target)),'#c8d4db',1.2);
 const indices=[];for(let i=0;i<=end;i+=step)indices.push(i);indices.sort((a,b)=>proj(route[b],target)[2]-proj(route[a],target)[2]);
 for(const i of indices){const [p,q,r]=plane(i,width).map(v=>proj(v,target));if(mode===2)line([proj(route[i],target),proj([route[i][0],-55,route[i][2]],target)],'#657e8c',.7);const img=textures[(i/step)%textures.length];ctx.save();ctx.transform((q[0]-p[0])/256,(q[1]-p[1])/256,(r[0]-p[0])/160,(r[1]-p[1])/160,p[0],p[1]);if(img.complete&&img.naturalWidth)ctx.drawImage(img,0,0,256,160);ctx.strokeStyle='#9cadb6';ctx.lineWidth=.8;ctx.strokeRect(0,0,256,160);ctx.restore();}
 ctx.fillStyle='#fff';ctx.beginPath();ctx.arc(480,290,5,0,Math.PI*2);ctx.fill();ctx.beginPath();ctx.arc(480,290,13,0,Math.PI*2);ctx.strokeStyle='#fff';ctx.lineWidth=1;ctx.stroke();
 line([[497,282],[528,260],[665,260]],'#fff',.8);ctx.font='13px Plex,monospace';ctx.fillText('YOU ARE HERE',533,251);
 const time=Math.floor(progress*180),stamp=String(Math.floor(time/60)).padStart(2,'0')+':'+String(time%60).padStart(2,'0');document.getElementById('elapsed').value=stamp;
 document.getElementById('map-meta').textContent=stamp+' / XYZ '+target.map(v=>(v<0?'':'+')+Math.round(v)).join(', ')+' / '+indices.length+' VIEWS';
 const src=textures[Math.floor(end/step)%textures.length].src,main=document.getElementById('main-image');if(main.src!==src)main.src=src;
}
function stop(){cancelAnimationFrame(frame);frame=0;last=0;}
function tick(now){frame=0;if(automatic||document.hidden||(!orbit&&!walking))return;if(!last)last=now;const dt=(now-last)/1000;if(dt>=1/30){last=now;if(orbit)angle+=dt*Math.PI*2/180;if(walking){progress=Math.min(1,progress+dt/180);slider.value=progress*100;if(progress===1){walking=false;walkBtn.textContent='Replay sample journey';walkBtn.setAttribute('aria-pressed','false');}}draw();}if(orbit||walking)frame=requestAnimationFrame(tick);}
function schedule(){stop();if(!automatic&&!document.hidden&&(orbit||walking))frame=requestAnimationFrame(tick);}
document.querySelectorAll('.choose').forEach(button=>button.addEventListener('click',()=>{mode=Number(button.dataset.mode);document.querySelectorAll('.choose').forEach(b=>b.setAttribute('aria-pressed',String(b===button)));document.getElementById('preview-title').textContent=String.fromCharCode(65+mode)+' / '+titles[mode];draw();}));
orbitBtn.addEventListener('click',()=>{if(reduced){angle+=Math.PI/12;draw();return;}orbit=!orbit;orbitBtn.textContent=orbit?'Pause orbit':'Start slow orbit';orbitBtn.setAttribute('aria-pressed',String(orbit));schedule();});
walkBtn.addEventListener('click',()=>{if(reduced){progress=progress>=1?.08:Math.min(1,progress+.08);slider.value=progress*100;draw();return;}walking=!walking;if(walking&&progress>=.99)progress=.08;walkBtn.textContent=walking?'Pause journey':'Play sample journey';walkBtn.setAttribute('aria-pressed',String(walking));schedule();});
autoBtn.addEventListener('click',()=>{automatic=!automatic;screen.classList.toggle('auto',automatic);autoBtn.textContent=automatic?'Return to human play':'Preview automatic flight';autoBtn.setAttribute('aria-pressed',String(automatic));document.getElementById('state-label').textContent=automatic?'Automatic flight / map black, capture stopped':'Human interaction / map visible';slider.disabled=automatic;walkBtn.disabled=automatic;orbitBtn.disabled=automatic;draw();schedule();});
slider.addEventListener('input',()=>{progress=Number(slider.value)/100;walking=false;walkBtn.textContent=reduced?'Advance sample journey':'Play sample journey';walkBtn.setAttribute('aria-pressed','false');draw();schedule();});
document.addEventListener('visibilitychange',schedule);new ResizeObserver(draw).observe(canvas);
if(reduced){orbitBtn.textContent='Rotate map 15 degrees';walkBtn.textContent='Advance sample journey';}
Promise.all(textures.map(img=>img.decode())).then(()=>{screen.classList.add('ready');draw();}).catch(()=>{document.getElementById('preview-note').textContent='The static map is available. Interactive image loading failed; reload this document to try again.';});
document.fonts.ready.then(draw);
'''.replace('__ROUTE__',json.dumps(route,separators=(',',':')))
page=page.replace('__FONTS__',fonts).replace('__DEFS__',defs).replace('__CARDS__',cards).replace('__MAIN__',textures[4]).replace('__STATIC__',static_scene(0)).replace('__SCRIPT__',script)
page += '\n<!-- Embedded font licenses\n' + '\n'.join((ROOT/'assets/fonts'/f).read_text(encoding='utf-8') for f in ['SpaceGrotesk-OFL.txt','IBMPlexMono-OFL.txt']) + '\n-->\n'
OUT.write_text(page,encoding='utf-8')
assert OUT.stat().st_size <= 512*1024
print(f'{OUT.name}: {OUT.stat().st_size:,} bytes')
