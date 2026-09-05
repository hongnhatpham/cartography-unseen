"""Isolated shape/color studies using the real proxy and diffusion backend.

Writes only study artifacts. It does not change live geometry, shaders or config.
The scene is static: collision/streaming for these meshes is not implemented here.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import asdict, replace
from math import pi
from pathlib import Path
from time import perf_counter
import json
import sys

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from app.config import AppConfig, configure_local_environment
from app.renderer import world
from app.renderer.camera import Camera
from app.renderer.proxy_renderer import ProxyRenderer

OUT = ROOT / 'logs/reference/form-study'
PALETTES = (
    ((.10,.72,.76), (.16,.31,.80), (.79,.89,.13), (.90,.42,.59), (.74,.70,.84)),
    ((.04,.69,.68), (.32,.25,.84), (.92,.78,.12), (.87,.30,.48), (.72,.77,.82)),
)


def tube(points, radius=.065, sides=6):
    points = np.asarray(points, dtype=float)
    rings = []
    for i, point in enumerate(points):
        tangent = points[min(i+1,len(points)-1)] - points[max(0,i-1)]
        tangent /= np.linalg.norm(tangent)
        axis = np.array([0.,1.,0.]) if abs(tangent[1]) < .9 else np.array([1.,0.,0.])
        u = np.cross(tangent, axis)
        u /= np.linalg.norm(u)
        v = np.cross(tangent,u)
        angles = np.arange(sides)*2*pi/sides
        normals = np.cos(angles)[:,None]*u + np.sin(angles)[:,None]*v
        rings.append(np.concatenate((point+radius*normals,normals),axis=1))
    rows = []
    for i in range(len(rings)-1):
        for j in range(sides):
            k=(j+1)%sides
            rows.extend((rings[i][j],rings[i+1][j],rings[i+1][k],
                         rings[i][j],rings[i+1][k],rings[i][k]))
    return np.asarray(rows,dtype='f4')


def rounded(exponent=.45):
    def vertex(u,v):
        raw=np.array([np.cos(v)*np.cos(u),np.sin(v),np.cos(v)*np.sin(u)])
        p=np.sign(raw)*np.abs(raw)**exponent
        n=np.sign(p)*np.abs(p)**(2/exponent-1)
        n/=max(np.linalg.norm(n),1e-8)
        return np.r_[p,n]
    rows=[]
    for i in range(24):
        for j in range(12):
            u0,u1=i*2*pi/24,(i+1)*2*pi/24
            v0,v1=-pi/2+j*pi/12,-pi/2+(j+1)*pi/12
            a,b,c,d=vertex(u0,v0),vertex(u1,v0),vertex(u1,v1),vertex(u0,v1)
            rows.extend((a,b,c,a,c,d))
    return np.asarray(rows,dtype='f4')


def meshes():
    cage=[]
    for axis in range(3):
        for a in (-.85,.85):
            for b in (-.85,.85):
                ends=np.zeros((2,3)); ends[:,axis]=[-.85,.85]
                other=[k for k in range(3) if k!=axis]
                ends[:,other[0]]=a; ends[:,other[1]]=b
                cage.append(tube(ends,.09))
    ribs=[]
    for z in (-.75,0.,.75):
        theta=np.linspace(0,pi,19)
        ribs.append(tube(np.c_[.86*np.cos(theta),1.55*np.sin(theta)-.75,np.full_like(theta,z)],.075))
    for x in (-.86,.86):
        ribs.append(tube([[x,-.75,-.85],[x,-.75,.85]],.08))
    weave=[]
    t=np.linspace(-.88,.88,11)
    for level in np.linspace(-.85,.85,7):
        # A curved saddle sheet, rather than a square pattern on a solid face.
        weave.append(tube(np.c_[t,np.full_like(t,level),.65*(t*t-level*level)],.043))
        weave.append(tube(np.c_[np.full_like(t,level),t,.65*(level*level-t*t)],.043))
    theta=np.linspace(0,2*pi,33)
    ring=tube(np.c_[.77*np.cos(theta),.77*np.sin(theta),np.zeros_like(theta)],.18,8)
    result={'rounded':rounded(), 'oval':rounded(1.), 'cage':np.concatenate(cage),
            'ribs':np.concatenate(ribs), 'weave':np.concatenate(weave), 'ring':ring}
    for name in ('weave','ribs','ring'):
        for axis,order in enumerate(([2,0,1],[0,2,1],[0,1,2])):
            source=result[name]
            result[f'{name}{axis}']=np.ascontiguousarray(np.c_[source[:,:3][:,order],source[:,3:][:,order]],dtype='f4')
    return result


def shape_for(cube, seed, variant):
    pick=world._unit_float(seed,0xC08BE,*[int(v*8) for v in cube.position])
    if variant=='current': return 'cube'
    if cube.role=='panel':
        # Select an optional companion form; the source panel stays in place.
        if variant=='a': return 'rounded' if pick<.18 else 'cube'
        if variant=='b': return 'weave' if pick<.36 else ('rounded' if pick<.45 else 'cube')
        return 'ribs' if pick<.20 else ('weave' if pick<.32 else ('rounded' if pick<.45 else 'cube'))
    if variant=='a': return 'rounded' if pick<.55 else ('ring' if pick<.70 else 'cube')
    if variant=='b': return 'cage' if pick<.25 else ('ribs' if pick<.55 else ('weave' if pick<.80 else ('rounded' if pick<.90 else 'cube')))
    return 'oval' if pick<.22 else ('ring' if pick<.45 else ('ribs' if pick<.68 else ('cage' if pick<.82 else 'cube')))


def colored(cube, seed, variant):
    if variant=='current': return cube
    x,y,z=cube.position
    group=tuple(int(np.floor(v/128.)) for v in cube.position)
    sample=world._unit_float(seed,0xC010B,*group)
    palette=PALETTES[0 if variant!='c' else 1]
    slot=min(4,int(sample*5))
    # Small accents share the broad region's palette rather than scattering
    # five unrelated full-strength colors over every object.
    if world._unit_float(seed,0xC010D,*[int(v//32) for v in cube.position])>.76:
        slot=(slot+2)%5
    chosen=np.array(palette[slot])
    field=world.value_noise(x,y,z,100.,seed,0xC010C)
    strong=field > {'a':.53,'b':.32,'c':.15}[variant]
    neutral=np.array([.77,.80,.82])
    color=chosen if strong else neutral*.75+chosen*.25
    # Preserve per-object light/dark variation, with color carried by whole
    # objects and broad patches instead of the surface checker.
    color*=.6+.4*cube.tone
    return replace(cube,color=tuple(color))


def study_root(variant):
    root=OUT/variant
    (root/'shaders').mkdir(parents=True,exist_ok=True)
    for source in (ROOT/'shaders').iterdir():
        text=source.read_text(encoding='utf-8')
        if source.name=='proxy.frag' and variant!='current':
            # Almost-zero coefficients keep the existing uniform binding alive.
            # This has no visible checker modulation in the 8-bit proxy image.
            text=text.replace('PANEL_COARSE_SWING = 0.22','PANEL_COARSE_SWING = 0.000001')
            text=text.replace('PANEL_FINE_SWING = 0.10','PANEL_FINE_SWING = 0.000001')
        if source.name in ('proxy.frag','sky.frag') and variant=='c':
            text=text.replace('mix(fog_color, zenith_color,', 'mix(fog_color, vec3(0.16, 0.36, 0.54) + zenith_color * 0.10,')
        if source.name=='proxy.vert':
            text=text.replace('mat3(model) * in_normal','transpose(inverse(mat3(model))) * in_normal')
        (root/'shaders'/source.name).write_text(text,encoding='utf-8')
    return root


def prepare(renderer,camera,variant,geometry):
    renderer._update_world(camera.position)
    if variant=='c': renderer.sky_color=(.12,.42,.46)
    groups=defaultdict(list)
    original=[cube for chunk in renderer._chunks.values() for cube in chunk.objects]
    objects=[('cube',colored(cube,renderer.world_seed,variant)) for cube in original]
    if variant!='current':
        for cube in original:
            pick=world._unit_float(renderer.world_seed,0xADD,*[int(v*8) for v in cube.position])
            share=.38 if cube.role=='mass' else (.0 if variant=='a' else .28)
            if pick>=share: continue
            name=shape_for(cube,renderer.world_seed,variant)
            if name=='cube': continue
            axis=int(np.argmin(cube.half_extents))
            half=np.array(cube.half_extents)*(.65 if name in ('oval','rounded') else .90)
            if name in ('weave','ribs','ring'):
                half[axis]=max(half[axis],min(v for k,v in enumerate(half) if k!=axis)*.4)
                name=f'{name}{axis}'
            position=np.array(cube.position)
            sign=-1 if pick<share/2 else 1
            position[axis]+=sign*(cube.half_extents[axis]+half[axis]*.8)
            item=colored(cube,renderer.world_seed,variant)
            if name.startswith(('weave','ribs')) or name=='cage':
                color=tuple(np.array(item.color)*.50+np.array([.68,.86,.92])*.50)
                item=replace(item,color=color)
            objects.append((name,replace(item,position=tuple(position),half_extents=tuple(half))))
    snapshot=camera.snapshot(renderer.render_width/renderer.render_height)
    clip=snapshot.projection_matrix@snapshot.view_matrix
    planes=np.array([clip[3]+sign*clip[axis] for axis in range(3) for sign in (-1,1)])
    planes/=np.linalg.norm(planes[:,:3],axis=1)[:,None]
    centers=np.array([cube.position for _,cube in objects])
    radius=np.linalg.norm([cube.half_extents for _,cube in objects],axis=1)*1.1
    visible=((centers@planes[:,:3].T+planes[:,3])>=-radius[:,None]).all(axis=1)
    for (name,item),shown in zip(objects,visible):
        if shown:
            groups[name].append(item)
    renderer._instance_counts={name:0 for name in renderer._instance_counts}
    for name,objects in groups.items():
        packed=renderer._pack_instances(tuple(objects),origin=renderer._render_origin)
        if name=='cube':
            renderer.instance_buffers[name].orphan(max(76,packed.nbytes))
            renderer.instance_buffers[name].write(packed)
        else:
            renderer.mesh_buffers[name]=renderer.ctx.buffer(geometry[name])
            renderer.instance_buffers[name]=renderer.ctx.buffer(packed)
            renderer.mesh_vaos[name]=renderer.ctx.vertex_array(renderer.proxy_program,[
                (renderer.mesh_buffers[name],'3f 3f','in_position','in_normal'),
                (renderer.instance_buffers[name],'4f 4f 4f 4f 3f /i',
                 'instance_model_0','instance_model_1','instance_model_2','instance_model_3','instance_color')])
        renderer._instance_counts[name]=len(objects)
    return {name:len(objects) for name,objects in groups.items()}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--ai',action='store_true')
    args=parser.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    configure_local_environment(ROOT,offline=True)
    config=AppConfig.load(ROOT/'config.json')
    (OUT/'settings.json').write_text(json.dumps(asdict(config),indent=2),encoding='utf-8')
    geometry=meshes()
    # Shared current-world open poses, discovered once and saved for replay.
    pose_path=OUT/'poses.json'
    if pose_path.exists():
        poses=json.loads(pose_path.read_text())
    else:
        renderer=ProxyRenderer(ROOT,config.diffusion_size,False,world_seed=config.world_seed)
        camera=Camera.create_default(); renderer.spawn_camera(camera)
        poses=[{'position':camera.position.tolist(),'yaw':camera.yaw,'pitch':camera.pitch}]
        camera.position += np.array([96.,32.,64.]); renderer.constrain_camera(camera)
        camera.yaw,camera.pitch,_=renderer.open_heading(camera)
        poses.append({'position':camera.position.tolist(),'yaw':camera.yaw,'pitch':camera.pitch})
        renderer.close()
        pose_path.write_text(json.dumps(poses,indent=2))
    backend=None
    if args.ai:
        from app.diffusion.factory import create_backend
        from tools.style_sheet import FrameClock
        backend=create_backend(config.backend); backend.load(config.backend_dict(ROOT))
    counts={}
    costs=[]
    try:
        for variant in ('current','a','b','c'):
            root=study_root(variant)
            for i,pose in enumerate(poses):
                renderer=ProxyRenderer(root,config.diffusion_size,False,world_seed=config.world_seed)
                try:
                    camera=Camera(position=np.array(pose['position']),yaw=pose['yaw'],pitch=pose['pitch'])
                    counts[f'{variant}-{i}']=prepare(renderer,camera,variant,geometry)
                    frame=renderer.capture_conditioning(renderer.render_scene(camera,manage_chunks=False),0.)
                    np.savez_compressed(OUT/f'{variant}-{i}-conditioning.npz',rgb=frame.rgb,depth=frame.depth)
                    Image.fromarray(frame.rgb).save(OUT/f'{variant}-{i}-proxy.png')
                    renderer.ctx.finish()
                    started=perf_counter()
                    for _ in range(15): renderer.render_scene(camera,manage_chunks=False)
                    renderer.ctx.finish()
                    costs.append({'variant':variant,'view':i,'proxy_ms':(perf_counter()-started)*1000/15})
                    if backend:
                        clock=FrameClock(10.); backend.set_clock(clock); backend.reseed(config.seed)
                        backend.apply_settings(config.backend_settings())
                        # Prompt held verbatim for all options; no palette words
                        # or family presets changed to force a candidate's look.
                        backend.set_prompt(config.prompt,config.negative_prompt)
                        clock.now=config.prompt_walk_seconds+1.; backend.prompt_walk.advance(); clock.now=0.
                        for _ in range(20):
                            output=backend.generate(frame); clock.tick()
                        Image.fromarray(output).save(OUT/f'{variant}-{i}-ai.png')
                    print('Rendered',variant,i,counts[f'{variant}-{i}'],flush=True)
                finally: renderer.close()
    finally:
        if backend: backend.unload()
    (OUT/'mesh-counts.json').write_text(json.dumps(counts,indent=2))
    (OUT/'render-cost.json').write_text(json.dumps(costs,indent=2))
    sheet=Image.new('RGB',(384*4,256*len(poses)+30),'#151515')
    draw=ImageDraw.Draw(sheet)
    for col,variant in enumerate(('current','a','b','c')):
        draw.text((col*384+8,8),variant,fill='white')
        for row in range(len(poses)):
            source=OUT/f'{variant}-{row}-{"ai" if args.ai else "proxy"}.png'
            sheet.paste(Image.open(source).resize((384,256)),(col*384,row*256+30))
    sheet.save(OUT/f'{"ai" if args.ai else "proxy"}-contact.jpg')


if __name__=='__main__': main()
