"""Compare one material-description change against the same B proxy."""
from dataclasses import asdict
import json
import numpy as np
from PIL import Image
from form_study import ROOT, OUT
from app.config import AppConfig, configure_local_environment
from app.diffusion.factory import create_backend
from app.renderer.camera import Camera
from app.types import ConditioningFrame
from tools.style_sheet import FrameClock


def main():
    configure_local_environment(ROOT,offline=True)
    settings=json.loads((OUT/'settings.json').read_text())
    config=AppConfig(**settings)
    prompt=config.prompt + ', curved wire lattice sheets, open ribbed arches, cable nets'
    backend=create_backend(config.backend)
    backend.load(config.backend_dict(ROOT))
    token_count=len(backend.tokenizer(prompt)['input_ids'])
    assert token_count <= backend.tokenizer.model_max_length, 'Added wording would be truncated'
    try:
        poses=json.loads((OUT/'poses.json').read_text())
        for i,pose in enumerate(poses):
            stored=np.load(OUT/f'b-{i}-conditioning.npz')
            camera=Camera(position=np.array(pose['position']),yaw=pose['yaw'],pitch=pose['pitch'])
            frame=ConditioningFrame(stored['rgb'],stored['depth'],None,camera.snapshot(1.5),0.,1)
            clock=FrameClock(10.); backend.set_clock(clock); backend.reseed(config.seed)
            backend.apply_settings(config.backend_settings())
            backend.set_prompt(prompt,config.negative_prompt)
            clock.now=config.prompt_walk_seconds+1.; backend.prompt_walk.advance(); clock.now=0.
            for _ in range(20):
                output=backend.generate(frame); clock.tick()
            Image.fromarray(output).save(OUT/f'b-{i}-wire-ai.png')
            print('Wire wording',i,flush=True)
    finally: backend.unload()
    (OUT/'wire-wording.json').write_text(json.dumps({'prompt':prompt,'original_prompt':config.prompt,
          'variant':'b','frames':20,'token_count':token_count,'settings':asdict(config)},indent=2))


if __name__=='__main__':main()
