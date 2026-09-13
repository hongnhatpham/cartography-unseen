"""Short real-GL dual-display preflight, without loading the AI models."""
from pathlib import Path
import json
import time
import numpy as np
import pygame
from app.renderer.proxy_renderer import ProxyRenderer
from app.map_view import MapWindow
from tools.display_topology import snapshot as topology

ROOT = Path(__file__).resolve().parents[1]

def run():
    output = ROOT / 'logs' / ('borderless-display-' + time.strftime('%Y%m%d_%H%M%S'))
    output.mkdir()
    result = dict(passed=False, samples=[], errors=[])
    game = map_window = None
    try:
        assert topology()['independent_sources'] >= 2
        game = ProxyRenderer(ROOT, (384, 256), True, display_monitor=0)
        result['renderer'] = game.ctx.info['GL_RENDERER']
        map_window = MapWindow(ROOT, display_monitor=1, fullscreen=True)
        scene = {'id':'display-preflight', 'images':[], 'segments':[], 'prompts':[]}
        start, sample_at, cycle = time.monotonic(), 0, 0
        while time.monotonic()-start < 45:
            elapsed = time.monotonic()-start
            if elapsed >= (cycle+1)*5 and cycle < 6:
                cycle += 1
                full = cycle % 2 == 0
                game.set_fullscreen(full)
                map_window.set_fullscreen(full)
            map_window.update(scene, [elapsed,0,0], [0,elapsed,0], True)
            map_window.poll()
            for event in game.poll_events():
                if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                    raise RuntimeError('Display preflight interrupted')
            frame = np.zeros((256,384,3), dtype=np.uint8)
            frame[:,:,0] = np.arange(384,dtype=np.uint16)%256
            frame[:,:,1] = np.arange(256,dtype=np.uint8)[:,None]
            frame[:,:,2] = int(elapsed*20)%256
            game.display(frame, [])
            assert game.ctx.error == 'GL_NO_ERROR'
            if elapsed >= sample_at:
                state = dict(elapsed=elapsed, topology=topology(),
                             game=game._window_call(game._placement.snapshot),
                             map=map_window.window_status)
                result['samples'].append(state)
                (output/'progress.json').write_text(json.dumps(state,indent=2))
                assert state['topology']['independent_sources'] >= 2, 'Windows disabled a display'
                assert state['game']['shown'], 'Game window hidden'
                assert map_window._process.is_alive(), 'Map exited'
                sample_at = elapsed+1
            time.sleep(1/60)
        for key, display in [('game',0),('map',1)]:
            state=result['samples'][-1][key]
            assert state and state['display']==display and state['shown'] and not state['minimized']
            assert state['presentation']=='borderless' and state['position']==state['bounds'][:2] and state['size']==state['bounds'][2:]
        result['passed']=True
    except BaseException as exc:
        result['errors'].append(repr(exc))
        raise
    finally:
        if map_window: map_window.close()
        if game: game.close()
        (output/'result.json').write_text(json.dumps(result,indent=2))
        print(json.dumps(dict(passed=result['passed'],errors=result['errors'],output=str(output))),flush=True)

if __name__=='__main__':
    run()
