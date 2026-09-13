"""Exercise F/F11/F1 with live AI and both windows; isolate settings/archives."""
from pathlib import Path
import ctypes
import json
import sys
from time import perf_counter, strftime
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

def run():
    from app import main
    main.configure_local_environment(ROOT, offline=True)
    import pygame
    from app.renderer.proxy_renderer import ProxyRenderer
    from app.diffusion.worker import DiffusionWorker
    from app.journey import JourneyRecorder
    from app.map_view import MapWindow
    from tools.display_topology import snapshot as display_topology
    output = ROOT / 'logs' / ('fullscreen-check-' + strftime('%Y%m%d_%H%M%S'))
    output.mkdir()
    config = json.loads((ROOT / 'cache/exhibition/config.json').read_text())
    config.update(map_sync_enabled=False, autowalk_idle_seconds=2, debug_overlay=True,
                  fullscreen=False, display_monitor=0, map_display_monitor=1)
    config_path = output / 'config.json'
    config_path.write_text(json.dumps(config), encoding='utf-8')
    report = {'ai_frames': 0, 'display_frames': 0, 'transitions': [], 'focus_calls': 0,
              'native_pointer_checks': 0, 'errors': [], 'duration_after_first_ai': 0}
    report['initial_topology'] = display_topology()
    assert report['initial_topology']['independent_sources'] >= 2, 'Windows must extend both displays before testing'
    started = None
    launched = perf_counter()
    action = 0
    renderer = None
    map_window = None
    # Ten complete cycles, then stay fullscreen with the overlay closed.
    keys = [pygame.K_f, pygame.K_F1, pygame.K_F1, pygame.K_f,
            pygame.K_F11, pygame.K_F1, pygame.K_F1, pygame.K_F11] * 10
    keys += [pygame.K_f, pygame.K_F1]
    original_init = ProxyRenderer.__init__
    original_poll = ProxyRenderer.poll_events
    original_present = ProxyRenderer._present_texture
    original_focus = ProxyRenderer.focus
    original_fullscreen = ProxyRenderer.set_fullscreen
    original_worker = DiffusionWorker.__init__
    original_recorder = JourneyRecorder.__init__
    original_map = MapWindow.__init__
    def init(self, *a, **kw):
        nonlocal renderer
        original_init(self, *a, **kw)
        renderer = self
        report['gl_renderer'] = self.ctx.info.get('GL_RENDERER')
        self._test_getdata = self._window_loop._api('SDL_GetWindowData', ctypes.c_void_p,
                                                   ctypes.c_void_p, ctypes.c_char_p)
    def verify():
        loop = renderer._window_loop
        address = loop.call(lambda: renderer._test_getdata(loop.window, b'pg_window'))
        assert address == id(renderer._placement.window), 'SDL points at a released window wrapper'
        report['native_pointer_checks'] += 1
    def focus(self):
        original_focus(self)
        verify()
        report['focus_calls'] += 1
    def fullscreen(self, enabled):
        result = original_fullscreen(self, enabled)
        verify()
        report['transitions'].append({'enabled': result, 'size': list(self.window_size),
                                     'actual': self._window_call(self._placement.snapshot)})
        topology = display_topology()
        report['transitions'][-1]['topology'] = topology
        assert topology['independent_sources'] >= 2, 'Fullscreen disabled a Windows display'
        return result
    def worker(self, *a, **kw):
        original_worker(self, *a, **kw)
        publish = self.generated.publish
        def generated(frame):
            nonlocal started
            if started is None:
                started = perf_counter()
                print('First AI frame; starting 180-second fullscreen test', flush=True)
            report['ai_frames'] += 1
            return publish(frame)
        self.generated.publish = generated
    def recorder(self, root, *a, **kw):
        return original_recorder(self, output / 'journeys', *a, **kw)
    def map_init(self, *a, **kw):
        nonlocal map_window
        original_map(self, *a, **kw)
        map_window = self
    def present(self, *a, **kw):
        result = original_present(self, *a, **kw)
        report['display_frames'] += 1
        return result
    def poll():
        nonlocal action
        events = original_poll()
        for event in events:
            window = getattr(event, 'window', None)
            assert window is None or window is renderer._placement.window
        if map_window is not None and not map_window._process.is_alive():
            raise RuntimeError('Journey map process exited during fullscreen test')
        now = perf_counter()
        if started is not None:
            elapsed = now - started
            if action < len(keys) and elapsed >= action * 1.5:
                events.append(pygame.event.Event(pygame.KEYDOWN, key=keys[action], mod=0))
                action += 1
            if elapsed >= 180:
                report['duration_after_first_ai'] = elapsed
                report['final_fullscreen'] = renderer.is_fullscreen
                report['map_alive_at_end'] = map_window._process.is_alive()
                report['game_window'] = renderer._window_call(renderer._placement.snapshot)
                report['map_window'] = map_window.window_status
                report['final_topology'] = display_topology()
                assert report['final_topology']['independent_sources'] >= 2
                for name, display in (('game_window', 0), ('map_window', 1)):
                    state = report[name]
                    assert state and state['display'] == display, (name, state)
                    assert state['fullscreen'] and not state['minimized'] and state['shown'], (name, state)
                    assert state['position'] == state['bounds'][:2] and state['size'] == state['bounds'][2:], (name, state)
                events.append(pygame.event.Event(pygame.QUIT))
        elif now - launched > 180:
            raise RuntimeError('Timed out waiting for AI')
        return events
    sys.argv = ['app.main', '--config', str(config_path)]
    try:
        with patch.object(ProxyRenderer, '__init__', init), \
             patch.object(ProxyRenderer, 'poll_events', staticmethod(poll)), \
             patch.object(ProxyRenderer, 'focus', focus), \
             patch.object(ProxyRenderer, 'set_fullscreen', fullscreen), \
             patch.object(ProxyRenderer, '_present_texture', present), \
             patch.object(DiffusionWorker, '__init__', worker), \
             patch.object(JourneyRecorder, '__init__', recorder), \
             patch.object(MapWindow, '__init__', map_init):
            result = main.run()
        assert result == 0, result
        assert action == len(keys), (action, len(keys))
        assert report['duration_after_first_ai'] >= 180
        assert report['ai_frames'] > 100
        assert report['final_fullscreen'] and report['map_alive_at_end']
        report['passed'] = True
    except BaseException as exc:
        report['errors'].append(repr(exc))
        report['passed'] = False
        raise
    finally:
        (output / 'result.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
        print(json.dumps({**report, 'transitions': len(report['transitions']), 'output': str(output)}), flush=True)

if __name__ == '__main__':
    run()
