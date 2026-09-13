"""Commissioning-only input driver. Normal startup never imports this module."""
from pathlib import Path
import json
import logging
import os
import sys
import time
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
BASE = Path(os.environ.get('CARTOGRAPHY_SOAK_DIRECTORY', ROOT.parent / ('Commissioning/soak-' + time.strftime('%Y%m%d'))))
sys.path.insert(0, str(ROOT))
from tools.soak_io import atomic, read_json
from tools.mock_visitors import build_plan, action_at


def run():
    from app import main
    main.configure_local_environment(ROOT, offline=True)
    import pygame
    from app.renderer.proxy_renderer import ProxyRenderer
    from app.diffusion.worker import DiffusionWorker
    from app.journey import JourneyRecorder
    from app.map_view import MapWindow

    output = Path(read_json(BASE / 'active.json')['directory'])
    # A supervisor restart must not silently restart or overwrite a failed test.
    if (output / 'app.json').exists():
        sys.argv = ['app.main', '--config', 'cache/exhibition/config.json']
        return main.run()
    data = dict(pid=os.getpid(), launched=time.time(), first_ai=None, ai_frames=0,
                display_frames=0, input_frames=0, saves=[], transitions=[],
                warnings=[], errors=[], map_alive=False, visitor_sessions=[], action_counts={})
    plan = build_plan(int(time.time()))
    atomic(output / 'visitor-plan.json', plan, retry=True)
    data['visitor_plan'] = dict(seed=plan['seed'], sessions=len(plan['sessions']),
                                saves=len(plan['saves']), end=plan['end'])
    last_input_time = time.monotonic()
    previous_action = None
    renderer = map_window = None
    last_write = 0
    enabled = True
    fired = set()
    original_log = main.configure_logging
    original_init = ProxyRenderer.__init__
    original_worker = DiffusionWorker.__init__
    original_map = MapWindow.__init__
    original_present = ProxyRenderer._present_texture
    original_input = ProxyRenderer.read_input
    original_poll = ProxyRenderer.poll_events
    original_reset = JourneyRecorder.reset

    class Errors(logging.Handler):
        def emit(self, record):
            if record.levelno >= logging.WARNING:
                field = 'errors' if record.levelno >= logging.ERROR else 'warnings'
                data[field].append(dict(time=time.time(), message=record.getMessage()[:1000]))

    def configure_logging(*args, **kwargs):
        path = original_log(*args, **kwargs)
        logging.getLogger().addHandler(Errors())
        data['log'] = str(path)
        return path

    def init(self, *args, **kwargs):
        nonlocal renderer
        original_init(self, *args, **kwargs)
        renderer = self

    def worker(self, *args, **kwargs):
        original_worker(self, *args, **kwargs)
        publish = self.generated.publish
        def generated(frame):
            if data['first_ai'] is None:
                data['first_ai'] = time.time()
            data['ai_frames'] += 1
            return publish(frame)
        self.generated.publish = generated

    def map_init(self, *args, **kwargs):
        nonlocal map_window
        original_map(self, *args, **kwargs)
        map_window = self

    def present(self, *args, **kwargs):
        value = original_present(self, *args, **kwargs)
        data['display_frames'] += 1
        return value

    def reset(self, *args, **kwargs):
        archive_id = self.id
        had_images = len(self._data['images'])
        result = original_reset(self, *args, **kwargs)
        data['saves'].append(dict(id=archive_id, images=had_images, time=time.time(),
                                 mock_visitor=data.get('pending_save')))
        return result

    def elapsed():
        return time.time() - data['first_ai'] if data['first_ai'] else -1

    def read_input(self):
        nonlocal last_input_time, previous_action
        mouse, keys, buttons = original_input(self)
        seconds = elapsed()
        now = time.monotonic()
        dt = min(now - last_input_time, 0.1)
        last_input_time = now
        action = action_at(plan, seconds) if enabled else None
        if action:
            visitor = action['visitor']
            if not data['visitor_sessions'] or data['visitor_sessions'][-1]['id'] != visitor:
                session = dict(plan['sessions'][visitor - 1], started=time.time())
                data['visitor_sessions'].append(session)
            if previous_action != action['start']:
                name = action['action']
                data['action_counts'][name] = data['action_counts'].get(name, 0) + 1
                previous_action = action['start']
            data['current_visitor'] = visitor
            data['current_action'] = action['action']
            pressed = {getattr(pygame, 'K_' + key) for key in action['keys']}
            original_keys = keys
            class InjectedKeys:
                def __getitem__(self, key):
                    return True if key in pressed else original_keys[key]
                def __iter__(self):
                    return iter([True])
            keys = InjectedKeys()
            mouse = (mouse[0] + action['yaw'] * dt, mouse[1])
            data['input_frames'] += 1
        else:
            data['current_visitor'] = None
            data['current_action'] = None
        if enabled:
            for session in data['visitor_sessions']:
                if seconds >= session['end'] and 'completed' not in session:
                    session['completed'] = time.time()
            data['visitor_workload_complete'] = seconds >= plan['end']
        return mouse, keys, buttons

    def poll():
        nonlocal last_write, enabled
        events = original_poll()
        now = time.time()
        if now - last_write >= 1:
            status_path = output / 'status.json'
            if status_path.exists():
                try:
                    status = read_json(status_path)['status']
                    enabled = status in ('warming_up', 'running')
                except (OSError, ValueError) as error:
                    enabled = False
                    data['errors'].append(dict(time=now, message='Test controller status unavailable: ' + repr(error)))
            data.update(updated=now, fullscreen=renderer.is_fullscreen,
                        map_alive=bool(map_window and map_window._process.is_alive()),
                        injection_enabled=enabled)
            data['game_window'] = renderer._window_call(renderer._placement.snapshot)
            data['game_window']['operator'] = renderer._operator_mode
            data['game_window']['gl_renderer'] = renderer.ctx.info.get('GL_RENDERER')
            data['map_window'] = map_window.window_status if map_window else None
            atomic(output / 'app.json', data)
            last_write = now
        seconds = elapsed()
        if enabled:
            # Configured displays start fullscreen; visitors never use operator keys.
            schedule = []
            schedule += [(save['at'], pygame.K_SPACE, save['reason']) for save in plan['saves']]
            for moment, key, reason in schedule:
                if seconds >= moment and moment not in fired:
                    if key == pygame.K_SPACE:
                        data['pending_save'] = next(save for save in plan['saves'] if save['at'] == moment)
                    events.append(pygame.event.Event(pygame.KEYDOWN, key=key, mod=0))
                    events.append(pygame.event.Event(pygame.KEYUP, key=key, mod=0))
                    fired.add(moment)
                    data['transitions'].append(dict(time=now, key=key, reason=reason))
                    break
        return events

    sys.argv = ['app.main', '--config', str(output / 'config.json')]
    try:
        with patch.object(main, 'configure_logging', configure_logging), \
             patch.object(ProxyRenderer, '__init__', init), \
             patch.object(DiffusionWorker, '__init__', worker), \
             patch.object(MapWindow, '__init__', map_init), \
             patch.object(ProxyRenderer, '_present_texture', present), \
             patch.object(ProxyRenderer, 'read_input', read_input), \
             patch.object(ProxyRenderer, 'poll_events', staticmethod(poll)), \
             patch.object(JourneyRecorder, 'reset', reset):
            return main.run()
    except BaseException as error:
        data['errors'].append(dict(time=time.time(), message=repr(error)))
        raise
    finally:
        data.update(exited=time.time(), updated=time.time())
        atomic(output / 'app.json', data)


if __name__ == '__main__':
    raise SystemExit(run())
