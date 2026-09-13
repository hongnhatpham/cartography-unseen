"""Independent five-second commissioning recorder; never repairs or hides failures."""
from pathlib import Path
from collections import deque
from datetime import datetime, timezone
import json
import os
import socket
import sqlite3
import statistics
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
BASE = Path(os.environ.get('CARTOGRAPHY_SOAK_DIRECTORY', ROOT.parent / ('Commissioning/soak-' + time.strftime('%Y%m%d'))))
sys.path.insert(0, str(ROOT))
from app.monitoring import SystemSampler, heartbeat
from tools.soak_io import atomic, read_json, append_json, exists
from tools.display_topology import snapshot as display_topology
import psutil


def utc(value):
    return datetime.fromtimestamp(value, timezone.utc).isoformat()


def run():
    output = Path(read_json(BASE / 'active.json')['directory'])
    state = dict(status='warming_up', watcher_pid=os.getpid(), launched=time.time(),
                 required_clean_seconds=7200, samples=0, errors=[], started=None,
                 thresholds=dict(min_free_gib=5, max_ram_percent=90, max_gpu_temperature_c=85,
                                 max_frame_age_seconds=5, min_60s_display_fps=24,
                                 min_60s_generation_fps=5, max_upload_delay_seconds=240,
                                 max_telemetry_queue_age_seconds=120, max_rss_growth_gib=1.5))
    atomic(output / 'status.json', state, retry=True)
    sampler = SystemSampler(ROOT)
    recent = deque(maxlen=12)
    baseline_rss = None
    started_monotonic = None
    previous_tick = None
    last_services = 0
    services = {}
    while state['status'] in ('warming_up', 'running'):
        tick = time.monotonic()
        now = time.time()
        errors = []
        if previous_tick is not None and tick - previous_tick > 15:
            errors.append('Independent sampling stopped for more than fifteen seconds')
        previous_tick = tick
        try:
            app_path = output / 'app.json'
            if not exists(app_path):
                if now - state['launched'] > 240:
                    raise RuntimeError('App did not publish commissioning state within four minutes')
                time.sleep(5)
                continue
            app = read_json(app_path)
            sample = heartbeat('commissioning-read-only', state['samples'], ROOT / 'cache/monitoring', sampler, now=now)
            supervisor = read_json(ROOT / 'cache/monitoring/supervisor.json')
            upload = read_json(ROOT / 'cache/monitoring/uploader.json')
            process = psutil.Process(app['pid'])
            processes = [process] + process.children(recursive=True)
            rss = 0
            for child in processes:
                try:
                    rss += child.memory_info().rss
                except psutil.NoSuchProcess:
                    pass
            with sqlite3.connect('file:' + (ROOT / 'cache/exhibition-monitor/outbox.sqlite3').as_posix() + '?mode=ro', uri=True, timeout=3) as db:
                queued, oldest = db.execute('select count(*),min(sampled) from outbox').fetchone()
            if now - last_services >= 60:
                services = {name: psutil.win_service_get(name).status() for name in ('sshd', 'Tailscale')}
                services['monitor_pids'] = []
                for candidate in psutil.process_iter(['pid', 'name', 'cmdline']):
                    if candidate.info['name'] in ('python.exe', 'pythonw.exe') and any(
                            argument.endswith('monitor_agent.py') for argument in (candidate.info['cmdline'] or [])):
                        services['monitor_pids'].append(candidate.pid)
                with socket.create_connection(('127.0.0.1', 22), timeout=3) as connection:
                    services['ssh_banner'] = connection.recv(128).decode().strip()
                last_services = now
            save_checks = []
            for saved in app['saves']:
                directory = ROOT / 'journeys' / saved['id']
                receipt = ROOT / 'journeys/.sync' / ('bundle-' + saved['id'] + '.json')
                complete = exists(directory / 'complete.json')
                verified = exists(receipt)
                save_checks.append(dict(id=saved['id'], complete=complete, verified=verified, images=saved['images']))
                if now - saved['time'] > 240 and (not complete or not verified or saved['images'] == 0):
                    errors.append('Journey not saved with images and verified within four minutes: ' + saved['id'])
            observation = dict(time=now, app=sample['app'], upload=sample['archiveSync'],
                               system=sample['system'], rss_bytes=rss, services=services,
                               queue_count=queued, queue_age=now-oldest if oldest else 0,
                               saves=save_checks, app_pid=app['pid'], supervisor_restarts=supervisor['restarts'])
            observation['visitors'] = dict(current=app.get('current_visitor'),
                                          action=app.get('current_action'),
                                          completed=sum('completed' in s for s in app.get('visitor_sessions', [])),
                                          actions=app.get('action_counts', {}))
            observation['windows'] = dict(game=app.get('game_window'), map=app.get('map_window'))
            observation['display_topology'] = display_topology()
            state['samples'] += 1
            append_json(output / 'samples.jsonl', observation)
            first = app['first_ai']
            if app['errors']:
                errors.append('Application error: ' + app['errors'][-1]['message'])
            if app.get('exited') or now-app.get('updated', 0) > 15:
                errors.append('Application exited or stopped responding')
            if upload.get('error_code') or upload.get('error_archives', 0):
                errors.append('Uploader reported an error')
            if first and now-first >= 60 and not psutil.pid_exists(upload['pid']):
                errors.append('Uploader process stopped')
            if first and now-first >= 60:
                if state['started'] is None:
                    state.update(status='running', started=now, start_utc=utc(now),
                                 expected_end_utc=utc(now+7200), baseline_restarts=supervisor['restarts'])
                    baseline_rss = rss
                    started_monotonic = time.monotonic()
                system, artwork = sample['system'], sample['app']
                if observation['display_topology']['independent_sources'] < 2:
                    errors.append('Windows disabled a display or stopped extending the desktop')
                if artwork['state'] != 'running' or not app['map_alive']:
                    errors.append('Artwork or map is not running')
                for name, display in (('game', 0), ('map', 1)):
                    window = observation['windows'][name]
                    if window and 'NVIDIA' not in window.get('gl_renderer', ''):
                        errors.append(f'{name} did not use the configured NVIDIA rendering GPU')
                    if not window or window['display'] != display or not window['fullscreen'] or window.get('presentation') != 'borderless' or window['minimized'] or not window['shown'] or window.get('operator', True) or window['position'] != window['bounds'][:2] or window['size'] != window['bounds'][2:]:
                        errors.append(f'{name} is not visibly fullscreen on its assigned display {display}')
                    elif name == 'map' and (now-window['updated'] > 10 or (window['active'] and now-window['last_draw'] > 5)):
                        errors.append('Map window rendering heartbeat stopped')
                if supervisor['childPid'] != app['pid'] or supervisor['restarts'] != state['baseline_restarts']:
                    errors.append('Supervisor restarted or replaced the app')
                supervisor_updated = datetime.fromisoformat(supervisor['updatedAt'].replace('Z', '+00:00')).timestamp()
                if supervisor['state'] != 'running' or now-supervisor_updated > 15:
                    errors.append('Supervisor heartbeat stopped')
                if len(services['monitor_pids']) != 1 or not psutil.pid_exists(services['monitor_pids'][0]):
                    errors.append('Monitoring collector missing or duplicated')
                if now-upload.get('updated_at', 0) > 180:
                    errors.append('Uploader progress heartbeat stopped')
                if artwork['statusAgeSeconds'] is None or artwork['statusAgeSeconds'] > 15:
                    errors.append('Artwork telemetry stale')
                if artwork['lastFrameAgeSeconds'] is None or artwork['lastFrameAgeSeconds'] > 5:
                    errors.append('No fresh AI frame within five seconds')
                recent.append((artwork['displayFps'] or 0, artwork['generationFps'] or 0))
                if len(recent) == 12:
                    if statistics.mean(x[0] for x in recent) < 24:
                        errors.append('Display performance below 24 FPS for one minute')
                    if statistics.mean(x[1] for x in recent) < 5:
                        errors.append('AI generation below 5 FPS for one minute')
                if system['diskFreeBytes'] is None or system['diskFreeBytes'] < 5*1024**3:
                    errors.append('Disk free space below five GiB')
                if psutil.virtual_memory().percent > 90:
                    errors.append('System RAM exceeds 90 percent')
                if system['gpuTemperatureC'] is None or system['gpuTemperatureC'] > 85:
                    errors.append('GPU temperature unavailable or above 85 C')
                if rss > baseline_rss + 1.5*1024**3:
                    errors.append('Application and child RSS grew more than 1.5 GiB')
                if oldest and now-oldest > 120:
                    errors.append('Monitoring delivery backlog older than two minutes')
                if not all(services[name] == 'running' for name in ('sshd', 'Tailscale')) or not services['ssh_banner'].startswith('SSH-2.0'):
                    errors.append('Remote maintenance service failed')
                state['clean_seconds'] = time.monotonic()-started_monotonic
                if state['clean_seconds'] >= 7200:
                    # The last visitor saves at the end of the two-hour input
                    # workload. Keep sampling during the normal upload window.
                    coverage = app.get('visitor_plan', {})
                    workload_done = app.get('visitor_workload_complete', False)
                    complete_visitors = observation['visitors']['completed'] == coverage.get('sessions')
                    all_saved = len(save_checks) == coverage.get('saves') and len(save_checks) >= 20
                    all_verified = all(s['verified'] and s['complete'] and s['images'] > 0 for s in save_checks)
                    if now-first > coverage.get('end', 7260) + 240 and not (workload_done and complete_visitors and all_saved and all_verified):
                        errors.append('Final mock visitor, save, or upload coverage incomplete')
                    if not app['fullscreen']:
                        errors.append('Artwork did not return to fullscreen')
                    if not errors and workload_done and complete_visitors and all_saved and all_verified:
                        state.update(status='awaiting_final_review', completed_utc=utc(now))
            elif now-state['launched'] > 300:
                errors.append('AI warmup did not complete within five minutes')
            if errors:
                state.update(status='failed', errors=errors, failed_utc=utc(now))
            state.update(updated=now, latest=observation, warnings=app['warnings'])
        except Exception as error:
            state.update(status='failed', errors=[repr(error)], traceback=traceback.format_exc(),
                         updated=now, failed_utc=utc(now))
        atomic(output / 'status.json', state, retry=True)
        time.sleep(max(0, 5-(time.monotonic()-tick)))


if __name__ == '__main__':
    run()
