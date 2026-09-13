import errno
import json
import threading
import time
import os
import pytest
from tools import soak_io


def test_atomic_replacement_waits_for_brief_reader_lock(tmp_path):
    path = tmp_path / 'snapshot.json'
    assert soak_io.atomic(path, {'version': 1}, retry=True)
    results = []
    with soak_io.shared_reader(path) as original:
        thread = threading.Thread(target=lambda: results.append(soak_io.atomic(path, {'version': 2}, retry=True)))
        thread.start()
        time.sleep(.05)
        assert json.loads(original.read()) == {'version': 1}
    thread.join()
    assert results == [True]
    assert soak_io.read_json(path) == {'version': 2}


def test_concurrent_publications_never_expose_partial_json(tmp_path):
    path = tmp_path / 'snapshot.json'
    soak_io.atomic(path, {'version': 0, 'payload': '0' * 1000}, retry=True)
    errors = []
    def writer():
        try:
            for version in range(1, 301):
                soak_io.atomic(path, {'version': version, 'payload': str(version) * 1000}, retry=True)
        except BaseException as error:
            errors.append(error)
    thread = threading.Thread(target=writer)
    thread.start()
    reads = 0
    while thread.is_alive():
        value = soak_io.read_json(path)
        assert value['payload'] == str(value['version']) * 1000
        reads += 1
        # Model periodic observers, not an unbroken loop that denies every
        # Windows rename a chance to acquire the file during the retry budget.
        time.sleep(.005)
    thread.join()
    assert not errors
    assert reads > 0
    assert soak_io.read_json(path)['version'] == 300


def test_only_bounded_sharing_failures_are_retried(monkeypatch):
    monkeypatch.setattr(soak_io.time, 'sleep', lambda _: None)
    attempts = []
    def locked():
        attempts.append(1)
        raise PermissionError(errno.EACCES, 'sharing conflict')
    with pytest.raises(PermissionError):
        soak_io.retry_sharing(locked)
    assert len(attempts) == 11
    attempts.clear()
    def full():
        attempts.append(1)
        raise OSError(errno.ENOSPC, 'disk full')
    with pytest.raises(OSError, match='disk full'):
        soak_io.retry_sharing(full)
    assert len(attempts) == 1


@pytest.mark.skipif(os.name != 'nt', reason='Windows ReplaceFile error')
def test_replacefile_remove_conflict_is_bounded(monkeypatch):
    import ctypes
    monkeypatch.setattr(soak_io.time, 'sleep', lambda _: None)
    attempts = []
    def conflict():
        attempts.append(1)
        if len(attempts) < 3:
            raise ctypes.WinError(1175)
        return 'published'
    assert soak_io.retry_sharing(conflict) == 'published'
    assert len(attempts) == 3


@pytest.mark.skipif(os.name != 'nt', reason='PowerShell supervisor')
def test_production_supervisor_publisher_with_concurrent_reader(tmp_path):
    import subprocess
    from pathlib import Path
    source = Path(__file__).resolve().parents[1] / 'tools/run_exhibition.ps1'
    path = tmp_path / 'supervisor.json'
    path.write_text('{"state":"running","sequence":0}')
    script = tmp_path / 'publish.ps1'
    script.write_text('''param($SupervisorSource, $Destination)
$ErrorActionPreference = 'Stop'
$tree = [Management.Automation.Language.Parser]::ParseFile($SupervisorSource, [ref]$null, [ref]$null)
$fn = $tree.Find({param($node) $node -is [Management.Automation.Language.FunctionDefinitionAst] -and $node.Name -eq 'Write-AtomicSupervisorFile'}, $true)
Invoke-Expression $fn.Extent.Text
for ($index = 1; $index -le 300; $index++) {
    [IO.File]::WriteAllText("$Destination.tmp", ('{"state":"running","sequence":' + $index + '}'))
    Write-AtomicSupervisorFile -Source "$Destination.tmp" -Destination $Destination
    Start-Sleep -Milliseconds 2
}
''')
    process = subprocess.Popen(['powershell.exe', '-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', str(script), str(source), str(path)],
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                               creationflags=subprocess.CREATE_NO_WINDOW)
    while process.poll() is None:
        assert soak_io.read_json(path)['state'] == 'running'
        time.sleep(.001)
    stdout, stderr = process.communicate(timeout=10)
    assert process.returncode == 0, stderr.decode(errors='replace')
    assert soak_io.read_json(path)['sequence'] == 300


def test_app_publication_failure_does_not_interrupt_rendering(tmp_path, monkeypatch):
    monkeypatch.setattr(soak_io.os, 'replace', lambda *_: (_ for _ in ()).throw(PermissionError(errno.EACCES, 'locked')))
    assert not soak_io.atomic(tmp_path / 'snapshot.json', {'version': 1})
    assert not list(tmp_path.glob('*.tmp'))


def test_snapshot_read_survives_brief_replacement_gap(tmp_path):
    path = tmp_path / 'supervisor.json'
    def replace():
        time.sleep(.05)
        soak_io.atomic(path, {'state': 'running'}, retry=True)
    thread = threading.Thread(target=replace)
    thread.start()
    assert soak_io.read_json(path) == {'state': 'running'}
    thread.join()


def test_persistently_missing_snapshot_still_fails(tmp_path, monkeypatch):
    monkeypatch.setattr(soak_io.time, 'sleep', lambda _: None)
    with pytest.raises(FileNotFoundError):
        soak_io.read_json(tmp_path / 'missing.json')


@pytest.mark.skipif(os.name != 'nt', reason='Supervisor uses Windows ReplaceFile')
def test_supervisor_windows_replacefile_with_concurrent_reads(tmp_path):
    import ctypes
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    replace = kernel.ReplaceFileW
    replace.argtypes = [ctypes.c_wchar_p, ctypes.c_wchar_p, ctypes.c_wchar_p,
                        ctypes.c_uint32, ctypes.c_void_p, ctypes.c_void_p]
    replace.restype = ctypes.c_int
    path = tmp_path / 'supervisor.json'
    path.write_text(json.dumps({'sequence': 0, 'state': 'running'}))
    errors = []
    def writer():
        try:
            for sequence in range(1, 1001):
                incoming = tmp_path / 'supervisor.tmp'
                incoming.write_text(json.dumps({'sequence': sequence, 'state': 'running'}))
                def publish():
                    if not replace(str(path), str(incoming), None, 0, None, None):
                        raise ctypes.WinError(ctypes.get_last_error())
                soak_io.retry_sharing(publish)
                # Give readers a window between publications, as the real
                # supervisor does (every five seconds, rather than 500 Hz).
                time.sleep(.002)
        except BaseException as error:
            errors.append(error)
    thread = threading.Thread(target=writer)
    thread.start()
    count = 0
    while thread.is_alive():
        assert soak_io.read_json(path)['state'] == 'running'
        count += 1
    thread.join()
    assert not errors
    assert count > 0
    assert soak_io.read_json(path)['sequence'] == 1000
