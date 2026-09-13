"""Commissioning file IO that cooperates with Windows atomic publishers."""
import errno
import ctypes
import json
import os
from pathlib import Path
import tempfile
import time


def retry_sharing(operation, *, allow_missing=False):
    for attempt in range(11):
        try:
            return operation()
        except OSError as error:
            sharing = error.errno == errno.EACCES or getattr(error, 'winerror', None) in (5, 32, 33, 1175)
            replacing = allow_missing and (error.errno == errno.ENOENT or getattr(error, 'winerror', None) in (2, 3))
            if not (sharing or replacing) or attempt == 10:
                raise
            time.sleep(.025)


def shared_reader(path):
    if os.name == 'nt':
        import msvcrt
        kernel = ctypes.WinDLL('kernel32', use_last_error=True)
        create = kernel.CreateFileW
        create.argtypes = [ctypes.c_wchar_p, ctypes.c_uint32, ctypes.c_uint32,
                           ctypes.c_void_p, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_void_p]
        create.restype = ctypes.c_void_p
        handle = create(str(path), 0x80000000, 7, None, 3, 0x80, None)
        if handle == ctypes.c_void_p(-1).value:
            error = ctypes.WinError(ctypes.get_last_error())
            error.filename = str(path)
            raise error
        descriptor = msvcrt.open_osfhandle(handle, os.O_RDONLY | os.O_BINARY)
        return os.fdopen(descriptor, 'rb')
    return Path(path).open('rb')


def read_json(path):
    def read():
        with shared_reader(path) as source:
            return json.loads(source.read())
    # Windows/.NET ReplaceFile can briefly make a changing snapshot unavailable.
    # This is only a short read retry; a persistently missing source still fails.
    return retry_sharing(read, allow_missing=True)


def exists(path):
    return retry_sharing(lambda: path.exists())


def atomic(path, value, *, retry=False):
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=path.parent,
                                         prefix=path.name + '.', suffix='.tmp', delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(value, stream)
        if retry:
            retry_sharing(lambda: os.replace(temporary, path))
        else:
            os.replace(temporary, path)
        return True
    except OSError:
        if retry:
            raise
        return False
    finally:
        if temporary is not None:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass


def append_json(path, value):
    # Only retry opening: retrying a partially written record could duplicate it.
    stream = retry_sharing(lambda: path.open('a', encoding='utf-8'))
    with stream:
        stream.write(json.dumps(value) + '\n')
