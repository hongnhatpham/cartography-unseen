from concurrent.futures import Future, TimeoutError
from types import SimpleNamespace
import threading
import time

import pytest

from app import window_loop


class FakeFunction:
    def __init__(self, function):
        self.function = function

    def __call__(self, *args):
        return self.function(*args)


@pytest.fixture
def native(monkeypatch):
    """Exercise the real owner-thread lifecycle with inert SDL calls."""
    calls = []
    fail_input = threading.Event()

    def wait(_timeout):
        if fail_input.is_set():
            raise RuntimeError("native input failed")
        return SimpleNamespace(type=0)

    fake = SimpleNamespace(
        __file__=__file__, NOEVENT=0, MOUSEMOTION=1, QUIT=2,
        WINDOWSHOWN=100, WINDOWDISPLAYCHANGED=120, VIDEOEXPOSE=130, VIDEORESIZE=131,
        display=SimpleNamespace(_set_autoresize=lambda value: None,
                                get_window_size=lambda: (640, 480)),
        key=SimpleNamespace(get_pressed=lambda: (False,) * 10),
        mouse=SimpleNamespace(get_rel=lambda: (0, 0), get_pressed=lambda: (False,) * 3),
        event=SimpleNamespace(wait=wait, get=lambda **kwargs: []),
        quit=lambda: calls.append(("quit", threading.get_ident())),
    )
    sdl = SimpleNamespace(
        SDL_GL_MakeCurrent=FakeFunction(lambda window, context: calls.append(
            ("context", threading.get_ident(), context)) or 0),
        SDL_GetError=FakeFunction(lambda: b"fake error"),
        SDL_SetWindowFullscreen=FakeFunction(lambda *_: 0),
        SDL_GetWindowFlags=FakeFunction(lambda *_: 0),
        SDL_GL_GetCurrentWindow=FakeFunction(lambda: 123),
        SDL_GL_GetCurrentContext=FakeFunction(lambda: 456),
    )
    monkeypatch.setattr(window_loop, "pygame", fake)
    monkeypatch.setattr(window_loop.ctypes, "CDLL", lambda _: sdl)
    return SimpleNamespace(calls=calls, fail_input=fail_input, pygame=fake)


def eventually(predicate):
    deadline = time.monotonic() + 1
    while not predicate():
        assert time.monotonic() < deadline, "owner thread did not reach expected state"
        time.sleep(.001)


def test_partial_window_creation_failure_cleans_up_on_owner(native):
    owners = []

    def create():
        owners.append(threading.current_thread())
        raise RuntimeError("window creation failed after SDL init")

    with pytest.raises(RuntimeError, match="window creation failed"):
        window_loop.WindowLoop(create)

    assert native.calls == [("quit", owners[0].ident)]
    assert not owners[0].is_alive()
    assert window_loop.WindowLoop.active is None


def test_readiness_timeout_stops_late_window_creation(native, monkeypatch):
    class FastReadiness(Future):
        def result(self, timeout=None):
            return super().result(.01 if timeout == 30 else timeout)

    monkeypatch.setattr(window_loop, "Future", FastReadiness)
    owners = []

    def create():
        owners.append(threading.current_thread())
        time.sleep(.03)

    with pytest.raises(TimeoutError):
        window_loop.WindowLoop(create)

    assert native.calls == [("quit", owners[0].ident)]
    assert not owners[0].is_alive()
    assert window_loop.WindowLoop.active is None


def test_input_failure_waits_for_render_context_release(native):
    host = window_loop.WindowLoop(lambda: None)
    try:
        native.fail_input.set()
        eventually(lambda: host.error is not None)
        assert not any(call[0] == "quit" for call in native.calls)
        assert host._thread.is_alive()
        with pytest.raises(RuntimeError, match="input loop failed"):
            host.call(lambda: None)
    finally:
        host.close()

    assert native.calls[-2:] == [
        ("context", threading.get_ident(), None), ("quit", host._thread.ident),
    ]
    assert not host._thread.is_alive()
    assert window_loop.WindowLoop.active is None


def test_shutdown_rejects_queued_and_future_commands(native):
    entered, release = threading.Event(), threading.Event()

    def wait(_timeout):
        entered.set()
        assert release.wait(1)
        return SimpleNamespace(type=0)

    native.pygame.event.wait = wait
    host = window_loop.WindowLoop(lambda: None)
    assert entered.wait(1)
    errors, executed = [], []

    def caller():
        try:
            host.call(lambda: executed.append(True))
        except RuntimeError as exc:
            errors.append(str(exc))

    caller_thread = threading.Thread(target=caller)
    caller_thread.start()
    eventually(lambda: not host._commands.empty())
    timer = threading.Timer(.02, release.set)
    timer.start()
    try:
        host.close()
    finally:
        release.set()
        timer.join()
        caller_thread.join(1)

    assert not executed
    assert errors == ["Window input loop stopped"]
    assert not caller_thread.is_alive()
    with pytest.raises(RuntimeError, match="input loop stopped"):
        host.call(lambda: None)
