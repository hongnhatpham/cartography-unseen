"""Check the display-side input contract without opening SDL or an OpenGL window."""
from collections import deque
import threading

import pygame
import pytest

from app.window_loop import WindowLoop


@pytest.fixture
def buffered_input():
    # Construct only the mailbox: no native window, event pump, or GPU thread.
    loop = WindowLoop.__new__(WindowLoop)
    loop._lock = threading.Lock()
    loop._events = deque()
    loop._window_events = {}
    loop._motion_event = None
    loop._quit_event = None
    loop._motion = (0, 0)
    loop._keys = (False, True)
    loop._buttons = (True, False, False)
    loop.error = None
    return loop


def test_motion_is_consumed_once_but_held_input_survives_owner_pause(buffered_input):
    loop = buffered_input
    loop._motion = (11, -7)
    first = loop.read_input()
    assert first == ((11, -7), (False, True), (True, False, False))
    # Rendering can read stale held state repeatedly while Windows blocks input.
    for _ in range(20):
        assert loop.read_input() == ((0, 0), first[1], first[2])
    loop._motion = (-3, 2)
    loop._keys = (False, False)
    loop._buttons = (False, False, False)
    assert loop.read_input() == ((-3, 2), (False, False), (False, False, False))
    assert first == ((11, -7), (False, True), (True, False, False))


def test_buffered_events_preserve_control_order_and_quit_under_event_pressure(buffered_input):
    loop = buffered_input
    controls = [pygame.event.Event(pygame.KEYDOWN, key=pygame.K_F1),
                pygame.event.Event(pygame.KEYDOWN, key=pygame.K_c),
                pygame.event.Event(pygame.KEYUP, key=pygame.K_c)]
    for index in range(1000):
        loop._record_events([pygame.event.Event(pygame.WINDOWMOVED, x=index, y=0)])
    motion = pygame.event.Event(pygame.MOUSEMOTION, rel=(5, 1), pos=(0, 0), buttons=(0, 0, 0))
    quit_event = pygame.event.Event(pygame.QUIT)
    loop._record_events(controls + [motion, quit_event])
    events = loop.poll_events()
    assert len(events) == 6
    assert events[:3] == controls
    assert events[3].type == pygame.WINDOWMOVED
    assert events[3].x == 999
    assert events[-2:] == [motion, quit_event]
    assert loop.poll_events() == []


def test_control_event_overflow_is_reported_instead_of_losing_key_edges(buffered_input):
    loop = buffered_input
    press = pygame.event.Event(pygame.KEYDOWN, key=pygame.K_w)
    loop._record_events([press] * 512)
    with pytest.raises(RuntimeError, match="overflow"):
        loop._record_events([pygame.event.Event(pygame.KEYUP, key=pygame.K_w)])
    assert len(loop._events) == 512


def test_owner_failure_reaches_display_without_waiting_for_another_input_event(buffered_input):
    buffered_input.error = ValueError("SDL event pump failed")
    with pytest.raises(RuntimeError, match="Window input loop failed") as caught:
        buffered_input.poll_events()
    assert caught.value.__cause__ is buffered_input.error
