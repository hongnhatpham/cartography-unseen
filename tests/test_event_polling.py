"""The event pump must yield to the AI worker and preserve queued controls."""
import pygame
import pytest

from app.renderer.proxy_renderer import ProxyRenderer


@pytest.mark.parametrize("has_first", [True, False])
def test_bounded_wait_preserves_event_order_without_pumping_again(monkeypatch, has_first):
    first = pygame.event.Event(pygame.KEYDOWN, key=pygame.K_RETURN)
    remaining = [pygame.event.Event(pygame.KEYUP, key=pygame.K_RETURN),
                 pygame.event.Event(pygame.QUIT)]
    waited = []
    def wait(timeout):
        waited.append(timeout)
        return first if has_first else pygame.event.Event(pygame.NOEVENT)
    def get(*, pump=True):
        assert not pump, "Draining the queue must not re-enter the GIL-holding pump"
        return list(remaining)
    monkeypatch.setattr(pygame.event, "wait", wait)
    monkeypatch.setattr(pygame.event, "get", get)
    assert ProxyRenderer.poll_events() == ([first, *remaining] if has_first else remaining)
    assert waited == [1]
