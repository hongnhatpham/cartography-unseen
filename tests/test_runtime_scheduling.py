"""The interactive app's thread cadence must not leak into its host process."""
import pytest

from app import main


@pytest.mark.parametrize("previous", [.005, .0005])
def test_app_uses_shorter_thread_interval_then_restores_host(monkeypatch, previous):
    calls = []
    monkeypatch.setattr(main.sys, "getswitchinterval", lambda: previous)
    monkeypatch.setattr(main.sys, "setswitchinterval", calls.append)

    def app():
        assert calls == [min(previous, .001)]
        return 3  # Configuration/startup failure still restores the host.

    monkeypatch.setattr(main, "_run", app)
    assert main.run() == 3
    assert calls == [min(previous, .001), previous]


def test_app_restores_thread_interval_on_exception(monkeypatch):
    calls = []
    monkeypatch.setattr(main.sys, "getswitchinterval", lambda: .005)
    monkeypatch.setattr(main.sys, "setswitchinterval", calls.append)

    def app():
        raise RuntimeError("startup failed")

    monkeypatch.setattr(main, "_run", app)
    with pytest.raises(RuntimeError, match="startup failed"):
        main.run()
    assert calls == [.001, .005]
