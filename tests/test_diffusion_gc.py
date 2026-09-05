"""Long-lived model objects stay out of exhibition-time full GC scans."""
import gc
from app.diffusion.latent_walk import LatentWalkBackend


def test_warmup_freezes_startup_graph_and_unload_restores_gc(monkeypatch):
    events = []
    frozen = [0]
    monkeypatch.setattr(gc, 'get_freeze_count', lambda: frozen[0])
    monkeypatch.setattr(gc, 'collect', lambda: events.append('collect'))

    def freeze():
        frozen[0] = 100
        events.append('freeze')

    def unfreeze():
        frozen[0] = 0
        events.append('unfreeze')

    monkeypatch.setattr(gc, 'freeze', freeze)
    monkeypatch.setattr(gc, 'unfreeze', unfreeze)
    backend = LatentWalkBackend()
    backend.warmup_passes = 0
    backend.warmup()
    backend.warmup()
    assert events == ['collect', 'freeze']
    assert gc.isenabled()
    backend.unload()
    backend.unload()
    assert events == ['collect', 'freeze', 'unfreeze']


def test_preexisting_frozen_objects_remain_owned_by_caller(monkeypatch):
    events = []
    monkeypatch.setattr(gc, 'get_freeze_count', lambda: 100)
    for name in ('collect', 'freeze', 'unfreeze'):
        monkeypatch.setattr(gc, name, lambda name=name: events.append(name))
    backend = LatentWalkBackend()
    backend.warmup_passes = 0
    backend.warmup()
    backend.unload()
    assert events == []
