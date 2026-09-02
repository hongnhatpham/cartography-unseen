from __future__ import annotations

from app.diffusion.base import DiffusionBackend


def create_backend(name: str) -> DiffusionBackend:
    if name == "sd_turbo_stream":
        from app.diffusion.sd_turbo_stream import SDTurboStreamBackend

        return SDTurboStreamBackend()
    if name == "proxy_passthrough":
        from app.diffusion.proxy_passthrough import ProxyPassthroughBackend

        return ProxyPassthroughBackend()
    raise RuntimeError(f"Unknown backend: {name}")
