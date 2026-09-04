from __future__ import annotations

from app.diffusion.base import DiffusionBackend

BACKEND_NAMES = ("latent_walk", "proxy_passthrough")


def create_backend(name: str) -> DiffusionBackend:
    """Construct a backend by config name without importing the others."""
    if name == "latent_walk":
        from app.diffusion.latent_walk import LatentWalkBackend

        return LatentWalkBackend()
    if name == "proxy_passthrough":
        from app.diffusion.proxy_passthrough import ProxyPassthroughBackend

        return ProxyPassthroughBackend()
    raise RuntimeError(f"Unknown backend: {name}. Available: {', '.join(BACKEND_NAMES)}")
