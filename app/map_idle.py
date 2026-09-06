"""Cached exhibition title artwork and an interruptible map/title crossfade."""
from pathlib import Path


class MapFade:
    """Map visibility starts at zero; changing direction preserves current opacity."""

    def __init__(self):
        self.active = False
        self.started = 0.0
        self.duration = 0.0
        self.origin = 0.0

    def value(self, now: float) -> float:
        if self.duration == 0:
            return float(self.active)
        progress = max(0.0, min(1.0, (now - self.started) / self.duration))
        eased = progress * progress * (3 - 2 * progress)
        return self.origin + (float(self.active) - self.origin) * eased

    def set_active(self, active: bool, now: float) -> None:
        if active != self.active:
            self.origin = self.value(now)
            self.active = active
            self.started = now
            self.duration = 0.6 if active else 1.2

    def running(self, now: float) -> bool:
        return now < self.started + self.duration


def title_surface(root: Path, size: tuple[int, int]):
    """Rasterize on resize only; bundled fonts and QR work without network access."""
    import pygame

    # Bound the cached texture at 1080p, including on a 4K projector.
    scale = min(1.0, 1920 / size[0], 1080 / size[1])
    width, height = (max(1, round(value * scale)) for value in size)
    surface = pygame.Surface((width, height))
    surface.fill((0, 0, 0))
    # Fit the approved 16:9 composition into any window without cropping credits.
    unit = min(width, height * 16 / 9) / 100
    layout_height = unit * 56.25
    top = (height - layout_height) / 2
    ink, muted = (234, 231, 221), (170, 170, 170)
    fonts = root / "assets/fonts"
    title = pygame.font.Font(str(fonts / "SpaceGrotesk-SemiBold.woff"), max(1, round(7.6 * unit)))
    author = pygame.font.Font(str(fonts / "IBMPlexMono-Regular.woff"), max(1, round(1.3 * unit)))
    small = pygame.font.Font(str(fonts / "IBMPlexMono-Regular.woff"), max(1, round(1.05 * unit)))

    def line(font, text, y, color=ink, tracking=0):
        if tracking:
            text_width = round(font.size(text)[0] + tracking * (len(text) - 1))
            x = (width - text_width) / 2
            for index, character in enumerate(text):
                offset = font.size(text[:index + 1])[0] - font.size(character)[0]
                surface.blit(font.render(character, True, color), (round(x + offset + index * tracking), round(y)))
        else:
            rendered = font.render(text, True, color)
            surface.blit(rendered, (round((width - rendered.get_width()) / 2), round(y)))

    y = top + layout_height * .19
    line(title, "Cartography", y, tracking=-.045 * title.get_height())
    line(title, "Unseen", y + 7.6 * 1.02 * unit, tracking=-.045 * title.get_height())
    y += (7.6 * 1.02 * 2 + 2.3) * unit
    line(author, "Nhat (Hong) Pham · Agnieszka Kiejziewicz", y)
    line(author, "Ricardo Arce · Kok Yoong Lim", y + 1.3 * 1.7 * unit)
    line(small, "Contributor · Tom Nguyen", y + (1.3 * 1.7 * 2 + .55) * unit, muted)
    qr = pygame.image.load(str(root / "assets/emergentplay-qr.png"))
    qr_size = max(1, round(8.8 * unit))
    qr_y = top + layout_height * .955 - (8.8 + .65 + 1.05 * 1.5) * unit
    surface.blit(pygame.transform.scale(qr, (qr_size, qr_size)),
                 (round((width - qr_size) / 2), round(qr_y)))
    line(small, "emergentplay.bynhat.com", qr_y + (8.8 + .65) * unit)
    return surface
