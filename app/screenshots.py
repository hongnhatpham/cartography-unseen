"""Save an AI-only display capture without compressing PNGs on the render thread."""
from concurrent.futures import Future, ThreadPoolExecutor
from datetime import datetime
import logging
from pathlib import Path

from PIL import Image


class ScreenshotWriter:
    def __init__(self, folder: Path) -> None:
        self.folder = folder
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="screenshot")
        self._pending: Future[Path] | None = None

    @property
    def busy(self) -> bool:
        return self._pending is not None and not self._pending.done()

    def submit(self, rgb: bytes, size: tuple[int, int]) -> None:
        # Keep just one image in flight; a held Enter key cannot grow a queue.
        if not self.busy:
            self._pending = self._executor.submit(self._save, rgb, size)

    def _save(self, rgb: bytes, size: tuple[int, int]) -> Path:
        self.folder.mkdir(parents=True, exist_ok=True)
        path = self.folder / f"cartography-unseen-{datetime.now():%Y%m%d-%H%M%S-%f}.png"
        image = Image.frombytes("RGB", size, rgb).transpose(Image.Transpose.FLIP_TOP_BOTTOM)
        with path.open("xb") as output:
            image.save(output, format="PNG")
        logging.info("Screenshot saved: %s", path)
        return path

    def poll(self) -> str | None:
        if self._pending is None or not self._pending.done():
            return None
        pending, self._pending = self._pending, None
        try:
            return f"SCREENSHOT SAVED: screenshot/{pending.result().name}"
        except Exception as exc:
            logging.exception("Screenshot save failed")
            return f"SCREENSHOT FAILED: {exc}"

    def close(self) -> None:
        self._executor.shutdown(wait=True)
        self.poll()
