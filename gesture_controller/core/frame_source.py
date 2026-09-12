"""Camera frame source -- device index or network stream.

The interesting part is `threaded`. A USB camera's `read()` blocks on hardware,
so it can never run ahead of us. A network stream (a phone camera app) is the
opposite: it pushes frames into a queue as fast as it likes, and every frame we
fail to consume becomes permanent latency. `CAP_PROP_BUFFERSIZE` is documented
to bound that queue but is a no-op on the FFMPEG backend used for http://
sources, so it cannot be relied on.

The fix is to drain continuously on a background thread and keep only the most
recent frame. The main loop then always gets the newest image, and surplus
frames are dropped instead of queued. This is also what makes a processing FPS
cap safe: without draining, consuming slower than the stream produces just
grows the backlog.
"""

from __future__ import annotations

import sys
import threading
import time
from typing import Optional, Tuple

import cv2
import numpy as np


class FrameSource:
    def __init__(self, source, threaded: bool = False):
        self.source = source
        self.is_url = isinstance(source, str)
        self.threaded = threaded and self.is_url

        if self.is_url:
            self.cap = cv2.VideoCapture(source)
        else:
            backend = cv2.CAP_DSHOW if sys.platform == "win32" else 0
            self.cap = cv2.VideoCapture(source, backend)
            if not self.cap.isOpened():
                self.cap = cv2.VideoCapture(source)

        self._lock = threading.Lock()
        self._latest: Optional[np.ndarray] = None
        self._seq = 0
        self._last_seq = -1
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

        # How many frames the grabber threw away because we were still busy.
        # A steady nonzero value means the stream outruns us -- which is fine,
        # and exactly what keeps latency flat.
        self.dropped = 0

    # -- lifecycle ---------------------------------------------------------
    def is_opened(self) -> bool:
        return bool(self.cap and self.cap.isOpened())

    def configure(self, width: int, height: int, fps: int) -> None:
        if self.is_url:
            # Resolution belongs to the phone app; asking here does nothing.
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            return
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, fps)

    def start(self) -> "FrameSource":
        if self.threaded and self._thread is None:
            self._thread = threading.Thread(target=self._pump, daemon=True)
            self._thread.start()
            # Wait briefly for the first frame so the caller isn't handed None.
            deadline = time.time() + 5.0
            while time.time() < deadline:
                with self._lock:
                    if self._latest is not None:
                        break
                time.sleep(0.01)
        return self

    def release(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.5)
            self._thread = None
        if self.cap:
            self.cap.release()

    # -- reading -----------------------------------------------------------
    def _pump(self) -> None:
        """Drain the stream forever, keeping only the newest frame."""
        while not self._stop.is_set():
            ok, frame = self.cap.read()
            if not ok or frame is None:
                time.sleep(0.005)
                continue
            with self._lock:
                if self._latest is not None and self._seq != self._last_seq:
                    self.dropped += 1     # previous frame was never consumed
                self._latest = frame
                self._seq += 1

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        if not self.threaded:
            return self.cap.read()
        with self._lock:
            if self._latest is None:
                return False, None
            self._last_seq = self._seq
            return True, self._latest.copy()

    def wait_for_new_frame(self, timeout: float = 1.0) -> bool:
        """Block until the grabber has produced a frame we have not seen.

        Keeps the loop from spinning on the same image when we are faster than
        the stream. Returns False on timeout so the caller stays responsive.
        """
        if not self.threaded:
            return True
        deadline = time.time() + timeout
        while time.time() < deadline and not self._stop.is_set():
            with self._lock:
                if self._seq != self._last_seq:
                    return True
            time.sleep(0.002)
        return False
