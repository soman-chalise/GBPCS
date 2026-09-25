"""Camera frame source -- device index or network stream.

The interesting part is `threaded`, and it is a genuine tradeoff, not a free
win -- measure before assuming it helps.

THE THEORY: a USB camera's `read()` blocks on hardware, so it can never run
ahead of us -- but that block (a driver-level USB transfer, milliseconds to
tens of milliseconds depending on wire format) is supposed to be CPU-idle
time. Pumping it on its own thread would let that wait overlap with MediaPipe
inference on the main thread instead of the two being paid for back to back
every frame. A network stream (a phone camera app) has a second, independent
reason to want this: it pushes frames into a queue as fast as it likes, and
every frame we fail to consume becomes permanent latency. `CAP_PROP_BUFFERSIZE`
is documented to bound that queue but is a no-op on the FFMPEG backend used
for http:// sources, so draining on a thread and keeping only the newest frame
is the only reliable fix there.

THE CATCH, measured on real hardware: that CPU-idle assumption only holds if
the driver actually grants hardware MJPG compression (see `configure()`'s
warning below). When it doesn't -- common on cheap/built-in webcams, not
expected on a proper deployment camera -- the "blocking" read is doing real
CPU work decoding the raw wire format, not idling, so the pump thread
directly competes with MediaPipe inference on the main thread for CPU
instead of overlapping it. On one affected dev laptop this measured >2x
WORSE effective fps and wildly spikier per-frame latency with threading on
vs off.

`camera.threaded_stream` in thresholds.yaml defaults to `true` -- correct for
a network stream (the queue-growth problem is real there and threading is
the only fix) and for the target deployment rig. If you hit the MJPG warning
on a local dev machine, flip it to `false` there rather than changing the
shared default -- see SETUP.md section 9.
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
        self.threaded = threaded

        if self.is_url:
            self.cap = cv2.VideoCapture(source)
        else:
            # DSHOW is the traditional default backend on Windows, but on at
            # least some UVC webcams/drivers it negotiates a markedly lower
            # real capture rate than MSMF at the same resolution/fourcc
            # request (measured ~14fps vs ~23fps on the same device here) --
            # try MSMF first and fall back if it can't open the device.
            backends = (
                [cv2.CAP_MSMF, cv2.CAP_DSHOW] if sys.platform == "win32" else [0]
            )
            self.cap = None
            for backend in backends:
                cap = cv2.VideoCapture(source, backend)
                if cap.isOpened():
                    self.cap = cap
                    break
                cap.release()
            if self.cap is None:
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
        # Force MJPG on the wire before negotiating resolution/fps. Most UVC
        # webcams fall back to raw/YUY2 by default, which saturates USB
        # bandwidth at 720p and silently throttles the driver down to a
        # handful of fps -- CPU stays idle because the process is blocked on
        # the slow transfer, not computing. MJPG moves the compression onto
        # the camera itself and is what actually unlocks the requested fps.
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, fps)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        # `.set()` returns success/failure but a LOT of drivers report success
        # while silently ignoring the request, so the only trustworthy check
        # is reading the negotiated format back. If this isn't MJPG, the wire
        # format is raw/YUY2 -- exactly the "saturates USB, driver throttles
        # to a handful of fps" failure mode described above, and it will look
        # like a CPU/recognition problem when it is actually a capture-layer
        # one. Surface it loudly instead of failing silently.
        got_fourcc = int(self.cap.get(cv2.CAP_PROP_FOURCC))
        got_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        got_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        got_fps = self.cap.get(cv2.CAP_PROP_FPS)
        tag = "".join(chr((got_fourcc >> (8 * i)) & 0xFF) for i in range(4)).strip()
        print("[camera] negotiated {}x{} @ {:.0f}fps, fourcc={!r}".format(
            got_w, got_h, got_fps, tag))
        if tag.upper() != "MJPG":
            print("[camera] WARNING: driver did not grant MJPG (got {!r}). The "
                  "wire format is likely uncompressed and may saturate USB "
                  "bandwidth at this resolution, throttling fps well below "
                  "what CPU load would predict. Try a lower camera.width/height "
                  "in thresholds.yaml, a different USB port, or check for a "
                  "driver-specific MJPG toggle.".format(tag))

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
