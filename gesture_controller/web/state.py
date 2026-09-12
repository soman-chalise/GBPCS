"""Thread-safe hand-off between the camera/recognition loop and Flask.

The loop (running in the main thread, `app.py`) is the ONLY writer of status
and video frames, and the ONLY consumer of commands. The Flask server (its
own background thread) only ever publishes commands and reads snapshots. That
keeps every actual state mutation -- toggling LIVE, rebinding a gesture,
starting a recording -- on the loop thread, one frame later, so nothing in
`App` needs its own extra locking.
"""

from __future__ import annotations

import queue
import threading
from typing import Any, Dict, List, Optional


class AppState:
    def __init__(self):
        self._lock = threading.Lock()
        self._status: Dict[str, Any] = {}
        self._frame_jpeg: Optional[bytes] = None
        self._commands: "queue.Queue[dict]" = queue.Queue()

    # -- written by the loop, read by the web layer -------------------------
    def publish_status(self, status: Dict[str, Any]) -> None:
        with self._lock:
            self._status = status

    def get_status(self) -> Dict[str, Any]:
        with self._lock:
            return dict(self._status)

    def publish_frame(self, jpeg_bytes: bytes) -> None:
        with self._lock:
            self._frame_jpeg = jpeg_bytes

    def get_frame(self) -> Optional[bytes]:
        with self._lock:
            return self._frame_jpeg

    # -- written by the web layer, read/drained by the loop ------------------
    def send_command(self, **cmd) -> None:
        self._commands.put(cmd)

    def drain_commands(self) -> List[dict]:
        out: List[dict] = []
        while True:
            try:
                out.append(self._commands.get_nowait())
            except queue.Empty:
                break
        return out
