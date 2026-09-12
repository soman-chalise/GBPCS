"""Laser pointer -- a continuous control, not a discrete keystroke.

Unlike every other control, this one is only meaningful while its bound
gesture is the currently-held stable pose (by default `gun_point`): the OS
cursor tracks the index fingertip for as long as the pose is held, and simply
stops moving (no snap-back) the instant it releases. Because it is continuous,
it bypasses `ActionMapper.trigger()`'s edge-fire/cooldown model entirely --
`app.py`'s main loop calls `update()` every frame regardless of cooldown
state.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import pyautogui

from .config import Config
from .hand_tracker import FINGERS

INDEX_TIP = FINGERS[0][2]


class LaserPointerController:
    def __init__(self, cfg: Config):
        lp = cfg.laser
        self.alpha = float(lp["smoothing_alpha"])
        self.screen_w, self.screen_h = pyautogui.size()
        self._ema: Optional[np.ndarray] = None
        self._active = False

    @property
    def active(self) -> bool:
        return self._active

    def reset(self) -> None:
        self._ema = None
        self._active = False

    def update(self, hand, active: bool, frame_w: int, frame_h: int) -> Optional[Tuple[int, int]]:
        """Move the cursor for one frame. Returns the screen point moved to,
        or None if nothing was moved this frame."""
        if not active or hand is None or not hand.present:
            self._ema = None
            self._active = False
            return None

        tip = np.asarray(hand.pixels[INDEX_TIP], dtype=np.float64)
        self._ema = tip if self._ema is None else (
            self.alpha * tip + (1.0 - self.alpha) * self._ema
        )
        self._active = True

        nx = min(max(self._ema[0] / max(frame_w, 1), 0.0), 1.0)
        ny = min(max(self._ema[1] / max(frame_h, 1), 0.0), 1.0)
        x = int(nx * (self.screen_w - 1))
        y = int(ny * (self.screen_h - 1))
        try:
            pyautogui.moveTo(x, y)
        except Exception as exc:                      # noqa: BLE001
            print("[laser] cursor move failed: {}".format(exc))
        return x, y
