"""Action mapper + shared cooldown (PRD 4.5).

The cooldown lives HERE, not in either recognizer, and it is a single shared
clock. That is deliberate: a swipe and a fist-close performed in the same
motion must not both advance the slide.

No recognizer -- built-in or custom -- knows a keystroke exists. Each emits a
gesture NAME; this module resolves that name through `BindingsStore` to a
control, then `controls.py` to an effect. Rebinding a gesture in the web UI
never touches recognizer code.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import List, Optional

import pyautogui

from . import controls as controls_mod
from .bindings_store import BindingsStore
from .config import Config

# A stray gesture near a screen corner should not raise a fail-safe exception
# mid-presentation.
pyautogui.FAILSAFE = False
pyautogui.PAUSE = 0.0


@dataclass
class ActionEvent:
    """One accepted gesture and what the mapper did about it."""

    gesture: str
    source: str                 # "trajectory", "pose", or "builtin"
    timestamp: float
    fired: bool                 # False when suppressed by cooldown
    control: Optional[str] = None
    detail: str = ""
    distance: float = float("nan")
    threshold: float = float("nan")


class ActionMapper:
    def __init__(self, cfg: Config, bindings: BindingsStore, dry_run: bool = False):
        self.cooldown = float(cfg.actions["cooldown_seconds"])
        self.enabled = bool(cfg.actions["keystroke_enabled"]) and not dry_run
        self.dry_run = dry_run
        self.bindings = bindings
        self._last_fire_time = 0.0
        self._last_fire_gesture: Optional[str] = None
        self.history: List[ActionEvent] = []

    # -- cooldown ----------------------------------------------------------
    def cooldown_remaining(self, now: Optional[float] = None) -> float:
        now = time.time() if now is None else now
        return max(0.0, self.cooldown - (now - self._last_fire_time))

    def in_cooldown(self, now: Optional[float] = None) -> bool:
        return self.cooldown_remaining(now) > 0.0

    def reset_cooldown(self) -> None:
        self._last_fire_time = 0.0

    # -- main entry point --------------------------------------------------
    def trigger(
        self,
        gesture: str,
        source: str,
        distance: float = float("nan"),
        threshold: float = float("nan"),
        now: Optional[float] = None,
    ) -> ActionEvent:
        now = time.time() if now is None else now
        remaining = self.cooldown_remaining(now)
        if remaining > 0.0:
            ev = ActionEvent(
                gesture, source, now, fired=False,
                detail="suppressed by shared cooldown ({:.2f}s left, last={})".format(
                    remaining, self._last_fire_gesture
                ),
                distance=distance, threshold=threshold,
            )
            self.history.append(ev)
            return ev

        control = self.bindings.get(gesture)
        kind = controls_mod.kind_for(control)
        if kind == controls_mod.KIND_NONE:
            detail = "not bound to a control"
        elif kind == controls_mod.KIND_LASER:
            # Continuous mode -- driven every frame by LaserPointerController,
            # not by this discrete trigger. Still consumes the cooldown so a
            # gun-point can't also fire something else in the same instant.
            detail = "laser pointer (continuous, not a keystroke)"
        elif not self.enabled:
            detail = "DRY RUN, would press {}".format(controls_mod.key_for(control))
        else:
            key = controls_mod.key_for(control)
            detail = "pressed {}".format(key)
            self._press(key)

        # The cooldown starts on any accepted gesture, even an unbound or
        # dry-run one -- so cooldown behaviour is identical when validating.
        self._last_fire_time = now
        self._last_fire_gesture = gesture

        ev = ActionEvent(
            gesture, source, now, fired=True, control=control, detail=detail,
            distance=distance, threshold=threshold,
        )
        self.history.append(ev)
        return ev

    def _press(self, key) -> None:
        try:
            if isinstance(key, (list, tuple)):
                pyautogui.hotkey(*key)
            else:
                pyautogui.press(str(key))
        except Exception as exc:                      # noqa: BLE001
            print("[action] keystroke failed: {}".format(exc))

    def describe_mapping(self) -> List[str]:
        mapping = self.bindings.all()
        if not mapping:
            return ["(no gestures bound yet)"]
        return [
            "{:<18} -> {}".format(n, controls_mod.get(c).label)
            for n, c in sorted(mapping.items())
        ]
