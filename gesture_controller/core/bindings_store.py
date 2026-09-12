"""Persisted gesture -> control bindings.

Replaces the old static `config/gestures.yaml`. Bindings are now runtime,
user-editable state (set from the web UI), not a file you hand-edit, so they
live in `data/bindings.json` next to `data/gestures.json` and follow the same
atomic-write persistence pattern as `TemplateStore`.

Any gesture name -- a built-in detector's or a custom recorded one's -- can be
bound to any control name from `core/controls.py`. Nothing here is locked;
the defaults below are just a starting point applied the first time a known
gesture is seen with no existing binding.
"""

from __future__ import annotations

import json
import os
import time
from typing import Dict, List

from . import controls

STORE_VERSION = 1

# Built-in gesture names, in the order the web UI should list them.
BUILTIN_GESTURES: List[str] = [
    "swipe_left",
    "swipe_right",
    "thumbs_up",
    "gun_point",
    "peace_sign",
    "open_palm_hold",
    "closed_fist_hold",
]

# Applied once, the first time each built-in gesture appears with no saved
# binding yet. Every one of these remains freely reassignable afterward.
DEFAULT_BINDINGS: Dict[str, str] = {
    "swipe_right": "next_slide",
    "swipe_left": "previous_slide",
    "thumbs_up": "start_presentation",
    "gun_point": "laser_pointer",
    "open_palm_hold": "blank_screen",
    "closed_fist_hold": "first_slide",
    "peace_sign": "end_presentation",
}


class BindingsStore:
    def __init__(self, path: str):
        self.path = path
        self.bindings: Dict[str, str] = {}

    # -- persistence ---------------------------------------------------------
    def load(self) -> "BindingsStore":
        if os.path.exists(self.path):
            with open(self.path, "r", encoding="utf-8") as fh:
                doc = json.load(fh)
            self.bindings = dict(doc.get("bindings") or {})

        changed = False
        for name in BUILTIN_GESTURES:
            if name not in self.bindings:
                self.bindings[name] = DEFAULT_BINDINGS.get(name, controls.DEFAULT_CONTROL)
                changed = True
        if changed:
            self.save()
        return self

    def save(self) -> None:
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        doc = {
            "version": STORE_VERSION,
            "saved_at": time.time(),
            "bindings": dict(sorted(self.bindings.items())),
        }
        tmp = self.path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(doc, fh, indent=2)
        os.replace(tmp, self.path)

    # -- access ----------------------------------------------------------
    def get(self, gesture: str) -> str:
        return self.bindings.get(gesture, controls.DEFAULT_CONTROL)

    def set(self, gesture: str, control: str) -> None:
        if not controls.is_valid(control):
            raise ValueError("unknown control '{}'".format(control))
        self.bindings[gesture] = control
        self.save()

    def ensure(self, gesture: str, default: str = controls.DEFAULT_CONTROL) -> None:
        """Register a gesture (e.g. a freshly recorded custom one) if unseen."""
        if gesture not in self.bindings:
            self.bindings[gesture] = default
            self.save()

    def all(self) -> Dict[str, str]:
        return dict(self.bindings)

    def gesture_for_control(self, control: str) -> List[str]:
        """Every gesture currently bound to `control` (used by the laser pointer)."""
        return [g for g, c in self.bindings.items() if c == control]
