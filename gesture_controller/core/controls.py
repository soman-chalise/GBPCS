"""Control registry -- the fixed list of things a gesture can be bound to.

Neither recognizer (built-in or custom) knows this file exists. A gesture only
ever has a NAME; `BindingsStore` maps that name to one of the control names
below, and `ActionMapper` resolves the control to an effect (a keystroke, or
the special continuous `laser_pointer` mode). This is the list the web UI's
per-gesture dropdown offers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Union

KeySpec = Union[str, List[str], None]

KIND_NONE = "none"
KIND_KEY = "key"
KIND_LASER = "laser"


@dataclass(frozen=True)
class Control:
    name: str
    label: str
    kind: str
    key: KeySpec = None


CONTROLS: Dict[str, Control] = {
    c.name: c
    for c in [
        Control("none", "(no action -- recognize only)", KIND_NONE),
        Control("next_slide", "Next slide", KIND_KEY, "right"),
        Control("previous_slide", "Previous slide", KIND_KEY, "left"),
        Control("start_presentation", "Start presentation (F5)", KIND_KEY, "f5"),
        Control("end_presentation", "End presentation (Esc)", KIND_KEY, "esc"),
        Control("blank_screen", "Blank / unblank screen", KIND_KEY, "b"),
        Control("first_slide", "First slide (Home)", KIND_KEY, "home"),
        Control("laser_pointer", "Laser pointer (hold to drag cursor)", KIND_LASER),
    ]
}

DEFAULT_CONTROL = "none"


def control_names() -> List[str]:
    return list(CONTROLS.keys())


def is_valid(name: str) -> bool:
    return name in CONTROLS


def get(name: str) -> Control:
    return CONTROLS.get(name, CONTROLS[DEFAULT_CONTROL])


def key_for(name: str) -> KeySpec:
    return get(name).key


def kind_for(name: str) -> str:
    return get(name).kind


def as_list() -> List[dict]:
    """JSON-friendly list, in registry order, for the web UI dropdown."""
    return [{"name": c.name, "label": c.label, "kind": c.kind} for c in CONTROLS.values()]
