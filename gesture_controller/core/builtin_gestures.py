"""Built-in, rule-based gesture detectors -- no recorded samples required.

These are deliberately NOT part of the TemplateStore / nearest-centroid system
`trajectory_recognizer.py` and `pose_recognizer.py` own. That system exists for
gestures a user records themselves; the gestures here ship pre-tuned and must
work "out of the box" the moment the app starts, at the cost of being less
adaptable to an individual's exact hand shape.

Two families, mirroring the custom-gesture architecture (PRD 4):

  * SWIPE (trajectory-family) -- classifies a closed `MotionSegmenter.Segment`
    by net displacement direction, straightness and axis dominance. Reuses the
    segmenter's output, not its own tracking.

  * POSE (pose-family) -- per-frame geometric checks on the wrist-relative
    landmark vector (`hand.shape`, reshaped to (21, 3)) and the curl ratios
    already computed by `HandTracker`. Gated through a small hold-then-fire
    state machine, the same shape as `PoseRecognizer`'s, so a pose fires once
    on a stable transition rather than every frame it is held.

All thresholds live in `thresholds.yaml`'s `[builtin]` section.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from .config import Config
from .hand_tracker import FINGERS, THUMB

# Landmark indices, for the direction checks the plain curl-ratio magnitude
# from HandTracker.curls can't express (which way a finger points, not just
# how extended it is).
THUMB_TIP = THUMB[2]
INDEX_TIP = FINGERS[0][2]

SWIPE_LEFT = "swipe_left"
SWIPE_RIGHT = "swipe_right"
THUMBS_UP = "thumbs_up"
GUN_POINT = "gun_point"
PEACE_SIGN = "peace_sign"
OPEN_PALM_HOLD = "open_palm_hold"
CLOSED_FIST_HOLD = "closed_fist_hold"

POSE_NAMES = (THUMBS_UP, GUN_POINT, PEACE_SIGN, OPEN_PALM_HOLD, CLOSED_FIST_HOLD)


@dataclass
class BuiltinPoseEvent:
    name: str
    previous: Optional[str]


class _HoldDebouncer:
    """Fire only on a stable-state transition. Same shape as PoseRecognizer's,
    kept local and tiny since it is only shared across the built-in pose
    detectors below -- see module docstring."""

    def __init__(self, hold_frames: int):
        self.hold_frames = hold_frames
        self._candidate: Optional[str] = None
        self._run = 0
        self._stable: Optional[str] = None

    @property
    def stable(self) -> Optional[str]:
        return self._stable

    def reset(self) -> None:
        self._candidate, self._run, self._stable = None, 0, None

    def freeze(self) -> None:
        """Drop the in-progress candidate run without touching the stable pose."""
        self._candidate, self._run = None, 0

    def update(self, name: Optional[str]) -> Optional[BuiltinPoseEvent]:
        if name == self._candidate:
            self._run += 1
        else:
            self._candidate, self._run = name, 1
        if self._run < self.hold_frames or name == self._stable:
            return None
        previous, self._stable = self._stable, name
        if name is None:
            return None
        return BuiltinPoseEvent(name=name, previous=previous)


class BuiltinGestureDetector:
    def __init__(self, cfg: Config):
        b = cfg.builtin
        self.finger_extended = float(b["finger_extended"])
        self.finger_curled = float(b["finger_curled"])
        self.thumb_extended = float(b["thumb_extended"])
        self.thumb_curled = float(b["thumb_curled"])
        self.thumb_up_y_margin = float(b["thumb_up_y_margin"])
        self.gun_point_z_margin = float(b["gun_point_z_margin"])
        self.suppress_above_speed = float(b["suppress_above_speed"])
        self.swipe_min_path_length = float(b["swipe_min_path_length"])
        self.swipe_straightness_min = float(b["swipe_straightness_min"])
        self.swipe_axis_dominance = float(b["swipe_axis_dominance"])
        self._debounce = _HoldDebouncer(int(b["hold_frames"]))

    @property
    def stable_pose(self) -> Optional[str]:
        return self._debounce.stable

    def reset_state(self) -> None:
        self._debounce.reset()

    # -- swipe family --------------------------------------------------------
    def classify_swipe(self, segment) -> Optional[str]:
        if segment is None or segment.rejected or segment.frames < 2:
            return None
        pts = segment.normalized_points()
        path_len = segment.path_length()
        if path_len < self.swipe_min_path_length:
            return None
        net = pts[-1] - pts[0]
        straightness = float(np.linalg.norm(net) / path_len) if path_len > 1e-9 else 0.0
        if straightness < self.swipe_straightness_min:
            return None
        dx, dy = float(net[0]), float(net[1])
        if abs(dx) < self.swipe_axis_dominance * abs(dy):
            return None
        return SWIPE_RIGHT if dx > 0 else SWIPE_LEFT

    # -- pose family ----------------------------------------------------------
    def _finger_curl(self, rel: np.ndarray, mcp: int, tip: int) -> float:
        return float(np.linalg.norm(rel[tip, :2] - rel[mcp, :2]))

    def _classify_pose_frame(self, hand) -> Optional[str]:
        rel = np.asarray(hand.shape, dtype=np.float64).reshape(-1, 3)

        curls: Dict[str, float] = {
            name: self._finger_curl(rel, mcp, tip) for name, mcp, tip in FINGERS
        }
        thumb_curl = self._finger_curl(rel, THUMB[1], THUMB[2])

        extended = {n: v > self.finger_extended for n, v in curls.items()}
        curled = {n: v < self.finger_curled for n, v in curls.items()}
        thumb_extended = thumb_curl > self.thumb_extended
        thumb_curled = thumb_curl < self.thumb_curled

        others = ("index", "middle", "ring", "pinky")
        matches: List[str] = []

        # thumbs_up: thumb extended and pointing up, everything else curled.
        if (
            thumb_extended
            and all(curled[n] for n in others)
            and rel[THUMB_TIP, 1] < -self.thumb_up_y_margin
        ):
            matches.append(THUMBS_UP)

        # gun_point: index + thumb extended, index tip closer to camera than
        # the wrist (the "pointing at camera" part), the rest curled.
        if (
            extended["index"]
            and thumb_extended
            and curled["middle"] and curled["ring"] and curled["pinky"]
            and rel[INDEX_TIP, 2] < -self.gun_point_z_margin
        ):
            matches.append(GUN_POINT)

        # peace_sign: index + middle extended, ring + pinky curled.
        if extended["index"] and extended["middle"] and curled["ring"] and curled["pinky"]:
            matches.append(PEACE_SIGN)

        # open_palm_hold: everything extended, including the thumb.
        if all(extended[n] for n in others) and thumb_extended:
            matches.append(OPEN_PALM_HOLD)

        # closed_fist_hold: everything curled, including the thumb.
        if all(curled[n] for n in others) and thumb_curled:
            matches.append(CLOSED_FIST_HOLD)

        # A frame that matches more than one rule is ambiguous (mid-transition
        # between two shapes) -- reject rather than guess, same philosophy as
        # the margin-over-runner-up check in the custom recognizers.
        if len(matches) != 1:
            return None
        return matches[0]

    def update_pose(self, hand, wrist_speed: float) -> Optional[BuiltinPoseEvent]:
        """One frame in, an event only on a stable-pose transition."""
        if hand is None or not hand.present:
            self._debounce.reset()
            return None
        if wrist_speed > self.suppress_above_speed:
            self._debounce.freeze()
            return None
        name = self._classify_pose_frame(hand)
        return self._debounce.update(name)
