"""Motion segmenter (PRD 4.2) -- trajectory gestures only.

Drives a WAITING -> IN_MOTION -> closed state machine from ONE signal:

    speed = || wrist_xy[t] - wrist_xy[t-1] || / hand_scale

That is 2D wrist displacement in hand-widths per frame. It never touches the
63-dim shape vector. In attempt 1 the speed signal came from the mixed
shape+trajectory vector, so finger jitter while the hand was stationary read as
motion and opened bogus segments.

The pose recognizer does NOT consume this module. Poses fire on their own
per-frame state transitions (PRD 4.4).
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional

import numpy as np

from .config import Config


class MotionState(Enum):
    WAITING = "WAITING"
    IN_MOTION = "IN_MOTION"


@dataclass
class Segment:
    """A closed motion segment: the 2D wrist path and why it ended."""

    points: np.ndarray                 # (T, 2) wrist path, aspect-corrected image units
    scales: np.ndarray                 # (T,) hand_scale per frame
    start_time: float
    end_time: float
    close_reason: str
    peak_speed: float = 0.0
    mean_speed: float = 0.0
    rejected: Optional[str] = None     # set when the segment failed a minimum gate

    @property
    def duration(self) -> float:
        return self.end_time - self.start_time

    @property
    def frames(self) -> int:
        return len(self.points)

    def normalized_points(self) -> np.ndarray:
        """Path in hand-widths, using the segment's mean hand scale.

        Dividing by a single scale (rather than per-frame) keeps the path shape
        intact; per-frame division would warp the trajectory whenever the hand
        moves toward or away from the camera mid-swipe.
        """
        s = float(np.mean(self.scales))
        return self.points / max(s, 1e-6)

    def path_length(self) -> float:
        p = self.normalized_points()
        if len(p) < 2:
            return 0.0
        return float(np.sum(np.linalg.norm(np.diff(p, axis=0), axis=1)))


@dataclass
class SegmenterStatus:
    """Per-frame view of the machine, for the HUD and the frame log."""

    state: MotionState
    speed: float = 0.0
    buffered_frames: int = 0
    segment: Optional[Segment] = None   # non-None only on the frame a segment closes


class MotionSegmenter:
    def __init__(self, cfg: Config):
        s = cfg.segmenter
        self._idle_speed = float(s["idle_speed_threshold"])
        self._start_frames = int(s["motion_start_frames"])
        self._end_still_frames = int(s["motion_end_still_frames"])
        self._min_frames = int(s["min_segment_frames"])
        self._max_frames = int(s["max_segment_frames"])
        self._preroll = int(s["preroll_frames"])
        self._min_path = float(s["min_segment_path_length"])
        self._lost_grace = int(s["hand_lost_grace_frames"])

        self._state = MotionState.WAITING
        self._preroll_buf: deque = deque(maxlen=max(self._preroll, 1))
        self._points: List[np.ndarray] = []
        self._scales: List[float] = []
        self._speeds: List[float] = []
        self._moving_run = 0
        self._still_run = 0
        self._lost_run = 0
        self._prev_wrist: Optional[np.ndarray] = None
        self._start_time = 0.0
        self._last_time = 0.0
        self._last_speed = 0.0

    # -- introspection -----------------------------------------------------
    @property
    def state(self) -> MotionState:
        return self._state

    @property
    def last_speed(self) -> float:
        """Most recent frame speed.

        Also read by the pose recognizer's `suppress_above_speed` gate. Note
        this hands over a plain scalar, not segmenter state -- the pose path
        stays independent of the WAITING/IN_MOTION machine.
        """
        return self._last_speed

    def reset(self) -> None:
        self._state = MotionState.WAITING
        self._preroll_buf.clear()
        self._points.clear()
        self._scales.clear()
        self._speeds.clear()
        self._moving_run = self._still_run = self._lost_run = 0
        self._prev_wrist = None
        self._last_speed = 0.0

    # -- main entry point --------------------------------------------------
    def update(self, wrist_xy, hand_scale: float, timestamp: float) -> SegmenterStatus:
        if wrist_xy is None:
            return self._on_hand_lost(timestamp)

        self._lost_run = 0
        self._last_time = timestamp

        if self._prev_wrist is None:
            speed = 0.0
        else:
            speed = float(
                np.linalg.norm(wrist_xy - self._prev_wrist) / max(hand_scale, 1e-6)
            )
        self._prev_wrist = np.asarray(wrist_xy, dtype=np.float64).copy()
        self._last_speed = speed

        moving = speed > self._idle_speed
        self._moving_run = self._moving_run + 1 if moving else 0
        self._still_run = 0 if moving else self._still_run + 1

        if self._state is MotionState.WAITING:
            self._preroll_buf.append((wrist_xy.copy(), hand_scale, timestamp, speed))
            if self._moving_run >= self._start_frames:
                self._open_segment()
            return SegmenterStatus(self._state, speed, len(self._points))

        # IN_MOTION
        self._points.append(np.asarray(wrist_xy, dtype=np.float64).copy())
        self._scales.append(hand_scale)
        self._speeds.append(speed)

        if self._still_run >= self._end_still_frames:
            return SegmenterStatus(
                MotionState.WAITING, speed, 0, self._close_segment("still", timestamp)
            )
        if len(self._points) >= self._max_frames:
            return SegmenterStatus(
                MotionState.WAITING, speed, 0,
                self._close_segment("max_length", timestamp),
            )
        return SegmenterStatus(self._state, speed, len(self._points))

    # -- internals ---------------------------------------------------------
    def _open_segment(self) -> None:
        self._state = MotionState.IN_MOTION
        self._points, self._scales, self._speeds = [], [], []
        for pt, sc, _ts, sp in self._preroll_buf:
            self._points.append(np.asarray(pt, dtype=np.float64).copy())
            self._scales.append(sc)
            self._speeds.append(sp)
        self._start_time = (
            self._preroll_buf[0][2] if self._preroll_buf else self._last_time
        )
        self._preroll_buf.clear()

    def _on_hand_lost(self, timestamp: float) -> SegmenterStatus:
        """Tolerate a short dropout, then CLOSE rather than discard.

        Discarding on hand loss is precisely how attempt 1 lost fast swipes.
        A partial fast swipe still carries its direction, so we hand it to the
        recognizer and let the distance threshold and margin check decide.
        """
        self._lost_run += 1
        self._last_speed = 0.0
        if self._state is MotionState.IN_MOTION and self._lost_run > self._lost_grace:
            return SegmenterStatus(
                MotionState.WAITING, 0.0, 0,
                self._close_segment("hand_lost", timestamp),
            )
        if self._state is MotionState.WAITING and self._lost_run > self._lost_grace:
            self._preroll_buf.clear()
            self._prev_wrist = None
        return SegmenterStatus(self._state, 0.0, len(self._points))

    def _close_segment(self, reason: str, timestamp: float) -> Segment:
        seg = Segment(
            points=np.array(self._points, dtype=np.float64).reshape(-1, 2),
            scales=(
                np.array(self._scales, dtype=np.float64)
                if self._scales
                else np.array([1.0])
            ),
            start_time=self._start_time,
            end_time=timestamp,
            close_reason=reason,
            peak_speed=float(max(self._speeds)) if self._speeds else 0.0,
            mean_speed=float(np.mean(self._speeds)) if self._speeds else 0.0,
        )
        if seg.frames < self._min_frames:
            seg.rejected = "too_short({}f<{}f)".format(seg.frames, self._min_frames)
        else:
            plen = seg.path_length()
            if plen < self._min_path:
                seg.rejected = "path_too_small({:.2f}<{})".format(plen, self._min_path)

        self._state = MotionState.WAITING
        self._points, self._scales, self._speeds = [], [], []
        self._moving_run = self._still_run = 0
        self._preroll_buf.clear()
        return seg
