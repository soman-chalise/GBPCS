"""Pose recognizer -- the fist family (PRD 4.4).

INPUT: the current frame's per-finger curl ratios. That is it. There is no
resampling, no template sequence, no DTW, no path -- and crucially no
dependency on the motion segmenter. A pose is a property of one frame.

FEATURE: 4 dims (index/middle/ring/pinky), optionally 5 with the thumb.
Each dim is  ||tip - MCP|| / hand_scale:  extended ~0.9-1.3, curled ~0.2-0.5.

CLASSIFICATION: per-gesture centroid over recorded samples, Euclidean distance
normalized by sqrt(dim), gated by a threshold plus the same
margin-over-runner-up rejection the trajectory side uses.

TRIGGERING: on a STATE CHANGE. A candidate pose must hold for `hold_frames`
consecutive frames to become the stable pose; an event fires only when the
stable pose actually changes. Holding a fist therefore fires once, not 30
times a second, and releasing it fires the open-hand pose once.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from .config import Config
from .template_store import POSE, Gesture, TemplateStore


@dataclass
class PoseResult:
    """Per-frame classification (not yet an event -- see PoseRecognizer.update)."""

    name: Optional[str]
    distance: float
    threshold: float
    accepted: bool
    runner_up_name: Optional[str] = None
    runner_up_distance: float = float("inf")
    margin_ratio: float = float("inf")
    reason: str = ""
    threshold_source: str = ""
    scores: Dict[str, float] = field(default_factory=dict)

    def describe(self) -> str:
        head = "MATCH {}".format(self.name) if self.accepted else "no-match"
        return "[pose] {}: best={} d={:.3f} thr={:.3f} {} | ru={} {:.3f} | {}".format(
            head, self.name, self.distance, self.threshold, self.threshold_source,
            self.runner_up_name, self.runner_up_distance, self.reason,
        )


@dataclass
class PoseEvent:
    """Emitted only on a stable-pose transition."""

    name: str
    previous: Optional[str]
    result: PoseResult


class PoseRecognizer:
    def __init__(self, cfg: Config, store: TemplateStore):
        p = cfg.pose
        self.include_thumb = bool(p["include_thumb"])
        self.default_threshold = float(p["default_threshold"])
        self.calibration_k = float(p["calibration_k"])
        self.threshold_floor = float(p["threshold_floor"])
        self.threshold_ceiling = float(p["threshold_ceiling"])
        self.margin_ratio = float(p["margin_ratio"])
        self.hold_frames = int(p["hold_frames"])
        self.suppress_above_speed = float(p["suppress_above_speed"])
        self.trim_ratio = float(p["record_trim_ratio"])
        self.hand_lost_grace = int(p["hand_lost_grace_frames"])
        self.store = store

        self._thresholds: Dict[str, float] = {}
        self._threshold_sources: Dict[str, str] = {}
        self._centroids: Dict[str, np.ndarray] = {}

        # transition state
        self._candidate: Optional[str] = None
        self._candidate_run = 0
        self._stable: Optional[str] = None
        self._last_result: Optional[PoseResult] = None
        self._absent_run = 0

        self.recalibrate()

    # -- introspection -----------------------------------------------------
    @property
    def stable_pose(self) -> Optional[str]:
        return self._stable

    @property
    def last_result(self) -> Optional[PoseResult]:
        return self._last_result

    def reset_state(self) -> None:
        self._candidate = None
        self._candidate_run = 0
        self._stable = None
        self._absent_run = 0

    # -- feature -----------------------------------------------------------
    def feature_from_curls(self, curls: np.ndarray) -> np.ndarray:
        return np.asarray(curls, dtype=np.float64).reshape(-1)

    def feature_from_recording(self, curl_frames: List[np.ndarray]) -> np.ndarray:
        """Average the stable middle of a pose recording.

        The first and last frames of a capture include the hand still forming
        or already releasing the pose, so trimming both ends gives a much
        cleaner centroid than averaging everything.
        """
        arr = np.array(curl_frames, dtype=np.float64)
        if len(arr) == 0:
            raise ValueError("no frames captured for pose sample")
        if len(arr) >= 5 and self.trim_ratio > 0:
            k = int(len(arr) * self.trim_ratio)
            if len(arr) - 2 * k >= 1:
                arr = arr[k: len(arr) - k]
        return arr.mean(axis=0)

    def distance(self, a: np.ndarray, b: np.ndarray) -> float:
        d = np.asarray(a) - np.asarray(b)
        return float(np.linalg.norm(d) / np.sqrt(len(d)))

    # -- thresholds --------------------------------------------------------
    def recalibrate(self) -> None:
        """Same 1-sample-default / >=2-sample-calibrated rule as the swipe side."""
        self._thresholds.clear()
        self._threshold_sources.clear()
        self._centroids.clear()
        for name, g in self.store.of_type(POSE).items():
            feats = g.matrix()
            self._centroids[name] = feats.mean(axis=0)
            if not g.is_calibrated:
                self._thresholds[name] = self.default_threshold
                self._threshold_sources[name] = "default(1 sample)"
                continue
            dists = self.intra_class_distances(g)
            if not dists:
                self._thresholds[name] = self.default_threshold
                self._threshold_sources[name] = "default(no pairs)"
                continue
            arr = np.array(dists)
            thr = float(arr.mean() + self.calibration_k * arr.std())
            thr = min(max(thr, self.threshold_floor), self.threshold_ceiling)
            self._thresholds[name] = thr
            self._threshold_sources[name] = "calibrated({} samples)".format(g.count)

    def intra_class_distances(self, g: Gesture) -> List[float]:
        feats = g.matrix()
        return [
            self.distance(feats[i], feats[j])
            for i in range(len(feats))
            for j in range(i + 1, len(feats))
        ]

    def threshold_for(self, name: str) -> float:
        return self._thresholds.get(name, self.default_threshold)

    def threshold_source(self, name: str) -> str:
        return self._threshold_sources.get(name, "default(unknown)")

    def centroid_for(self, name: str) -> Optional[np.ndarray]:
        return self._centroids.get(name)

    # -- classification ----------------------------------------------------
    def classify(self, curls: np.ndarray) -> PoseResult:
        if not self._centroids:
            return PoseResult(None, float("inf"), 0.0, False,
                              reason="no pose gestures recorded")
        feature = self.feature_from_curls(curls)
        per_gesture: Dict[str, float] = {}
        for name, c in self._centroids.items():
            per_gesture[name] = (
                self.distance(feature, c) if len(c) == len(feature) else float("inf")
            )

        ranked = sorted(per_gesture.items(), key=lambda kv: kv[1])
        best_name, best_dist = ranked[0]
        runner_name, runner_dist = (ranked[1] if len(ranked) > 1 else (None, float("inf")))
        ratio = best_dist / runner_dist if runner_dist > 0 else float("inf")

        thr = self.threshold_for(best_name)
        res = PoseResult(
            name=best_name,
            distance=best_dist,
            threshold=thr,
            accepted=False,
            runner_up_name=runner_name,
            runner_up_distance=runner_dist,
            margin_ratio=ratio,
            threshold_source=self.threshold_source(best_name),
            scores=per_gesture,
        )
        if best_dist > thr:
            res.reason = "distance over threshold"
            return res
        if runner_name is not None and ratio > self.margin_ratio:
            res.reason = "ambiguous vs {} (ratio {:.2f} > {:.2f})".format(
                runner_name, ratio, self.margin_ratio
            )
            return res
        res.accepted = True
        res.reason = "ok"
        return res

    # -- per-frame update / transition detection ---------------------------
    def update(self, curls, wrist_speed: float) -> Optional[PoseEvent]:
        """Run one frame. Returns an event only on a stable-pose transition.

        `wrist_speed` is a scalar read of current motion, used only for the
        `suppress_above_speed` gate that stops a swiping hand's mid-flight
        shape from registering as a pose. This is NOT the motion segmenter:
        a completely stationary hand still triggers poses normally, which is
        exactly what PRD validation test 3 checks.
        """
        if curls is None:
            # Tolerate a short tracking dropout (MediaPipe flickers
            # present/absent for a single frame often enough that treating
            # every blip as "hand gone" wiped hold-progress and the stable
            # pose before a held thumbs_up/fist could ever fire -- see
            # thresholds.yaml's hand_lost_grace_frames note). Only past the
            # grace window do we actually drop the stable pose so a
            # re-appearing hand re-triggers.
            self._absent_run += 1
            self._last_result = None
            if self._absent_run <= self.hand_lost_grace:
                self._candidate, self._candidate_run = None, 0
                return None
            self._candidate, self._candidate_run, self._stable = None, 0, None
            return None

        self._absent_run = 0
        result = self.classify(curls)
        self._last_result = result

        if wrist_speed > self.suppress_above_speed:
            # Freeze transition tracking while the hand is travelling fast.
            self._candidate, self._candidate_run = None, 0
            return None

        candidate = result.name if result.accepted else None
        if candidate == self._candidate:
            self._candidate_run += 1
        else:
            self._candidate, self._candidate_run = candidate, 1

        if self._candidate_run < self.hold_frames:
            return None
        if candidate == self._stable:
            return None

        previous, self._stable = self._stable, candidate
        if candidate is None:
            return None      # falling out of a pose is not itself an event
        return PoseEvent(name=candidate, previous=previous, result=result)
