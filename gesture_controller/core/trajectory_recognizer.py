"""Trajectory recognizer -- the swipe family (PRD 4.3).

INPUT: the 2D wrist path of a closed segment. Nothing else. This module never
sees the 63-dim shape vector, so finger jitter cannot perturb its resampling or
its distances.

FEATURE PIPELINE
    1. Path in hand-widths (segment already divides by mean hand_scale).
    2. Resample to N points by CUMULATIVE PATH DISTANCE, not by frame index or
       time. This is what makes a slow swipe and a fast swipe produce the same
       feature -- speed invariance by construction, no time warping needed.
    3. Translate so point 0 is the origin (translation invariance; direction
       is preserved, which is the whole point for left-vs-right).
    4. Divide by total path length (size invariance). After this a clean
       straight swipe ends near a unit vector in its direction, and a
       back-and-forth wiggle ends near the origin.
    5. Append net displacement (last - first) weighted by
       `net_displacement_weight` -- the dominant swipe signal per the PRD.

CLASSIFICATION: nearest neighbour over stored templates (Euclidean, normalized
by sqrt(dim) so a threshold means the same thing regardless of resample_points),
gated by BOTH a distance threshold and a margin-over-runner-up check.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from .config import Config
from .template_store import TRAJECTORY, Gesture, TemplateStore


@dataclass
class TrajectoryResult:
    """Outcome of one classification attempt -- accepted or not."""

    accepted: bool
    name: Optional[str]
    distance: float
    threshold: float
    runner_up_name: Optional[str] = None
    runner_up_distance: float = float("inf")
    margin_ratio: float = float("inf")     # distance / runner_up_distance
    reason: str = ""
    threshold_source: str = ""             # "default(1 sample)" or "calibrated(n samples)"
    scores: Dict[str, float] = field(default_factory=dict)   # best distance per gesture

    def describe(self) -> str:
        head = "ACCEPT {}".format(self.name) if self.accepted else "reject"
        ru = (
            "runner-up {} {:.3f} (ratio {:.2f})".format(
                self.runner_up_name, self.runner_up_distance, self.margin_ratio
            )
            if self.runner_up_name
            else "runner-up none (single gesture)"
        )
        return "[traj] {}: best={} d={:.3f} thr={:.3f} {} | {} | {}".format(
            head, self.name, self.distance, self.threshold, self.threshold_source,
            ru, self.reason,
        )


def resample_by_path_distance(points: np.ndarray, n: int) -> np.ndarray:
    """Resample a polyline to exactly `n` points spaced evenly by arc length.

    Deliberately arc-length based, NOT index/time based: two performances of
    the same swipe at different speeds have different frame counts and
    different per-frame spacing, but the same shape along their path.
    """
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    if len(pts) == 0:
        return np.zeros((n, 2), dtype=np.float64)
    if len(pts) == 1:
        return np.repeat(pts, n, axis=0)

    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    total = cum[-1]
    if total <= 1e-9:
        # Degenerate: no actual travel. Return the single location n times;
        # the segmenter's min_segment_path_length gate normally catches this.
        return np.repeat(pts[:1], n, axis=0)

    targets = np.linspace(0.0, total, n)
    out = np.empty((n, 2), dtype=np.float64)
    out[:, 0] = np.interp(targets, cum, pts[:, 0])
    out[:, 1] = np.interp(targets, cum, pts[:, 1])
    return out


class TrajectoryRecognizer:
    """Owns every gesture tagged `trajectory` in the store."""

    def __init__(self, cfg: Config, store: TemplateStore):
        t = cfg.trajectory
        self.n_points = int(t["resample_points"])
        self.disp_weight = float(t["net_displacement_weight"])
        self.default_threshold = float(t["default_threshold"])
        self.calibration_k = float(t["calibration_k"])
        self.threshold_floor = float(t["threshold_floor"])
        self.threshold_ceiling = float(t["threshold_ceiling"])
        self.margin_ratio = float(t["margin_ratio"])
        self.store = store
        self._thresholds: Dict[str, float] = {}
        self._threshold_sources: Dict[str, str] = {}
        self.recalibrate()

    # -- feature extraction ------------------------------------------------
    def feature_from_points(self, points_in_hand_widths: np.ndarray) -> np.ndarray:
        pts = resample_by_path_distance(points_in_hand_widths, self.n_points)

        pts = pts - pts[0]                                   # translation invariance
        seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        path_len = float(np.sum(seg))
        if path_len > 1e-9:
            pts = pts / path_len                             # size invariance

        net = pts[-1] - pts[0]
        return np.concatenate([pts.reshape(-1), net * self.disp_weight])

    def feature_from_segment(self, segment) -> np.ndarray:
        return self.feature_from_points(segment.normalized_points())

    @property
    def feature_dim(self) -> int:
        return self.n_points * 2 + 2

    # -- distance ----------------------------------------------------------
    def distance(self, a: np.ndarray, b: np.ndarray) -> float:
        """Euclidean, normalized by sqrt(dim).

        The normalization keeps thresholds meaningful if `resample_points`
        changes -- otherwise every retune of N would silently require retuning
        every threshold too.
        """
        d = np.asarray(a) - np.asarray(b)
        return float(np.linalg.norm(d) / np.sqrt(len(d)))

    # -- thresholds --------------------------------------------------------
    def recalibrate(self) -> None:
        """Recompute per-gesture thresholds. Called on every new sample.

        1 sample  -> generic pre-tuned default (PRD: 1 sample is enough to go live)
        >=2       -> mean(intra-class pairwise) + k * std, clamped to [floor, ceiling]
        """
        self._thresholds.clear()
        self._threshold_sources.clear()
        for name, g in self.store.of_type(TRAJECTORY).items():
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

    # -- classification ----------------------------------------------------
    def classify(self, feature: np.ndarray) -> TrajectoryResult:
        gestures = self.store.of_type(TRAJECTORY)
        if not gestures:
            return TrajectoryResult(
                False, None, float("inf"), 0.0,
                reason="no trajectory gestures recorded",
            )

        per_gesture: Dict[str, float] = {}
        for name, g in gestures.items():
            feats = g.matrix()
            if feats.shape[1] != len(feature):
                per_gesture[name] = float("inf")
                continue
            per_gesture[name] = min(self.distance(feature, f) for f in feats)

        ranked = sorted(per_gesture.items(), key=lambda kv: kv[1])
        best_name, best_dist = ranked[0]
        runner_name, runner_dist = (ranked[1] if len(ranked) > 1 else (None, float("inf")))
        ratio = best_dist / runner_dist if runner_dist > 0 else float("inf")

        thr = self.threshold_for(best_name)
        src = self.threshold_source(best_name)
        res = TrajectoryResult(
            accepted=False,
            name=best_name,
            distance=best_dist,
            threshold=thr,
            runner_up_name=runner_name,
            runner_up_distance=runner_dist,
            margin_ratio=ratio,
            threshold_source=src,
            scores=per_gesture,
        )

        if best_dist > thr:
            res.reason = "distance over threshold"
            return res
        # Margin check. With a single recorded gesture there is no runner-up,
        # so this passes trivially -- documented, expected, and the reason the
        # distance threshold still has to do real work in that case.
        if runner_name is not None and ratio > self.margin_ratio:
            res.reason = "ambiguous vs {} (ratio {:.2f} > {:.2f})".format(
                runner_name, ratio, self.margin_ratio
            )
            return res

        res.accepted = True
        res.reason = "ok"
        return res
