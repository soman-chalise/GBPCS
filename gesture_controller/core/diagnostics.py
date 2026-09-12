"""Diagnostics for the PRD section 7 validation plan.

Three things live here:

  * `distance_report()`  -- intra-class vs inter-class distance table
                            (validation test 1). Printed automatically every
                            time a sample is added, and on demand with 'd'.
  * `EventLog`           -- append-only CSV of every accepted/rejected
                            classification (tests 2, 4, 5 read this).
  * `FrameLog`           -- optional per-frame CSV of speed / segmenter state /
                            pose distance (test 2 and 3 read this).

Nothing here changes recognition behaviour; it only makes it measurable.
"""

from __future__ import annotations

import csv
import os
import time
from typing import Dict, List, Optional

import numpy as np

from .template_store import POSE, TRAJECTORY, TemplateStore


# --------------------------------------------------------------------------
# Validation test 1: intra-class vs inter-class separation
# --------------------------------------------------------------------------
def _pairwise(recognizer, feats_a: np.ndarray, feats_b: np.ndarray) -> List[float]:
    return [
        recognizer.distance(a, b)
        for a in feats_a
        for b in feats_b
        if len(a) == len(b)
    ]


def distance_report(
    store: TemplateStore, traj_rec, pose_rec, gtype: Optional[str] = None
) -> str:
    """Build the intra/inter distance table for one or both recognizer families."""
    out: List[str] = []
    families = [(TRAJECTORY, traj_rec), (POSE, pose_rec)]
    if gtype:
        families = [f for f in families if f[0] == gtype]

    for family, rec in families:
        gestures = store.of_type(family)
        out.append("")
        out.append("=" * 74)
        out.append("{} RECOGNIZER -- distance report".format(family.upper()))
        out.append("=" * 74)
        if not gestures:
            out.append("  (no {} gestures recorded)".format(family))
            continue

        # --- intra-class -------------------------------------------------
        out.append("")
        out.append("INTRA-CLASS (same gesture, different samples) -- want SMALL")
        out.append(
            "  {:<14} {:>7} {:>8} {:>8} {:>8} {:>8}  {}".format(
                "gesture", "n", "mean", "std", "min", "max", "threshold"
            )
        )
        intra_stats: Dict[str, tuple] = {}
        for name, g in sorted(gestures.items()):
            thr = rec.threshold_for(name)
            src = rec.threshold_source(name)
            if not g.is_calibrated:
                out.append(
                    "  {:<14} {:>7} {:>8} {:>8} {:>8} {:>8}  {:.3f}  {}".format(
                        name, g.count, "-", "-", "-", "-", thr, src
                    )
                )
                out.append(
                    "      ^ 1 sample: no intra-class distance yet. Expected, "
                    "not an error (PRD 5)."
                )
                continue
            d = np.array(rec.intra_class_distances(g))
            intra_stats[name] = (d.mean(), d.std(), d.min(), d.max())
            out.append(
                "  {:<14} {:>7} {:>8.3f} {:>8.3f} {:>8.3f} {:>8.3f}  {:.3f}  {}".format(
                    name, g.count, d.mean(), d.std(), d.min(), d.max(), thr, src
                )
            )

        # --- inter-class ---------------------------------------------------
        names = sorted(gestures)
        if len(names) < 2:
            out.append("")
            out.append(
                "INTER-CLASS: needs >=2 {} gestures; only '{}' recorded. "
                "The margin-over-runner-up check is inactive with one gesture, "
                "so the distance threshold is doing all the rejection work.".format(
                    family, names[0] if names else "-"
                )
            )
            continue

        out.append("")
        out.append("INTER-CLASS (different gestures) -- want LARGE")
        out.append(
            "  {:<14} {:<14} {:>8} {:>8}".format("gesture A", "gesture B", "min", "mean")
        )
        inter_min: Dict[str, float] = {n: float("inf") for n in names}
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                d = np.array(_pairwise(rec, gestures[a].matrix(), gestures[b].matrix()))
                if not len(d):
                    continue
                out.append(
                    "  {:<14} {:<14} {:>8.3f} {:>8.3f}".format(a, b, d.min(), d.mean())
                )
                inter_min[a] = min(inter_min[a], float(d.min()))
                inter_min[b] = min(inter_min[b], float(d.min()))

        # --- the verdict line ---------------------------------------------
        out.append("")
        out.append("SEPARATION VERDICT (per gesture)")
        for name in names:
            g = gestures[name]
            lo = inter_min.get(name, float("inf"))
            thr = rec.threshold_for(name)
            if name in intra_stats:
                intra_max = intra_stats[name][3]
                gap = lo - intra_max
                verdict = (
                    "GOOD" if gap > 0 else "OVERLAP -- samples of different "
                    "gestures are as close as samples of the same one"
                )
                out.append(
                    "  {:<14} intra_max={:.3f}  inter_min={:.3f}  gap={:+.3f}  {}".format(
                        name, intra_max, lo, gap, verdict
                    )
                )
            else:
                verdict = (
                    "GOOD" if lo > thr else
                    "RISK -- nearest other gesture is inside this gesture's threshold"
                )
                out.append(
                    "  {:<14} intra=n/a (1 sample)  inter_min={:.3f}  thr={:.3f}  {}".format(
                        name, lo, thr, verdict
                    )
                )
        out.append("")
        out.append(
            "Read it as: every inter_min should sit well above that gesture's "
            "threshold, and above its intra_max once calibrated. If a row says "
            "OVERLAP or RISK, record 1-2 more samples of that gesture."
        )
    return "\n".join(out)


# --------------------------------------------------------------------------
# CSV logs
# --------------------------------------------------------------------------
class EventLog:
    """Every classification attempt, accepted or rejected."""

    FIELDS = [
        "wall_time", "session_t", "mode", "source", "outcome", "gesture",
        "distance", "threshold", "threshold_source", "runner_up",
        "runner_up_distance", "margin_ratio", "fired", "key", "detail",
        "seg_frames", "seg_duration", "seg_peak_speed", "seg_mean_speed",
        "seg_close_reason", "session_label",
    ]

    def __init__(self, path: str, session_label: str = ""):
        self.path = path
        self.session_label = session_label
        self.t0 = time.time()
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self._new = not os.path.exists(path) or os.path.getsize(path) == 0
        self._fh = open(path, "a", newline="", encoding="utf-8")
        self._w = csv.DictWriter(self._fh, fieldnames=self.FIELDS)
        if self._new:
            self._w.writeheader()
            self._fh.flush()

    def write(self, **row) -> None:
        rec = {k: "" for k in self.FIELDS}
        rec["wall_time"] = "{:.3f}".format(time.time())
        rec["session_t"] = "{:.3f}".format(time.time() - self.t0)
        rec["session_label"] = self.session_label
        for k, v in row.items():
            if k in rec:
                rec[k] = v
        self._w.writerow(rec)
        self._fh.flush()

    def close(self) -> None:
        try:
            self._fh.close()
        except Exception:                              # noqa: BLE001
            pass


class FrameLog:
    """Optional per-frame trace. Only opened with app.py --log-frames."""

    FIELDS = [
        "session_t", "hand", "wrist_x", "wrist_y", "hand_scale", "speed",
        "seg_state", "seg_buffered", "curls", "pose_best", "pose_distance",
        "pose_stable",
    ]

    def __init__(self, path: str):
        self.path = path
        self.t0 = time.time()
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self._fh = open(path, "w", newline="", encoding="utf-8")
        self._w = csv.DictWriter(self._fh, fieldnames=self.FIELDS)
        self._w.writeheader()

    def write(self, **row) -> None:
        rec = {k: "" for k in self.FIELDS}
        rec["session_t"] = "{:.3f}".format(time.time() - self.t0)
        for k, v in row.items():
            if k in rec:
                rec[k] = v
        self._w.writerow(rec)

    def close(self) -> None:
        try:
            self._fh.close()
        except Exception:                              # noqa: BLE001
            pass


# --------------------------------------------------------------------------
# Validation test 4/5: end-of-session tally
# --------------------------------------------------------------------------
def session_summary(mapper, live_seconds: float, label: str = "") -> str:
    fired = [e for e in mapper.history if e.fired]
    suppressed = [e for e in mapper.history if not e.fired]
    lines = [
        "",
        "=" * 74,
        "SESSION SUMMARY" + (" [{}]".format(label) if label else ""),
        "=" * 74,
        "LIVE time            : {:.1f}s ({:.1f} min)".format(
            live_seconds, live_seconds / 60.0
        ),
        "Triggers fired       : {}".format(len(fired)),
        "Suppressed (cooldown): {}".format(len(suppressed)),
    ]
    if live_seconds > 0:
        lines.append(
            "Trigger rate         : {:.2f} / minute".format(
                len(fired) / (live_seconds / 60.0)
            )
        )
    if fired:
        counts: Dict[str, int] = {}
        for e in fired:
            key = "{} ({})".format(e.gesture, e.source)
            counts[key] = counts.get(key, 0) + 1
        lines.append("Breakdown:")
        for k, v in sorted(counts.items(), key=lambda kv: -kv[1]):
            lines.append("  {:<28} {}".format(k, v))
    lines.append("")
    lines.append(
        "For validation test 4, run this with --dry-run --session-label talking "
        "and no deliberate gestures. Every trigger above is a FALSE POSITIVE."
    )
    return "\n".join(lines)
