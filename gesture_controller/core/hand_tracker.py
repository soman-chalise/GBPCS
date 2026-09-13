"""MediaPipe HandLandmarker wrapper + base feature extraction (PRD 4.1).

This module is the ONLY place that talks to MediaPipe. It produces two
deliberately SEPARATE streams from every frame:

  1. `shape`      -- 63-dim normalized landmark vector (pose recognizer's input)
  2. `wrist_xy`   -- 2D wrist position    (motion segmenter + trajectory recognizer's input)

They are smoothed by two independent EMA filters and are never concatenated
into a single vector. That separation is the central architectural constraint
from the PRD: attempt 1 mixed them into one 66-dim vector and shape jitter
became indistinguishable from spatial motion.

NOTE ON ROI CROPPING: there is none. Detection runs on the full frame every
frame, on purpose -- see README "Dropped optimizations".
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python import vision

from .config import Config, resolve

WRIST = 0
MIDDLE_MCP = 9

# (finger_name, mcp_idx, tip_idx) -- used by the pose feature
FINGERS = [
    ("index", 5, 8),
    ("middle", 9, 12),
    ("ring", 13, 16),
    ("pinky", 17, 20),
]
THUMB = ("thumb", 2, 4)

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
]


@dataclass
class HandFrame:
    """One frame of tracking output. Two independent streams, plus raw pixels."""

    present: bool
    timestamp: float
    shape: Optional[np.ndarray] = None       # (63,) smoothed, wrist-relative, scale-normalized
    wrist_xy: Optional[np.ndarray] = None    # (2,) smoothed, aspect-corrected image coords
    hand_scale: float = 0.0                  # wrist -> middle-MCP distance, in aspect-corrected units
    curls: Optional[np.ndarray] = None       # (4,) or (5,) per-finger curl ratios
    curl_names: tuple = ()
    pixels: Optional[np.ndarray] = None      # (21,2) int pixel coords, for drawing only
    landmarks_rel: Optional[np.ndarray] = None  # (21,3) wrist-relative, scale-normalized,
                                                 # RAW -- no EMA smoothing, no z_weight scaling.
                                                 # For any per-frame geometric rule (built-in
                                                 # poses) that needs an instantaneous, undistorted
                                                 # read. `shape` below is smoothed and z-weighted
                                                 # for the nearest-centroid path and must not be
                                                 # reused for rule-based thresholds -- see
                                                 # builtin_gestures.py.


class HandTracker:
    def __init__(self, cfg: Config):
        t = cfg.tracker
        self._include_thumb = bool(cfg.pose["include_thumb"])
        self._z_weight = float(t["z_weight"])
        self._alpha_shape = float(t["shape_smooth_alpha"])
        self._alpha_traj = float(t["trajectory_smooth_alpha"])

        grid = int(t["clahe_tile_grid"])
        self._clahe = cv2.createCLAHE(
            clipLimit=float(t["clahe_clip_limit"]), tileGridSize=(grid, grid)
        )

        options = vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=resolve(t["model_path"])),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=int(t["num_hands"]),
            min_hand_detection_confidence=float(t["min_hand_detection_confidence"]),
            min_hand_presence_confidence=float(t["min_hand_presence_confidence"]),
            min_tracking_confidence=float(t["min_tracking_confidence"]),
        )
        self._landmarker = vision.HandLandmarker.create_from_options(options)

        # Two independent EMA states -- never a shared filter.
        self._ema_shape: Optional[np.ndarray] = None
        self._ema_wrist: Optional[np.ndarray] = None
        self._last_ts_ms = -1

    # -- lifecycle ---------------------------------------------------------
    def close(self) -> None:
        self._landmarker.close()

    def reset_smoothing(self) -> None:
        """Drop EMA state, e.g. after the hand has been absent for a while."""
        self._ema_shape = None
        self._ema_wrist = None

    # -- preprocessing -----------------------------------------------------
    def normalize_lighting(self, bgr: np.ndarray) -> np.ndarray:
        """CLAHE on the L channel -- basic lighting robustness (PRD section 3)."""
        lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
        lab[:, :, 0] = self._clahe.apply(lab[:, :, 0])
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    # -- main entry point --------------------------------------------------
    def process(self, bgr: np.ndarray, timestamp: float) -> HandFrame:
        h, w = bgr.shape[:2]
        aspect = w / float(h)

        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        # detect_for_video requires strictly increasing timestamps.
        ts_ms = int(timestamp * 1000)
        if ts_ms <= self._last_ts_ms:
            ts_ms = self._last_ts_ms + 1
        self._last_ts_ms = ts_ms

        result = self._landmarker.detect_for_video(mp_image, ts_ms)
        if not result.hand_landmarks:
            return HandFrame(present=False, timestamp=timestamp)

        lm = result.hand_landmarks[0]
        # Aspect-correct x so that a horizontal swipe isn't compressed relative
        # to a vertical one -- both streams below use these units.
        pts = np.array([[p.x * aspect, p.y, p.z * aspect] for p in lm], dtype=np.float64)

        scale = float(np.linalg.norm(pts[MIDDLE_MCP, :2] - pts[WRIST, :2]))
        if scale < 1e-6:
            return HandFrame(present=False, timestamp=timestamp)

        # --- stream 1: shape vector (nearest-centroid path) ---------------
        # Keep the raw wrist-relative geometry around BEFORE z-weighting/EMA
        # for anything that needs an instantaneous, undistorted read (the
        # built-in rule-based poses) -- only the copy fed to the smoothed
        # `shape` output below gets z_weight applied and gets EMA'd.
        landmarks_rel = (pts - pts[WRIST]) / scale
        rel = landmarks_rel.copy()
        rel[:, 2] *= self._z_weight
        shape = rel.reshape(-1)
        shape = self._ema(shape, "_ema_shape", self._alpha_shape)

        # --- stream 2: wrist trajectory (segmenter + trajectory recognizer)
        # Absolute wrist position, NOT wrist-relative (that would be constant 0).
        wrist_xy = pts[WRIST, :2].copy()
        wrist_xy = self._ema(wrist_xy, "_ema_wrist", self._alpha_traj)

        curls, curl_names = self._curl_ratios(pts, scale)

        pixels = np.stack(
            [(pts[:, 0] / aspect * w).astype(int), (pts[:, 1] * h).astype(int)], axis=1
        )

        return HandFrame(
            present=True,
            timestamp=timestamp,
            shape=shape,
            wrist_xy=wrist_xy,
            hand_scale=scale,
            curls=curls,
            curl_names=curl_names,
            pixels=pixels,
            landmarks_rel=landmarks_rel,
        )

    # -- helpers -----------------------------------------------------------
    def _ema(self, value: np.ndarray, attr: str, alpha: float) -> np.ndarray:
        prev = getattr(self, attr)
        out = value if prev is None else alpha * value + (1.0 - alpha) * prev
        setattr(self, attr, out)
        return out.copy()

    def _curl_ratios(self, pts: np.ndarray, scale: float):
        """Per-finger curl ratio = ||tip - MCP|| / hand_scale  (PRD 4.4).

        Extended finger ~0.9-1.3, curled finger ~0.2-0.5. Uses raw (unsmoothed)
        landmarks by design -- the pose recognizer does its own temporal
        stabilization via hold_frames, and double-smoothing adds lag to what is
        supposed to be an instantaneous per-frame read.
        """
        fingers = list(FINGERS) + ([THUMB] if self._include_thumb else [])
        names, vals = [], []
        for name, mcp, tip in fingers:
            names.append(name)
            vals.append(float(np.linalg.norm(pts[tip, :2] - pts[mcp, :2]) / scale))
        return np.array(vals, dtype=np.float64), tuple(names)
