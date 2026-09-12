"""Persisted gesture-template store (PRD section 6).

One JSON file, `data/gestures.json`, holds every recorded sample so a recording
session survives a restart. Each gesture carries an explicit `type` --
"trajectory" or "pose" -- chosen by the user at record time (PRD section 5,
explicit tagging). The type decides which recognizer owns the gesture; nothing
is inferred from the feature vector at load time.

Samples are stored as plain lists of floats so the file stays diffable and
hand-editable.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

TRAJECTORY = "trajectory"
POSE = "pose"
VALID_TYPES = (TRAJECTORY, POSE)

STORE_VERSION = 1


@dataclass
class Sample:
    feature: np.ndarray
    recorded_at: float
    meta: dict = field(default_factory=dict)

    def to_json(self) -> dict:
        return {
            "feature": [float(v) for v in np.asarray(self.feature).reshape(-1)],
            "recorded_at": self.recorded_at,
            "meta": self.meta,
        }

    @staticmethod
    def from_json(doc: dict) -> "Sample":
        return Sample(
            feature=np.array(doc["feature"], dtype=np.float64),
            recorded_at=float(doc.get("recorded_at", 0.0)),
            meta=doc.get("meta", {}) or {},
        )


@dataclass
class Gesture:
    name: str
    type: str
    samples: List[Sample] = field(default_factory=list)

    @property
    def count(self) -> int:
        return len(self.samples)

    @property
    def is_calibrated(self) -> bool:
        """>=2 samples means a per-gesture threshold can be computed (PRD 4.3)."""
        return self.count >= 2

    def matrix(self) -> np.ndarray:
        return np.array([s.feature for s in self.samples], dtype=np.float64)


class TemplateStore:
    def __init__(self, path: str):
        self.path = path
        self.gestures: Dict[str, Gesture] = {}

    # -- persistence -------------------------------------------------------
    def load(self) -> "TemplateStore":
        if not os.path.exists(self.path):
            return self
        with open(self.path, "r", encoding="utf-8") as fh:
            doc = json.load(fh)
        for name, g in (doc.get("gestures") or {}).items():
            gtype = g.get("type")
            if gtype not in VALID_TYPES:
                raise ValueError(
                    "gesture '{}' in {} has invalid type '{}' (expected one of {})".format(
                        name, self.path, gtype, VALID_TYPES
                    )
                )
            self.gestures[name] = Gesture(
                name=name,
                type=gtype,
                samples=[Sample.from_json(s) for s in g.get("samples", [])],
            )
        return self

    def save(self) -> None:
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        doc = {
            "version": STORE_VERSION,
            "saved_at": time.time(),
            "gestures": {
                name: {
                    "type": g.type,
                    "samples": [s.to_json() for s in g.samples],
                }
                for name, g in sorted(self.gestures.items())
            },
        }
        tmp = self.path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(doc, fh, indent=2)
        os.replace(tmp, self.path)

    # -- mutation ----------------------------------------------------------
    def add_sample(
        self, name: str, gtype: str, feature: np.ndarray, meta: Optional[dict] = None
    ) -> Gesture:
        if gtype not in VALID_TYPES:
            raise ValueError("gesture type must be one of {}".format(VALID_TYPES))
        g = self.gestures.get(name)
        if g is None:
            g = Gesture(name=name, type=gtype)
            self.gestures[name] = g
        elif g.type != gtype:
            raise ValueError(
                "gesture '{}' already exists as type '{}'; refusing to add a "
                "'{}' sample. Rename it or delete it first.".format(
                    name, g.type, gtype
                )
            )
        feat = np.asarray(feature, dtype=np.float64).reshape(-1)
        if g.samples and g.samples[0].feature.shape != feat.shape:
            raise ValueError(
                "feature length mismatch for '{}': stored {} vs new {}. "
                "This usually means a config change (e.g. resample_points or "
                "include_thumb) invalidated existing templates.".format(
                    name, g.samples[0].feature.shape[0], feat.shape[0]
                )
            )
        g.samples.append(Sample(feature=feat, recorded_at=time.time(), meta=meta or {}))
        self.save()
        return g

    def delete_gesture(self, name: str) -> bool:
        if name in self.gestures:
            del self.gestures[name]
            self.save()
            return True
        return False

    def delete_last_sample(self, name: str) -> bool:
        g = self.gestures.get(name)
        if not g or not g.samples:
            return False
        g.samples.pop()
        if not g.samples:
            del self.gestures[name]
        self.save()
        return True

    # -- queries -----------------------------------------------------------
    def of_type(self, gtype: str) -> Dict[str, Gesture]:
        return {n: g for n, g in self.gestures.items() if g.type == gtype and g.samples}

    def summary_lines(self) -> List[str]:
        if not self.gestures:
            return ["(no gestures recorded yet)"]
        out = []
        for name, g in sorted(self.gestures.items()):
            mode = "calibrated" if g.is_calibrated else "default-threshold"
            out.append(
                "{:<14} {:<10} {} sample(s)  [{}]".format(
                    name, g.type, g.count, mode
                )
            )
        return out
