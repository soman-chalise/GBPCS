"""Camera diagnostic -- find out why the preview is blank.

Tries every OpenCV backend and camera index, reads a burst of frames from each,
and reports what the pixels ACTUALLY contain (not just that read() returned
True -- a blocked or shuttered webcam usually returns True with a black frame).

Run:  venv/Scripts/python.exe gesture_controller/tools/camera_check.py
      ...                                                  camera_check.py --save

`--save` additionally writes one captured frame per backend into logs/. It is
off by default because that frame is a photo of whoever is in front of the
camera.
"""

from __future__ import annotations

import os
import sys

import cv2
import numpy as np

OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "logs")

BACKENDS = [
    ("CAP_DSHOW", cv2.CAP_DSHOW),
    ("CAP_MSMF", getattr(cv2, "CAP_MSMF", None)),
    ("CAP_ANY", cv2.CAP_ANY),
]

# A webcam's first frames are often black while auto-exposure ramps up, so
# always look at a late frame, never frame 0.
WARMUP = 15
SAMPLE = 10


def describe(frames) -> str:
    a = np.array(frames, dtype=np.float32)
    mean = a.mean()
    b, g, r = (a[..., i].mean() for i in range(3))
    # Spatial detail must be measured PER CHANNEL. A uniform tinted fill such
    # as BGR(10,0,0) has zero real detail, but its across-channel spread would
    # show up as a healthy-looking std if the channels are pooled.
    spatial = float(max(a[-1][..., i].std() for i in range(3)))
    # Does the image change between frames at all? A frozen or synthetic feed
    # gives near-zero inter-frame difference.
    motion = float(np.abs(np.diff(a, axis=0)).mean()) if len(a) > 1 else 0.0

    if spatial < 1.0 and motion < 0.05:
        verdict = ("UNIFORM STATIC FILL -- not camera output at all. Usually "
                   "another app holds the camera and Windows is handing this "
                   "process placeholder frames.")
    elif mean < 6:
        verdict = "NEARLY BLACK -- lens covered, or no light reaching the sensor"
    elif motion < 0.15:
        verdict = "FROZEN -- pixels vary in space but never change over time"
    else:
        verdict = "LOOKS LIKE REAL VIDEO"
    return (
        "mean={:6.1f}  B={:5.1f} G={:5.1f} R={:5.1f}  spatial detail={:5.2f}  "
        "inter-frame delta={:.3f}\n      -> {}".format(
            mean, b, g, r, spatial, motion, verdict)
    )


def try_open(name, backend, index) -> bool:
    if backend is None:
        return False
    cap = cv2.VideoCapture(index, backend)
    if not cap.isOpened():
        print("  [{}] index {}: could not open".format(name, index))
        cap.release()
        return False

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    for _ in range(WARMUP):
        cap.read()

    frames, fails = [], 0
    for _ in range(SAMPLE):
        ok, f = cap.read()
        if ok and f is not None:
            frames.append(f)
        else:
            fails += 1

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
    cc = "".join(chr((fourcc >> 8 * i) & 0xFF) for i in range(4)).strip()
    exposure = cap.get(cv2.CAP_PROP_EXPOSURE)
    brightness = cap.get(cv2.CAP_PROP_BRIGHTNESS)
    cap.release()

    if not frames:
        print("  [{}] index {}: opened but every read failed".format(name, index))
        return False

    print("  [{}] index {}: {}x{} fourcc={} exposure={} brightness={} "
          "failed_reads={}".format(name, index, w, h, cc or "?", exposure,
                                   brightness, fails))
    print("      " + describe(frames))

    # Saving is opt-in: this writes a photo of whoever is in front of the
    # camera, so it should never happen as a silent side effect.
    if "--save" in sys.argv:
        path = os.path.join(OUT_DIR, "camtest_{}_{}.png".format(name, index))
        cv2.imwrite(path, frames[-1])
        print("      last frame saved: {}  (delete when done)".format(path))
    return True


def main() -> int:
    os.makedirs(OUT_DIR, exist_ok=True)
    print("=" * 74)
    print("CAMERA DIAGNOSTIC")
    print("=" * 74)
    print("OpenCV {}".format(cv2.__version__))
    print("Reading {} warmup + {} sample frames per backend/index.".format(
        WARMUP, SAMPLE))
    print("Pass --save to also write one frame per backend into logs/ -- that "
          "is a photo of you, so it is off by default.\n")

    any_ok = False
    for name, backend in BACKENDS:
        print("{}:".format(name))
        for index in (0, 1, 2):
            if try_open(name, backend, index):
                any_ok = True
        print("")

    if not any_ok:
        print("No camera could be opened on any backend.")
        return 1

    print("=" * 74)
    print("If any entry says LOOKS LIKE REAL VIDEO, the camera is fine.")
    print("")
    print("UNIFORM STATIC FILL means the device opened and streamed, but the")
    print("pixels are a constant synthetic fill -- most often because another")
    print("app already holds the camera. Windows gives the second app")
    print("placeholder frames rather than refusing to open the device. Close")
    print("the Windows Camera app, Teams, Zoom, Slack, OBS, or a browser tab")
    print("with a camera preview, then re-run.")
    print("")
    print("NEARLY BLACK means real sensor output with no light: check for a")
    print("privacy shutter, a hardware/Fn camera kill switch, or a dark room.")
    print("")
    print("Nothing opening at all: Settings > Privacy & security > Camera")
    print("-> 'Let desktop apps access your camera' must be ON.")
    print("=" * 74)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
