"""Find a phone streaming as a camera over USB tethering (or Wi-Fi).

Probes the likely addresses and ports for the common Android phone-camera
apps, then verifies each hit by actually pulling a frame through OpenCV and
checking it contains a real image.

Prints a ready-to-paste command for whatever it finds.

Run:  venv/Scripts/python.exe gesture_controller/tools/find_phone_camera.py
"""

from __future__ import annotations

import socket
import subprocess
import sys
import urllib.request

import cv2
import numpy as np

# app name -> (port, [stream paths to try])
APPS = {
    "IP Webcam": (8080, ["/video", "/videofeed"]),
    "DroidCam": (4747, ["/video", "/mjpegfeed?640x480"]),
    "Iriun": (8080, ["/video"]),
}

# Android USB tethering hands the phone .129 on this subnet almost universally.
WELL_KNOWN = ["192.168.42.129", "192.168.43.1", "172.20.10.1"]

TIMEOUT = 0.6


def local_subnet_hosts():
    """Gateways and neighbours of every active adapter -- the phone is one."""
    out = set()
    try:
        res = subprocess.run(["ipconfig"], capture_output=True, text=True,
                             timeout=10)
        for line in res.stdout.splitlines():
            if "Default Gateway" in line or "IPv4 Address" in line:
                part = line.split(":")[-1].strip()
                if part.count(".") == 3 and not part.startswith("0."):
                    out.add(part)
                    # The phone is typically .1 or .129 on the same /24.
                    a, b, c, _ = part.split(".")
                    out.add("{}.{}.{}.1".format(a, b, c))
                    out.add("{}.{}.{}.129".format(a, b, c))
    except Exception:                                   # noqa: BLE001
        pass
    return sorted(out)


def port_open(host: str, port: int) -> bool:
    s = socket.socket()
    s.settimeout(TIMEOUT)
    try:
        return s.connect_ex((host, port)) == 0
    finally:
        s.close()


def http_ok(url: str) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=TIMEOUT + 1) as r:
            return r.status == 200
    except Exception:                                   # noqa: BLE001
        return False


def verify_stream(url: str):
    """Pull real frames and confirm they are an image, not a placeholder."""
    cap = cv2.VideoCapture(url)
    if not cap.isOpened():
        cap.release()
        return None
    frames = []
    for _ in range(12):
        ok, f = cap.read()
        if ok and f is not None:
            frames.append(f)
    cap.release()
    if len(frames) < 3:
        return None
    a = np.array(frames[-6:], dtype=np.float32)
    spatial = float(max(a[-1][..., i].std() for i in range(3)))
    motion = float(np.abs(np.diff(a, axis=0)).mean())
    h, w = frames[-1].shape[:2]
    sharp = cv2.Laplacian(cv2.cvtColor(frames[-1], cv2.COLOR_BGR2GRAY),
                          cv2.CV_64F).var()
    return {
        "size": "{}x{}".format(w, h),
        "brightness": float(a.mean()),
        "spatial": spatial,
        "motion": motion,
        "sharpness": float(sharp),
        "real": spatial > 1.0 and motion > 0.05,
    }


def main() -> int:
    print("=" * 74)
    print("PHONE CAMERA FINDER")
    print("=" * 74)

    hosts = []
    for h in WELL_KNOWN + local_subnet_hosts():
        if h not in hosts:
            hosts.append(h)
    print("Probing {} address(es): {}".format(len(hosts), ", ".join(hosts)))
    print("")

    candidates = []
    for host in hosts:
        for app, (port, paths) in APPS.items():
            if not port_open(host, port):
                continue
            print("  {}:{} is OPEN  (looks like {})".format(host, port, app))
            for path in paths:
                url = "http://{}:{}{}".format(host, port, path)
                if http_ok(url) or True:      # some apps refuse HEAD/GET probes
                    candidates.append((app, url))

    if not candidates:
        print("  nothing found.")
        print("")
        print("Checklist:")
        print("  1. Phone connected by USB and UNLOCKED.")
        print("  2. USB tethering ON:")
        print("     Settings > Network & internet > Hotspot & tethering >")
        print("     USB tethering")
        print("  3. The camera app is running AND you pressed 'Start server'.")
        print("  4. The app shows a URL on the phone screen -- if it is not in")
        print("     the probed list above, pass it directly:")
        print("     app.py --camera http://THAT-IP:8080/video")
        return 1

    print("")
    print("Verifying streams (pulling real frames)...")
    print("")
    working = []
    for app, url in candidates:
        info = verify_stream(url)
        if info is None:
            print("  [dead ] {}".format(url))
            continue
        verdict = "REAL VIDEO" if info["real"] else "blank/placeholder"
        print("  [{}] {}".format("LIVE " if info["real"] else "blank", url))
        print("          {} brightness={:.0f}/255 sharpness={:.0f} "
              "detail={:.1f} motion={:.2f} -> {}".format(
                  info["size"], info["brightness"], info["sharpness"],
                  info["spatial"], info["motion"], verdict))
        if info["real"]:
            working.append((url, info))

    if not working:
        print("")
        print("Ports were open but no usable video came back. Make sure the")
        print("phone app is actively streaming (screen on, server started).")
        return 1

    best = max(working, key=lambda t: t[1]["sharpness"])
    print("")
    print("=" * 74)
    print("USE THIS:")
    print("")
    print("  venv\\Scripts\\python.exe gesture_controller\\app.py --dry-run "
          "--camera {}".format(best[0]))
    print("")
    print("To make it permanent, set this in config/thresholds.yaml:")
    print('  camera:')
    print('    source: "{}"'.format(best[0]))
    print("")
    print("Compare against your built-in camera, which measured brightness 71")
    print("and sharpness 80 -- higher sharpness means less motion blur and")
    print("far fewer hand_lost segments.")
    print("=" * 74)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
