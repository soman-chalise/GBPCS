"""Gesture-based presentation controller -- main loop.

The camera + recognition loop runs here, in the main thread, exactly as
before: HandTracker -> MotionSegmenter -> {custom trajectory/pose recognizers,
built-in rule-based detectors} -> ActionMapper / LaserPointerController. The
difference from v2 is the interface: there is no OpenCV window and no
keyboard shortcuts. All control (LIVE toggle, recording, gesture -> control
bindings, keystrokes on/off) happens through the local web control panel,
served by `web/server.py` on a background thread and opened automatically in
the default browser.

CLI:
  --dry-run             never send keystrokes (use for validation test 4)
  --camera SRC           device index (0, 1) or stream URL for a phone camera
  --fps N                cap processing to N frames/sec (0 = uncapped)
  --log-frames           write a per-frame CSV trace to logs/frames.csv
  --session-label NAME   tag rows in logs/events.csv (e.g. "talking")
  --stats                print the distance report and exit, no camera
  --port N               override web.port in thresholds.yaml
  --no-browser           don't auto-open the control panel in a browser
"""

from __future__ import annotations

import argparse
import os
import sys
import threading
import time
import webbrowser
from typing import List, Optional

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.action_mapper import ActionMapper
from core.bindings_store import BUILTIN_GESTURES, BindingsStore
from core.builtin_gestures import BuiltinGestureDetector
from core.config import Config, resolve
from core import controls as controls_mod
from core.diagnostics import EventLog, FrameLog, distance_report, session_summary
from core.frame_source import FrameSource
from core.hand_tracker import HAND_CONNECTIONS, HandTracker
from core.laser_pointer import LaserPointerController
from core.motion_segmenter import MotionSegmenter, MotionState
from core.pose_recognizer import PoseRecognizer
from core.template_store import POSE, TRAJECTORY, TemplateStore
from core.trajectory_recognizer import TrajectoryRecognizer
from web.server import create_app
from web.state import AppState

STORE_PATH = resolve("data/gestures.json")
BINDINGS_PATH = resolve("data/bindings.json")

WHITE = (255, 255, 255)
GREY = (170, 170, 170)
GREEN = (80, 230, 120)
AMBER = (60, 200, 250)
RED = (70, 70, 240)
BLUE = (240, 180, 90)

BUILTIN_TYPE = {
    "swipe_left": TRAJECTORY, "swipe_right": TRAJECTORY,
    "thumbs_up": POSE, "gun_point": POSE, "peace_sign": POSE,
    "open_palm_hold": POSE, "closed_fist_hold": POSE,
}


class App:
    def __init__(self, args):
        self.args = args
        self.cfg = Config.load()
        self.store = TemplateStore(STORE_PATH).load()
        self.bindings = BindingsStore(BINDINGS_PATH).load()
        self.tracker = HandTracker(self.cfg)
        self.segmenter = MotionSegmenter(self.cfg)
        self.traj = TrajectoryRecognizer(self.cfg, self.store)
        self.pose = PoseRecognizer(self.cfg, self.store)
        self.builtin = BuiltinGestureDetector(self.cfg)
        self.laser = LaserPointerController(self.cfg)
        self.mapper = ActionMapper(self.cfg, self.bindings, dry_run=args.dry_run)

        self.verbose = bool(self.cfg.logging["verbose_scores"])
        self.events = EventLog(
            resolve(self.cfg.logging["event_csv"]), args.session_label or ""
        )
        self.frames_log = (
            FrameLog(resolve(self.cfg.logging["frame_csv"])) if args.log_frames else None
        )

        self.live = False
        self.recording = False
        self.record_name: Optional[str] = None
        self.record_type: Optional[str] = None
        self.record_points: List[np.ndarray] = []
        self.record_scales: List[float] = []
        self.record_curls: List[np.ndarray] = []
        self.record_times: List[float] = []
        self.countdown_until = 0.0

        self.live_seconds = 0.0
        self._live_since: Optional[float] = None
        self.last_event_text = ""
        self.last_event_time = 0.0
        self.fps = 0.0
        self._fps_t = time.time()
        self._fps_n = 0
        self.cap = None
        cap_cfg = int(self.cfg.camera.get("process_fps_cap", 0) or 0)
        self.fps_cap = int(args.fps) if args.fps else cap_cfg
        self._min_interval = 1.0 / self.fps_cap if self.fps_cap > 0 else 0.0
        self._next_frame_at = 0.0

        self.state = AppState()
        self._last_hand_present = False

    # ==================================================================
    # setup / teardown
    # ==================================================================
    def open_camera(self) -> bool:
        c = self.cfg.camera
        raw = self.args.camera if self.args.camera is not None else c.get(
            "source", c.get("index", 0)
        )
        source, is_url = self._parse_source(raw)

        self.cap = FrameSource(source, threaded=bool(c.get("threaded_stream", True)))

        if not self.cap.is_opened():
            print("ERROR: could not open camera source {!r}.".format(source))
            if is_url:
                print("Check the phone app is streaming and the URL is reachable")
                print("in a browser on this PC.")
            else:
                print("Close any other app using the webcam, or set a different")
                print("camera.source in config/thresholds.yaml (or --camera N).")
            print("List what is available:  venv/Scripts/python.exe "
                  "gesture_controller/tools/camera_check.py")
            return False
        self.cap.configure(int(c["width"]), int(c["height"]), int(c["fps"]))
        self.cap.start()
        print("[camera] opened source {!r}{}".format(
            source, "  (threaded: newest-frame only)" if self.cap.threaded else ""))
        if self.fps_cap:
            print("[camera] processing capped at {} fps".format(self.fps_cap))
        return True

    @staticmethod
    def _parse_source(raw):
        if isinstance(raw, int):
            return raw, False
        text = str(raw).strip()
        if text.isdigit():
            return int(text), False
        return text, True

    def start_web_server(self) -> None:
        w = self.cfg.web
        host = self.args.host or str(w.get("host", "127.0.0.1"))
        port = int(self.args.port or w.get("port", 5000))
        flask_app = create_app(self.state)

        def _run():
            flask_app.run(host=host, port=port, debug=False, use_reloader=False,
                           threaded=True)

        threading.Thread(target=_run, daemon=True).start()
        url = "http://{}:{}/".format(host, port)
        print("[web] control panel at {}".format(url))
        if bool(w.get("auto_open_browser", True)) and not self.args.no_browser:
            time.sleep(0.6)
            try:
                webbrowser.open(url)
            except Exception as exc:                  # noqa: BLE001
                print("[web] could not auto-open a browser: {}".format(exc))

    def shutdown(self) -> None:
        if self._live_since is not None:
            self.live_seconds += time.time() - self._live_since
            self._live_since = None
        print(session_summary(self.mapper, self.live_seconds, self.args.session_label or ""))
        self.events.close()
        if self.frames_log:
            self.frames_log.close()
            print("Frame trace written to {}".format(self.frames_log.path))
        print("Event log: {}".format(self.events.path))
        if self.cap:
            self.cap.release()
        self.tracker.close()

    # ==================================================================
    # main loop
    # ==================================================================
    def run(self) -> int:
        if not self.open_camera():
            return 1
        self.start_web_server()
        print("=" * 74)
        print("GESTURE CONTROLLER -- control everything from the web panel above")
        print("=" * 74)

        try:
            while True:
                self.process_commands()

                if self._min_interval:
                    wait = self._next_frame_at - time.time()
                    if wait > 0:
                        time.sleep(min(wait, 0.05))
                        continue
                    self._next_frame_at = time.time() + self._min_interval

                self.cap.wait_for_new_frame(timeout=0.5)
                ok, frame = self.cap.read()
                if not ok:
                    print("camera read failed; stopping")
                    break
                if bool(self.cfg.camera["flip_horizontal"]):
                    frame = cv2.flip(frame, 1)
                h, w = frame.shape[:2]

                now = time.time()
                hand = self.tracker.process(
                    self.tracker.normalize_lighting(frame), now
                )

                status = self.segmenter.update(
                    hand.wrist_xy if hand.present else None,
                    hand.hand_scale if hand.present else 1.0,
                    now,
                )

                if self.recording and hand.present and now >= self.countdown_until:
                    if self.record_type == TRAJECTORY:
                        self.record_points.append(hand.wrist_xy.copy())
                        self.record_scales.append(hand.hand_scale)
                        self.record_times.append(now)
                    else:
                        self.record_curls.append(hand.curls.copy())

                if self.live and not self.recording:
                    if status.segment is not None:
                        self.handle_segment(status.segment)
                    pose_event = self.pose.update(
                        hand.curls if hand.present else None, self.segmenter.last_speed
                    )
                    if pose_event is not None:
                        self.handle_pose_event(pose_event)

                    builtin_ev = self.builtin.update_pose(
                        hand if hand.present else None, self.segmenter.last_speed
                    )
                    if builtin_ev is not None:
                        self.handle_builtin_trigger(builtin_ev.name)

                    self.update_laser(hand, w, h)
                elif hand.present:
                    self.pose.classify(hand.curls)

                if self.frames_log:
                    self.log_frame(hand, status)

                self.tick_fps()
                self.annotate(frame, hand, status)
                self.publish_frame(frame)
                self.publish_status()
        except KeyboardInterrupt:
            print("\ninterrupted")
        finally:
            self.shutdown()
        return 0

    # ==================================================================
    # commands from the web layer
    # ==================================================================
    def process_commands(self) -> None:
        for cmd in self.state.drain_commands():
            t = cmd.get("type")
            try:
                if t == "set_live":
                    self.set_live(bool(cmd.get("value")))
                elif t == "set_keystrokes":
                    self.mapper.enabled = bool(cmd.get("value")) and not self.args.dry_run
                elif t == "set_binding":
                    self.bindings.set(cmd["gesture"], cmd["control"])
                elif t == "record_start":
                    self.start_recording(cmd["name"], cmd["gtype"])
                elif t == "record_stop":
                    self.stop_recording()
                elif t == "record_cancel":
                    self.cancel_recording()
                elif t == "delete_sample":
                    self.delete_sample(cmd["name"])
                elif t == "reset_stats":
                    self.reset_counters()
                elif t == "reload_config":
                    self.reload_config()
            except Exception as exc:                  # noqa: BLE001
                print("[command] {} failed: {}".format(t, exc))

    # ==================================================================
    # recognizer plumbing
    # ==================================================================
    def handle_segment(self, seg) -> None:
        builtin_name = self.builtin.classify_swipe(seg)
        if builtin_name is not None:
            self.handle_builtin_trigger(builtin_name)

        if seg.rejected:
            if self.verbose:
                print("[traj] segment discarded: {} (frames={} path={:.2f} "
                      "peak_speed={:.3f} close={})".format(
                          seg.rejected, seg.frames, seg.path_length(),
                          seg.peak_speed, seg.close_reason))
            return

        feature = self.traj.feature_from_segment(seg)
        res = self.traj.classify(feature)
        if self.verbose:
            print(res.describe() + "  | frames={} dur={:.2f}s peak_speed={:.3f} "
                  "close={}".format(seg.frames, seg.duration, seg.peak_speed,
                                    seg.close_reason))
        if res.accepted:
            action = self.mapper.trigger(res.name, "trajectory", res.distance, res.threshold)
            self.note_event(res.name, "traj", action)

    def handle_pose_event(self, ev) -> None:
        res = ev.result
        if self.verbose:
            print("[pose] transition {} -> {} | {}".format(
                ev.previous, ev.name, res.describe()))
        if self._is_laser(ev.name):
            return
        action = self.mapper.trigger(ev.name, "pose", res.distance, res.threshold)
        self.note_event(ev.name, "pose", action)

    def handle_builtin_trigger(self, name: str) -> None:
        if self._is_laser(name):
            return
        action = self.mapper.trigger(name, "builtin")
        self.note_event(name, "builtin", action)

    def _is_laser(self, gesture: str) -> bool:
        return controls_mod.kind_for(self.bindings.get(gesture)) == controls_mod.KIND_LASER

    def update_laser(self, hand, frame_w: int, frame_h: int) -> None:
        laser_gestures = set(self.bindings.gesture_for_control("laser_pointer"))
        active = (
            (self.builtin.stable_pose in laser_gestures) if self.builtin.stable_pose else False
        ) or (
            (self.pose.stable_pose in laser_gestures) if self.pose.stable_pose else False
        )
        self.laser.update(hand, active, frame_w, frame_h)

    def note_event(self, name: str, source: str, action) -> None:
        mark = "FIRED" if action.fired else "COOLDOWN"
        self.last_event_text = "{} [{}] {} - {}".format(name, source, mark, action.detail)
        self.last_event_time = time.time()
        print("[action] {}".format(self.last_event_text))

    # ==================================================================
    # session controls (formerly keyboard shortcuts)
    # ==================================================================
    def set_live(self, value: bool) -> None:
        if value == self.live:
            return
        self.live = value
        if self.live:
            self._live_since = time.time()
            self.segmenter.reset()
            self.pose.reset_state()
            self.builtin.reset_state()
            self.laser.reset()
            self.mapper.reset_cooldown()
            print("[mode] LIVE ON")
        else:
            if self._live_since is not None:
                self.live_seconds += time.time() - self._live_since
                self._live_since = None
            print("[mode] LIVE OFF ({:.1f}s total)".format(self.live_seconds))

    def current_live_seconds(self) -> float:
        extra = time.time() - self._live_since if self._live_since else 0.0
        return self.live_seconds + extra

    def reset_counters(self) -> None:
        self.mapper.history.clear()
        self.live_seconds = 0.0
        self._live_since = time.time() if self.live else None
        print("[stats] counters reset -- validation window starts now")

    def reload_config(self) -> None:
        self.cfg = Config.load()
        self.traj = TrajectoryRecognizer(self.cfg, self.store)
        self.pose = PoseRecognizer(self.cfg, self.store)
        self.builtin = BuiltinGestureDetector(self.cfg)
        self.laser = LaserPointerController(self.cfg)
        self.segmenter = MotionSegmenter(self.cfg)
        enabled = self.mapper.enabled
        self.mapper = ActionMapper(self.cfg, self.bindings, dry_run=self.args.dry_run)
        self.mapper.enabled = enabled and not self.args.dry_run
        self.verbose = bool(self.cfg.logging["verbose_scores"])
        print("[config] reloaded thresholds.yaml")

    def delete_sample(self, name: str) -> None:
        if self.store.delete_last_sample(name):
            self.traj.recalibrate()
            self.pose.recalibrate()

    # ==================================================================
    # recording
    # ==================================================================
    def start_recording(self, name: str, gtype: str) -> None:
        existing = self.store.gestures.get(name)
        if existing and existing.type != gtype:
            raise ValueError(
                "'{}' already exists as {}; delete it first to change type".format(
                    name, existing.type
                )
            )
        self.record_name = name
        self.record_type = gtype
        self.record_points, self.record_scales, self.record_curls = [], [], []
        self.record_times = []
        self.recording = True
        countdown = float(self.cfg.recording["countdown_seconds"])
        self.countdown_until = time.time() + countdown
        print("[record] '{}' ({}) -- get ready, {:.0f}s".format(name, gtype, countdown))

    def cancel_recording(self) -> None:
        self.recording = False
        self.record_name = self.record_type = None

    def stop_recording(self) -> None:
        self.recording = False
        name, gtype = self.record_name, self.record_type
        self.record_name = self.record_type = None
        if not name:
            return
        if time.time() < self.countdown_until:
            print("  cancelled during the get-ready countdown; nothing captured.")
            return

        try:
            if gtype == TRAJECTORY:
                min_frames = int(self.cfg.segmenter["min_segment_frames"])
                if len(self.record_points) < min_frames:
                    print("  discarded: only {} frames with a hand visible "
                          "(need {}).".format(len(self.record_points), min_frames))
                    return
                pts, path_len, note = self.extract_trajectory_sample()
                if pts is None:
                    print("  discarded: {}".format(note))
                    return
                print("  {}".format(note))
                feature = self.traj.feature_from_points(pts)
                meta = {
                    "frames": len(pts),
                    "path_length": round(path_len, 4),
                    "captured_frames": len(self.record_points),
                    "segmentation": note,
                }
            else:
                if not self.record_curls:
                    print("  discarded: no hand was visible during the capture.")
                    return
                feature = self.pose.feature_from_recording(self.record_curls)
                meta = {
                    "frames": len(self.record_curls),
                    "curls": [round(float(v), 4) for v in feature],
                }

            g = self.store.add_sample(name, gtype, feature, meta)
        except ValueError as exc:
            print("  NOT SAVED: {}".format(exc))
            return

        self.traj.recalibrate()
        self.pose.recalibrate()
        self.bindings.ensure(name)
        print("  saved '{}' ({}) -- now {} sample(s).".format(name, gtype, g.count))

    def extract_trajectory_sample(self):
        """Turn a raw capture into a template, trimmed exactly like a live segment.

        Replays the captured wrist stream through a fresh MotionSegmenter --
        the same class, same thresholds, that produces segments in LIVE mode.
        Takes the FIRST valid segment, not the longest: you perform the
        gesture first, then reach for the stop button.

        Returns (points_in_hand_widths | None, path_length, note).
        """
        min_path = float(self.cfg.segmenter["min_segment_path_length"])
        raw = np.array(self.record_points, dtype=np.float64)
        mean_scale = max(float(np.mean(self.record_scales)), 1e-6)

        def as_hand_widths(a, scales):
            return a / max(float(np.mean(scales)), 1e-6)

        if bool(self.cfg.recording["resegment_trajectory"]):
            seg = MotionSegmenter(self.cfg)
            found = []
            for p, sc, t in zip(self.record_points, self.record_scales,
                                self.record_times):
                st = seg.update(p, sc, t)
                if st.segment is not None:
                    found.append(st.segment)
            tail = self.record_times[-1] if self.record_times else 0.0
            for i in range(int(self.cfg.segmenter["hand_lost_grace_frames"]) + 2):
                st = seg.update(None, 1.0, tail + (i + 1) / 30.0)
                if st.segment is not None:
                    found.append(st.segment)

            valid = [s for s in found if not s.rejected]
            if valid:
                s = valid[0]
                note = (
                    "re-segmented: kept {}/{} frames, path {:.2f} hand-widths"
                    .format(s.frames, len(raw), s.path_length())
                )
                if len(valid) > 1:
                    note += ("  [{} motion bursts found; used the first.]"
                             .format(len(valid)))
                return s.normalized_points(), s.path_length(), note
            if found:
                return None, 0.0, (
                    "the capture held {} motion burst(s) but all failed the "
                    "minimum gates ({}). Swipe further or faster -- or, if you "
                    "meant a hand shape, record it as type 'pose' instead.".format(
                        len(found), found[0].rejected)
                )

        pts = as_hand_widths(raw, self.record_scales)
        path_len = float(np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1)))
        if path_len < min_path:
            return None, path_len, (
                "hand barely moved (path {:.2f} < {:.2f} hand-widths). If you "
                "meant a hand shape, record it as type 'pose' instead."
                .format(path_len, min_path)
            )
        return pts, path_len, "used the raw capture ({} frames, path {:.2f})".format(
            len(pts), path_len)

    # ==================================================================
    # status / logging / drawing
    # ==================================================================
    def gesture_rows(self) -> List[dict]:
        rows = []
        for name in BUILTIN_GESTURES:
            rows.append({
                "name": name, "type": BUILTIN_TYPE.get(name, "pose"),
                "builtin": True, "samples": 0, "threshold_source": "built-in",
                "control": self.bindings.get(name),
            })
        for name, g in sorted(self.store.gestures.items()):
            rec = self.traj if g.type == TRAJECTORY else self.pose
            rows.append({
                "name": name, "type": g.type, "builtin": False,
                "samples": g.count, "threshold_source": rec.threshold_source(name),
                "control": self.bindings.get(name),
            })
        return rows

    def publish_status(self) -> None:
        recording_frames = 0
        if self.recording:
            recording_frames = (
                len(self.record_points) if self.record_type == TRAJECTORY
                else len(self.record_curls)
            )
        rep = ""
        if self.store.gestures:
            rep = distance_report(self.store, self.traj, self.pose)

        self.state.publish_status({
            "live": self.live,
            "keystrokes_enabled": self.mapper.enabled,
            "fps": round(self.fps, 1),
            "hand_present": self._last_hand_present,
            "cooldown_remaining": round(self.mapper.cooldown_remaining(), 2),
            "cooldown_total": self.mapper.cooldown,
            "last_event": self.last_event_text,
            "recording": self.recording,
            "record_name": self.record_name,
            "record_type": self.record_type,
            "record_countdown": max(0.0, self.countdown_until - time.time()) if self.recording else 0.0,
            "record_captured": recording_frames,
            "gestures": self.gesture_rows(),
            "bindings": self.bindings.all(),
            "distance_report": rep,
        })

    def publish_frame(self, frame) -> None:
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
        if ok:
            self.state.publish_frame(buf.tobytes())

    def log_frame(self, hand, status) -> None:
        pr = self.pose.last_result
        self.frames_log.write(
            hand=int(hand.present),
            wrist_x="{:.5f}".format(hand.wrist_xy[0]) if hand.present else "",
            wrist_y="{:.5f}".format(hand.wrist_xy[1]) if hand.present else "",
            hand_scale="{:.5f}".format(hand.hand_scale) if hand.present else "",
            speed="{:.5f}".format(status.speed),
            seg_state=status.state.value,
            seg_buffered=status.buffered_frames,
            curls=(
                "|".join("{:.3f}".format(v) for v in hand.curls)
                if hand.present else ""
            ),
            pose_best=(pr.name if pr else ""),
            pose_distance=("{:.4f}".format(pr.distance) if pr else ""),
            pose_stable=(self.pose.stable_pose or ""),
        )

    def tick_fps(self) -> None:
        self._fps_n += 1
        dt = time.time() - self._fps_t
        if dt >= 0.5:
            self.fps = self._fps_n / dt
            self._fps_n = 0
            self._fps_t = time.time()

    def annotate(self, frame, hand, status) -> None:
        """Draw the hand skeleton + a small HUD onto the frame for the MJPEG preview."""
        self._last_hand_present = hand.present
        h, w = frame.shape[:2]

        if hand.present:
            for a, b in HAND_CONNECTIONS:
                cv2.line(frame, tuple(hand.pixels[a]), tuple(hand.pixels[b]),
                         (90, 90, 90), 2)
            for i, (px, py) in enumerate(hand.pixels):
                colour = AMBER if i == 0 else GREEN
                cv2.circle(frame, (int(px), int(py)), 4, colour, -1)

        cv2.rectangle(frame, (0, 0), (w, 84), (24, 24, 24), -1)
        if self.recording:
            mode, colour = "RECORDING '{}' [{}]".format(
                self.record_name, self.record_type), RED
        elif self.live:
            mode, colour = "LIVE", GREEN
        else:
            mode, colour = "IDLE -- use the web control panel", GREY
        self.text(frame, mode, 12, 26, colour, 0.7, 2)
        self.text(frame, "{:.0f} fps | hand {}".format(
            self.fps, "yes" if hand.present else "NO"), 12, 50, GREY)

        seg_colour = AMBER if status.state is MotionState.IN_MOTION else GREY
        stable = self.builtin.stable_pose or self.pose.stable_pose or "-"
        self.text(frame, "speed {:.3f} {}  |  pose: {}".format(
            status.speed, status.state.value, stable), 12, 72, seg_colour)

    @staticmethod
    def text(img, s, x, y, colour=WHITE, scale=0.55, thick=1) -> None:
        cv2.putText(img, s, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0),
                    thick + 2, cv2.LINE_AA)
        cv2.putText(img, s, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, colour,
                    thick, cv2.LINE_AA)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true",
                    help="recognize and log but never send keystrokes")
    ap.add_argument("--log-frames", action="store_true",
                    help="write a per-frame CSV trace to logs/frames.csv")
    ap.add_argument("--session-label", default="",
                    help="tag rows in logs/events.csv, e.g. 'talking'")
    ap.add_argument("--stats", action="store_true",
                    help="print the distance report and exit (no camera)")
    ap.add_argument("--fps", type=int, default=0,
                    help="cap processing to N frames/sec (e.g. 18). 0 = "
                         "uncapped. Overrides camera.process_fps_cap.")
    ap.add_argument("--camera", default=None,
                    help="camera source, overriding camera.source in the "
                         "config: a device index (0, 1, 2) or a stream URL")
    ap.add_argument("--host", default=None, help="override web.host")
    ap.add_argument("--port", type=int, default=None, help="override web.port")
    ap.add_argument("--no-browser", action="store_true",
                    help="don't auto-open the control panel in a browser")
    args = ap.parse_args()

    if args.stats:
        cfg = Config.load()
        store = TemplateStore(STORE_PATH).load()
        traj = TrajectoryRecognizer(cfg, store)
        pose = PoseRecognizer(cfg, store)
        print("\nRecorded gestures:")
        for line in store.summary_lines():
            print("  " + line)
        print(distance_report(store, traj, pose))
        return 0

    return App(args).run()


if __name__ == "__main__":
    raise SystemExit(main())
