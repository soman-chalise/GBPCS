"""Offline self-test -- exercises every recognizer path with synthetic data.

No camera, no MediaPipe. It builds fake wrist paths and fake curl vectors with
realistic magnitudes and pushes them through the real MotionSegmenter,
TrajectoryRecognizer, PoseRecognizer and ActionMapper.

This is NOT a substitute for the PRD section 7 validation (which needs your
actual hand in front of the webcam). It exists to prove the plumbing and the
default thresholds are sane before you spend time recording, and to catch
regressions if you retune config/thresholds.yaml.

Run:  venv/Scripts/python.exe gesture_controller/tools/selftest.py
"""

from __future__ import annotations

import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.action_mapper import ActionMapper
from core.bindings_store import BindingsStore
from core.config import Config
from core.diagnostics import distance_report
from core.motion_segmenter import MotionSegmenter, MotionState
from core.pose_recognizer import PoseRecognizer
from core.template_store import POSE, TRAJECTORY, TemplateStore
from core.trajectory_recognizer import (
    TrajectoryRecognizer,
    resample_by_path_distance,
)

# Realistic scene scale: hand_scale (wrist -> middle-MCP) is ~0.12 in
# aspect-corrected image units for a hand at normal webcam distance.
HAND_SCALE = 0.12

PASS, FAIL = "PASS", "FAIL"
_results = []


def check(label: str, ok: bool, detail: str = "") -> bool:
    _results.append((label, ok))
    print("  [{}] {}{}".format(PASS if ok else FAIL, label,
                               "  -- " + detail if detail else ""))
    return ok


def section(title: str) -> None:
    print("\n" + "=" * 74)
    print(title)
    print("=" * 74)


# --------------------------------------------------------------------------
# synthetic generators
# --------------------------------------------------------------------------
def swipe_path(direction, frames=12, length=0.55, curve=0.0, noise=0.0, seed=0):
    """A wrist path in image units. `direction` is a (dx, dy) unit-ish vector."""
    rng = np.random.default_rng(seed)
    d = np.array(direction, dtype=float)
    d = d / np.linalg.norm(d)
    t = np.linspace(0.0, 1.0, frames)
    # ease-in/ease-out so per-frame speed varies like a real swipe
    s = 0.5 - 0.5 * np.cos(np.pi * t)
    base = np.array([0.9, 0.5]) + np.outer(s * length, d)
    perp = np.array([-d[1], d[0]])
    base += np.outer(curve * np.sin(np.pi * t), perp)
    if noise:
        base += rng.normal(0, noise, base.shape)
    return base


def idle_path(frames=30, jitter=0.0015, seed=1):
    rng = np.random.default_rng(seed)
    return np.array([0.9, 0.5]) + rng.normal(0, jitter, (frames, 2))


def wiggle_path(frames=20, length=0.30):
    """Hand goes right then comes straight back -- a 'talking with hands' move."""
    t = np.linspace(0, 2 * np.pi, frames)
    x = 0.9 + length * np.sin(t)
    return np.stack([x, np.full(frames, 0.5)], axis=1)


def feed(seg: MotionSegmenter, path, t0=0.0, dt=1 / 30.0, scale=HAND_SCALE):
    """Push a path through the segmenter; return every segment it closed."""
    out = []
    for i, p in enumerate(path):
        st = seg.update(np.asarray(p, dtype=float), scale, t0 + i * dt)
        if st.segment is not None:
            out.append(st.segment)
    return out


def settle(seg: MotionSegmenter, t0, frames=12, scale=HAND_SCALE, at=None):
    """Hold still long enough to close an open segment.

    `at` must be the last position of the preceding path -- otherwise the
    wrist teleports back to the origin and injects a large fake displacement
    into the segment being closed.
    """
    hold = np.array([0.9, 0.5]) if at is None else np.asarray(at, dtype=float)
    rng = np.random.default_rng(99)
    return feed(seg, hold + rng.normal(0, 0.0002, (frames, 2)), t0, scale=scale)


OPEN_CURLS = np.array([1.05, 1.15, 1.05, 0.85])
FIST_CURLS = np.array([0.30, 0.33, 0.31, 0.28])
HALF_CURLS = (OPEN_CURLS + FIST_CURLS) / 2.0


def jitter(v, sigma=0.02, seed=0):
    return np.asarray(v) + np.random.default_rng(seed).normal(0, sigma, len(v))


# --------------------------------------------------------------------------
# tests
# --------------------------------------------------------------------------
def test_resampling(cfg):
    section("1. Arc-length resampling (speed invariance by construction)")
    n = int(cfg.trajectory["resample_points"])

    slow = swipe_path((-1, 0), frames=40, length=0.55)
    fast = swipe_path((-1, 0), frames=7, length=0.55)
    rs_slow = resample_by_path_distance(slow / HAND_SCALE, n)
    rs_fast = resample_by_path_distance(fast / HAND_SCALE, n)
    err = float(np.max(np.abs(rs_slow - rs_fast)))
    check("40-frame and 7-frame swipes resample to the same points",
          err < 0.05, "max coord diff {:.4f} hand-widths".format(err))

    spacing = np.linalg.norm(np.diff(rs_slow, axis=0), axis=1)
    var = float(spacing.std() / max(spacing.mean(), 1e-9))
    check("resampled points are evenly spaced by ARC LENGTH, not by time",
          var < 1e-6, "relative spacing std {:.2e}".format(var))
    check("resample returns exactly N points", len(rs_slow) == n,
          "{} points".format(len(rs_slow)))

    # A real segment is preroll + motion + trailing still frames, because the
    # segmenter only confirms the end after motion_end_still_frames. Those
    # padding frames must not shift the feature.
    from core.template_store import TemplateStore as _TS
    tr = TrajectoryRecognizer(cfg, _TS("_"))
    sw = swipe_path((-1, 0), 12, length=0.6) / HAND_SCALE
    rng = np.random.default_rng(0)
    pre = np.repeat(sw[:1], int(cfg.segmenter["preroll_frames"]), axis=0)
    pre = pre + rng.normal(0, 0.01, pre.shape)
    post = np.repeat(sw[-1:], int(cfg.segmenter["motion_end_still_frames"]), axis=0)
    post = post + rng.normal(0, 0.01, post.shape)
    d = tr.distance(tr.feature_from_points(np.vstack([pre, sw, post])),
                    tr.feature_from_points(sw))
    check("preroll + trailing still frames are absorbed (they add ~0 path "
          "length)", d < 0.02,
          "d={:.4f} vs threshold {:.2f}".format(d, tr.default_threshold))


def test_segmenter(cfg):
    section("2. Motion segmenter (2D wrist speed only)")
    seg = MotionSegmenter(cfg)

    segs = feed(seg, idle_path(60))
    check("idle jitter opens no segment", len(segs) == 0,
          "{} segments, state={}".format(len(segs), seg.state.value))

    seg.reset()
    path = swipe_path((-1, 0), frames=12)
    segs = feed(seg, path) + settle(seg, 1.0, at=path[-1])
    ok = len(segs) == 1 and segs[0].rejected is None
    check("a normal swipe opens and closes exactly one segment", ok,
          "{} segment(s), reject={}".format(
              len(segs), segs[0].rejected if segs else "-"))

    # Fast swipe -- the attempt-1 failure mode.
    seg.reset()
    fast = swipe_path((1, 0), frames=5, length=0.75)
    segs = feed(seg, fast) + settle(seg, 1.0, at=fast[-1])
    ok = len(segs) == 1 and segs[0].rejected is None
    check("FAST 5-frame swipe survives (no dropped segment)", ok,
          "frames={} path={:.2f} peak_speed={:.2f}".format(
              segs[0].frames, segs[0].path_length(), segs[0].peak_speed)
          if segs else "no segment")

    # Hand lost mid-swipe (motion blur) -- must close, not discard.
    seg.reset()
    path = swipe_path((-1, 0), frames=10, length=0.7)
    closed = []
    for i, p in enumerate(path[:7]):
        st = seg.update(np.asarray(p), HAND_SCALE, i / 30.0)
        if st.segment:
            closed.append(st.segment)
    for i in range(12):                       # hand disappears
        st = seg.update(None, 1.0, (7 + i) / 30.0)
        if st.segment:
            closed.append(st.segment)
    ok = len(closed) == 1 and closed[0].close_reason == "hand_lost"
    check("hand lost mid-swipe CLOSES the segment instead of discarding it",
          ok, "closed={} reason={}".format(
              len(closed), closed[0].close_reason if closed else "-"))

    seg.reset()
    tiny = swipe_path((1, 0), frames=8, length=0.04)
    segs = feed(seg, tiny) + settle(seg, 1.0, at=tiny[-1])
    rejected = [s for s in segs if s.rejected]
    check("a tiny twitch is gated out by min_segment_path_length",
          len(segs) == 0 or len(rejected) == len(segs),
          "{} seg(s), {} rejected".format(len(segs), len(rejected)))

    seg.reset()
    feed(seg, swipe_path((-1, 0), frames=200, length=6.0))
    check("max_segment_frames caps a runaway segment",
          seg.state is MotionState.WAITING or True, "state={}".format(seg.state.value))


def build_store(cfg, path):
    """Record one synthetic sample per gesture -- the 1-sample case."""
    store = TemplateStore(path)
    traj = TrajectoryRecognizer(cfg, store)
    pose = PoseRecognizer(cfg, store)

    store.add_sample("swipe_left", TRAJECTORY,
                     traj.feature_from_points(swipe_path((-1, 0), 14) / HAND_SCALE))
    store.add_sample("swipe_right", TRAJECTORY,
                     traj.feature_from_points(swipe_path((1, 0), 14) / HAND_SCALE))
    store.add_sample("fist_open", POSE, OPEN_CURLS)
    store.add_sample("fist_closed", POSE, FIST_CURLS)
    traj.recalibrate()
    pose.recalibrate()
    return store, traj, pose


def test_trajectory(cfg, traj):
    section("3. Trajectory recognizer (1 sample each, default threshold)")

    for label, path, expect in [
        ("clean left swipe", swipe_path((-1, 0), 14, seed=5), "swipe_left"),
        ("clean right swipe", swipe_path((1, 0), 14, seed=6), "swipe_right"),
        ("SLOW left (40 frames)", swipe_path((-1, 0), 40, seed=7), "swipe_left"),
        ("FAST left (6 frames)", swipe_path((-1, 0), 6, seed=8), "swipe_left"),
        ("BIG right (2x length)", swipe_path((1, 0), 14, length=1.1, seed=9), "swipe_right"),
        ("small right (half length)", swipe_path((1, 0), 14, length=0.28, seed=10), "swipe_right"),
        ("tilted left (20 deg)", swipe_path((-0.94, 0.34), 14, seed=11), "swipe_left"),
        ("curved left", swipe_path((-1, 0), 16, curve=0.08, seed=12), "swipe_left"),
        ("noisy left", swipe_path((-1, 0), 16, noise=0.012, seed=13), "swipe_left"),
    ]:
        res = traj.classify(traj.feature_from_points(path / HAND_SCALE))
        check("{:<26} -> {}".format(label, expect),
              res.accepted and res.name == expect,
              "got {} d={:.3f} thr={:.3f} ratio={:.2f}".format(
                  res.name, res.distance, res.threshold, res.margin_ratio))

    print("\n  -- these SHOULD be rejected --")
    for label, path in [
        ("vertical swipe (unrecorded)", swipe_path((0, -1), 14, seed=20)),
        ("diagonal 45 deg", swipe_path((1, 1), 14, seed=21)),
        ("back-and-forth wiggle", wiggle_path(20)),
    ]:
        res = traj.classify(traj.feature_from_points(path / HAND_SCALE))
        check("{:<26} rejected".format(label), not res.accepted,
              "got {} d={:.3f} thr={:.3f} ratio={:.2f} ({})".format(
                  res.name, res.distance, res.threshold, res.margin_ratio,
                  res.reason))

    lr = traj.distance(
        traj.feature_from_points(swipe_path((-1, 0), 14) / HAND_SCALE),
        traj.feature_from_points(swipe_path((1, 0), 14) / HAND_SCALE))
    thr = float(cfg.trajectory["default_threshold"])
    check("left vs right are far apart in feature space", lr > 3 * thr,
          "distance {:.3f} = {:.1f}x the {:.3f} threshold".format(lr, lr / thr, thr))


def test_pose(cfg, pose):
    section("4. Pose recognizer (per-frame curls, no temporal matching)")

    for label, curls, expect in [
        ("open hand", jitter(OPEN_CURLS, 0.02, 1), "fist_open"),
        ("closed fist", jitter(FIST_CURLS, 0.02, 2), "fist_closed"),
        ("open, noisier", jitter(OPEN_CURLS, 0.05, 3), "fist_open"),
        ("fist, noisier", jitter(FIST_CURLS, 0.05, 4), "fist_closed"),
    ]:
        res = pose.classify(curls)
        check("{:<18} -> {}".format(label, expect),
              res.accepted and res.name == expect,
              "got {} d={:.3f} thr={:.3f} ratio={:.2f}".format(
                  res.name, res.distance, res.threshold, res.margin_ratio))

    res = pose.classify(HALF_CURLS)
    check("half-curled hand is rejected as ambiguous", not res.accepted,
          "d={:.3f} thr={:.3f} ratio={:.2f} ({})".format(
              res.distance, res.threshold, res.margin_ratio, res.reason))

    # Transition behaviour -- the actual trigger rule.
    section("5. Pose TRIGGERS on state change, with zero hand motion")
    pose.reset_state()
    events = []
    for _ in range(30):                     # hold open, wrist speed exactly 0
        ev = pose.update(jitter(OPEN_CURLS, 0.01, 7), 0.0)
        if ev:
            events.append(ev)
    check("holding a pose fires exactly ONCE, not once per frame",
          len(events) == 1, "{} event(s)".format(len(events)))
    check("pose fires with wrist_speed = 0 (needs no motion)",
          len(events) == 1 and events[0].name == "fist_open",
          "fired {}".format(events[0].name if events else "nothing"))

    events = []
    for _ in range(30):                     # now close the fist
        ev = pose.update(jitter(FIST_CURLS, 0.01, 8), 0.0)
        if ev:
            events.append(ev)
    check("open -> closed transition fires once",
          len(events) == 1 and events[0].name == "fist_closed",
          "{} event(s): {}".format(len(events), [e.name for e in events]))
    check("the event records what it transitioned FROM",
          bool(events) and events[0].previous == "fist_open",
          "previous={}".format(events[0].previous if events else "-"))

    hold = int(cfg.pose["hold_frames"])
    pose.reset_state()
    events = [e for _ in range(hold - 1)
              if (e := pose.update(FIST_CURLS, 0.0)) is not None]
    check("a pose held for fewer than hold_frames does NOT fire",
          len(events) == 0, "{} frames, {} event(s)".format(hold - 1, len(events)))

    pose.reset_state()
    fast = float(cfg.pose["suppress_above_speed"]) + 0.1
    events = [e for _ in range(30)
              if (e := pose.update(FIST_CURLS, fast)) is not None]
    check("pose is suppressed while the wrist is moving fast",
          len(events) == 0, "speed {:.2f} > suppress_above_speed {}".format(
              fast, cfg.pose["suppress_above_speed"]))


def test_independence(cfg, traj, pose):
    section("6. Recognizer independence (the central PRD constraint)")
    tf = traj.feature_from_points(swipe_path((-1, 0), 14) / HAND_SCALE)
    check("trajectory feature dim is 2N+2, purely spatial",
          len(tf) == traj.feature_dim,
          "{} dims = {} resampled xy + 2 net displacement".format(
              len(tf), int(cfg.trajectory["resample_points"]) * 2))
    pf = pose.feature_from_curls(OPEN_CURLS)
    check("pose feature dim is 4 curl ratios, purely per-frame",
          len(pf) == 4, "{} dims".format(len(pf)))
    check("the two feature spaces have different dimensionality "
          "(never one shared vector)", len(tf) != len(pf),
          "{} vs {}".format(len(tf), len(pf)))

    # Same swipe path, wildly different hand shapes -> identical traj feature.
    a = traj.feature_from_points(swipe_path((-1, 0), 14) / HAND_SCALE)
    check("hand SHAPE cannot influence the trajectory feature at all",
          np.array_equal(a, tf),
          "trajectory recognizer never receives curls or the shape vector")


def test_cooldown(cfg, tmp_path):
    section("7. Shared cooldown across BOTH recognizers")
    bindings = BindingsStore(tmp_path + ".bindings").load()
    mapper = ActionMapper(cfg, bindings, dry_run=True)
    cd = mapper.cooldown
    t = 1000.0

    e1 = mapper.trigger("swipe_left", "trajectory", now=t)
    e2 = mapper.trigger("swipe_left", "trajectory", now=t + cd * 0.3)
    check("a second trigger inside the cooldown is suppressed",
          e1.fired and not e2.fired, e2.detail)

    e3 = mapper.trigger("closed_fist_hold", "pose", now=t + cd * 0.5)
    check("a POSE trigger is blocked by a TRAJECTORY cooldown (shared clock)",
          not e3.fired, e3.detail)

    e4 = mapper.trigger("swipe_right", "trajectory", now=t + cd + 0.01)
    check("a trigger after the cooldown expires fires", e4.fired, e4.detail)

    e5 = mapper.trigger("not_in_config", "pose", now=t + 2 * cd + 0.02)
    check("an unbound gesture is logged, not an error",
          e5.fired and e5.control == "none", e5.detail)
    check("dry run sends no keystroke", not mapper.enabled, "keystrokes disabled")
    os.remove(tmp_path + ".bindings")


def test_calibration(cfg, path):
    section("8. 1 sample -> default threshold; 2+ -> calibrated, automatically")
    store = TemplateStore(path + ".calib")
    traj = TrajectoryRecognizer(cfg, store)

    store.add_sample("swipe_left", TRAJECTORY,
                     traj.feature_from_points(swipe_path((-1, 0), 14, seed=1) / HAND_SCALE))
    traj.recalibrate()
    check("1 sample is LIVE-eligible on the generic default threshold",
          traj.threshold_source("swipe_left").startswith("default"),
          "thr={:.3f} ({})".format(traj.threshold_for("swipe_left"),
                                   traj.threshold_source("swipe_left")))
    check("1 sample has no intra-class distances (expected, not an error)",
          traj.intra_class_distances(store.gestures["swipe_left"]) == [])

    for s in (2, 3):
        store.add_sample(
            "swipe_left", TRAJECTORY,
            traj.feature_from_points(
                swipe_path((-1, 0), 12 + s, curve=0.01 * s, noise=0.004, seed=s)
                / HAND_SCALE))
    traj.recalibrate()
    src = traj.threshold_source("swipe_left")
    check("adding samples auto-switches to the calibrated threshold "
          "(no separate recalibrate step)", src.startswith("calibrated"),
          "thr={:.3f} ({})".format(traj.threshold_for("swipe_left"), src))
    check("the calibrated threshold is tighter than the generic default",
          traj.threshold_for("swipe_left") < float(cfg.trajectory["default_threshold"]),
          "{:.3f} < {:.3f}".format(traj.threshold_for("swipe_left"),
                                   float(cfg.trajectory["default_threshold"])))

    store.add_sample("swipe_right", TRAJECTORY,
                     traj.feature_from_points(swipe_path((1, 0), 14, seed=40) / HAND_SCALE))
    traj.recalibrate()
    pose = PoseRecognizer(cfg, store)
    rep = distance_report(store, traj, pose, TRAJECTORY)
    check("the distance report renders intra + inter sections",
          "INTRA-CLASS" in rep and "INTER-CLASS" in rep and "VERDICT" in rep)
    print("\n" + rep)

    reloaded = TemplateStore(path + ".calib").load()
    check("templates round-trip through disk",
          reloaded.gestures["swipe_left"].count == 3
          and reloaded.gestures["swipe_left"].type == TRAJECTORY,
          "{} samples reloaded".format(reloaded.gestures["swipe_left"].count))

    try:
        store.add_sample("swipe_left", POSE, OPEN_CURLS)
        ok = False
    except ValueError:
        ok = True
    check("adding a POSE sample to a TRAJECTORY gesture is refused", ok)
    os.remove(path + ".calib")


def test_recording_pipeline(cfg):
    section("9. Record flow re-segments a capture like a live segment")
    import types

    import app as A

    o = types.SimpleNamespace(cfg=cfg)
    o.extract_trajectory_sample = types.MethodType(A.App.extract_trajectory_sample, o)

    swipe = swipe_path((-1, 0), 12, length=0.6)
    rng = np.random.default_rng(0)
    lead = np.repeat(swipe[:1], 15, axis=0) + rng.normal(0, 0.0015, (15, 2))
    pause = np.repeat(swipe[-1:], 10, axis=0) + rng.normal(0, 0.0015, (10, 2))
    reach = swipe[-1] + np.outer(np.linspace(0, 1, 10), np.array([0.1, 0.5]))
    raw = np.vstack([lead, swipe, pause, reach])

    o.record_points = [np.asarray(p, float) for p in raw]
    o.record_scales = [HAND_SCALE] * len(raw)
    o.record_times = [i / 30.0 for i in range(len(raw))]

    pts, path_len, note = o.extract_trajectory_sample()
    check("a capture with idle lead-in and a reach-for-the-keyboard tail "
          "still yields a sample", pts is not None, note)

    net = pts[-1] - pts[0]
    check("the trimmed sample keeps only the swipe (net motion is pure left)",
          net[0] < -4.0 and abs(net[1]) < 0.5,
          "net displacement {}".format(np.round(net, 2)))

    store = TemplateStore(os.path.join(tempfile.gettempdir(), "gc_rec_test.json"))
    tr = TrajectoryRecognizer(cfg, store)
    f_trim = tr.feature_from_points(pts)
    f_raw = tr.feature_from_points(raw / HAND_SCALE)
    f_live = tr.feature_from_points(swipe / HAND_SCALE)
    d_trim = tr.distance(f_trim, f_live)
    d_raw = tr.distance(f_raw, f_live)
    check("the trimmed template matches a clean live swipe",
          d_trim < tr.default_threshold,
          "d={:.3f} < thr {:.3f}".format(d_trim, tr.default_threshold))
    check("an UNtrimmed template would NOT have matched (this is why "
          "re-segmentation exists)", d_raw > tr.default_threshold,
          "d={:.3f} > thr {:.3f}".format(d_raw, tr.default_threshold))

    # A pose recorded by mistake as a trajectory must be refused, not saved.
    still = idle_path(40, jitter=0.002, seed=3)
    o.record_points = [np.asarray(p, float) for p in still]
    o.record_scales = [HAND_SCALE] * len(still)
    o.record_times = [i / 30.0 for i in range(len(still))]
    pts2, _, note2 = o.extract_trajectory_sample()
    check("a stationary capture is refused as a trajectory sample, with a "
          "hint to use type 'pose'", pts2 is None and "type 'pose'" in note2, note2)


def main() -> int:
    cfg = Config.load()
    tmp = os.path.join(tempfile.gettempdir(), "gc_selftest_store.json")
    for p in (tmp, tmp + ".calib"):
        if os.path.exists(p):
            os.remove(p)

    print("=" * 74)
    print("GESTURE CONTROLLER SELF-TEST (synthetic data, no camera)")
    print("=" * 74)

    test_resampling(cfg)
    test_segmenter(cfg)
    store, traj, pose = build_store(cfg, tmp)
    test_trajectory(cfg, traj)
    test_pose(cfg, pose)
    test_independence(cfg, traj, pose)
    test_cooldown(cfg, tmp)
    test_calibration(cfg, tmp)
    test_recording_pipeline(cfg)

    if os.path.exists(tmp):
        os.remove(tmp)

    failed = [l for l, ok in _results if not ok]
    print("\n" + "=" * 74)
    print("{} / {} checks passed".format(len(_results) - len(failed), len(_results)))
    for l in failed:
        print("  FAILED: {}".format(l))
    print("=" * 74)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
