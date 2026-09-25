# Design & Architecture

This document explains **how the gesture controller works internally**: the
pipeline, every algorithm used and why, and the reasoning behind each design
decision. `README.md` is the quick overview; `SETUP.md` is how to get it
running. This is the "why does it work this way" reference.

---

## 1. The problem this architecture solves

Two earlier prototypes (see `prd.md`, "Attempt 1") pushed every gesture
through **one shared feature vector** (hand shape + trajectory mixed together)
and **one matching algorithm** (DTW or resampled-nearest-neighbour). That
failed for a structural reason:

- A **swipe** (left/right) is a *trajectory* gesture — the signal is almost
  entirely in *where the wrist moved over time*. Hand shape during a swipe is
  noise.
- A **fist / thumbs-up / peace sign** is a *pose* gesture — the signal is
  entirely in *one frame's finger geometry*. There is no meaningful
  trajectory to align.

Mixing them meant every threshold/weighting change that helped one gesture
family hurt the other, and shape jitter (fingers wobbling while the hand held
still) was indistinguishable from real spatial motion.

**The fix, and the rule every module below obeys: two completely independent
pipelines, never merged.** One recognizes trajectories, one recognizes poses.
They share only the camera input upstream and the cooldown/action-firing
logic downstream.

A second axis was added on top of that (this build, v3): **built-in vs.
custom gestures** (see section 4). That is a *second*, orthogonal split —
independent of trajectory-vs-pose — and the two axes combine into four
possible pipelines (custom trajectory, custom pose, built-in trajectory,
built-in pose), all converging on one shared cooldown/binding/action layer.

---

## 2. End-to-end pipeline

```
Camera (webcam or phone stream)
   │
   ▼
core/frame_source.py         -- backend abstraction; threaded "newest frame only" draining
                                 (camera.threaded_stream, default true -- correct for a
                                 network/phone stream and the deployment rig; per-machine
                                 caveat for a local dev webcam in SETUP.md §9)
   │
   ▼
core/hand_tracker.py         -- MediaPipe GestureRecognizer (21 landmarks + pretrained
                                 static-gesture label), CLAHE lighting fix
   │
   ├── shape (63-dim, EMA-smoothed, wrist-relative, scale-normalized)
   ├── wrist_xy (2D, EMA-smoothed, SEPARATE filter from shape)
   ├── curls (4 or 5-dim, per-finger extension, UNsmoothed)
   └── pixels (21x2 raw image coords, for drawing + laser pointer)
   │
   ├─────────────────────────────┬───────────────────────────────┐
   ▼                             ▼                               ▼
core/motion_segmenter.py    core/pose_recognizer.py        core/builtin_gestures.py
(wrist_xy speed only ->     (curls, per-frame,              (shape + curls, per-frame
 WAITING/IN_MOTION state     nearest-centroid vs.            geometric rules, zero
 machine -> closed Segment)  YOUR recorded samples)          recording needed)
   │                             │                               │
   ▼                             │                               ├─ classify_swipe(segment)
core/trajectory_recognizer.py    │                               └─ update_pose(hand, speed)
(resampled path vs.              │                                       │
 YOUR recorded samples)          │                                       │
   │                             │                                       │
   └──────────────┬──────────────┴───────────────────────────────────────┘
                  ▼
         gesture NAME (+ source: "trajectory" | "pose" | "builtin")
                  │
                  ▼
      core/bindings_store.py  -- gesture NAME -> control NAME  (data/bindings.json)
                  │
                  ▼
      core/controls.py        -- control NAME -> effect (a keystroke, or "laser")
                  │
                  ▼
      core/action_mapper.py   -- SHARED cooldown gate, then fires the effect
                  │                          │
                  ▼                          ▼
           pyautogui.press()      core/laser_pointer.py (continuous, bypasses cooldown)
```

Everything above the `bindings_store` line only ever produces a **name**. No
recognizer, custom or built-in, knows what a keystroke is, what "next slide"
means, or that a web page exists. That separation is deliberate at every
layer, not just the top one — see section 7 for why.

---

## 3. Shared layer: hand tracking

**File:** `core/hand_tracker.py`. The only module that talks to MediaPipe.

- **Model:** MediaPipe Tasks `GestureRecognizer` (not plain `HandLandmarker`),
  `VIDEO` running mode, `num_hands=1`. It runs hand detection internally, so
  it still produces the same 21 (x, y, z) landmarks per hand per frame that
  everything below is built on — one model, one inference per frame, not
  two — and additionally ships a pretrained static-gesture classifier
  (`Closed_Fist`, `Open_Palm`, `Pointing_Up`, `Thumb_Down`, `Thumb_Up`,
  `Victory`, `ILoveYou`, `None`), exposed per-frame as
  `HandFrame.gesture_label`/`gesture_score` and gated by
  `builtin.gesture_min_confidence` (`canned_gesture_classifier_options.score_threshold`).
  Only `core/builtin_gestures.py` consumes this label — the custom
  recognizers (section 6) only ever see landmarks, never this classifier.
- **Lighting normalization:** CLAHE (Contrast-Limited Adaptive Histogram
  Equalization) on the L channel of LAB color space, before detection. Cheap,
  and meaningfully improves detection in dim rooms.
- **Aspect correction:** x is multiplied by the frame's aspect ratio before
  any geometry is computed, so a horizontal swipe and a vertical swipe are
  measured in the same units — otherwise a wide 16:9 frame would compress
  horizontal motion relative to vertical.
- **Two independent output streams**, each with its **own** EMA
  (exponential moving average) smoothing filter — this separation is the
  single most important invariant in the codebase:
  1. `shape` (63-dim): all 21 landmarks, made position-invariant (subtract
     the wrist landmark) and scale-invariant (divide by `hand_scale`, the
     wrist→middle-MCP distance — a proxy for "how big the hand looks", which
     stands in for distance-from-camera). `z` is downweighted ×0.3 because
     MediaPipe's depth estimate is noisy. Smoothed with `shape_smooth_alpha`
     (0.45).
  2. `wrist_xy` (2D): just the wrist position, same scale-normalized space.
     Smoothed with its **own** `trajectory_smooth_alpha` (0.45) — a separate
     filter instance, not a slice of the same one. If these two streams
     shared a filter, finger jitter would leak into the trajectory signal
     (exactly the Attempt 1 bug).
- **Curl ratios** (`curls`): for each finger, `‖tip − MCP‖ / hand_scale`.
  Extended ≈ 0.9–1.3, curled ≈ 0.2–0.5. Computed from the **raw**
  (unsmoothed) landmarks on purpose — the pose recognizer does its own
  temporal stabilization (`hold_frames`), so smoothing here would just add
  lag to what's supposed to be an instantaneous read. `pose.include_thumb`
  controls whether a 5th (thumb) dimension is included for the *custom* pose
  recognizer — the built-in detectors always compute their own thumb curl
  regardless of this flag (see section 5).
- **No ROI (region-of-interest) cropping.** Detection runs on the full frame
  every time. An earlier attempt cropped to a padded box around the previous
  frame's hand for speed; a fast swipe could leave that crop mid-motion, the
  hand was "lost", and the whole segment was discarded — exactly the
  gesture that most needed to be caught. Full-frame detection was measured at
  ~18ms of a 33ms frame budget; the webcam's own frame rate is the actual
  bottleneck, not detection.
- **EMA state is dropped the instant a frame has no hand** (`process()`
  calls `reset_smoothing()` before returning `present=False`). MediaPipe
  flickers present/absent for a single frame often enough in practice
  (visible in `logs/frames.csv`) that leaving the old EMA state in place was
  a real bug: the next real detection would blend the true (possibly far
  away) new position with the stale pre-loss value, producing a phantom
  multi-frame "glide" from the old position toward the new one. The motion
  segmenter read that glide as genuine swipe motion — so a swipe would fire
  based on where the hand *ended up relative to before it flickered*, not on
  an actual swiping motion — and the pose recognizers' `suppress_above_speed`
  gate read the same phantom speed spike as fast wrist motion and froze
  hold-tracking, making a held `thumbs_up`/fist unreliable. Resetting on loss
  means the first frame back is always the raw, undistorted reading.

---

## 4. Two gesture families × two provenances

|  | **Trajectory family** | **Pose family** |
|---|---|---|
| Signal | 2D wrist path over time | one frame's finger geometry |
| Needs the motion segmenter? | yes | no |
| **Custom** (you record it) | `core/trajectory_recognizer.py` | `core/pose_recognizer.py` |
| **Built-in** (ships pre-tuned) | `core/builtin_gestures.py` — swipe half | `core/builtin_gestures.py` — pose half |

**Custom** gestures are recognized by matching against samples *you*
recorded through the web panel (nearest-centroid / nearest-neighbour). They
adapt to your exact hand and motion, but need at least one recording.

**Built-in** gestures (`swipe_left`, `swipe_right`, `thumbs_up`, `gun_point`,
`peace_sign`, `closed_fist_hold`) are fixed geometric rules
tuned once, in `thresholds.yaml`'s `[builtin]` section, and need **no
recording at all** — they must work the moment the app starts. The tradeoff
is they're less personalized: they use population-reasonable thresholds, not
thresholds calibrated to your hand.

Both provenances run **every frame, concurrently** — a user can have a
custom gesture *and* all six built-ins active at once. They only ever
collide at the shared cooldown (section 8), which guarantees at most one
fires per physical gesture.

---

## 5. Motion segmenter — trajectory-only speed

**File:** `core/motion_segmenter.py`.

A state machine, `WAITING → IN_MOTION → (closed)`, driven by **one scalar**:

```
speed = ‖wrist_xy[t] − wrist_xy[t-1]‖ / hand_scale     (hand-widths per frame)
```

This is computed *only* from the 2D wrist stream — never from the 63-dim
shape vector. That is the direct fix for Attempt 1's bug, where finger
jitter (shape noise) was indistinguishable from spatial motion, so a
completely still hand could "open" a bogus segment.

- `idle_speed_threshold` (0.055): below this, the hand counts as "still".
- `motion_start_frames` (2): consecutive fast frames needed to confirm motion
  actually started (debounces a single noisy spike).
- `motion_end_still_frames` (6): consecutive still frames needed to confirm
  the motion ended.
- `preroll_frames` (4): a small ring buffer of frames from *before* motion
  was confirmed gets prepended to the segment, so the very start of a fast
  swipe isn't clipped by the 2-frame confirmation delay.
- `min_segment_frames` / `min_segment_path_length`: gates that reject
  segments too short or too small to be a deliberate gesture (jitter).
- `max_segment_frames` (90, ~3s): safety cap on a runaway "motion" reading.
- **Hand-lost handling:** if the hand disappears mid-swipe (motion blur) for
  up to `hand_lost_grace_frames` (6), the segmenter **closes and classifies**
  the segment instead of discarding it. A partial fast swipe still carries
  its direction — discarding it is exactly how Attempt 1 lost fast swipes to
  motion blur.

The segmenter's output (a closed `Segment`: the 2D path, its scale-per-frame,
timing, peak/mean speed, and why it closed) feeds **both** the custom
`TrajectoryRecognizer` and the built-in swipe classifier — they consume the
same `Segment` object independently.

Poses do **not** go through this state machine at all (see section 6) — they
fire on their own per-frame state transitions, because a pose gesture can
happen with zero wrist motion.

---

## 6. Custom recognizers (user-recorded)

### 6.1 Trajectory recognizer (`core/trajectory_recognizer.py`)

Input: a closed `Segment`'s 2D wrist path (hand-widths). Pipeline:

1. **Arc-length resampling** to a fixed `N=16` points — spaced evenly by
   *cumulative path distance*, not by frame index or time. This is what
   makes a slow-performed swipe and a fast-performed swipe produce the
   (nearly) same feature vector: speed invariance by construction, no
   dynamic time warping needed.
2. **Translate** so the first point is the origin (position invariance,
   direction preserved).
3. **Divide by total path length** (size invariance) — a clean swipe now
   ends near a unit vector pointing in its direction; a back-and-forth
   wiggle ends near the origin (its net displacement cancels out).
4. **Append net displacement** (`end − start`), weighted ×4
   (`net_displacement_weight`) — this is deliberately the dominant signal,
   since "did the hand end up meaningfully left or right of where it
   started" is the actual swipe signal.

**Classification:** nearest-neighbour (Euclidean distance, normalized by
`√dim` so the threshold's meaning doesn't shift if `resample_points`
changes) against every stored sample of every recorded trajectory gesture,
gated by:

- a **distance threshold** — `default_threshold` (0.42) when a gesture has
  only 1 sample; once it has ≥2, automatically switches to a **calibrated**
  threshold: `mean(intra-class pairwise distances) + k·std`, clamped to
  `[threshold_floor, threshold_ceiling]`. Recalculated on every new sample —
  no separate "recalibrate" step.
- a **margin-over-runner-up** check: the best match must beat the
  second-best by a ratio (`margin_ratio`, 0.80) or the match is rejected as
  ambiguous. With only one gesture recorded there's no runner-up, so this
  passes trivially and the distance threshold does all the rejection work.

### 6.2 Pose recognizer (`core/pose_recognizer.py`)

Input: the current frame's curl-ratio vector only (4 or 5 dims). No temporal
component, no resampling, no path.

**Classification:** distance to each recorded gesture's **centroid** (mean
of its samples), same threshold/margin gating philosophy as the trajectory
side, with its own default (0.30) / calibration constants.

**Triggering — this is the important part:** a pose does not fire every
frame it's held. A small state machine tracks:

```
candidate = best-matching gesture this frame (or None if rejected)
if candidate == previous candidate: run += 1  else: run = 1
if run < hold_frames: no event
if candidate == current "stable" pose: no event  (already fired for this)
else: STABLE POSE CHANGED -> fire an event, remember previous
```

So holding a fist fires once, not 30 times a second; releasing it (into
"open" or "no pose") fires the *next* pose once. `suppress_above_speed`
(0.20) freezes this transition tracking while the wrist is moving fast —
stops a swiping hand's mid-flight shape from being misread as a held pose —
but reads only a scalar speed value, **not** the segmenter's state, so a
genuinely stationary hand still triggers poses with zero motion.

---

## 7. Built-in recognizers (`core/builtin_gestures.py`)

Independent of `TemplateStore` entirely — no samples, no calibration, fixed
rules tuned once in `thresholds.yaml`'s `[builtin]` section. This file is
deliberately **not** built on top of the custom recognizers above (per an
explicit design decision): rule-based geometric checks are more predictable
and don't require the user to understand "1 sample vs. calibrated
threshold" just to get `swipe_right` working.

### 7.1 Built-in swipes — `classify_swipe(segment)`

Given the *same* closed `Segment` the custom trajectory recognizer sees:

```python
net = end_point - start_point
straightness = ‖net‖ / path_length        # 1.0 = perfectly straight, ~0 = wavy/back-and-forth
if path_length < swipe_min_path_length: reject      # too small, likely jitter
if straightness < swipe_straightness_min: reject     # too wavy to be a deliberate swipe
if |dx| < swipe_axis_dominance * |dy|: reject         # not clearly horizontal
return "swipe_right" if dx > 0 else "swipe_left"
```

No resampling, no template matching — just three geometric gates on the
segment's net displacement. Cheap and deterministic.

### 7.2 Built-in poses — pretrained classifier + one geometric fallback

`thumbs_up`, `closed_fist_hold` and `peace_sign` are read straight off
`hand.gesture_label` (section 3) — MediaPipe GestureRecognizer's pretrained
classifier, mapped `Thumb_Up → thumbs_up`, `Closed_Fist → closed_fist_hold`,
`Victory → peace_sign` (`ML_LABEL_TO_POSE` in `builtin_gestures.py`). This
replaced hand-tuned geometric rules (thumb-tip-vs-wrist-y, curl thresholds)
that misfired on wrist rotation — the original motivation for switching to a
pretrained model.

`gun_point` has no equivalent in that model's fixed vocabulary
(`Closed_Fist, Open_Palm, Pointing_Up, Thumb_Down, Thumb_Up, Victory,
ILoveYou, None`) and is the one pose still classified by a geometric rule, on
`hand.landmarks_rel` reshaped to `(21, 3)` (wrist-relative, scale-normalized,
raw/unsmoothed): index + thumb extended, middle/ring/pinky curled, **and**
index fingertip's relative z is closer to the camera than the wrist by
`gun_point_z_margin` (0.05) — the "pointing at the camera" check, using
MediaPipe's depth axis (smaller/more-negative z = closer to camera).
Extension/curl for this rule use per-finger thresholds (`finger_extended`
0.88, `finger_curled` 0.50, and a separate `thumb_extended`, since the
thumb's geometry differs from the other four fingers) — these three
constants exist purely for `gun_point` now.

`open_palm_hold` (all five fingers extended) shipped but never fired
reliably in practice and was removed rather than kept as a source of dead
weight in the control list.

**If a frame's geometry matches more than one rule simultaneously** (e.g.
mid-transition between two shapes), the frame is treated as **no match** —
rejected rather than guessed, the same philosophy as the custom recognizers'
margin-over-runner-up check.

Matches are gated through a small, self-contained hold/transition state
machine (`_HoldDebouncer`) — the same shape as `PoseRecognizer`'s (fire only
on a stable-state change, after `hold_frames` (5) consecutive frames), kept
as its own tiny class here rather than sharing code with
`pose_recognizer.py`, since it's only reused across the five built-in poses.
`suppress_above_speed` (0.20) freezes it during fast wrist motion, same
rationale as the custom recognizer.

**Tolerating a brief tracking dropout.** Both `_HoldDebouncer` (here) and
`PoseRecognizer` (6.2) used to fully reset — clearing the *stable* pose, not
just the in-progress hold count — on any single frame with no hand detected.
Since MediaPipe genuinely flickers present/absent for one frame at a time
fairly often, this made a held `thumbs_up`/`closed_fist_hold` unreliable: one
stray blip mid-hold threw away all accumulated `hold_frames` progress (and
forgot that the pose was already stable, so it would need to re-qualify from
scratch even though the user never released it). Both now tolerate up to
`hand_lost_grace_frames` (4) consecutive absent frames by freezing — pausing
the in-progress hold count without forgetting the stable pose — and only
fully reset past that, mirroring the segmenter's own `hand_lost_grace_frames`
philosophy (section 5) for the pose side.

---

## 8. Bindings, controls, and the action mapper

This is the layer that makes gestures **remappable without touching any
recognizer code** — the entire point of the web control panel.

- **`core/controls.py`** — a fixed registry of possible *effects*:
  `next_slide` (→), `previous_slide` (←), `start_presentation` (F5),
  `end_presentation` (Esc), `blank_screen` (b), `first_slide` (Home), and
  `laser_pointer` (special — see 8.1), plus `none` (recognized but does
  nothing). This is the list the web UI's per-gesture dropdown offers.
- **`core/bindings_store.py`** — persists **gesture name → control name** to
  `data/bindings.json` (atomic write, same pattern as `template_store.py`).
  Any gesture — built-in or custom — can be bound to any control. Built-in
  gestures get sensible defaults seeded on first run
  (`swipe_right→next_slide`, `swipe_left→previous_slide`,
  `thumbs_up→start_presentation`, `gun_point→laser_pointer`,
  `closed_fist_hold→first_slide`,
  `peace_sign→end_presentation`), but every one of those is freely
  reassignable from the web panel afterward — nothing is hard-locked in
  code.
- **`core/action_mapper.py`** — `ActionMapper.trigger(gesture, source)` is
  the single entry point every recognizer (custom or built-in) calls when it
  accepts a gesture. It:
  1. Checks a **shared cooldown clock** (`actions.cooldown_seconds`, 0.8s) —
     shared across *all four* pipelines, so a swipe and a fist-close
     performed in the same motion can't both fire. Whichever recognizer's
     event reaches `trigger()` first in a given frame wins; everything else
     within the cooldown window is suppressed and logged as such.
  2. Resolves `gesture → BindingsStore.get() → control name`.
  3. Resolves `control name → controls.py → effect` and either presses the
     keystroke (`pyautogui.press`/`hotkey`) or, for `laser_pointer`, does
     nothing here (see 8.1 — it's handled outside the cooldown-gated path).

### 8.1 The laser pointer is not a keystroke

Every other control is a **discrete** event: gesture recognized once →
keystroke sent once. The laser pointer is **continuous**: it needs to track
the fingertip every frame for as long as the pose is held, which doesn't fit
a cooldown-gated single trigger at all.

`core/laser_pointer.py`'s `LaserPointerController.update()` is called every
frame directly from the main loop (`app.py`), independent of
`ActionMapper.trigger()`:

```python
active = the gesture currently bound to "laser_pointer" is the presently-stable pose
         (checked against BOTH core.builtin_gestures.stable_pose and
          core.pose_recognizer.stable_pose, since either provenance could be bound to it)
if active:
    EMA-smooth the index fingertip's pixel position (own smoothing state --
        hand.pixels is intentionally raw/unsmoothed elsewhere)
    normalize to [0,1] against the current frame size
    scale to the OS screen resolution (pyautogui.size())
    pyautogui.moveTo(x, y)
else:
    do nothing -- cursor simply stops where it is, no snap-back
```

`app.py` explicitly skips calling `ActionMapper.trigger()` for a pose
transition when its bound control resolves to `laser_pointer` — starting to
point wouldn't consume the shared cooldown and briefly block a legitimate
next/previous-slide gesture right after.

---

## 9. Web control panel architecture

**Files:** `web/state.py`, `web/server.py`, `web/static/*`.

The camera/recognition loop (`app.py`) runs in the **main thread**, exactly
as it always has. Flask runs on a **background daemon thread**. They talk
through one small, deliberately dumb hand-off object:

```python
class AppState:
    publish_status(dict)   # main thread writes, web thread reads
    get_status() -> dict
    send_command(**cmd)    # web thread writes (a POST route), main thread reads
    drain_commands() -> [cmd, ...]
```

Every mutating web route (`/api/live`, `/api/bindings`, `/api/record/start`,
`/api/panel_focus`, `/api/preview/pin`, etc.) does **nothing but enqueue a
command** and return immediately. The main loop drains the queue once per
frame (`App.process_commands()`) and is the **only** thread that ever
actually calls `self.bindings.set(...)`, `self.store.add_sample(...)`,
toggles `self.live`, flips `self.preview_pinned`, etc. This means:

- No locks needed around `App`'s own state — only `AppState`'s tiny
  internal lock, guarding a dict swap and a queue.
- A rebind or a "start recording" click takes effect one frame later
  (≤ ~50ms) — imperceptible, and avoids any risk of tearing state mid-frame.
- GET routes (`/api/status`, `/api/controls`) just read the latest published
  snapshot.

**There is no video in the browser, on purpose** — no MJPEG stream, no
`<img>`/`<video>` tag, nothing for Flask to encode or serve frame-by-frame.
The camera feed only ever exists as the native `cv2` floating window
(`App.show_frame`, section below); the web panel is pure status/control
JSON. That is a real processing saving, not just a UI choice: encoding and
serving a live JPEG stream every frame is real, blocking work this build
does not pay for at all.

The frontend (`web/static/app.js`) is plain JS, no framework: it polls
`/api/status` every 400ms and re-renders whichever tab is currently visible
from whatever it gets back. Three tabs, plain show/hide (`.tab-panel.active`
in CSS, no client-side routing) so none of them are ever a long page the
user has to scroll past the others to reach:

1. **Main** — status, session controls (LIVE/keystrokes/reset/reload), the
   camera-preview pin toggle (see below), and a "live keystrokes" feed —
   the last few entries of `ActionMapper.history`, reused rather than kept
   as a second parallel log, rendered newest-first.
2. **Record gesture** — just the recording card, on its own.
3. **Gestures** — the control↔gesture bindings table, the gesture list, and
   the distance report.

Every button click is a `fetch()` POST; the page also runs a focus/blur/
visibility heartbeat (see below) independent of the 400ms status poll.

The page binds to `127.0.0.1` only (`web.host` in `thresholds.yaml`) — this
build deliberately does not attempt Bluetooth HID or a Jetson-hosted
version; see `prd.md`'s open items and the "Future: Jetson / Bluetooth HID"
note in `SETUP.md`.

### 9.1 The floating camera preview

`App.show_frame` (app.py) owns a single native `cv2` window, shown or hidden
by `App._want_preview_visible()`:

```
show it if:  recording (only visual feedback while capturing a sample)
          or preview_pinned (web panel's "always keep it on top" toggle)
          or the web panel does NOT currently have focus
otherwise: hide it (destroy the window -- no imshow, no draw cost at all)
```

"Does the web panel have focus" is not guessed from window titles — the
page itself reports it via `/api/panel_focus`, driven by `focus`/`blur`/
`visibilitychange` plus a 1s heartbeat (in case a browser fails to fire
those reliably). A stale heartbeat (tab/browser closed outright) times out
after `PANEL_FOCUS_TIMEOUT` (3s) and is treated as "not focused" too, so the
preview does not stay hidden forever with no page left to un-hide it. This
is the mechanism behind "switch away from the browser (e.g. to your slides)
and the camera preview appears on top by default" — and the *only* way to
change that default is the pin toggle on the Main tab, not some hidden
keyboard shortcut or the native window's own close button (closing it
manually just makes it pop back up the next frame it's wanted, same as
before it was closed — the pin toggle is the one place that actually
changes the behaviour, per design).

When shown, the window is created `WINDOW_NORMAL` (resizable) and sized
once via `cv2.resizeWindow` to a small aspect-correct default — sized from
the *client* area directly, which is why the video is never clipped by
title-bar/border chrome. (An earlier version forced the window's outer
rect via a raw win32 `SetWindowPos` width/height, which does eat into that
chrome and was clipping the visible video — the "small but cropped" bug.)
Position and always-on-top are only actively (re)applied once, at the
moment the window transitions from hidden to shown (bottom-left corner of
whichever monitor has the user's current foreground window, or wherever it
already is if none is found) — after that, on-top is passively reasserted
every `PREVIEW_TOPMOST_RECHECK_SECONDS` (0.5s) without moving or resizing
anything, so the user is free to drag or resize it and it stays put (the
"dynamic, don't keep it fixed" behaviour) instead of snapping back every
half second like the previous always-repositioning version did.

---

## 10. Why nothing above knows about pyautogui, Flask, or each other

Every layer boundary in this document exists to keep one specific kind of
change cheap:

| you want to... | you touch... | you do NOT touch... |
|---|---|---|
| retune a threshold | `thresholds.yaml` | any `.py` file |
| remap a gesture to a different key | the web panel (persists to `data/bindings.json`) | any recognizer |
| add a new *custom* gesture | record it via the web panel | any code at all |
| add a new *built-in* gesture | `core/builtin_gestures.py` + a few `[builtin]` constants | `action_mapper.py`, `controls.py`, the web layer |
| add a new *control* (e.g. "next section") | `core/controls.py` | every recognizer, `bindings_store.py` |
| change how gestures are displayed/edited | `web/static/*` | any `core/*.py` |
| swap `pyautogui` for a different keystroke backend (e.g. future BLE HID) | `action_mapper.py`'s `_press()` (and `laser_pointer.py`'s `pyautogui.moveTo` call) | every recognizer, the web layer, bindings |

That last row is exactly the seam left open for the deferred Jetson /
Bluetooth-HID work: nothing about recognition, bindings, or the web UI
assumes `pyautogui` — swapping the output backend is a change to two call
sites, not an architecture change.
