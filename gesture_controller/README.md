# Gesture-Based Presentation Controller (v3)

Webcam hand-gesture control for slide decks. MediaPipe hand tracking → two
independent recognizer families (trajectory / pose), each with a **built-in**
zero-recording half and a **custom** you-record-it half → a
gesture-to-control binding layer → keystrokes/mouse via `pyautogui`.
Controlled entirely from a local web control panel that opens automatically
in your browser — there is no OpenCV window or keyboard shortcut anymore.

**Read next, depending on what you need:**

| doc | for |
|---|---|
| **[SETUP.md](SETUP.md)** | installing it, running it, the web panel, troubleshooting (including the "Application Control policy" error), using a phone as the camera |
| **[DESIGN.md](DESIGN.md)** | the full architecture: every algorithm, every module, why each design decision was made |
| **[VALIDATION.md](VALIDATION.md)** | the 5-test procedure for proving gestures actually work against your hand, not just synthetic data |

---

## What it does

- Recognizes hand gestures from a webcam (or a phone camera, see
  `SETUP.md`) in real time, no GPU needed.
- Seven gestures ship **built-in** — work immediately, no setup:
  `swipe_left`, `swipe_right`, `thumbs_up`, `gun_point`, `peace_sign`,
  `open_palm_hold`, `closed_fist_hold`.
- You can also **record your own custom gestures** through the web panel —
  one sample is enough to go live.
- Any gesture, built-in or custom, can be **bound to any control** (next
  slide, previous slide, start/end presentation, blank screen, first slide,
  or a continuous laser pointer that drags the cursor) from a dropdown in the
  web panel — no code changes, takes effect instantly.
- Everything is controlled from one local web page
  (`http://127.0.0.1:5000/`), including recording new gestures, live camera
  preview, and rebinding controls.

## Quick start

```bash
# from the repo root
venv/Scripts/python.exe app.py
```

Your browser opens automatically to the control panel. Click **Go LIVE** and
try a gesture. See `SETUP.md` for the full walkthrough, CLI flags, and
troubleshooting.

## The one architectural rule

Two earlier prototypes failed because trajectory gestures (swipes) and pose
gestures (fists) were pushed through **one shared feature vector** and one
matching algorithm — every change that helped one hurt the other. This build
keeps them **completely separate**, for both the custom recognizers and the
built-in detectors:

```
Camera ─► HandTracker (MediaPipe, full frame, num_hands=1, VIDEO mode)
             │
             ├── wrist_xy (2D) ──► MotionSegmenter ─┬─► TrajectoryRecognizer (custom)  ─┐
             │                                       └─► builtin swipe classifier       │
             │                                                                          ├─► BindingsStore ─► ActionMapper ─► keystroke
             └── curls / shape ──┬─► PoseRecognizer (custom) ────────────────────────────┤        (or LaserPointerController, continuous)
                                 └─► builtin pose detector ────────────────────────────┘
```

Full explanation of every box above, and every threshold and algorithm
involved: **[DESIGN.md](DESIGN.md)**.

## Layout

```
gesture_controller/
  README.md / SETUP.md / DESIGN.md / VALIDATION.md
  config/
    thresholds.yaml           every tunable constant
  core/
    config.py                  YAML loading, fails loudly on typos
    hand_tracker.py             MediaPipe wrapper, 2 independent smoothed streams
    motion_segmenter.py         wrist-speed state machine (trajectory only)
    trajectory_recognizer.py    custom swipe-family recognizer (nearest-centroid)
    pose_recognizer.py          custom fist-family recognizer (nearest-centroid)
    builtin_gestures.py         rule-based, pre-tuned, zero-recording gestures
    controls.py                 control registry (keystrokes + laser_pointer)
    bindings_store.py           gesture -> control persistence (data/bindings.json)
    laser_pointer.py            continuous cursor drag while a pose is held
    action_mapper.py            gesture -> binding -> control -> effect, shared cooldown
    template_store.py           persisted custom samples (data/gestures.json)
    diagnostics.py              distance report, CSV logs, session summary
    frame_source.py             camera backend abstraction (USB + network stream)
  web/
    server.py                   Flask app: status/control JSON API + MJPEG stream
    state.py                    thread-safe hand-off between the loop and Flask
    static/                     index.html, style.css, app.js -- the control panel
  data/gestures.json             custom recorded templates (survives restarts)
  data/bindings.json              gesture -> control mapping (survives restarts)
  logs/events.csv                 every classification, accepted and rejected
  logs/frames.csv                  per-frame trace (only with --log-frames)
  models/hand_landmarker.task
  tools/                           selftest.py, camera_check.py, find_phone_camera.py
  app.py                           camera + recognition loop, no GUI
```
