# Setup & Running

Step-by-step instructions to get the gesture controller running on your own
machine, plus every gotcha we actually hit while building it. For *how it
works*, see `DESIGN.md`. For a quick overview, see `README.md`.

---

## 1. Requirements

- **Windows** (this build has been developed/tested on Windows; the code has
  no Windows-only APIs in the recognition/web layers, but the camera backend
  defaults to `CAP_DSHOW` on `win32`).
- **Python 3.12** (this was built and tested against 3.12; other 3.x versions
  likely work but aren't verified).
- **A webcam** — built-in laptop camera, a USB webcam, or a phone camera app
  (see section 6). 1280x720 @ 30fps is the tuned default; other resolutions
  work but thresholds were tuned around this.
- **No GPU required.** Everything runs on CPU in real time.

---

## 2. Install dependencies

From inside this folder (`gesture_controller/`), create a virtual
environment and install into it:

```powershell
python -m venv venv
venv\Scripts\pip install -r requirements.txt
```

(If you cloned this alongside a sibling `venv/` someone else already built,
just reuse that instead of making a new one — the commands throughout this
doc assume a `venv/` sitting next to `app.py`, adjust the path if yours is
elsewhere.)

Dependencies (`requirements.txt`):

```
mediapipe==1.0.1       # hand landmark detection
opencv-python==5.0.0.93 # camera capture, image processing, JPEG encode
numpy==2.5.2
pyautogui==0.9.54      # sends the actual keystrokes / mouse moves
PyYAML==6.0.3           # thresholds.yaml
Flask==3.1.0             # the local web control panel
```

### If `pip install` fails with an SSL error

We hit this ourselves: some machines have a Python whose `_ssl` module is
blocked (by the same Application Control policy discussed in section 8),
which breaks `pip`'s HTTPS entirely — every install attempt fails with
`SSLError`/`Could not fetch URL`. If that happens, use
[**uv**](https://github.com/astral-sh/uv) instead — it has its own
independent networking stack and isn't affected:

```powershell
uv pip install --python venv\Scripts\python.exe -r requirements.txt
```

This is exactly how `Flask` got installed during development on a locked-down
machine.

---

## 3. Run it

From the repo root:

```powershell
venv\Scripts\python.exe app.py
```

What happens, in order:

1. It opens your camera (device `0` by default).
2. It starts a local Flask server and — after a short delay — opens your
   default browser to `http://127.0.0.1:5000/`. **That page is the entire
   interface.** There is no OpenCV window and no keyboard shortcuts anymore.
3. In the page:
   - Click **Go LIVE**.
   - Perform a built-in gesture (swipe left/right with an open palm, a
     thumbs-up, or point two fingers like a gun at the camera and hold it).
   - Watch the **last event** line and the **gesture table** for what fired.
4. Stop the app with **Ctrl+C** in the terminal it's running in (there is no
   quit button in the page yet).

### CLI flags

```powershell
venv\Scripts\python.exe app.py --dry-run
```

| flag | what it does |
|---|---|
| `--dry-run` | recognizes and logs gestures but never sends a real keystroke/mouse-move — safe for testing without hijacking your other windows |
| `--camera N` or `--camera URL` | use a different camera device index, or a phone-stream URL (section 6) |
| `--fps N` | cap processing to N frames/sec (0 = uncapped) |
| `--log-frames` | write a per-frame CSV trace to `logs/frames.csv` |
| `--session-label NAME` | tags rows in `logs/events.csv`, e.g. `--session-label talking` for a false-positive test |
| `--stats` | print the distance report for your recorded custom gestures and exit (no camera opened) |
| `--host HOST` | override `web.host` in `thresholds.yaml` |
| `--port N` | override `web.port` in `thresholds.yaml` |
| `--no-browser` | don't auto-open a browser tab (useful if you're launching it from a script) |

### IMPORTANT: give the target window focus

Keystrokes go to whichever window has OS focus, exactly like a physical
keyboard would. Before triggering gestures for real (not `--dry-run`), click
into PowerPoint (or whatever you're presenting) so it — not your browser or
terminal — receives the arrow keys / F5 / Esc.

---

## 4. Using the web control panel

Open `http://127.0.0.1:5000/` (it opens itself on launch).

| card | what it's for |
|---|---|
| **Live preview** | the camera feed with the hand skeleton drawn on it, plus a small HUD (mode, fps, wrist speed, current stable pose) |
| **Session** | Go LIVE / stop, toggle keystrokes on/off (the web equivalent of `--dry-run`, but flippable live), reset stats, reload `thresholds.yaml` |
| **Gestures & bindings** | every built-in gesture (marked `*`) and every custom gesture you've recorded, each with a dropdown to change what control it's bound to |
| **Record a custom gesture** | name it, pick **trajectory** (swipe-like) or **pose** (hand-shape), Start recording → a short countdown → perform the gesture → Stop recording |
| **Distance report** | live intra/inter-class separation stats for your custom gestures (see `VALIDATION.md`) |

### Recording your own gesture

1. Type a name (e.g. `wave`), pick **trajectory** or **pose**.
2. Click **Start recording**. A "get ready" countdown runs on the page —
   nothing is captured yet.
3. Perform the gesture, then click **Stop recording**.
4. **One sample is enough** — the gesture is immediately usable, on a
   generic default threshold. Record 1–2 more samples later if it misfires;
   it automatically switches to a tighter, calibrated threshold once it has
   ≥2 samples. No separate "recalibrate" step.
5. Bind it to a control from its row's dropdown.

### Rebinding a built-in gesture

Just change its dropdown in the **Gestures & bindings** table — e.g. if you'd
rather `thumbs_up` mean "next slide" and `swipe_right` mean "start
presentation", set that there. Takes effect within one frame, no restart.

---

## 5. Testing without a camera / without sending real input

- `--dry-run` (or the "Keystrokes" toggle in the web panel) — the full
  pipeline runs, gestures are recognized and logged, nothing is actually sent
  to the OS.
- `venv\Scripts\python.exe tools\selftest.py` — offline,
  synthetic-data checks of every recognizer's logic (no camera, no MediaPipe
  video pipeline needed beyond import). Run this after any threshold or
  recognizer change.
- `venv\Scripts\python.exe app.py --stats` — prints the
  distance report for your recorded custom gestures without opening a camera.

For validating gestures against your *actual* hand and camera (the tests
that matter for a real ship decision), see `VALIDATION.md`.

---

## 6. Using a phone as the camera instead of a webcam

A phone sensor usually beats a laptop webcam, and its flashlight fixes the
dim-room motion blur that otherwise drops fast swipes.

1. Install **IP Webcam** (Android) — no PC driver needed.
2. Set resolution 1280x720, quality ~70, FPS 30, and turn the **torch on**.
3. Connect the phone by USB-C, unlock it, enable **USB tethering**.
4. Tap **Start server** at the bottom of the IP Webcam app.
5. Find the exact stream URL:

   ```powershell
   venv\Scripts\python.exe tools\find_phone_camera.py
   ```

6. Run with it:

   ```powershell
   venv\Scripts\python.exe app.py --camera http://192.168.42.129:8080/video
   ```

You can also set `camera.source` in `config/thresholds.yaml` to that URL
permanently, instead of passing `--camera` every time.

A network stream pushes frames faster than the app consumes them; a
background thread (`core/frame_source.py`) drains it continuously and keeps
only the newest frame so latency doesn't grow unbounded. That thread is
controlled by `camera.threaded_stream` in `thresholds.yaml`, which defaults
to `true` — correct here, and for the Jetson/4K-camera deployment rig.
Section 9 covers the one case where you'd want it `false` instead (a local
dev laptop whose USB/built-in webcam driver won't grant hardware MJPG).

---

## 7. Troubleshooting: camera opens but the preview is blank / one solid colour

Almost always means **another app already has the camera** — Teams, Zoom,
Slack, OBS, the Windows Camera app, or a browser tab with an active camera
preview. Windows hands the *second* app placeholder frames instead of
refusing to open the device, so it looks like the app is broken when it's
actually just getting fed nothing.

```powershell
venv\Scripts\python.exe tools\camera_check.py
```

This tries every backend/index combination and reports what the pixels
actually contain (`LOOKS LIKE REAL VIDEO` / `UNIFORM STATIC FILL` /
`NEARLY BLACK` / `FROZEN`). Add `--save` to also write a sample frame per
backend into `logs/` for inspection.

---

## 8. Troubleshooting: `OSError: ... An Application Control policy has blocked this file`

This is **not a bug in the app** — it's Windows itself refusing to load
`mediapipe`'s native DLL (`venv\Lib\site-packages\mediapipe\tasks\c\libmediapipe.dll`).
We hit this on a managed/corporate laptop; the exact error looks like:

```
Code Integrity determined that a process (...python.exe) attempted to load
...\mediapipe\tasks\c\libmediapipe.dll that did not meet the Enterprise
signing level requirements or violated code integrity policy
(Policy ID: {xxxxxxxx-xxxx-...}).
```

The **Policy ID** in the message means this is a managed Windows Defender
Application Control (WDAC) policy (from your organization's IT/MDM), on top
of — or instead of — consumer **Smart App Control**. `mediapipe`'s compiled
extension is unsigned, so it fails that check unconditionally; nothing in
this codebase can work around it. Your options, in order:

1. **Ask your IT/security team** to allow `libmediapipe.dll` (or the whole
   `venv\Lib\site-packages\mediapipe\` path) under that WDAC policy. This is
   the correct fix on a managed device.
2. **Run this on an unmanaged personal machine** instead — same repo, same
   `venv`, same command.
3. Check **Settings → Privacy & security → Windows Security → App & browser
   control → Smart App Control**. If it's in "Evaluation" (not "On"), you can
   turn it off yourself. If it's already "On", Microsoft only allows
   disabling it via a clean Windows reinstall — not worth doing just for
   this.

You can confirm which policy engine is involved with (as admin):

```powershell
Get-WinEvent -LogName "Microsoft-Windows-CodeIntegrity/Operational" -MaxEvents 5 |
    Select-Object TimeCreated, Id, Message | Format-List
```

Look for event ID `3077`/`3033` (the block) and `3118`/`8045` ("Smart App
Control Block Details").

---

## 9. Troubleshooting: hand detection feels slow / a couple seconds behind

If this happens **throughout the whole session**, not just at startup, and
per-frame processing feels fine in isolation, this is almost always the
camera capture thread fighting MediaPipe for CPU, not a lighting or model
problem. Look for this line at startup:

```
[camera] WARNING: driver did not grant MJPG (got '...'). The wire format is
likely uncompressed and may saturate USB bandwidth...
```

If you see it: your webcam's driver isn't giving the app hardware-compressed
frames, so `core/frame_source.py`'s background camera-reading thread has to
do real CPU work decoding raw frames every read — and with
`camera.threaded_stream: true`, that competes directly with MediaPipe
inference on the main thread for CPU, instead of overlapping idle I/O wait
the way the threaded design assumes. Measured on real hardware hitting this:
effective fps roughly **halved** (28fps → 13fps) and per-frame recognizer
latency went from a consistent ~16ms to averaging 63ms with spikes past
250ms, with threading on vs off.

`camera.threaded_stream` in `thresholds.yaml` defaults to `true` — correct
for a network stream (section 6) and for a proper deployment camera (e.g. the
Jetson/4K rig), where it only helps. If you hit this on a **local dev
laptop** with a USB/built-in webcam, set it to `false` **there** (don't
change the shared default other people/rigs rely on) and re-test. If the
slowness persists with it `false` too, try a lower `camera.width`/`height`,
and see `tools/camera_check.py` (section 7) to rule out a camera-layer issue
underneath it.

This matters most if you're deploying somewhere with harder detection
conditions to begin with (e.g. dim auditorium lighting) — a starved,
sub-20fps pipeline stretches every frame-count threshold in
`thresholds.yaml` (`hold_frames`, `motion_start_frames`,
`hand_lost_grace_frames`, ...) proportionally longer in wall-clock time on
top of whatever the lighting itself costs you, so fix this first and
re-measure before concluding a detection problem is a lighting problem. A
dev laptop with a flaky webcam driver is not a reliable stand-in for the
actual deployment rig on this axis — validate the auditorium claim on the
Jetson + 4K camera itself, not just on the laptop.

---

## 10. Data that persists between runs

| file | what |
|---|---|
| `data/gestures.json` | your recorded **custom** gesture samples |
| `data/bindings.json` | the live gesture→control mapping (built-ins + custom) |
| `logs/events.csv` | every classification attempt, accepted and rejected, across all runs |
| `logs/frames.csv` | per-frame trace, only written with `--log-frames` |

All four are safe to delete if you want a clean slate — they're recreated
automatically (bindings re-seed to the defaults in
`core/bindings_store.py`'s `DEFAULT_BINDINGS`).

---

## 11. Future: Jetson Orin / Bluetooth HID (not built yet)

The eventual target is a headless NVIDIA Jetson Orin with the camera
attached to it, connecting to the presentation laptop over Bluetooth as a
HID keyboard/mouse (so the laptop needs no drivers or software at all), with
this same web control panel served by the Jetson and reachable from any
browser on the network.

**That port is intentionally not part of this build.** Today, everything
runs on one machine (your laptop or the Jetson, whichever has the camera),
the web panel binds to `127.0.0.1` only, and keystrokes/mouse-moves go out
through `pyautogui` locally. The architecture already isolates the one seam
this future work needs — see `DESIGN.md` section 10, "swap `pyautogui` for a
different keystroke backend" — so that port should be a change to
`action_mapper.py` and `laser_pointer.py`'s output calls, plus wiring the
Flask server to listen on the Jetson's network address instead of localhost,
not a redesign.
