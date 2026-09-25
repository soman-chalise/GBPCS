# Validation plan (PRD section 7)

Five tests. Run them in order — test 1 gates the rest. Every test below has a
concrete pass/fail number you read off the tool, not a judgement call.

Nothing here has been run against your hand yet. The offline self-test
(`tools/selftest.py`) proves the plumbing and default thresholds are sane with
synthetic data; it cannot tell you whether *your* swipe is repeatable. That is
what this document is for.

**Before you start** — the four core gestures below (`swipe_left`,
`swipe_right`, plus `thumbs_up`/`gun_point`/etc.) are now **built-in**: they
ship pre-tuned and need no recording at all. Launch the app and open the web
panel it starts:

```bash
venv/Scripts/python.exe app.py
```

If instead you're validating **custom** gestures you've recorded, use the
"Record a custom gesture" card in the web panel: name it, pick
trajectory/pose, click **Start recording**, perform the gesture, click **Stop
recording**. Repeat for each custom gesture under test.

---

## Test 1 — Intra-class vs inter-class separation

**Run**

```bash
venv/Scripts/python.exe app.py --stats
```

or read the "Distance report" card in the web panel, which updates live and
covers custom gestures only (built-ins have no samples to compare).

**With 1 sample per gesture** you get the inter-class half only. Read the
`SEPARATION VERDICT` block:

```
SEPARATION VERDICT (per gesture)
  swipe_left    intra=n/a (1 sample)  inter_min=1.578  thr=0.420  GOOD
```

**PASS**: every row says `GOOD` — i.e. `inter_min` comfortably exceeds that
gesture's threshold. Aim for `inter_min` at least 2× the threshold.
**FAIL**: any row says `RISK` — the nearest *other* gesture sits inside this
gesture's own acceptance radius, so they will be confused live.

**With ≥2 samples** the intra-class half appears and the verdict switches to a
gap:

```
  swipe_left    intra_max=0.028  inter_min=1.578  gap=+1.551  GOOD
```

**PASS**: `gap` is positive and large. **FAIL**: `OVERLAP` — two samples of
*different* gestures are as close as two samples of the *same* gesture. Fixing
that means re-recording the offending gesture more consistently, not retuning
thresholds.

For the trajectory family specifically, `swipe_left` vs `swipe_right` should
land around 1.5–1.6 against a 0.42 threshold. Anything under ~1.0 means your
two swipes are not as different as you think.

> **1-sample note:** with only one gesture recorded in a family there is no
> runner-up, so the margin check is inactive and the report says so. Record
> both swipes (and both poses) before trusting test 1.

---

## Test 2 — Swipe speed (slow / normal / fast)

The attempt-1 failure mode. This is the test that ROI cropping used to fail.

**Run**

```bash
venv/Scripts/python.exe app.py --dry-run --log-frames --session-label speed
```

Click **Go LIVE** in the web panel. Perform **5 slow, 5 normal, 5 fast** swipes
in each direction, pausing ~1.5s between them (longer than the 0.8s cooldown,
so nothing is suppressed for the wrong reason).

**Watch the terminal.** Every segment prints a line:

```
[traj] ACCEPT swipe_left: best=swipe_left d=0.118 thr=0.420 default(1 sample) | runner-up swipe_right 1.601 (ratio 0.07) | ok  | frames=11 dur=0.37s peak_speed=0.412 close=still
```

**PASS criteria**

| | target |
|---|---|
| Accepted | 30 / 30 |
| `frames=` on fast swipes | ≥ 5 (the `min_segment_frames` floor) |
| `close=` | `still` — never `hand_lost` |
| `d=` on fast vs slow | same ballpark; arc-length resampling should make speed irrelevant |

**FAIL signatures and what they mean**

- `segment discarded: too_short(3f<5f)` — your fast swipe finished in under 5
  frames. Lower `segmenter.min_segment_frames`, or raise camera FPS.
- `close=hand_lost` — MediaPipe lost the hand to motion blur. The segment is
  still classified (by design), but if it is also rejected, improve lighting
  to shorten the camera's exposure time.
- Fast swipes accepted but slow ones rejected (or vice versa) — genuinely
  unexpected given arc-length resampling; check `logs/frames.csv` for a
  `seg_state` that flapped `IN_MOTION`→`WAITING` mid-swipe, which would mean
  `segmenter.idle_speed_threshold` is too high for your slow swipe.

**Then check the CSV** for the speed spread you actually produced:

```bash
venv/Scripts/python.exe -c "
import csv;rows=[r for r in csv.DictReader(open('logs/events.csv')) if r['source']=='trajectory']
for r in rows: print(r['outcome'], r['gesture'], 'd='+r['distance'], 'frames='+r['seg_frames'], 'peak='+r['seg_peak_speed'], r['seg_close_reason'])
"
```

You want `peak` to span roughly 3× between your slowest and fastest swipe. If
it does not, you did not actually test three speeds.

---

## Test 3 — Poses, held and quick, with no hand motion

Confirms the pose recognizer needs no trajectory at all. Applies to both
built-in poses (`thumbs_up`, `gun_point`, `peace_sign`, `closed_fist_hold`)
and any custom pose gestures you've recorded.

**Run** (same session is fine; click **Reset stats** in the web panel first)

```bash
venv/Scripts/python.exe app.py --dry-run --session-label poses
```

**3a — held, completely stationary.** Rest your elbow so the wrist does not
drift. Open hand → hold 3s → close to a fist → hold 3s → open. Repeat 5×.

**PASS**: exactly one trigger per transition. The preview's HUD line shows the
stable pose name flipping once per change. Terminal:

```
[pose] transition fist_open -> fist_closed | [pose] MATCH fist_closed: d=0.041 thr=0.300 ...
```

**FAIL**: repeated fires while holding → `pose.hold_frames` is too low, or your
hand is oscillating across the decision boundary. Check the HUD `d=` value: it
should sit well under the threshold while held, not hover at it.

**3b — quick.** Snap open→closed→open as fast as you can, 5×.

**PASS**: each *pair* produces one trigger, then the 0.8s cooldown eats the
second. That is correct behaviour, not a miss — it is what test 5 covers.

**3c — the "no motion required" assertion.** With the hand perfectly still,
poses must still fire. If they do not, `pose.suppress_above_speed` (custom
gestures) or `builtin.suppress_above_speed` (built-ins) is too low for your
camera's noise floor. Set it to `99` in `thresholds.yaml`, click **Reload
config** in the web panel, and retry — if it fires now, raise the value rather
than disabling it (0.20 is the default; try 0.35).

---

## Test 4 — False positives while talking with your hands

**The real acceptance test.** A system that is perfect on deliberate gestures
and fires during conversation is not shippable.

**Run**

```bash
venv/Scripts/python.exe app.py --dry-run --session-label talking
```

`--dry-run` matters: no keystrokes reach your other windows, but the cooldown
and all counting behave identically.

Click **Go LIVE**, then **Reset stats** to zero the counters. Now **talk for 5
minutes** with your hands in frame — gesticulate, point, scratch your face,
adjust your glasses. Do **not** perform any recorded or built-in gesture
deliberately.

Stop the app with `Ctrl+C` in its terminal to get the tally automatically:

```
SESSION SUMMARY [talking]
LIVE time            : 302.4s (5.0 min)
Triggers fired       : 1
Suppressed (cooldown): 0
Trigger rate         : 0.20 / minute
Breakdown:
  swipe_right (trajectory)     1
```

**Every trigger in that list is a false positive.**

| rate | verdict |
|---|---|
| 0–1 per 5 min | ship it |
| 2–4 per 5 min | tighten (below) |
| 5+ per 5 min | something is mistuned, not marginal |

**Tightening, in the order to try it** — change one at a time, click **Reload
config** in the web panel, re-run:

1. Record 1–2 more samples of whichever gesture false-fires. This switches it
   from the generic default threshold to a calibrated one, which is usually a
   large tightening on its own, and it is the PRD's recommended first move.
2. Lower `trajectory.margin_ratio` (0.80 → 0.70). Rejects more ambiguous
   matches. Cheap and safe.
3. Raise `segmenter.min_segment_path_length` (0.55 → 0.8). Conversational hand
   movement is usually short; real swipes are long.
4. Lower `trajectory.default_threshold` (0.42 → 0.35) — only for gestures still
   on 1 sample.
5. Raise `pose.hold_frames` if the false positives are poses.

Diagnose *which* to use with the rejected rows, which are logged too:

```bash
venv/Scripts/python.exe -c "
import csv,collections
rows=list(csv.DictReader(open('logs/events.csv')))
rows=[r for r in rows if r['session_label']=='talking']
print(collections.Counter((r['source'],r['outcome'],r['detail'][:40]) for r in rows).most_common(12))
"
```

If accepted-but-wrong rows show `margin_ratio` near your 0.80 limit, option 2
is the fix. If they show small `distance` values, option 1 or 4 is.

---

## Test 5 — Cooldown prevents double-fires

**Run** (any session)

```bash
venv/Scripts/python.exe app.py --dry-run --session-label cooldown
```

Click **Go LIVE**, then **Reset stats**. Perform **10 deliberate,
well-separated** swipes (≥1.5s apart), then **5 deliberately sloppy** ones — a
swipe that rebounds, or a swipe that ends in a fist/pose.

**PASS**

- The 10 clean swipes → 10 fired, 0 suppressed.
- The 5 sloppy ones → 5 fired and some suppressed. A suppressed line reads:

```
[action] swipe_left [traj] COOLDOWN - suppressed by shared cooldown (0.43s left, last=swipe_left)
```

- Critically, a **swipe followed by a fist** in one motion must fire **once**,
  not twice — that is the shared clock. The suppressed entry will show
  `source=pose` with `last=swipe_left`, proving both recognizers share it.

The web panel's status row shows the cooldown counting down in real time.

**FAIL**: a single deliberate gesture firing twice → raise
`actions.cooldown_seconds`. Legitimate consecutive gestures being eaten →
lower it. 0.8s is the default; the PRD's range is 0.6–1.0s.

---

## Recording the results

`logs/events.csv` accumulates across runs and is tagged with
`--session-label`, so all five tests stay separable in one file. Columns worth
knowing:

| column | use |
|---|---|
| `outcome` | `accepted` / `rejected` / `segment_rejected` |
| `distance`, `threshold`, `threshold_source` | test 1, 4 |
| `runner_up`, `margin_ratio` | test 4 tightening |
| `fired` | `False` here with `outcome=accepted` means cooldown — test 5 |
| `seg_frames`, `seg_peak_speed`, `seg_close_reason` | test 2 |

`logs/frames.csv` (only with `--log-frames`) has the per-frame `speed`,
`seg_state` and `pose_distance` traces — reach for it when a test fails and
you need to see *when* the state machine changed its mind.
