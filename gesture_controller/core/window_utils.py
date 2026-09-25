"""Small win32 helpers for pinning the local camera-preview window.

Windows-only (pywin32). Used by `App.show_frame` (app.py) to keep the preview
always-on-top and parked in the bottom-left corner -- but only actively
repositioned/resized while whatever the user is looking at (the foreground
window) is on the SAME physical monitor as the preview. If the user's active
window is on a different, extended monitor, there's no risk of the preview
covering anything there, so we leave it alone rather than fight the user for
window placement on a screen we have no reason to touch.
"""

from __future__ import annotations

import ctypes
from dataclasses import dataclass
from typing import Optional

import win32api
import win32con
import win32gui

# A process that hasn't declared itself DPI-aware gets monitor rects/window
# coordinates silently scaled by Windows display scaling (e.g. a 1920x1080
# monitor reporting as 1536x864 at 125%), which would misplace/mis-size the
# preview on anything but a 100%-scaled setup. Must happen once, as early as
# possible, before any window/monitor query -- and can only be set once per
# process, so a second call (or an older Windows without shcore) just fails
# quietly and whatever awareness already exists stands.
try:
    ctypes.windll.shcore.SetProcessDpiAwareness(2)    # PROCESS_PER_MONITOR_DPI_AWARE
except Exception:                                      # noqa: BLE001
    try:
        ctypes.windll.user32.SetProcessDPIAware()
    except Exception:                                  # noqa: BLE001
        pass


@dataclass
class Rect:
    left: int
    top: int
    right: int
    bottom: int


def monitor_for_window(hwnd: Optional[int]) -> Optional[Rect]:
    if not hwnd:
        return None
    try:
        hmon = win32api.MonitorFromWindow(hwnd, win32con.MONITOR_DEFAULTTONEAREST)
        left, top, right, bottom = win32api.GetMonitorInfo(hmon)["Monitor"]
        return Rect(left, top, right, bottom)
    except Exception:                                  # noqa: BLE001
        return None


def foreground_window() -> Optional[int]:
    return win32gui.GetForegroundWindow() or None


def find_window_by_title(title: str) -> Optional[int]:
    return win32gui.FindWindow(None, title) or None


def set_topmost(hwnd: int) -> None:
    win32gui.SetWindowPos(
        hwnd, win32con.HWND_TOPMOST, 0, 0, 0, 0,
        win32con.SWP_NOMOVE | win32con.SWP_NOSIZE | win32con.SWP_NOACTIVATE,
    )


def window_rect(hwnd: int) -> Optional[Rect]:
    try:
        left, top, right, bottom = win32gui.GetWindowRect(hwnd)
        return Rect(left, top, right, bottom)
    except Exception:                                  # noqa: BLE001
        return None


def pin_bottom_left(hwnd: int, monitor: Rect, margin: int) -> None:
    """Move (never resize) the window to the monitor's bottom-left corner.

    Deliberately never touches width/height: a forced SetWindowPos outer size
    eats into the title bar/border chrome and clips the video (the previous
    "feed is cropped" bug). Sizing is cv2's job now (WINDOW_NORMAL +
    cv2.resizeWindow in app.py, which sets the *client* area directly and is
    chrome-correct by construction) -- this reads the window's OWN current
    outer rect just to know its height for bottom-alignment, and only ever
    positions.
    """
    rect = window_rect(hwnd)
    if rect is None:
        return
    width, height = rect.right - rect.left, rect.bottom - rect.top
    x = monitor.left + margin
    y = monitor.bottom - height - margin
    win32gui.SetWindowPos(
        hwnd, win32con.HWND_TOPMOST, x, y, 0, 0,
        win32con.SWP_NOSIZE | win32con.SWP_SHOWWINDOW | win32con.SWP_NOACTIVATE,
    )
