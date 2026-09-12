"""Flask control panel -- status/control JSON API + MJPEG preview.

Every mutating route only enqueues a command onto `AppState`; the camera loop
in `app.py` is what actually applies it (see `web/state.py`). Every GET route
reads the most recent snapshot the loop published. Binds to localhost only for
now -- see prd.md's open item on the Jetson/Bluetooth HID port.
"""

from __future__ import annotations

import os
import time

from flask import Flask, Response, jsonify, request

from core import controls as controls_mod

from .state import AppState

STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")


def create_app(state: AppState) -> Flask:
    app = Flask(__name__, static_folder=STATIC_DIR, static_url_path="")
    app.url_map.strict_slashes = False

    @app.get("/")
    def index():
        return app.send_static_file("index.html")

    @app.get("/api/status")
    def status():
        return jsonify(state.get_status())

    @app.get("/api/controls")
    def controls_list():
        return jsonify(controls_mod.as_list())

    @app.post("/api/live")
    def set_live():
        body = request.get_json(force=True, silent=True) or {}
        state.send_command(type="set_live", value=bool(body.get("value")))
        return jsonify(ok=True)

    @app.post("/api/keystrokes")
    def set_keystrokes():
        body = request.get_json(force=True, silent=True) or {}
        state.send_command(type="set_keystrokes", value=bool(body.get("value")))
        return jsonify(ok=True)

    @app.post("/api/bindings")
    def set_binding():
        body = request.get_json(force=True, silent=True) or {}
        gesture, control = body.get("gesture"), body.get("control")
        if not gesture or not controls_mod.is_valid(control or ""):
            return jsonify(ok=False, error="bad gesture/control"), 400
        state.send_command(type="set_binding", gesture=gesture, control=control)
        return jsonify(ok=True)

    @app.post("/api/record/start")
    def record_start():
        body = request.get_json(force=True, silent=True) or {}
        name = (body.get("name") or "").strip().lower().replace(" ", "_")
        gtype = body.get("type")
        if not name or gtype not in ("trajectory", "pose"):
            return jsonify(ok=False, error="name and type ('trajectory'|'pose') required"), 400
        state.send_command(type="record_start", name=name, gtype=gtype)
        return jsonify(ok=True)

    @app.post("/api/record/stop")
    def record_stop():
        state.send_command(type="record_stop")
        return jsonify(ok=True)

    @app.post("/api/record/cancel")
    def record_cancel():
        state.send_command(type="record_cancel")
        return jsonify(ok=True)

    @app.post("/api/samples/delete")
    def delete_sample():
        body = request.get_json(force=True, silent=True) or {}
        name = body.get("name")
        if not name:
            return jsonify(ok=False, error="name required"), 400
        state.send_command(type="delete_sample", name=name)
        return jsonify(ok=True)

    @app.post("/api/stats/reset")
    def reset_stats():
        state.send_command(type="reset_stats")
        return jsonify(ok=True)

    @app.post("/api/config/reload")
    def reload_config():
        state.send_command(type="reload_config")
        return jsonify(ok=True)

    @app.get("/stream.mjpg")
    def stream():
        def gen():
            boundary = b"--frame"
            while True:
                jpeg = state.get_frame()
                if jpeg is not None:
                    yield (
                        boundary + b"\r\n"
                        b"Content-Type: image/jpeg\r\n"
                        b"Content-Length: " + str(len(jpeg)).encode() + b"\r\n\r\n"
                        + jpeg + b"\r\n"
                    )
                time.sleep(0.03)

        return Response(gen(), mimetype="multipart/x-mixed-replace; boundary=frame")

    return app
