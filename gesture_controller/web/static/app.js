(function () {
  "use strict";

  let controlOptions = [];
  let recording = false;
  let lastGesturesKey = null;
  let dropdownOpen = false;
  let lastEventsKey = null;

  function api(path, opts) {
    return fetch(path, opts).then((r) => r.json());
  }

  function post(path, body) {
    return api(path, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body || {}),
    });
  }

  function renderGestures(status) {
    const rows = status.gestures || [];
    const tbody = document.getElementById("gesture-rows");
    tbody.innerHTML = "";
    for (const g of rows) {
      const tr = document.createElement("tr");
      if (g.builtin) tr.className = "builtin";

      tr.innerHTML =
        '<td class="name">' + g.name + "</td>" +
        "<td>" + g.type + "</td>" +
        "<td>" + (g.builtin ? "-" : g.samples + " (" + g.threshold_source + ")") + "</td>";
      const tdActions = document.createElement("td");
      if (!g.builtin) {
        const del = document.createElement("button");
        del.className = "btn small";
        del.textContent = "delete last";
        del.addEventListener("click", () => post("/api/samples/delete", { name: g.name }).then(poll));
        tdActions.appendChild(del);
      }
      tr.appendChild(tdActions);
      tbody.appendChild(tr);
    }
  }

  // Controls are the fixed, listed side; each gets a dropdown of which
  // gesture (built-in or custom) currently triggers it -- the inverse of the
  // old per-gesture control picker.
  function renderControls(status) {
    const rows = status.gestures || [];
    const tbody = document.getElementById("control-rows");
    tbody.innerHTML = "";
    for (const c of controlOptions) {
      if (c.name === "none") continue;
      const assigned = rows.find((g) => g.control === c.name);
      const currentGesture = assigned ? assigned.name : "";

      const tr = document.createElement("tr");
      const tdControl = document.createElement("td");
      tdControl.className = "name";
      tdControl.textContent = c.label;
      tr.appendChild(tdControl);

      const select = document.createElement("select");
      const optNone = document.createElement("option");
      optNone.value = "";
      optNone.textContent = "-- unassigned --";
      select.appendChild(optNone);
      for (const g of rows) {
        const opt = document.createElement("option");
        opt.value = g.name;
        opt.textContent = g.builtin
          ? g.name + " *"
          : g.name + " (" + g.type + ", " + g.samples + " sample" + (g.samples === 1 ? "" : "s") + ")";
        if (g.name === currentGesture) opt.selected = true;
        select.appendChild(opt);
      }
      select.addEventListener("focus", () => { dropdownOpen = true; });
      select.addEventListener("blur", () => { dropdownOpen = false; });
      select.addEventListener("change", () => {
        // The native option list is already closed once "change" fires (the
        // user has committed a choice) -- safe to let the next poll rebuild
        // right away, which is what makes a reassignment (stealing this
        // gesture from whatever control had it) show up on the OTHER
        // control's dropdown immediately instead of only after it loses focus.
        dropdownOpen = false;
        const newGesture = select.value;
        if (currentGesture && currentGesture !== newGesture) {
          post("/api/bindings", { gesture: currentGesture, control: "none" });
        }
        if (newGesture) {
          post("/api/bindings", { gesture: newGesture, control: c.name }).then(poll);
        } else {
          poll();
        }
      });

      const tdGesture = document.createElement("td");
      tdGesture.appendChild(select);
      tr.appendChild(tdGesture);
      tbody.appendChild(tr);
    }
  }

  function renderEventFeed(status) {
    const rows = status.recent_events || [];
    const key = JSON.stringify(rows);
    if (key === lastEventsKey) return;
    lastEventsKey = key;

    const ul = document.getElementById("event-feed");
    ul.innerHTML = "";
    if (!rows.length) {
      const li = document.createElement("li");
      li.className = "feed-empty";
      li.textContent = "no events yet";
      ul.appendChild(li);
      return;
    }
    for (const ev of rows) {
      const li = document.createElement("li");
      li.className = ev.fired ? "fired" : "suppressed";
      const label = document.createElement("span");
      label.textContent = ev.gesture + " -- " + ev.detail;
      const tag = document.createElement("span");
      tag.className = "tag";
      tag.textContent = ev.fired ? ev.source : "skipped";
      li.appendChild(label);
      li.appendChild(tag);
      ul.appendChild(li);
    }
  }

  function renderPreview(status) {
    const chk = document.getElementById("chk-pin");
    if (document.activeElement !== chk) chk.checked = !!status.preview_pinned;
    document.getElementById("preview-hint").textContent = status.preview_visible
      ? "floating preview: visible right now"
      : "floating preview: hidden right now (panel has focus)";
  }

  function renderStatus(status) {
    document.getElementById("s-fps").textContent = status.fps ?? "-";
    const handDot = document.getElementById("s-hand-dot");
    const handTxt = document.getElementById("s-hand");
    handDot.className = "dot " + (status.hand_present ? "on" : "off");
    handTxt.textContent = status.hand_present ? "hand tracked" : "no hand";
    document.getElementById("s-cooldown").textContent =
      (status.cooldown_remaining || 0).toFixed(2) + "s";
    document.getElementById("s-last-event").textContent =
      status.last_event || "no events yet";
    document.getElementById("report").textContent =
      status.distance_report || "(no custom gestures recorded yet)";

    const modePill = document.getElementById("mode-pill");
    if (status.recording) {
      modePill.textContent = "RECORDING";
      modePill.className = "pill rec";
    } else if (status.live) {
      modePill.textContent = "LIVE";
      modePill.className = "pill live";
    } else {
      modePill.textContent = "IDLE";
      modePill.className = "pill idle";
    }

    const btnLive = document.getElementById("btn-live");
    btnLive.textContent = status.live ? "Stop LIVE" : "Go LIVE";
    btnLive.classList.toggle("primary", !status.live);

    const btnKeys = document.getElementById("btn-keys");
    btnKeys.textContent = "Keystrokes: " + (status.keystrokes_enabled ? "ON" : "OFF");

    recording = !!status.recording;
    document.getElementById("btn-cancel").disabled = !recording;
    document.getElementById("btn-record").textContent = recording ? "Stop recording" : "Start recording";
    const hint = document.getElementById("rec-hint");
    if (recording) {
      if (status.record_countdown > 0) {
        hint.textContent = "get ready... starts in " + status.record_countdown.toFixed(1) + "s";
      } else {
        hint.textContent = "capturing '" + status.record_name + "' -- " +
          status.record_captured + " frames. Click Stop recording when done.";
      }
    } else {
      hint.textContent = "";
    }

    // Rebuilding these tables tears down and recreates every <select>, which
    // was closing an open dropdown's native option list out from under the
    // user on every 400ms poll tick -- even when nothing had changed. Skip
    // the rebuild while a dropdown is actually open, and skip it entirely
    // when the underlying data hasn't changed (also removes needless DOM
    // churn on every tick).
    const gesturesKey = JSON.stringify(status.gestures || []);
    if (gesturesKey !== lastGesturesKey && !dropdownOpen) {
      renderControls(status);
      renderGestures(status);
      lastGesturesKey = gesturesKey;
    }

    renderEventFeed(status);
    renderPreview(status);
  }

  function poll() {
    api("/api/status").then(renderStatus).catch(() => {});
  }

  document.getElementById("btn-live").addEventListener("click", () => {
    const live = document.getElementById("btn-live").textContent === "Go LIVE";
    post("/api/live", { value: live }).then(poll);
  });

  document.getElementById("btn-keys").addEventListener("click", () => {
    const on = document.getElementById("btn-keys").textContent.indexOf("ON") !== -1;
    post("/api/keystrokes", { value: !on }).then(poll);
  });

  document.getElementById("btn-reset").addEventListener("click", () => post("/api/stats/reset").then(poll));
  document.getElementById("btn-reload").addEventListener("click", () => post("/api/config/reload").then(poll));

  document.getElementById("btn-record").addEventListener("click", () => {
    if (recording) {
      post("/api/record/stop").then(poll);
      return;
    }
    const name = document.getElementById("rec-name").value.trim();
    if (!name) { alert("enter a gesture name first"); return; }
    const gtype = document.querySelector('input[name="rec-type"]:checked').value;
    post("/api/record/start", { name: name, type: gtype }).then(poll);
  });

  document.getElementById("btn-cancel").addEventListener("click", () => post("/api/record/cancel").then(poll));

  document.getElementById("chk-pin").addEventListener("change", (e) => {
    post("/api/preview/pin", { value: e.target.checked }).then(poll);
  });

  // -- tabs -------------------------------------------------------------
  // Plain show/hide, no routing -- keeps the Record and Gestures pages from
  // ever needing to scroll past each other, and each renders independently
  // of which one is currently visible (they all just read the same status).
  for (const btn of document.querySelectorAll(".tab")) {
    btn.addEventListener("click", () => {
      for (const b of document.querySelectorAll(".tab")) b.classList.remove("active");
      for (const p of document.querySelectorAll(".tab-panel")) p.classList.remove("active");
      btn.classList.add("active");
      document.getElementById("tab-" + btn.dataset.tab).classList.add("active");
    });
  }

  // -- panel focus heartbeat --------------------------------------------
  // Tells app.py whether this page currently has the user's attention, so it
  // knows whether to keep the floating camera preview hidden (panel focused)
  // or show it always-on-top (panel not focused, e.g. switched to slides).
  // Events cover the common cases; the interval is just a safety net against
  // a missed event (e.g. a browser that doesn't fire blur reliably).
  function sendFocus() {
    const focused = document.hasFocus() && document.visibilityState === "visible";
    post("/api/panel_focus", { focused: focused }).catch(() => {});
  }
  window.addEventListener("focus", sendFocus);
  window.addEventListener("blur", sendFocus);
  document.addEventListener("visibilitychange", sendFocus);
  setInterval(sendFocus, 1000);
  sendFocus();

  api("/api/controls").then((data) => {
    controlOptions = data;
    poll();
  });

  setInterval(poll, 400);
})();
