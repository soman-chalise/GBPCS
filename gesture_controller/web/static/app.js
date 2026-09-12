(function () {
  "use strict";

  let controlOptions = [];
  let recording = false;

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

      const select = document.createElement("select");
      for (const c of controlOptions) {
        const opt = document.createElement("option");
        opt.value = c.name;
        opt.textContent = c.label;
        if (c.name === g.control) opt.selected = true;
        select.appendChild(opt);
      }
      select.addEventListener("change", () => {
        post("/api/bindings", { gesture: g.name, control: select.value });
      });

      tr.innerHTML =
        '<td class="name">' + g.name + "</td>" +
        "<td>" + g.type + "</td>" +
        "<td>" + (g.builtin ? "-" : g.samples + " (" + g.threshold_source + ")") + "</td>";
      const tdControl = document.createElement("td");
      tdControl.appendChild(select);
      if (!g.builtin) {
        const del = document.createElement("button");
        del.className = "btn small";
        del.textContent = "delete last";
        del.style.marginLeft = "8px";
        del.addEventListener("click", () => post("/api/samples/delete", { name: g.name }));
        tdControl.appendChild(del);
      }
      tr.appendChild(tdControl);
      tbody.appendChild(tr);
    }
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

    renderGestures(status);
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

  api("/api/controls").then((data) => {
    controlOptions = data;
    poll();
  });

  setInterval(poll, 400);
})();
