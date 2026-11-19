(() => {
  if (globalThis.voiceStudioLiveInstalled) return;
  globalThis.voiceStudioLiveInstalled = true;

  let session;
  let layer;
  let boxes;
  let status;
  let caption;
  let clearBoxes;
  let geometryChangedAt = -Infinity;

  /** Remove the overlay when the capture session ends. */
  function stop() {
    clearTimeout(clearBoxes);
    layer?.remove();
    layer = null;
    session = null;
    geometryChangedAt = -Infinity;
  }

  /** Render an isolated overlay that leaves page controls clickable. */
  function start(id) {
    stop();
    session = id;
    layer = document.createElement("div");
    layer.id = "voice-studio-live-overlay";
    Object.assign(layer.style, {
      position: "fixed", inset: "0", pointerEvents: "none", zIndex: "2147483647",
      display: "block", contain: "strict",
    });
    const root = layer.attachShadow({mode: "closed"});
    const style = new CSSStyleSheet();
    style.replaceSync(`
      :host { all: initial; }
      * { box-sizing: border-box; }
      .boxes { position: absolute; inset: 0; }
      .box { position: absolute; border: 2px solid #fff2db; border-radius: 4px;
        box-shadow: 0 0 0 1px #271b14; }
      .box span { position: absolute; left: -2px; top: 0; padding: 2px 5px;
        color: #fff2db; background: #211810; border-radius: 0 0 3px 0;
        font: 600 12px/1.4 system-ui, sans-serif; white-space: nowrap; }
      .status, .caption { position: absolute; color: #fff2db; background: #211810eb;
        border: 1px solid #916b4d; border-radius: 6px; overflow-wrap: anywhere; }
      .status { top: 12px; right: 12px; max-width: min(420px, 80vw); padding: 6px 9px;
        font: 500 12px/1.4 system-ui, sans-serif; color: #ffc190; }
      .caption { bottom: max(45px, 8vh); left: 50%; transform: translateX(-50%);
        padding: 7px 12px; max-width: min(720px, 88vw); text-align: center;
        font: 600 16px/1.35 system-ui, sans-serif; }
      .caption div + div { margin-top: 4px; }
      [hidden] { display: none; }
    `);
    root.adoptedStyleSheets = [style];
    boxes = document.createElement("div");
    boxes.className = "boxes";
    boxes.setAttribute("aria-hidden", "true");
    status = document.createElement("div");
    status.className = "status";
    status.setAttribute("role", "status");
    status.textContent = "Voice Studio Live · запускаем анализ";
    caption = document.createElement("div");
    caption.className = "caption";
    caption.hidden = true;
    root.append(boxes, status, caption);
    positionOverlay();
  }

  function positionOverlay() {
    // Put the overlay inside a fullscreen container when its subtree is rendered.
    const fullscreen = document.fullscreenElement;
    const container = fullscreen && !(fullscreen instanceof HTMLVideoElement) ? fullscreen : document.documentElement;
    if (layer && layer.parentElement !== container) container.append(layer);
  }

  function stamp(ms) {
    const seconds = Math.max(0, Math.floor(ms / 1000));
    return `${Math.floor(seconds / 60)}:${String(seconds % 60).padStart(2, "0")}`;
  }

  /** Only show captures that precede this moment by less than five seconds. */
  function drawBoxes(frames, elapsedMs) {
    const last = frames.at(-1);
    if (!last || !Number.isFinite(last.timeMs) || !Array.isArray(last.faces)) return;
    const age = elapsedMs - last.timeMs;
    const geometryAge = performance.now() - geometryChangedAt;
    if (age >= 5000 || age < -1000 || age >= geometryAge) return;
    for (const face of last.faces.slice(0, 8)) {
      if (!Array.isArray(face.box) || face.box.length !== 4 || !face.box.every(Number.isFinite)) continue;
      const [x, y, w, h] = face.box.map(value => Math.max(0, Math.min(value, 1)) * 100);
      if (!w || !h || x + w > 101 || y + h > 101) continue;
      const box = document.createElement("div");
      box.className = "box";
      Object.assign(box.style, {left: x + "%", top: y + "%", width: w + "%", height: h + "%"});
      const label = document.createElement("span");
      label.textContent = `Лицо ${String(face.id).slice(0, 12)} · ${stamp(last.timeMs)}`;
      box.append(label);
      boxes.append(box);
    }
    clearBoxes = setTimeout(() => boxes?.replaceChildren(), Math.max(0, Math.min(1500, 5000 - age)));
    return age;
  }

  /** Show the most recent results and report the age of a displayed frame. */
  function update(report) {
    clearTimeout(clearBoxes);
    boxes.replaceChildren();
    const segments = Array.isArray(report.segments) ? report.segments : [];
    caption.replaceChildren();
    for (const segment of segments) {
      if (typeof segment.text !== "string" || !segment.text.trim()) continue;
      const line = document.createElement("div");
      line.textContent = `${stamp(segment.startMs)} · ${segment.text}`;
      caption.append(line);
    }
    caption.hidden = !caption.childElementCount;
    const notices = Array.isArray(report.warnings) ? report.warnings.filter(text => typeof text === "string") : [];
    const elapsedMs = Number.isFinite(report.elapsedMs) ? report.elapsedMs : 0;
    const requestMs = Number.isFinite(report.requestMs) ? (report.requestMs / 1000).toFixed(1) + " с анализа" : "";
    const dropped = report.dropped > 0 ? `пропущено ${report.dropped}` : "";
    const age = drawBoxes(Array.isArray(report.faceFrames) ? report.faceFrames : [], elapsedMs);
    const delay = Number.isFinite(age) ? `рамки ${(Math.max(0, age) / 1000).toFixed(1)} с назад` : "";
    status.textContent = ["Voice Studio Live · предыдущий фрагмент", delay, requestMs, dropped, ...notices.slice(0, 2)]
      .filter(Boolean).join(" · ");
  }

  /** Drop boxes from viewport geometry that was changed by the user. */
  function invalidateBoxes() {
    if (!session) return;
    clearTimeout(clearBoxes);
    boxes.replaceChildren();
    geometryChangedAt = performance.now();
  }

  window.addEventListener("resize", invalidateBoxes);
  document.addEventListener("scroll", invalidateBoxes, true);
  document.addEventListener("seeking", event => {
    if (event.target instanceof HTMLVideoElement) invalidateBoxes();
  }, true);
  document.addEventListener("fullscreenchange", () => {
    positionOverlay();
    invalidateBoxes();
  });
  chrome.runtime.onMessage.addListener((message, sender, respond) => {
    if (sender.id !== chrome.runtime.id) return false;
    if (message.type === "voice-live-start") {
      start(message.session);
      respond({ok: true});
    } else if (session && message.session === session) {
      if (message.type === "voice-live-update") update(message.report);
      if (message.type === "voice-live-stop") stop();
      respond({ok: true});
    }
    return false;
  });
})();
