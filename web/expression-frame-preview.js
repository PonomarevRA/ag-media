import {attachStudioIcon} from "./studio-icons.js";
import {uiText} from "./i18n.js";
import {expressionLabel, expressionMarker} from "./facial-expressions.js";
import {FRAME_COLORS} from "./video-overlay.js";
import {momentTimecode} from "./profile-moment-timeline.js";

const mounted = new WeakMap();

/** Resolve normalized geometry only from the requested face AND track in its exact timeline cell. */
export function expressionFaceBox(evidence, {timeMs, startMs, faceId, track}) {
  const stamp = timeMs ?? startMs;
  const timeline = evidence?.timeline;
  if (!Number.isInteger(stamp) || stamp < 0 || !(timeline?.stepMs > 0)) return null;
  const hits = timeline.frames?.[Math.floor(stamp / timeline.stepMs)] ?? [];
  const matches = hits.filter(hit => hit[0] === track && (hit[6] ?? hit[0]) === faceId);
  if (matches.length !== 1) return null;
  const box = matches[0].slice(1, 5);
  if (box.length !== 4 || !box.every(Number.isFinite) || box[0] < 0 || box[1] < 0
      || box[2] <= 0 || box[3] <= 0 || box[0] + box[2] > 1.001 || box[1] + box[3] > 1.001) return null;
  return box;
}

/** Seek to a decoded source frame, removing every listener/timer on success, failure or cancellation. */
export async function waitForExpressionFrame(video, timeMs, {signal, timeoutMs = 5000} = {}) {
  signal?.throwIfAborted();
  if (video.hidden || video.error || !(video.readyState > 0) || !Number.isInteger(timeMs) || timeMs < 0
      || Number.isFinite(video.duration) && timeMs >= video.duration * 1000) throw Error(uiText("Кадр недоступен"));
  video.pause();
  await new Promise((resolve, reject) => {
    let timer;
    const events = ["seeked", "loadeddata"];
    function cleanup() {
      clearTimeout(timer);
      for (const event of events) video.removeEventListener(event, check);
      video.removeEventListener("error", failed);
      video.removeEventListener("emptied", failed);
      signal?.removeEventListener("abort", aborted);
    }
    function finish(error) {
      cleanup();
      if (error) reject(error);
      else resolve();
    }
    function failed() { finish(Error(uiText("Кадр недоступен"))); }
    function aborted() { finish(signal.reason ?? new DOMException("Aborted", "AbortError")); }
    function check() {
      if (!video.seeking && video.readyState >= 2 && video.videoWidth > 0 && video.videoHeight > 0
          && Math.abs(video.currentTime * 1000 - timeMs) < 8) finish();
    }
    for (const event of events) video.addEventListener(event, check);
    video.addEventListener("error", failed);
    video.addEventListener("emptied", failed);
    signal?.addEventListener("abort", aborted, {once: true});
    timer = setTimeout(failed, timeoutMs);
    try {
      video.currentTime = timeMs / 1000;
      check();
    } catch (error) { finish(error); }
  });
  signal?.throwIfAborted();
}

/** Freeze intrinsic source pixels, so normalized geometry is independent of player letterboxing. */
export function drawExpressionFrame(canvas, video, box) {
  const scale = Math.min(1, 1440 / Math.max(video.videoWidth, video.videoHeight));
  canvas.width = Math.max(1, Math.round(video.videoWidth * scale));
  canvas.height = Math.max(1, Math.round(video.videoHeight * scale));
  const ctx = canvas.getContext("2d");
  if (!ctx) throw Error(uiText("Кадр недоступен"));
  ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
  const rect = box.map((value, index) => value * (index % 2 ? canvas.height : canvas.width));
  // The same dark rim and pale face color as the monitor keep the selected head legible.
  ctx.strokeStyle = "rgba(10, 8, 6, .78)";
  ctx.lineWidth = 6;
  ctx.strokeRect(...rect);
  ctx.strokeStyle = FRAME_COLORS.face;
  ctx.lineWidth = 3;
  ctx.strokeRect(...rect);
}

/** Share one source-frame dialog between montage and profile expression controls. */
export function mountExpressionFramePreview(video) {
  if (mounted.has(video)) return mounted.get(video);
  function element(tag, text = "", className = "") {
    const node = document.createElement(tag);
    node.textContent = text;
    node.className = className;
    return node;
  }
  const dialog = element("dialog", "", "speaker-profile-dialog editor-object-dialog expression-frame-dialog");
  dialog.id = "media-expression-dialog";
  const header = element("header", "", "speaker-profile-header");
  const labels = element("div");
  const title = element("h3");
  title.id = "media-expression-dialog-title";
  const time = element("p", "", "muted");
  labels.append(title, time);
  const close = element("button", "", "studio-icon");
  close.dataset.icon = "close";
  close.type = "button";
  close.setAttribute("aria-label", uiText("Закрыть кадр"));
  close.title = uiText("Закрыть кадр");
  attachStudioIcon(close);
  header.append(labels, close);
  const status = element("p", "", "muted expression-frame-status");
  status.setAttribute("role", "status");
  const canvas = element("canvas", "", "expression-source-frame");
  canvas.setAttribute("role", "img");
  dialog.setAttribute("aria-labelledby", title.id);
  dialog.append(header, status, canvas);
  document.body.append(dialog);
  let operation;
  let opener;

  /** Clear both pending work and frozen pixels before another sample or source is selected. */
  function release() {
    operation?.abort();
    operation = undefined;
    canvas.hidden = true;
    canvas.width = canvas.height = 1;
  }
  close.onclick = () => dialog.close();
  dialog.addEventListener("close", () => {
    release();
    if (opener?.isConnected) opener.focus({preventScroll: true});
    opener = undefined;
  });
  for (const event of ["emptied", "loadstart", "error"]) video.addEventListener(event, () => {
    release();
    if (dialog.open) status.textContent = uiText("Кадр недоступен");
  });
  const controller = {
    /** Open the exact measured source frame; a failed seek never exposes the old canvas. */
    async open(selection, evidence, button = document.activeElement) {
      release();
      opener = button;
      const current = operation = new AbortController();
      const timeMs = selection.timeMs ?? selection.startMs;
      title.textContent = `${expressionMarker(selection.label) ?? ""} ${expressionLabel(selection.label)} · ${uiText`Лицо ${selection.faceId}`}`;
      const confidence = Number.isFinite(selection.confidence) ? ` · ${Math.round(selection.confidence * 100)}%` : "";
      time.textContent = momentTimecode(timeMs) + confidence;
      canvas.setAttribute("aria-label", uiText`Исходный кадр · лицо ${selection.faceId} · ${momentTimecode(timeMs)}`);
      status.textContent = uiText("Подготавливаем кадр…");
      if (!dialog.open) dialog.showModal();
      close.focus();
      try {
        const box = expressionFaceBox(evidence, selection);
        if (!box) throw Error(uiText("Рамка этого лица в указанном кадре недоступна."));
        await waitForExpressionFrame(video, timeMs, {signal: current.signal});
        if (operation !== current || current.signal.aborted) return;
        drawExpressionFrame(canvas, video, box);
        dialog.style.setProperty("--object-frame-width", `${65 * canvas.width / canvas.height}dvh`);
        canvas.hidden = false;
        status.textContent = uiText("Исходный кадр с рамкой выбранного лица. Выражение — оценка классификатора.");
      } catch (error) {
        if (operation !== current || current.signal.aborted) return;
        canvas.hidden = true;
        canvas.width = canvas.height = 1;
        status.textContent = error.message || uiText("Кадр недоступен");
      }
    },
    clear() {
      release();
      if (dialog.open) dialog.close();
    },
  };
  mounted.set(video, controller);
  return controller;
}
