const CHUNK_MS = 3000;
const REQUEST_TIMEOUT_MS = 12000;
const MAX_CHUNK_BYTES = 1024 * 1024;
const ENDPOINT = "http://127.0.0.1:8766/api/live-chunk";

let current = null;

function send(run, message) {
  return chrome.runtime.sendMessage({target: "worker", session: run.session, ...message}).catch(() => {});
}

/** Release the recorder, pending upload and playback monitor as a unit. */
function cleanup(run, reason) {
  if (current !== run) return;
  current = null;
  clearTimeout(run.timer);
  run.request?.abort();
  if (run.recorder?.state !== "inactive") run.recorder?.stop();
  run.stream?.getTracks().forEach(track => track.stop());
  run.monitor?.close().catch(() => {});
  if (reason) void send(run, {type: "ended", message: reason});
}

function publish(run, data) {
  return send(run, {type: "update", report: {
    ...data,
    elapsedMs: Math.max(0, Math.round(performance.now() - run.started)),
    dropped: run.dropped,
  }});
}

/** Submit one self-contained fragment with timestamps relative to the capture start. */
async function upload(run, blob, startMs) {
  // Keep one encoded chunk in flight. Native capture may outrun the models.
  if (run.request || !blob.size || blob.size > MAX_CHUNK_BYTES) {
    run.dropped++;
    const message = blob.size > MAX_CHUNK_BYTES ? "Фрагмент слишком велик." : "";
    if (message) void publish(run, {warnings: [message]});
    return;
  }
  const controller = new AbortController();
  run.request = controller;
  const timeout = setTimeout(() => controller.abort(), REQUEST_TIMEOUT_MS);
  const url = new URL(ENDPOINT);
  url.search = new URLSearchParams({session: run.session, lang: run.language, start_ms: String(startMs)});
  const requested = performance.now();
  let serviceError;
  try {
    const response = await fetch(url, {
      method: "POST",
      headers: {"X-Voice-Live-Token": run.token, "Content-Type": "video/webm"},
      body: blob,
      signal: controller.signal,
      credentials: "omit",
      redirect: "error",
    });
    if (current !== run) return;
    if (response.status === 401 || response.status === 403) {
      cleanup(run, "Студия отклонила токен. Проверьте его в настройках расширения.");
      return;
    }
    if (response.status === 429 || response.status === 503) {
      run.dropped++;
      void publish(run, {warnings: ["Локальный анализатор занят. Фрагмент пропущен."]});
      return;
    }
    if (!response.ok) {
      serviceError = `Студия вернула код ${response.status}.`;
      throw new Error(serviceError);
    }
    const result = await response.json();
    if (current !== run) return;
    run.failures = 0;
    void publish(run, {
      ...result,
      startMs,
      requestMs: Math.round(performance.now() - requested),
    });
  } catch (error) {
    if (current !== run) return;
    run.failures++;
    run.dropped++;
    const fallback = error.name === "AbortError" ? "Студия не успела обработать фрагмент." : "Нет ответа от локальной студии.";
    const message = serviceError || fallback;
    if (run.failures >= 3) cleanup(run, message);
    else void publish(run, {warnings: [message]});
  } finally {
    clearTimeout(timeout);
    if (run.request === controller) run.request = null;
  }
}

/** Each recorder produces an independent WebM header so FFmpeg can decode every chunk. */
function record(run) {
  if (current !== run) return;
  const recorder = new MediaRecorder(run.stream, {
    mimeType: run.mimeType, videoBitsPerSecond: 640000, audioBitsPerSecond: 64000,
  });
  run.recorder = recorder;
  const startMs = Math.round(performance.now() - run.started);
  const parts = [];
  recorder.ondataavailable = event => {
    if (current === run && event.data.size) parts.push(event.data);
  };
  recorder.onerror = () => cleanup(run, "Chrome не смог закодировать фрагмент вкладки.");
  recorder.onstop = () => {
    if (current !== run) return;
    const chunk = new Blob(parts, {type: "video/webm"});
    try {
      record(run);
    } catch {
      cleanup(run, "Не удалось продолжить захват вкладки.");
      return;
    }
    void upload(run, chunk, startMs);
  };
  recorder.start();
  run.timer = setTimeout(() => {
    if (current === run && recorder.state !== "inactive") recorder.stop();
  }, CHUNK_MS);
}

/** Play captured sound locally because tabCapture disconnects the tab from its normal output. */
async function enablePlayback(run, stream) {
  if (!stream.getAudioTracks().length) return;
  const monitor = new AudioContext();
  run.monitor = monitor;
  monitor.createMediaStreamSource(stream).connect(monitor.destination);
  let timeout;
  try {
    await Promise.race([monitor.resume(), new Promise((_, reject) => {
      timeout = setTimeout(() => reject(new Error("Chrome не вернул звук вкладки. Захват остановлен.")), 1500);
    })]);
  } finally {
    clearTimeout(timeout);
  }
}

/** Open a new tab media stream and start its recorder loop. */
async function start(message) {
  if (current) throw new Error("Захват уже запущен.");
  const run = {...message, dropped: 0, failures: 0};
  current = run;
  try {
    const capture = {chromeMediaSource: "tab", chromeMediaSourceId: message.streamId};
    const stream = await navigator.mediaDevices.getUserMedia({
      audio: {mandatory: capture},
      video: {mandatory: {...capture, maxWidth: 640, maxHeight: 480, maxFrameRate: 8}},
    });
    if (current !== run) {
      stream.getTracks().forEach(track => track.stop());
      throw new Error("Захват отменён.");
    }
    run.stream = stream;
    for (const track of stream.getTracks()) {
      track.addEventListener("ended", () => cleanup(run, "Источник захвата завершился."), {once: true});
    }
    run.mimeType = ["video/webm;codecs=vp8,opus", "video/webm"].find(type => MediaRecorder.isTypeSupported(type));
    if (!run.mimeType) throw new Error("Эта версия Chromium не поддерживает захват WebM.");
    await enablePlayback(run, stream);
    if (current !== run) throw new Error("Захват отменён.");
    run.started = performance.now();
    record(run);
    return {ok: true};
  } catch (error) {
    cleanup(run);
    throw error;
  }
}

chrome.runtime.onMessage.addListener((message, sender, respond) => {
  if (message.target !== "offscreen" || sender.id !== chrome.runtime.id) return false;
  if (message.type === "stop") {
    if (current?.session === message.session) cleanup(current);
    respond({ok: true});
    return false;
  }
  if (message.type !== "start") return false;
  start(message).then(respond, error => respond({ok: false, error: error.message}));
  return true;
});
