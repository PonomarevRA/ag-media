import {apiText, uiText} from "./i18n.js";
import {mountAnalysisStepWindow} from "./analysis-step-window.js";
import {attachStudioHint} from "./studio-icons.js";
const STAGES = [
  ["upload", uiText("Загрузка файла"), 1],
  ["source-probe", uiText("Проверка файла"), .3],
  ["audio", uiText("FFmpeg · извлечение звука"), 1],
  ["diarization", uiText("Sherpa · голоса"), 1.5, .045],
  ["whisper", uiText("Whisper · текст и таймкоды"), 1, .12],
  ["whisper-recovery", uiText("Whisper Turbo · повтор проблемных фраз"), 1, .18],
  ["dialogue-pauses", uiText("Паузы · измерение тишины"), .3, .004],
  ["audio-layers", uiText("Звук · музыка, фон и интонация"), .5, .012],
  ["light-asd", uiText("YuNet + Light-ASD · кто говорит в кадре"), 1, .64],
  ["yunet", uiText("YuNet · фото спикеров"), 1, .045],
  ["speaker-review", uiText("Повторная проверка · цифровые профили"), 1, .065],
  ["voice-profiles", uiText("Голосовые профили · аудио и видео"), .4, .012],
  ["dialogue-audit", uiText("Диалог · проверка границ и фраз"), .1, .001],
  ["person-layer", uiText("Технический слой · фигуры и взгляд"), 1, .09],
  ["object-review", uiText("Сверка предметов по ракурсам"), .2],
  ["scene", uiText("Tesseract + NanoDet · надписи и предметы"), 1, .23],
  ["speaker-context", uiText("Контекстный граф · сохранённые метки говорящих"), .2],
];

const STATE = {queued: uiText("Ожидает"), running: uiText("В работе"), done: uiText("Готово"), skipped: uiText("Пропущен"), partial: uiText("Частично"), failed: uiText("Ошибка"), pending: uiText("Ещё не выполнен"), invalidated: uiText("Нужен пересчёт")};
const CACHE_LAYERS = {
  speech: ["diarization", "whisper", "whisper-recovery", "dialogue-pauses"],
  audio: ["audio-layers"], visual: ["light-asd", "yunet", "speaker-review"],
  "speaker-review": ["speaker-review"], people: ["person-layer"], scene: ["scene"],
  voice: ["voice-profiles"], object_review: ["object-review"],
};
const RETRY_GROUPS = {audio: "speech", "dialogue-audit": "voice"};
for (const [group, stages] of Object.entries(CACHE_LAYERS)) {
  if (group !== "speaker-review") for (const stage of stages) RETRY_GROUPS[stage] = group;
}
const SYMBOLS = {queued: "·", running: "…", done: "✓", skipped: "↷", partial: "!", failed: "×", pending: "·", invalidated: "↻"};

/**
 * Distinguish checkpoints from shared inference without hiding failed/skipped states.
 * @param {object} event Actual server stage event, including optional cacheHit and elapsedMs.
 * @param {string|undefined} cacheMode Previously observed cache decision for this worker group.
 * @returns {string} Localized status text for the existing progress row.
 */
export function analysisStageStatus(event, cacheMode) {
  if (event.state === "failed") return STATE.failed + (event.reason ? " · " + apiText(event.reason) : "");
  if (event.cacheHit === true) {
    if (event.state === "skipped") {
      return uiText("Из кэша · пропущен") + (event.reason ? " · " + apiText(event.reason) : "");
    }
    return event.state === "partial" ? uiText("Из кэша · частично") : uiText("Из кэша");
  }
  if (event.reused) return uiText("Готово в общем процессе");
  if (cacheMode === "recompute" && event.state === "running") return uiText("Пересчитывается");
  const measured = typeof event.elapsedMs === "number" && ["done", "partial"].includes(event.state);
  const label = cacheMode === "recompute" && event.state === "done" ? uiText("Пересчитано") : STATE[event.state];
  const time = measured ? uiText` · ${(event.elapsedMs / 1000).toFixed(1)} с` : "";
  return label + time + (event.reason ? " · " + apiText(event.reason) : "");
}

const REPORT_FIELDS = {
  speech: ["segments", "text", "srt", "language", "durationMs", "languageMode", "recognitionModel",
    "recognitionRecovery", "recognitionWarning", "dialoguePauses", "speechOverlaps", "speakerCount",
    "speakerWarning", "corrections", "contextVersion", "speechTurns", "dialogueAudit"],
  voice: ["segments", "voiceProfiles", "speakerProfileIndex", "voiceProfileWarning", "dialogueAudit", "srt"],
  audio: ["audioLayers", "soundAnalysisStatus", "soundClassifier"],
  visual: ["videoEvidence", "segments", "speakerReview", "srt"],
  "scene-preview": ["sceneEvidence"],
  scene: ["sceneEvidence", "shotColors", "objectTimeline", "objectProfiles", "videoEvidence",
    "segments", "language", "durationMs", "text", "srt", "corrections", "ocrTranscriptChecks"],
};

/** Apply a known report layer only after the base and keep earlier layers available. */
export function mergeAnalysisPartial(current, update) {
  const report = update?.report;
  if (!report || typeof report !== "object" || Array.isArray(report)) return current;
  if (update.phase === "base") {
    return current === null && Array.isArray(report.segments) ? {...report, analysisComplete: false} : current;
  }
  if (!current || update.phase !== "detail" || !Object.hasOwn(REPORT_FIELDS, update.layer)) return current;
  const fields = REPORT_FIELDS[update.layer].filter(field => Object.hasOwn(report, field)
    && (field !== "segments" || Array.isArray(report.segments)));
  if (!fields.length) return current;
  const next = {...current};
  for (const field of fields) next[field] = report[field];
  return next;
}

/** Read real stage events as they arrive and keep the final transcript contract. */
export async function readAnalysisStream(response, update) {
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  try {
    for (;;) {
      const {value, done} = await reader.read();
      if (done) throw new Error(uiText("Анализ прервался до получения результата."));
      buffer += decoder.decode(value, {stream: true});
      let end;
      while ((end = buffer.indexOf("\n")) >= 0) {
        const line = buffer.slice(0, end);
        buffer = buffer.slice(end + 1);
        if (!line) continue;
        const event = JSON.parse(line);
        if (!event.stage && typeof event.error === "string" && event.error) {
          if (event.pipeline) update({pipeline: event.pipeline});
          throw new Error(apiText(event.error));
        }
        if (event.result) return event.result;
        update(event);
      }
    }
  } finally {
    await reader.cancel().catch(() => {});
    reader.releaseLock();
  }
}

/** Format measured retry progress independently of the general ETA. */
export function recoveryProgress(event) {
  if (!Number.isInteger(event.windowsDone) || !Number.isInteger(event.windowsTotal) ||
      event.windowsDone < 0 || event.windowsTotal < 1 || event.windowsDone > event.windowsTotal) return "";
  const progress = uiText`фрагменты ${event.windowsDone} из ${event.windowsTotal}`;
  return event.recoveryPass === "alternate" ? uiText`другой язык · ${progress}` : progress;
}

/** Show wall time independently of accumulated work in parallel stages. */
export function pipelineDuration(ms) {
  if (!Number.isFinite(ms) || ms < 0) return "";
  if (ms < 60_000) return uiText`${(ms / 1000).toFixed(1)} с`;
  const seconds = Math.floor(ms / 1000);
  return [Math.floor(seconds / 60), String(seconds % 60).padStart(2, "0")].join(":");
}

/** Display actual server stages, measured wall time and the current ETA. */
export function mountMediaAnalysis(find, retry) {
  const region = find("media-analysis");
  const clock = find("media-analysis-elapsed");
  const meter = find("media-analysis-meter");
  const status = find("media-analysis-status");
  const list = find("media-analysis-stages");
  const details = mountAnalysisStepWindow(find, retry);
  const rows = new Map();
  const cacheModes = new Map();
  let timer;
  let duration;
  let started;
  let terminal;

  function estimate() {
    if (started !== undefined) clock.textContent = "· " + pipelineDuration(performance.now() - started);
    for (const row of rows.values()) {
      if (row.state !== "running" || row.started === undefined) continue;
      row.elapsed = pipelineDuration(performance.now() - row.started);
      details.update(row);
    }
    const pending = [...rows.values()].filter(row => ["queued", "running"].includes(row.state));
    const active = pending.filter(row => row.state === "running");
    const current = active[0];
    if (!current) {
      const waiting = pending.find(row => row.waitingReason);
      if (waiting) {
        status.textContent = [waiting.name, apiText(waiting.waitingReason)].join(" · ");
        return;
      }
      status.textContent = pending.length ? uiText("Подготовка следующего этапа…") : uiText("Собираем результат…");
      return;
    }
    if (duration === undefined || current.id === "upload") {
      meter.removeAttribute("value");
      status.textContent = `${current.name}…`;
      return;
    }
    if (current.id === "whisper-recovery" && current.started !== undefined) {
      const seconds = Math.floor((performance.now() - current.started) / 1000);
      const passed = uiText`прошло ${seconds} с`;
      status.textContent = [current.name, current.progress, passed].filter(Boolean).join(" · ");
      return;
    }
    let remaining = 0;
    let activeRemaining = 0;
    let overdue = false;
    for (const row of pending) {
      const expected = row.base + Math.min(duration, 900) * row.rate;
      const spent = row.started === undefined ? 0 : (performance.now() - row.started) / 1000;
      const left = Math.max(0, expected - spent);
      if (row.state === "running") {
        activeRemaining = Math.max(activeRemaining, left);
        overdue ||= spent > expected;
      } else remaining += left;
    }
    meter.value = 100 * (rows.size - pending.length) / rows.size;
    const rounded = Math.max(5, Math.ceil((remaining + activeRemaining) / 5) * 5);
    const hint = overdue ? uiText("время уточняется") : uiText`ещё около ${rounded} с`;
    status.textContent = `${active.map(row => row.name).join(" + ")} · ${hint}`;
  }

  /** Add a compact status button while preserving the order of known pipeline steps. */
  function addRow(id, name, base, rate = 0) {
    const line = document.createElement("li");
    line.dataset.stage = id;
    const button = document.createElement("button");
    button.type = "button";
    button.className = "studio-icon pipeline-step";
    button.setAttribute("aria-haspopup", "dialog");
    button.setAttribute("aria-controls", "media-step-window");
    attachStudioHint(button, "");
    const placeHint = () => {
      const root = button.ownerDocument.documentElement;
      const bounds = button.getBoundingClientRect();
      button.dataset.hintSide = root.clientWidth - bounds.left < root.clientWidth * .48 ? "end" : "start";
    };
    button.addEventListener("pointerenter", placeHint);
    button.addEventListener("focus", placeHint);
    line.append(button);
    const row = {id, line, button, name, base, rate, retryGroup: RETRY_GROUPS[id]};
    button.onclick = () => details.open(row, button);
    rows.set(id, row);
    if (id === "whisper-recovery" && rows.has("whisper")) rows.get("whisper").line.after(line);
    else list.append(line);
    return row;
  }

  /** Update the actual worker outcome and any diagnostic delivered by the server. */
  function applyStep(event) {
    if (!event.stage || !Object.hasOwn(SYMBOLS, event.state)) return;
    const definition = STAGES.find(([id]) => id === event.stage);
    if (!rows.has(event.stage) && !definition) return;
    const row = rows.get(event.stage) ?? addRow(...definition);
    row.state = event.state;
    row.reason = event.reason;
    row.waitingReason = row.state === "queued" ? row.reason : undefined;
    if (row.state === "queued") row.started = undefined;
    if (row.state === "running" && row.started === undefined) row.started = performance.now();
    if (event.detail && row.id === "whisper") row.name = event.detail + uiText(" · текст и таймкоды");
    row.error = event.error || (row.state === "failed" && {message: event.reason || uiText("Этап завершился с ошибкой.")});
    row.lastFailure = event.lastFailure || row.error || row.lastFailure;
    row.windowStartMs = event.windowStartMs;
    row.retryGroup = event.retryGroup || row.retryGroup;
    row.elapsed = pipelineDuration(event.elapsedMs);
    row.status = analysisStageStatus({state: event.state, cacheHit: event.cacheHit, reused: event.reused}, cacheModes.get(row.id));
    row.progress = row.id === "whisper-recovery" ? recoveryProgress(event) : "";
    if (event.totalChunks > 1) row.progress = uiText`Блок ${event.chunk} из ${event.totalChunks}`;
    row.line.dataset.state = row.state;
    row.line.dataset.cache = event.cacheHit === true ? "hit" : cacheModes.get(row.id) || "none";
    row.button.textContent = SYMBOLS[row.state];
    const label = [row.name, row.status, row.elapsed].filter(Boolean).join(" · ");
    row.button.setAttribute("aria-label", label);
    row.button.dataset.hint = label;
    details.update(row);
  }

  function update(event) {
    if (event.pipeline) {
      terminal = event.pipeline;
      return;
    }
    if (event.runId) region.dataset.runId = event.runId;
    if (typeof event.durationMs === "number") duration = event.durationMs / 1000;
    if (Object.hasOwn(CACHE_LAYERS, event.cacheLayer) && typeof event.cacheHit === "boolean") {
      for (const id of CACHE_LAYERS[event.cacheLayer]) cacheModes.set(id, event.cacheHit ? "cached" : "recompute");
    }
    applyStep(event);
    estimate();
  }

  function reset() {
    clearInterval(timer);
    rows.clear();
    cacheModes.clear();
    duration = undefined;
    started = terminal = undefined;
    delete region.dataset.runId;
    clock.textContent = "";
    const returnStep = details.close();
    region.hidden = true;
    list.replaceChildren();
    meter.removeAttribute("value");
    return returnStep;
  }

  return {
    reset,
    update,
    setAvailable: details.setAvailable,
    start(flags) {
      const returnStep = reset();
      const enabled = new Set(["upload", "source-probe", "audio", "whisper", "dialogue-pauses", "audio-layers", "speaker-context"]);
      if (flags.speakers === "1") enabled.add("diarization").add("voice-profiles").add("dialogue-audit");
      if (flags.visual === "1") enabled.add("light-asd").add("yunet");
      if (flags.visual === "1" && flags.speakers === "1" && !flags.speaker_count) enabled.add("speaker-review");
      if (flags.body === "1" || flags.gaze === "1") enabled.add("person-layer");
      if (flags.scene === "1") enabled.add("scene");
      if (flags.object_reconciliation === "1" || flags.selected_stages?.split(",").includes("object_review")) enabled.add("object-review");
      for (const layer of (flags.selected_stages ?? "").split(",")) {
        for (const id of CACHE_LAYERS[layer] ?? []) enabled.add(id);
      }
      for (const [id, name, base, rate] of STAGES) {
        if (!enabled.has(id)) continue;
        addRow(id, id === "whisper" && flags.quality === "1" ? uiText("Whisper Small · текст и таймкоды") : name, base, rate);
        applyStep({stage: id, state: "queued"});
      }
      region.hidden = false;
      meter.hidden = false;
      started = performance.now();
      update({stage: "upload", state: "running"});
      if (returnStep) rows.get(returnStep)?.button.focus({preventScroll: true});
      timer = setInterval(estimate, 1000);
    },
    finish(error, stageReport, pipeline = terminal) {
      clearInterval(timer);
      if (pipeline?.runId) region.dataset.runId = pipeline.runId;
      for (const event of pipeline?.stages ?? []) applyStep(event);
      // The final manifest distinguishes deliberately pending work from skipped workers.
      if (!error && stageReport) {
        for (const [layer, info] of Object.entries(stageReport)) {
          const state = typeof info === "string" ? info : info?.state;
          if (!state) continue;
          for (const id of CACHE_LAYERS[layer] ?? []) {
            const observed = rows.get(id)?.state;
            if (["done", "failed", "partial", "skipped"].includes(observed)) continue;
            if (["pending", "invalidated", "failed"].includes(state)) applyStep({stage: id, state, reason: info?.reason});
          }
        }
      }
      for (const row of rows.values()) {
        if (row.state === "running") applyStep({stage: row.id, state: error ? "failed" : "done", reason: error});
        if (row.state === "queued") applyStep({stage: row.id, state: error ? "pending" : "skipped"});
      }
      const wallTime = pipeline?.elapsedMs ?? (performance.now() - started);
      clock.textContent = "· " + pipelineDuration(wallTime);
      if (error) {
        meter.removeAttribute("value");
        meter.hidden = true;
        status.textContent = uiText("Анализ остановлен");
      } else {
        meter.hidden = false;
        meter.value = 100;
        const hasIssues = [...rows.values()].some(row => ["failed", "partial"].includes(row.state) || row.lastFailure
          || (row.id === "person-layer" && row.state === "skipped" && row.reason));
        status.textContent = uiText(hasIssues ? "Анализ завершён с замечаниями" : "Анализ завершён");
      }
    },
  };
}
