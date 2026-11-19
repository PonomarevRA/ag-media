import {apiText, uiText} from "./i18n.js";
const STAGES = [
  ["upload", uiText("Загрузка файла"), 1],
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
  people: ["person-layer"], scene: ["scene"],
  voice: ["voice-profiles"], object_review: ["object-review"],
};

/**
 * Distinguish checkpoints from shared inference without hiding failed/skipped states.
 * @param {object} event Actual server stage event, including optional cacheHit and elapsedMs.
 * @param {string|undefined} cacheMode Previously observed cache decision for this worker group.
 * @returns {string} Localized status text for the existing progress row.
 */
export function analysisStageStatus(event, cacheMode) {
  if (event.cacheHit === true) {
    if (event.state === "skipped") return uiText("Из кэша · пропущен");
    return event.state === "partial" ? uiText("Из кэша · частично") : uiText("Из кэша");
  }
  if (event.reused) return uiText("Готово в общем процессе");
  if (cacheMode === "recompute" && event.state === "running") return uiText("Пересчитывается");
  const measured = typeof event.elapsedMs === "number" && ["done", "partial"].includes(event.state);
  const label = cacheMode === "recompute" && event.state === "done" ? uiText("Пересчитано") : STATE[event.state];
  return label + (measured ? uiText` · ${(event.elapsedMs / 1000).toFixed(1)} с` : "");
}

const REPORT_FIELDS = {
  speech: ["segments", "text", "srt", "language", "durationMs", "languageMode", "recognitionModel",
    "recognitionRecovery", "recognitionWarning", "dialoguePauses", "speechOverlaps", "speakerCount",
    "speakerWarning", "corrections", "contextVersion", "speechTurns", "dialogueAudit"],
  voice: ["segments", "voiceProfiles", "speakerProfileIndex", "voiceProfileWarning", "dialogueAudit", "srt"],
  audio: ["audioLayers", "soundAnalysisStatus", "soundClassifier"],
  visual: ["videoEvidence", "segments", "speakerReview", "srt"],
  scene: ["sceneEvidence", "shotColors", "objectTimeline", "objectProfiles", "videoEvidence",
    "segments", "language", "durationMs", "text", "srt", "corrections"],
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
        if (event.error) throw new Error(apiText(event.error));
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

/** Display actual server stages and an approximate, explicitly qualified time estimate. */
export function mountMediaAnalysis(find) {
  const region = find("media-analysis");
  const meter = find("media-analysis-meter");
  const status = find("media-analysis-status");
  const list = find("media-analysis-stages");
  const rows = new Map();
  const cacheModes = new Map();
  let timer;
  let duration;
  let started;

  function estimate() {
    const pending = [...rows.values()].filter(row => ["queued", "running"].includes(row.state));
    const active = pending.filter(row => row.state === "running");
    const current = active[0];
    if (!current) return;
    if (duration === undefined || current.id === "upload") {
      meter.removeAttribute("value");
      status.textContent = `${current.label.textContent}…`;
      return;
    }
    if (current.id === "whisper-recovery" && current.started !== undefined) {
      const seconds = Math.floor((performance.now() - current.started) / 1000);
      const passed = uiText`прошло ${seconds} с`;
      status.textContent = [current.label.textContent, current.recoveryProgress, passed].filter(Boolean).join(" · ");
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
    status.textContent = `${active.map(row => row.label.textContent).join(" + ")} · ${hint}`;
  }

  function update(event) {
    if (typeof event.durationMs === "number") duration = event.durationMs / 1000;
    if (Object.hasOwn(CACHE_LAYERS, event.cacheLayer) && typeof event.cacheHit === "boolean") {
      for (const id of CACHE_LAYERS[event.cacheLayer]) cacheModes.set(id, event.cacheHit ? "cached" : "recompute");
    }
    if (event.stage === "whisper-recovery" && ["running", "done", "partial"].includes(event.state) && !rows.has(event.stage)) {
      const [, name, base, rate] = STAGES.find(([id]) => id === event.stage);
      const line = document.createElement("li");
      const label = document.createElement("span");
      label.textContent = name;
      const detail = document.createElement("span");
      line.append(label, detail);
      rows.get("whisper")?.line.after(line);
      rows.set(event.stage, {id: event.stage, line, label, detail, base, rate, state: "queued"});
    }
    const row = rows.get(event.stage);
    if (row) {
      row.state = event.state;
      row.line.dataset.state = event.state;
      row.line.dataset.cache = event.cacheHit === true ? "hit" : cacheModes.get(event.stage) || "none";
      if (event.state === "running" && row.started === undefined) row.started = performance.now();
      if (event.detail && row.id === "whisper") row.label.textContent = event.detail + uiText(" · текст и таймкоды");
      row.detail.textContent = analysisStageStatus(event, cacheModes.get(event.stage));
      if (event.stage === "whisper-recovery") {
        row.recoveryProgress = recoveryProgress(event);
        if (row.recoveryProgress) row.detail.textContent += " · " + row.recoveryProgress;
      }
      if (event.totalChunks > 1) row.detail.textContent += uiText` · блок ${event.chunk}/${event.totalChunks}`;
    }
    estimate();
  }

  function reset() {
    clearInterval(timer);
    rows.clear();
    cacheModes.clear();
    duration = undefined;
    region.hidden = true;
    list.replaceChildren();
    meter.removeAttribute("value");
  }

  return {
    reset,
    update,
    start(flags) {
      reset();
      const enabled = new Set(["upload", "audio", "whisper", "dialogue-pauses", "audio-layers", "speaker-context"]);
      if (flags.speakers === "1") enabled.add("diarization").add("voice-profiles").add("dialogue-audit");
      if (flags.visual === "1") enabled.add("light-asd").add("yunet");
      if (flags.visual === "1" && flags.speakers === "1" && !flags.speaker_count) enabled.add("speaker-review");
      if (flags.body === "1" || flags.gaze === "1") enabled.add("person-layer");
      if (flags.scene === "1") enabled.add("scene");
      if (flags.object_reconciliation === "1" || flags.selected_stages?.split(",").includes("object_review")) enabled.add("object-review");
      for (const layer of (flags.selected_stages ?? "").split(",")) {
        for (const id of CACHE_LAYERS[layer] ?? []) enabled.add(id);
      }
      for (const [id, name, base, rate = 0] of STAGES) {
        if (!enabled.has(id)) continue;
        const line = document.createElement("li");
        const label = document.createElement("span");
        label.textContent = id === "whisper" && flags.quality === "1" ? uiText("Whisper Small · текст и таймкоды") : name;
        const detail = document.createElement("span");
        detail.textContent = STATE.queued;
        line.dataset.state = "queued";
        line.append(label, detail);
        rows.set(id, {id, line, label, detail, base, rate, state: "queued"});
        list.append(line);
      }
      region.hidden = false;
      region.open = true;
      meter.hidden = false;
      started = performance.now();
      update({stage: "upload", state: "running"});
      timer = setInterval(estimate, 1000);
    },
    finish(error, stageReport) {
      // The final manifest distinguishes deliberately pending work from skipped workers.
      if (!error && stageReport) {
        for (const [layer, info] of Object.entries(stageReport)) {
          const state = typeof info === "string" ? info : info?.state;
          if (!state) continue;
          for (const id of CACHE_LAYERS[layer] ?? []) {
            if (["pending", "invalidated", "failed"].includes(state)) update({stage: id, state});
          }
        }
      }
      clearInterval(timer);
      for (const row of rows.values()) {
        if (row.state === "running") update({stage: row.id, state: error ? "failed" : "done"});
        if (row.state === "queued") update({stage: row.id, state: "skipped"});
      }
      if (error) {
        meter.removeAttribute("value");
        meter.hidden = true;
        status.textContent = uiText("Анализ остановлен");
      } else {
        meter.hidden = false;
        meter.value = 100;
        region.open = false;
        status.textContent = uiText`Анализ завершён за ${Math.round((performance.now() - started) / 1000)} с`;
      }
    },
  };
}
