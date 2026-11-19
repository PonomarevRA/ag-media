import {uiText, apiText} from "./i18n.js";
import {attachStudioIcon, attachStudioHint} from "./studio-icons.js";
import {readAnalysisStream} from "./video-analysis.js";

const reasons = {
  "multiple-acoustic-labels": "Несколько голосовых меток",
  "speaker-transition-inside-cue": "Возможная смена говорящего внутри фразы",
  "simultaneous-voices": "Наложение голосов",
  "sparse-text-over-voiced-audio": "Мало текста относительно измеренной речи",
  "cue-boundary-drift": "Граница фразы далеко от речи",
  "unknown-acoustic-label": "Голос не привязан",
  "voiced-audio-without-text": "Измеренная речь без текста",
};

/** Snapshot only audited speech inputs; edits to the same report object must invalidate late results. */
function speechSnapshot(report) {
  const fields = ["segments", "speechTurns", "durationMs", "language", "sourceDigest", "recognitionRecovery"];
  return JSON.stringify(Object.fromEntries(fields.filter(key => Object.hasOwn(report, key))
    .map(key => [key, report[key]])));
}

/** Format source milliseconds without rounding a transition into the following second. */
export function auditTime(ms) {
  const value = Math.max(0, Math.floor(ms));
  return `${Math.floor(value / 60000)}:${String(Math.floor(value / 1000) % 60).padStart(2, "0")}.${String(value % 1000).padStart(3, "0").slice(0, 2)}`;
}

/** Compose the host's native disclosure and existing icon controls; never change raw cue text. */
export function mountDialogueAudit(owner, {currentReport, currentSource, seek}) {
  if (!owner) return {update() {}};
  const summary = document.createElement("summary");
  summary.textContent = uiText("Проверка диалога");
  const controls = document.createElement("div");
  controls.className = "actions dialogue-audit-actions";
  const status = document.createElement("p");
  status.className = "muted";
  status.setAttribute("role", "status");
  const list = document.createElement("ol");
  list.className = "dialogue-audit-list";
  let busy = false;
  let visible = 20;
  let shown = null;
  let controller = null;
  let generation = 0;
  let viewedSource = currentSource();
  let disposed = false;
  let waitTimer = null;
  status.className = "muted dialogue-audit-status";
  status.style?.setProperty("overflow-wrap", "anywhere");
  status.dataset.agWidth = "content";
  status.dataset.agWidthMinEm = "4";
  list.dataset.agWidth = "fill";
  controls.dataset.agWidth = "fill";

  /** Reuse the current Studio icon and tooltip behavior with an explicit accessible action. */
  function action(icon, label, callback) {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "studio-icon";
    button.dataset.icon = icon;
    button.setAttribute("aria-label", label);
    attachStudioIcon(button);
    attachStudioHint(button, label);
    button.onclick = callback;
    controls.append(button);
    return button;
  }
  const inspect = action("technical", uiText("Проверить границы реплик"), () => run(false));
  const retry = action("voices", uiText("Перепроверить аудио проблемных мест"), () => run(true));
  const cancel = action("close", uiText("Прервать ожидание проверки"), () => controller?.abort());
  const more = document.createElement("button");
  more.type = "button";
  more.textContent = uiText("Показать ещё");
  more.onclick = () => {visible += 20; render(shown);};
  owner.append(summary, controls, status, list, more);

  /** Paint only a bounded page of findings; ASR proposals remain separate, labeled drafts. */
  function render(inventory) {
    if (inventory !== shown) visible = 20;
    shown = inventory;
    const candidates = Array.isArray(inventory?.candidates) ? inventory.candidates.slice(0, 2000) : [];
    summary.textContent = uiText("Проверка диалога") + (inventory?.summary ? ` · ${candidates.length}` : "");
    list.replaceChildren();
    for (const item of candidates.slice(0, visible)) {
      if (!item || !Number.isFinite(item.startMs) || !Number.isFinite(item.endMs)) continue;
      const row = document.createElement("li");
      row.style?.setProperty("display", "flex");
      row.style?.setProperty("flex-wrap", "wrap");
      const stamp = document.createElement("button");
      stamp.className = "dialogue-audit-time";
      stamp.type = "button";
      stamp.textContent = auditTime(item.startMs);
      stamp.title = `${auditTime(item.startMs)}–${auditTime(item.endMs)}`;
      stamp.onclick = () => seek(item.startMs);
      const description = document.createElement("span");
      description.className = "dialogue-audit-description";
      description.dataset.agWidth = "content";
      description.dataset.agWidthMinEm = "4";
      description.style?.setProperty("flex", "1 1 14em");
      description.style?.setProperty("min-width", "0");
      const codes = Array.isArray(item.reasons) ? item.reasons.filter(value => typeof value === "string").slice(0, 12) : [];
      description.textContent = codes.map(reason => uiText(reasons[reason] ?? reason)).join(" · ");
      row.append(stamp, description);
      const drafts = Array.isArray(inventory?.repairs?.suggestions) ? inventory.repairs.suggestions : [];
      const suggestions = drafts.filter(value => value?.candidateId === item.id);
      if (suggestions.length) {
        const drafts = document.createElement("details");
        drafts.className = "dialogue-audit-drafts";
        drafts.dataset.agWidth = "fill";
        drafts.style?.setProperty("flex", "1 1 100%");
        drafts.style?.setProperty("min-width", "0");
        drafts.style?.setProperty("overflow-wrap", "anywhere");
        const label = document.createElement("summary");
        label.textContent = uiText("Варианты ASR · требуют проверки");
        drafts.append(label);
        for (const suggestion of suggestions.slice(0, 20)) {
          const text = document.createElement("p");
          text.textContent = `${auditTime(suggestion.startMs)} · ${suggestion.text}`;
          drafts.append(text);
        }
        row.append(drafts);
      }
      list.append(row);
    }
    more.hidden = candidates.length <= visible;
    inspect.disabled = retry.disabled = busy || !currentReport() || !currentSource();
    cancel.hidden = !busy;
    if (!busy) {
      const failures = {timeout: "Истёк лимит проверки. Предыдущие данные сохранены.",
        failed: "Перепроверка не завершена. Исходные данные сохранены.",
        cancelled: "Перепроверка прервана. Исходные данные сохранены."};
      const state = failures[inventory?.status] ?? failures[inventory?.repairs?.status];
      if (state) {
        const detail = typeof inventory?.error === "string" ? inventory.error.slice(0, 300) : "";
        status.textContent = uiText(state) + (detail ? ` ${apiText(detail)}` : "");
      } else if (inventory?.status === "unavailable") {
        status.textContent = apiText(inventory.message ?? uiText("Для проверки нужны измеренные голосовые интервалы."));
      } else if (candidates.length) {
        status.textContent = uiText("Кандидаты требуют проверки. Исходная расшифровка сохранена.");
      } else {
        status.textContent = inventory ? uiText("Проверка завершена. Структурных кандидатов не найдено.")
          : uiText("Запусти отдельную проверку. Видео останется доступным.");
      }
    }
  }

  /** Upload source plus the small speech-only report; fetch streams the Blob without a full RAM copy. */
  async function run(rerun) {
    const source = currentSource();
    const report = currentReport();
    if (!source || !report || busy || disposed) return;
    const requestGeneration = generation;
    let failure = null;
    const auditedSpeech = speechSnapshot(report);
    const metadata = new TextEncoder().encode(auditedSpeech);
    const length = new Uint8Array(4);
    new DataView(length.buffer).setUint32(0, metadata.length);
    const requestController = new AbortController();
    controller = requestController;
    let timedOut = false;
    waitTimer = setTimeout(() => {
      if (requestGeneration !== generation || controller !== requestController) return;
      timedOut = true;
      requestController.abort();
    }, 190000);
    busy = true;
    owner.open = true;
    render(shown);
    status.textContent = uiText("Проверка голосовых границ…");
    try {
      const response = await fetch(`/api/dialogue-audit?progress=1&rerun=${Number(rerun)}`, {
        method: "POST", headers: {"X-Voice-Studio": "1"},
        body: new Blob([length, metadata, source]), signal: requestController.signal,
      });
      if (!response.ok) throw new Error(apiText((await response.json()).error));
      const result = await readAnalysisStream(response, event => {
        if (requestGeneration !== generation || source !== currentSource() || disposed) return;
        if (event.candidateCount !== undefined) status.textContent = uiText("Найдено кандидатов: ") + event.candidateCount;
        else if (event.stage === "dialogue-audit") status.textContent = uiText("Перепроверка выбранных голосовых отрезков…");
      });
      if (source === currentSource() && requestGeneration === generation && currentReport()) {
        if (auditedSpeech === speechSnapshot(currentReport())) {
          currentReport().dialogueAudit = result;
        } else {
          failure = uiText("Расшифровка изменилась во время проверки. Устаревший результат отброшен; запусти проверку заново.");
        }
      }
    } catch (error) {
      failure = timedOut ? uiText("Истёк лимит ожидания. Предыдущий результат сохранён.")
        : error.name === "AbortError" ? uiText("Ожидание прервано. Предыдущий результат сохранён.") : error.message;
    } finally {
      if (requestGeneration !== generation || controller !== requestController || disposed) return;
      clearTimeout(waitTimer);
      waitTimer = null;
      busy = false;
      controller = null;
      inspect.disabled = retry.disabled = !currentSource() || !currentReport();
      cancel.hidden = true;
      render(currentReport()?.dialogueAudit ?? shown);
      if (failure && requestGeneration === generation) status.textContent = failure;
    }
  }

  /** Invalidate the waiting client immediately; an old completion cannot repaint a new source. */
  function invalidate() {
    generation++;
    controller?.abort();
    controller = null;
    clearTimeout(waitTimer);
    waitTimer = null;
    busy = false;
    visible = 20;
  }

  return {
    /** Refresh host evidence; source resets invalidate outstanding client work. */
    update(report) {
      if (disposed) return;
      const source = currentSource();
      if (!report || source !== viewedSource) invalidate();
      viewedSource = source;
      owner.hidden = !report;
      render(report?.dialogueAudit ?? null);
    },
    /** Abort waiting and detach the native disclosure from its host. */
    dispose() {
      if (disposed) return;
      invalidate();
      disposed = true;
      owner.replaceChildren();
      owner.hidden = true;
    },
  };
}
