import {uiText} from "./i18n.js";

export const ANALYSIS_STAGE_CHOICES = [
  ["speech", "Речь и таймкоды"], ["audio", "Музыка, фон и интонация"],
  ["visual", "Лица и говорящие в кадре"], ["people", "Фигуры и взгляд"],
  ["scene", "Надписи, предметы и камера"], ["voice", "Голосовые профили"],
  ["object_review", "Сверка предметов по ракурсам"],
];
const STATES = {
  done: "Готово", reused: "Из кэша", pending: "Ещё не выполнен",
  invalidated: "Нужен пересчёт", failed: "Ошибка",
};

/** Validate a bounded stage selection before serializing it to the shared API. */
export function stageSelectionQuery(selected, force = false) {
  if (selected === null) return {};
  const allowed = new Set(ANALYSIS_STAGE_CHOICES.map(([id]) => id));
  if (!Array.isArray(selected) || selected.some(id => !allowed.has(id)) || new Set(selected).size !== selected.length) {
    throw new Error("Invalid analysis stage selection");
  }
  const value = selected.join(",");
  return {selected_stages: value, ...(force ? {force_stages: value} : {})};
}

/** Match a manual speaker choice only to the unchanged source cue, never a nearby new phrase. */
export function speakerEditKey(cue) {
  return JSON.stringify([cue.startMs, cue.endMs, cue.text]);
}

/** Carry explicit user choices into a retained-source rerun without pinning automatic model labels. */
export function retainedSpeakerEdits(report) {
  const edits = new Map();
  for (const cue of report?.segments ?? []) {
    if (cue.speakerNumberEvidence?.source === "user" && Number.isSafeInteger(cue.speakerNumber)
        && cue.speakerNumber > 0) edits.set(speakerEditKey(cue), String(cue.speakerNumber));
  }
  return edits;
}

/**
 * Mount the existing settings-style native controls for partial analysis.
 * The host owns the source and execution; this panel only selects stages and shows server states.
 * @param {function(string):HTMLElement} find Host element lookup.
 * @param {function(string[], boolean):Promise<void>} run Execute selected IDs, optionally bypassing their cache.
 * @returns {object} Source/busy/report setters and the selection query for a new upload.
 */
export function mountAnalysisStageSelection(find, run) {
  const manual = find("media-stage-manual");
  const list = find("media-stage-choices");
  const start = find("media-stage-run");
  const force = find("media-stage-force");
  const sourceNote = find("media-stage-source-note");
  const controls = new Map();
  let busy = false;
  let hasSource = false;

  /** Apply source availability and global work ownership without altering saved selections. */
  function refresh() {
    manual.disabled = busy;
    for (const {check, repeat} of controls.values()) {
      check.disabled = busy;
      repeat.disabled = busy || !hasSource;
    }
    const empty = !selected().length;
    start.disabled = force.disabled = busy || !hasSource || empty;
    sourceNote.hidden = hasSource;
  }

  function selected() {
    return [...controls].filter(([, row]) => row.check.checked).map(([id]) => id);
  }

  for (const [id, label] of ANALYSIS_STAGE_CHOICES) {
    const row = document.createElement("div");
    row.className = "media-stage-choice";
    row.dataset.stage = id;
    const field = document.createElement("label");
    field.className = "inline-check";
    const check = document.createElement("input");
    check.type = "checkbox";
    check.checked = id !== "object_review";
    check.value = id;
    check.addEventListener("change", () => { manual.checked = true; refresh(); });
    field.append(check, document.createTextNode(uiText(label)));
    const state = document.createElement("small");
    state.className = "muted";
    state.textContent = uiText("Ещё не выполнен");
    const repeat = document.createElement("button");
    repeat.type = "button";
    repeat.className = "studio-icon media-stage-repeat";
    repeat.textContent = "↻";
    const hint = `${uiText("Пересчитать этап")}: ${uiText(label)}`;
    repeat.setAttribute("aria-label", hint);
    repeat.title = hint;
    repeat.addEventListener("click", () => { if (!busy && hasSource) run([id], true); });
    row.append(field, state, repeat);
    controls.set(id, {check, state, repeat});
    list.append(row);
  }
  start.addEventListener("click", () => { if (!start.disabled) run(selected(), false); });
  force.addEventListener("click", () => { if (!force.disabled) run(selected(), true); });
  refresh();

  return {
    query: () => stageSelectionQuery(manual.checked ? selected() : null),
    setBusy(value) { busy = Boolean(value); refresh(); },
    setSource(value) { hasSource = Boolean(value); refresh(); },
    /** Show authoritative completion separately from a user's next selection. */
    setReport(report) {
      for (const [id, {state}] of controls) {
        const info = report?.analysisStages?.[id];
        const value = typeof info === "string" ? info : info?.state;
        state.textContent = uiText(STATES[value] ?? "Ещё не выполнен");
        state.dataset.state = value ?? "pending";
        state.title = typeof info?.reason === "string" ? info.reason : "";
      }
    },
  };
}
