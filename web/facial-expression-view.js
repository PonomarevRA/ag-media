import {createMomentTimeline} from "./profile-moment-timeline.js";
import {uiLocale, uiText} from "./i18n.js";
import {expressionLabel, expressionMarker} from "./facial-expressions.js";

const number = new Intl.NumberFormat(uiLocale, {maximumFractionDigits: 2});
const percent = new Intl.NumberFormat(uiLocale, {style: "percent", maximumFractionDigits: 1});
const duration = ms => uiText`${number.format(ms / 1000)} с`;

/** Create plain text nodes; report contents never become markup. */
function node(tag, text, className = "") {
  const element = document.createElement(tag);
  element.textContent = text;
  if (className) element.className = className;
  return element;
}

/** Format a source time in milliseconds for an exact seek button. */
function clock(ms) {
  const seconds = Math.floor(ms / 1000);
  return `${Math.floor(seconds / 60)}:${String(seconds % 60).padStart(2, "0")}.${String(ms % 1000).padStart(3, "0")}`;
}

/** Provide a native keyboard button that seeks the same original source video. */
function moment(startMs, endMs, seek, label) {
  const text = `${clock(startMs)}–${clock(endMs)}`;
  const button = node("button", label ? `${text} · ${label}` : text);
  button.type = "button";
  button.title = uiText`К моменту ${clock(startMs)}`;
  button.setAttribute("aria-label", `${button.title} · ${label ?? text}`);
  button.onclick = () => seek(startMs);
  return button;
}

/** Explain each measured denominator and retain unknown visible time as a separate amount. */
function statistics(face) {
  const group = node("dl", "", "speaker-profile-facts");
  const rows = [["Лицо видно", duration(face.observedMs)], ["Классифицировано", duration(face.classifiedMs)],
    ["Не классифицировано", duration(face.unknownMs)],
    ["Покрытие классификацией", face.observedMs > 0 ? percent.format(face.classifiedMs / face.observedMs) : uiText("Нет данных")]];
  for (const [label, value] of rows) group.append(node("dt", uiText(label)), node("dd", value));
  for (const [label, value] of Object.entries(face.byLabel ?? {})) {
    if (!(value.durationMs > 0) || !Number.isFinite(value.share)) continue;
    group.append(node("dt", expressionLabel(label)), node("dd", `${duration(value.durationMs)} · ${percent.format(value.share)}`));
  }
  return group;
}

/** Add twenty rows per explicit request so long recordings keep a bounded initial DOM. */
function pagedRows(container, rows, render, moreLabel) {
  let offset = 0;
  const more = node("button", moreLabel);
  more.type = "button";
  /** Retain keyboard position at the first newly revealed observation. */
  function appendPage() {
    const restoreFocus = document.activeElement === more;
    let first;
    more.remove();
    const end = Math.min(rows.length, offset + 20);
    for (; offset < end; offset++) {
      const row = render(rows[offset]);
      first ??= row.matches("button") ? row : row.querySelector("button");
      container.append(row);
    }
    if (offset < rows.length) container.append(more);
    if (restoreFocus && first) first.focus();
  }
  more.onclick = appendPage;
  appendPage();
}

/** Render classifier observations and explicit question windows in the existing speaker dialog. */
export function expressionSummary(layer, seek, openFrame) {
  const disposers = [];
  const dispose = () => { for (const release of disposers) release(); };
  const section = node("section", "", "speaker-expression-summary");
  section.append(node("h4", uiText("Выражения лица")),
    node("p", uiText("Оценки видимого выражения лица. По ним нельзя определить чувства, намерения или правдивость."), "muted"));
  if (!layer?.faces?.length) {
    section.append(node("p", uiText("Выражения лица не измерены для этого участника."), "muted"));
    if (layer?.status === "failed" || layer?.status === "unavailable") {
      section.append(node("p", uiText("Классификатор выражений недоступен."), "muted"));
    }
    return {element: section, dispose};
  }
  section.append(node("p", uiText("Доли выражений рассчитаны по классифицированному времени каждого лица. Пропуски остаются неизвестными."), "muted"));
  if (layer.model) section.append(node("p", uiText`Модель: ${layer.model}`, "muted expression-model"));
  for (const face of layer.faces) {
    const block = node("section", "", "speaker-expression-face");
    block.append(node("h5", uiText`Лицо ${face.faceId}`), statistics(face));
    const moments = node("details", "", "speaker-profile-moments expression-moments");
    moments.append(node("summary", uiText`Измеренные интервалы: ${face.intervals.length}`));
    if (face.intervals.length) {
      const entries = face.intervals.map(interval => ({...interval, timeMs: interval.startMs}));
      const timeline = createMomentTimeline(entries, {
        label: uiText("Измеренные выражения лица"), className: "profile-expression-timeline",
        describe: entry => `${clock(entry.startMs)}–${clock(entry.endMs)} · ${expressionLabel(entry.label)} · ${percent.format(entry.confidence)}`,
        marker: entry => expressionMarker(entry.label),
        onActivate: entry => openFrame ? openFrame(entry, document.activeElement) : seek(entry.timeMs),
      });
      moments.append(timeline.element);
      disposers.push(timeline.dispose);
    }
    if (face.intervals.length) block.append(moments);
    const reactions = node("details", "", "expression-reactions");
    reactions.append(node("summary", uiText`Окна вопросов: ${face.reactions.length}`),
      node("p", uiText("Окно охватывает вопрос и 3 секунды после него. Связь по времени не доказывает реакцию на вопрос."), "muted"));
    pagedRows(reactions, face.reactions, reaction => {
      const item = node("div", "", "expression-reaction");
      const basis = {"question-mark": "Вопрос по пунктуации транскрипции", "explicit-annotation": "Явная метка вопроса",
        "interview-prompt-pattern": "Приглашение к ответу: оценка по формулировке"}[reaction.questionBasis];
      if (basis) item.append(node("p", uiText(basis), "muted"));
      item.append(node("p", reaction.text), moment(reaction.windowStartMs, reaction.windowEndMs, seek), statistics(reaction));
      return item;
    }, uiText("Показать ещё вопросы"));
    if (face.reactions.length) block.append(reactions);
    section.append(block);
  }
  return {element: section, dispose};
}
