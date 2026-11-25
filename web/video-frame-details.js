import {uiText} from "./i18n.js";
import {attachStudioIcon} from "./studio-icons.js";
import {FRAME_COLORS} from "./video-overlay.js";

const rows = [
  ["shot", "film", "План"],
  ["speech", "mic", "Речь"],
  ["face", "face", "Лица"],
  ["body", "person", "Фигуры"],
  ["body-count", "person", "Люди (оценка)"],
  ["interface", "ocr", "Интерфейс"],
  ["change", "frame-changes", "Зоны изменений"],
  ["graphic", "frame-changes", "Возможные графические вставки"],
  ["object", "cube", "Предметы"],
  ["text", "ocr", "Текст"],
];

/** Summarize only the marks the monitor drew at this timestamp. */
export function frameDetailValues(frame) {
  const bodies = [...frame.marks.values()].filter(mark => mark.box).length;
  return {
    shot: frame.shot === null ? "" : String(frame.shot),
    speech: frame.phrase ? [frame.speechLabel, frame.phrase.text].filter(Boolean).join(" · ") : "",
    face: frame.faces.length || "",
    body: bodies || "",
    "body-count": frame.personCount ?? "",
    interface: frame.panels?.map(panel => uiText(panel.panelType)).join(" · ") ?? "",
    change: frame.changes?.length || "",
    graphic: frame.graphics?.length || "",
    object: frame.objects?.map(item => uiText(item.label ?? "")).filter(Boolean).join(" · ") ?? "",
    text: frame.text?.map(item => item.text).filter(Boolean).join(" · ") ?? "",
  };
}

/** Keep DOM writes limited to clock ticks and changed values during playback. */
export function mountFrameDetails(panel, inspect) {
  const workbench = panel.closest(".editor-workbench");
  for (const [kind, color] of Object.entries(FRAME_COLORS)) {
    workbench.style.setProperty("--frame-" + kind, color);
  }
  const time = panel.querySelector("[data-frame-time]");
  const empty = panel.querySelector("[data-frame-empty]");
  const list = panel.querySelector("dl");
  const items = new Map();
  let currentFrame;
  for (const [kind, iconName, caption] of rows) {
    const row = document.createElement("div");
    row.dataset.frameKind = kind;
    const label = document.createElement("dt");
    const icon = document.createElement("span");
    icon.dataset.icon = iconName;
    attachStudioIcon(icon);
    label.append(icon, uiText(caption));
    const cell = document.createElement("dd");
    const value = document.createElement("button");
    value.type = "button";
    value.className = "frame-detail-button";
    value.setAttribute("aria-haspopup", "dialog");
    value.onclick = () => currentFrame && inspect(kind, uiText(caption), currentFrame, value);
    label.onclick = () => value.click();
    cell.append(value);
    row.append(label, cell);
    list.append(row);
    items.set(kind, {row, value, caption: uiText(caption)});
  }
  let lastSecond = -1;

  /** Show the current paint result, or hide the panel with the overlay. */
  return function update(frame) {
    currentFrame = frame;
    const hidePanel = frame === null;
    if (panel.hidden !== hidePanel) panel.hidden = hidePanel;
    if (!frame) return;

    const seconds = Math.floor(frame.timeMs / 1000);
    if (lastSecond !== seconds) {
      const minutes = String(Math.floor(seconds / 60)).padStart(2, "0");
      time.textContent = minutes + ":" + String(seconds % 60).padStart(2, "0");
      lastSecond = seconds;
    }

    const values = frameDetailValues(frame);
    let hasDetails = false;
    for (const [kind, {row, value, caption}] of items) {
      const copy = String(values[kind]);
      const hideRow = copy === "";
      hasDetails ||= !hideRow;
      if (row.hidden !== hideRow) row.hidden = hideRow;
      if (kind === "speech") {
        const confirmed = String(frame.speakingFaces.length > 0);
        if (row.dataset.confirmed !== confirmed) row.dataset.confirmed = confirmed;
      }
      if (value.textContent === copy) continue;
      value.textContent = copy;
      value.title = copy;
      value.setAttribute("aria-label", caption + " · " + copy);
    }
    if (empty.hidden !== hasDetails) empty.hidden = hasDetails;
  };
}
