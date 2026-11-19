import {uiText} from "./i18n.js";

/** Place measured appearances on their own time axis; an optional millisecond domain keeps paged views aligned. */
export function layoutObjectAppearances(entries, domain) {
  const spans = [];
  for (const entry of entries) {
    const hasDuration = entry.endMs - entry.startMs > 1;
    spans.push({entry, hasDuration, startMs: hasDuration ? entry.startMs : entry.timeMs,
      endMs: hasDuration ? entry.endMs : entry.timeMs});
  }
  if (!spans.length) return [];
  const first = domain?.[0] ?? spans.reduce((value, span) => Math.min(value, span.startMs), Infinity);
  const last = domain?.[1] ?? spans.reduce((value, span) => Math.max(value, span.endMs), -Infinity);
  const percent = timeMs => last === first ? 50 : 100 * (timeMs - first) / (last - first);
  return spans.map(span => ({entry: span.entry, hasDuration: span.hasDuration,
    left: percent(span.startMs), right: percent(span.endMs), dot: percent(span.entry.timeMs)}));
}

/** Mount the shared measurement rail. Optional labels/markers adapt profiles; navigateOnly keeps arrows from activating a modal. */
export function mountObjectAppearanceTimeline(find, formatTime, onSelect, options = {}) {
  const root = find("media-object-visit-track");
  const rail = find("media-object-view-buttons");
  const tooltip = find("media-object-visit-tooltip");
  const status = find("media-object-visit-status");
  const previous = find("media-object-previous-visit");
  const next = find("media-object-next-visit");
  let points = [];
  let selected = -1;
  let hovered = -1;
  let focused = -1;
  let navigationFocused = false;
  let dismissed = false;

  /** Anchor the tooltip to the whole rail so edge and mobile labels stay visible. */
  function showTooltip() {
    const index = hovered >= 0 ? hovered : focused >= 0 ? focused : navigationFocused ? selected : -1;
    tooltip.hidden = dismissed || index < 0;
    tooltip.textContent = points[index]?.label ?? "";
  }

  /** Move to an existing appearance, keeping focus on the invoking control. */
  function choose(index, focusMarker = false, activate = true) {
    if (index < 0 || index >= points.length) return;
    const point = points[index];
    dismissed = false;
    if (focusMarker) point.button.focus({preventScroll: true});
    setSelection(point.entry.sample ?? {timeMs: point.entry.timeMs});
    options.onNavigate?.(point.entry);
    if (activate) onSelect(point.entry);
    status.textContent = point.button.getAttribute("aria-label");
  }

  previous.onclick = () => choose(selected > 0 ? selected - 1 : -1, false, !options.navigateOnly);
  next.onclick = () => choose(selected + 1, false, !options.navigateOnly);
  for (const button of [previous, next]) {
    button.addEventListener("focus", () => { navigationFocused = true; dismissed = false; showTooltip(); });
    button.addEventListener("blur", () => { navigationFocused = false; showTooltip(); });
  }
  root.addEventListener("keydown", event => {
    if (event.key !== "Escape" || tooltip.hidden) return;
    dismissed = true;
    showTooltip();
    event.stopPropagation();
  });

  /** Create one tab stop per selection and page through the dots with arrows or Home/End. */
  function keys(event, index) {
    const position = {ArrowLeft: index - 1, ArrowRight: index + 1, Home: 0, End: points.length - 1}[event.key];
    if (position === undefined) return;
    event.preventDefault();
    event.stopPropagation();
    choose(position, true, !options.navigateOnly);
  }

  /** Keep the marked occurrence tied to the selected source sample rather than playback. */
  function setSelection(sample) {
    const viewId = sample?.observationId ?? sample?.timeMs;
    selected = points.findIndex(point => point.button.dataset.viewId === String(viewId));
    if (selected < 0 && Number.isFinite(sample?.timeMs)) {
      selected = points.findIndex(({entry}) => entry.startMs <= sample.timeMs && sample.timeMs < entry.endMs);
    }
    points.forEach(({button, zone}, index) => {
      const active = index === selected;
      button.setAttribute("aria-pressed", String(active));
      button.tabIndex = index === Math.max(0, selected) ? 0 : -1;
      if (zone) zone.dataset.selected = String(active);
    });
    previous.setAttribute("aria-disabled", String(selected <= 0));
    next.setAttribute("aria-disabled", String(points.length === 0 || selected >= points.length - 1));
    showTooltip();
  }

  /** Keep touch stepping available when nearby dot hit areas overlap. */
  function updateSteps() {
    const axisWidth = root.clientWidth - 28;
    const percent = points.map(point => point.position).sort((a, b) => a - b);
    const dense = axisWidth > 0 && percent.some((value, index) => index &&
      (value - percent[index - 1]) * axisWidth / 100 < 28);
    previous.hidden = next.hidden = !dense;
  }

  const observer = new ResizeObserver(updateSteps);
  observer.observe(root);

  /** Rebuild from measured intervals; restore focus if progressive evidence kept that sample. */
  function render(entries, sample, viewId) {
    points = [];
    hovered = focused = -1;
    dismissed = false;
    status.textContent = "";
    rail.replaceChildren();
    for (const {entry, hasDuration, left, right, dot} of layoutObjectAppearances(entries, options.domain)) {
      const index = points.length;
      const timestamp = formatTime(entry.timeMs);
      const frame = uiText`Кадр предмета ${timestamp}`;
      const explanation = hasDuration ?
        `${formatTime(entry.startMs)}–${formatTime(entry.endMs)} · ${frame}` : frame;
      const label = uiText`Появление ${index + 1} из ${entries.length}`;
      const description = options.describe?.(entry, index, entries.length);
      const button = document.createElement("button");
      button.className = "editor-object-visit-dot";
      button.type = "button";
      button.dataset.viewId = String(entry.sample?.observationId ?? entry.timeMs);
      button.style.insetInlineStart = `${dot}%`;
      button.setAttribute("aria-label", description ?? `${label} · ${explanation}`);
      if (options.marker) {
        button.dataset.marker = "emoji";
        button.textContent = options.marker(entry);
      }
      button.onkeydown = event => keys(event, index);
      button.onclick = () => choose(index, true);
      button.addEventListener("focus", () => { focused = index; dismissed = false; showTooltip(); });
      button.addEventListener("blur", () => { focused = -1; showTooltip(); });
      let zone;
      if (hasDuration) {
        zone = document.createElement("span");
        zone.className = "editor-object-visit-zone";
        zone.setAttribute("aria-hidden", "true");
        zone.style.insetInlineStart = `${left}%`;
        zone.style.inlineSize = `${right - left}%`;
        zone.onclick = () => choose(index, true);
        rail.append(zone);
      }
      for (const target of [button, zone].filter(Boolean)) {
        target.addEventListener("pointerenter", () => { hovered = index; dismissed = false; showTooltip(); });
        target.addEventListener("pointerleave", () => { hovered = -1; showTooltip(); });
      }
      points.push({entry, button, zone, label: description ?? explanation, position: dot});
      rail.append(button);
    }
    root.hidden = !points.length;
    updateSteps();
    setSelection(sample);
    const restored = points.find(({button}) => button.dataset.viewId === viewId);
    restored?.button.focus({preventScroll: true});
  }

  return {render, setSelection, dispose: () => observer.disconnect()};
}
