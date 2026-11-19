import {uiText} from "./i18n.js";
import {mountObjectAppearanceTimeline} from "./video-object-appearance-timeline.js";

const PAGE_SIZE = 200;

/** Format a source millisecond position without rounding into the next measurement. */
export function momentTimecode(ms) {
  const seconds = Math.floor(ms / 1000);
  return `${Math.floor(seconds / 60)}:${String(seconds % 60).padStart(2, "0")}.${String(ms % 1000).padStart(3, "0")}`;
}

/** Keep a bounded marker page on the full first-to-last measurement axis. */
export function momentPage(entries, offset = 0) {
  const ordered = entries.filter(entry => Number.isInteger(entry.timeMs) && entry.timeMs >= 0)
    .sort((a, b) => a.timeMs - b.timeMs);
  if (!ordered.length) return {items: [], total: 0, domain: [0, 0], nextOffset: null};
  const first = ordered[0].startMs ?? ordered[0].timeMs;
  let last = first;
  for (const entry of ordered) last = Math.max(last, entry.endMs ?? entry.timeMs);
  return {items: ordered.slice(offset, offset + PAGE_SIZE), total: ordered.length, domain: [first, last],
    nextOffset: offset + PAGE_SIZE < ordered.length ? offset + PAGE_SIZE : null};
}

/** Compose the existing appearance rail for measured profile moments, with arrow-only selection. */
export function createMomentTimeline(entries, {label, describe, marker, onActivate, className = ""}) {
  const nodes = new Map();
  function element(tag, text = "", classes = "") {
    const node = document.createElement(tag);
    node.textContent = text;
    node.className = classes;
    return node;
  }
  const elementRoot = element("div", "", `editor-object-views profile-moment-timeline ${className}`);
  const axis = element("div", "", "profile-moment-axis");
  const start = element("span");
  const end = element("span");
  axis.append(start, end);
  const track = element("div", "", "editor-object-visit-track");
  track.setAttribute("role", "group");
  track.setAttribute("aria-label", label);
  nodes.set("media-object-visit-track", track);
  const previous = element("button", "‹", "editor-object-visit-step");
  const next = element("button", "›", "editor-object-visit-step");
  for (const [button, name] of [[previous, "Предыдущая отметка"], [next, "Следующая отметка"]]) {
    button.type = "button";
    button.setAttribute("aria-label", uiText(name));
    button.title = uiText(name);
  }
  nodes.set("media-object-previous-visit", previous);
  nodes.set("media-object-next-visit", next);
  const rail = element("div", "", "editor-object-visit-rail");
  nodes.set("media-object-view-buttons", rail);
  const tooltip = element("div", "", "editor-object-visit-tooltip");
  tooltip.setAttribute("aria-hidden", "true");
  tooltip.hidden = true;
  nodes.set("media-object-visit-tooltip", tooltip);
  const status = element("span", "", "editor-object-visit-status");
  status.setAttribute("role", "status");
  nodes.set("media-object-visit-status", status);
  track.append(previous, rail, next, tooltip, status);
  const pageLabel = element("p", "", "muted");
  const controls = element("div", "", "profile-moment-controls");
  const open = element("button", uiText("Открыть выбранный момент"));
  open.type = "button";
  open.className = "profile-moment-open";
  controls.append(open);
  const previousPage = element("button", uiText("Предыдущие отметки"));
  const nextPage = element("button", uiText("Следующие отметки"));
  previousPage.type = nextPage.type = "button";
  controls.append(previousPage, nextPage);
  elementRoot.append(axis, track, pageLabel, controls);
  let offset = 0;
  let selected;
  const page = momentPage(entries);
  const timeline = mountObjectAppearanceTimeline(key => nodes.get(key), momentTimecode, entry => onActivate(entry), {
    domain: page.domain, describe, marker, navigateOnly: true,
    onNavigate: entry => { selected = entry; open.setAttribute("aria-label", `${uiText("Открыть выбранный момент")} · ${describe(entry)}`); },
  });

  /** Swap only the marker page; the min/max axis continues to describe the complete set. */
  function render(focusPage = false) {
    const page = momentPage(entries, offset);
    selected = page.items[0];
    start.textContent = momentTimecode(page.domain[0]);
    end.textContent = momentTimecode(page.domain[1]);
    timeline.render(page.items, selected?.sample ?? (selected && {timeMs: selected.timeMs}));
    open.disabled = !selected;
    if (selected) open.setAttribute("aria-label", `${uiText("Открыть выбранный момент")} · ${describe(selected)}`);
    previousPage.hidden = nextPage.hidden = page.total <= PAGE_SIZE;
    previousPage.disabled = offset === 0;
    nextPage.disabled = page.nextOffset === null;
    pageLabel.hidden = page.total <= PAGE_SIZE;
    pageLabel.textContent = uiText`Показано ${offset + 1}–${offset + page.items.length} из ${page.total}`;
    if (focusPage) rail.querySelector("button")?.focus({preventScroll: true});
  }
  previousPage.onclick = () => { offset = Math.max(0, offset - PAGE_SIZE); render(true); };
  nextPage.onclick = () => { offset += PAGE_SIZE; render(true); };
  open.onclick = () => { if (selected) onActivate(selected); };
  render();
  return {element: elementRoot, dispose: timeline.dispose};
}
