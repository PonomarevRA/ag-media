import {uiLocale, uiText} from "./i18n.js";

const number = new Intl.NumberFormat(uiLocale, {maximumFractionDigits: 2});
const durationNumber = new Intl.NumberFormat(uiLocale, {maximumSignificantDigits: 5});
const clock = ms => `${Math.floor(ms / 60000)}:${String(Math.floor(ms / 1000) % 60).padStart(2, "0")}`;

/** Preserve millisecond intervals, rounding the whole timestamp before splitting its fields. */
export function statisticsClock(ms) {
  const rounded = Math.max(0, Math.round(ms));
  return clock(rounded) + "." + String(rounded % 1000).padStart(3, "0");
}

/** Keep short positive measurements distinguishable from zero and missing measurements. */
export function formatStatisticsValue(value, seconds = false) {
  if (!Number.isFinite(value)) return uiText("Нет измерений");
  if (!seconds) return number.format(value);
  if (value > 0 && value < .01) return durationNumber.format(value * 1000) + " " + uiText("мс");
  return durationNumber.format(value) + " " + uiText("с");
}

/** Map a plot-local X to one bounded, equal-duration source bin; invalid/empty plots have no selection. */
export function statisticsBinAt(x, left, right, count) {
  if (![x, left, right].every(Number.isFinite) || right <= left || !Number.isInteger(count) || count < 1) return null;
  return Math.min(count - 1, Math.max(0, Math.floor((x - left) / (right - left) * count)));
}

/** Use integer count ticks and a compact1/2/5 duration ceiling, without forcing sub-second data to1s. */
export function statisticsYTop(maximum, seconds = false) {
  if (!Number.isFinite(maximum) || maximum <= 0) return 1;
  if (!seconds) return Math.max(1, Math.ceil(maximum));
  const magnitude = 10 ** Math.floor(Math.log10(maximum));
  const scaled = maximum / magnitude;
  return (scaled <= 1 ? 1 : scaled <= 2 ? 2 : scaled <= 5 ? 5 : 10) * magnitude;
}

function el(tag, text, className) {
  const node = document.createElement(tag);
  if (text !== undefined) node.textContent = text;
  if (className) node.className = className;
  return node;
}

/** Own responsive SVG geometry, exact bin inspection and lifecycle for one fixed report series group. */
export function createStatisticsChart({title, subtitle, series, model, label, onSelect, seconds = false}) {
  const section = el("section", undefined, "video-stat-chart");
  section.dataset.agWidth = "fill";
  section.append(el("h3", title), el("p", subtitle, "muted"),
    el("p", uiText("Серые участки — нет измерений. Наведи курсор, коснись графика или используй стрелки."), "muted"));
  const host = el("div", undefined, "video-stat-plot");
  host.tabIndex = 0;
  host.setAttribute("role", "group");
  host.setAttribute("aria-label", title);
  host.setAttribute("aria-keyshortcuts", "ArrowLeft ArrowRight Home End Escape");
  const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
  svg.setAttribute("role", "img");
  svg.setAttribute("aria-label", title + " · " + uiText("Точные значения в таблице ниже"));
  const tooltip = el("div", undefined, "video-stat-tooltip");
  tooltip.hidden = true;
  tooltip.setAttribute("role", "status");
  tooltip.setAttribute("aria-live", "off");
  host.append(svg, tooltip);
  section.append(host);
  const legend = el("ul", undefined, "video-stat-legend");
  for (const [i, row] of series.entries()) legend.append(el("li", label(row.id), "stat-tone-" + i));
  section.append(legend);
  let selected = 0, cursor, geometry, drawFrame, pointerFrame, pendingPointer, disposed = false, pinned = false;
  let lastPoint = {x: 0, y: 0}, keyboard = false, tooltipIndex = -1;
  const maximum = Math.max(0, ...series.flatMap(row => row.values.filter(Number.isFinite)));
  const high = statisticsYTop(maximum, seconds);

  /** Keep all values, including true zero and missing cells, available without hover-only decoding. */
  function updateTooltip(index, point, fromKeyboard) {
    if (!model.bins[index]) return;
    lastPoint = point;
    keyboard = fromKeyboard;
    tooltip.setAttribute("aria-live", fromKeyboard ? "polite" : "off");
    if (tooltipIndex !== index) {
      tooltipIndex = index;
      const bin = model.bins[index];
      const heading = el("strong", `${statisticsClock(bin.startMs)}–${statisticsClock(bin.endMs)}`);
      const rows = el("dl");
      for (const [i, row] of series.entries()) {
        const value = row.values[index];
        rows.append(el("dt", label(row.id), "stat-tone-" + i),
          el("dd", formatStatisticsValue(value, seconds)));
      }
      tooltip.replaceChildren(heading, rows);
      if (!series.length) tooltip.append(el("p", uiText("Нет измерений этого слоя")));
    }
    tooltip.hidden = false;
    const left = Math.max(6, Math.min(point.x + 12, host.clientWidth - tooltip.offsetWidth - 6));
    const top = Math.max(6, Math.min(point.y + 12, svg.clientHeight - tooltip.offsetHeight - 6));
    tooltip.style.left = left + "px";
    tooltip.style.top = top + "px";
  }

  function select(index) {
    selected = Math.max(0, Math.min(model.bins.length - 1, index));
    const bin = model.bins[selected];
    if (!geometry || !cursor || !bin) return;
    const x = geometry.x((bin.startMs + bin.endMs) / 2);
    cursor.setAttribute("d", `M${x} ${geometry.top}V${geometry.bottom}`);
    if (!tooltip.hidden) updateTooltip(selected, keyboard ? {x, y: geometry.top} : lastPoint, keyboard);
  }

  /** Rebuild only on layout/font-size changes; hover updates just the crosshair and a bounded value panel. */
  function draw() {
    drawFrame = 0;
    if (disposed || !host.clientWidth || !svg.clientHeight) return;
    // On narrow layouts the value panel sits below the SVG, outside its plotting coordinates.
    const width = host.clientWidth, height = svg.clientHeight;
    const font = Number.parseFloat(getComputedStyle(svg).fontSize) || 12;
    const ticks = !seconds && high === 1 ? [0, 1] : [0, seconds ? high / 2 : Math.floor(high / 2), high];
    const left = Math.max(36, Math.min(width * .4, Math.max(...ticks.map(value => formatStatisticsValue(value, seconds).length)) * font * .65 + 12));
    const right = width - 10, top = font + 10, bottom = height - font * 2.3;
    const x = ms => left + (right - left) * ms / Math.max(1, model.durationMs);
    const y = value => bottom - (bottom - top) * value / high;
    geometry = {left, right, top, bottom, x};
    svg.setAttribute("viewBox", `0 0 ${width} ${height}`);
    svg.replaceChildren();
    const node = (tag, attrs, text) => {
      const child = document.createElementNS("http://www.w3.org/2000/svg", tag);
      for (const [key, value] of Object.entries(attrs)) child.setAttribute(key, value);
      if (text !== undefined) child.textContent = text;
      svg.append(child); return child;
    };
    for (const [i, bin] of model.bins.entries()) {
      if (!series.some(row => row.values[i] !== null)) node("rect", {x: x(bin.startMs), y: top, width: x(bin.endMs) - x(bin.startMs), height: bottom - top, class: "video-stat-missing"});
    }
    for (const value of ticks) {
      node("path", {d: `M${left} ${y(value)}H${right}`, class: "video-stat-grid"});
      node("text", {x: left - 7, y: y(value) + font * .35, "text-anchor": "end"}, formatStatisticsValue(value, seconds));
    }
    const divisions = width / font < 22 ? 1 : width < 500 ? 2 : 4;
    for (let i = 0; i <= divisions; i++) node("text", {x: x(model.durationMs * i / divisions), y: height - font * .4,
      "text-anchor": i === 0 ? "start" : i === divisions ? "end" : "middle"}, clock(model.durationMs * i / divisions));
    for (const [s, row] of series.entries()) {
      let path = "", connected = false;
      for (const [i, value] of row.values.entries()) {
        if (value === null) { connected = false; continue; }
        const bin = model.bins[i], a = x(bin.startMs), b = x(bin.endMs);
        const tone = "stat-tone-" + s;
        if (seconds) {
          const bar = (b - a) * .82 / Math.max(1, series.length);
          node("rect", {x: a + (b - a) * .09 + s * bar, y: y(value), width: Math.max(.4, bar - .5), height: Math.max(value === 0 ? 1 : 0, bottom - y(value)), class: "video-stat-bar " + tone});
        } else {
          path += `${connected ? "L" : "M"}${a} ${y(value)}H${b} `;
          connected = true;
        }
      }
      if (!seconds) node("path", {d: path, class: "video-stat-series stat-tone-" + s, "stroke-dasharray": s % 3 === 1 ? "6 3" : s % 3 === 2 ? "2 3" : "none"});
    }
    cursor = node("path", {class: "video-stat-cursor", "data-stat-cursor": ""});
    // Transparent source-bin targets keep pointer/touch lookup independent of mark size.
    for (const [i, bin] of model.bins.entries()) node("rect", {x: x(bin.startMs), y: top, width: x(bin.endMs) - x(bin.startMs), height: bottom - top, fill: "transparent", "data-stat-bin": i});
    select(selected);
  }

  function requestDraw() { if (!disposed && !drawFrame) drawFrame = requestAnimationFrame(draw); }
  function inspect(index, point, fromKeyboard = false) {
    if (disposed || index === null) return;
    onSelect(index);
    updateTooltip(index, point, fromKeyboard);
  }
  const pointOf = event => { const rect = host.getBoundingClientRect(); return {x: event.clientX - rect.left, y: event.clientY - rect.top}; };
  host.addEventListener("pointermove", event => {
    if (event.pointerType === "touch" || !geometry) return;
    pendingPointer = pointOf(event);
    if (pointerFrame) return;
    pointerFrame = requestAnimationFrame(() => {
      pointerFrame = 0;
      const point = pendingPointer;
      if (disposed || !point || point.y < geometry.top || point.y > geometry.bottom || point.x < geometry.left || point.x > geometry.right) { tooltip.hidden = true; return; }
      inspect(statisticsBinAt(point.x, geometry.left, geometry.right, model.bins.length), point);
    });
  });
  host.addEventListener("pointerdown", event => {
    if (!geometry) return;
    pinned = event.pointerType === "touch";
    const point = pointOf(event);
    inspect(statisticsBinAt(point.x, geometry.left, geometry.right, model.bins.length), point);
  });
  function clearPointer() { cancelAnimationFrame(pointerFrame); pointerFrame = 0; pendingPointer = null; }
  host.addEventListener("pointerleave", () => { clearPointer(); if (!pinned) tooltip.hidden = true; });
  host.addEventListener("blur", () => { clearPointer(); tooltip.hidden = true; pinned = false; });
  host.addEventListener("keydown", event => {
    if (event.key === "Escape" && !tooltip.hidden) { tooltip.hidden = true; event.preventDefault(); event.stopPropagation(); return; }
    let index = selected;
    if (event.key === "ArrowLeft") index--;
    else if (event.key === "ArrowRight") index++;
    else if (event.key === "Home") index = 0;
    else if (event.key === "End") index = model.bins.length - 1;
    else return;
    event.preventDefault();
    index = Math.max(0, Math.min(model.bins.length - 1, index));
    inspect(index, {x: geometry?.x(model.bins[index]?.startMs ?? 0) ?? 0, y: geometry?.top ?? 0}, true);
  });
  const observer = typeof ResizeObserver === "function" ? new ResizeObserver(requestDraw) : null;
  observer?.observe(host);
  if (!observer) window.addEventListener("resize", requestDraw);
  requestDraw();

  const details = el("details"); details.dataset.statSection = title;
  details.append(el("summary", uiText("Точные значения")));
  const tableHost = el("div", undefined, "video-stat-table-scroll"), table = el("table"), head = el("tr");
  head.append(el("th", uiText("Интервал")));
  for (const row of series) head.append(el("th", label(row.id)));
  const thead = el("thead"); thead.append(head); table.append(thead);
  const tbody = el("tbody");
  model.bins.forEach((bin, i) => {
    const row = el("tr"), button = el("button", `${statisticsClock(bin.startMs)}–${statisticsClock(bin.endMs)}`);
    button.type = "button"; button.dataset.statFocus = `${title}-${i}`; button.onclick = () => onSelect(i);
    const cell = el("td"); cell.append(button); row.append(cell);
    for (const item of series) row.append(el("td", formatStatisticsValue(item.values[i], seconds)));
    tbody.append(row);
  });
  table.append(tbody); tableHost.append(table); details.append(tableHost); section.append(details);
  if (!series.some(row => row.values.some(value => value !== null))) section.append(el("p", uiText("Нет измерений этого слоя"), "muted"));
  return {element: section, select,
    dispose() {
      disposed = true; observer?.disconnect(); window.removeEventListener("resize", requestDraw);
      cancelAnimationFrame(drawFrame); cancelAnimationFrame(pointerFrame); tooltip.hidden = true;
    },
  };
}
