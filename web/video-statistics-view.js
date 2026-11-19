import {createStatisticsChart, formatStatisticsValue, statisticsClock} from "./video-statistics-chart.js";
import {buildVideoStatistics} from "./video-statistics.js";
import {uiLocale, uiText} from "./i18n.js";
import {attachStudioIcon} from "./studio-icons.js";
import {expressionLabel, expressionMarker} from "./facial-expressions.js";
import {gazeIcon, gazeLabel} from "./speaker-gaze.js";

const number = new Intl.NumberFormat(uiLocale, {maximumFractionDigits: 2});
const percent = new Intl.NumberFormat(uiLocale, {style: "percent", maximumFractionDigits: 1});
const time = statisticsClock;
const entityLabels = {faces: "Лица", people: "Фигуры людей", objects: "Предметы"};

/** Create data-bearing DOM as literal text, never transcript-provided HTML. */
function el(tag, text, className) {
  const node = document.createElement(tag);
  if (text !== undefined) node.textContent = text;
  if (className) node.className = className;
  return node;
}

/** Show an on-demand report dashboard; hidden dialogs perform no aggregation or chart redraw. */
export function mountVideoStatistics(find, video) {
  const trigger = find("media-statistics");
  const dialog = find("media-statistics-dialog");
  const body = find("media-statistics-body");
  const close = find("media-statistics-close");
  let report, model;
  let chartViews = [];
  function releaseCharts() { for (const chart of chartViews) chart.dispose(); chartViews = []; }
  function timelineChart(title, subtitle, series, model, label, onSelect, seconds = false) {
    const chart = createStatisticsChart({title, subtitle, series, model, label, onSelect, seconds});
    chartViews.push(chart);
    return chart.element;
  }
  // A queued close from a previous opening must not dispose newly opened charts.
  dialog.addEventListener("close", () => { if (!dialog.open) releaseCharts(); });
  let excludeCommon = true;
  const range = find("media-statistics-time");
  const readout = find("media-statistics-position");
  const seek = find("media-statistics-seek");
  const setPosition = index => {
    range.value = String(Math.max(0, index));
    const bin = model?.bins[Number(range.value)];
    readout.textContent = bin ? `${time(bin.startMs)}–${time(bin.endMs)}` : "—";
    range.setAttribute("aria-valuetext", readout.textContent);
    if (bin) {
      for (const chart of chartViews) chart.select(Number(range.value));
    }
    seek.disabled = !bin || video.hidden || !Number.isFinite(video.duration);
  };
  range.oninput = () => setPosition(Number(range.value));
  seek.onclick = () => {
    const bin = model?.bins[Number(range.value)];
    if (!bin || seek.disabled) return;
    video.pause();
    video.currentTime = Math.min(bin.startMs / 1000, Math.max(0, video.duration - .001));
    dialog.close();
  };
  close.onclick = () => dialog.close();

  /** Recompute from the current canonical report so live patches and manual edits stay visible. */
  function render() {
    const focusKey = document.activeElement?.dataset.statFocus;
    const expanded = new Set([...body.querySelectorAll("details[open]")].map(node => node.dataset.statSection));
    model = buildVideoStatistics(report, {excludeCommon});
    releaseCharts();
    body.replaceChildren();
    const note = el("p", uiText(model.partial ? "Анализ продолжается · статистика дополняется" : "По данным текущего анализа"), "muted");
    body.append(note);
    const metrics = el("dl", undefined, "video-stat-metrics");
    for (const [key, caption] of [["speakers", "Говорящие"], ["faces", "ID лиц"], ["people", "Фигур одновременно, макс."], ["objects", "Профили предметов"], ["expressions", "Классы выражений"], ["expressionEpisodes", "Эпизоды выражений"]]) {
      const card = el("div");
      card.dataset.agWidth = "content";
      card.dataset.agWidthMinEm = "4";
      card.append(el("dt", uiText(caption)), el("dd", model.counts[key] === null ? "—" : number.format(model.counts[key])));
      metrics.append(card);
    }
    body.append(metrics, el("p", uiText("ID зависят от трекинга. Выражения — оценки видимых признаков лица, а не чувств. Пропуски означают отсутствие измерений."), "muted"));
    const words = el("section", undefined, "video-stat-words");
    words.append(el("h3", uiText("Топ-10 слов")), el("p", uiText`Всего слов: ${number.format(model.words.total)}`, "muted"));
    const filter = el("label");
    const input = el("input"); input.type = "checkbox"; input.checked = excludeCommon;
    input.dataset.statFocus = "word-filter";
    input.onchange = () => { excludeCommon = input.checked; render(); body.querySelector("input")?.focus(); };
    filter.append(input, uiText("Скрыть частые служебные слова RU/EN")); words.append(filter);
    const list = el("ol", undefined, "video-stat-word-list");
    for (const item of model.words.top) {
      const row = el("li"); row.append(el("bdi", item.word), el("strong", String(item.count))); list.append(row);
    }
    words.append(list);
    if (!model.words.top.length) words.append(el("p", uiText("Нет слов для подсчёта"), "muted"));
    body.append(words);
    const summary = (title, data, label) => {
      const section = el("section", undefined, "video-stat-summary");
      section.append(el("h3", title));
      if (!data.measuredMs) section.append(el("p", uiText("Нет измерений этого слоя"), "muted"));
      const facts = el("dl");
      for (const item of data.summary) facts.append(el("dt", label(item.label)), el("dd", `${formatStatisticsValue(item.ms / 1000, true)} · ${percent.format(item.share)}`));
      section.append(facts); body.append(section);
    };
    const expression = id => `${expressionMarker(id)} ${expressionLabel(id)}`;
    const gaze = id => `${gazeIcon(id)} ${gazeLabel(id)}`;
    summary(uiText("Взгляды · измеренное время"), model.gaze, gaze);
    body.append(el("p", uiText("Взгляд учитывается только для лиц, однозначно связанных со спикером."), "muted"));
    summary(uiText("Выражения лица · измеренное время"), model.expressions, expression);
    body.append(el("p", uiText("Время суммируется по людям: при нескольких лицах секунды наблюдений могут превышать длину ролика. Доли считаются только от измеренного времени."), "muted"));
    if (model.bins.length) {
      const charts = el("div", undefined, "video-stat-charts");
      charts.append(timelineChart(uiText("Люди и предметы во времени"), uiText("X — время ролика. Y — максимум в интервале; для предметов — профили, встреченные в интервале."), model.entities, model, id => uiText(entityLabels[id]), setPosition));
      charts.append(timelineChart(uiText("Взгляды во времени"), uiText("X — время ролика. Y — секунды наблюдений в интервале."), model.gaze.series, model, gaze, setPosition, true));
      charts.append(timelineChart(uiText("Выражения лица во времени"), uiText("X — время ролика. Y — секунды наблюдений в интервале."), model.expressions.series, model, expression, setPosition, true));
      body.insertBefore(charts, words);
    }
    range.max = String(Math.max(0, model.bins.length - 1));
    range.disabled = !model.bins.length;
    setPosition(Math.min(Number(range.value), Math.max(0, model.bins.length - 1)));
    for (const node of body.querySelectorAll("details")) node.open = expanded.has(node.dataset.statSection);
    if (focusKey) [...body.querySelectorAll("[data-stat-focus]")].find(node => node.dataset.statFocus === focusKey)?.focus({preventScroll: true});
  }
  trigger.onclick = () => { if (report) { render(); dialog.showModal(); } };
  for (const node of [trigger, close]) if (!node.querySelector("svg")) attachStudioIcon(node);
  return {setData(value) {
    report = value;
    trigger.disabled = !report;
    if (!report) { if (dialog.open) dialog.close(); releaseCharts(); body.replaceChildren(); model = null; }
    else if (dialog.open) render();
  }};
}
