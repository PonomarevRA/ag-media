import {apiText, uiLocale, uiText} from "./i18n.js";
import {imageFileError, imageItems} from "./image-analysis-model.js";

const SVG = "http://www.w3.org/2000/svg";
const ROLES = {button_candidate: "Кнопка · кандидат", field_candidate: "Поле · кандидат",
  block_candidate: "Блок · кандидат"};

/** Own one browser-selected image, its local analysis and object URLs independently of speech results. */
export function mountImageAnalysis(find, request, {onOpen, setBusy, workbench}) {
  const panel = find("image-analysis"), image = find("image-preview"), overlay = find("image-overlay");
  const status = find("image-status"), list = find("image-detections"), summary = find("image-summary");
  const run = find("image-run"), cancel = find("image-cancel"), download = find("image-json");
  const markdown = find("image-markdown");
  const objects = find("image-objects"), ui = find("image-ui"), mode = find("image-detail");
  const layers = {objects: find("image-layer-objects"), ui: find("image-layer-ui"), text: find("image-layer-text")};
  let source = null, url = null, report = null, items = [], selected = null, controller = null, epoch = 0;

  /** Release only this image's pending UI request and in-memory source when another material is opened. */
  function clear() {
    epoch++;
    if (controller) { controller.abort(); controller = null; setBusy(false); }
    if (url) URL.revokeObjectURL(url);
    source = url = report = null;
    items = [];
    image.removeAttribute("src");
    overlay.replaceChildren();
    list.replaceChildren();
    panel.hidden = true;
    workbench.hidden = false;
  }

  /** Set competing-source and model-setting state while preserving an available cancel action. */
  function busy(value) {
    setBusy(value);
    run.disabled = objects.disabled = ui.disabled = mode.disabled = value;
    cancel.hidden = !value;
  }

  /** Render a bounded text-only list and matching outline layer; selection never trusts model text as HTML. */
  function render() {
    overlay.replaceChildren();
    list.replaceChildren();
    for (const item of items) {
      if (!layers[item.layer].checked) continue;
      const [x, y, width, height] = item.box;
      const rect = document.createElementNS(SVG, "rect");
      for (const [key, value] of Object.entries({x: x * 1000, y: y * 1000, width: width * 1000, height: height * 1000}))
        rect.setAttribute(key, value);
      rect.setAttribute("class", `image-box image-box-${item.layer}${selected === item.id ? " selected" : ""}`);
      rect.setAttribute("vector-effect", "non-scaling-stroke");
      overlay.append(rect);
      const button = document.createElement("button");
      button.type = "button";
      button.className = "image-detection";
      button.dataset.layer = item.layer;
      button.setAttribute("aria-pressed", String(selected === item.id));
      const label = document.createElement("span");
      label.textContent = ROLES[item.role] ? `${uiText(ROLES[item.role])}: ${item.label}` : item.label;
      button.append(label);
      const score = document.createElement("small");
      score.textContent = item.score === null ? uiText("Кандидат") : `${Math.round(item.score * 1000) / 10}%`;
      score.title = uiText("Уверенность модели");
      button.append(score);
      button.onclick = () => {
        selected = item.id;
        // Keep keyboard focus and scroll position instead of rebuilding the list on each selection.
        for (const row of list.children) row.setAttribute("aria-pressed", String(row === button));
        for (const box of overlay.children) box.classList.toggle("selected", box === rect);
      };
      list.append(button);
    }
    find("image-empty").hidden = list.children.length > 0 || !report;
  }

  /** Submit the same source with current independent layer options and ignore every stale completion. */
  async function analyze() {
    if (!source || controller) return;
    if (!objects.checked && !ui.checked) {
      status.textContent = uiText("Выбери предметы или распознавание интерфейса.");
      return;
    }
    const generation = ++epoch;
    const abort = controller = new AbortController();
    let timedOut = false;
    const timeout = setTimeout(() => { timedOut = true; abort.abort(); }, 130000);
    busy(true);
    panel.setAttribute("aria-busy", "true");
    status.textContent = uiText("Анализируем картинку локально…");
    try {
      const data = await request("analyze-image", source, {mode: mode.value, ui: ui.checked ? "1" : "0", markdown: uiLocale,
        objects: objects.checked ? "1" : "0", signal: abort.signal, onProgress(event) {
          if (generation !== epoch) return;
          if (event.stage === "image_validate") status.textContent = uiText("Проверяем изображение…");
          if (event.stage === "image_vision") status.textContent = uiText("Ищем предметы, текст и интерфейс…");
        }});
      if (generation !== epoch || abort.signal.aborted) return;
      report = {...data, source: {...data.source, name: source.name}};
      items = imageItems(report);
      selected = null;
      const count = layer => items.filter(item => item.layer === layer).length;
      summary.textContent = uiText`Предметы: ${count("objects")} · UI: ${count("ui")} · текст: ${count("text")}`;
      const dimensions = Array.isArray(data.imageSize) ? data.imageSize.join(" × ") : "";
      find("image-source").textContent = `${source.name} · ${dimensions}`;
      const warnings = [];
      if (["unavailable", "failed"].includes(data.objectStatus)) warnings.push(uiText("Модель предметов недоступна."));
      if (data.uiStatus === "unavailable") warnings.push(uiText("OCR недоступен; текст не проверен."));
      if (data.uiStatus === "partial") warnings.push(uiText("OCR завершён частично. Проверь текст или повтори анализ."));
      const seconds = ((data.requestElapsedMs ?? data.elapsedMs ?? 0) / 1000).toFixed(1);
      status.textContent = warnings.length ? warnings.join(" ") : uiText`Готово · ${seconds} с`;
      download.disabled = false;
      markdown.disabled = !report.markdown;
      render();
    } catch (error) {
      if (generation === epoch) status.textContent = abort.signal.aborted
        ? uiText(timedOut ? "Истёк лимит анализа картинки. Попробуй быстрый режим." : "Запрос отменён. Картинку можно проанализировать снова.")
        : apiText(error.message);
    } finally {
      clearTimeout(timeout);
      if (generation === epoch) {
        controller = null;
        panel.setAttribute("aria-busy", "false");
        busy(false);
      }
    }
  }

  /** Validate the selected format first so an accidental unsupported file does not discard other work. */
  function open(file) {
    const error = imageFileError(file);
    if (error) { find("media-status").textContent = uiText(error); return false; }
    onOpen();
    source = file;
    url = URL.createObjectURL(file);
    image.src = url;
    find("image-source").textContent = file.name;
    summary.textContent = status.textContent = "";
    download.disabled = markdown.disabled = true;
    panel.hidden = false;
    workbench.hidden = true;
    analyze();
    return true;
  }

  run.onclick = analyze;
  cancel.onclick = () => controller?.abort();
  find("image-close").onclick = clear;
  for (const control of Object.values(layers)) control.onchange = render;
  download.onclick = () => {
    if (!report) return;
    const {markdown: _document, ...measurements} = report;
    saveImageReport(JSON.stringify(measurements, null, 2), "application/json", "image-analysis.json");
  };
  markdown.onclick = () => {
    if (report?.markdown) saveImageReport(report.markdown, "text/markdown;charset=utf-8", "image-analysis.md");
  };
  return {open, clear, snapshot: () => report};
}

/** Save the requested representation of an already completed image analysis. */
function saveImageReport(content, type, filename) {
  const href = URL.createObjectURL(new Blob([content], {type}));
  const link = document.createElement("a");
  link.href = href;
  link.download = filename;
  try { link.click(); } finally { setTimeout(() => URL.revokeObjectURL(href), 30000); }
}
