import {apiText, uiText} from "./i18n.js";

const MODELS = {
  transcription: ["Транскрипция", "Whisper Base · распознавание речи и субтитры"],
  speakers: ["Разделение говорящих", "Sherpa · определение голосов и границ реплик"],
  sounds: ["Музыка и фоновые звуки", "Zipformer Small INT8 · разметка музыки и шумов"],
  vision: ["Лица, предметы и выражения", "YuNet, SFace, NanoDet и OpenCV"],
  qwen: ["Синтез Qwen3-TTS", "MLX · синтез речи по образцу голоса"],
  chatterbox: ["Синтез Chatterbox", "MLX · многоязычный синтез по образцу голоса"],
};

/** Build stable controls so progress updates preserve focus and reading order. */
export function mountModelInstall(find, request, onComplete) {
  const list = find("model-install-list");
  const notice = find("model-install-status");
  const rows = new Map();
  let pending = false;
  let active = null;
  let timer;
  let sequence = 0;

  for (const [id, [title, description]] of Object.entries(MODELS)) {
    const row = document.createElement("li");
    row.className = "studio-model-install-row";
    row.dataset.agWidth = "fill";
    const detail = document.createElement("div");
    const name = document.createElement("strong");
    name.textContent = uiText(title);
    const summary = document.createElement("p");
    summary.className = "muted";
    summary.textContent = uiText(description);
    const state = document.createElement("p");
    state.id = `model-install-${id}`;
    const progress = document.createElement("progress");
    progress.hidden = true;
    progress.setAttribute("aria-label", uiText(title));
    const button = document.createElement("button");
    button.type = "button";
    button.disabled = true;
    button.setAttribute("aria-describedby", state.id);
    button.textContent = uiText("Установить");
    button.addEventListener("click", () => start(id));
    detail.append(name, summary, state, progress);
    row.append(detail, button);
    list.append(row);
    rows.set(id, {button, name, progress, state});
  }

  function describe(item, busy) {
    if (item.ready) return uiText("Готово к работе");
    if (item.error) return apiText(item.error);
    if (!item.supported) return uiText("Нужен Mac с Apple Silicon и macOS 26+");
    if (!busy) return uiText("Можно установить");
    if (item.stage !== "weights") return uiText("Установка зависимостей");
    const status = uiText("Загрузка и проверка весов");
    return item.artifact ? `${status} · ${item.artifact.split("/").at(-1)}` : status;
  }

  /** Render exact byte progress when available and indeterminate runtime installation. */
  function render(report) {
    const completed = active && !report.active;
    active = report.active;
    for (const item of report.items) {
      const row = rows.get(item.id);
      if (!row) continue;
      const busy = active === item.id;
      row.button.disabled = pending || Boolean(active) || item.ready || !item.supported;
      row.button.textContent = uiText(item.ready ? "Установлено" : item.error ? "Повторить установку" : "Установить");
      row.button.setAttribute("aria-label", `${row.button.textContent} · ${row.name.textContent}`);
      row.state.textContent = describe(item, busy);
      row.progress.hidden = !busy;
      if (busy && item.total > 0 && item.received >= 0) {
        row.progress.max = item.total;
        row.progress.value = item.received;
      } else row.progress.removeAttribute("value");
    }
    const result = completed && report.items.find(item => item.id === completed);
    if (result && !result.error && result.ready) onComplete();
    notice.textContent = result ? apiText(result.error || "Установка завершена.") : "";
    clearTimeout(timer);
    if (active) timer = setTimeout(refresh, 1200);
  }

  /** Recover an active installation when the settings route is reopened. */
  async function refresh() {
    clearTimeout(timer);
    if (pending) return;
    const version = ++sequence;
    try {
      const report = await request("model-install");
      if (version === sequence) render(report);
    } catch (error) {
      if (version !== sequence) return;
      notice.textContent = error.message;
      if (active) timer = setTimeout(refresh, 2500);
    }
  }

  async function start(id) {
    if (active || pending) return;
    pending = true;
    sequence++;
    for (const row of rows.values()) row.button.disabled = true;
    notice.textContent = uiText("Начинаем установку…");
    try {
      const report = await request("model-install", {id});
      pending = false;
      render(report);
    } catch (error) {
      pending = false;
      await refresh();
      notice.textContent = error.message;
    }
  }

  return {open: refresh};
}
