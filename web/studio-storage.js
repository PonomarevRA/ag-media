import {uiLocale, uiText} from "./i18n.js";
import {attachStudioIcon} from "./studio-icons.js";

const LABELS = {
  models: "Модели", runtime: "Среды выполнения", "test-assets": "Тестовые материалы",
  exports: "Экспорты", media: "Записи и синтез", checkpoints: "Обученные веса",
  "media-graphs": "Сохранённые графы", "analysis-checkpoints": "Кэш анализа",
  "analysis-cache": "Кэш анализа", "temporary-media": "Временные загрузки",
  "swift-cache": "Кэш компилятора", export: "Экспорт",
};
const number = new Intl.NumberFormat(uiLocale, {maximumFractionDigits: 1});
const date = new Intl.DateTimeFormat(uiLocale, {dateStyle: "medium"});
const UNITS = ["Б", "КиБ", "МиБ", "ГиБ", "ТиБ"];

/** Show a stable unit for small cache files as well as large exports. */
function formatBytes(bytes) {
  const scale = Math.max(0, Math.min(UNITS.length - 1, Math.floor(Math.log2(bytes || 1) / 10)));
  return `${number.format(bytes / 1024 ** scale)} ${uiText(UNITS[scale])}`;
}

/** Build text nodes without treating a local filename as HTML. */
function textNode(tag, text, className = "") {
  const node = document.createElement(tag);
  node.className = className;
  node.textContent = text;
  return node;
}

/** Load workspace usage and confirm individual removals or a batch of regenerable caches. */
export function mountStudioStorage(find, request) {
  const usage = find("storage-groups");
  const list = find("storage-list");
  const status = find("storage-status");
  const refresh = find("storage-refresh");
  const clean = find("storage-clean");
  const candidates = find("storage-candidates");
  const dialog = find("storage-dialog");
  const confirm = find("storage-confirm");
  const cancel = find("storage-cancel");
  let data;
  let selected;
  let loading = false;
  let deleting = false;

  /** Bind confirmation to the exact inventory revision behind the selected action. */
  function prepare(action, title, detail, note, groups = []) {
    selected = action;
    find("storage-confirm-title").textContent = uiText(title);
    find("storage-confirm-detail").textContent = detail;
    find("storage-confirm-note").textContent = uiText(note);
    const groupList = find("storage-confirm-groups");
    groupList.replaceChildren(...groups.map(group => textNode("li",
      `${uiText(LABELS[group.kind])} · ${formatBytes(group.bytes)}`)));
    groupList.hidden = !groups.length;
    confirm.textContent = uiText(action.route === "storage-clean" ? "Очистить кэш" : "Удалить");
    dialog.showModal();
  }

  function render() {
    const shown = candidates.checked ? data.items.filter(row => row.candidate) : data.items;
    list.replaceChildren();
    for (const row of shown) {
      const item = document.createElement("li");
      item.className = "storage-item";
      item.dataset.agWidth = "fill";
      item.dataset.candidate = String(row.candidate);
      const name = textNode("strong", row.label);
      const stamp = date.format(new Date(row.lastUsed * 1000));
      const activity = row.activitySource === "studio" ? uiText`Активность в студии: ${stamp}`
        : uiText`Дата чтения или изменения файла: ${stamp}`;
      const description = textNode("span",
        [uiText(LABELS[row.kind]), formatBytes(row.bytes), activity].join(" · "), "muted");
      const detail = document.createElement("div");
      detail.dataset.agWidth = "content";
      detail.dataset.agWidthMinEm = "6";
      detail.append(name, description);
      if (row.referenceUnknown) detail.append(textNode("span", uiText("Связи с графами не удалось проверить"), "muted"));
      if (row.referenced) detail.append(textNode("span", uiText("Связан с сохранённым графом"), "muted"));
      if (row.candidate) {
        const hint = row.automatic ? uiText("Подлежит автоочистке") : uiText("Кандидат на удаление");
        detail.append(textNode("span", hint, "storage-item-hint"));
        const button = textNode("button", "", "studio-icon studio-quiet-action");
        button.type = "button";
        button.dataset.icon = "trash";
        button.dataset.hint = uiText("Удалить");
        button.setAttribute("aria-label", uiText`Удалить ${row.label}`);
        attachStudioIcon(button);
        button.onclick = () => {
          const {id, revision} = row;
          prepare({route: "storage-delete", body: {id, revision}}, "Удалить файл или папку?",
            `${row.label} · ${formatBytes(row.bytes)}`, "Удаление освободит место. Восстановить данные из студии будет нельзя.");
        };
        item.append(detail, button);
      } else {
        item.append(detail);
      }
      list.append(item);
    }
    find("storage-empty").hidden = shown.length > 0;
  }

  async function load(message = "") {
    if (loading) return;
    loading = refresh.disabled = clean.disabled = true;
    status.textContent = uiText("Считаем объём данных…");
    try {
      data = await request("storage");
      usage.replaceChildren(...data.groups.map(group => {
        const item = document.createElement("div");
        item.append(textNode("dt", uiText(LABELS[group.kind])), textNode("dd", formatBytes(group.bytes)));
        return item;
      }));
      find("storage-free").textContent = uiText`Свободно на диске: ${formatBytes(data.freeBytes)}`;
      const {cacheDays, temporaryDays, exportDays} = data.policy;
      find("storage-policy").textContent = uiText`Автоочистка: временные загрузки через ${temporaryDays} дн., кэши через ${cacheDays} дн. без использования. Экспорты старше ${exportDays} дн. доступны для ручной проверки.`;
      find("storage-limit").hidden = data.totalItems <= data.items.length;
      const bytes = data.cleanup?.bytes ?? 0;
      find("storage-cleanup-summary").textContent = bytes ? uiText`Пересоздаваемый кэш: ${formatBytes(bytes)}`
        : uiText("Пересоздаваемый кэш пуст.");
      render();
      clean.disabled = bytes === 0;
      status.textContent = message;
    } catch (error) {
      status.textContent = error.message;
    } finally {
      loading = refresh.disabled = false;
    }
  }

  refresh.onclick = () => load();
  clean.onclick = () => {
    const {bytes, groups, revision} = data.cleanup;
    prepare({route: "storage-clean", body: {revision}}, "Очистить кэш проекта?",
      uiText`Объём кэша: ${formatBytes(bytes)}`,
      "Записи, материалы, графы, модели и экспорты сохранятся. Студия пропустит используемый кэш и пересоздаст удалённые данные по мере необходимости.",
      groups);
  };
  candidates.onchange = () => { if (data) render(); };
  dialog.addEventListener("cancel", event => { if (deleting) event.preventDefault(); });
  confirm.onclick = async () => {
    if (!selected || deleting) return;
    deleting = confirm.disabled = cancel.disabled = true;
    const {route, body} = selected;
    try {
      const result = await request(route, body);
      dialog.close();
      const message = result.skipped
        ? uiText`Удалено: ${formatBytes(result.bytes)}. Пропущено элементов: ${number.format(result.skipped)}.`
        : uiText`Удалено: ${formatBytes(result.bytes)}`;
      await load(message);
    } catch (error) {
      dialog.close();
      await load(error.message);
    } finally {
      deleting = confirm.disabled = cancel.disabled = false;
      (route === "storage-clean" && !clean.disabled ? clean : refresh).focus();
    }
  };

  return {open: load};
}
