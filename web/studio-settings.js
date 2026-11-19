import {mountStudioPipeline} from "./studio-pipeline.js";
import {mountStudioMods} from "./studio-mods.js";
import {uiLocale, uiText} from "./i18n.js";
import {attachStudioIcon} from "./studio-icons.js";

const pageSize = 25;
const dateFormat = new Intl.DateTimeFormat(uiLocale, {dateStyle: "medium", timeStyle: "short"});

/** List, open and remove saved transcripts with an explicit language and revision. */
export function mountStudioSettings(find, request, view) {
  const pipeline = mountStudioPipeline(find, request);
  const mods = mountStudioMods(find, request, () => pipeline.catalogChanged());
  const language = find("settings-material-language");
  const status = find("settings-material-status");
  const list = find("settings-material-list");
  const newer = find("materials-newer");
  const older = find("materials-older");
  const confirmation = find("material-delete");
  let offset = 0;
  let nextOffset = null;
  let sequence = 0;
  let deletion;
  let removing = false;

  /** Discard late list responses after the user changes language or page. */
  async function refresh(message = "") {
    const version = ++sequence;
    const selected = language.value;
    list.replaceChildren();
    newer.hidden = older.hidden = true;
    status.textContent = uiText("Загружаем материалы…");
    try {
      const {items, nextOffset: next} = await request("transcript-materials", undefined,
        {language: selected, limit: pageSize, offset});
      if (version !== sequence) return;
      if (!items.length && offset > 0) {
        offset = Math.max(0, offset - pageSize);
        return refresh(message);
      }
      nextOffset = next;
      newer.hidden = offset === 0;
      older.hidden = nextOffset === null;
      for (const item of items) list.append(materialRow(item, selected));
      status.textContent = message || (items.length ? "" : uiText("Сохранённых материалов пока нет."));
    } catch (error) {
      if (version === sequence) status.textContent = error.message;
    }
  }

  /** Keep both row actions tied to the version and language that the user saw. */
  function materialRow(item, selected) {
    const row = document.createElement("li");
    const detail = document.createElement("div");
    const title = document.createElement("strong");
    const updated = document.createElement("time");
    const date = new Date(item.updated * 1000);
    title.textContent = item.title;
    updated.dateTime = date.toISOString();
    updated.textContent = dateFormat.format(date);
    detail.append(title, updated);

    const link = document.createElement("a");
    const url = view.materialUrl(item.id, selected);
    link.className = "material-open studio-icon";
    link.href = url;
    link.textContent = uiText("Открыть");
    link.setAttribute("aria-label", uiText`Открыть материал «${item.title}»`);
    link.addEventListener("click", event => {
      if (event.button || event.metaKey || event.ctrlKey || event.altKey || event.shiftKey) return;
      event.preventDefault();
      view.openMaterial(url);
    });

    const remove = document.createElement("button");
    remove.type = "button";
    remove.className = "studio-icon material-remove";
    remove.dataset.icon = "trash";
    remove.dataset.hint = uiText("Удалить материал");
    remove.setAttribute("aria-label", uiText`Удалить материал «${item.title}»`);
    attachStudioIcon(remove);
    remove.addEventListener("click", () => {
      if (removing) return;
      deletion = {item, language: selected};
      find("material-delete-name").textContent = item.title;
      confirmation.returnValue = "";
      confirmation.showModal();
    });
    row.append(detail, link, remove);
    return row;
  }

  confirmation.addEventListener("close", async () => {
    const target = deletion;
    deletion = null;
    if (confirmation.returnValue !== "delete" || !target) return;
    removing = true;
    const {item, language: selected} = target;
    try {
      await request("transcript-material-delete", {id: item.id, revision: item.revision}, {language: selected});
      view.dismissMaterial(item.id, selected);
      await refresh(uiText("Материал удалён."));
    } catch (error) {
      await refresh(error.message);
    } finally {
      removing = false;
      if (!find("settings-page").hidden) find("materials-title").focus({preventScroll: true});
    }
  });
  language.addEventListener("change", () => { offset = 0; refresh(); });
  newer.addEventListener("click", () => { offset = Math.max(0, offset - pageSize); refresh(); });
  older.addEventListener("click", () => {
    if (nextOffset === null) return;
    offset = nextOffset;
    refresh();
  });

  return {
    open(defaultLanguage) {
      if (!language.dataset.initialized) {
        language.value = defaultLanguage;
        language.dataset.initialized = "true";
      }
      return Promise.all([refresh(), mods.open(), pipeline.open()]);
    },
  };
}
