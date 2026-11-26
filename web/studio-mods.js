import {uiText} from "./i18n.js";
import {modSettingsForm} from "./studio-mod-fields.js";

const STATUS = {ready: "Готов", enabled: "Включён", disabled: "Выключен", changed: "Изменён",
  invalid: "Ошибка манифеста", error: "Ошибка манифеста", blocked: "Нет зависимостей", unavailable: "Недоступен",
  "missing-requirements": "Нет зависимостей", "missing-dependency": "Не включена зависимость",
  "dependency-cycle": "Цикл зависимостей", "invalid-settings": "Ошибка параметров"};

const CAPABILITY_LABELS = {
  "speech.asr": "Распознавание речи", "speech.diarization": "Разделение говорящих",
  "speech.embedding": "Сравнение голосов", "audio.classification": "Музыка и фоновые звуки",
  "audio.denoise": "Шумоподавление", "audio.separation": "Разделение звуковых дорожек",
  "audio.background": "Фон для озвучки", "vision.faces": "Поиск лиц",
  "vision.face-embedding": "Сопоставление лиц", "vision.objects": "Поиск предметов",
  "vision.active-speaker": "Кто говорит в кадре", "vision.expressions": "Выражения лица",
  "vision.ocr": "Текст в кадре", "vision.gaze": "Взгляд и поворот головы",
  "vision.segmentation": "Маски персонажей", "vision.appearance": "Внешние признаки лица",
  "translation.text": "Перевод текста", "synthesis.tts": "Синтез речи", "prosody.context": "Ударения по контексту",
  "prosody.lexical": "Ударения в словах", "vision.visual-batch": "Полный анализ лиц",
  "vision.scene-batch": "Полный анализ сцены", "vision.person-batch": "Фигуры, взгляд и маски",
};

/** Match names, IDs and capability names as plain text; an empty query keeps every entry. */
export function filterMediaMods(items, query, origin = "all") {
  const needle = query.trim().toLocaleLowerCase();
  return items.filter(item => {
    if (origin !== "all" && (item.origin === "builtin" ? "builtin" : "user") !== origin) return false;
    const manifest = item.manifest ?? {};
    const text = [item.id, manifest.name, manifest.description,
      ...(manifest.capabilities ?? []).map(cap => cap.id)].join(" ").toLocaleLowerCase();
    return !needle || text.includes(needle);
  });
}

/** Group declared replacement providers, retaining inactive entries so unavailable choices stay explainable. */
export function mediaModProviders(items, selections = {}) {
  const groups = new Map();
  for (const item of items) {
    for (const capability of item.manifest?.capabilities ?? []) {
      if (capability.kind !== "provider") continue;
      if (!groups.has(capability.id)) groups.set(capability.id, []);
      groups.get(capability.id).push(item);
    }
  }
  for (const capability of Object.keys(selections)) {
    if (!groups.has(capability)) groups.set(capability, []);
  }
  return groups;
}

/** Mount mod management without running any model or user code while listing the catalog. */
export function mountStudioMods(find, request, onChange = () => {}) {
  const root = find("settings-mods");
  const list = find("mods-list");
  const providers = find("mods-providers");
  const status = find("mods-status");
  const query = find("mods-search");
  const origin = find("mods-origin");
  const directory = find("mods-directory");
  const template = find("mods-template-form");
  let catalog = null;
  let sequence = 0;
  let working = false;
  let disabled = new Map();

  /** Lock competing changes and restore the previous disabled state after failures. */
  function busy(value) {
    working = value;
    if (value) {
      disabled = new Map();
      for (const control of root.querySelectorAll("button,input,select,textarea")) {
        disabled.set(control, control.disabled);
        control.disabled = true;
      }
    } else {
      for (const [control, wasDisabled] of disabled) if (control.isConnected) control.disabled = wasDisabled;
      disabled.clear();
    }
    list.setAttribute("aria-busy", String(value));
    providers.setAttribute("aria-busy", String(value));
  }

  /** Apply one explicit mutation and render only a validated catalog response. */
  async function change(operation, body) {
    if (working) return;
    ++sequence;
    const focused = root.contains(document.activeElement) ? document.activeElement.dataset.modsFocus : null;
    busy(true);
    status.textContent = uiText("Сохраняем настройки модов…");
    try {
      const result = await request(`mods/${operation}`, body);
      render(Array.isArray(result.mods) ? result : await request("mods"));
      status.textContent = uiText("Сохранено. Изменения применятся к следующему запуску.");
      onChange();
    } catch (error) {
      status.textContent = error.message;
    } finally {
      busy(false);
      if (focused) {
        const target = [...root.querySelectorAll("[data-mods-focus]")].find(node => node.dataset.modsFocus === focused);
        target?.focus({preventScroll: true});
      }
    }
  }

  /** Read refreshed metadata, ignoring late responses from an older settings visit. */
  async function refresh() {
    if (working) return;
    const revision = ++sequence;
    status.textContent = uiText("Читаем каталог модов…");
    try {
      const result = await request("mods");
      if (revision !== sequence) return;
      render(result);
      status.textContent = result.truncated ? uiText("Каталог ограничен. Проверь количество папок модов.") : "";
    } catch (error) {
      if (revision === sequence) status.textContent = error.message;
    }
  }

  /** Preserve expanded cards and focused controls when server metadata is replaced. */
  function render(result) {
    if (!result || !Array.isArray(result.mods)) throw new Error(uiText("Некорректный каталог модов."));
    const focused = root.contains(document.activeElement) ? document.activeElement.dataset.modsFocus : null;
    const open = new Set([...list.querySelectorAll("details[open][data-mod-id]")].map(node => node.dataset.modId));
    catalog = result;
    directory.textContent = result.directory ?? "";
    renderList(open);
    renderProviders();
    if (focused) {
      const target = [...root.querySelectorAll("[data-mods-focus]")].find(node => node.dataset.modsFocus === focused);
      target?.focus({preventScroll: true});
    }
  }

  /** Rebuild only filtered cards; parameter forms are created lazily when opened. */
  function renderList(open = new Set()) {
    list.replaceChildren();
    const items = filterMediaMods(catalog?.mods ?? [], query.value, origin.value);
    for (const item of items) list.append(modCard(item, open.has(item.id)));
    find("mods-empty").hidden = items.length !== 0;
  }

  /** Build one disclosure card; mount settings only after its details are opened. */
  function modCard(item, expanded) {
    const row = document.createElement("li");
    const details = document.createElement("details");
    details.dataset.modId = item.id;
    details.dataset.agWidth = "fill";
    const heading = document.createElement("summary");
    const title = document.createElement("strong");
    title.textContent = item.manifest?.name || item.id;
    const state = document.createElement("small");
    state.textContent = `${item.manifest?.version ?? ""} · ${uiText(item.origin === "builtin" ? "Встроенный" : "Пользовательский")} · ${uiText(STATUS[item.status] ?? item.status ?? "Недоступен")}`;
    heading.append(title, document.createTextNode(" "), state);
    details.append(heading);
    let mounted = false;
    const populate = () => {
      if (mounted || !details.open) return;
      mounted = true;
      const body = document.createElement("div");
      body.className = "studio-mod-body";
      body.dataset.agWidth = "fill";
      const description = document.createElement("p");
      description.textContent = item.manifest?.description ?? "";
      const identity = document.createElement("code");
      identity.textContent = item.id;
      const capabilities = document.createElement("p");
      capabilities.className = "muted";
      const descriptions = [];
      for (const capability of item.manifest?.capabilities ?? []) {
        let description = capability.id;
        if (capability.contract) description += ` → ${capability.contract.id} v${capability.contract.version}`;
        descriptions.push(description);
      }
      capabilities.textContent = descriptions.join(" · ");
      const contracts = document.createElement("p");
      contracts.className = "muted";
      const adapters = [];
      for (const capability of item.manifest?.capabilities ?? []) {
        if (capability.inputContract) adapters.push(`${uiText("Вход")}: ${capability.inputContract.id} v${capability.inputContract.version}`);
        if (capability.prepare_input) adapters.push(`${uiText("Подготовка входа")}: ${capability.prepare_input}`);
        if (capability.convert_output) adapters.push(`${uiText("Преобразование результата")}: ${capability.convert_output}`);
      }
      contracts.textContent = adapters.join(" · ");
      const enabled = document.createElement("label");
      enabled.className = "inline-check";
      const check = document.createElement("input");
      check.type = "checkbox";
      check.setAttribute("aria-label", uiText`Включён · ${item.manifest?.name || item.id}`);
      check.checked = Boolean(item.enabled);
      check.disabled = !item.manifest || !item.fingerprint || Boolean(item.requirementsMissing?.length);
      check.dataset.modsFocus = `${item.id}:enable`;
      check.addEventListener("change", () => {
        const next = check.checked;
        check.checked = Boolean(item.enabled);
        change("enable", {id: item.id, fingerprint: item.fingerprint, enabled: next});
      });
      enabled.append(check, document.createTextNode(uiText("Включён")));
      body.append(identity, description, capabilities);
      if (adapters.length) body.append(contracts);
      body.append(enabled);
      const availability = item.manifest?.availability;
      if (availability) {
        const resources = document.createElement("p");
        resources.className = "muted";
        resources.textContent = uiText(availability.ready ? "Ресурсы найдены" : "Ресурсы или среда не найдены либо не проверены");
        body.append(resources);
      }
      if (item.origin !== "builtin") {
        const trust = document.createElement("small");
        trust.className = "muted";
        trust.textContent = uiText("Пользовательский Python-код выполняется локально с правами студии. Проверяй его перед включением.");
        body.append(trust);
      }
      if (item.status === "changed") {
        const accept = document.createElement("button");
        accept.type = "button";
        accept.textContent = uiText("Включить текущую версию");
        accept.dataset.modsFocus = `${item.id}:accept`;
        accept.addEventListener("click", () => change("enable", {id: item.id, fingerprint: item.fingerprint, enabled: true}));
        body.append(accept);
      }
      const problem = [item.error, ...(item.requirementsMissing ?? [])].filter(Boolean);
      if (problem.length) {
        const note = document.createElement("p");
        note.className = "muted";
        note.textContent = problem.join(" · ");
        body.append(note);
      }
      if (item.manifest) {
        body.append(modSettingsForm(item, settings => change("configure", {id: item.id, settings})));
        const metadata = document.createElement("details");
        const label = document.createElement("summary");
        label.textContent = uiText("Манифест и происхождение");
        const json = document.createElement("pre");
        json.textContent = JSON.stringify(item.manifest, null, 2);
        metadata.append(label, json);
        body.append(metadata);
      }
      details.append(body);
    };
    details.addEventListener("toggle", populate);
    details.open = expanded;
    populate();
    row.append(details);
    return row;
  }

  /** Publish one selector per replacement capability; the server remains the authority on compatibility. */
  function renderProviders() {
    providers.replaceChildren();
    for (const [capability, items] of mediaModProviders(catalog.mods, catalog.selections)) {
      const field = document.createElement("div");
      field.className = "select-field";
      field.dataset.agFormField = "";
      field.dataset.agWidth = "content";
      field.dataset.agWidthMinEm = "7";
      const caption = document.createElement("label");
      const fieldId = `mods-provider-${capability}`;
      caption.htmlFor = fieldId;
      caption.id = `${fieldId}-label`;
      caption.textContent = uiText(CAPABILITY_LABELS[capability] ?? capability);
      const select = document.createElement("select");
      select.id = fieldId;
      select.setAttribute("aria-labelledby", caption.id);
      select.dataset.modsFocus = `provider:${capability}`;
      select.title = capability;
      const defaultOption = document.createElement("option");
      defaultOption.value = "";
      defaultOption.textContent = uiText("По умолчанию");
      select.append(defaultOption);
      const current = catalog.selections?.[capability];
      if (current && !items.some(item => item.id === current)) {
        const missing = document.createElement("option");
        missing.value = current;
        missing.textContent = uiText`Недоступный мод: ${current}`;
        missing.disabled = true;
        select.append(missing);
      }
      for (const item of items) {
        const option = document.createElement("option");
        option.value = item.id;
        option.textContent = item.manifest.name;
        option.disabled = !item.enabled || Boolean(item.error) || Boolean(item.requirementsMissing?.length);
        select.append(option);
      }
      select.value = catalog.selections?.[capability] ?? "";
      select.addEventListener("change", () => {
        const next = select.value || null;
        select.value = catalog.selections?.[capability] ?? "";
        change("select", {capability, id: next});
      });
      field.append(caption, select);
      providers.append(field);
    }
  }

  find("mods-rescan").addEventListener("click", () => change("rescan", {}));
  query.addEventListener("input", () => renderList());
  origin.addEventListener("change", () => renderList());
  template.addEventListener("submit", event => {
    event.preventDefault();
    if (template.reportValidity()) change("template", {id: find("mods-template-id").value.trim(), template: find("mods-template-kind")?.value ?? "word-count"});
  });
  return {open: refresh};
}
