import {uiText} from "./i18n.js";

/** Parse one native settings field without coercing empty values or non-finite numbers. */
export function modSettingValue(raw, schema) {
  if (schema.enum) return JSON.parse(raw);
  if (schema.type === "boolean") return Boolean(raw);
  if (schema.type === "number" || schema.type === "integer") {
    const value = raw.trim() === "" ? NaN : Number(raw);
    if (!Number.isFinite(value) || schema.type === "integer" && !Number.isInteger(value)) {
      throw new Error(uiText("Укажи корректное число."));
    }
    return value;
  }
  return schema.type === "string" ? raw : JSON.parse(raw);
}

/** Keep absent optional fields absent, and allow clearing optional structured/numeric values. */
export function readModSetting(raw, schema, {required, present, changed}) {
  if (!required) {
    if (!present && !changed) return {include: false};
    const blank = typeof raw === "string" && raw.trim() === "";
    const clearable = Boolean(schema.enum) || !["string", "boolean"].includes(schema.type);
    if (blank && clearable) return {include: false};
  }
  return {include: true, value: modSettingValue(raw, schema)};
}

/**
 * Render the declared typed settings using native controls; the server validates the same schema.
 * @param {object} item Catalog entry with effective settings and a manifest settingsSchema.
 * @param {function(object):void} save Submit a full settings object after native and JSON validation.
 * @returns {HTMLFormElement} A labelled, lazy-mounted settings form without executable manifest markup.
 */
export function modSettingsForm(item, save) {
  const form = document.createElement("form");
  form.className = "studio-mod-settings";
  form.dataset.agWidth = "fill";
  form.dataset.agFormRow = "stack";
  const fields = [];
  const schema = item.manifest.settingsSchema ?? {properties: {}};
  for (const [key, definition] of Object.entries(schema.properties ?? {})) {
    const field = document.createElement("div");
    field.dataset.agFormField = "";
    field.dataset.agWidth = "fill";
    const caption = document.createElement("label");
    caption.textContent = definition.title || key;
    const id = `mod-setting-${item.id}-${fields.length}`;
    caption.id = `${id}-label`;
    caption.htmlFor = id;
    // Explicit null is a supported setting; only a missing key uses the schema default.
    const hasSetting = Object.hasOwn(item.settings ?? {}, key);
    const value = hasSetting ? item.settings[key] : definition.default;
    const present = hasSetting || Object.hasOwn(definition, "default");
    const required = (schema.required ?? []).includes(key);
    let input;
    if (Array.isArray(definition.enum)) {
      input = document.createElement("select");
      if (!required) {
        const unset = document.createElement("option");
        unset.value = "";
        unset.textContent = uiText("Не задано");
        input.append(unset);
      }
      for (const choice of definition.enum) {
        const option = document.createElement("option");
        option.value = JSON.stringify(choice);
        option.textContent = String(choice);
        input.append(option);
      }
      input.value = value === undefined ? "" : JSON.stringify(value);
    } else {
      input = document.createElement(definition.type === "object" || definition.type === "array" ? "textarea" : "input");
      if (definition.type === "boolean") {
        input.type = "checkbox";
        input.checked = value === true;
        input.indeterminate = !present;
      } else if (["number", "integer"].includes(definition.type)) {
        input.type = "number";
        input.step = definition.type === "integer" ? "1" : "any";
        input.value = value === undefined ? "" : String(value);
        if (definition.minimum !== undefined) input.min = definition.minimum;
        if (definition.maximum !== undefined) input.max = definition.maximum;
      } else {
        input.value = definition.type === "string" ? value ?? "" : value === undefined ? "" : JSON.stringify(value, null, 2);
        if (definition.maxLength !== undefined) input.maxLength = definition.maxLength;
      }
    }
    input.id = id;
    input.setAttribute("aria-labelledby", caption.id);
    input.dataset.modsFocus = `${item.id}:setting:${key}`;
    input.required = definition.type !== "boolean" && required;
    const entry = {key, definition, input, required, present, changed: false};
    input.addEventListener("input", () => { entry.changed = true; input.setCustomValidity(""); });
    field.append(caption, input);
    if (definition.description) {
      const help = document.createElement("small");
      help.id = `${id}-help`;
      help.className = "muted";
      help.textContent = definition.description;
      input.setAttribute("aria-describedby", help.id);
      field.append(help);
    }
    fields.push(entry);
    form.append(field);
  }
  if (!fields.length) {
    const note = document.createElement("p");
    note.className = "muted";
    note.textContent = uiText("У этого мода нет пользовательских параметров.");
    form.append(note);
    return form;
  }
  const apply = document.createElement("button");
  apply.type = "submit";
  apply.textContent = uiText("Сохранить параметры");
  apply.dataset.modsFocus = `${item.id}:save`;
  form.append(apply);
  form.addEventListener("submit", event => {
    event.preventDefault();
    const entries = [];
    for (const entry of fields) {
      const {key, definition, input} = entry;
      try {
        const raw = definition.type === "boolean" && !definition.enum ? input.checked : input.value;
        const value = readModSetting(raw, definition, entry);
        if (value.include) entries.push([key, value.value]);
      } catch {
        input.setCustomValidity(uiText("Проверь значение: ожидается тип из схемы мода."));
        input.reportValidity();
        return;
      }
    }
    if (form.reportValidity()) save(Object.fromEntries(entries));
  });
  return form;
}
