import english from "./messages-en.js";

const STORAGE_KEY = "voice-studio:interface-language";

/** Resolve the explicit route override, then the saved preference. */
function initialLocale() {
  if (typeof window === "undefined") return "ru";
  const explicit = new URLSearchParams(window.location.search).get("ui");
  if (explicit === "en" || explicit === "ru") return explicit;
  try { return window.localStorage.getItem(STORAGE_KEY) === "en" ? "en" : "ru"; }
  catch { return "ru"; }
}

export const uiLocale = initialLocale();

/** Translate an interface message without treating data as a translation key. */
export function textFor(locale, key, ...values) {
  const message = locale === "en" && Object.hasOwn(english, key) ? english[key] : key;
  return message.replace(/\{(\d+)\}/g, (match, index) => Number(index) < values.length ? values[index] : match);
}

/** Format an ordinary message or a template literal in the active interface locale. */
export function uiText(message, ...values) {
  if (typeof message === "string") return textFor(uiLocale, message, ...values);
  const key = message.map((part, index) => (index ? "{" + (index - 1) + "}" : "") + part).join("");
  return textFor(uiLocale, key, ...values);
}

/** Translate studio diagnostics and preserve the original when it has no catalog entry. */
export function apiText(message) {
  if (!message || uiLocale !== "en") return message;
  const count = /^Размечено говорящих: (\d+) из (\d+)\. Проверь метки\.$/.exec(message);
  if (count) return `${count[1]} of ${count[2]} speakers labeled. Review the labels.`;
  const size = /^Размер файла должен быть не больше (\d+) МиБ\.$/.exec(message);
  if (size) return `The file cannot exceed ${size[1]} MiB.`;
  return textFor(uiLocale, message);
}

/** Translate the initial HTML before recordings, transcripts or filenames reach the DOM. */
function translateStaticLabels(document) {
  const roots = [document.body, ...[...document.querySelectorAll("template")].map(node => node.content)];
  const attrs = ["aria-label", "aria-description", "title", "placeholder", "alt", "label"];
  for (const root of roots) {
    const labels = document.createTreeWalker(root, NodeFilter.SHOW_TEXT);
    while (labels.nextNode()) {
      const node = labels.currentNode;
      if (node.parentElement?.closest("script,style,textarea,[data-endonym]")) continue;
      const key = node.data.trim().replace(/\s+/g, " ");
      const translated = textFor(uiLocale, key);
      if (translated !== key) node.data = node.data.replace(/\S[\s\S]*\S|\S/u, () => translated);
    }
    for (const node of root.querySelectorAll(attrs.map(attr => "[" + attr + "]").join(","))) {
      for (const attr of attrs) {
        if (node.hasAttribute(attr)) node.setAttribute(attr, textFor(uiLocale, node.getAttribute(attr)));
      }
    }
  }
}

/** Install the independent presentation setting and preserve the current route on changes. */
export function installInterfaceLocale(document, hasUnsavedWork = () => false) {
  document.documentElement.lang = uiLocale;
  if (uiLocale === "en") translateStaticLabels(document);
  const select = document.getElementById("interface-language");
  select.value = uiLocale;
  try { window.localStorage.setItem(STORAGE_KEY, uiLocale); } catch {}
  select.addEventListener("change", () => {
    const locale = select.value === "en" ? "en" : "ru";
    if (hasUnsavedWork() && !window.confirm(uiText("Смена языка перезагрузит страницу. Несохранённая работа будет потеряна. Продолжить?"))) {
      select.value = uiLocale;
      return;
    }
    const url = new URL(window.location.href);
    url.searchParams.set("ui", locale);
    window.location.replace(url.href);
  });
}
