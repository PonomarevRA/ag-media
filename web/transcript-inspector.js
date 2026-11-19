import {uiText} from "./i18n.js";

/** Keep the existing transcript disclosure and its preference stable across source changes. */
export function mountTranscriptPanel(panel, toggle) {
  const storageKey = "voice-studio:transcript-panel-collapsed";
  let collapsed = false;
  try { collapsed = localStorage.getItem(storageKey) === "true"; } catch {}

  /** Synchronize visual, accessible and persisted disclosure state without changing the report. */
  function update() {
    const label = uiText(collapsed ? "Показать расшифровку" : "Свернуть расшифровку");
    if (collapsed && panel.contains?.(panel.ownerDocument?.activeElement)) toggle.focus({preventScroll: true});
    panel.dataset.collapsed = String(collapsed);
    toggle.setAttribute("aria-expanded", String(!collapsed));
    toggle.setAttribute("aria-label", label);
    toggle.dataset.hint = label;
    toggle.title = label;
    panel.dispatchEvent(new Event("editor-inspector-change", {bubbles: true}));
  }

  toggle.onclick = () => {
    collapsed = !collapsed;
    try { localStorage.setItem(storageKey, String(collapsed)); } catch {}
    update();
  };
  update();
  return {get collapsed() { return collapsed; }};
}
