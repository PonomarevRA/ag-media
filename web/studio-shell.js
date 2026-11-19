import {uiText} from "./i18n.js";

const NAVIGATION_KEY = "voice-studio:sidebar-collapsed";
const mounted = new WeakMap();

/** Resolve only this layout preference; unavailable storage preserves the expanded navigation. */
export function readSidebarPreference(storage) {
  try { return storage?.getItem(NAVIGATION_KEY) === "1"; }
  catch { return false; }
}

/** Keep the existing mounted navigation/data controls while adapting its desktop presentation. */
export function mountStudioShell(document) {
  if (mounted.has(document)) return mounted.get(document);
  const root = document.documentElement;
  const toggle = document.getElementById("nav-collapse");
  const desktop = document.getElementById("desktop-navigation");
  const navigation = document.getElementById("studio-navigation");
  if (!toggle || !desktop || !navigation) return {dispose() {}};
  let storage;
  try { storage = document.defaultView?.localStorage; } catch {}
  let collapsed = readSidebarPreference(storage);
  toggle.hidden = false;
  const lifecycle = new AbortController();
  const options = {signal: lifecycle.signal};
  root.dataset.studioShell = "true";
  const hint = document.createElement("div");
  hint.className = "studio-navigation-hint";
  hint.popover = "manual";
  hint.hidden = true;
  hint.setAttribute("aria-hidden", "true");
  document.body.append(hint);
  let hintOwner;

  /** Dismiss the top-layer hint before the sidebar moves or a menu opens. */
  function hideHint() {
    if (typeof hint.hidePopover === "function" && hint.matches(":popover-open")) hint.hidePopover();
    hint.hidden = true;
    hintOwner = undefined;
  }

  /** Keep label, pressed/disclosure state and the visible rail synchronized. */
  function render() {
    root.dataset.sidebarCollapsed = String(collapsed);
    toggle.setAttribute("aria-expanded", String(!collapsed));
    const label = uiText(collapsed ? "Развернуть панель навигации" : "Свернуть панель навигации");
    toggle.setAttribute("aria-label", label);
    toggle.dataset.hint = label;
    toggle.title = label;
    hideHint();
  }

  /** Reveal the full destination for pointer and keyboard users when only its icon is visible. */
  function showHint(link) {
    if (!collapsed || !desktop.contains(link) || typeof hint.showPopover !== "function") return;
    hideHint();
    hint.textContent = link.getAttribute("aria-label") || link.textContent.trim();
    hint.hidden = false;
    hint.showPopover();
    hintOwner = link;
    const box = link.getBoundingClientRect();
    const size = hint.getBoundingClientRect();
    const view = document.defaultView;
    hint.style.left = `${Math.max(8, Math.min(box.right + 8, view.innerWidth - size.width - 8))}px`;
    hint.style.top = `${Math.max(8, Math.min(box.top, view.innerHeight - size.height - 8))}px`;
  }

  toggle.addEventListener("click", () => {
    collapsed = !collapsed;
    try { storage?.setItem(NAVIGATION_KEY, collapsed ? "1" : "0"); } catch {}
    render();
  }, options);
  for (const link of navigation.querySelectorAll("nav a")) {
    let hovered = false;
    link.title = link.getAttribute("aria-label") || link.textContent.trim();
    link.addEventListener("focus", () => showHint(link), options);
    link.addEventListener("pointerenter", () => { hovered = true; showHint(link); }, options);
    link.addEventListener("pointerleave", () => {
      hovered = false;
      if (hintOwner === link && document.activeElement !== link) hideHint();
    }, options);
    link.addEventListener("blur", () => {
      if (hintOwner === link && !hovered) hideHint();
    }, options);
    link.addEventListener("click", hideHint, options);
    link.addEventListener("keydown", event => {
      if (event.key === "Escape") hideHint();
    }, options);
  }
  desktop.addEventListener("scroll", hideHint, options);
  document.defaultView?.addEventListener("resize", hideHint, options);
  document.getElementById("mobile-navigation")?.addEventListener("close", hideHint, options);
  render();
  const controller = {
    /** Remove listeners and supplementary presentation without changing any media state. */
    dispose() {
      lifecycle.abort();
      hideHint();
      hint.remove();
      toggle.hidden = true;
      toggle.setAttribute("aria-expanded", "true");
      delete root.dataset.studioShell;
      delete root.dataset.sidebarCollapsed;
      mounted.delete(document);
    },
  };
  mounted.set(document, controller);
  return controller;
}
