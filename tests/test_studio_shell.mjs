import assert from "node:assert/strict";
import test from "node:test";
import {mountStudioShell, readSidebarPreference} from "../web/studio-shell.js";
import {textFor} from "../web/i18n.js";

/** Emulate only the native navigation/storage boundary, including movement into the existing drawer. */
function environment({collapsed = false, blocked = false, popover = true} = {}) {
  const store = new Map([["voice-studio:design-profile", "retro"], ["voice-studio:color-mode", "dark"],
    ["voice-studio:sidebar-collapsed", collapsed ? "1" : "0"]]);
  const writes = [];
  class Node extends EventTarget {
    attrs = new Map();
    dataset = {};
    style = {};
    textContent = "";
    hidden = false;
    open = false;
    removed = false;
    setAttribute(key, value) { this.attrs.set(key, value); }
    getAttribute(key) { return this.attrs.get(key) ?? null; }
    matches() { return this.open; }
    showPopover() { this.open = true; }
    hidePopover() { this.open = false; }
    getBoundingClientRect() { return {top: 70, right: 64, width: 160, height: 44}; }
    remove() { this.removed = true; }
  }
  const toggle = new Node();
  const link = new Node();
  link.setAttribute("aria-label", "Настройки");
  const navigation = new Node();
  navigation.querySelectorAll = () => [link];
  const desktop = new Node();
  desktop.current = true;
  desktop.contains = () => desktop.current;
  const window = new Node();
  window.innerWidth = 900;
  window.innerHeight = 600;
  window.localStorage = {
    getItem(key) { if (blocked) throw Error("Storage unavailable"); return store.get(key); },
    setItem(key, value) { if (blocked) throw Error("Storage unavailable"); writes.push([key, value]); store.set(key, value); },
  };
  const drawer = new Node();
  const nodes = {"nav-collapse": toggle, "desktop-navigation": desktop, "studio-navigation": navigation, "mobile-navigation": drawer};
  const appended = [];
  const document = {documentElement: {dataset: {}}, defaultView: window, body: {append: node => appended.push(node)},
    getElementById: key => nodes[key], createElement() {
      const node = new Node();
      if (!popover) node.showPopover = node.hidePopover = undefined;
      return node;
    }};
  return {document, toggle, link, desktop, drawer, store, writes, appended};
}

test("navigation collapse is idempotent and persists only view state, preserving design/color preferences", () => {
  const env = environment({collapsed: true});
  const controller = mountStudioShell(env.document);
  assert.equal(mountStudioShell(env.document), controller);
  assert.equal(env.appended.length, 1);
  assert.equal(env.document.documentElement.dataset.sidebarCollapsed, "true");
  assert.equal(env.toggle.getAttribute("aria-expanded"), "false");
  env.toggle.dispatchEvent(new Event("click"));
  assert.equal(env.document.documentElement.dataset.sidebarCollapsed, "false");
  assert.equal(env.toggle.getAttribute("aria-expanded"), "true");
  assert.deepEqual(env.writes, [["voice-studio:sidebar-collapsed", "0"]]);
  assert.equal(env.store.get("voice-studio:design-profile"), "retro");
  assert.equal(env.store.get("voice-studio:color-mode"), "dark");
  controller.dispose();
  const writes = env.writes.length;
  env.toggle.dispatchEvent(new Event("click"));
  assert.equal(env.writes.length, writes, "disposed handlers cannot continue mutating layout state");
  assert.equal(env.appended[0].removed, true);
  assert.equal(env.toggle.hidden, true);
  const fresh = mountStudioShell(env.document);
  assert.notEqual(fresh, controller);
  assert.equal(env.toggle.hidden, false);
  fresh.dispose();
});

test("icon rail reveals destination labels on keyboard focus, dismisses hints and keeps drawer links fully native", () => {
  const env = environment({collapsed: true});
  const controller = mountStudioShell(env.document);
  const hint = env.appended[0];
  env.link.dispatchEvent(new Event("focus"));
  assert.equal(hint.open, true);
  assert.equal(hint.hidden, false);
  assert.equal(hint.textContent, "Настройки");
  assert.equal(hint.style.left, "72px");
  env.document.activeElement = env.link;
  env.link.dispatchEvent(new Event("pointerenter"));
  env.link.dispatchEvent(new Event("pointerleave"));
  assert.equal(hint.open, true, "pointer departure must retain the keyboard-focus hint");
  env.document.activeElement = undefined;
  env.link.dispatchEvent(new Event("blur"));
  assert.equal(hint.open, false);
  assert.equal(hint.hidden, true);
  env.desktop.current = false;
  env.link.dispatchEvent(new Event("focus"));
  assert.equal(hint.open, false, "the existing mobile drawer does not inherit icon-only hints");
  controller.dispose();
});

test("blocked storage and absent Popover API retain functional navigation and an inert hint fallback", () => {
  const env = environment({blocked: true, popover: false});
  assert.equal(readSidebarPreference(env.document.defaultView.localStorage), false);
  const controller = mountStudioShell(env.document);
  env.toggle.dispatchEvent(new Event("click"));
  assert.equal(env.document.documentElement.dataset.sidebarCollapsed, "true");
  env.link.dispatchEvent(new Event("focus"));
  assert.equal(env.appended[0].hidden, true);
  assert.equal(env.link.title, "Настройки");
  controller.dispose();
});

test("shell and workbench command labels use the shared English catalog", () => {
  assert.equal(textFor("en", "Свернуть панель навигации"), "Collapse navigation");
  assert.equal(textFor("en", "Развернуть панель навигации"), "Expand navigation");
  assert.equal(textFor("en", "Подготовка материала"), "Media preparation");
  assert.equal(textFor("en", "Изменить ширину инспектора"), "Resize inspector width");
  assert.equal(textFor("en", "Вписать шкалу"), "Fit timeline");
});
