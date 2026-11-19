import {uiText} from "./i18n.js";
import {bindResizablePanel} from "./editor-panel-resizer.js";
import {performVideoAction, videoActionEnabled, videoFrameStepMs} from "./video-shortcuts.js";
import {mountMonitorActions} from "./editor-monitor-actions.js";
import {attachStudioHint} from "./studio-icons.js";

const HEIGHT_KEY = "voice-studio:editor-timeline-height";
const WIDTH_KEY = "voice-studio:editor-inspector-width";
const MOUNTED = new WeakMap();

/** Clamp local view preferences; these dimensions never enter the media/report model. */
export function workspaceDimensions(value = {}, width = 1200, height = 720) {
  const bounded = (input, fallback, low, high) => Math.round(Math.max(low, Math.min(high,
    Number.isFinite(input) ? input : fallback)));
  return {
    inspectorWidth: bounded(value.inspectorWidth, 320, 240, Math.max(240, Math.min(540, width * .46))),
    timelineHeight: bounded(value.timelineHeight, 320, 140, Math.max(140, Math.min(560, height - 240))),
  };
}

/** Reserve visible track rows below the sticky ruler after measured toolbar and footer chrome. */
export function minimumTimelineHeight(chromeHeight, rulerHeight, trackBodyHeight = 128) {
  const finite = value => Number.isFinite(value) ? Math.max(0, value) : 0;
  return Math.ceil(finite(chromeHeight) + finite(rulerHeight) + Math.max(96, finite(trackBodyHeight)));
}

/** Restore a finite dimension while tolerating blocked/private browser storage. */
function storedDimension(storage, key, fallback) {
  try {
    const raw = storage?.getItem(key);
    if (raw !== null && raw !== undefined && Number.isFinite(Number(raw))) return Number(raw);
  } catch {}
  return fallback;
}

/** Add analysis-grid transport beside the native player, preserving all native controls. */
function mountTransport(document, preview, video) {
  const root = document.createElement("div");
  root.id = "editor-transport";
  root.className = "editor-transport";
  root.dataset.agWidth = "fill";
  root.setAttribute("role", "group");
  root.setAttribute("aria-label", uiText("Транспорт видео"));
  const controls = document.createElement("div");
  controls.className = "editor-transport-controls";
  controls.dataset.agWidth = "content";
  controls.dataset.agWidthMinEm = "1";
  root.append(controls);
  const commands = [
    ["start", "В начало", "M5 5v14m13-14-9 7 9 7z"],
    ["previous", "Шаг назад", "M6 5v14m12-14-8 7 8 7z"],
    ["play", "Воспроизвести", "m8 5 11 7-11 7z"],
    ["next", "Шаг вперёд", "M18 5v14M6 5l8 7-8 7z"],
    ["end", "В конец", "M19 5v14M6 5l9 7-9 7z"],
  ];
  const buttons = [];
  for (const [action, label, path] of commands) {
    const button = document.createElement("button");
    button.type = "button";
    button.dataset.editorAction = action;
    button.className = "editor-transport-button studio-icon";
    button.title = uiText(label);
    button.setAttribute("aria-label", uiText(label));
    attachStudioHint(button, uiText(label));
    const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
    svg.setAttribute("viewBox", "0 0 24 24");
    svg.setAttribute("aria-hidden", "true");
    const shape = document.createElementNS("http://www.w3.org/2000/svg", "path");
    shape.setAttribute("d", path);
    svg.append(shape);
    button.append(svg);
    button.onclick = () => performVideoAction(video, action, Number(video.dataset.editorFrameStepMs) || 40);
    controls.append(button);
    buttons.push(button);
  }
  const clock = document.createElement("output");
  clock.className = "editor-transport-clock";
  clock.setAttribute("aria-label", uiText("Позиция видео"));
  clock.dataset.agWidth = "content";
  clock.dataset.agWidthMinEm = "4";
  root.append(clock);
  preview.querySelector(".media-monitor")?.after(root);

  /** Update transport state from the native media clock without rendering timeline rows. */
  function update() {
    root.hidden = video.hidden || !video.videoWidth;
    for (const button of buttons) button.disabled = !videoActionEnabled(video, button.dataset.editorAction);
    const step = videoFrameStepMs(video);
    for (const button of [buttons[1], buttons[3]]) {
      button.title = `${button.getAttribute("aria-label")} · ${Math.round(step * 100) / 100} ${uiText("мс")} · ${uiText("сетка анализа")}`;
    }
    const play = buttons[2];
    const label = uiText(video.paused ? "Воспроизвести" : "Пауза");
    play.setAttribute("aria-label", label);
    play.title = label;
    play.querySelector("path").setAttribute("d", video.paused ? commands[2][2] : "M8 5v14m8-14v14");
    for (const button of buttons) button.dataset.hint = button.title;
    const seconds = Math.max(0, Number(video.currentTime) || 0);
    const minutes = Math.floor(seconds / 60);
    const fraction = String(Math.floor(seconds % 60)).padStart(2, "0");
    clock.value = `${String(minutes).padStart(2, "0")}:${fraction}.${String(Math.floor(seconds % 1 * 100)).padStart(2, "0")}`;
  }
  const events = ["loadedmetadata", "emptied", "play", "pause", "timeupdate", "ended", "error"];
  for (const event of events) video.addEventListener(event, update);
  update();
  return {update, dispose() {
    for (const event of events) video.removeEventListener(event, update);
    root.remove();
  }};
}

/** Mount a horizontal separator because AG's width-only panel contract has no height axis. */
function bindTimelineHeight(document, workbench, timeline, dimensions, limits, setHeight, persist) {
  const view = document.defaultView;
  const handle = document.createElement("div");
  handle.className = "editor-timeline-resizer";
  handle.setAttribute("role", "separator");
  handle.setAttribute("aria-orientation", "horizontal");
  handle.setAttribute("aria-label", uiText("Изменить высоту шкалы"));
  handle.setAttribute("aria-controls", timeline.id);
  handle.title = uiText("Потяните границу или используйте стрелки; Home / End — минимальный / максимальный размер");
  handle.tabIndex = 0;
  workbench.insertBefore(handle, timeline);
  let start = null;

  /** Apply identical bounds to pointer and keyboard changes and expose the current value. */
  function resize(value, remember = false) {
    const {minimum, maximum} = limits();
    const height = Math.round(Math.max(minimum, Math.min(maximum, value)));
    setHeight(height);
    handle.setAttribute("aria-valuemin", String(minimum));
    handle.setAttribute("aria-valuemax", String(maximum));
    handle.setAttribute("aria-valuenow", String(height));
    if (remember) persist(height);
  }
  const move = event => {
    if (!start || start.id !== event.pointerId) return;
    event.preventDefault();
    resize(start.height - (event.clientY - start.y));
  };
  const stop = () => {
    if (!start) return;
    start = null;
    view.removeEventListener("pointermove", move);
    view.removeEventListener("pointerup", stop);
    view.removeEventListener("pointercancel", stop);
    workbench.classList.remove("editor-resizing");
    persist(dimensions().timelineHeight);
  };
  const down = event => {
    if (event.button !== 0) return;
    event.preventDefault();
    start = {id: event.pointerId, y: event.clientY, height: dimensions().timelineHeight};
    workbench.classList.add("editor-resizing");
    view.addEventListener("pointermove", move);
    view.addEventListener("pointerup", stop);
    view.addEventListener("pointercancel", stop);
    handle.focus();
  };
  const key = event => {
    const step = event.shiftKey ? 32 : 8;
    const height = dimensions().timelineHeight;
    const value = {ArrowUp: height + step, ArrowDown: height - step, Home: limits().minimum, End: limits().maximum}[event.key];
    if (value === undefined) return;
    event.preventDefault();
    resize(value, true);
  };
  handle.addEventListener("pointerdown", down);
  handle.addEventListener("keydown", key);
  return {handle, resize, dispose() {
    stop();
    handle.removeEventListener("pointerdown", down);
    handle.removeEventListener("keydown", key);
    handle.remove();
  }};
}

/** Keep the existing monitor, transcript and timeline in a bounded, persistent desktop workspace. */
export function mountEditorWorkspace(document) {
  const workbench = document.querySelector(".editor-workbench");
  if (!workbench || MOUNTED.has(workbench)) return MOUNTED.get(workbench);
  const view = document.defaultView;
  const video = document.getElementById("media-preview");
  const inspector = document.getElementById("media-result");
  const timeline = document.getElementById("media-editor-timeline");
  const preview = workbench.querySelector(".editor-preview");
  if (!video || !inspector || !timeline || !preview) return;
  let storage;
  const originalInspectorWidth = inspector.style.width;
  try { storage = view.localStorage; } catch {}
  let dimensions = workspaceDimensions({
    inspectorWidth: storedDimension(storage, WIDTH_KEY, 320),
    timelineHeight: storedDimension(storage, HEIGHT_KEY, 320),
  });
  const save = (key, value) => { try { storage?.setItem(key, String(value)); } catch {} };
  workbench.dataset.agWidth = "fill";
  workbench.dataset.workspace = "editor";
  let requestedHeight = dimensions.timelineHeight;
  let requestedWidth = dimensions.inspectorWidth;
  const transport = mountTransport(document, preview, video);
  // The action row stays fixed; expanded mask fields and status messages use the scrollable area below.
  const secondary = document.createElement("div");
  secondary.className = "editor-preview-secondary";
  secondary.dataset.agWidth = "fill";
  const transportRoot = document.getElementById("editor-transport");
  const monitorActions = mountMonitorActions(document, transportRoot.querySelector(".editor-transport-controls"));
  while (transportRoot.nextElementSibling) secondary.append(transportRoot.nextElementSibling);
  transportRoot.after(secondary);
  const panelResize = bindResizablePanel({panel: inspector, edge: "left", minWidth: 240, maxWidth: 540,
    initialWidth: dimensions.inspectorWidth, storageKey: WIDTH_KEY, storage,
    reservedWidth: () => 200,
    onResize: width => {
      if (workbench.clientWidth && workbench.dataset.bounded === "true" && view.matchMedia("(min-width: 65rem)").matches) {
        dimensions.inspectorWidth = width;
      }
      workbench.style.setProperty("--editor-inspector-width", `${width}px`);
    }});
  panelResize.handle.setAttribute("aria-label", uiText("Изменить ширину инспектора"));
  panelResize.handle.setAttribute("aria-controls", inspector.id);
  panelResize.handle.title = uiText("Потяните границу или используйте стрелки; Home / End — минимальный / максимальный размер");
  let timelineLimits = {minimum: 280, maximum: 560};
  const heightResize = bindTimelineHeight(document, workbench, timeline, () => dimensions, () => timelineLimits,
    height => {
      dimensions.timelineHeight = height;
      workbench.style.setProperty("--editor-timeline-height", `${height}px`);
    }, height => { requestedHeight = height; save(HEIGHT_KEY, height); });
  const rememberWidth = () => {
    requestedWidth = panelResize.width;
    view.removeEventListener("pointerup", rememberWidth);
    view.removeEventListener("pointercancel", rememberWidth);
  };
  const beginWidth = () => {
    view.addEventListener("pointerup", rememberWidth);
    view.addEventListener("pointercancel", rememberWidth);
  };
  panelResize.handle.addEventListener("keydown", rememberWidth);
  panelResize.handle.addEventListener("pointerdown", beginWidth);
  const focusButton = document.createElement("button");
  focusButton.id = "editor-workspace-focus";
  focusButton.type = "button";
  focusButton.className = "studio-icon editor-workspace-focus";
  const focusIcon = document.createElementNS("http://www.w3.org/2000/svg", "svg");
  focusIcon.setAttribute("viewBox", "0 0 24 24");
  focusIcon.setAttribute("aria-hidden", "true");
  const focusPath = document.createElementNS("http://www.w3.org/2000/svg", "path");
  focusIcon.append(focusPath);
  focusButton.append(focusIcon);
  focusButton.setAttribute("aria-controls", "media-result media-editor-timeline media-preview");
  focusButton.setAttribute("aria-keyshortcuts", "Escape");
  attachStudioHint(focusButton, uiText("Развернуть рабочую область"));
  const inspectorToggle = document.getElementById("editor-inspector-toggle");
  inspectorToggle?.before(focusButton);
  let focused = false;

  /** Enlarge the same DOM workspace locally; playback, source URL and report remain untouched. */
  function setFocused(value, restoreFocus = true) {
    focused = value;
    workbench.dataset.focused = String(value);
    document.body.classList.toggle("editor-workspace-focused", value);
    const label = uiText(value ? "Вернуть обычный вид" : "Развернуть рабочую область");
    focusPath.setAttribute("d", value ? "M4 9h5V4m6 0v5h5M4 15h5v5m6 0v-5h5" : "M9 4H4v5m11-5h5v5M4 15v5h5m11-5v5h-5");
    focusButton.title = label;
    focusButton.dataset.hint = label;
    focusButton.setAttribute("aria-label", label);
    focusButton.setAttribute("aria-pressed", String(value));
    if (!value && restoreFocus) focusButton.focus({preventScroll: true});
    schedule();
  }
  focusButton.onclick = () => setFocused(!focused);
  const escapeFocus = event => {
    if (!focused || event.key !== "Escape" || event.defaultPrevented || event.isComposing || document.querySelector("dialog[open]")) return;
    try {
      const menus = [...document.querySelectorAll(":popover-open")];
      if (menus.some(menu => menu.getAttribute("aria-hidden") !== "true")) return;
    } catch {}
    event.preventDefault();
    setFocused(false);
  };
  document.addEventListener("keydown", escapeFocus);

  /** Recompute only layout preferences when a surface appears, collapses or changes viewport. */
  function layout() {
    if (!workbench.clientWidth) {
      if (focused) setFocused(false, false);
      transport.update();
      return;
    }
    const viewport = view.visualViewport?.height || view.innerHeight;
    const top = workbench.getBoundingClientRect().top;
    const topbar = document.querySelector(".studio-topbar");
    const sidebar = document.querySelector(".studio-sidebar");
    const topInset = Math.max(0, topbar?.getBoundingClientRect().bottom || 0);
    const sideInset = sidebar && view.getComputedStyle(sidebar).display !== "none" ? sidebar.getBoundingClientRect().right : 0;
    workbench.style.setProperty("--editor-focus-top", `${topInset}px`);
    workbench.style.setProperty("--editor-focus-left", `${sideInset}px`);
    const available = focused ? viewport - topInset - 16 : viewport - (top > 0 && top < viewport - 400 ? top : 96) - 16;
    const budget = Math.max(0, viewport - topInset - 16);
    const active = !video.hidden && Boolean(video.videoWidth);
    const desktop = view.matchMedia("(min-width: 65rem)").matches;
    const scroll = document.getElementById("media-editor-scroll");
    const ruler = timeline.querySelector(".editor-band-ruler");
    const chrome = Math.max(0, timeline.getBoundingClientRect().height - (scroll?.clientHeight || 0));
    const minimum = timeline.hidden ? 0 : minimumTimelineHeight(chrome, ruler?.getBoundingClientRect().height || 36,
      viewport >= 800 ? 128 : 96);
    let previewChrome = 0;
    for (const node of preview.children) {
      if (!node.hidden && !node.classList.contains("media-monitor")) previewChrome += node.getBoundingClientRect().height;
    }
    const previewMinimum = Math.max(180, previewChrome + 120);
    const required = minimum + previewMinimum + 6;
    const enoughSpace = !desktop || timeline.hidden || required <= budget;
    const workspaceHeight = Math.min(960, budget, Math.max(available, required));
    workbench.style.setProperty("--editor-workspace-height", `${Math.max(400, workspaceHeight)}px`);
    timelineLimits = {minimum: Math.max(140, minimum), maximum: Math.max(minimum, workspaceHeight - previewMinimum - 6)};
    focusButton.disabled = !active;
    workbench.dataset.bounded = String(active && enoughSpace);
    workbench.dataset.timelineOpen = String(!timeline.hidden);
    workbench.dataset.inspectorOpen = String(!inspector.hidden && inspector.dataset.collapsed !== "true");
    if (active && desktop && enoughSpace) {
      heightResize.resize(requestedHeight);
      panelResize.setWidth(requestedWidth, {remember: false});
    }
    const focus = document.activeElement;
    const hiddenSplitter = (!desktop || !enoughSpace) && (focus === panelResize.handle || focus === heightResize.handle);
    const hiddenInspector = (inspector.hidden || inspector.dataset.collapsed === "true") && inspector.contains(focus);
    if (hiddenSplitter || hiddenInspector) {
      const toggle = document.getElementById("editor-inspector-toggle");
      if (toggle && !toggle.hidden) toggle.focus({preventScroll: true});
      else video.focus({preventScroll: true});
    }
    transport.update();
  }
  let scheduled = 0;
  const schedule = () => {
    if (!scheduled) scheduled = view.requestAnimationFrame(() => { scheduled = 0; layout(); });
  };
  const observer = new view.ResizeObserver(schedule);
  observer.observe(workbench);
  for (const node of [...timeline.children, ...preview.children]) {
    if (!node.classList.contains("editor-scroll") && !node.classList.contains("media-monitor")) observer.observe(node);
  }
  const changes = new view.MutationObserver(schedule);
  for (const node of [video, inspector, timeline, workbench.closest("main")].filter(Boolean)) {
    changes.observe(node, {attributes: true, attributeFilter: ["hidden", "data-collapsed"]});
  }
  changes.observe(document.documentElement, {attributes: true, attributeFilter: ["data-sidebar-collapsed"]});
  view.addEventListener("resize", schedule);
  document.addEventListener("studio-design-change", schedule);
  video.addEventListener("loadedmetadata", schedule);
  video.addEventListener("emptied", schedule);
  workbench.addEventListener("editor-inspector-change", schedule);
  setFocused(false, false);
  const controller = {refresh: schedule, dispose() {
    observer.disconnect();
    changes.disconnect();
    view.cancelAnimationFrame(scheduled);
    view.removeEventListener("resize", schedule);
    document.removeEventListener("studio-design-change", schedule);
    video.removeEventListener("loadedmetadata", schedule);
    video.removeEventListener("emptied", schedule);
    workbench.removeEventListener("editor-inspector-change", schedule);
    document.removeEventListener("keydown", escapeFocus);
    panelResize.handle.removeEventListener("keydown", rememberWidth);
    panelResize.handle.removeEventListener("pointerdown", beginWidth);
    view.removeEventListener("pointerup", rememberWidth);
    view.removeEventListener("pointercancel", rememberWidth);
    document.body.classList.remove("editor-workspace-focused");
    focusButton.remove();
    panelResize.dispose();
    heightResize.dispose();
    secondary.replaceWith(...secondary.childNodes);
    monitorActions.dispose();
    transport.dispose();
    inspector.style.width = originalInspectorWidth;
    for (const key of ["workspace", "bounded", "focused", "timelineOpen", "inspectorOpen"]) delete workbench.dataset[key];
    for (const key of ["--editor-inspector-width", "--editor-timeline-height", "--editor-workspace-height", "--editor-focus-top", "--editor-focus-left"]) {
      workbench.style.removeProperty(key);
    }
    MOUNTED.delete(workbench);
  }};
  MOUNTED.set(workbench, controller);
  return controller;
}
