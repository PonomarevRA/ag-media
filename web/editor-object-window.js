import {uiText} from "./i18n.js";

/** Mount the shared nonmodal window controls and its own recovery entry. */
export function mountEditorWindow(find, {prefix, panel, heading, onClose, onVisibility = () => {}}) {
  const shelf = find(prefix + "-shelf");
  const restore = find(prefix + "-restore");
  const restoreLabel = find(prefix + "-restore-label");
  const expand = find(prefix + "-expand");
  let expanded = false;
  const align = bindWindowPlacement(panel, find("media-result"));

  /** Keep the panel and its recovery entry mutually exclusive. */
  function setVisibility(visible, minimized = false) {
    panel.hidden = !visible;
    if (!visible) panel.removeAttribute("data-front");
    shelf.hidden = !minimized;
    onVisibility(visible);
  }

  /** Raise this inspector above other visible detail windows. */
  function raise() {
    const previous = panel.ownerDocument.querySelector(".editor-object-window[data-front]");
    if (previous === panel) return;
    previous?.removeAttribute("data-front");
    panel.dataset.front = "";
  }
  panel.addEventListener("pointerdown", raise);
  panel.addEventListener("focusin", raise);

  /** Restore the selected content and return focus to its heading. */
  function show() {
    setVisibility(true);
    raise();
    align();
    heading.focus({preventScroll: true});
  }

  /** Preserve the current sample while exposing the recovery action. */
  function minimize() {
    setVisibility(false, true);
    restore.focus({preventScroll: true});
  }

  find(prefix + "-minimize").onclick = minimize;
  find(prefix + "-close").onclick = onClose;
  restore.onclick = show;
  expand.onclick = () => {
    expanded = !expanded;
    panel.dataset.expanded = String(expanded);
    expand.setAttribute("aria-pressed", String(expanded));
    align();
  };
  panel.addEventListener("keydown", event => {
    if (event.key !== "Escape" || event.defaultPrevented) return;
    event.preventDefault();
    event.stopPropagation();
    onClose();
  });

  return {
    show,
    clear: () => setVisibility(false),
    visible: () => !panel.hidden,
    setLabel(label) {
      restoreLabel.textContent = label;
      restore.setAttribute("aria-label", uiText`Восстановить окно: ${label}`);
    },
  };
}

/** Drag the heading in desktop layouts, keeping the title and controls within reach. */
function bindWindowPlacement(panel, transcript) {
  const header = panel.querySelector(".editor-object-window-heading");
  const view = panel.ownerDocument.defaultView;
  const desktop = view.matchMedia("(min-width: 40.01rem)");
  let drag;
  let position;

  /** Clamp coordinates against actual content and viewport dimensions. */
  function place(x, y) {
    const gutter = 12;
    const limitX = view.document.documentElement.clientWidth - panel.offsetWidth - gutter;
    const limitY = view.innerHeight - header.offsetHeight - gutter * 2;
    position = {x: Math.max(gutter, Math.min(x, limitX)), y: Math.max(gutter, Math.min(y, limitY))};
    panel.style.setProperty("--object-window-x", position.x + "px");
    panel.style.setProperty("--object-window-y", position.y + "px");
  }

  header.addEventListener("pointerdown", event => {
    if (!desktop.matches || panel.dataset.expanded === "true" || event.button !== 0 ||
        event.target.closest("button")) return;
    const rect = panel.getBoundingClientRect();
    drag = {id: event.pointerId, dx: event.clientX - rect.left, dy: event.clientY - rect.top};
    header.setPointerCapture(drag.id);
  });
  header.addEventListener("pointermove", event => {
    if (event.pointerId === drag?.id) place(event.clientX - drag.dx, event.clientY - drag.dy);
  });
  header.addEventListener("lostpointercapture", () => { drag = undefined; });
  /** Keep the transcript visible when the inspector fits over the monitor. */
  function align() {
    const width = panel.offsetWidth;
    if (!desktop.matches || !width || panel.dataset.expanded === "true") return;
    if (position) {
      place(position.x, position.y);
      return;
    }
    const transcriptBounds = transcript.getBoundingClientRect();
    const workspace = transcript.parentElement.getBoundingClientRect();
    const left = transcriptBounds.left - width - 12;
    if (transcriptBounds.width && left >= workspace.left + 12) {
      panel.style.setProperty("--object-window-default-x", left + "px");
    } else {
      panel.style.removeProperty("--object-window-default-x");
    }
  }
  view.addEventListener("resize", align);
  transcript.addEventListener("editor-inspector-change", () => view.requestAnimationFrame(align));
  return align;
}
