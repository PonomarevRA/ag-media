/** Reuse the existing export, checkbox and clothing disclosure in one monitor action row. */
export function mountMonitorActions(document, controls) {
  const overlay = document.getElementById("media-overlay-options");
  const download = document.getElementById("media-video-export");
  const masks = document.getElementById("editor-clothing-toggle");
  const details = document.getElementById("media-clothing-export");
  const positions = [];
  for (const node of [overlay, download, masks]) {
    if (!node) continue;
    positions.push({node, parent: node.parentNode, next: node.nextSibling});
    controls.append(node);
  }
  const label = overlay?.querySelector("label");
  const restoreHint = () => label?.removeAttribute("data-hint-dismissed");
  label?.addEventListener("focusin", restoreHint);

  /** Mirror readiness and native open state; the original mask-export controller owns the form. */
  function syncDisclosure() {
    if (!masks || !details) return;
    masks.hidden = details.hidden;
    masks.setAttribute("aria-expanded", String(!details.hidden && details.open));
  }
  const toggle = () => { if (details && !details.hidden) details.open = !details.open; };
  masks?.addEventListener("click", toggle);
  details?.addEventListener("toggle", syncDisclosure);
  const observer = new document.defaultView.MutationObserver(syncDisclosure);
  if (details) {
    details.dataset.toolbarDisclosure = "true";
    observer.observe(details, {attributes: true, attributeFilter: ["hidden"]});
  }
  syncDisclosure();
  return {
    /** Restore owned nodes so remounting cannot lose their handlers, fields or source state. */
    dispose() {
      observer.disconnect();
      label?.removeEventListener("focusin", restoreHint);
      masks?.removeEventListener("click", toggle);
      details?.removeEventListener("toggle", syncDisclosure);
      if (details) delete details.dataset.toolbarDisclosure;
      // Restore the last moved sibling first; an earlier node may use it as its insertion anchor.
      for (let index = positions.length - 1; index >= 0; index--) {
        const {node, parent, next} = positions[index];
        if (parent) parent.insertBefore(node, next?.parentNode === parent ? next : null);
      }
    },
  };
}
