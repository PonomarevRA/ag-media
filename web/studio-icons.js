const paths = {
  edit: "m4 16 12-12 4 4L8 20H4v-4Zm10-10 4 4",
  check: "m5 12 4 4L20 5",
  pin: "m9 3 6 0-1 6 4 4H6l4-4-1-6Zm3 10v8",
  voices: "M3 8v8m4-11v14m4-9v4m3-6v8m4-13v18m4-12v6",
  spell: "M3 15 7 4l4 11M5 10h4m5 8 3 3 5-6",
  info: "M12 10v7m0-10h.01M21 12a9 9 0 1 1-18 0 9 9 0 0 1 18 0Z",
  photo: "M3 4h18v16H3Zm0 12 5-5 5 5 3-3 5 5M16 8h.01",
  "face-check": "M9 9h.01M14 9h.01M9 14q2 2 4 0M20 11a9 9 0 1 0-8 10m3-3 3 3 5-6",
  "face-partial": "M9 9h.01M14 9h.01M8 14h6M20 11a9 9 0 1 0-8 10m3-3h7",
  "face-question": "M9 9h.01M14 9h.01M8 14h6M20 11a9 9 0 1 0-8 10m4-5a2 2 0 1 1 3 2v1m0 3h.01",
  subtitles: "M4 3h11l5 5v13H4Zm11 0v6h5M7 13h4m2 0h4m-10 4h3m2 0h5",
  json: "M4 3h11l5 5v13H4Zm11 0v6h5m-9 3H9v2l-2 1 2 1v2h2m2-6h2v2l2 1-2 1v2h-2",
  chart: "M4 3v18h17M8 16v-5m5 5V6m5 10V9",
  studio: "M4 10v4m4-7v10m4-14v18m4-14v10m4-7v4",
  mic: "M9 6a3 3 0 1 1 6 0v5a3 3 0 1 1-6 0Zm-4 5a7 7 0 0 0 14 0m-7 7v3m-3 0h6",
  history: "M3 12a9 9 0 1 1 2 5.6M3 6v6h6m3-5v5l3 2",
  storage: "M4 5h16v14H4zM4 14h16m-12 2.5h1m3 0h1",
  book: "M12 6v15M3 4q5-2 9 2 4-4 9-2v15q-5-2-9 2-4-4-9-2Z",
  download: "M12 3v12m-5-5 5 5 5-5M4 16v5h16v-5",
  technical: "M3 8V3h5m8 0h5v5m0 8v5h-5m-8 0H3v-5m5-8h8v8H8Z",
  sparkles: "m9 7 1.8 5.2L16 14l-5.2 1.8L9 21l-1.8-5.2L2 14l5.2-1.8ZM18 2l1.1 3.9L23 7l-3.9 1.1L18 12l-1.1-3.9L13 7l3.9-1.1ZM4 2l.7 2.3L7 5l-2.3.7L4 8l-.7-2.3L1 5l2.3-.7Z",
  settings: "M4 7h3m4 0h9M4 12h9m4 0h3M4 17h3m4 0h9M7 5v4m6 1v4m-6 1v4",
  menu: "M4 6h16M4 12h16M4 18h16",
  close: "m6 6 12 12M18 6 6 18",
  inspector: "M4 5h16v14H4zM14 5v14m2-10h2m-2 4h2",
  film: "M3 5h18v14H3zM7 5v14m10-14v14M3 10h4m10 0h4M3 14h4m10 0h4",
  face: "M9 10h.01M15 10h.01M9 15q3 3 6 0M3 8V3h5m8 0h5v5m0 8v5h-5m-8 0H3v-5",
  person: "M12 3a3 3 0 1 0 0 6 3 3 0 0 0 0-6ZM5 21v-3a7 7 0 0 1 14 0v3",
  cube: "m12 3 8 4.5v9L12 21l-8-4.5v-9L12 3Zm-8 4.5 8 4.5 8-4.5m-8 4.5v9",
  ocr: "M3 8V4h4m10 0h4v4m0 8v4h-4m-10 0H3v-4m5-7h8m-8 3h8m-8 3h5",
  undo: "M8 4 3 9l5 5M3 9h11a6 6 0 0 1 0 12h-3",
  redo: "m16 4 5 5-5 5m5-5H10a6 6 0 0 0 0 12h3",
  refresh: "M20 10a8 8 0 0 0-13.5-4L4 8m0-5v5h5m-5 6a8 8 0 0 0 13.5 4l2.5-2m0 5v-5h-5",
  trash: "M4 6h16M9 6V3h6v3m3 0-1 15H7L6 6m4 4v7m4-7v7",
};

/** Attach a non-interactive icon and let the action supply its accessible name. */
export function attachStudioIcon(node) {
  const ns = "http://www.w3.org/2000/svg";
  const svg = document.createElementNS(ns, "svg");
  svg.setAttribute("viewBox", "0 0 24 24");
  svg.setAttribute("aria-hidden", "true");
  svg.setAttribute("focusable", "false");
  const path = document.createElementNS(ns, "path");
  path.setAttribute("d", paths[node.dataset.icon]);
  svg.append(path);
  node.prepend(svg);
  if (node.hasAttribute("data-hint")) attachStudioHint(node, node.dataset.hint || node.getAttribute("aria-label"));
}

/** Display the same short label on hover and keyboard focus, with Escape dismissal. */
export function attachStudioHint(node, label = node.getAttribute("aria-label")) {
  node.dataset.hint = label;
  const restoreHint = () => node.removeAttribute("data-hint-dismissed");
  node.addEventListener("focus", restoreHint);
  node.addEventListener("pointerenter", restoreHint);
  node.addEventListener("keydown", event => {
    if (event.key !== "Escape" || node.hasAttribute("data-hint-dismissed")) return;
    if (getComputedStyle(node, "::after").display === "none") return;
    node.setAttribute("data-hint-dismissed", "");
    event.preventDefault();
  });
}
