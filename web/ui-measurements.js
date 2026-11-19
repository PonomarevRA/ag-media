import {px, rectangle, unionRectangles, insets, compareRectangles} from "./ui-measurement-geometry.js";

export const MEASUREMENT_LIMITS = Object.freeze({targets: 32, pairs: 32, catalogScan: 4000, textNodes: 256, textFragments: 64});
const ID = /^[^\s\u0000-\u001f]{1,128}$/u;
const NATIVE_TEXT = new Set(["INPUT", "TEXTAREA", "SELECT"]);

/** Bound standalone calls too. Strings identify exact DOM IDs, never selectors or scripts. */
function validate(options) {
  const allowed = new Set(["targets", "pairs", "tolerance_px", "offset", "limit", "include_text"]);
  if (!options || typeof options !== "object" || Array.isArray(options) || Object.keys(options).some(key => !allowed.has(key)))
    throw new TypeError("Use a measurement options object with known fields.");
  const {targets = [], pairs = [], tolerance_px = 1, offset = 0, limit = 32, include_text = true} = options;
  const validId = id => typeof id === "string" && ID.test(id);
  if (!Array.isArray(targets) || targets.length > 32 || !targets.every(validId) || !Array.isArray(pairs) || pairs.length > 32 ||
      !Number.isFinite(tolerance_px) || tolerance_px < 0 || tolerance_px > 16 || !Number.isSafeInteger(offset) || offset < 0 || offset > 4000 ||
      !Number.isSafeInteger(limit) || limit < 1 || limit > 50 || typeof include_text !== "boolean")
    throw new TypeError("Invalid measurement batch; use up to32 element IDs and32 pairs.");
  const ids = new Set(targets);
  for (const pair of pairs) {
    if (!pair || typeof pair !== "object" || Array.isArray(pair) || Object.keys(pair).some(key => key !== "from" && key !== "to") ||
        !Object.hasOwn(pair, "from") || !Object.hasOwn(pair, "to") ||
        !validId(pair.from) || !validId(pair.to)) throw new TypeError("Each pair needs literal from/to element IDs.");
    ids.add(pair.from); ids.add(pair.to);
  }
  if (ids.size > 32) throw new TypeError("The combined batch exceeds32 element IDs.");
  return {ids: [...ids], pairs, tolerance: tolerance_px, offset, limit, includeText: include_text};
}

/** Read a synchronous viewport-CSS-pixel snapshot without scrolling, mutation, models or text/value export.
 * Empty targets discover rendered IDs. Unsupported facts remain unknown rather than guessed dimensions.
 */
export function measureUi(options = {}, document = globalThis.document) {
  const settings = validate(options);
  if (!document?.defaultView) throw new TypeError("A connected browser document is required.");
  const view = document.defaultView, started = view.performance.now();
  const candidates = document.querySelectorAll("[id]"), idCounts = new Map();
  for (let index = 0; index < Math.min(candidates.length, MEASUREMENT_LIMITS.catalogScan); index++) {
    const id = candidates[index].id;
    idCounts.set(id, (idCounts.get(id) ?? 0) + 1);
  }
  const styles = new WeakMap(), rawBoxes = new Map();
  const styleOf = element => {
    if (!styles.has(element)) styles.set(element, view.getComputedStyle(element));
    return styles.get(element);
  };

  /** Separate hidden/unsupported layout and clipped accessible-name text from ordinary boxes. */
  function availability(element, text = false) {
    let cursor = element, depth = 0;
    while (cursor) {
      if (++depth > 64) return "ancestor-limit";
      const style = styleOf(cursor);
      if (cursor.hidden || style.display === "none" || ["hidden", "collapse"].includes(style.visibility) ||
          style.contentVisibility === "hidden" || Number(style.opacity) === 0) return "hidden";
      if (text && ((style.clipPath && style.clipPath !== "none") || (style.clip && style.clip !== "auto"))) return "clipped-text";
      if ((style.transform && style.transform !== "none") ||
          [style.scale, style.rotate, style.translate].some(value => value && value !== "none") ||
          (style.zoom && !["1", "normal"].includes(style.zoom))) return "transformed";
      cursor = cursor.parentElement;
    }
    return null;
  }

  /** Bound Range traversal; native values and generated pseudo-content are never inferred. */
  function textGeometry(element, box, contentBox) {
    if (!settings.includeText) return {status: "skipped"};
    if (NATIVE_TEXT.has(element.tagName)) return {status: "unknown", reason: "native-control-text"};
    const walker = document.createTreeWalker(element, 4), fragments = [], omitted = new Set();
    let node, scanned = 0, truncated = false;
    while ((node = walker.nextNode())) {
      if (++scanned > MEASUREMENT_LIMITS.textNodes) { truncated = true; break; }
      if (!node.nodeValue?.trim()) continue;
      const parent = node.parentElement;
      if (parent.closest("script,style,noscript,template")) continue;
      const unavailable = availability(parent, true);
      if (unavailable === "hidden") continue;
      const unsupported = unavailable || (parent.closest("svg") ? "svg-text" :
        parent.closest("select,textarea,input") ? "native-control-text" : null);
      if (unsupported) { omitted.add(unsupported); continue; }
      const range = document.createRange();
      range.selectNodeContents(node);
      for (const fragment of range.getClientRects()) {
        if (!fragment.width || !fragment.height) continue;
        if (fragments.length === MEASUREMENT_LIMITS.textFragments) { truncated = true; break; }
        fragments.push(rectangle(fragment));
      }
      if (truncated) break;
    }
    const bounds = unionRectangles(fragments);
    if (truncated) omitted.add("text-scan-limit");
    const incomplete = omitted.size > 0;
    return {status: incomplete ? bounds ? "partial" : "unknown" : bounds ? "ok" : "empty", ...(incomplete && {reasons: [...omitted]}),
      box: bounds, fragmentCount: fragments.length,
      insetsPx: incomplete ? null : insets(box, bounds), contentInsetsPx: incomplete ? null : insets(contentBox, bounds)};
  }

  /** Read physical CSS slots, keeping unresolved properties as null. */
  function edges(style, prefix, suffix = "") {
    const values = {};
    for (const side of ["Top", "Right", "Bottom", "Left"]) values[side.toLowerCase()] = px(Number.parseFloat(style[prefix + side + suffix]));
    return values;
  }

  /** Read an exact element ID; punctuation cannot turn into a query selector. */
  function target(id) {
    const element = document.getElementById(id);
    if (!element) return {id, status: "unknown", reason: "not-found"};
    if (candidates.length > MEASUREMENT_LIMITS.catalogScan) return {id, status: "unknown", reason: "id-scan-limit"};
    if (idCounts.get(id) !== 1) return {id, status: "unknown", reason: "ambiguous-id"};
    const unavailable = availability(element);
    if (unavailable === "hidden") return {id, tag: element.tagName.toLowerCase(), status: "excluded", reason: "hidden"};
    if (!element.getClientRects().length) return {id, status: "unknown", reason: "no-layout-box"};
    const raw = element.getBoundingClientRect(), box = rectangle(raw);
    if (unavailable) return {id, status: "unknown", reason: unavailable, observedBox: box};
    const style = styleOf(element), padding = edges(style, "padding"), border = edges(style, "border", "Width");
    if (!box || style.display === "inline" || style.display === "contents")
      return {id, status: "unknown", reason: "fragmented-inline-box", observedBox: box};
    const contentBox = {x: px(box.x + element.clientLeft + padding.left), y: px(box.y + element.clientTop + padding.top),
      width: px(Math.max(0, element.clientWidth - padding.left - padding.right)),
      height: px(Math.max(0, element.clientHeight - padding.top - padding.bottom))};
    rawBoxes.set(id, {x: raw.left, y: raw.top, width: raw.width, height: raw.height});
    const icons = [];
    let visitedIcons = 0;
    for (const icon of element.querySelectorAll("svg")) {
      if (++visitedIcons > 16 || icons.length === 4) break;
      if (!availability(icon)) icons.push(rectangle(icon.getBoundingClientRect()));
    }
    return {id, status: "ok", tag: element.tagName.toLowerCase(), box, paddingPx: padding, borderPx: border, marginPx: edges(style, "margin"), contentBox,
      fontSizePx: px(Number.parseFloat(style.fontSize)),
      inViewport: box.x + box.width > 0 && box.y + box.height > 0 && box.x < view.innerWidth && box.y < view.innerHeight,
      text: textGeometry(element, box, contentBox), ...(icons.length && {iconLayoutBoxes: icons})};
  }

  const result = {schemaVersion: "ag-media/ui-measurements/1", coordinateSpace: "viewport-css-px",
    viewport: {width: view.innerWidth, height: view.innerHeight, scrollX: px(view.scrollX), scrollY: px(view.scrollY),
      devicePixelRatio: view.devicePixelRatio, fonts: document.fonts?.status ?? "unknown"}, limits: MEASUREMENT_LIMITS,
    capturedAtMs: Date.now(),
    notes: ["DOM Range layout boxes are not painted glyph ink; pseudo-content and native field text are unknown.",
      "Reported pixels round to0.01; alignment uses unrounded boxes. Content dimensions use integer CSSOM client sizes.",
      "Rectangles do not certify hit testing, occlusion, clipping, visual quality or accessibility."]};
  if (!settings.ids.length) {
    const catalog = [];
    let index = settings.offset;
    for (; index < Math.min(candidates.length, MEASUREMENT_LIMITS.catalogScan) && catalog.length < settings.limit; index++) {
      const element = candidates[index];
      if (!ID.test(element.id) || availability(element) === "hidden" || !element.getClientRects().length) continue;
      if (idCounts.get(element.id) === 1) catalog.push({id: element.id, tag: element.tagName.toLowerCase()});
    }
    return {...result, catalog, nextOffset: index < Math.min(candidates.length, MEASUREMENT_LIMITS.catalogScan) ? index : null,
      truncated: candidates.length > MEASUREMENT_LIMITS.catalogScan, elapsedMs: px(view.performance.now() - started)};
  }
  const targets = settings.ids.map(target), measured = new Map(targets.map(item => [item.id, item]));
  const pairs = [];
  for (const pair of settings.pairs) {
    const from = measured.get(pair.from), to = measured.get(pair.to);
    pairs.push(from.status === "ok" && to.status === "ok"
      ? {...pair, status: "ok", ...compareRectangles(rawBoxes.get(from.id), rawBoxes.get(to.id), settings.tolerance)}
      : {...pair, status: "unknown", reason: "target-unmeasured"});
  }
  return {...result, targets, pairs, tolerancePx: settings.tolerance, elapsedMs: px(view.performance.now() - started)};
}
