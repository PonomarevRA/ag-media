/** Round measured CSS pixels for bounded transport; never replace unavailable geometry with zero. */
export function px(value) {
  return Number.isFinite(value) ? Math.round(value * 100) / 100 : null;
}

/** Copy a viewport rectangle without live DOM references or device-pixel conversion. */
export function rectangle(value) {
  if (!value || ![value.left, value.top, value.width, value.height].every(Number.isFinite)) return null;
  return {x: px(value.left), y: px(value.top), width: px(value.width), height: px(value.height)};
}

/** Union visible text layout fragments; this is a layout box, not painted glyph ink. */
export function unionRectangles(rectangles) {
  if (!rectangles.length) return null;
  let left = Infinity, top = Infinity, right = -Infinity, bottom = -Infinity;
  for (const box of rectangles) {
    left = Math.min(left, box.x); top = Math.min(top, box.y);
    right = Math.max(right, box.x + box.width); bottom = Math.max(bottom, box.y + box.height);
  }
  return {x: px(left), y: px(top), width: px(right - left), height: px(bottom - top)};
}

/** Signed inside distances; a negative result means the inner box extends beyond that edge. */
export function insets(outer, inner) {
  if (!outer || !inner) return null;
  return {top: px(inner.y - outer.y), right: px(outer.x + outer.width - inner.x - inner.width),
    bottom: px(outer.y + outer.height - inner.y - inner.height), left: px(inner.x - outer.x)};
}

/** Compare axis-aligned boxes. Gaps and overlap are separate so crossing boxes never become a negative margin. */
export function compareRectangles(from, to, tolerance = 1) {
  const rightA = from.x + from.width, bottomA = from.y + from.height;
  const rightB = to.x + to.width, bottomB = to.y + to.height;
  const differences = {left: to.x - from.x, top: to.y - from.y, right: rightB - rightA, bottom: bottomB - bottomA,
    centerX: to.x + to.width / 2 - from.x - from.width / 2,
    centerY: to.y + to.height / 2 - from.y - from.height / 2,
    width: to.width - from.width, height: to.height - from.height};
  const aligned = {}, alignment = {};
  for (const [key, value] of Object.entries(differences)) {
    aligned[key] = Math.abs(value) <= tolerance;
    alignment[key] = px(value);
  }
  return {gapPx: {x: px(Math.max(to.x - rightA, from.x - rightB, 0)),
    y: px(Math.max(to.y - bottomA, from.y - bottomB, 0))},
  overlapPx: {x: px(Math.max(0, Math.min(rightA, rightB) - Math.max(from.x, to.x))),
    y: px(Math.max(0, Math.min(bottomA, bottomB) - Math.max(from.y, to.y)))},
  alignmentDeltaPx: alignment, aligned};
}
