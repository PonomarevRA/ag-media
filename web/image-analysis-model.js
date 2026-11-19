/** Decide the import route even when a local file has no browser-supplied MIME type. */
export function isImageFile(file) {
  return file.type?.startsWith("image/") || /\.(png|jpe?g|webp|gif|bmp|avif|heic|svg)$/i.test(file.name || "");
}

/** Explain upload limits before replacing an existing media selection. The server validates actual bytes. */
export function imageFileError(file) {
  if (!/\.(png|jpe?g)$/i.test(file.name || "") && !["image/png", "image/jpeg"].includes(file.type))
    return "Выбери картинку PNG или JPEG.";
  if (!file.size || file.size > 24 * 1024 * 1024) return "Картинка должна занимать не больше 24 МиБ.";
  return null;
}

/** Reject nonfinite, empty or out-of-frame proposals before drawing normalized image coordinates. */
export function imageBox(value) {
  if (!Array.isArray(value) || value.length !== 4 || !value.every(Number.isFinite)) return null;
  const [x, y, width, height] = value;
  if (x < 0 || y < 0 || width <= 0 || height <= 0 || x + width > 1.01 || y + height > 1.01) return null;
  return [x, y, Math.min(width, 1 - x), Math.min(height, 1 - y)];
}

/** Keep detected objects, inferred UI controls and OCR observations distinct; confidence is never invented. */
export function imageItems(report) {
  const items = [];
  const groups = [["objects", report.objects], ["ui", report.interface?.elements], ["text", report.ocr]];
  for (const [layer, rows] of groups) {
    if (!Array.isArray(rows)) continue;
    for (const row of rows.slice(0, 120)) {
      if (layer === "ui" && row.role === "text_line") continue;
      const box = imageBox(row.box);
      if (!box) continue;
      const score = typeof row.score === "number" && Number.isFinite(row.score) && row.score >= 0 && row.score <= 1
        ? row.score : null;
      items.push({id: `${layer}-${items.length}`, layer, box, score,
        label: String(row.label || row.text || row.role || "").slice(0, 400), role: row.role || ""});
    }
  }
  return items;
}
