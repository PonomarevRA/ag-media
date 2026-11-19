import {uiLocale} from "./i18n.js";

const percent = new Intl.NumberFormat(uiLocale, {style: "percent", maximumFractionDigits: 1});

/** Render each measured swatch with its color and share as text. */
export function colorPalette(samples, label) {
  const group = document.createElement("div");
  group.className = "video-color-palette";
  if (label) group.setAttribute("aria-label", label);
  for (const item of Array.isArray(samples) ? samples : []) {
    if (!/^#[0-9a-f]{6}$/i.test(item?.hex) || !Number.isFinite(item.share) || item.share < 0 || item.share > 1) continue;
    const swatch = document.createElement("span");
    swatch.style.setProperty("--swatch", item.hex);
    swatch.textContent = `${item.hex} · ${percent.format(item.share)}`;
    group.append(swatch);
  }
  return group;
}
