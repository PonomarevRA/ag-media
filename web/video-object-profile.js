import {uiLocale, uiText} from "./i18n.js";
import {colorPalette} from "./video-color-palette.js";

const coefficient = new Intl.NumberFormat(uiLocale, {maximumFractionDigits: 3});
const number = new Intl.NumberFormat(uiLocale, {maximumFractionDigits: 2});
const validCorrelation = value => Number.isFinite(value) && Math.abs(value) <= 1;
const validScale = value => Number.isFinite(value) && value > 0;

/** Use the graph's ranked pair so its two source views match the displayed scores. */
function firstComparison(matches) {
  return (Array.isArray(matches) ? matches : []).find(match => match &&
    (validCorrelation(match.objectSimilarity) || validCorrelation(match.backgroundSimilarity) ||
      validScale(match.scaleRatio)));
}

/** Format all three measurements from the chosen pair, including missing values. */
function comparisonRows({objectSimilarity, backgroundSimilarity, scaleRatio}) {
  return [
    ["Сходство предмета", validCorrelation(objectSimilarity) ? coefficient.format(objectSimilarity) : null],
    ["Сходство фона", validCorrelation(backgroundSimilarity) ? coefficient.format(backgroundSimilarity) : null],
    ["Размер второй рамки относительно первой", validScale(scaleRatio) ? `×${number.format(scaleRatio)}` : null],
  ];
}

/** Return labeled palette and comparison blocks for profiles with measured visual evidence. */
export function objectAppearance(appearance, paletteOverride, timecode, profile) {
  const content = document.createDocumentFragment();
  const paletteTitle = uiText(paletteOverride ? "Цвета внутри рамки" : "Цвета основного кадра");
  const palette = colorPalette(paletteOverride ?? appearance?.palette, paletteTitle);
  if (palette.childElementCount) {
    const title = document.createElement("h5");
    title.textContent = paletteTitle;
    content.append(title, palette);
  }
  const match = firstComparison(appearance?.matches);
  if (match) {
    const title = document.createElement("h5");
    title.textContent = uiText("Сравнение кадров");
    if (match.fromLabel && match.toLabel && match.fromLabel !== match.toLabel) {
      title.textContent += ` · ${match.fromLabel} → ${match.toLabel}`;
    }
    if (Number.isInteger(match.fromTimeMs) && Number.isInteger(match.toTimeMs)) {
      title.textContent += ` · ${timecode(match.fromTimeMs)} → ${timecode(match.toTimeMs)}`;
    }
    const values = document.createElement("dl");
    values.className = "editor-object-comparison";
    for (const [label, value] of comparisonRows(match)) {
      const term = document.createElement("dt");
      const detail = document.createElement("dd");
      term.textContent = uiText(label);
      detail.textContent = value ?? uiText("Нет данных");
      values.append(term, detail);
    }
    content.append(title, values);
  }
  if (profile?.objectGallery) {
    const title = document.createElement("h5");
    title.textContent = uiText("Исходные ракурсы предмета");
    const note = document.createElement("p");
    note.className = "muted";
    note.textContent = uiText("Измеренные кадры и совпадающие детали; не 3D-развёртка.");
    content.append(title, note);
    const links = profile.objectReconciliation?.acceptedLinks;
    if (Array.isArray(links)) {
      const evidence = document.createElement("p");
      evidence.textContent = uiText`Подтверждённые связи между появлениями: ${links.length}`;
      content.append(evidence);
    }
  }
  return content;
}

/** Use the dominant measured color while leaving legacy observations on the layer default. */
export function objectAccent(palette) {
  const first = Array.isArray(palette) && palette.find(sample => /^#[0-9a-f]{6}$/i.test(sample?.hex));
  return first?.hex;
}

/** Summarize repeated labels for display while retaining per-track facts in the report. */
export function objectLabelSummary(labels) {
  const entries = new Map();
  for (const item of Array.isArray(labels) ? labels : []) {
    if (typeof item?.label !== "string") continue;
    if (!entries.has(item.label)) entries.set(item.label, {label: item.label, sampleCount: 0});
    const entry = entries.get(item.label);
    if (Number.isInteger(item.sampleCount) && item.sampleCount > 0) entry.sampleCount += item.sampleCount;
    if (Number.isFinite(item.score) && item.score >= 0 && item.score <= 1) {
      entry.score = Math.max(entry.score ?? 0, item.score);
    }
  }
  return [...entries.values()].sort((a, b) => (b.score ?? -1) - (a.score ?? -1));
}

/** Combine bounded measured gallery views with legacy profile views without duplicating observations. */
export function objectProfileViews(profile) {
  const views = new Map();
  for (const list of [profile?.objectGallery?.views, profile?.appearance?.views]) {
    for (const sample of Array.isArray(list) ? list : []) {
      if (!Number.isInteger(sample?.timeMs) || sample.timeMs < 0) continue;
      const key = sample.observationId ?? `${sample.timeMs}:${JSON.stringify(sample.box)}`;
      if (views.has(key) || views.size < 32) views.set(key, {...views.get(key), ...sample});
    }
  }
  return [...views.values()].sort((a, b) => a.timeMs - b.timeMs);
}

/** Return seekable observations for a profile. Legacy analyses supply their track clips. */
export function objectProfileAppearances(profile, tracks, duration) {
  const appearanceSamples = new Map();
  const views = objectProfileViews(profile);
  for (const sample of Array.isArray(views) ? views : []) {
    if (Number.isFinite(sample?.timeMs)) appearanceSamples.set(sample.timeMs, sample);
  }
  for (const clip of tracks) {
    const sample = clip.representative;
    if (Number.isFinite(sample?.timeMs)) appearanceSamples.set(sample.timeMs, sample);
  }
  const reported = Array.isArray(profile.appearances) ? profile.appearances : [];
  const appearances = [];
  for (const item of reported.length ? reported : tracks) {
    if (!item || !Number.isInteger(item.startMs) || !Number.isInteger(item.endMs) ||
        item.startMs < 0 || item.startMs >= duration || item.endMs <= item.startMs) continue;
    const startMs = item.startMs;
    const endMs = Math.min(item.endMs, duration);
    const candidate = item.representativeMs ?? item.representative?.timeMs;
    const timeMs = Number.isInteger(candidate) && candidate >= startMs && candidate < endMs ? candidate : startMs;
    const sample = item.representative?.timeMs === timeMs ? item.representative : appearanceSamples.get(timeMs);
    appearances.push({startMs, endMs, timeMs, sample, sampleCount: item.sampleCount});
  }
  return appearances.sort((a, b) => a.startMs - b.startMs || a.timeMs - b.timeMs);
}

/** Read a tiny version of the frozen frame once, then sample local element interiors. */
function frameColors(frame) {
  const tiny = document.createElement("canvas");
  tiny.width = 96;
  tiny.height = 96;
  const ctx = tiny.getContext("2d", {willReadFrequently: true});
  ctx.drawImage(frame, 0, 0, 96, 96);
  try {
    const pixels = ctx.getImageData(0, 0, 96, 96).data;
    return ([x, y, w, h]) => {
      const channels = [[], [], []];
      for (const dy of [.24, .5, .76]) {
        for (const dx of [.24, .5, .76]) {
          const col = Math.max(0, Math.min(95, Math.floor(96 * (x + dx * w))));
          const row = Math.max(0, Math.min(95, Math.floor(96 * (y + dy * h))));
          channels.forEach((values, i) => values.push(pixels[(row * 96 + col) * 4 + i]));
        }
      }
      return `rgb(${channels.map(values => values.sort((a, b) => a - b)[4]).join(", ")})`;
    };
  } catch {
    return () => "#99e198";
  }
}

/** Add a contrasting edge around pixel-derived colors so pale or dark samples remain readable. */
function traceElement(ctx, box, color, weight = 1.5) {
  const rect = box.map((value, index) => value * (index % 2 ? ctx.canvas.height : ctx.canvas.width));
  ctx.strokeStyle = "#15110e";
  ctx.lineWidth = weight + 2;
  ctx.strokeRect(...rect);
  ctx.strokeStyle = color;
  ctx.lineWidth = weight;
  ctx.strokeRect(...rect);
}

/** Freeze a single selected frame and use the shared monitor drawing for its technical view. */
export function mountObjectFrameDialog(find, paintSnapshot) {
  const dialog = find("media-object-dialog");
  const trigger = find("media-object-open");
  const title = find("media-object-dialog-title");
  const time = find("media-object-dialog-time");
  const toggle = find("media-object-technical");
  const canvas = find("media-object-detail-frame");
  const source = document.createElement("canvas");
  let selection;

  /** Repaint only on open or when the technical checkbox changes. */
  function draw() {
    if (!selection) return;
    canvas.width = source.width;
    canvas.height = source.height;
    const ctx = canvas.getContext("2d");
    ctx.drawImage(source, 0, 0);
    if (!toggle.checked) return;
    const {sample, sampleColor} = selection;
    ctx.save();
    const interfaces = paintSnapshot?.(ctx, canvas.width, canvas.height, sample.timeMs) ?? [];
    ctx.restore();
    for (const region of interfaces) {
      for (const element of region.elements?.slice(0, 20) ?? []) {
        if (element.role === "text_line" || !Array.isArray(element.box) || element.box.length !== 4 ||
            !element.box.every(Number.isFinite)) continue;
        traceElement(ctx, element.box, sampleColor(element.box));
      }
    }
    traceElement(ctx, sample.box, objectAccent(sample.palette) ?? sampleColor(sample.box), 3);
  }

  toggle.addEventListener("change", draw);
  trigger.addEventListener("click", () => {
    if (!selection) return;
    toggle.checked = true;
    draw();
    dialog.showModal();
  });
  dialog.addEventListener("close", () => {
    if (!trigger.hidden && trigger.isConnected) trigger.focus({preventScroll: true});
  });

  /** Keep accessible naming aligned with the current object profile. */
  function setLabel(label) {
    title.textContent = label;
    canvas.setAttribute("aria-label", uiText`Предмет в кадре: ${label}`);
  }

  return {
    setLabel,
    /** Release the previous capture when the user selects another source sample. */
    clear() {
      if (dialog.open) dialog.close();
      selection = undefined;
      source.width = source.height = 1;
      canvas.width = canvas.height = 1;
    },
    /** Save decoded pixels independently of subsequent playhead movements. */
    capture(video, sample, label, stamp) {
      const scale = Math.min(1, 1440 / Math.max(video.videoWidth, video.videoHeight));
      source.width = Math.max(1, Math.round(video.videoWidth * scale));
      source.height = Math.max(1, Math.round(video.videoHeight * scale));
      dialog.style.setProperty("--object-frame-width", `${65 * source.width / source.height}dvh`);
      source.getContext("2d").drawImage(video, 0, 0, source.width, source.height);
      selection = {sample, sampleColor: frameColors(source)};
      setLabel(label);
      time.textContent = stamp;
    },
  };
}

/** Only load same-origin URLs generated for the private scroll cache. */
export function scrollPreviewUrl(value) {
  return typeof value === "string" && /^\/api\/scene-scroll-preview\/[a-f0-9]{32}\/[0-9]{1,2}\.png$/.test(value)
    ? value : null;
}
