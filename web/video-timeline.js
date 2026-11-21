import {mountExpressionFramePreview} from "./expression-frame-preview.js";
import {expressionLanes, expressionGroupKey, unboundExpressionGroups} from "./facial-expressions.js";
import {gazeIcon, gazeLabel} from "./speaker-gaze.js";
import {uiLocale, uiText} from "./i18n.js";
import {effectiveSpeaker, mixedSpeakerLabel} from "./speaker-label.js";
import {
  mountObjectFrameDialog, objectAccent, objectAppearance, objectLabelSummary, objectProfileAppearances, objectProfileViews, scrollPreviewUrl,
} from "./video-object-profile.js";
import {mountObjectAppearanceTimeline} from "./video-object-appearance-timeline.js";
import {attachStudioIcon} from "./studio-icons.js";

const LEVELS = [.035, .065, .11, .19, .32];

/** Fit the entire source duration into the visible timeline track, retaining its leading ruler margin. */
export function fitTimelineScale(durationMs, availableWidth) {
  if (!Number.isFinite(durationMs) || durationMs <= 0 || !Number.isFinite(availableWidth) || availableWidth <= 0) return .11;
  return Math.max(.00001, Math.min(.64, availableWidth * .76 / durationMs));
}

/** Index overlapping intervals and return their clips in display order on the half-open timeline. */
export function timelineIntervalLookup(ranges) {
  const sorted = ranges.map(([start, end, item], order) => ({start: Number(start), end: Number(end), item, order}))
    .filter(({start, end}) => start < end)
    .sort((left, right) => left.start - right.start);
  const size = 2 ** Math.ceil(Math.log2(Math.max(1, sorted.length)));
  const maxEnd = new Float64Array(size * 2).fill(-Infinity);
  sorted.forEach(({end}, index) => { maxEnd[size + index] = end; });
  for (let index = size - 1; index > 0; index--) {
    maxEnd[index] = Math.max(maxEnd[2 * index], maxEnd[2 * index + 1]);
  }
  return time => {
    const matches = [];
    function visit(index, start, length) {
      if (maxEnd[index] <= time || sorted[start].start > time) return;
      if (length === 1) {
        matches.push(sorted[start]);
        return;
      }
      const half = length / 2;
      visit(index * 2, start, half);
      visit(index * 2 + 1, start + half, half);
    }
    if (sorted.length && Number.isFinite(time)) visit(1, 0, size);
    return matches.sort((left, right) => left.order - right.order).map(({item}) => item);
  };
}

/** Find the center of the visible part of a marker, in coordinates inside its one-pixel border. */
export function timelineMarkerCenter(start, end, left, right) {
  const visibleStart = Math.max(start, left);
  const visibleEnd = Math.min(end, right);
  if (visibleStart >= visibleEnd || (visibleStart === start && visibleEnd === end)) return null;
  return (visibleStart + visibleEnd) / 2 - start - 1;
}

const TICKS = [250, 500, 1000, 2000, 5000, 10000, 15000, 30000, 60000, 120000];
const MOTION_LABELS = {
  left: ["←", "Влево"], right: ["→", "Вправо"], up: ["↑", "Вверх"], down: ["↓", "Вниз"],
  in: ["↗", "Приближение камеры"], out: ["↙", "Отдаление камеры"],
};
const DEPTH_LABELS = {foreground: "Передний план", midground: "Средний план", background: "Задний план"};
const DEPTH_MARKERS = {foreground: "●", midground: "◐", background: "○"};
const DEPTH_BASIS = {
  "relative-face-scale": "Оценка по размеру лица",
  "relative-face-scale-body-position": "Оценка по размеру лица и положению фигуры",
};

function timecode(ms, compact = false) {
  const value = Math.max(0, Math.floor(ms));
  const seconds = Math.floor(value / 1000);
  const clock = `${String(Math.floor(seconds / 60)).padStart(2, "0")}:${String(seconds % 60).padStart(2, "0")}`;
  return compact ? clock : `${clock}.${String(Math.floor(value % 1000 / 10)).padStart(2, "0")}`;
}

/** Use the measured scene boundaries, retaining matching thumbnails from the face timeline. */
function timelineShots(data, media) {
  const native = media?.shots ?? [];
  const colors = (Array.isArray(data.shotColors) ? data.shotColors : []).filter(shot => shot && Number.isInteger(shot.startMs) &&
    /^#[0-9a-f]{6}$/i.test(shot.average));
  if (!colors.length) return native;
  let index = 0;
  return colors.sort((a, b) => a.startMs - b.startMs).map(shot => {
    while (native[index + 1]?.startMs <= shot.startMs) index++;
    const preview = [native[index], native[index + 1]].find(frame =>
      frame?.jpeg && Math.abs(frame.startMs - shot.startMs) <= 40);
    return {...shot, jpeg: preview?.jpeg};
  });
}

/** Keep measured clips on separate rows and display the identity shared by their profile. */
function objectLanes(data, duration) {
  const frames = data.sceneEvidence?.frames ?? [];
  const profiles = new Map((data.objectProfiles ?? []).map(profile => [profile.id, profile]));
  const clips = data.objectTimeline ?? frames.flatMap(frame => (frame.objects ?? []).map(object => ({
    startMs: frame.timeMs, endMs: frame.timeMs + 1, label: object.label, sampleCount: 1,
    representative: {timeMs: frame.timeMs, box: object.box, score: object.score},
  })));
  const lanes = [];
  const ordered = clips.filter(clip => clip.startMs >= 0 && clip.startMs < duration &&
    clip.endMs > clip.startMs && typeof clip.label === "string")
    .sort((a, b) => a.startMs - b.startMs);
  const legacyFrames = ordered.some(clip => !clip.representative) ?
    new Map(frames.map(frame => [frame.timeMs, frame.objects])) : null;
  const byProfile = new Map();
  for (const clip of ordered) {
    let row = lanes.find(lane => lane.right <= clip.startMs);
    if (!row) {
      const index = lanes.length;
      const label = index ? uiText`Предметы · ${index + 1}` : uiText("Предметы");
      row = {id: `objects-${index}`, kind: "objects", label, clips: [], right: 0};
      lanes.push(row);
    }
    const endMs = Math.min(duration, clip.endMs);
    const count = uiText`Замеров: ${clip.sampleCount}`;
    const representative = clip.representative ?? legacyObjectSample(clip, legacyFrames);
    const profile = profiles.get(clip.profileId);
    const screen = representative?.interfaceEvidence ?? profile?.interfaceEvidence;
    const label = screen?.displayLabel ?? profile?.label ?? clip.label;
    let description = label === clip.label ? count : `${count} · ${uiText`Метка кадра: ${clip.label}`}`;
    if (screen) description += ` · ${screen.type}`;
    const item = {...clip, endMs, text: uiText(label === "monitor" ? "Монитор" : label), description, representative, profile, screen};
    row.clips.push(item);
    if (profile) {
      if (!byProfile.has(profile)) byProfile.set(profile, []);
      byProfile.get(profile).push(item);
    }
    row.right = Math.max(endMs, clip.startMs + 10 / LEVELS[0]);
  }
  for (const [profile, tracks] of byProfile) {
    const appearances = objectProfileAppearances(profile, tracks, duration);
    for (const clip of tracks) clip.appearances = appearances;
  }
  return lanes;
}

/** Older analyses identify the first exact observation in the track ID. */
function legacyObjectSample(clip, frames) {
  const match = /^object_track:(\d+):(\d+):/.exec(clip.id);
  if (!match || Number(match[1]) !== clip.startMs) return undefined;
  const object = frames?.get(clip.startMs)?.[Number(match[2])];
  if (object?.label === clip.label) return {timeMs: clip.startMs, box: object.box, score: object.score};
}

const AUDIO_LANES = {
  music: "Музыка",
  intonation: "Интонация",
  background: "Фоновые звуки",
};
const AUDIO_LABELS = {
  music: "Музыка", rising: "Восходящая", falling: "Нисходящая", steady: "Ровная",
  "rise-fall": "Подъём и спад", "fall-rise": "Спад и подъём",
  applause: "Аплодисменты", laughter: "Смех", crowd: "Толпа", cricket_chirp: "Сверчок",
  bird: "Птицы", dog_bark: "Лай", rain: "Дождь", wind: "Ветер", traffic_noise: "Дорожный шум",
  door_slam: "Хлопок двери", knock: "Стук", typing_computer_keyboard: "Клавиатура",
  glass_breaking: "Звон разбитого стекла", siren: "Сирена", cough: "Кашель",
};
const PITCH_ICONS = {rising: "↗", falling: "↘", steady: "→", "rise-fall": "↗↘", "fall-rise": "↘↗"};

/** Build one independently switchable track per measurement type. */
function audioLanes(rows, duration) {
  if (!Array.isArray(rows)) return [];
  const groups = new Map();
  const percent = new Intl.NumberFormat(uiLocale, {style: "percent", maximumFractionDigits: 0});
  for (const row of rows) {
    if (!row || !Object.hasOwn(AUDIO_LANES, row.kind) || !Object.hasOwn(AUDIO_LABELS, row.label)
        || !Number.isInteger(row.startMs) || !Number.isInteger(row.endMs)
        || row.startMs < 0 || row.endMs > duration || row.startMs >= row.endMs) continue;
    if ((row.kind === "music") !== (row.label === "music")
        || (row.kind === "intonation") !== Object.hasOwn(PITCH_ICONS, row.label)) continue;
    const graph = Array.isArray(row.pitchHz) && row.pitchHz.length === 3 && row.pitchHz.every(Number.isFinite);
    const description = graph ? uiText`Высота: ${row.pitchHz.map(Math.round).join(" → ")} Гц`
      : Number.isFinite(row.confidence) ? uiText`Оценка модели: ${percent.format(row.confidence)}` : "";
    const text = uiText(AUDIO_LABELS[row.label]);
    const marker = row.kind === "intonation" ? PITCH_ICONS[row.label] ?? text : text;
    if (!groups.has(row.kind)) groups.set(row.kind, []);
    groups.get(row.kind).push({...row, text, marker, description, cutMs: Math.floor((row.startMs + row.endMs) / 2)});
  }
  return Object.entries(AUDIO_LANES).filter(([kind]) => groups.has(kind)).map(([kind, label]) => ({
    id: kind, kind, label: uiText(label), clips: groups.get(kind).sort((a, b) => a.startMs - b.startMs),
  }));
}

/** Keep camera estimates and apparent character motion apart, using exact sampled spans. */
function motionLanes(scene, editor, faces, duration) {
  const isTimed = clip => clip && Number.isInteger(clip.startMs) && Number.isInteger(clip.endMs)
    && 0 <= clip.startMs && clip.startMs < clip.endMs && clip.endMs <= duration;
  const describe = (row, camera) => {
    const directions = [row.horizontal, row.vertical, camera ? row.zoom : null]
      .filter(key => Object.hasOwn(MOTION_LABELS, key)).map(key => MOTION_LABELS[key]);
    const knownStationary = row.horizontal === "still" && row.vertical === "still"
      && (!camera || !row.zoom || row.zoom === "still");
    const depth = Object.hasOwn(DEPTH_LABELS, row.depth) ? uiText(DEPTH_LABELS[row.depth]) : "";
    const direction = directions.map(([, label]) => uiText(label)).join(" · ");
    const motion = direction || uiText(knownStationary ? "Без заметного движения" : "Движение не определено");
    const text = [motion, depth].filter(Boolean).join(" · ");
    const marker = directions.map(([arrow]) => arrow).join("") || (knownStationary ? "·" : "");
    const basis = camera ? "" : row.basis === "camera-compensated"
      ? uiText("С учётом движения камеры") : row.basis === "screen-space" ? uiText("Движение в кадре") : "";
    const depthMarker = depth ? DEPTH_MARKERS[row.depth] : "";
    const depthBasis = depth && Object.hasOwn(DEPTH_BASIS, row.depthBasis) ? uiText(DEPTH_BASIS[row.depthBasis]) : "";
    return {...row, text, marker: [marker, depthMarker].filter(Boolean).join(" ") || "…",
      description: [basis, depthBasis].filter(Boolean).join(" · ")};
  };
  const cameras = (Array.isArray(scene?.cameraMotion) ? scene.cameraMotion : []).filter(isTimed);
  const camera = cameras.length ? {
    id: "camera-motion", kind: "camera-motion", label: uiText("Движение камеры"),
    clips: cameras.map(row => describe(row, true)),
  } : null;
  const characters = new Map();
  const bySpeaker = new Map();
  for (const row of Array.isArray(editor?.personMotion) ? editor.personMotion : []) {
    if (!isTimed(row) || !Number.isInteger(row.faceId) || row.faceId < 1) continue;
    const binding = faces[row.faceId];
    const number = binding?.basis !== "visible-only" ? binding?.speaker : undefined;
    if (!characters.has(row.faceId)) {
      const lane = {id: `person-motion-${row.faceId}`, kind: "person-motion", faceId: row.faceId,
        label: uiText`Движение персонажа · лицо ${row.faceId}`, clips: []};
      characters.set(row.faceId, lane);
      if (Number.isInteger(number)) {
        lane.number = number;
        if (!bySpeaker.has(number)) bySpeaker.set(number, []);
        bySpeaker.get(number).push(lane);
      }
    }
    characters.get(row.faceId).clips.push(describe(row, false));
  }
  return {camera, bySpeaker, unbound: [...characters.values()].filter(lane => lane.number === undefined)};
}

/** Build numbered lanes, keeping unattributed mixed speech off individual speaker tracks. */
export function buildTimeline(data, evidence, gazes = new Map()) {
  const speechEnd = data.segments.reduce((end, cue) => Math.max(end, cue.endMs), 0);
  const sceneEnd = data.sceneEvidence?.frames?.reduce((end, frame) => Math.max(end, frame.timeMs + 1), 0) ?? 0;
  const duration = Math.max(data.durationMs ?? 0, evidence?.editorTimeline?.durationMs ?? 0, speechEnd, sceneEnd);
  const faces = evidence?.faceSpeakers ?? data.videoEvidence?.faceSpeakers ?? {};
  const media = evidence?.editorTimeline ?? data.videoEvidence?.editorTimeline;
  const profiles = new Map((data.speakerProfiles ?? []).map(profile => [profile.number, profile]));
  const boundSpeakers = new Set();
  const lanes = new Map();
  function speakerLane(number) {
    const id = `speaker-${number}`;
    if (!lanes.has(id)) lanes.set(id, {id, number,
      label: profiles.get(number)?.name || uiText`Спикер ${number}`, kind: "speaker", clips: []});
    return lanes.get(id);
  }
  for (const [face, {speaker, basis}] of Object.entries(faces)) {
    if (basis === "visible-only") continue;
    const lane = speakerLane(speaker);
    const jpeg = media?.facePhotos?.[face];
    boundSpeakers.add(speaker);
    if (jpeg && (profiles.get(speaker)?.face?.id === Number(face) || !lane.photo)) lane.photo = jpeg;
  }
  const extras = [];
  const shots = timelineShots(data, media).filter(shot => 0 <= shot.startMs && shot.startMs < duration);
  if (shots.length) {
    extras.push({id: "shots", label: uiText("Планы"), kind: "shots", clips: shots.map((shot, index) => ({
      ...shot, endMs: shots[index + 1]?.startMs ?? (shot.average ? duration : media?.durationMs || duration),
      text: uiText`План ${index + 1}`,
    })).filter(shot => shot.endMs > shot.startMs)});
  }
  const pauses = (data.dialoguePauses ?? []).map(pause => {
    const length = uiText`${((pause.endMs - pause.startMs) / 1000).toFixed(2)} с`;
    const location = pause.position === "between-cues" ? uiText("Между репликами") : uiText("У границы или внутри реплики");
    const basis = pause.basis === "speaker-gap" ? uiText("Голос не обнаружен") : uiText("Тихий звук");
    return {...pause, length, text: `${length} · ${basis} · ${location} · ${uiText("Точка проверки")}: ${timecode(pause.cutMs)}`};
  });
  if (pauses.length) extras.push({id: "pauses", label: uiText("Паузы"), kind: "pause", clips: pauses});
  let mixed;
  let unknown;
  for (const cue of data.segments) {
    const number = effectiveSpeaker(cue);
    let lane;
    if (mixedSpeakerLabel(cue)) {
      mixed ||= {id: "overlap", label: uiText("Перебивания"), kind: "overlap", clips: []};
      if (cue.overlaps?.length) {
        for (const {startMs, endMs} of cue.overlaps) mixed.clips.push({...cue, startMs, endMs});
      } else {
        mixed.clips.push(cue);
      }
      continue;
    } else if (number === null) {
      unknown ||= {id: "unassigned", label: uiText("Без спикера"), kind: "unknown", clips: []};
      lane = unknown;
    } else {
      lane = speakerLane(number);
      if (!boundSpeakers.has(number) && cue.speakerPhoto?.speaker === number) lane.photo ||= cue.speakerPhoto.jpeg;
    }
    if (lane.kind === "speaker" && cue.speakerName) lane.label = cue.speakerName;
    lane.clips.push(cue);
  }
  const result = extras.filter(lane => lane.kind === "shots");
  const motion = motionLanes(data.sceneEvidence, media, faces, duration);
  const expressions = expressionLanes(evidence ?? data.videoEvidence, duration);
  if (motion.camera) result.push(motion.camera);
  for (const lane of [...lanes.values()].sort((a, b) => a.number - b.number)) {
    result.push(lane);
    const gaze = gazes.get(lane.number);
    if (gaze?.intervals.length) {
      result.push({id: `gaze-${lane.number}`, number: lane.number,
        label: uiText`Взгляд · ${lane.label}`, kind: "gaze",
        clips: gaze.intervals.map(interval => ({...interval, text: gazeLabel(interval.direction)}))});
    }
    for (const track of expressions.filter(track => track.number === lane.number)) {
      track.label = uiText`Выражения · ${lane.label} · лицо ${track.faceId}`;
      result.push(track);
    }
    for (const track of motion.bySpeaker.get(lane.number) ?? []) {
      track.label = uiText`Движение персонажа · ${lane.label}`;
      result.push(track);
    }
  }
  if (mixed) result.push(mixed);
  if (unknown) result.push(unknown);
  const unboundFaces = Object.keys(media?.facePhotos ?? {}).map(Number).filter(id =>
    Number.isInteger(id) && id > 0 && (!faces[id] || faces[id].basis === "visible-only"));
  result.push(...unboundExpressionGroups([...expressions, ...motion.unbound], media?.facePhotos, unboundFaces), ...extras.filter(lane => lane.kind !== "shots"),
    ...audioLanes(data.audioLayers, duration), ...objectLanes(data, duration));
  return {duration, lanes: result};
}

function image(jpeg) {
  if (typeof jpeg !== "string" || !/^[A-Za-z0-9+/]+={0,2}$/.test(jpeg)) return null;
  const img = document.createElement("img");
  img.src = "data:image/jpeg;base64," + jpeg;
  img.alt = "";
  img.loading = "lazy";
  img.decoding = "async";
  return img;
}

const ICON_PATHS = {
  music: "M9 18V5l11-2v13M9 8l11-2M9 18a3 2 0 1 1-6 0 3 2 0 0 1 6 0Zm11-2a3 2 0 1 1-6 0 3 2 0 0 1 6 0Z",
  intonation: "M3 15c3 0 3-8 6-8s3 10 6 10 3-5 6-5M3 21h18",
  background: "M3 9v6m4-9v12m5-15v18m5-15v12m4-9v6",
  objects: "m12 3 9 5v8l-9 5-9-5V8l9-5Zm-9 5 9 5 9-5m-9 5v8",
  fit: "M8 3H3v5m13-5h5v5M3 16v5h5m13-5v5h-5M8 12h8m-6-2-2 2 2 2m4-4 2 2-2 2",
  density: "M3 5h18M3 9h18M3 13h18M3 17h18M3 21h18",
  more: "M12 5h.01M12 12h.01M12 19h.01",
  expression: "M8 9h.01M16 9h.01M8 15h8M21 12a9 9 0 1 1-18 0 9 9 0 0 1 18 0Z",
  motion: "M5 6h3m-4 4h4m7-6a2 2 0 1 0 0 4 2 2 0 0 0 0-4Zm-7 10 4-4 3 2 4 .5m-9 7 3-4 4 4h3",
  gaze: "M11 5.75C6.8 5.75 3.55 8.1 1.75 12c1.8 3.9 5.05 6.25 9.25 6.25 2.65 0 5-1 6.8-2.7M11 5.75c2.65 0 5 1 6.8 2.7M11 9a3 3 0 1 0 0 6 3 3 0 0 0 0-6Zm0 3h11m-3.5-3.5L22 12l-3.5 3.5",
  zoomOut: "M15.25 15.25 21 21M6.75 10.5h7.5m3.25 0a7 7 0 1 1-14 0 7 7 0 0 1 14 0Z",
  zoomIn: "M15.25 15.25 21 21M6.75 10.5h7.5m-3.75-3.75v7.5m7-3.75a7 7 0 1 1-14 0 7 7 0 0 1 14 0Z",
};

/** Give a timeline key a labeled 24-unit icon and a hint on hover or keyboard focus. */
function installTimelineIcon(button, icon, label) {
  const ns = "http://www.w3.org/2000/svg";
  const svg = document.createElementNS(ns, "svg");
  svg.setAttribute("viewBox", "0 0 24 24");
  svg.setAttribute("aria-hidden", "true");
  svg.setAttribute("focusable", "false");
  const path = document.createElementNS(ns, "path");
  path.setAttribute("d", ICON_PATHS[icon]);
  svg.append(path);
  button.classList.add("editor-icon");
  button.setAttribute("aria-label", label);
  button.title = label;
  button.dataset.hint = label;
  button.replaceChildren(svg);
  button.addEventListener("keydown", event => {
    if (event.key !== "Escape") return;
    button.dataset.hintDismissed = "";
  });
  for (const type of ["focusin", "pointerenter"]) {
    button.addEventListener(type, () => { delete button.dataset.hintDismissed; });
  }
}

/** Retain the native checkbox and its change handler inside an icon-sized label. */
function installTimelineLayerIcon(control, toggle, icon, label) {
  installTimelineIcon(control, icon, label);
  toggle.setAttribute("aria-label", label);
  control.prepend(toggle);
}

/** Keep a speaker's actions above the scrollable timeline and inside the viewport. */
function speakerActions(lane, actions) {
  const owner = lane.number ?? `face-${lane.faceId}`;
  const trigger = document.createElement("button");
  trigger.type = "button";
  trigger.className = "editor-speaker-more";
  trigger.dataset.timelineControl = `more-${owner}`;
  installTimelineIcon(trigger, "more", uiText`Действия · ${lane.label}`);

  const popover = document.createElement("div");
  popover.className = "editor-speaker-menu";
  popover.popover = "auto";
  trigger.popoverTargetElement = popover;
  for (const [index, config] of actions.entries()) {
    const action = document.createElement("button");
    action.type = "button";
    action.autofocus = index === 0;
    action.dataset.timelineControl = `${config.icon}-${owner}`;
    action.className = "editor-speaker-motion";
    installTimelineIcon(action, config.icon, config.label);
    action.append(config.label);
    delete action.dataset.hint;
    action.setAttribute("aria-controls", config.ids.join(" "));
    action.setAttribute("aria-pressed", String(config.visible));
    action.onclick = () => {
      const show = action.getAttribute("aria-pressed") !== "true";
      action.setAttribute("aria-pressed", String(show));
      config.toggle(show);
    };
    if (config.bind) config.bind(action);
    popover.append(action);
  }
  popover.addEventListener("beforetoggle", event => {
    if (event.newState !== "open") return;
    const {left, top, bottom} = trigger.getBoundingClientRect();
    const placeAbove = window.innerHeight - bottom < 112 && top > 112;
    popover.style.left = `${Math.max(8, Math.min(left, window.innerWidth - 236))}px`;
    popover.style.top = placeAbove ? "auto" : `${bottom + 4}px`;
    popover.style.bottom = placeAbove ? `${window.innerHeight - top + 4}px` : "auto";
  });
  return [trigger, popover];
}

/** Draw a bounded source-frame crop with a small margin around the measured box. */
function drawObjectCrop(canvas, video, box, accent) {
  const [x, y, width, height] = box;
  const sourceWidth = video.videoWidth;
  const sourceHeight = video.videoHeight;
  const left = Math.max(0, x - width * .16);
  const top = Math.max(0, y - height * .16);
  const right = Math.min(1, x + width * 1.16);
  const bottom = Math.min(1, y + height * 1.16);
  const cropWidth = (right - left) * sourceWidth;
  const cropHeight = (bottom - top) * sourceHeight;
  const scale = Math.min(480 / cropWidth, 320 / cropHeight);
  canvas.width = Math.max(1, Math.round(cropWidth * scale));
  canvas.height = Math.max(1, Math.round(cropHeight * scale));
  const context = canvas.getContext("2d");
  if (!context) return false;
  context.drawImage(video, left * sourceWidth, top * sourceHeight, cropWidth, cropHeight,
    0, 0, canvas.width, canvas.height);
  const rect = [(x - left) * sourceWidth * scale, (y - top) * sourceHeight * scale,
    width * sourceWidth * scale, height * sourceHeight * scale];
  context.strokeStyle = "#18130d";
  context.lineWidth = 5;
  context.strokeRect(...rect);
  context.strokeStyle = accent ?? "#ffd398";
  context.lineWidth = 2;
  context.strokeRect(...rect);
  return true;
}

/** Reject coordinates that cannot provide a crop inside a source frame. */
function hasObjectBox(box) {
  return Array.isArray(box) && box.length === 4 && box.every(Number.isFinite) &&
    box[0] >= 0 && box[1] >= 0 && box[0] < 1 && box[1] < 1 && box[2] > 0 && box[3] > 0;
}

/** Keep the crop pinned to the clicked observation while playback continues. */
function mountObjectPreview(find, video, seek, paintSnapshot, highlightProfile) {
  const panel = find("media-object-preview");
  const canvas = find("media-object-image");
  const openPhoto = find("media-object-open");
  const screenBadge = find("media-object-interface");
  const photoDialog = mountObjectFrameDialog(find, paintSnapshot);
  const status = find("media-object-image-status");
  const name = find("media-object-name");
  const time = find("media-object-time");
  const score = find("media-object-score");
  const frameLabel = find("media-object-frame-label");
  const profile = find("media-object-profile");
  const labels = find("media-object-profile-labels");
  const appearance = find("media-object-appearance");
  const views = find("media-object-views");
  const viewsTitle = find("media-object-views-title");
  const extraViews = find("media-object-extra-views");
  const extraViewButtons = find("media-object-extra-view-buttons");
  const scroll = find("media-object-scroll");
  const scrollPages = find("media-object-scroll-pages");
  const textBlock = find("media-object-text");
  const textLines = find("media-object-text-lines");
  const percent = new Intl.NumberFormat(uiLocale, {style: "percent", maximumFractionDigits: 1});
  let selectedButton;
  let selectedSample;
  let selectedLabel;
  let renderedProfile;
  let renderedAppearances;
  let pending;
  const appearanceTimeline = mountObjectAppearanceTimeline(find, timecode, openView);

  function displayScore(value) {
    return Number.isFinite(value) && 0 <= value && value <= 1 ? percent.format(value) : uiText("Нет данных");
  }

  /** Show detector labels and visual evidence, retaining the user's disclosure on refresh. */
  function renderProfile(data, appearances, preserveDisclosure = false) {
    if (preserveDisclosure && renderedProfile === data && renderedAppearances === appearances) return;
    renderedProfile = data;
    renderedAppearances = appearances;
    labels.replaceChildren();
    for (const entry of objectLabelSummary(data?.labels)) {
      const row = document.createElement("li");
      const label = document.createElement("span");
      label.textContent = entry.label;
      const evidence = document.createElement("span");
      const count = uiText`Замеров: ${entry.sampleCount}`;
      evidence.textContent = `${displayScore(entry.score)} · ${count}`;
      row.append(label, evidence);
      labels.append(row);
    }
    renderAppearance();
    renderScrollPreviews(data?.scrollPreviews, preserveDisclosure);
    renderViews(appearances, objectProfileViews(data));
    const visible = labels.childElementCount || appearance.childElementCount || !views.hidden;
    if (!preserveDisclosure || profile.hidden) {
      profile.open = labels.childElementCount > 1 || Boolean(appearance.childElementCount) || !views.hidden;
    }
    profile.hidden = !visible;
  }

  /** Lazy-load the original stitched pages when the reader opens the disclosure. */
  function renderScrollPreviews(pages, preserveDisclosure) {
    scrollPages.replaceChildren();
    for (const page of Array.isArray(pages) ? pages : []) {
      const url = scrollPreviewUrl(page?.previewUrl);
      if (!url || !Number.isFinite(page.startMs) || !Number.isFinite(page.endMs)) continue;
      const figure = document.createElement("figure");
      const image = document.createElement("img");
      image.loading = "lazy";
      image.dataset.preview = url;
      image.alt = uiText("Склейка прокрутки интерфейса");
      const caption = document.createElement("figcaption");
      const time = document.createElement("span");
      time.textContent = `${timecode(page.startMs)}–${timecode(page.endMs)}`;
      if (Number.isFinite(page.height) && page.viewportHeight > 0 && page.addedPixels > 0) {
        const screens = (page.height / page.viewportHeight).toFixed(1);
        time.textContent += uiText` · ${screens} экрана`;
      }
      const save = document.createElement("a");
      save.className = "studio-icon studio-quiet-action";
      save.dataset.icon = "download";
      save.setAttribute("data-hint", "");
      save.setAttribute("aria-label", uiText("Скачать склейку PNG"));
      save.href = url;
      save.download = `scroll-${Math.round(page.startMs)}-${Math.round(page.endMs)}.png`;
      attachStudioIcon(save);
      caption.append(time, save);
      const markdown = page.markdownAvailable === true ? document.createElement("a") : null;
      if (markdown) {
        markdown.className = "studio-icon studio-quiet-action";
        markdown.dataset.icon = "markdown";
        markdown.setAttribute("data-hint", "");
        markdown.setAttribute("aria-label", uiText("Скачать Markdown"));
        markdown.href = url.replace(/\.png$/, ".md");
        markdown.download = save.download.replace(/\.png$/, ".md");
        attachStudioIcon(markdown);
        caption.append(markdown);
      }
      image.addEventListener("error", () => {
        image.hidden = true;
        time.textContent = uiText("Временная склейка недоступна. Запусти анализ заново.");
        save.hidden = true;
        if (markdown) markdown.hidden = true;
      });
      figure.append(caption, image);
      scrollPages.append(figure);
    }
    scroll.hidden = !scrollPages.childElementCount;
    if (!preserveDisclosure || scroll.hidden) scroll.open = false;
    loadScrollImages();
  }

  /** Avoid downloading temporary media for cards the reader has only selected. */
  function loadScrollImages() {
    if (!scroll.open) return;
    for (const image of scrollPages.querySelectorAll("img[data-preview]")) {
      image.src = image.dataset.preview;
      delete image.dataset.preview;
    }
  }

  scroll.addEventListener("toggle", loadScrollImages);

  /** Show the explicit source frame's colors whenever a measured view is selected. */
  function renderAppearance() {
    const visual = renderedProfile?.appearance;
    const samples = objectProfileViews(renderedProfile);
    const selected = selectedSample?.observationId ?? selectedSample?.timeMs;
    const view = selected === undefined ? undefined :
      samples.find(sample => (sample?.observationId ?? sample?.timeMs) === selected);
    appearance.replaceChildren(objectAppearance(visual, selectedSample?.palette ?? view?.palette, timecode, renderedProfile));
  }

  /** Link measured intervals to representative source frames, using old views for legacy reports. */
  function renderViews(appearances, samples) {
    const previousFocus = views.contains(document.activeElement) && document.activeElement.dataset.viewId;
    const grouped = Boolean(appearances?.length);
    viewsTitle.textContent = uiText(grouped ? "Появления предмета" : "Кадры предмета");
    const frames = fallbackViews(samples);
    const representativeTimes = new Set(appearances?.map(item => item.timeMs));
    const details = grouped ? frames.filter(frame => !representativeTimes.has(frame.timeMs)) : [];
    const entries = grouped ? appearances : frames;
    appearanceTimeline.render(entries, selectedSample, previousFocus);
    renderViewList(details, previousFocus);
    extraViews.hidden = !details.length;
    views.hidden = !entries.length;
    updateViewButtons();
  }

  /** Open the measured frame, retaining the seek even when crop data is unavailable. */
  function openView({timeMs, sample = {timeMs}}) {
    selectSample(sample, sample.label || name.textContent, timeMs);
    renderAppearance();
    seek(timeMs);
    capture();
  }

  /** Keep secondary measured crops available under their own disclosure. */
  function renderViewList(entries, previousFocus) {
    extraViewButtons.replaceChildren();
    for (const entry of entries) {
      const {timeMs, sample} = entry;
      const row = document.createElement("li");
      const button = document.createElement("button");
      button.type = "button";
      button.textContent = timecode(timeMs);
      button.title = uiText`Кадр предмета ${button.textContent}`;
      if (sample.label) button.title += ` · ${sample.label}`;
      button.setAttribute("aria-label", button.title);
      button.dataset.viewId = String(sample.observationId ?? timeMs);
      button.onclick = () => openView(entry);
      row.append(button);
      extraViewButtons.append(row);
      if (button.dataset.viewId === previousFocus) button.focus({preventScroll: true});
    }
  }

  /** Mark the open crop even while the video playhead moves away from it. */
  function updateViewButtons() {
    appearanceTimeline.setSelection(selectedSample);
    const selected = selectedSample?.observationId ?? selectedSample?.timeMs;
    for (const button of extraViewButtons.querySelectorAll("[data-view-id]")) {
      button.setAttribute("aria-pressed", String(button.dataset.viewId === String(selected)));
    }
  }

  /** Normalize exact sample buttons when an imported profile lacks appearance intervals. */
  function fallbackViews(samples) {
    if (!Array.isArray(samples)) return [];
    return samples.filter(sample => Number.isFinite(sample?.timeMs) && sample.timeMs >= 0 && hasObjectBox(sample.box))
      .map(sample => ({timeMs: sample.timeMs, sample}));
  }

  /** Display the original recognized lettering as plain text. */
  function renderObjectText(data) {
    textLines.replaceChildren();
    for (const entry of data?.visibleText ?? []) {
      const row = document.createElement("li");
      row.textContent = entry.text;
      textLines.append(row);
    }
    textBlock.hidden = !textLines.childElementCount;
  }

  function clear() {
    if (selectedButton) selectedButton.setAttribute("aria-expanded", "false");
    highlightProfile();
    selectedButton = selectedSample = selectedLabel = pending = undefined;
    panel.hidden = true;
    photoDialog.clear();
  }

  function showUnavailable() {
    if (!pending) return;
    pending = undefined;
    status.textContent = uiText("Кадр недоступен");
  }

  function capture() {
    if (!pending || video.seeking || video.readyState < 2 || !video.videoWidth ||
        Math.abs(video.currentTime * 1000 - pending.timeMs) >= 8) return;
    const accent = objectAccent(pending.palette ?? renderedProfile?.appearance?.palette);
    if (!drawObjectCrop(canvas, video, pending.box, accent)) {
      showUnavailable();
      return;
    }
    photoDialog.capture(video, pending, name.textContent, time.textContent);
    pending = undefined;
    openPhoto.hidden = false;
    status.hidden = true;
  }

  /** Update the panel to the exact sample before the caller moves the playhead. */
  function selectSample(sample, label, fallbackTime = 0) {
    selectedSample = sample;
    selectedLabel = label;
    time.textContent = uiText`Кадр ${timecode(sample?.timeMs ?? fallbackTime)}`;
    score.textContent = displayScore(sample?.score);
    frameLabel.textContent = uiText`Метка кадра: ${label}`;
    frameLabel.hidden = name.textContent === label;
    openPhoto.hidden = true;
    photoDialog.clear();
    status.hidden = false;
    pending = hasObjectBox(sample?.box) && Number.isFinite(sample.timeMs) && !video.hidden && !video.error
      ? sample : undefined;
    status.textContent = uiText(pending ? "Подготавливаем кадр…" : "Кадр недоступен");
    updateViewButtons();
  }

  for (const event of ["seeked", "loadeddata"]) video.addEventListener(event, capture);
  video.addEventListener("error", showUnavailable);

  /** Refresh inferred names and kinds when new evidence reaches an open card. */
  function updateLabel(clip) {
    name.textContent = clip.text;
    screenBadge.hidden = !clip.screen;
    screenBadge.textContent = clip.screen?.type ?? "";
    score.closest("dl").hidden = clip.screen?.basis === "verified-panel";
    if (clip.screen) {
      screenBadge.dataset.kind = clip.screen.type;
      screenBadge.title = uiText("Тип по признакам интерфейса");
    }
    if (selectedLabel) frameLabel.hidden = name.textContent === selectedLabel;
    photoDialog.setLabel(clip.text);
  }

  return {
    clear, capture,
    selected: () => selectedButton,
    /** Reconnect the open preview to its rebuilt timeline button without seeking. */
    rebind(button, clip) {
      if (!button) {
        clear();
        return;
      }
      selectedButton = button;
      button.setAttribute("aria-expanded", "true");
      updateLabel(clip);
      highlightProfile(clip.profile?.id);
      renderProfile(clip.profile, clip.appearances, true);
      renderObjectText(clip.profile);
    },
    show(button, clip) {
      const alreadySelected = selectedButton === button;
      clear();
      if (alreadySelected) return;
      selectedButton = button;
      button.setAttribute("aria-expanded", "true");
      updateLabel(clip);
      highlightProfile(clip.profile?.id);
      selectSample(clip.representative, clip.label, clip.startMs);
      renderProfile(clip.profile, clip.appearances);
      renderObjectText(clip.profile);
      panel.hidden = false;
      panel.scrollIntoView({block: "nearest", inline: "nearest"});
    },
  };
}

/** Drive the preview from the scroll position and keep a fixed, labeled playhead. */
export function mountVideoTimeline(find, video, openProfile, paintSnapshot) {
  const root = find("media-editor-timeline");
  const scroll = find("media-editor-scroll");
  const sheet = find("media-editor-sheet");
  const clock = find("media-editor-clock");
  const selected = find("media-editor-selection");
  const zoomOut = find("media-editor-zoom-out");
  const zoomIn = find("media-editor-zoom-in");
  const actions = zoomIn.closest(".editor-actions");
  const fit = document.createElement("button");
  fit.id = "media-editor-zoom-fit";
  fit.type = "button";
  fit.className = "editor-view-button";
  installTimelineIcon(fit, "fit", uiText("Вписать шкалу"));
  fit.setAttribute("aria-pressed", "false");
  const zoomSummary = document.createElement("output");
  zoomSummary.className = "editor-zoom-summary";
  zoomSummary.dataset.agWidth = "content";
  zoomSummary.dataset.agWidthMinEm = "3";
  zoomSummary.setAttribute("aria-label", uiText("Масштаб шкалы"));
  actions.append(fit, zoomSummary);
  const density = document.createElement("button");
  density.id = "media-editor-density";
  density.type = "button";
  density.className = "editor-view-button";
  installTimelineIcon(density, "density", uiText("Плотные дорожки"));
  const densityKey = "voice-studio:editor-track-density";
  let compact = true;
  try { compact = localStorage.getItem(densityKey) !== "comfortable"; } catch {}
  const updateDensity = () => {
    root.dataset.density = compact ? "compact" : "comfortable";
    density.setAttribute("aria-pressed", String(compact));
  };
  density.onclick = () => {
    compact = !compact;
    try { localStorage.setItem(densityKey, compact ? "compact" : "comfortable"); } catch {}
    updateDensity();
  };
  updateDensity();
  actions.append(density);
  const allGazeButton = find("media-editor-gaze");
  const allExpressionButton = find("media-editor-expressions");
  const allMotionButton = find("media-editor-motion");
  const objectsControl = find("media-editor-objects-control");
  const objectsCheckbox = find("media-editor-objects");
  const objectPreview = mountObjectPreview(find, video, drive, paintSnapshot, highlightObjectProfile);
  const expressionPreview = mountExpressionFramePreview(video);
  let expressionEvidence;
  const objectButtonsByProfile = new Map();
  let highlightedObjects = [];
  let objectRows = [];
  const audioControls = new Map(Object.keys(AUDIO_LANES).map(kind => [kind, {
    control: find(`media-editor-${kind}-control`), toggle: find(`media-editor-${kind}`), row: null,
  }]));
  const audioControlLabels = {music: "Музыка", intonation: "Интонация", background: "Фоновые звуки"};
  for (const [kind, layer] of audioControls) {
    installTimelineLayerIcon(layer.control, layer.toggle, kind, uiText(audioControlLabels[kind]));
    layer.toggle.onchange = () => {
      if (layer.row) layer.row.hidden = !layer.toggle.checked;
      if (needle) paint(position, false);
    };
  }
  installTimelineLayerIcon(objectsControl, objectsCheckbox, "objects", uiText("Предметы"));
  objectsCheckbox.onchange = () => {
    for (const row of objectRows) row.hidden = !objectsCheckbox.checked;
    if (!objectsCheckbox.checked) objectPreview.clear();
    paint(position, false);
  };
  installTimelineIcon(allExpressionButton, "expression", uiText("Дорожки выражений лица"));
  installTimelineIcon(allGazeButton, "gaze", uiText("Дорожки взгляда"));
  installTimelineIcon(allMotionButton, "motion", uiText("Движение персонажа"));
  installTimelineIcon(zoomOut, "zoomOut", uiText("Уменьшить масштаб"));
  installTimelineIcon(zoomIn, "zoomIn", uiText("Увеличить масштаб"));
  const hiddenGaze = new Set();
  let showGaze = true;
  let gazeRows = new Map();
  let gazeButtons = new Map();
  const hiddenExpressions = new Set();
  let showExpressions = true;
  let expressionRows = new Map();
  let expressionButtons = new Map();
  let expressionHeadings = new Map();
  const hiddenMotion = new Set();
  let showMotion = true;
  let motionRows = new Map();
  let motionButtons = new Map();

  function refreshGaze() {
    allGazeButton.hidden = !gazeRows.size;
    allGazeButton.setAttribute("aria-pressed", String(showGaze));
    for (const [number, band] of gazeRows) {
      band.hidden = !showGaze || hiddenGaze.has(number);
      const button = gazeButtons.get(number);
      button.setAttribute("aria-pressed", String(!band.hidden));
      button.disabled = !showGaze;
    }
    if (needle) paint(position, false);
  }
  allGazeButton.onclick = () => { showGaze = !showGaze; refreshGaze(); };

  /** Build the same overflow action for a bound speaker or an explicitly unbound face group. */
  function expressionAction(key, ids) {
    return {icon: "expression", label: uiText("Выражения лица"), ids,
      visible: showExpressions && !hiddenExpressions.has(key),
      bind: button => expressionButtons.set(key, button),
      toggle: show => {
        if (show) hiddenExpressions.delete(key);
        else hiddenExpressions.add(key);
        refreshExpressions();
      }};
  }

  /**
   * Build a local motion action without losing its preference when globally hidden.
   * @param {number|string} key Speaker number or unbound face group key.
   * @param {string[]} ids IDs of this owner's measured motion rows.
   * @returns {object} Menu action bound to the shared local/global visibility state.
   */
  function motionAction(key, ids) {
    return {icon: "motion", label: uiText("Движение персонажа"), ids,
      visible: showMotion && !hiddenMotion.has(key),
      bind: button => motionButtons.set(key, button),
      toggle: show => {
        if (show) hiddenMotion.delete(key);
        else hiddenMotion.add(key);
        refreshMotion();
      }};
  }

  /**
   * Apply the global motion switch while retaining each owner's local choice.
   * Updates row visibility and accessible button state; camera tracks stay independent.
   */
  function refreshMotion() {
    allMotionButton.hidden = !motionRows.size;
    allMotionButton.setAttribute("aria-pressed", String(showMotion));
    const ids = [];
    for (const [key, rows] of motionRows) {
      const visible = showMotion && !hiddenMotion.has(key);
      for (const band of rows) {
        band.hidden = !visible;
        ids.push(band.id);
      }
      const button = motionButtons.get(key);
      if (button) {
        button.setAttribute("aria-pressed", String(visible));
        button.disabled = !showMotion;
      }
    }
    if (ids.length) allMotionButton.setAttribute("aria-controls", ids.join(" "));
    else allMotionButton.removeAttribute("aria-controls");
    if (needle) paint(position, false);
  }
  allMotionButton.onclick = () => { showMotion = !showMotion; refreshMotion(); };

  /** Share the global expression switch while preserving each speaker's local choice. */
  function refreshExpressions() {
    allExpressionButton.hidden = !expressionRows.size;
    allExpressionButton.setAttribute("aria-pressed", String(showExpressions));
    // Keep the profile entry point available when its measured subtracks are collapsed.
    for (const heading of expressionHeadings.values()) heading.hidden = false;
    for (const [key, rows] of expressionRows) {
      const visible = showExpressions && !hiddenExpressions.has(key);
      for (const band of rows) band.hidden = !visible;
      const button = expressionButtons.get(key);
      if (button) {
        button.setAttribute("aria-pressed", String(visible));
        button.disabled = !showExpressions;
      }
    }
    if (needle) paint(position, false);
  }
  allExpressionButton.onclick = () => { showExpressions = !showExpressions; refreshExpressions(); };

  let duration = 0;
  let zoomScale = LEVELS[2];
  let fitMode = false;
  let fittedScale = LEVELS[2];
  let position = 0;
  let lead = 0;
  let labelWidth = 0;
  let trackWidth = 0;
  let expectedScroll = 0;
  let animation = 0;
  let scrollSeek = 0;
  let queuedSeek = null;
  let clips = [];
  let clipsAt = timelineIntervalLookup([]);
  let markersAt = timelineIntervalLookup([]);
  let markerScroll = -1;
  let edgeMarkers = new Set();
  let active = new Set();
  let selectedItem;
  let ruler;
  let head;
  let needle;

  const scale = () => fitMode ? fittedScale : zoomScale;
  const playable = () => !video.hidden && video.readyState > 0;

  /** Move only marker labels that are clipped by an edge of the track viewport. */
  function centerVisibleMarkers(offset) {
    if (markerScroll === offset || !trackWidth) return;
    markerScroll = offset;
    const right = offset + trackWidth;
    const candidates = new Set([...markersAt(offset), ...markersAt(right)]);
    const adjusted = new Set();
    for (const item of candidates) {
      const center = timelineMarkerCenter(item.start, item.end, offset, right);
      if (center === null) continue;
      const left = `${center}px`;
      if (item.marker.style.left !== left) item.marker.style.left = left;
      adjusted.add(item);
    }
    for (const item of edgeMarkers) {
      if (!adjusted.has(item)) item.marker.style.removeProperty("left");
    }
    edgeMarkers = adjusted;
  }

  /** Keep the cursor aligned with the native scroll position. */
  function alignViewport(offset) {
    const left = `${labelWidth + lead + (fitMode ? position * scale() : offset)}px`;
    if (needle.style.left !== left) needle.style.left = left;
    centerVisibleMarkers(offset);
  }

  /** Move the cursor and update just the clips entering or leaving the playhead. */
  function paint(ms, follow) {
    position = Math.min(Math.max(0, ms), duration);
    const time = timecode(position);
    if (clock.value !== time) clock.value = time;
    let offset = scroll.scrollLeft;
    if (follow && !fitMode) {
      const next = position * scale();
      if (Math.abs(offset - next) >= .75) {
        scroll.scrollLeft = next;
        offset = scroll.scrollLeft;
      }
      expectedScroll = offset;
    }
    alignViewport(offset);
    const current = new Set(clipsAt(position).filter(({lane}) =>
      (lane.kind !== "gaze" || !gazeRows.get(lane.number)?.hidden)
      && (lane.kind !== "expression" || showExpressions && !hiddenExpressions.has(expressionGroupKey(lane)))
      && (lane.kind !== "person-motion" || showMotion && !hiddenMotion.has(expressionGroupKey(lane)))
      && (lane.kind !== "objects" || objectsCheckbox.checked)
      && audioControls.get(lane.kind)?.toggle.checked !== false));
    for (const item of active) if (!current.has(item)) item.button.removeAttribute("data-active");
    for (const item of current) if (!active.has(item)) item.button.dataset.active = "true";
    active = current;
    // Some seek targets land just before an exact sample on the browser's media clock.
    const objectSelection = selectedItem?.lane.kind === "objects" && objectsCheckbox.checked && video.paused &&
      Math.abs(position - (selectedItem.clip.representative?.timeMs ?? selectedItem.clip.startMs)) < 8;
    if (!current.has(selectedItem) && !objectSelection) {
      selectedItem = [...current].find(item => !["shots", "gaze", "expression", "pause", "objects"].includes(item.lane.kind)
        && !audioControls.has(item.lane.kind)) ?? current.values().next().value;
    }
    const hideSelection = !selectedItem;
    if (selected.hidden !== hideSelection) selected.hidden = hideSelection;
    if (selectedItem) {
      const {clip, lane} = selectedItem;
      const label = `${timecode(clip.startMs)} · ${lane.label} · ${clip.text}`;
      if (selected.textContent !== label) selected.textContent = label;
    }
  }

  /** Seek without resuming playback; coalesce trackpad events to one seek per frame. */
  function drive(ms, follow = true) {
    paint(ms, follow);
    if (!playable()) return;
    queuedSeek = Math.min(position / 1000, Number.isFinite(video.duration) ? video.duration : Infinity);
    video.pause();
    if (!scrollSeek) {
      scrollSeek = requestAnimationFrame(() => {
        scrollSeek = 0;
        const target = queuedSeek;
        queuedSeek = null;
        if (target === null || !playable()) return;
        if (Math.abs(video.currentTime - target) >= .008) video.currentTime = target;
        else sync();
      });
    }
  }

  /** Scale the ruler and lanes together, keeping the current media time stationary. */
  function layout() {
    if (root.hidden || !head || !scroll.clientWidth) return;
    labelWidth = head.offsetWidth;
    trackWidth = Math.max(1, scroll.clientWidth - labelWidth);
    lead = Math.round(trackWidth * .24);
    fittedScale = fitTimelineScale(duration, trackWidth);
    if (fitMode) { scroll.scrollLeft = 0; expectedScroll = 0; }
    root.dataset.zoomFit = String(fitMode);
    sheet.style.width = `${fitMode ? scroll.clientWidth : Math.ceil(duration * scale()) + scroll.clientWidth}px`;
    const rootFont = Number.parseFloat(getComputedStyle(document.documentElement).fontSize);
    const markerRanges = [];
    for (const {button, clip, lane, marker} of clips) {
      const start = lead + clip.startMs * scale();
      const width = Math.max(fitMode ? 2 : 10, (clip.endMs - clip.startMs) * scale());
      button.style.left = `${start}px`;
      button.style.width = `${width}px`;
      if (lane.kind === "gaze" || lane.kind === "expression" || lane.kind === "pause" || audioControls.has(lane.kind)) {
        const end = start + width;
        markerRanges.push([start, end, {marker, start, end}]);
      }
      if (lane.kind === "pause") marker.textContent = width < rootFont * 2.6 ? "‖" : clip.length;
      if (audioControls.has(lane.kind)) {
        const short = {music: "♫", background: "♪"};
        marker.textContent = width < rootFont * 4 && short[lane.kind] ? short[lane.kind] : clip.marker;
      }
    }
    for (const item of edgeMarkers) item.marker.style.removeProperty("left");
    edgeMarkers.clear();
    markersAt = timelineIntervalLookup(markerRanges);
    markerScroll = -1;
    ruler.replaceChildren();
    const tickWidth = Math.max(68, rootFont * 4.9);
    const step = TICKS.find(tick => tick * scale() >= tickWidth && duration / tick < 300) ?? TICKS.at(-1);
    for (let ms = 0; ms <= duration; ms += step) {
      const label = document.createElement("span");
      label.className = "editor-tick";
      label.style.left = `${lead + ms * scale()}px`;
      label.textContent = timecode(ms, step >= 1000);
      ruler.append(label);
    }
    zoomOut.disabled = scale() <= .00001;
    zoomIn.disabled = scale() >= .64;
    fit.setAttribute("aria-pressed", String(fitMode));
    zoomSummary.value = `${Math.round(scale() * 1000)} px/s`;
    fit.title = fit.dataset.hint = `${uiText("Вписать шкалу")} · ${zoomSummary.value}`;
    paint(position, true);
  }

  /** Render a sticky track header, and a scrollable track with the same row height. */
  function row(kind, caption) {
    const band = document.createElement("div");
    band.className = `editor-band editor-band-${kind}`;
    const label = document.createElement("div");
    label.className = "editor-label";
    label.dataset.agWidth = "content";
    label.dataset.agWidthMinEm = "3.5";
    const track = document.createElement("div");
    track.className = "editor-track";
    if (caption) label.textContent = caption;
    band.append(label, track);
    sheet.append(band);
    return {band, label, track};
  }

  function render(model) {
    sheet.replaceChildren();
    objectButtonsByProfile.clear();
    gazeRows = new Map();
    gazeButtons = new Map();
    expressionRows = new Map();
    expressionButtons = new Map();
    expressionHeadings = new Map();
    motionRows = new Map();
    motionButtons = new Map();
    objectRows = [];
    for (const layer of audioControls.values()) layer.row = null;
    clips = [];
    markersAt = timelineIntervalLookup([]);
    edgeMarkers.clear();
    active = new Set();
    selectedItem = undefined;
    const axis = row("ruler", uiText("Время"));
    head = axis.label;
    ruler = axis.track;
    // Tick text is decorative: the clock and each seekable clip already expose exact times accessibly.
    ruler.setAttribute("aria-hidden", "true");
    ruler.addEventListener("click", event => {
      drive((event.clientX - ruler.getBoundingClientRect().left - lead) / scale());
    });
    const withGaze = new Set(model.lanes.filter(lane => lane.kind === "gaze").map(lane => lane.number));
    const expressionIds = new Map();
    for (const lane of model.lanes) {
      if (lane.kind !== "expression") continue;
      const key = expressionGroupKey(lane);
      if (!expressionIds.has(key)) expressionIds.set(key, []);
      expressionIds.get(key).push(`media-editor-${lane.id}`);
    }
    const motionIds = new Map();
    for (const lane of model.lanes) {
      if (lane.kind !== "person-motion") continue;
      const key = expressionGroupKey(lane);
      if (!motionIds.has(key)) motionIds.set(key, []);
      motionIds.get(key).push(`media-editor-${lane.id}`);
    }
    for (const lane of model.lanes) {
      const band = row(lane.kind);
      band.track.setAttribute("role", "group");
      band.track.setAttribute("aria-label", lane.label);
      if (lane.kind === "speaker") {
        const profileButton = document.createElement("button");
        profileButton.type = "button";
        profileButton.className = "editor-speaker-title";
        profileButton.dataset.timelineControl = `profile-${lane.number}`;
        profileButton.setAttribute("aria-label", uiText`Открыть профиль · ${lane.label}`);
        profileButton.onclick = () => openProfile(lane.number, profileButton);
        const avatar = document.createElement("span");
        avatar.className = "editor-avatar";
        const photo = image(lane.photo);
        if (photo) avatar.append(photo);
        else avatar.textContent = lane.number;
        avatar.setAttribute("aria-hidden", "true");
        const title = document.createElement("span");
        title.textContent = lane.label;
        title.title = lane.label;
        profileButton.append(avatar, title);
        band.label.append(profileButton);
        if (withGaze.has(lane.number)) {
          const toggle = document.createElement("button");
          toggle.type = "button";
          toggle.className = "editor-speaker-gaze";
          toggle.dataset.timelineControl = `gaze-${lane.number}`;
          installTimelineIcon(toggle, "gaze", uiText`Взгляд · ${lane.label}`);
          toggle.setAttribute("aria-controls", `media-editor-gaze-${lane.number}`);
          toggle.onclick = () => {
            if (hiddenGaze.has(lane.number)) hiddenGaze.delete(lane.number);
            else hiddenGaze.add(lane.number);
            refreshGaze();
          };
          band.label.append(toggle);
          gazeButtons.set(lane.number, toggle);
        }
        const actions = [];
        if (expressionIds.has(lane.number)) {
          actions.push(expressionAction(lane.number, expressionIds.get(lane.number)));
        }
        if (motionIds.has(lane.number)) {
          actions.push(motionAction(lane.number, motionIds.get(lane.number)));
        }
        if (actions.length) band.label.append(...speakerActions(lane, actions));
      } else if (lane.kind === "expression-face") {
        const title = document.createElement("button");
        title.type = "button";
        title.className = "editor-speaker-title editor-unbound-face-title";
        title.dataset.agWidth = "content";
        title.dataset.agWidthMinEm = "2";
        title.dataset.timelineControl = `profile-face-${lane.faceId}`;
        title.setAttribute("aria-label", uiText`Открыть профиль · ${lane.label}`);
        title.onclick = () => openProfile({faceId: lane.faceId}, title);
        const photo = image(lane.photo);
        if (photo) {
          const avatar = document.createElement("span");
          avatar.className = "editor-avatar";
          avatar.setAttribute("aria-hidden", "true");
          avatar.append(photo);
          title.append(avatar);
        }
        title.append(lane.label);
        const key = expressionGroupKey(lane);
        band.band.id = `media-editor-${lane.id}`;
        expressionHeadings.set(key, band.band);
        const actions = [];
        if (expressionIds.has(key)) actions.push(expressionAction(key, expressionIds.get(key)));
        if (motionIds.has(key)) actions.push(motionAction(key, motionIds.get(key)));
        band.label.append(title, ...speakerActions(lane, actions));
      } else {
        band.label.textContent = lane.kind === "gaze" ? uiText("Взгляд") : lane.label;
      }
      if (lane.kind === "gaze") {
        band.band.id = `media-editor-gaze-${lane.number}`;
        gazeRows.set(lane.number, band.band);
      }
      if (lane.kind === "expression") {
        band.band.id = `media-editor-${lane.id}`;
        const key = expressionGroupKey(lane);
        if (!expressionRows.has(key)) expressionRows.set(key, []);
        expressionRows.get(key).push(band.band);
        band.label.textContent = uiText`Выражения · лицо ${lane.faceId}`;
      }
      if (lane.kind === "person-motion") {
        const key = expressionGroupKey(lane);
        band.band.id = `media-editor-${lane.id}`;
        if (!motionRows.has(key)) motionRows.set(key, []);
        motionRows.get(key).push(band.band);
        band.band.hidden = !showMotion || hiddenMotion.has(key);
        band.label.textContent = uiText`Движение персонажа · лицо ${lane.faceId}`;
      }
      if (lane.kind === "objects") {
        objectRows.push(band.band);
        band.band.hidden = !objectsCheckbox.checked;
      }
      const audioLayer = audioControls.get(lane.kind);
      if (audioLayer) {
        audioLayer.row = band.band;
        band.band.id = `media-editor-lane-${lane.kind}`;
        audioLayer.toggle.setAttribute("aria-controls", band.band.id);
        band.band.hidden = !audioLayer.toggle.checked;
      }
      for (const clip of lane.clips) {
        const button = document.createElement("button");
        button.type = "button";
        button.className = `editor-clip editor-clip-${lane.kind}`;
        const time = timecode(clip.startMs);
        const label = `${time} · ${clip.text}`;
        const color = lane.kind === "shots" && clip.average;
        const description = color ? `${label} · ${uiText("Средний цвет")} ${color}` :
          (clip.description ? `${label} · ${clip.description}` : label);
        button.title = description;
        button.setAttribute("aria-label", `${lane.label} · ${description}`);
        if (color) button.style.setProperty("--shot-color", color);
        const preview = image(clip.jpeg);
        if (preview && lane.kind === "shots") button.append(preview);
        const text = document.createElement("span");
        text.textContent = lane.kind === "gaze" ? gazeIcon(clip.direction) : clip.length ?? clip.marker ?? clip.text;
        if (lane.kind === "gaze") button.dataset.direction = clip.direction;
        if (lane.kind === "expression") button.dataset.expression = clip.label;
        if (lane.kind === "person-motion" && clip.depth) button.dataset.depth = clip.depth;
        if (lane.kind === "objects") {
          if (clip.id) button.dataset.graphNode = clip.id;
          if (clip.profile) {
            const id = clip.profile.id;
            button.dataset.objectProfile = id;
            if (!objectButtonsByProfile.has(id)) objectButtonsByProfile.set(id, []);
            objectButtonsByProfile.get(id).push(button);
          }
          if (clip.screen) {
            button.dataset.interfaceKind = clip.screen.type;
            const badge = document.createElement("small");
            badge.className = "editor-object-kind";
            badge.dataset.kind = clip.screen.type;
            badge.textContent = clip.screen.type;
            button.append(badge);
          }
          const accent = objectAccent(clip.representative?.palette ?? clip.profile?.appearance?.palette);
          if (accent) button.style.setProperty("--object-color", accent);
          button.setAttribute("aria-controls", "media-object-preview");
          button.setAttribute("aria-expanded", "false");
        }
        if (lane.kind === "pause") {
          button.dataset.position = clip.position;
          button.dataset.basis = clip.basis;
        }
        button.append(text);
        const item = {button, clip, lane, marker: text};
        button.onclick = () => {
          if (lane.kind === "expression") {
            objectPreview.clear();
            selectedItem = item;
            // A queued trackpad seek must not overwrite the selected expression frame.
            queuedSeek = null;
            cancelAnimationFrame(scrollSeek);
            scrollSeek = 0;
            paint(clip.startMs, true);
            expressionPreview.open(clip, expressionEvidence, button);
            return;
          }
          const object = lane.kind === "objects";
          if (object) objectPreview.show(button, clip);
          else objectPreview.clear();
          selectedItem = item;
          drive(clip.representative?.timeMs ?? clip.cutMs ?? clip.startMs);
          if (object) objectPreview.capture();
        };
        clips.push(item);
        band.track.append(button);
      }
    }
    objectsControl.hidden = !objectRows.length;
    for (const layer of audioControls.values()) layer.control.hidden = !layer.row;
    needle = document.createElement("div");
    needle.className = "editor-needle";
    needle.setAttribute("aria-hidden", "true");
    sheet.append(needle);
    clipsAt = timelineIntervalLookup(clips.map(item => [item.clip.startMs, item.clip.endMs, item]));
    refreshGaze();
    refreshExpressions();
    refreshMotion();
  }

  /** Highlight linked observations without filling gaps or moving the playhead. */
  function highlightObjectProfile(id) {
    for (const button of highlightedObjects) delete button.dataset.profileRelated;
    highlightedObjects = objectButtonsByProfile.get(id) ?? [];
    for (const button of highlightedObjects) button.dataset.profileRelated = "true";
  }

  /** Identify the same clip even when speaker evidence moves it to another speech lane. */
  function clipKey(item) {
    const {lane, clip} = item;
    if (clip.id) return clip.id;
    const group = ["speaker", "unknown", "overlap"].includes(lane.kind) ? "speech" : lane.id;
    const reference = lane.kind === "objects" ? clip.representative?.timeMs ?? clip.label : "";
    return JSON.stringify([group, clip.startMs, clip.endMs, reference]);
  }

  /** Remember keyboard focus and object inspection before replacing timeline rows. */
  function timelineInteraction() {
    const focused = document.activeElement;
    const focusedClip = clips.find(item => item.button === focused);
    const object = clips.find(item => item.button === objectPreview.selected());
    return {focus: focusedClip && clipKey(focusedClip), control: focused?.dataset.timelineControl,
      selection: selectedItem && clipKey(selectedItem), object: object && clipKey(object),
      scrollTop: scroll.scrollTop};
  }

  /** Restore interaction against the refreshed evidence at the current playhead. */
  function restoreTimelineInteraction(state) {
    const lookup = key => key === undefined ? undefined : clips.find(item => clipKey(item) === key);
    selectedItem = lookup(state.selection);
    if (state.object) {
      const item = lookup(state.object);
      objectPreview.rebind(item?.button, item?.clip);
    }
    scroll.scrollTop = state.scrollTop;
    const control = state.control && [...sheet.querySelectorAll("[data-timeline-control]")]
      .find(button => button.dataset.timelineControl === state.control);
    (lookup(state.focus)?.button || control)?.focus({preventScroll: true});
  }

  /** Follow the media clock once both the queued scrub and the browser's seek have settled. */
  function sync() {
    // The player may land at a different time if another seek takes over or its seekable range changes.
    if (!root.hidden && playable() && queuedSeek === null && !video.seeking) paint(video.currentTime * 1000, true);
  }

  function animate() {
    animation = 0;
    sync();
    startAnimation();
  }

  /** Resume frame updates when analysis appears during already-running playback. */
  function startAnimation() {
    if (!animation && !root.hidden && !video.paused && !video.ended) animation = requestAnimationFrame(animate);
  }

  /** Change view scale without seeking the source, including when leaving overview mode. */
  function zoom(factor) {
    zoomScale = Math.max(.00001, Math.min(.64, scale() * factor));
    fitMode = false;
    layout();
  }
  zoomOut.onclick = () => zoom(1 / 1.7);
  zoomIn.onclick = () => zoom(1.7);
  fit.onclick = () => { fitMode = true; layout(); };
  scroll.addEventListener("wheel", event => {
    if ((!event.ctrlKey && !event.metaKey) || root.hidden) return;
    event.preventDefault();
    zoom(Math.exp(-Math.max(-100, Math.min(100, event.deltaY)) * .008));
  }, {passive: false});
  scroll.addEventListener("scroll", () => {
    if (root.hidden) return;
    const offset = scroll.scrollLeft;
    if (Math.abs(offset - expectedScroll) < .75) return;
    expectedScroll = offset;
    drive(offset / scale(), false);
  });
  scroll.addEventListener("keydown", event => {
    if (event.target !== scroll || playable() && Boolean(video.videoWidth)) return;
    const target = {ArrowLeft: position - 1000, ArrowRight: position + 1000, Home: 0, End: duration}[event.key];
    if (target === undefined) return;
    event.preventDefault();
    drive(target);
  });
  video.addEventListener("play", startAnimation);
  video.addEventListener("timeupdate", () => { if (!animation) sync(); });
  for (const event of ["loadedmetadata", "seeked", "pause", "ended"]) video.addEventListener(event, sync);
  new ResizeObserver(layout).observe(scroll);

  return {
    setData(data, evidence, gazes = new Map(), preserveInteraction = false) {
      video.dataset.editorFrameStepMs = String(evidence?.timeline?.stepMs ?? data?.videoEvidence?.timeline?.stepMs ?? 40);
      expressionEvidence = evidence ?? data?.videoEvidence;
      if (!data) expressionPreview.clear();
      const interaction = preserveInteraction ? timelineInteraction() : null;
      if (!interaction) objectPreview.clear();
      const model = data && buildTimeline(data, evidence, gazes);
      if (!model?.lanes.length) {
        objectPreview.clear();
        root.hidden = true;
        showGaze = true;
        showExpressions = true;
        hiddenExpressions.clear();
        objectsCheckbox.checked = true;
        for (const layer of audioControls.values()) {
          layer.toggle.checked = true;
          layer.row = null;
        }
        hiddenGaze.clear();
        hiddenMotion.clear();
        showMotion = true;
        motionRows.clear();
        motionButtons.clear();
        allMotionButton.hidden = true;
        allMotionButton.setAttribute("aria-pressed", "true");
        allMotionButton.removeAttribute("aria-controls");
        position = duration = expectedScroll = 0;
        queuedSeek = null;
        cancelAnimationFrame(scrollSeek);
        cancelAnimationFrame(animation);
        scrollSeek = animation = 0;
        sheet.replaceChildren();
        clips = [];
        clipsAt = timelineIntervalLookup([]);
        markersAt = timelineIntervalLookup([]);
        edgeMarkers.clear();
        active.clear();
        selectedItem = undefined;
        selected.hidden = true;
        return;
      }
      duration = model.duration;
      if (playable() && queuedSeek === null && !video.seeking) position = video.currentTime * 1000;
      root.hidden = false;
      selected.hidden = true;
      render(model);
      if (interaction) restoreTimelineInteraction(interaction);
      layout();
      startAnimation();
    },
  };
}
