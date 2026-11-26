import {uiText} from "./i18n.js";
import {effectiveSpeaker, mixedSpeakerLabel} from "./speaker-label.js";
import {packMedia} from "./person-layer.js";
import {timelineIntervalLookup} from "./timeline-intervals.js";

export const FRAME_COLORS = Object.freeze({
  face: "#f5f1e8", speech: "#ffb34e", body: "#70dddb", object: "#99e198", text: "#c7adfb",
  interface: "#98bfff", change: "#ffdf66", graphic: "#ff9ddc",
});
const {face: FACE, speech: SPEAKING, body: BODY} = FRAME_COLORS;
const OBJECT_LIMIT = 3;
const DISPLAY_LABELS = new Set(["tv", "tv screen", "monitor", "screen"]);
const TEXT_LIMIT = 4;
const DEPTH_LABELS = {foreground: "Передний план", midground: "Средний план", background: "Задний план"};
const GAZE_LABELS = {left: "←", right: "→", center: "•"};
const VERTICAL_DIRECTIONS = {up: "↑", down: "↓"};
const SCENE_RADIUS_MS = 100;
const PANEL_RADIUS_MS = 750;
const CHANGE_PREVIEW_MS = 700;
const CHANGE_VERIFICATION_LIMIT_MS = 2400;
const intervalStart = row => row.startMs;
const isGraphic = region => region.basis === "overlay-persistence";

/** Resolve all rectangles covering the timestamp, including independently appearing graphics. */
function regionLookup(rows = []) {
  const at = timelineIntervalLookup(rows.map(row => [row.startMs, row.endMs, row.regions]));
  return stamp => at(stamp).flat();
}

/** Keep the same color in the browser and the serialized MP4 drawing instructions. */
function outlineColor(hex, alpha) {
  const value = Number.parseInt(hex.slice(1), 16);
  return `rgba(${value >> 16}, ${(value >> 8) & 255}, ${value & 255}, ${alpha})`;
}
const OBJECT_OUTLINE = outlineColor(FRAME_COLORS.object, ".50");
const TEXT_OUTLINE = outlineColor(FRAME_COLORS.text, ".48");

/** Index ordered disjoint intervals, preserving first-match behavior for imported data. */
function intervalLookup(rows = []) {
  let previousEnd = -Infinity;
  for (const row of rows) {
    if (!Number.isFinite(row.startMs) || !Number.isFinite(row.endMs)
        || row.startMs < previousEnd || row.endMs <= row.startMs) {
      return time => rows.find(item => item.startMs <= time && time < item.endMs);
    }
    previousEnd = row.endMs;
  }
  return time => {
    const row = rows[after(rows, time, intervalStart) - 1];
    return row && time < row.endMs ? row : undefined;
  };
}

/** Build overlay lookups once per report or export rather than once per painted frame. */
function frameLookups(evidence, phrases) {
  const tracks = new Map();
  for (const row of evidence?.editorTimeline?.personMotion ?? []) {
    if (!Object.hasOwn(DEPTH_LABELS, row?.depth)) continue;
    const key = `${row.track}:${row.faceId}`;
    if (!tracks.has(key)) tracks.set(key, []);
    tracks.get(key).push(row);
  }
  for (const [key, rows] of tracks) tracks.set(key, intervalLookup(rows));
  return {
    phraseAt: intervalLookup(phrases),
    simultaneousAt: intervalLookup(evidence?.simultaneousFaces ?? []),
    localSpeechAt: intervalLookup(evidence?.visibleSpeechIntervals ?? []),
    depthAt: (track, face, stamp) => tracks.get(`${track}:${face}`)?.(stamp),
    personCuts: (evidence?.editorTimeline?.shots ?? []).map(shot => shot.startMs).sort((a, b) => a - b),
  };
}

/** Find a unique, sustained speaking face near the current frame. */
function sustainedFace(timeline, index, track) {
  const window = timeline.frames.slice(Math.max(0, index - 2), index + 3);
  return window.length === 5 && window.every(frame => {
    const active = frame.filter(hit => hit[5] === 1);
    return active.length === 1 && active[0][0] === track;
  });
}

/** Resolve the final timeline label for the visible face, excluding unassigned faces. */
function faceSpeaker(evidence, hit) {
  const binding = evidence?.faceSpeakers?.[hit[6] ?? hit[0]];
  return binding?.basis === "visible-only" ? null : binding?.speaker ?? null;
}

/** A frame can highlight only the final cue speaker or explicitly timed overlap speakers. */
function speakersAt(phrase, timeMs) {
  if (!phrase) return [];
  if (mixedSpeakerLabel(phrase, timeMs)) {
    return (phrase.overlaps ?? []).filter(span => span.startMs <= timeMs && timeMs < span.endMs)
      .flatMap(span => span.speakers);
  }
  const speaker = effectiveSpeaker(phrase);
  return speaker === null ? [] : [speaker];
}

/** Map the playback clock to detector hits and keep ambiguous speech detached from faces. */
export function frameAnnotation(evidence, phrases, timeMs, lookups) {
  const timeline = evidence?.timeline;
  const index = timeline ? Math.floor(timeMs / timeline.stepMs) : -1;
  const faces = timeline?.frames?.[index] ?? [];
  const phrase = lookups ? lookups.phraseAt(timeMs)
    : phrases.find(row => row.startMs <= timeMs && timeMs < row.endMs);
  const observation = phrase && evidence?.phrases?.[phrase.startMs]?.activity;
  const activeFaces = faces.filter(face => face[5] === 1);
  const allowed = speakersAt(phrase, timeMs);
  const confirmed = activeFaces.filter(hit => allowed.includes(faceSpeaker(evidence, hit)));
  let speakerFace;
  if (activeFaces.length === 1 && confirmed.length && observation?.state === "visible-active"
      && activeFaces[0][0] === observation.track) {
    speakerFace = observation.track;
  }
  if (!speakerFace && phrase && !mixedSpeakerLabel(phrase, timeMs) && activeFaces.length === 1 && confirmed.length) {
    const track = activeFaces[0][0];
    if (sustainedFace(timeline, index, track)) speakerFace = track;
  }
  const local = lookups?.localSpeechAt
    ? lookups.localSpeechAt(timeMs)
    : evidence?.visibleSpeechIntervals?.find(row => row.startMs <= timeMs && timeMs < row.endMs);
  const single = activeFaces.length === 1 ? activeFaces[0] : null;
  const bound = single ? faceSpeaker(evidence, single) : null;
  const localActive = !speakerFace && phrase && !mixedSpeakerLabel(phrase, timeMs) && single
    && local?.basis === "light-asd-local" && local.cueStartMs === phrase.startMs
    && local.track === single[0] && local.faceId === (single[6] ?? single[0])
    && (bound === null || allowed.includes(bound));
  if (localActive) speakerFace = local.track;
  const simultaneous = lookups ? lookups.simultaneousAt(timeMs)
    : evidence?.simultaneousFaces?.find(span => span.startMs <= timeMs && timeMs < span.endMs);
  const speakingFaces = simultaneous && !localActive
    ? confirmed.filter(hit => simultaneous.tracks.includes(hit[0])).map(hit => hit[0])
    : (speakerFace ? [speakerFace] : []);
  return {faces, phrase, speakerFace, speakingFaces};
}

/** Use sampled geometry only while the exact corresponding face track is present. */
export function personMarks(layer, faces, at) {
  const tracks = new Set(faces.map(face => face[0]));
  const output = new Map();
  const index = Math.round(at / 250);
  const nearby = (layer?.frames ?? []).slice(Math.max(0, index - 1), index + 2)
    .sort((a, b) => Math.abs(at - a.timeMs) - Math.abs(at - b.timeMs));
  for (const row of nearby) {
    const distance = Math.abs(row.timeMs - at);
    for (const person of row.people) {
      if (!tracks.has(person.track)) continue;
      const mark = output.get(person.track) ?? {};
      if (!mark.box && distance <= 150) mark.box = person.box;
      if (!mark.gaze && distance <= 300) mark.gaze = person.gaze;
      if (!mark.headPose && distance <= 250) mark.headPose = person.headPose;
      output.set(person.track, mark);
    }
  }
  return output;
}

/** Position the intrinsic video within a player using centered object-fit: contain. */
function imageBounds(video, width, height) {
  const scale = Math.min(width / video.videoWidth, height / video.videoHeight);
  const imageWidth = video.videoWidth * scale;
  const imageHeight = video.videoHeight * scale;
  return [(width - imageWidth) / 2, (height - imageHeight) / 2, imageWidth, imageHeight];
}

/** Draw a measured transcript window with an ellipsis on exceptionally long lines. */
function captionLines(ctx, text, width) {
  const lines = [];
  let line = "";
  let cropped = false;
  for (const word of text.split(/\s+/)) {
    const joined = line ? line + " " + word : word;
    if (ctx.measureText(joined).width > width && line) {
      lines.push(line);
      if (lines.length === 3) {
        cropped = true;
        break;
      }
      line = word;
    } else {
      line = joined;
    }
  }
  if (!cropped && line) lines.push(line);
  return lines.map((value, index) => {
    if (ctx.measureText(value).width <= width && !(cropped && index === 2)) return value;
    const letters = Array.from(value);
    while (letters.length && ctx.measureText(letters.join("") + "…").width > width) letters.pop();
    return letters.join("") + "…";
  });
}

/** Draw face tracks and retain the label bounds for collision-free caption placement. */
function drawFaces(ctx, faces, bounds, speakingFaces, faceSpeakers, marks) {
  const [left, top, width, height] = bounds;
  let anchor;
  const labels = [];
  ctx.font = "600 12px system-ui";
  for (const [track, x, y, w, h, , face = track] of faces) {
    const box = [left + x * width, top + y * height, w * width, h * height];
    const active = speakingFaces.includes(track);
    const color = active ? SPEAKING : FACE;
    ctx.lineWidth = 4;
    ctx.strokeStyle = "rgba(10, 8, 6, .78)";
    ctx.strokeRect(...box);
    ctx.lineWidth = 2;
    ctx.strokeStyle = color;
    ctx.strokeRect(...box);

    const binding = faceSpeakers?.[face];
    let caption = binding?.speaker && binding.basis !== "visible-only"
      ? uiText`Спикер ${binding.speaker}` : uiText`Лицо ${face}`;
    if (active && (!binding?.speaker || binding.basis === "visible-only")) {
      caption += " · " + uiText("Говорит в кадре");
    }
    const mark = marks.get(track);
    if (mark?.depth) caption += " · " + uiText(DEPTH_LABELS[mark.depth])
      + (mark.depthBasis === "relative-face-scale" ? " ≈" : "");
    const gaze = GAZE_LABELS[mark?.gaze?.horizontal];
    if (gaze) caption += " · " + uiText("Взгляд") + " " + gaze;
    if (mark?.headPose) {
      const head = GAZE_LABELS[mark.headPose.horizontal] ?? "";
      const tilt = VERTICAL_DIRECTIONS[mark.headPose.vertical] ?? "";
      if (head || tilt) caption += " · " + uiText("Голова") + " " + (head === "•" && tilt ? tilt : head + tilt);
    }
    ctx.font = "600 12px system-ui";
    const textWidth = ctx.measureText(caption).width;
    if (textWidth + 12 > width) {
      ctx.font = `600 ${Math.max(8, Math.floor(12 * (width - 12) / textWidth))}px system-ui`;
    }
    const size = Math.min(width, ctx.measureText(caption).width + 12);
    const labelX = Math.min(box[0], left + width - size);
    const labelY = Math.max(top, box[1] - 20);
    ctx.fillStyle = "rgba(16, 13, 10, .94)";
    const labelBounds = [labelX, labelY, size, 20];
    ctx.fillRect(...labelBounds);
    labels.push(labelBounds);
    ctx.fillStyle = color;
    ctx.fillText(caption, labelX + 6, labelY + 14);
    if (active) anchor = box;
  }
  return {anchor, labels};
}

/** Outline the visible body beneath the existing face layer. */
function drawBodies(ctx, marks, bounds) {
  const [left, top, width, height] = bounds;
  ctx.save();
  ctx.lineWidth = 2;
  ctx.strokeStyle = BODY;
  ctx.setLineDash([7, 4]);
  for (const {box} of marks.values()) {
    if (!box) continue;
    const [x, y, w, h] = box;
    ctx.strokeRect(left + x * width, top + y * height, w * width, h * height);
  }
  ctx.restore();
}

/** Keep the caption near an active face, or centered above the player controls. */
function drawCaption(ctx, phrase, anchor, width, height, label, labels) {
  const fontSize = width < 380 ? 13 : 15;
  const lineHeight = fontSize + 5;
  ctx.font = `600 ${fontSize}px system-ui`;
  const text = label ? `${label} · ${phrase.text}` : phrase.text;
  const lines = captionLines(ctx, text, Math.min(420, width - 40));
  const boxWidth = Math.max(...lines.map(line => ctx.measureText(line).width)) + 20;
  const boxHeight = lines.length * lineHeight + 12;
  const controlsHeight = width < 400 ? 72 : 56;
  const bottom = Math.max(8, height - controlsHeight - boxHeight);
  let left = (width - boxWidth) / 2;
  let top = bottom;
  if (anchor) {
    const [x, y, w, h] = anchor;
    left = Math.min(Math.max(8, x + (w - boxWidth) / 2), width - boxWidth - 8);
    const below = y + h + 9;
    top = below <= bottom ? below : Math.max(8, Math.min(bottom, y - 28 - boxHeight));
  }
  top = captionTop(left, top, boxWidth, boxHeight, height, bottom, labels);
  ctx.fillStyle = "rgba(16, 13, 10, .92)";
  ctx.fillRect(left, top, boxWidth, boxHeight);
  if (anchor) {
    ctx.fillStyle = SPEAKING;
    ctx.fillRect(left, top, 3, boxHeight);
  }
  ctx.fillStyle = FACE;
  lines.forEach((line, index) => ctx.fillText(line, left + 10, top + 6 + fontSize + index * lineHeight));
}

/** Move a caption clear of face labels, preferring the area above player controls. */
function captionTop(left, preferred, width, height, frameHeight, controlsLimit, labels) {
  const neighbors = labels.filter(([x, , w]) => x < left + width && left < x + w);
  const candidates = [preferred, ...neighbors.map(([, top, , h]) => top + h + 4)];
  for (const [, top] of neighbors) candidates.push(top - height - 4);
  const clear = top => neighbors.every(([, y, , h]) => top + height <= y || y + h <= top);
  for (const limit of [controlsLimit, frameHeight - height - 4]) {
    const position = candidates.find(top => top >= 4 && top <= limit && clear(top));
    if (position !== undefined) return position;
  }
  return preferred;
}

/** Draw OCR lines and object boxes on the single sampled frame selected by the user. */
function drawScene(ctx, frame, bounds) {
  const [left, top, width, height] = bounds;
  ctx.save();
  ctx.lineWidth = 2;
  ctx.setLineDash([5, 3]);
  ctx.strokeStyle = FRAME_COLORS.text;
  for (const {box: [x, y, w, h]} of frame.text) ctx.strokeRect(left + x * width, top + y * height, w * width, h * height);
  ctx.strokeStyle = FRAME_COLORS.object;
  for (const {box: [x, y, w, h]} of frame.objects) ctx.strokeRect(left + x * width, top + y * height, w * width, h * height);
  ctx.restore();
}

/** Return the first item strictly after a playback timestamp. */
function after(items, time, stamp) {
  let left = 0, right = items.length;
  while (left < right) {
    const middle = (left + right) >>> 1;
    if (stamp(items[middle]) <= time) left = middle + 1;
    else right = middle;
  }
  return left;
}

const frameTime = frame => frame.timeMs;

/** Show a sampled person count only near its timestamp on the same shot. */
export function personCountAt(samples, cuts, time, visualCuts = []) {
  const index = after(samples, time, frameTime);
  const candidates = [samples[index - 1], samples[index]].filter(Boolean);
  const shot = after(cuts, time, value => value);
  const visualShot = after(visualCuts, time, value => value);
  const hit = candidates
    .filter(row => Math.abs(row.timeMs - time) <= 500
      && after(cuts, row.timeMs, value => value) === shot
      && after(visualCuts, row.timeMs, value => value) === visualShot)
    .sort((a, b) => Math.abs(a.timeMs - time) - Math.abs(b.timeMs - time))[0];
  return hit?.count ?? null;
}

/** Prefer a measured object check, including a check with zero detections. */
export function sceneFrameAt(frames, cuts, time, radius = SCENE_RADIUS_MS) {
  const index = after(frames, time, frameTime);
  const previous = frames[index - 1], next = frames[index];
  const shot = after(cuts, time, value => value);
  const eligible = frame => frame && Math.abs(frame.timeMs - time) < radius &&
    after(cuts, frame.timeMs, value => value) === shot;
  if (eligible(previous) && (!eligible(next) || time - previous.timeMs <= next.timeMs - time)) return previous;
  return eligible(next) ? next : null;
}

const sampleTime = ([time]) => time;

/** Index canonical track boxes once. Legacy reports without measurements use raw frames. */
function indexObjectTracks(clips) {
  if (!Array.isArray(clips) || clips.some(clip => !Array.isArray(clip.boxSamples) || !clip.boxSamples.length)) {
    return null;
  }
  const rows = [];
  for (const clip of clips) {
    // The independently measured panel has its own outline.
    if (clip.label === "mobile-ui" || clip.label === "ui") continue;
    const isolated = clip.boxSamples.length === 1;
    rows.push({clip, isolated, startMs: clip.startMs - (isolated ? SCENE_RADIUS_MS : 0),
      endMs: clip.endMs + (isolated ? SCENE_RADIUS_MS : 0)});
  }
  rows.sort((a, b) => a.startMs - b.startMs);
  const edges = new Set();
  const endThrough = [];
  let latestEnd = -Infinity;
  for (const {startMs, endMs, clip: {boxSamples}} of rows) {
    edges.add(startMs).add(endMs);
    for (const [stamp] of boxSamples) edges.add(stamp);
    latestEnd = Math.max(latestEnd, endMs);
    endThrough.push(latestEnd);
  }
  return {rows, endThrough, edges};
}

/** Hold the last measured box until the next sample within the same shot-local track. */
function objectMarksAt(tracks, cuts, timeMs, nearbyFrame) {
  const {rows, endThrough} = tracks;
  const marks = [];
  const cutIndex = after(cuts, timeMs, value => value);
  const shotStart = cuts[cutIndex - 1] ?? -Infinity;
  const first = after(endThrough, timeMs, value => value);
  const limit = after(rows, timeMs, intervalStart);
  for (let index = first; index < limit; index++) {
    const {clip, isolated, endMs} = rows[index];
    if (timeMs >= endMs) continue;
    // A single detection needs the existing nearest-frame check to be visible at playback FPS.
    const sample = isolated ? clip.boxSamples[0] : clip.boxSamples[after(clip.boxSamples, timeMs, sampleTime) - 1];
    if (isolated && sample[0] !== nearbyFrame?.timeMs) continue;
    if (!sample || sample[0] < shotStart) continue;
    marks.push({label: clip.label, box: sample[1], score: sample[2], trackId: clip.id, timeMs: sample[0]});
  }
  return marks;
}

/** Check whether the measured panel covers the requested fraction of an object's box. */
function coveredByPanel([x, y, w, h], panels, fraction) {
  return panels.some(({box: [a, b, u, v]}) => {
    const commonWidth = Math.max(0, Math.min(x + w, a + u) - Math.max(x, a));
    const commonHeight = Math.max(0, Math.min(y + h, b + v) - Math.max(y, b));
    return commonWidth * commonHeight >= fraction * w * h;
  });
}

/** Suppress small contained marks in automatic preview and keep larger details first. */
function prominentMarks(rows, panels, limit) {
  if (!rows) return undefined;
  const area = ({box: [, , width, height]}) => width * height;
  return rows.filter(({box}) => !coveredByPanel(box, panels, .85))
    .sort((a, b) => area(b) - area(a) || (b.score ?? 0) - (a.score ?? 0)).slice(0, limit);
}

/** Draw the verified panel over its original source area in both preview and export. */
function drawPanels(ctx, panels, width, height) {
  if (!panels.length) return;
  ctx.save();
  ctx.lineWidth = 1.5;
  ctx.strokeStyle = outlineColor(FRAME_COLORS.interface, ".85");
  for (const {box: [x, y, w, h]} of panels) {
    const left = Math.max(1, x * width), top = Math.max(1, y * height);
    ctx.strokeRect(left, top, Math.min(width - 1, (x + w) * width) - left,
                   Math.min(height - 1, (y + h) * height) - top);
  }
  ctx.restore();
}

/** Draw quiet outlines at measured positions, with speech and faces layered above them. */
function drawObjects(ctx, objects, width, height) {
  if (!objects?.length) return;
  ctx.save();
  ctx.lineWidth = 1.25;
  ctx.strokeStyle = OBJECT_OUTLINE;
  for (const {box: [x, y, w, h]} of objects) {
    ctx.strokeRect(x * width, y * height, w * width, h * height);
  }
  ctx.restore();
}

/** Mark OCR regions near their measured frame without covering the lettering. */
function drawTextRegions(ctx, text, width, height) {
  if (!text?.length) return;
  ctx.save();
  ctx.lineWidth = 1.25;
  ctx.strokeStyle = TEXT_OUTLINE;
  ctx.setLineDash([5, 3]);
  for (const {box: [x, y, w, h]} of text) {
    ctx.strokeRect(x * width, y * height, w * width, h * height);
  }
  ctx.restore();
}

/** Draw measured changes or sustained graphic candidates as a separate technical layer. */
function drawChanges(ctx, changes, width, height, style) {
  if (!changes?.length) return;
  ctx.save();
  ctx.lineWidth = style === "graphic" ? 2 : 1.5;
  ctx.strokeStyle = FRAME_COLORS[style];
  ctx.setLineDash(style === "graphic" ? [7, 4] : [3, 3]);
  for (const {box: [x, y, w, h], recognition} of changes) {
    ctx.strokeRect(x * width, y * height, w * width, h * height);
    const label = recognition?.text?.[0]?.text ?? recognition?.objects?.[0]?.label;
    if (typeof label === "string" && label.trim()) {
      const caption = label.slice(0, 28);
      ctx.font = "600 12px system-ui";
      const size = Math.min(width, ctx.measureText(caption).width + 8);
      const left = Math.min(x * width, width - size);
      const top = Math.max(0, y * height - 18);
      ctx.fillStyle = "rgba(16, 13, 10, .94)";
      ctx.fillRect(left, top, size, 18);
      ctx.fillStyle = FRAME_COLORS[style];
      ctx.fillText(caption, left + 4, top + 13);
    }
  }
  ctx.restore();
}

/** Show matching inspected crops only up to their last visual check and within their shot. */
export function previewChangeRegions(exact, inspected, cuts, time) {
  const boundary = cuts[after(cuts, time, value => value) - 1] ?? 0;
  const visible = [];
  const overlaps = ({box}) => coveredByPanel(box, visible, .65)
    || visible.some(region => coveredByPanel(region.box, [{box}], .65));
  for (let index = after(inspected, time, intervalStart) - 1; index >= 0; index--) {
    const row = inspected[index];
    if (row.startMs < boundary || time - row.startMs > CHANGE_VERIFICATION_LIMIT_MS) break;
    if (time > (row.endMs ?? row.startMs + CHANGE_PREVIEW_MS)) continue;
    for (const region of row.regions) {
      if (!overlaps(region)) visible.push(region);
      if (visible.length >= 4) return visible;
    }
  }
  for (const region of exact ?? []) {
    if (!overlaps(region)) visible.push(region);
    if (visible.length >= 4) break;
  }
  return visible;
}

/** Use one layout and draw order for browser pixels and the MP4 render instructions. */
export function paintTechnicalFrame(ctx, evidence, phrases, sceneFrames, cuts, timeMs, width, height,
                                    selected, textFrames = [], lookups, onFrame, panelFrames = [], objectTracks = null, changes = []) {
  const bounds = [0, 0, width, height];
  const {faces, phrase, speakerFace, speakingFaces} = frameAnnotation(evidence, phrases, timeMs, lookups);
  const overlap = phrase && (mixedSpeakerLabel(phrase, timeMs) ||
    (speakingFaces.length > 1 && !phrase.speakerNumber ? uiText("Голоса пересекаются") : null));
  const nearbyFrame = selected ?? sceneFrameAt(sceneFrames, cuts, timeMs);
  const objectMarks = selected || !objectTracks ? nearbyFrame?.objects
    : objectMarksAt(objectTracks, cuts, timeMs, nearbyFrame);
  const textFrame = selected ?? sceneFrameAt(textFrames, cuts, timeMs);
  const panelFrame = selected?.panelChecked ? selected : sceneFrameAt(panelFrames, cuts, timeMs, PANEL_RADIUS_MS);
  const panels = panelFrame?.interfaces?.filter(panel => panel?.panelType === "mobile-ui" || panel?.panelType === "ui") ?? [];
  const objects = selected
    ? objectMarks.filter(item => !DISPLAY_LABELS.has(item.label) || !coveredByPanel(item.box, panels, .9))
    : prominentMarks(objectMarks, panels, OBJECT_LIMIT);
  const text = selected ? textFrame.text : prominentMarks(textFrame?.text, panels, TEXT_LIMIT);
  if (selected) drawScene(ctx, {...selected, objects}, bounds);
  else {
    drawObjects(ctx, objects, width, height);
    drawTextRegions(ctx, text, width, height);
  }
  const graphics = changes.filter(isGraphic);
  const frameChanges = changes.filter(region => !isGraphic(region));
  drawChanges(ctx, frameChanges, width, height, "change");
  drawChanges(ctx, graphics, width, height, "graphic");
  drawPanels(ctx, panels, width, height);
  const marks = personMarks(evidence?.personLayer, faces, timeMs);
  for (const hit of faces) {
    const track = hit[0], face = hit[6] ?? track;
    const depth = lookups?.depthAt?.(track, face, timeMs) ?? (!lookups
      ? evidence?.editorTimeline?.personMotion?.find(row => row?.track === track && row.faceId === face
        && row.startMs <= timeMs && timeMs < row.endMs && Object.hasOwn(DEPTH_LABELS, row.depth))
      : undefined);
    if (depth) marks.set(track, {...marks.get(track), depth: depth.depth, depthBasis: depth.depthBasis});
  }
  drawBodies(ctx, marks, bounds);
  const {anchor, labels} = drawFaces(ctx, faces, bounds, speakingFaces, evidence?.faceSpeakers, marks);
  const anchored = !overlap && speakingFaces.includes(speakerFace) ? anchor : null;
  const speaker = phrase && effectiveSpeaker(phrase);
  const speechLabel = overlap ?? (speaker ? uiText`Спикер ${speaker}` : null);
  if (phrase) drawCaption(ctx, phrase, anchored, width, height, speechLabel, labels);
  onFrame?.({
    timeMs, faces, marks, speakingFaces, phrase, speechLabel,
    personCount: personCountAt(evidence?.editorTimeline?.personPilot?.samples ?? [], cuts, timeMs, lookups?.personCuts),
    shot: cuts.length ? Math.max(1, after(cuts, timeMs, value => value) + (cuts[0] > 0 ? 1 : 0)) : null,
    objects, text, panels, changes: frameChanges, graphics,
    panelTimeMs: panelFrame?.timeMs, objectTimeMs: nearbyFrame?.timeMs,
  });
}

/** Keep drawing data declarative. Browser text measurement also sets the export positions. */
function recordDrawing(measure, target) {
  const states = [];
  const round = values => values.map(value => Math.round(value * 10) / 10);
  return {
    font: "600 12px system-ui", fillStyle: FACE, strokeStyle: FACE, lineWidth: 1, dash: [],
    save() {
      states.push({font: this.font, fillStyle: this.fillStyle, strokeStyle: this.strokeStyle,
        lineWidth: this.lineWidth, dash: this.dash});
    },
    restore() { Object.assign(this, states.pop()); },
    setLineDash(dash) { this.dash = dash; },
    measureText(text) {
      measure.font = this.font;
      return measure.measureText(text);
    },
    fillRect(...rect) { target.push(["fill", this.fillStyle, ...round(rect)]); },
    strokeRect(...rect) { target.push(["stroke", this.strokeStyle, ...round(rect), this.lineWidth, this.dash]); },
    fillText(text, x, y) {
      const size = Number(/(\d+)px/.exec(this.font)[1]);
      target.push(["text", this.fillStyle, ...round([x, y]), size, text]);
    },
  };
}

/** Build sparse timed vectors and coalesce identical adjacent painted samples. */
export function previewExportPlan(measure, evidence, phrases, sceneFrames, cuts, width, height, durationMs, textFrames = [], panelFrames = [], objectClips, changes = []) {
  const boundaries = new Set([0, durationMs]);
  const objectTracks = indexObjectTracks(objectClips);
  const changesAt = regionLookup(changes);
  for (const row of changes) {
    for (const edge of [row.startMs, row.endMs]) {
      if (0 < edge && edge < durationMs) boundaries.add(edge);
    }
  }
  for (const edge of objectTracks?.edges ?? []) {
    if (0 < edge && edge < durationMs) boundaries.add(edge);
  }
  const visual = evidence?.timeline;
  const visualEnd = Math.min(durationMs, (visual?.frames?.length ?? 0) * (visual?.stepMs ?? 40));
  const step = visual?.stepMs >= 20 ? visual.stepMs : 40;
  for (let at = 0; at <= visualEnd; at += step) boundaries.add(at);
  for (const boundary of cuts) {
    if (0 < boundary && boundary < durationMs) boundaries.add(boundary);
  }
  for (const row of evidence?.editorTimeline?.personMotion ?? []) {
    if (!Object.hasOwn(DEPTH_LABELS, row?.depth)) continue;
    for (const edge of [row.startMs, row.endMs]) {
      if (Number.isInteger(edge) && 0 < edge && edge < durationMs) boundaries.add(edge);
    }
  }
  for (const [frames, radius] of [[sceneFrames, SCENE_RADIUS_MS], [textFrames, SCENE_RADIUS_MS],
                                  [panelFrames, PANEL_RADIUS_MS]]) {
    for (const [index, frame] of frames.entries()) {
      const stamp = frame.timeMs;
      const next = frames[index + 1];
      const edges = [stamp - radius + 1, stamp + radius];
      if (next) edges.push(Math.floor((stamp + next.timeMs) / 2) + 1);
      for (const edge of edges) if (0 < edge && edge < durationMs) boundaries.add(edge);
    }
  }
  for (const cue of phrases) {
    for (const row of [cue, ...(cue.overlaps ?? [])]) {
      if (row.startMs > 0 && row.startMs < durationMs) boundaries.add(row.startMs);
      if (row.endMs > 0 && row.endMs < durationMs) boundaries.add(row.endMs);
    }
  }
  const times = [...boundaries].sort((a, b) => a - b);
  const lookups = frameLookups(evidence, phrases);
  const samples = [];
  let lastKey;
  for (let index = 0; index + 1 < times.length; index++) {
    const start = times[index], end = times[index + 1];
    const commands = [];
    paintTechnicalFrame(recordDrawing(measure, commands), evidence, phrases, sceneFrames, cuts, start, width, height,
                        null, textFrames, lookups, undefined, panelFrames, objectTracks, changesAt(start));
    const key = JSON.stringify(commands);
    const previous = samples.at(-1);
    if (commands.length && previous && previous[1] === start && lastKey === key) previous[1] = end;
    else if (commands.length) samples.push([start, end, commands]);
    lastKey = key;
  }
  return {format: "preview-v1", width, height, samples};
}

/** Render one synchronized overlay frame without consuming player input. */
/** Mount the live overlay and notify the monitor after painting or hiding it. */
export function mountVideoOverlay(video, canvas, options, toggle, onFrame, changeToggle) {
  const ctx = canvas.getContext("2d");
  const frameCallbacks = typeof video.requestVideoFrameCallback === "function";
  let evidence;
  let phrases = [];
  let lookups = frameLookups(evidence, phrases);
  let objectClips;
  let objectTracks;
  let sceneFrame;
  let sceneFrames = [];
  let textFrames = [];
  let interfaceFrames = [];
  let panelFrames = [];
  let cuts = [];
  let sceneCuts = null;
  let inferredCuts = [];
  let changeRows = [];
  let inspectedChangeRows = [];
  let graphicRows = [];
  let changesAt = intervalLookup();
  let graphicsAt = regionLookup();
  let pending;

  /** Keep instant measurements and give reviewed crops a short, shot-local preview. */
  function visibleChanges(at) {
    // FFmpeg timestamps are rounded to milliseconds, while the player uses fractional seconds.
    if (!toggle.checked) return [];
    const time = Math.round(at);
    const graphics = graphicsAt(time);
    if (!changeToggle?.checked) return graphics;
    const changes = previewChangeRegions(changesAt(time)?.regions, inspectedChangeRows,
                                         sceneCuts ?? (cuts.length ? cuts : inferredCuts), time);
    return graphics.concat(changes ?? []);
  }

  function paint(time = video.currentTime) {
    const width = video.clientWidth;
    const height = video.clientHeight;
    if (canvas.hidden || !video.videoWidth || width < 80 || !height) return;
    const ratio = window.devicePixelRatio || 1;
    const pixelsWide = Math.round(width * ratio);
    const pixelsHigh = Math.round(height * ratio);
    if (canvas.width !== pixelsWide || canvas.height !== pixelsHigh) {
      canvas.width = pixelsWide;
      canvas.height = pixelsHigh;
    }
    ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
    ctx.clearRect(0, 0, width, height);
    const timeMs = time * 1000;
    const [left, top, imageWidth, imageHeight] = imageBounds(video, width, height);
    const selected = sceneFrame && video.paused && Math.abs(timeMs - sceneFrame.timeMs) < 120 ? sceneFrame : null;
    ctx.translate(left, top);
    paintTechnicalFrame(ctx, evidence, phrases, sceneFrames, sceneCuts ?? (cuts.length ? cuts : inferredCuts), timeMs,
                        Math.round(imageWidth), Math.round(imageHeight), selected, textFrames, lookups, onFrame, panelFrames, objectTracks, visibleChanges(timeMs));
  }

  function schedule() {
    if (pending !== undefined || canvas.hidden || video.paused || video.ended) return;
    if (frameCallbacks) {
      pending = video.requestVideoFrameCallback((_, frame) => tick(frame.mediaTime));
    } else {
      pending = requestAnimationFrame(() => tick(video.currentTime));
    }
  }

  function tick(time) {
    pending = undefined;
    paint(time);
    schedule();
  }

  function cancel() {
    if (pending === undefined) return;
    if (frameCallbacks) video.cancelVideoFrameCallback(pending);
    else cancelAnimationFrame(pending);
    pending = undefined;
  }

  function update() {
    const hasVisualData = evidence?.timeline?.frames?.length || sceneFrame || sceneFrames.length || objectTracks?.rows.length ||
      textFrames.length || panelFrames.length || changeRows.length || graphicRows.length;
    const available = Boolean(ctx && hasVisualData && video.videoWidth);
    options.hidden = !available;
    if (changeToggle) {
      changeToggle.closest("label").hidden = !available || !changeRows.length;
      changeToggle.disabled = !toggle.checked;
    }
    canvas.hidden = !available || !toggle.checked;
    if (canvas.hidden) {
      cancel();
      onFrame?.(null);
    }
    else {
      paint();
      schedule();
    }
  }

  toggle.addEventListener("change", update);
  changeToggle?.addEventListener("change", update);
  for (const event of ["loadedmetadata", "emptied", "error"]) video.addEventListener(event, update);
  video.addEventListener("play", schedule);
  for (const event of ["seeked", "pause", "ended"]) {
    video.addEventListener(event, () => {
      paint();
      if (video.paused || video.ended) cancel();
    });
  }
  new ResizeObserver(() => paint()).observe(video);

  return {
    exportBody(source) {
      if (!video.videoWidth || !Number.isFinite(video.duration)) {
        throw new Error(uiText("Дождись загрузки предпросмотра."));
      }
      const [, , width, height] = imageBounds(video, video.clientWidth, video.clientHeight);
      const lastCue = phrases.reduce((end, cue) => Math.max(end, cue.endMs), 0);
      const duration = Math.max(lastCue, Math.round(video.duration * 1000));
      const plan = previewExportPlan(ctx, evidence, phrases, sceneFrames, sceneCuts ?? (cuts.length ? cuts : inferredCuts),
                                     Math.round(width), Math.round(height), duration, textFrames, panelFrames, objectClips,
                                     toggle.checked ? graphicRows.concat(changeToggle?.checked ? changeRows : []) : []);
      return packMedia(source, plan);
    },
    /** Use the monitor renderer on a frozen source frame and return its nearby UI geometry. */
    paintSnapshot(context, width, height, stamp) {
      const boundaries = sceneCuts ?? (cuts.length ? cuts : inferredCuts);
      paintTechnicalFrame(context, evidence, phrases, sceneFrames, boundaries, stamp, width, height,
                          null, textFrames, lookups, undefined, panelFrames, objectTracks, visibleChanges(stamp));
      const rows = sceneFrameAt(interfaceFrames, boundaries, stamp)?.interfaces ??
        sceneFrameAt(panelFrames, boundaries, stamp, PANEL_RADIUS_MS)?.interfaces ?? [];
      return rows.map(panel => {
        if (panel.elements?.length || !Number.isInteger(panel.readingTimeMs)) return panel;
        const index = after(interfaceFrames, panel.readingTimeMs, frame => frame.timeMs) - 1;
        const reading = interfaceFrames[index];
        if (reading?.timeMs !== panel.readingTimeMs) return panel;
        const prior = reading.interfaces.find(row => row.elements?.length &&
          coveredByPanel(row.box, [panel], .98) && coveredByPanel(panel.box, [row], .98));
        return prior ? {...panel, elements: prior.elements, connections: prior.connections} : panel;
      });
    },
    setSceneData(data) {
      const samples = data?.frames ?? [];
      changeRows = (data?.frameChanges?.rows ?? []).filter(row => Array.isArray(row?.regions) && row.regions.length);
      inspectedChangeRows = changeRows.flatMap(row => row.regions.filter(region => region.recognition)
        .map(region => ({startMs: region.recognition.sampleTimeMs ?? row.startMs,
                        endMs: region.recognition.verifiedUntilMs, regions: [region]})))
        .sort((first, second) => first.startMs - second.startMs);
      changesAt = intervalLookup(changeRows);
      graphicRows = (data?.frameChanges?.graphicRows ?? []).filter(row => Array.isArray(row?.regions) && row.regions.length);
      graphicsAt = regionLookup(graphicRows);
      sceneFrames = samples.filter(frame => frame.objectChecked !== false);
      textFrames = samples.filter(frame => frame.text?.length);
      interfaceFrames = samples.filter(frame => frame.interfaces?.length);
      panelFrames = samples.filter(frame => frame.panelChecked ||
        frame.interfaces?.some(panel => panel?.panelType === "mobile-ui" || panel?.panelType === "ui"));
      sceneCuts = data?.shotStartsMs ?? null;
      inferredCuts = sceneCuts === null ? samples.filter((frame, index) => index && frame.shot !== samples[index - 1].shot)
                                                .map(frame => frame.timeMs) : [];
      update();
    },
    setSceneFrame(frame) {
      sceneFrame = frame;
      update();
    },
    /** Replace the monitor data and reuse the object index across speech-only edits. */
    setData(nextEvidence, nextPhrases, nextObjectClips) {
      if (objectClips !== nextObjectClips) {
        objectClips = nextObjectClips;
        objectTracks = indexObjectTracks(objectClips);
      }
      evidence = nextEvidence;
      phrases = nextPhrases;
      lookups = frameLookups(evidence, phrases);
      cuts = (evidence?.editorTimeline?.shots ?? []).map(shot => shot.startMs).sort((a, b) => a - b);
      update();
    },
  };
}
