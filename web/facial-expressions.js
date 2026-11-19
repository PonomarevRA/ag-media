import {uiText} from "./i18n.js";

const LABELS = {
  angry: "Сердитое выражение", disgust: "Выражение отвращения", fearful: "Испуганное выражение",
  happy: "Радостное выражение", neutral: "Нейтральное выражение", sad: "Грустное выражение",
  surprised: "Удивлённое выражение",
};
const MARKERS = {angry: "😠", disgust: "🤢", fearful: "😨", happy: "🙂", neutral: "😐", sad: "🙁", surprised: "😮"};

/** Name the classifier's visible expression class without inferring a person's state. */
export const expressionLabel = label => uiText(LABELS[label] ?? "Выражение не определено");

/** Keep lane markers short; the complete class remains in the accessible name and tooltip. */
export const expressionMarker = label => MARKERS[label];

/** Retain only finite classifier estimates on exact, bounded source-time cells. */
export function expressionIntervals(layer, durationMs = Infinity) {
  const result = [];
  for (const row of Array.isArray(layer?.intervals) ? layer.intervals : []) {
    if (!row || !Object.hasOwn(LABELS, row.label) || !Number.isInteger(row.startMs)
        || !Number.isInteger(row.endMs) || row.startMs < 0 || row.endMs <= row.startMs
        || row.startMs >= durationMs || !Number.isInteger(row.faceId) || !Number.isInteger(row.track)
        || !Number.isFinite(row.confidence) || row.confidence < 0 || row.confidence > 1) continue;
    result.push({...row, endMs: Math.min(row.endMs, durationMs)});
  }
  return result.sort((a, b) => a.startMs - b.startMs || a.faceId - b.faceId || a.track - b.track);
}

/** Preserve each face/track's cells and gaps; speaker attribution comes only from current bindings. */
export function expressionLanes(evidence, durationMs) {
  const lanes = new Map();
  for (const row of expressionIntervals(evidence?.facialExpressions, durationMs)) {
    const id = `expression-${row.faceId}-${row.track}`;
    if (!lanes.has(id)) {
      const binding = evidence?.faceSpeakers?.[row.faceId];
      const number = binding?.basis !== "visible-only" && Number.isInteger(binding?.speaker) ? binding.speaker : undefined;
      lanes.set(id, {id, number, faceId: row.faceId, kind: "expression",
        label: uiText`Выражения · лицо ${row.faceId}`, clips: []});
    }
    const score = Math.round(row.confidence * 100);
    lanes.get(id).clips.push({...row, text: expressionLabel(row.label), marker: expressionMarker(row.label),
      description: uiText`Оценка классификатора: ${score}%`});
  }
  return [...lanes.values()];
}

/** Share one local collapse state across a speaker's rows or all tracks of an unbound face. */
export const expressionGroupKey = lane => lane.number ?? `face-${lane.faceId}`;

/** Group expressions and motion under one unbound face without guessing a speaker identity. */
export function unboundExpressionGroups(lanes, photos = {}, faceIds = []) {
  const groups = new Map();
  for (const faceId of faceIds) groups.set(faceId, []);
  for (const lane of lanes) {
    if (lane.number !== undefined) continue;
    if (!groups.has(lane.faceId)) groups.set(lane.faceId, []);
    groups.get(lane.faceId).push(lane);
  }
  const result = [];
  for (const [faceId, tracks] of [...groups].sort(([left], [right]) => left - right)) {
    result.push({id: `expression-face-${faceId}`, kind: "expression-face", faceId,
      label: uiText`Лицо ${faceId}`, photo: photos[faceId], clips: []}, ...tracks);
  }
  return result;
}

/** Attach separate face statistics so simultaneous faces never inflate a speaker-time denominator. */
export function profileExpressions(profile, evidence) {
  const layer = evidence?.facialExpressions;
  if (!layer) return undefined;
  const ids = new Set(profile.faceIds ?? []);
  if (profile.face?.id != null) ids.add(profile.face.id);
  const faces = [];
  const intervals = expressionIntervals(layer);
  for (const face of Array.isArray(layer.statistics?.faces) ? layer.statistics.faces : []) {
    if (!ids.has(face.faceId)) continue;
    const reactions = [];
    for (const reaction of Array.isArray(layer.reactions) ? layer.reactions : []) {
      const summary = reaction.faces?.find(item => item.faceId === face.faceId);
      if (summary) reactions.push({...reaction, faces: undefined, ...summary});
    }
    faces.push({...face, intervals: intervals.filter(row => row.faceId === face.faceId), reactions});
  }
  return {status: layer.status, model: layer.model, faces};
}

/** Page raw estimates with source times; old reports are explicitly unmeasured. */
export function expressionPage(report, {kind = "intervals", face_id, start_ms = 0, end_ms = Infinity,
  offset = 0, limit = 10} = {}) {
  const layer = report?.videoEvidence?.facialExpressions;
  const rows = kind === "samples" ? layer?.samples : expressionIntervals(layer);
  const filtered = [];
  for (const row of Array.isArray(rows) ? rows : []) {
    const start = kind === "samples" ? row?.timeMs : row?.startMs;
    const end = kind === "samples" ? start + 1 : row?.endMs;
    if (!Number.isFinite(start) || end <= start_ms || start >= end_ms || face_id !== undefined && row.faceId !== face_id) continue;
    if (kind === "samples") {
      const probabilities = {};
      for (const label of Object.keys(LABELS)) {
        const score = row.probabilities?.[label];
        if (Number.isFinite(score) && score >= 0 && score <= 1) probabilities[label] = score;
      }
      filtered.push({timeMs: start, track: Number.isInteger(row.track) ? row.track : null,
        faceId: Number.isInteger(row.faceId) ? row.faceId : null,
        label: Object.hasOwn(LABELS, row.label) ? row.label : "unknown",
        confidence: Number.isFinite(row.confidence) ? row.confidence : null,
        status: row.status === "ok" ? "ok" : "unknown", reason: String(row.reason ?? "").slice(0, 160), probabilities});
    } else {
      filtered.push({startMs: start, endMs: end, track: row.track, faceId: row.faceId, label: row.label, confidence: row.confidence});
    }
  }
  return page(layer, filtered, offset, limit);
}

/** Flatten question/face windows into bounded pages without attributing causation or feelings. */
export function reactionPage(report, {face_id, offset = 0, limit = 10} = {}) {
  const layer = report?.videoEvidence?.facialExpressions;
  const rows = [];
  for (const reaction of Array.isArray(layer?.reactions) ? layer.reactions : []) {
    for (const face of Array.isArray(reaction.faces) ? reaction.faces : []) {
      if (face_id !== undefined && face.faceId !== face_id) continue;
      rows.push({questionCue: reaction.questionCue, questionStartMs: reaction.questionStartMs,
        questionEndMs: reaction.questionEndMs, questionBasis: reaction.questionBasis ?? null,
        text: String(reaction.text ?? "").slice(0, 1000),
        windowStartMs: reaction.windowStartMs, windowEndMs: reaction.windowEndMs, faceId: face.faceId,
        ...boundedStatistics(face), dominantLabel: Object.hasOwn(LABELS, face.dominantLabel) ? face.dominantLabel : null,
        dominantShare: Number.isFinite(face.dominantShare) ? face.dominantShare : null});
    }
  }
  return page(layer, rows, offset, limit);
}

/** Return aggregate face-time separately from each face's visible-time denominator. */
export function expressionStatisticsPage(report, {offset = 0, limit = 10} = {}) {
  const layer = report?.videoEvidence?.facialExpressions;
  const statistics = layer?.statistics ?? {};
  const faces = [];
  for (const face of Array.isArray(statistics.faces) ? statistics.faces : []) {
    faces.push({faceId: face.faceId, ...boundedStatistics(face)});
  }
  return {...page(layer, faces, offset, limit), statistics: boundedStatistics(statistics)};
}

/** Bound the class map to this model's seven labels and retain explicit time denominators. */
function boundedStatistics(source) {
  const result = {};
  for (const key of ["durationMs", "observedMs", "classifiedMs", "unknownMs", "coverage", "observedFaceMs", "classifiedFaceMs"]) {
    if (Number.isFinite(source[key])) result[key] = source[key];
  }
  result.byLabel = {};
  for (const label of Object.keys(LABELS)) {
    const value = source.byLabel?.[label];
    if (Number.isFinite(value?.durationMs) && Number.isFinite(value?.share)) {
      result.byLabel[label] = {durationMs: value.durationMs, share: value.share};
    }
  }
  return result;
}

/** Keep shared provenance and page continuation explicit for each read. */
function page(layer, rows, offset, limit) {
  const items = rows.slice(offset, offset + limit);
  const next = offset + items.length;
  return {status: layer?.status ?? "not-measured", basis: layer?.basis ?? null, model: layer?.model ?? null,
    version: layer?.version ?? null, sampleStepMs: layer?.sampleStepMs ?? null, confidenceThreshold: layer?.confidenceThreshold ?? null,
    estimate: "visible-expression-classifier; not mental state, intent, truthfulness or causation",
    total: rows.length, items, nextOffset: next < rows.length ? next : null};
}
