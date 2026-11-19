import {uiText} from "./i18n.js";

const DIRECTIONS = {
  camera: {label: "В сторону камеры", icon: "◎"},
  left: {label: "Влево", icon: "←"},
  right: {label: "Вправо", icon: "→"},
  center: {label: "Глаза по центру", icon: "•"},
};

export const gazeLabel = direction => uiText(DIRECTIONS[direction].label);
export const gazeIcon = direction => DIRECTIONS[direction].icon;

/** Distinguish a frontal, centered sample from centered eyes on a turned head. */
function directionOf({gaze, headPose}) {
  const direction = gaze?.horizontal;
  if (direction === "left" || direction === "right") return direction;
  if (direction !== "center") return null;
  const frontal = [headPose?.yawDeg, headPose?.pitchDeg].every(angle => Number.isFinite(angle) && Math.abs(angle) <= 12);
  return frontal ? "camera" : "center";
}

/** Bound a sample to the visible run of the same track, with a 250 ms maximum hold. */
function visibleEnd(timeline, track, stamp, end) {
  if (!timeline?.frames?.length || !(timeline.stepMs > 0)) return end;
  const first = Math.floor(stamp / timeline.stepMs);
  for (let index = first; index * timeline.stepMs < end; index++) {
    if (!timeline.frames[index]?.some(hit => hit[0] === track)) return Math.max(stamp, index * timeline.stepMs);
  }
  return end;
}

/** Convert exact, unambiguous speaker face observations into non-overlapping gaze intervals. */
function buildGaze(evidence, durationMs, byFace) {
  const result = new Map();
  const rows = evidence?.personLayer?.frames ?? [];
  const bindings = evidence?.faceSpeakers ?? {};
  const shots = evidence?.editorTimeline?.shots ?? [];
  let cut = 0;
  for (const [index, row] of rows.entries()) {
    const startMs = row.timeMs;
    if (!Number.isInteger(startMs) || startMs < 0 || startMs >= durationMs) continue;
    while (cut < shots.length && shots[cut].startMs <= startMs) cut++;
    const end = Math.min(startMs + 250, rows[index + 1]?.timeMs ?? Infinity,
                         shots[cut]?.startMs ?? Infinity, durationMs);
    const assigned = new Map();
    for (const person of row.people) {
      const binding = bindings[person.faceId];
      if (!byFace && (!binding || binding.basis === "visible-only" || !binding.speaker)) continue;
      const speaker = byFace ? person.faceId : binding.speaker;
      if (!Number.isInteger(speaker) || speaker < 1) continue;
      assigned.set(speaker, assigned.has(speaker) ? null : person);
    }
    for (const [speaker, person] of assigned) {
      if (!person) continue;
      const direction = directionOf(person);
      const endMs = visibleEnd(evidence.timeline, person.track, startMs, end);
      if (!direction || endMs <= startMs) continue;
      if (!result.has(speaker)) result.set(speaker, {observedMs: 0, directionsMs: {}, intervals: []});
      const summary = result.get(speaker);
      summary.observedMs += endMs - startMs;
      summary.directionsMs[direction] = (summary.directionsMs[direction] ?? 0) + endMs - startMs;
      const last = summary.intervals.at(-1);
      if (last?.endMs === startMs && last.direction === direction && last.track === person.track && last.shot === cut) {
        last.endMs = endMs;
      } else {
        summary.intervals.push({startMs, endMs, direction, track: person.track, shot: cut});
      }
    }
  }
  return result;
}

/** Summarize only faces that have a current, unambiguous speech association. */
export const buildSpeakerGaze = (evidence, durationMs) => buildGaze(evidence, durationMs, false);

/** Summarize a visible face independently of whether its voice has been identified. */
export const buildFaceGaze = (evidence, durationMs) => buildGaze(evidence, durationMs, true);
