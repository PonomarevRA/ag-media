import {effectiveSpeaker, mixedSpeakerLabel} from "./speaker-label.js";
import {expressionIntervals} from "./facial-expressions.js";
import {buildSpeakerGaze} from "./speaker-gaze.js";

const STOP_WORDS = new Set(("и в во не на но что это как а я ты он она мы вы они с со у к ко по за от до из для о об же ли бы то так да нет ну вот там тут был была были быть есть его ее их мне вам нам тебя меня свой this that the a an and or but to of in on at for from with is are was were be been it its i you he she we they my your our their not no yes so as if then than have has had do does did can could will would just" ).split(/\s+/u));
const array = value => Array.isArray(value) ? value : [];
const validTime = value => Number.isFinite(value) && value >= 0;

/** Count literal words case-insensitively; filtering never changes the transcript or invents lemmas. */
export function transcriptWords(segments, excludeCommon = true) {
  const counts = new Map();
  let total = 0;
  for (const cue of array(segments)) {
    const tokens = String(cue.text ?? "").normalize("NFC").toLocaleLowerCase().match(/[\p{L}\p{M}]+(?:['’−-][\p{L}\p{M}]+)*/gu) ?? [];
    for (const word of tokens) {
      total++;
      if (excludeCommon && (STOP_WORDS.has(word) || [...word].length < 2)) continue;
      counts.set(word, (counts.get(word) ?? 0) + 1);
    }
  }
  const top = [...counts].map(([word, count]) => ({word, count}));
  top.sort((a, b) => b.count - a.count || a.word.localeCompare(b.word));
  return {total, top: top.slice(0, 10)};
}

/** Split overlapping estimates by face; at each instant retain only its strongest active class. */
function faceExpressionCells(layer, duration) {
  const faces = new Map();
  for (const row of expressionIntervals(layer, duration)) {
    if (!faces.has(row.faceId)) faces.set(row.faceId, []);
    faces.get(row.faceId).push(row);
  }
  const result = [];
  for (const rows of faces.values()) {
    const events = [];
    for (const row of rows) events.push({time: row.startMs, row, add: true}, {time: row.endMs, row, add: false});
    events.sort((a, b) => a.time - b.time);
    const active = new Set();
    let best, cursor = 0;
    for (let i = 0; i < events.length;) {
      const stamp = events[i].time;
      if (best && stamp > cursor) {
        const previous = result.at(-1);
        if (previous?.faceId === best.faceId && previous.label === best.label && previous.endMs === cursor) previous.endMs = stamp;
        else result.push({...best, startMs: cursor, endMs: stamp});
      }
      while (i < events.length && events[i].time === stamp) {
        const {row, add} = events[i++];
        if (add) active.add(row);
        else active.delete(row);
      }
      best = null;
      for (const row of active) if (!best || row.confidence > best.confidence) best = row;
      cursor = stamp;
    }
  }
  return result;
}

/** Integrate measured person-time into bounded bins; gaps remain null rather than inferred neutral states. */
function durationSeries(cells, key, bins) {
  const totals = new Map();
  const series = new Map();
  const measured = bins.map(() => false);
  for (const cell of cells) {
    if (!validTime(cell.startMs) || !validTime(cell.endMs) || cell.endMs <= cell.startMs) continue;
    const label = cell[key];
    if (!series.has(label)) series.set(label, bins.map(() => null));
    const values = series.get(label);
    for (let i = 0; i < bins.length; i++) {
      const ms = Math.max(0, Math.min(cell.endMs, bins[i].endMs) - Math.max(cell.startMs, bins[i].startMs));
      if (!ms) continue;
      measured[i] = true;
      values[i] = (values[i] ?? 0) + ms / 1000;
      totals.set(label, (totals.get(label) ?? 0) + ms);
    }
  }
  for (const values of series.values()) {
    for (let i = 0; i < bins.length; i++) if (measured[i] && values[i] === null) values[i] = 0;
  }
  const measuredMs = [...totals.values()].reduce((a, b) => a + b, 0);
  return {measuredMs, summary: [...totals].map(([label, ms]) => ({label, ms, share: ms / measuredMs})),
    series: [...series].map(([id, values]) => ({id, values}))};
}

/** Take a peak over sampled frames in each bin, retaining explicit unobserved bins. */
function sampledCounts(rows, bins, count) {
  const values = bins.map(() => null);
  let index = 0;
  for (const row of rows) {
    if (!validTime(row.timeMs)) continue;
    while (index < bins.length && row.timeMs >= bins[index].endMs) index++;
    if (index === bins.length) break;
    if (row.timeMs < bins[index].startMs) continue;
    values[index] = Math.max(values[index] ?? 0, count(row));
  }
  return values;
}

/** Read only the published report: profiles are unique IDs, people are simultaneous sampled figures. */
export function buildVideoStatistics(report, {excludeCommon = true, maxBins = 60} = {}) {
  const cues = array(report?.segments);
  const evidence = report?.videoEvidence ?? {};
  let durationMs = Math.max(0, report?.durationMs ?? 0, evidence.editorTimeline?.durationMs ?? 0);
  for (const cue of cues) if (validTime(cue.endMs)) durationMs = Math.max(durationMs, cue.endMs);
  if (!Number.isFinite(durationMs)) durationMs = 0;
  const count = durationMs ? Math.min(120, Math.max(1, Math.floor(maxBins) || 60), Math.ceil(durationMs / 1000)) : 0;
  const bins = Array.from({length: count}, (_, i) => ({startMs: durationMs * i / count, endMs: durationMs * (i + 1) / count}));
  const speakers = new Set();
  for (const cue of cues) {
    if (mixedSpeakerLabel(cue)) {
      for (const overlap of array(cue.overlaps)) for (const id of array(overlap.speakers)) if (Number.isInteger(id) && id > 0) speakers.add(id);
    } else {
      const speaker = effectiveSpeaker(cue);
      if (speaker !== null) speakers.add(speaker);
    }
  }
  const faces = new Set(Object.keys(evidence.editorTimeline?.facePhotos ?? {}));
  for (const id of Object.keys(evidence.faceSpeakers ?? {})) faces.add(String(id));
  const peopleRows = array(evidence.personLayer?.frames);
  for (const row of peopleRows) for (const person of array(row.people)) if (person.faceId != null) faces.add(String(person.faceId));
  const faceRows = array(evidence.timeline?.frames).map((hits, i) => ({timeMs: i * evidence.timeline.stepMs, hits: array(hits)}));
  for (const row of faceRows) for (const hit of row.hits) if (hit[6] ?? hit[0]) faces.add(String(hit[6] ?? hit[0]));
  const faceCounts = sampledCounts(faceRows, bins, row => row.hits.length);
  const peopleCounts = sampledCounts(peopleRows, bins, row => array(row.people).filter(p => Array.isArray(p.box)).length);
  const objectTracks = array(report?.objectTimeline);
  const objectIds = new Set();
  for (const track of objectTracks) if (track.profileId ?? track.id) objectIds.add(track.profileId ?? track.id);
  for (const profile of array(report?.objectProfiles)) if (profile.id) objectIds.add(profile.id);
  // A profile can have many tracks. Count its presence only once in a time slice, including overlaps.
  const objectCounts = bins.map(bin => {
    if (!Array.isArray(report?.objectTimeline)) return null;
    const ids = new Set();
    for (const track of objectTracks) if (track.startMs < bin.endMs && track.endMs > bin.startMs && (track.profileId ?? track.id)) ids.add(track.profileId ?? track.id);
    return ids.size;
  });
  const expressionCells = faceExpressionCells(evidence.facialExpressions, durationMs);
  const expressions = durationSeries(expressionCells, "label", bins);
  const gazeCells = [];
  for (const gaze of buildSpeakerGaze(evidence, durationMs).values()) gazeCells.push(...gaze.intervals);
  const gaze = durationSeries(gazeCells, "direction", bins);
  return {durationMs, partial: report?.analysisComplete === false, bins, words: transcriptWords(cues, excludeCommon),
    counts: {speakers: speakers.size, faces: evidence.timeline || faces.size ? faces.size : null,
      people: peopleRows.length ? Math.max(0, ...peopleCounts.filter(v => v !== null)) : null,
      objects: Array.isArray(report?.objectTimeline) || Array.isArray(report?.objectProfiles) ? objectIds.size : null,
      expressions: expressionCells.length ? new Set(expressionCells.map(row => row.label)).size : null,
      expressionEpisodes: evidence.facialExpressions?.status === "completed" || expressionCells.length ? expressionCells.length : null},
    entities: [{id: "faces", values: faceCounts}, {id: "people", values: peopleCounts}, {id: "objects", values: objectCounts}],
    expressions, gaze};
}
