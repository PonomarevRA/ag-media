import test from "node:test";
import assert from "node:assert/strict";
import {buildVideoStatistics, transcriptWords} from "../web/video-statistics.js";

test("top words retain literal forms, multilingual letters and an explicit common-word filter", () => {
  const cues = [{text: "И кот, КОТ! The cat and cat. café বাংলা привет-привет"}];
  const words = transcriptWords(cues);
  assert.equal(words.total, 10);
  assert.deepEqual(words.top.slice(0, 2).map(row => row.count), [2, 2]);
  assert.ok(words.top.some(row => row.word === "বাংলা"));
  assert.ok(!words.top.some(row => row.word === "the"));
  assert.ok(transcriptWords(cues, false).top.some(row => row.word === "the"));
});

test("multiple tracks of one profile count once while two concurrent cups stay distinct", () => {
  const report = {durationMs: 3000, segments: [{speaker: 1, startMs: 0, endMs: 1000, text: "Да"}], objectTimeline: [
    {id: "track-a", profileId: "cup-a", startMs: 0, endMs: 1200},
    {id: "track-b", profileId: "cup-a", startMs: 700, endMs: 1800},
    {id: "track-c", profileId: "cup-b", startMs: 0, endMs: 1000},
  ], objectProfiles: [{id: "cup-a"}, {id: "cup-b"}]};
  const stats = buildVideoStatistics(report);
  assert.equal(stats.counts.objects, 2);
  assert.deepEqual(stats.entities.find(row => row.id === "objects").values, [2, 1, 0]);
  assert.equal(stats.counts.people, null);
});

test("strongest overlapping expression per face wins and person-time can exceed source time", () => {
  const cell = (faceId, label, confidence, startMs = 0, endMs = 1000) => ({faceId, track: faceId, label, confidence, startMs, endMs});
  const stats = buildVideoStatistics({durationMs: 1000, segments: [], videoEvidence: {facialExpressions: {intervals: [
    cell(1, "neutral", .5), cell(1, "happy", .9, 500), cell(2, "sad", .8),
    cell(1, "happy", .9, 500), cell(2, "invalid", 1),
  ]}}});
  assert.equal(stats.expressions.measuredMs, 2000);
  assert.equal(stats.counts.expressionEpisodes, 3);
  assert.deepEqual(Object.fromEntries(stats.expressions.summary.map(row => [row.label, row.ms])), {neutral: 500, happy: 500, sad: 1000});
  assert.equal(stats.expressions.summary.reduce((sum, row) => sum + row.share, 0), 1);
});

test("face IDs survive track changes, figures are a peak, and overlap adds speaking participants", () => {
  const stats = buildVideoStatistics({durationMs: 3000, segments: [{text: "Together", speaker: 1, overlap: true, overlaps: [{speakers: [1, 2]}]}],
    videoEvidence: {timeline: {stepMs: 1000, frames: [[[4, 0, 0, .2, .2, 0, 1]], [[8, 0, 0, .2, .2, 0, 1]], []]},
      personLayer: {frames: [{timeMs: 0, people: [{faceId: 1, box: [0, 0, 1, 1]}, {box: [0, 0, .2, .3]}]}]}}});
  assert.equal(stats.counts.faces, 1);
  assert.equal(stats.counts.people, 2);
  assert.equal(stats.counts.speakers, 2);
  assert.deepEqual(stats.entities.find(row => row.id === "people").values, [2, null, null]);
});

test("manual speaker assignment overrides acoustic overlap without losing the original evidence on reset", () => {
  const cue = {speaker: 9, speakerNumber: 3, overlap: true, overlaps: [{speakers: [1, 2]}]};
  const report = {durationMs: 1000, segments: [cue]};
  assert.equal(buildVideoStatistics(report).counts.speakers, 1);
  delete cue.speakerNumber;
  assert.equal(buildVideoStatistics(report).counts.speakers, 2);
  assert.equal(cue.speaker, 9);
});

test("gaze uses existing source-time cells and keeps unmeasured intervals as gaps", () => {
  const stats = buildVideoStatistics({durationMs: 3000, segments: [], videoEvidence: {
    faceSpeakers: {1: {speaker: 1}}, timeline: {stepMs: 250, frames: [[[1]], [[1]]]},
    personLayer: {frames: [{timeMs: 0, people: [{faceId: 1, track: 1, gaze: {horizontal: "center"}, headPose: {yawDeg: 0, pitchDeg: 0}}]}]},
  }});
  assert.equal(stats.gaze.measuredMs, 250);
  assert.deepEqual(stats.gaze.series, [{id: "camera", values: [.25, null, null]}]);
});

test("empty/partial reports expose missing layers and two-hour reports keep a bounded chart model", () => {
  const empty = buildVideoStatistics({segments: [], analysisComplete: false});
  assert.equal(empty.partial, true);
  assert.deepEqual(empty.bins, []);
  assert.equal(empty.counts.expressions, null);
  assert.equal(empty.counts.objects, null);
  const long = buildVideoStatistics({durationMs: 7200000, segments: []}, {maxBins: 100000});
  assert.equal(long.bins.length, 120);
  assert.equal(long.bins.at(-1).endMs, 7200000);
  assert.ok(long.entities.every(row => row.values.length === 120));
});

test("a measured class switch is zero for the absent class while a truly unmeasured bin stays null", () => {
  const stats = buildVideoStatistics({durationMs: 3000, segments: [], videoEvidence: {
    facialExpressions: {intervals: [
      {faceId: 1, track: 1, label: "happy", confidence: .9, startMs: 0, endMs: 1000},
      {faceId: 1, track: 1, label: "neutral", confidence: .9, startMs: 1000, endMs: 2000},
    ]},
    faceSpeakers: {1: {speaker: 1}}, timeline: {stepMs: 1000, frames: [[[1]], [[1]], []]},
    personLayer: {frames: [
      {timeMs: 0, people: [{faceId: 1, track: 1, gaze: {horizontal: "left"}}]},
      {timeMs: 1000, people: [{faceId: 1, track: 1, gaze: {horizontal: "right"}}]},
    ]},
  }});
  assert.deepEqual(stats.expressions.series, [{id: "happy", values: [1, 0, null]}, {id: "neutral", values: [0, 1, null]}]);
  assert.deepEqual(stats.gaze.series, [{id: "left", values: [.25, 0, null]}, {id: "right", values: [0, .25, null]}]);
});
