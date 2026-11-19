import assert from "node:assert/strict";
import {test} from "node:test";
import {buildSpeakerGaze} from "../web/speaker-gaze.js";
import {buildTimeline} from "../web/video-timeline.js";

const visible = [7, 0, 0, .3, .4, 0, 2];
const person = (horizontal, headPose) => ({track: 7, faceId: 2, gaze: {horizontal}, headPose});
const frontal = {yawDeg: 3, pitchDeg: -2};

test("gaze time is clipped to cuts and visible samples, with a distinct camera estimate", () => {
  const evidence = {
    faceSpeakers: {"2": {speaker: 4, basis: "audio-visual"}},
    personLayer: {frames: [
      {timeMs: 0, people: [person("center", frontal)]},
      {timeMs: 250, people: [person("center", {yawDeg: 30, pitchDeg: 0})]},
      {timeMs: 500, people: [person("left", frontal)]},
    ]},
    editorTimeline: {durationMs: 800, shots: [{startMs: 0}, {startMs: 210}]},
    timeline: {stepMs: 40, frames: Array.from({length: 20}, (_, i) => i < 15 ? [visible] : [])},
  };
  const map = buildSpeakerGaze(evidence, 800);
  const stats = map.get(4);
  assert.deepEqual(stats.directionsMs, {camera: 210, center: 250, left: 100});
  assert.equal(stats.observedMs, 560);
  const model = buildTimeline({durationMs: 800, segments: []}, evidence, map);
  assert.deepEqual(model.lanes.map(lane => lane.id), ["shots", "speaker-4", "gaze-4"]);
});

test("unassigned and conflicting face samples yield no speaker gaze", () => {
  const evidence = {
    faceSpeakers: {"2": {speaker: 4, basis: "visible-only"}, "3": {speaker: 1, basis: "editorial"},
                   "4": {speaker: 1, basis: "audio-visual"}},
    personLayer: {frames: [{timeMs: 0, people: [
      person("center", frontal), {...person("center", frontal), faceId: 3},
      {...person("left", frontal), faceId: 4},
    ]}]},
  };
  assert.equal(buildSpeakerGaze(evidence, 300).size, 0);
});
