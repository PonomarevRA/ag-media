import test from "node:test";
import assert from "node:assert/strict";
import {buildTimeline} from "../web/video-timeline.js";
import {faceProfile, speakerProfiles} from "../web/speaker-profiles.js";
import {buildFaceGaze, buildSpeakerGaze} from "../web/speaker-gaze.js";

test("motion and expressions belong to the same face header; motion-only and photo-only faces remain inspectable", () => {
  const evidence = {faceSpeakers: {17: {speaker: 2, basis: "visible-only"}},
    editorTimeline: {facePhotos: {17: "photo", 20: "photo"}, personMotion: [
      {faceId: 17, startMs: 0, endMs: 250}, {faceId: 19, startMs: 0, endMs: 250},
    ]}, facialExpressions: {intervals: [
      {faceId: 17, track: 8, startMs: 0, endMs: 250, label: "neutral", confidence: .9},
    ]}};
  const before = structuredClone(evidence);
  const {lanes} = buildTimeline({durationMs: 300, segments: []}, evidence);
  assert.deepEqual(lanes.map(lane => lane.id), ["expression-face-17", "expression-17-8", "person-motion-17",
    "expression-face-19", "person-motion-19", "expression-face-20"]);
  assert.equal(lanes[2].faceId, 17);
  assert.equal(lanes[2].number, undefined);
  assert.deepEqual(evidence, before);
});

test("a face profile keeps exact attributes and provenance without creating a speech identity", () => {
  const observations = {age: 31, ageSource: "measured-model", ageSamples: 3,
    appearanceClass: "male", hairSwatch: "#33282b", sampleTimesMs: [100, 300, 500]};
  const evidence = {editorTimeline: {faceAttributes: {17: observations},
    clothingByFace: {17: {swatch: "#ff0000", pattern: "plain"}}}};
  const profile = faceProfile(17, evidence, [{faceId: 18, appearance: {age: 60}, clothing: {swatch: "#ffffff"}}]);
  assert.equal(profile.name, "Лицо 17");
  assert.equal(profile.number, undefined);
  assert.equal(profile.face.basis, "visual");
  assert.deepEqual(profile.face.observations, observations);
  assert.equal(profile.clothing.swatch, "#ff0000");
  assert.equal(profile.reference, undefined);
  assert.deepEqual(speakerProfiles([], evidence), []);
});

test("legacy face profiles use only an exact saved face reference and preserve missing attributes", () => {
  const index = [{id: "saved", faceId: 17, pinned: true, appearance: {age: 33, hairSwatch: "#112233"}}];
  const evidence = {editorTimeline: {faceAttributes: {17: {age: 31}}}};
  assert.equal(faceProfile(17, evidence, index).face.observations.age, 31);
  assert.equal(faceProfile(17, evidence, index).face.observations.hairSwatch, "#112233");
  assert.equal(faceProfile(17, evidence, index).reference.id, "saved");
  const missing = faceProfile(18, evidence, index);
  assert.deepEqual(missing.face.observations, {});
  assert.equal(missing.clothing, undefined);
  assert.equal(faceProfile("17", evidence), undefined);
  assert.equal(faceProfile(17, {}, [{faceId: 17, age: 35}]).face.observations.age, 35);
  assert.deepEqual(faceProfile(17, {}, [{faceId: 17, age: 35}, {faceId: 17, age: 60}]).face.observations, {});
});

test("face gaze preserves simultaneous independent observations without attributing speech", () => {
  const person = (faceId, track, horizontal) => ({faceId, track, gaze: {horizontal}, headPose: {yawDeg: 0, pitchDeg: 0}});
  const evidence = {faceSpeakers: {17: {speaker: 2, basis: "visible-only"}}, personLayer: {frames: [
    {timeMs: 0, people: [person(17, 1, "left"), person(19, 2, "center")]},
    {timeMs: 250, people: [person(17, 1, "right"), person(19, 2, "center")]},
  ]}};
  const gaze = buildFaceGaze(evidence, 400);
  assert.deepEqual(gaze.get(17).directionsMs, {left: 250, right: 150});
  assert.deepEqual(gaze.get(19).directionsMs, {camera: 400});
  assert.equal(buildSpeakerGaze(evidence, 400).size, 0);
  assert.deepEqual(faceProfile(17, evidence, [], 400).gaze, gaze.get(17));
});
