import test from "node:test";
import assert from "node:assert/strict";
import {buildTimeline, timelineIntervalLookup, timelineMarkerCenter} from "../web/video-timeline.js";
import {editorFaceBindings} from "../web/speaker-profiles.js";
import {objectLabelSummary} from "../web/video-object-profile.js";
import {layoutObjectAppearances} from "../web/video-object-appearance-timeline.js";

test("the timeline index returns overlapping clips in display order across seeks", () => {
  const lookup = timelineIntervalLookup([
    [1500, 2100, "reply"], [0, 5000, "shot"], [1000, 2000, "speech"],
    [2100, 2110, "next"], [2100, 2100, "empty"],
  ]);
  assert.deepEqual(lookup(1900), ["reply", "shot", "speech"]);
  assert.deepEqual(lookup(2100), ["shot", "next"]);
  assert.deepEqual(lookup(600), ["shot"]);
  assert.deepEqual(lookup(5000), []);
  assert.deepEqual(timelineIntervalLookup([])(0), []);
});

test("marker labels use the visible intersection only when a viewport edge clips the marker", () => {
  assert.equal(timelineMarkerCenter(100, 300, 120, 260), 89);
  assert.equal(timelineMarkerCenter(100, 200, 150, 220), 74);
  assert.equal(timelineMarkerCenter(100, 200, 40, 150), 24);
  assert.equal(timelineMarkerCenter(100, 200, 0, 250), null);
  assert.equal(timelineMarkerCenter(100, 200, 200, 260), null);
});

test("a conflicting cue photo cannot replace a verified speaker portrait", () => {
  const segments = [{startMs: 0, endMs: 1000, text: "Да", speaker: 1, visualFace: 4,
    speakerPhoto: {speaker: 1, jpeg: "other"}}];
  const evidence = {
    faceSpeakers: {3: {speaker: 1, basis: "audio-visual"}},
    editorTimeline: {facePhotos: {3: "verified"}},
    phrases: {0: {activePhoto: "other"}},
  };
  const {lanes} = buildTimeline({durationMs: 1000, segments}, evidence);
  assert.equal(lanes[0].photo, "verified");
  evidence.faceSpeakers[4] = {speaker: 1, basis: "audio-visual"};
  assert.deepEqual(editorFaceBindings(evidence, []).faceSpeakers, {});
});

test("shot boundaries, saved speaker corrections, silent avatars and mixed speech get their own lanes", () => {
  const data = {durationMs: 5000, segments: [
    {startMs: 500, endMs: 1500, text: "Привет", speaker: 1},
    {startMs: 1700, endMs: 2600, text: "Да", speaker: 1, speakerNumber: 2},
    {startMs: 3000, endMs: 4000, text: "Подожди", speaker: 1, overlap: true},
  ]};
  const evidence = {
    faceSpeakers: {2: {speaker: 2}, 3: {speaker: 3}},
    editorTimeline: {durationMs: 4800, shots: [{startMs: 0}, {startMs: 2200}], facePhotos: {3: "cGhvdG8="}},
  };
  const {duration, lanes} = buildTimeline(data, evidence);
  assert.equal(duration, 5000);
  assert.deepEqual(lanes.map(row => row.id), ["shots", "speaker-1", "speaker-2", "speaker-3", "overlap"]);
  assert.deepEqual(lanes[0].clips.map(clip => clip.endMs), [2200, 4800]);
  assert.equal(lanes[2].clips[0].text, "Да");
  assert.equal(lanes[3].photo, "cGhvdG8=");
  assert.deepEqual(lanes[3].clips, []);
  assert.equal(lanes[4].clips[0].text, "Подожди");
});

test("shot colors use the full scene timeline and retain matching thumbnails", () => {
  const data = {durationMs: 66000, segments: [], shotColors: [
    {startMs: 0, average: "#FF0000", sampleCount: 10},
    {startMs: 62000, average: "#0000FF", sampleCount: 12},
  ]};
  const evidence = {editorTimeline: {durationMs: 60000, shots: [{startMs: 0, jpeg: "dGVzdA=="}]}};
  const {lanes} = buildTimeline(data, evidence);
  assert.deepEqual(lanes[0].clips.map(clip => [clip.endMs, clip.average, clip.jpeg]), [
    [62000, "#FF0000", "dGVzdA=="], [66000, "#0000FF", undefined],
  ]);
});

test("object clips preserve graph IDs and concurrent samples have separate rows", () => {
  const model = buildTimeline({durationMs: 1000, segments: [], objectTimeline: [
    {id: "object_track:cup-a", startMs: 0, endMs: 401, label: "cup", sampleCount: 2, shotId: "shot:0"},
    {id: "object_track:cup-b", startMs: 0, endMs: 1, label: "cup", sampleCount: 1, shotId: "shot:0"},
    {id: "object_track:chair", startMs: 600, endMs: 1001, label: "chair", sampleCount: 2, shotId: "shot:500"},
  ]});
  const rows = model.lanes.filter(row => row.kind === "objects");
  assert.equal(rows.length, 2);
  assert.deepEqual(rows[0].clips.map(clip => [clip.id, clip.endMs]), [["object_track:cup-a", 401], ["object_track:chair", 1000]]);
  assert.equal(rows[1].clips[0].id, "object_track:cup-b");
  const old = buildTimeline({durationMs: 1000, segments: [], sceneEvidence: {frames: [
    {timeMs: 200, objects: [{label: "cup"}]},
  ]}});
  assert.equal(old.lanes[0].clips[0].startMs, 200);
  assert.equal(old.lanes[0].clips[0].endMs, 201);
});

test("object clips retain the scored frame and old clips resolve their own same-class detection", () => {
  const box = [.25, .35, .2, .3];
  const representative = {timeMs: 410, box, score: .95, observationId: "object_observation:410:0:boat"};
  const {lanes} = buildTimeline({durationMs: 1000, segments: [], sceneEvidence: {frames: [
    {timeMs: 200, objects: [{label: "boat", score: .49}, {label: "boat", box, score: .69}]},
  ]}, objectTimeline: [
    {id: "object_track:200:0:boat", startMs: 200, endMs: 411, label: "boat", representative},
    {id: "object_track:200:1:boat", startMs: 200, endMs: 201, label: "boat"},
  ]});
  assert.deepEqual(lanes[0].clips[0].representative, representative);
  assert.deepEqual(lanes[1].clips[0].representative, {timeMs: 200, box, score: .69});
});

test("related clips share a displayed profile name and retain their own detector label", () => {
  const profile = {id: "object_profile:table", label: "dining table", labels: []};
  const representative = {timeMs: 620, score: .324, box: [.01, .4, .98, .3]};
  const {lanes} = buildTimeline({durationMs: 1000, segments: [], objectProfiles: [profile], objectTimeline: [
    {id: "object_track:boat", startMs: 500, endMs: 621, label: "boat",
      profileId: profile.id, sampleCount: 2, representative},
  ]});
  const clip = lanes[0].clips[0];
  assert.equal(clip.text, "dining table");
  assert.equal(clip.label, "boat");
  assert.match(clip.description, /boat/);
  assert.equal(clip.representative, representative);
  assert.equal(clip.profile, profile);
});

test("all appearances remain ordered and link to their own measured observations", () => {
  const box = [.25, .35, .2, .3];
  const first = {timeMs: 120, observationId: "first", box};
  const second = {timeMs: 470, observationId: "second", box};
  const fromProfile = {timeMs: 820, observationId: "third", box};
  const profile = {id: "phone", label: "cell phone", appearance: {views: [fromProfile]}, appearances: [
    {startMs: 800, endMs: 850, representativeMs: 820, sampleCount: 2},
    {startMs: 400, endMs: 500, representativeMs: 470, sampleCount: 4},
    {startMs: 100, endMs: 200, representativeMs: 120, sampleCount: 3, representative: first},
    {startMs: 900, endMs: 1000, representativeMs: 1000, sampleCount: 1},
    {startMs: 1000, endMs: 1001, representativeMs: 1000},
  ]};
  const data = {durationMs: 1000, segments: [], objectProfiles: [profile], objectTimeline: [
    {startMs: 100, endMs: 200, label: "remote", profileId: profile.id, representative: first},
    {startMs: 400, endMs: 500, label: "cell phone", profileId: profile.id, representative: second},
  ]};
  const {lanes} = buildTimeline(data);
  const [clip, peer] = lanes[0].clips;
  assert.deepEqual(clip.appearances.map(item => [item.startMs, item.endMs, item.timeMs, item.sample, item.sampleCount]), [
    [100, 200, 120, first, 3], [400, 500, 470, second, 4],
    [800, 850, 820, fromProfile, 2], [900, 1000, 900, undefined, 1],
  ]);
  assert.equal(peer.appearances, clip.appearances);
  assert.equal(profile.appearances[0].startMs, 800);
});

test("legacy shared profiles derive appearances from measured track bounds without joining another object", () => {
  const phone = {id: "phone", label: "cell phone"};
  const otherPhone = {id: "another-phone", label: "cell phone"};
  const {lanes} = buildTimeline({durationMs: 3000, segments: [], objectProfiles: [phone, otherPhone], objectTimeline: [
    {startMs: 200, endMs: 300, label: phone.label, profileId: phone.id},
    {startMs: 400, endMs: 600, label: phone.label, profileId: otherPhone.id},
    {startMs: 2000, endMs: 2400, label: phone.label, profileId: phone.id, representative: {timeMs: 2200}},
  ]});
  assert.deepEqual(lanes[0].clips[0].appearances.map(item => item.timeMs), [200, 2200]);
  const otherClip = lanes.flatMap(lane => lane.clips).find(clip => clip.profile === otherPhone);
  assert.deepEqual(otherClip.appearances.map(item => item.timeMs), [400]);
});

test("each displayed detector label combines its measurements and keeps its highest confidence", () => {
  const labels = [
    {label: "cell phone", sampleCount: 3, score: .68, trackId: "first"},
    {label: "remote", sampleCount: 2, score: .37, trackId: "second"},
    {label: "cell phone", sampleCount: 6, score: .76, trackId: "third"},
  ];
  assert.deepEqual(objectLabelSummary(labels), [
    {label: "cell phone", sampleCount: 9, score: .76},
    {label: "remote", sampleCount: 2, score: .37},
  ]);
  assert.equal(labels[0].trackId, "first");
});

test("the object timeline uses the measured span and places each dot at its source frame", () => {
  const appearances = [
    {startMs: 1000, endMs: 1200, timeMs: 1100},
    {startMs: 1600, endMs: 1601, timeMs: 1600},
    {startMs: 1800, endMs: 2000, timeMs: 1900},
  ];
  const positions = layoutObjectAppearances(appearances);
  assert.deepEqual(positions.map(({hasDuration, left, right, dot}) => [hasDuration, left, right, dot]), [
    [true, 0, 20, 10], [false, 60, 60, 60], [true, 80, 100, 90],
  ]);
});

test("a single observed frame is centered without a false duration", () => {
  const entry = {startMs: 700, endMs: 701, timeMs: 700};
  assert.deepEqual(layoutObjectAppearances([entry]), [{entry, hasDuration: false, left: 50, right: 50, dot: 50}]);
  const span = {startMs: 700, endMs: 900, timeMs: 800};
  assert.deepEqual(layoutObjectAppearances([span]), [{entry: span, hasDuration: true, left: 0, right: 100, dot: 50}]);
  assert.deepEqual(layoutObjectAppearances([]), []);
});

test("music, pitch and concurrent ambience preserve independent seekable lanes", () => {
  const {lanes} = buildTimeline({segments: [], durationMs: 8000, audioLayers: [
    {kind: "music", label: "music", startMs: 100, endMs: 7000, confidence: .8},
    {kind: "background", label: "cricket_chirp", startMs: 2500, endMs: 4000, confidence: .62},
    {kind: "intonation", label: "rising", startMs: 5000, endMs: 6800, pitchHz: [120, 134, 150]},
    {kind: "background", label: "music", startMs: 100, endMs: 7000},
  ]});
  assert.deepEqual(lanes.map(lane => lane.kind), ["music", "intonation", "background"]);
  assert.equal(lanes[1].clips[0].marker, "↗");
  assert.match(lanes[1].clips[0].description, /120 → 134 → 150/);
  assert.equal(lanes[2].clips[0].cutMs, 3250);
  assert.equal(lanes[2].clips.length, 1);
});

test("camera and actor motion use separate timed lanes and retain relative depth", () => {
  const startMs = 100, endMs = 400;
  const {lanes} = buildTimeline({durationMs: 1000, segments: [], sceneEvidence: {cameraMotion: [
    {startMs, endMs, horizontal: "right", vertical: "still", zoom: "still"},
    {startMs: 900, endMs: 1100, horizontal: "left", vertical: "still", zoom: "still"},
  ]}}, {faceSpeakers: {7: {speaker: 1, basis: "audio-visual"}}, editorTimeline: {personMotion: [
    {startMs, endMs, faceId: 7, horizontal: "left", vertical: "still", depth: "foreground", basis: "camera-compensated"},
    {startMs, endMs, faceId: 9, horizontal: "unknown", vertical: "unknown", depth: "background", basis: "unknown"},
  ]}});
  const camera = lanes.find(lane => lane.kind === "camera-motion");
  assert.deepEqual(camera.clips.map(({startMs, endMs, marker}) => [startMs, endMs, marker]), [[100, 400, "→"]]);
  const people = lanes.filter(lane => lane.kind === "person-motion");
  assert.deepEqual(people.map(lane => lane.id), ["person-motion-7", "person-motion-9"]);
  assert.match(people[0].label, /1/);
  assert.equal(people[0].clips[0].marker, "← ●");
  assert.equal(people[1].clips[0].marker, "○");
});

test("a named speaker owns only the motion lanes bound to that speaker", () => {
  const motion = faceId => ({faceId, startMs: 50, endMs: 300, horizontal: "right", vertical: "still"});
  const data = {
    durationMs: 500,
    speakerProfiles: [{number: 2, name: "Максим"}],
    segments: [{speaker: 1, startMs: 0, endMs: 500, text: "Привет"}],
  };
  const evidence = {
    faceSpeakers: {3: {speaker: 2}, 4: {speaker: 2}, 6: {speaker: 1, basis: "visible-only"}},
    editorTimeline: {personMotion: [motion(3), motion(4), motion(6)]},
  };
  const {lanes} = buildTimeline(data, evidence);
  assert.deepEqual(lanes.map(lane => [lane.id, lane.number]), [
    ["speaker-1", 1], ["speaker-2", 2], ["person-motion-3", 2],
    ["person-motion-4", 2], ["expression-face-6", undefined], ["person-motion-6", undefined],
  ]);
  assert.equal(lanes[1].label, "Максим");
  assert.match(lanes[2].label, /Максим/);
});
