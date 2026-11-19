import test from "node:test";
import assert from "node:assert/strict";
import {applyReportEvidenceLayers} from "../web/media-transcript.js";
import {editorFaceBindings, speakerProfiles} from "../web/speaker-profiles.js";
import {buildTimeline} from "../web/video-timeline.js";

/** Reproduce the saved-report boundary: a bound face has real expression measurements and a scene. */
function savedReport() {
  const face = {faceId: 5, observedMs: 480, classifiedMs: 480, unknownMs: 0,
    byLabel: {happy: {durationMs: 480, share: 1}}};
  return {durationMs: 5000, segments: [{startMs: 2000, endMs: 3000, text: "Да", speaker: 2, visualFace: 5}],
    speakerProfileIndex: [{id: "profile-2", number: 2, faceId: 5}],
    videoEvidence: {faceSpeakers: {5: {speaker: 2, basis: "audio-visual"}}, timeline: {stepMs: 40, frames: []},
      facialExpressions: {status: "ok", intervals: [{startMs: 2200, endMs: 2680,
        faceId: 5, track: 9, label: "happy", confidence: .9}], statistics: {faces: [face]}}},
    sceneEvidence: {status: "ok", frames: [{timeMs: 2200, objects: [{label: "cup"}]}]}};
}

test("cold saved-report restore feeds identical expression identity to profile and timeline plus scene", () => {
  const report = savedReport();
  const before = structuredClone(report);
  const state = {};
  const calls = [];
  applyReportEvidenceLayers(report, state, {
    videoEvidence: value => calls.push(["video", value]), sceneEvidence: value => calls.push(["scene", value]),
  });
  assert.equal(state.videoEvidence, report.videoEvidence);
  assert.equal(state.sceneEvidence, report.sceneEvidence);
  assert.deepEqual(calls.map(([kind]) => kind), ["video", "scene"]);
  const displayed = editorFaceBindings(state.videoEvidence, report.segments);
  const profiles = speakerProfiles(report.segments, displayed, report.speakerProfileIndex);
  const speaker = profiles.find(profile => profile.number === 2);
  assert.equal(speaker.face.id, 5);
  assert.equal(speaker.facialExpressions.faces[0].faceId, 5);
  assert.equal(speaker.facialExpressions.faces[0].byLabel.happy.share, 1);
  const timeline = buildTimeline(report, displayed);
  const expression = timeline.lanes.find(lane => lane.kind === "expression");
  assert.equal(expression.number, 2);
  assert.equal(expression.clips[0].startMs, 2200);
  assert.equal(expression.clips[0].label, "happy");
  assert.deepEqual(report, before);
});

test("live cumulative reports reuse equal layers and absent layers clear previous source observations", () => {
  const report = savedReport();
  const state = {};
  const calls = [];
  const setters = {videoEvidence: value => calls.push(["video", value]), sceneEvidence: value => calls.push(["scene", value])};
  applyReportEvidenceLayers(report, state, setters);
  applyReportEvidenceLayers({...report, segments: [...report.segments]}, state, setters);
  assert.equal(calls.length, 2);
  applyReportEvidenceLayers({segments: []}, state, setters);
  assert.deepEqual(state, {videoEvidence: null, sceneEvidence: null});
  assert.deepEqual(calls.slice(2), [["video", null], ["scene", null]]);
});
