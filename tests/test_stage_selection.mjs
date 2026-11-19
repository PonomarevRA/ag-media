import test from "node:test";
import assert from "node:assert/strict";
import {stageSelectionQuery, retainedSpeakerEdits, speakerEditKey} from "../web/analysis-stage-selection.js";
import {objectProfileViews} from "../web/video-object-profile.js";

test("omitted selection keeps defaults, while an empty selection explicitly requests no stage", () => {
  assert.deepEqual(stageSelectionQuery(null), {});
  assert.deepEqual(stageSelectionQuery([]), {selected_stages: ""});
  assert.deepEqual(stageSelectionQuery(["scene", "object_review"], true),
    {selected_stages: "scene,object_review", force_stages: "scene,object_review"});
});

test("stage transport rejects unknown, duplicate and non-array selections", () => {
  for (const invalid of [["ffmpeg"], ["scene", "scene"], "scene", {}, undefined]) {
    assert.throws(() => stageSelectionQuery(invalid));
  }
});

test("source gallery keeps exact independent observations and merges only duplicate observation IDs", () => {
  const first = {observationId: "one", timeMs: 400, box: [0, 0, .5, .5], palette: [{hex: "#ff0000"}]};
  const different = {observationId: "two", timeMs: 400, box: [.5, 0, .5, .5]};
  const legacy = {observationId: "one", timeMs: 400, score: .8};
  const profile = {objectGallery: {views: [first, different, {timeMs: -1}]}, appearance: {views: [legacy]}};
  const views = objectProfileViews(profile);
  assert.equal(views.length, 2);
  assert.equal(views[0].score, .8);
  assert.deepEqual(views[0].box, first.box);
  assert.deepEqual(views[0].palette, first.palette);
  assert.equal(views[1].observationId, "two");
  assert.deepEqual(profile.objectGallery.views[0], first);
});

test("retained-source reruns keep user speaker choices only on the exact unchanged cue", () => {
  const manual = {startMs: 0, endMs: 1000, text: "Hello", speakerNumber: 2, speakerNumberEvidence: {source: "user"}};
  const auto = {...manual, startMs: 1000, speakerNumberEvidence: {source: "light-asd"}};
  const edits = retainedSpeakerEdits({segments: [manual, auto]});
  assert.equal(edits.size, 1);
  assert.equal(edits.get(speakerEditKey({...manual})), "2");
  assert.equal(edits.has(speakerEditKey({...manual, endMs: 1100})), false);
  assert.equal(edits.has(speakerEditKey({...manual, text: "Different cue"})), false);
  assert.equal(retainedSpeakerEdits(null).size, 0);
});
