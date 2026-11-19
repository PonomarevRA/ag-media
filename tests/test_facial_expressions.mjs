import test from "node:test";
import assert from "node:assert/strict";
import {expressionIntervals, expressionLanes, expressionGroupKey, unboundExpressionGroups, profileExpressions, expressionPage, reactionPage} from "../web/facial-expressions.js";
import {buildTimeline} from "../web/video-timeline.js";
import {editorFaceBindings, speakerProfiles} from "../web/speaker-profiles.js";
import {studioWebTools} from "../web/web-mcp.js";
import {textFor} from "../web/i18n.js";

/** Keep separate face-time denominators and explicit gaps in a small saved-report fixture. */
function evidence() {
  const face = {faceId: 3, observedMs: 1200, classifiedMs: 960, unknownMs: 240, coverage: .8,
    byLabel: {neutral: {durationMs: 480, share: .5}, happy: {durationMs: 480, share: .5}}};
  return {faceSpeakers: {3: {speaker: 2, basis: "audio-visual"}}, facialExpressions: {
    version: "facial-expressions-1", status: "partial", model: "fixture", basis: "visible-expression-classifier",
    intervals: [{startMs: 0, endMs: 480, faceId: 3, track: 7, label: "neutral", confidence: .8},
      {startMs: 720, endMs: 1200, faceId: 3, track: 7, label: "happy", confidence: .9},
      {startMs: 720, endMs: 1200, faceId: 4, track: 8, label: "happy", confidence: .7}],
    samples: [{timeMs: 0, faceId: 3, track: 7, label: "neutral", confidence: .8, status: "ok",
      probabilities: {neutral: .8, happy: .2}}, {timeMs: 480, faceId: 3, track: 7, label: "unknown",
      confidence: .4, status: "unknown", reason: "low-confidence"}],
    statistics: {durationMs: 1200, observedMs: 1200, classifiedMs: 960, unknownMs: 240, coverage: .8,
      classifiedFaceMs: 1440, observedFaceMs: 1680, byLabel: face.byLabel,
      faces: [face, {...face, faceId: 4, observedMs: 480, classifiedMs: 480, unknownMs: 0}]},
    reactions: [{questionCue: 1, questionStartMs: 0, questionEndMs: 500, text: "Вопрос <script>?",
      windowStartMs: 0, windowEndMs: 1200, faces: [{...face, dominantLabel: "happy", dominantShare: .5}, {...face, faceId: 4}]}],
  }};
}

test("exact expression cells preserve unmeasured gaps and retain independent unbound faces", () => {
  const visual = evidence();
  visual.faceSpeakers[4] = {speaker: 99, basis: "visible-only"};
  const lanes = expressionLanes(visual, 1000);
  assert.deepEqual(lanes.map(row => [row.id, row.number]), [["expression-3-7", 2], ["expression-4-8", undefined]]);
  assert.deepEqual(lanes[0].clips.map(row => [row.startMs, row.endMs]), [[0, 480], [720, 1000]]);
  assert.match(lanes[0].clips[0].description, /80%/);
  const model = buildTimeline({durationMs: 1200, segments: []}, visual);
  assert.deepEqual(model.lanes.map(row => row.id), ["speaker-2", "expression-3-7", "expression-face-4", "expression-4-8"]);
  assert.equal(buildTimeline({durationMs: 1200, segments: []}).lanes.length, 0);
});

test("unbound face parents group every track independently from bound speaker controls", () => {
  const first = {id: "expression-1-4", faceId: 1, track: 4, kind: "expression", clips: []};
  const second = {...first, id: "expression-1-9", track: 9};
  const other = {...first, id: "expression-2-7", faceId: 2, track: 7};
  const bound = {...first, id: "expression-3-5", faceId: 3, number: 2};
  const grouped = unboundExpressionGroups([other, first, bound, second], {1: "photo"});
  assert.deepEqual(grouped.map(lane => lane.id), ["expression-face-1", first.id, second.id, "expression-face-2", other.id]);
  assert.equal(grouped[0].photo, "photo");
  assert.equal(grouped[0].number, undefined);
  assert.equal(expressionGroupKey(first), expressionGroupKey(second));
  assert.equal(expressionGroupKey(bound), 2);
  assert.notEqual(expressionGroupKey(bound), expressionGroupKey(other));
  assert.deepEqual(first.clips, []);
});

test("invalid imported expression intervals do not fabricate time or confidence", () => {
  const row = evidence().facialExpressions.intervals[0];
  assert.equal(expressionIntervals({intervals: [row, {...row, label: "truthful"}, {...row, confidence: NaN},
    {...row, startMs: -1}, {...row, endMs: 0}, {...row, track: null}, {...row, faceId: "3"}]}).length, 1);
});

test("profiles use current editorial bindings and keep each face's own statistics", () => {
  const visual = evidence();
  const cues = [{startMs: 0, endMs: 1000, text: "Да", speaker: 1, speakerNumber: 5, visualFace: 3}];
  const edited = editorFaceBindings(visual, cues);
  const profiles = speakerProfiles(cues, edited);
  assert.equal(profiles[0].number, 5);
  assert.equal(profiles[0].facialExpressions.faces[0].faceId, 3);
  assert.equal(profiles[0].facialExpressions.faces[0].unknownMs, 240);
  assert.equal(profiles[0].facialExpressions.faces[0].byLabel.happy.share, .5);
  assert.equal(profiles[0].facialExpressions.faces[0].reactions[0].questionCue, 1);
  assert.equal(profileExpressions({face: {id: 99}}, visual).faces.length, 0);
  assert.equal(profileExpressions({}, null), undefined);
});

test("WebMCP pages source measurements, per-face statistics and question windows without a giant nested report", async () => {
  const report = {videoEvidence: evidence()};
  const tools = studioWebTools({ui: {transcript: () => report}, browserML: {dispose() {}}}, new AbortController().signal);
  const expressions = tools.find(tool => tool.name === "ui_facial_expressions");
  const first = await expressions.execute({limit: 1});
  assert.equal(first.items.length, 1);
  assert.equal(first.total, 3);
  assert.equal(first.nextOffset, 1);
  assert.equal(first.model, "fixture");
  assert.match(first.estimate, /not mental state/);
  const second = await expressions.execute({face_id: 3, offset: 1, limit: 1});
  assert.equal(second.items[0].startMs, 720);
  assert.equal(second.nextOffset, null);
  const unknown = await expressions.execute({kind: "samples", start_ms: 480, end_ms: 481});
  assert.equal(unknown.items[0].status, "unknown");
  for (const input of [{limit: 31}, {offset: -1}, {face_id: "3"}, {start_ms: 100, end_ms: 50}, {other: true}]) {
    assert((await expressions.execute(input)).error);
  }
  const stats = await tools.find(tool => tool.name === "ui_expression_statistics").execute({limit: 1});
  assert.equal(stats.items.length, 1);
  assert.equal(stats.statistics.classifiedFaceMs, 1440);
  assert.equal(stats.statistics.faces, undefined);
  const reactions = await tools.find(tool => tool.name === "ui_question_reactions").execute({face_id: 3, limit: 1});
  assert.equal(reactions.total, 1);
  assert.equal(reactions.items[0].text, "Вопрос <script>?");
  assert.equal(reactions.items[0].windowEndMs, 1200);
  assert.equal(reactions.items[0].faces, undefined);
  assert.equal(expressions.annotations.readOnlyHint, true);
  assert.equal(expressions.annotations.untrustedContentHint, true);
});

test("legacy reports are explicitly unmeasured and outputs bound nested classifier maps and question text", () => {
  assert.deepEqual(expressionPage(null).items, []);
  assert.equal(expressionPage({}).status, "not-measured");
  assert.equal(reactionPage({}).nextOffset, null);
  const report = {videoEvidence: evidence()};
  report.videoEvidence.facialExpressions.samples[0].probabilities.injected = 1;
  report.videoEvidence.facialExpressions.samples[0].large = "x".repeat(10000);
  const row = expressionPage(report, {kind: "samples", limit: 1}).items[0];
  assert.equal(row.large, undefined);
  assert.equal(row.probabilities.injected, undefined);
  report.videoEvidence.facialExpressions.reactions[0].text = "x".repeat(5000);
  assert.equal(reactionPage(report).items[0].text.length, 1000);
});

test("English expression controls and caveats are localized independently of report content", () => {
  assert.equal(textFor("en", "Выражения лица"), "Facial expressions");
  assert.equal(textFor("en", "Выражения · {0} · лицо {1}", "Пример", 3), "Expressions · Пример · face 3");
  assert.equal(textFor("en", "Прочитать окна вопросов"), "Read question windows");
  assert.match(textFor("en", "Оценки видимого выражения лица. По ним нельзя определить чувства, намерения или правдивость."), /truthfulness/);
});
