import test from "node:test";
import assert from "node:assert/strict";
import {cleanSpeakerNames, speakerNamesKey, transcriptSides, mountTranscriptChat, installChatHint, transcriptPortrait} from "../web/transcript-chat.js";

test("a single unconfirmed frame photo is cue-local; it cannot override a bound portrait", () => {
  const cue = {speaker: 1, startMs: 100};
  const evidence = {phrases: {100: {photos: ["Y2FuZGlkYXRl"]}},
    faceSpeakers: {8: {speaker: 1, basis: "audio-visual"}}, editorTimeline: {facePhotos: {8: "Ym91bmQ="}}};
  assert.deepEqual(transcriptPortrait(cue, null, evidence), {jpeg: "Y2FuZGlkYXRl", confirmed: false});
  assert.equal(transcriptPortrait({...cue, startMs: 200}, null, evidence), null);
  assert.deepEqual(transcriptPortrait(cue, {face: {id: 8}}, evidence), {jpeg: "Ym91bmQ=", confirmed: true});
  assert.equal(transcriptPortrait({...cue, overlap: true}, null, evidence), null);
  evidence.phrases[100].photos.push("b3RoZXI=");
  assert.equal(transcriptPortrait(cue, null, evidence), null);
  assert.equal(cue.speakerPhoto, undefined);
});

test("an avatar respects explicit photo selection and speaker edits", () => {
  const cue = {speaker: 1, startMs: 100, speakerPhoto: {speaker: 1, jpeg: "Y2hvc2Vu"}};
  assert.deepEqual(transcriptPortrait(cue, null, null), {jpeg: "Y2hvc2Vu", confirmed: true});
  assert.equal(transcriptPortrait({...cue, speakerNumber: 2}, null, null), null);
});

test("old active-face evidence cannot confirm the avatar after a speaker reassignment", () => {
  const cue = {startMs: 0, speaker: 1};
  const evidence = {faceSpeakers: {8: {speaker: 1, basis: "audio-visual"}},
    phrases: {0: {activity: {state: "visible-active", face: 8, speaker: 1}, activePhoto: "AAAA"}}};
  assert.deepEqual(transcriptPortrait(cue, null, evidence), {jpeg: "AAAA", confirmed: true});
  assert.deepEqual(transcriptPortrait({...cue, speakerNumber: 2}, null, evidence), {jpeg: "AAAA", confirmed: false});
  delete evidence.faceSpeakers;
  assert.deepEqual(transcriptPortrait(cue, null, evidence), {jpeg: "AAAA", confirmed: false});
});
import {activityStatus} from "../web/video-evidence.js";

test("speaker aliases accept literal bounded names keyed by known numeric IDs only", () => {
  const names = cleanSpeakerNames(JSON.parse('{"1":"  Алина  ","2":"<b>Имя</b>","0":"bad","__proto__":"bad","3":42,"4":"\\u0000 ","5":"' + "x".repeat(100) + '"}'));
  assert.deepEqual(names, {1: "Алина", 2: "<b>Имя</b>", 5: "x".repeat(80)});
  assert.deepEqual(cleanSpeakerNames(["Имя"]), {});
});

test("the source digest scopes aliases across reopenings without filename collisions", () => {
  assert.equal(speakerNamesKey({sourceDigest: "abc", id: "first"}), speakerNamesKey({sourceDigest: "abc", id: "other"}));
  assert.notEqual(speakerNamesKey({sourceDigest: "abc", title: "video.mp4"}), speakerNamesKey({sourceDigest: "def", title: "video.mp4"}));
  assert.equal(speakerNamesKey({title: "same.mp4"}), null);
  assert.notEqual(speakerNamesKey({id: "1"}), null);
});

test("speaker changes alternate turns; unknown/overlap stays centred and manual attribution owns its side", () => {
  const cues = [1, 1, 2, 2, 3, 1].map(speaker => ({speaker}));
  assert.deepEqual(transcriptSides(cues), ["left", "left", "right", "right", "left", "right"]);
  assert.deepEqual(transcriptSides([{speaker: 1}, {}, {speaker: 1}, {speaker: 2, overlap: true}, {speaker: 2}]), ["left", "center", "left", "center", "right"]);
  assert.deepEqual(transcriptSides([{speaker: 1}, {speaker: 2, overlap: true, speakerNumber: 3}]), ["left", "right"]);
});

test("browser aliases survive same-source report patches but do not leak to another source or an anonymous reset", () => {
  const original = globalThis.document;
  globalThis.document = {createElement: () => ({setAttribute() {}})};
  try {
    const data = new Map([[speakerNamesKey({sourceDigest: "a"}), '{"1":"Алина"}']]);
    const storage = {getItem: key => data.get(key) ?? null};
    const chat = mountTranscriptChat({children: [], after() {}}, storage);
    const a = {sourceDigest: "a"}; chat.setData(a, [], null);
    assert.deepEqual(a.speakerNames, {1: "Алина"});
    const patch = {sourceDigest: "a"}; chat.setData(patch, [], null);
    assert.deepEqual(patch.speakerNames, {1: "Алина"});
    const b = {sourceDigest: "b"}; chat.setData(b, [], null);
    assert.deepEqual(b.speakerNames, {});
    chat.reset();
    const anonymous = {}; chat.setData(anonymous, [], null);
    assert.deepEqual(anonymous.speakerNames, {});
    const unavailable = mountTranscriptChat({children: [], after() {}}, {getItem() { throw Error("blocked"); }});
    const imported = {sourceDigest: "a", speakerNames: {2: "Максим"}};
    unavailable.setData(imported, [], null);
    assert.deepEqual(imported.speakerNames, {2: "Максим"});
  } finally { globalThis.document = original; }
});

test("compact activity badges never turn missing/ambiguous activity into a confirmed speaker", () => {
  assert.equal(activityStatus(null), null);
  assert.equal(activityStatus({state: "visible-active", activeFrames: 20, phraseFrames: 25}).icon, "face-check");
  assert.equal(activityStatus({state: "partial-or-ambiguous"}).icon, "face-partial");
  const unconfirmed = activityStatus({state: "not-confirmed"});
  assert.equal(unconfirmed.icon, "face-question");
  assert.match(unconfirmed.label, /не подтверждён/u);
});

test("hover waits250ms, cancels on leave/blur, and never reveals a detached status hint", () => {
  const saved = {set: globalThis.setTimeout, clear: globalThis.clearTimeout};
  const calls = [], cancelled = [];
  globalThis.setTimeout = (callback, delay) => { calls.push({callback, delay}); return calls.length; };
  globalThis.clearTimeout = id => cancelled.push(id);
  try {
    const listeners = {}, button = {dataset: {}, isConnected: true, hovered: true,
      matches() { return this.hovered; }, closest() { return null; },
      addEventListener(type, callback) { listeners[type] = callback; }};
    installChatHint(button);
    listeners.pointerenter();
    assert.equal(calls[0].delay, 250);
    assert.equal(Object.hasOwn(button.dataset, "hintReady"), false);
    listeners.pointerleave(); button.hovered = false; calls[0].callback();
    assert.equal(Object.hasOwn(button.dataset, "hintReady"), false);
    button.hovered = true; listeners.pointerenter(); calls[1].callback();
    assert.equal(Object.hasOwn(button.dataset, "hintReady"), true);
    listeners.blur();
    assert.equal(Object.hasOwn(button.dataset, "hintReady"), false);
    listeners.pointerenter(); button.isConnected = false; calls[2].callback();
    assert.equal(Object.hasOwn(button.dataset, "hintReady"), false);
    assert.ok(cancelled.includes(1) && cancelled.includes(2));
  } finally { globalThis.setTimeout = saved.set; globalThis.clearTimeout = saved.clear; }
});
