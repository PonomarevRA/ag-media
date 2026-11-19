import test from "node:test";
import assert from "node:assert/strict";
import {mountTranscriptEditor} from "../web/transcript-editor.js";

globalThis.document = {getElementById: () => null};
globalThis.Option = class {
  constructor(text, value) { this.text = text; this.value = value; }
};

/** Model only the editor controls; asynchronous callbacks remain directly awaitable. */
class Control {
  constructor() { this.listeners = new Map(); this.value = ""; this.children = []; }
  addEventListener(name, listener) { this.listeners.set(name, listener); }
  fire(name) { return this.listeners.get(name)?.(); }
  replaceChildren(...children) { this.children = children; }
  add(child) { this.children.push(child); }
}

/** Hold one material operation while a synthetic host changes its active source. */
function fixture(source = null) {
  const controls = new Map();
  const find = id => {
    if (!controls.has(id)) controls.set(id, new Control());
    return controls.get(id);
  };
  let current = source;
  let complete, fail;
  const applies = [];
  const calls = [];
  const request = (route, body, flags) => {
    calls.push({route, body, flags});
    if (route === "transcript-materials") return Promise.resolve({items: []});
    return new Promise((resolve, reject) => { complete = resolve; fail = reject; });
  };
  const editor = mountTranscriptEditor(find, request, {
    current: () => current, title: () => "Synthetic material",
    apply: data => { applies.push(data); current = data; },
  });
  return {editor, find, applies, calls, current: () => current,
    replace: value => { current = value; },
    complete: value => complete(value), fail: error => fail(error)};
}

test("a late open cannot replace a new source or its status, and releases pending ownership", async () => {
  const f = fixture();
  const opening = f.editor.open("saved-A", "ru");
  const upload = {sourceDigest: "upload-B", segments: []};
  f.replace(upload);
  f.editor.setBusy(true);
  f.find("transcript-material-status").textContent = "New source active";
  f.complete({id: "saved-A", revision: 1, segments: []});
  assert.equal(await opening, undefined);
  assert.equal(f.applies.length, 0);
  assert.equal(f.current(), upload);
  assert.equal(f.find("transcript-material-status").textContent, "New source active");
  f.editor.setBusy(false);
  const reopened = f.editor.open("saved-C", "ru");
  f.complete({id: "saved-C", revision: 1, segments: []});
  assert.equal((await reopened).id, "saved-C");
  assert.equal(f.applies.length, 1);
});

test("a stale load failure leaves the new source status intact", async () => {
  const f = fixture();
  const opening = f.editor.open("saved-A", "ru");
  f.editor.setBusy(true);
  f.find("transcript-material-status").textContent = "New source active";
  f.fail(Error("Old request failed"));
  await opening;
  assert.equal(f.find("transcript-material-status").textContent, "New source active");
  assert.equal(f.applies.length, 0);
});

test("source replacement fences a load even without a busy transition, and same-source reload works", async () => {
  const original = {id: "saved-A", revision: 1, segments: []};
  const f = fixture(original);
  const replaced = f.editor.open("saved-A", "ru");
  f.replace({id: "saved-B", segments: []});
  f.complete({...original, revision: 2});
  await replaced;
  assert.equal(f.applies.length, 0);
  const current = f.current();
  const reload = f.editor.open("saved-B", "ru");
  f.complete({...current, revision: 2});
  assert.equal((await reload).revision, 2);
  assert.equal(f.applies.length, 1);
});

test("late successful save persists once without applying to a replacement upload", async () => {
  const f = fixture({sourceDigest: "upload-A", language: "ru", segments: [{text: "Synthetic"}]});
  const saving = f.find("transcript-material-save").fire("click");
  const replacement = {sourceDigest: "upload-B", segments: []};
  f.replace(replacement);
  f.editor.setBusy(true);
  f.find("transcript-material-status").textContent = "New source active";
  f.complete({id: "saved-A", revision: 1, segments: []});
  assert.equal((await saving).id, "saved-A");
  assert.equal(f.calls.length, 1);
  assert.equal(f.calls[0].route, "transcript-material");
  assert.equal(f.applies.length, 0);
  assert.equal(f.current(), replacement);
  assert.equal(f.find("transcript-material-status").textContent, "New source active");
});

test("late patch persists once without applying to a replacement source", async () => {
  const f = fixture({id: "saved-A", revision: 1, language: "ru", segments: []});
  const patching = f.editor.patch(0, {text: "Corrected synthetic cue"});
  const replacement = {sourceDigest: "upload-B", segments: []};
  f.replace(replacement);
  f.editor.setBusy(true);
  f.complete({id: "saved-A", revision: 2, segments: []});
  assert.equal((await patching).revision, 2);
  assert.equal(f.calls.length, 1);
  assert.equal(f.calls[0].route, "transcript-cue");
  assert.equal(f.applies.length, 0);
  assert.equal(f.current(), replacement);
});

test("late save failure does not change replacement status and pending state clears", async () => {
  const f = fixture({sourceDigest: "upload-A", language: "ru", segments: [{text: "Synthetic"}]});
  const saving = f.find("transcript-material-save").fire("click");
  f.replace({sourceDigest: "upload-B", segments: []});
  f.editor.setBusy(true);
  f.find("transcript-material-status").textContent = "New source active";
  f.fail(Error("Old save failed"));
  await saving;
  assert.equal(f.find("transcript-material-status").textContent, "New source active");
  f.editor.setBusy(false);
  const opening = f.editor.open("saved-C", "ru");
  f.complete({id: "saved-C", revision: 1, segments: []});
  assert.equal((await opening).id, "saved-C");
});

test("a completed host operation invalidates an old load even when the source object is retained", async () => {
  const f = fixture({id: "saved-A", segments: []});
  const opening = f.editor.open("saved-A", "ru");
  f.editor.setBusy(true);
  f.editor.setBusy(false);
  f.complete({id: "saved-A", revision: 2, segments: []});
  assert.equal(await opening, undefined);
  assert.equal(f.applies.length, 0);
});

test("same-source save and patch apply their durable results and return their revisions", async () => {
  const f = fixture({sourceDigest: "upload-A", language: "ru", segments: [{text: "Synthetic"}]});
  const saving = f.find("transcript-material-save").fire("click");
  const saved = {id: "saved-A", revision: 1, language: "ru", segments: [{text: "Synthetic"}]};
  f.complete(saved);
  assert.equal(await saving, saved);
  assert.equal(f.current(), saved);
  assert.equal(f.applies.length, 1);
  const patching = f.editor.patch(0, {text: "Synthetic correction"});
  const patched = {...saved, revision: 2};
  f.complete(patched);
  assert.equal(await patching, patched);
  assert.equal(f.current(), patched);
  assert.equal(f.applies.length, 2);
});

test("current save failures remain visible and patch failures release pending ownership", async () => {
  const f = fixture({sourceDigest: "upload-A", language: "ru", segments: [{text: "Synthetic"}]});
  const saving = f.find("transcript-material-save").fire("click");
  f.fail(Error("Save rejected"));
  await saving;
  assert.equal(f.find("transcript-material-status").textContent, "Save rejected");
  f.replace({id: "saved-A", revision: 1, language: "ru", segments: []});
  const patching = f.editor.patch(0, {text: "Synthetic correction"});
  f.fail(Error("Revision conflict"));
  await assert.rejects(patching, /Revision conflict/);
  const opening = f.editor.open("saved-C", "ru");
  f.complete({id: "saved-C", revision: 1, segments: []});
  assert.equal((await opening).id, "saved-C");
});
