import test from "node:test";
import assert from "node:assert/strict";
import {mountDialogueAudit} from "../web/dialogue-audit.js";

/** Supply the DOM protocol used by the disclosure without browser rendering claims. */
class Element {
  constructor(tag) { this.tag = tag; this.children = []; this.dataset = {}; this.attributes = {}; }
  setAttribute(key, value) { this.attributes[key] = value; }
  getAttribute(key) { return this.attributes[key]; }
  hasAttribute(key) { return Object.hasOwn(this.attributes, key); }
  removeAttribute(key) { delete this.attributes[key]; }
  addEventListener() {}
  append(...items) { this.children.push(...items); }
  prepend(...items) { this.children.unshift(...items); }
  replaceChildren(...items) { this.children = items; }
}

/** Restore global DOM/fetch ownership after each isolated panel check. */
function setup(t) {
  const originalDocument = globalThis.document;
  const originalFetch = globalThis.fetch;
  globalThis.document = {createElement: tag => new Element(tag), createElementNS: (ns, tag) => new Element(tag)};
  t.after(() => { globalThis.document = originalDocument; globalThis.fetch = originalFetch; });
  const owner = new Element("details");
  let report = {durationMs: 5000, segments: [{startMs: 100, endMs: 1500, text: "Raw text"}],
    speechTurns: [{startMs: 100, endMs: 1500, speaker: 1}]};
  const source = new Blob(["Source media bytes"]);
  const seeks = [];
  const panel = mountDialogueAudit(owner, {currentReport: () => report, currentSource: () => source,
    seek: stamp => seeks.push(stamp)});
  panel.update(report);
  return {owner, get report() { return report; }, source, seeks, panel,
    replaceReport(value) { report = value; panel.update(report); }};
}

test("dialogue audit sends source-clock facts and keeps drafts separate from original text", async t => {
  const {owner, report, seeks, panel} = setup(t);
  const inventory = {status: "complete", summary: {candidateCount: 21},
    candidates: Array.from({length: 21}, (_, index) => ({id: String(index), startMs: 100 + index,
      endMs: 1500, reasons: ["speaker-transition-inside-cue"]})),
    repairs: {suggestions: [{candidateId: "0", startMs: 100, text: "Unverified ASR draft"}]}};
  let body;
  globalThis.fetch = async (url, options) => {
    body = options.body;
    return new Response(new TextEncoder().encode(JSON.stringify({result: inventory}) + "\n"));
  };
  await owner.children[1].children[0].onclick();
  assert.equal(report.segments[0].text, "Raw text");
  assert.equal(report.dialogueAudit.candidates.length, 21);
  assert.equal(owner.children[3].children.length, 20);
  owner.children[4].onclick();
  assert.equal(owner.children[3].children.length, 21);
  owner.children[3].children[0].children[0].onclick();
  assert.deepEqual(seeks, [100]);
  const bytes = new Uint8Array(await body.arrayBuffer());
  const metadataLength = new DataView(bytes.buffer).getUint32(0);
  const sent = JSON.parse(new TextDecoder().decode(bytes.subarray(4, 4 + metadataLength)));
  assert.deepEqual(sent.speechTurns, report.speechTurns);
  assert.equal(owner.children[3].children[0].children[2].tag, "details");
  panel.update(null);
  assert.equal(owner.hidden, true);
});

test("partial retry failures stay visible while reviewable candidates remain available", t => {
  const {owner, panel} = setup(t);
  for (const state of ["timeout", "failed", "cancelled"]) {
    panel.update({dialogueAudit: {status: state, error: `Specific ${state} detail`,
      candidates: [{id: "issue", startMs: 100, endMs: 1500, reasons: ["multiple-acoustic-labels"]}]}});
    assert.match(owner.children[2].textContent, new RegExp(`Specific ${state} detail`));
    assert.equal(owner.children[3].children.length, 1);
  }
});

/** Complete a delayed old audit after a same-source editorial change. */
async function staleSpeechReply(t, mode) {
  const view = setup(t);
  let finish;
  globalThis.fetch = () => new Promise(resolve => { finish = resolve; });
  const pending = view.owner.children[1].children[0].onclick();
  const corrected = "Corrected transcript with the missing words restored";
  if (mode === "replace") {
    view.replaceReport({...view.report, segments: [{...view.report.segments[0], text: corrected}]});
  } else {
    view.report.segments[0].text = corrected;
    if (mode === "refresh") view.panel.update(view.report);
  }
  const oldInventory = {status: "complete", candidates: [{id: "old-speech-snapshot", startMs: 100,
    endMs: 1500, reasons: ["sparse-text-over-voiced-audio"]}]};
  finish(new Response(JSON.stringify({result: oldInventory}) + "\n"));
  await pending;
  assert.equal(view.report.segments[0].text, corrected);
  assert.equal(view.report.dialogueAudit?.candidates?.some(row => row.id === "old-speech-snapshot") ?? false, false);
}

test("an in-place same-source edit cannot receive findings for the prior speech snapshot", async t => {
  await staleSpeechReply(t, "in-place");
});

test("refreshing an edited same-source report invalidates its outstanding audit", async t => {
  await staleSpeechReply(t, "refresh");
});

test("a replacement report for the same source cannot receive the prior report's audit", async t => {
  await staleSpeechReply(t, "replace");
});
