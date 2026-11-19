import test from "node:test";
import assert from "node:assert/strict";
import {compareRectangles, unionRectangles, insets} from "../web/ui-measurement-geometry.js";
import {measureUi} from "../web/ui-measurements.js";
import {studioWebTools} from "../web/web-mcp.js";
import {createUiBridge} from "../web/mcp-bridge.js";

test("geometry separates direction, gap, overlap and edge alignment at fractional CSS pixels", () => {
  const a = {x: 10, y: 20, width: 100, height: 44};
  const b = {x: 126.5, y: 20, width: 80, height: 44};
  const forward = compareRectangles(a, b, .5), reverse = compareRectangles(b, a, .5);
  assert.deepEqual(forward.gapPx, {x: 16.5, y: 0});
  assert.deepEqual(reverse.gapPx, forward.gapPx);
  assert.deepEqual(forward.overlapPx, {x: 0, y: 44});
  assert.equal(forward.aligned.top, true);
  assert.equal(forward.aligned.width, false);
  assert.equal(reverse.alignmentDeltaPx.left, -116.5);
  assert.equal(compareRectangles(a, {...a, y: a.y + 1.004}, 1).aligned.top, false);
  assert.deepEqual(compareRectangles(a, {...a, x: 50}).overlapPx, {x: 60, y: 44});
});

test("text union retains signed overflow and no fabricated empty bounds", () => {
  assert.equal(unionRectangles([]), null);
  assert.equal(insets(null, {}), null);
  const union = unionRectangles([{x: 8, y: 25, width: 80, height: 12}, {x: 12, y: 39, width: 60, height: 12}]);
  assert.deepEqual(union, {x: 8, y: 25, width: 80, height: 26});
  assert.deepEqual(insets({x: 10, y: 20, width: 100, height: 44}, union), {top: 5, right: 22, bottom: 13, left: -2});
});

test("standalone batches reject malformed, excessive and nonfinite input before touching the DOM", () => {
  const neverRead = {get defaultView() {throw Error("DOM touched");}};
  for (const options of [{targets: Array(33).fill("x")}, {targets: ["x y"]}, {pairs: [{from: "x", to: "y", script: "evil"}]},
    {pairs: [{from: null, to: "x"}]}, {tolerance_px: Infinity}, {tolerance_px: -1}, {include_text: "yes"},
    {targets: Array.from({length: 32}, (_, i) => "a" + i), pairs: [{from: "extra", to: "a0"}]}]) {
    assert.throws(() => measureUi(options, neverRead), {name: "TypeError"});
  }
});

test("ordinary bridge discovers canonical measurement and shares strict nested validation and revision fencing", async () => {
  const lifecycle = new AbortController();
  const tools = studioWebTools({ui: {}, browserML: {dispose() {}}}, lifecycle.signal);
  assert(tools.some(tool => tool.name === "ui_measure"));
  assert(tools.every(tool => !tool.name.startsWith("voice_")));
  const measurement = tools.find(tool => tool.name === "ui_measure");
  assert.equal(measurement.annotations.readOnlyHint, true);
  for (const input of [{targets: [7]}, {pairs: [{from: "x", to: "y", extra: true}]}, {pairs: [{from: "x"}]}, {tolerance_px: NaN},
    {tolerance_px: 17}, {selector: "button"}, {script: "document.body.innerHTML"}]) assert((await measurement.execute(input)).error);
  let epoch = 0;
  const ui = {inspect: () => ({page: "/tools"}), transcript: () => null, contextRevision: () => epoch};
  const bridge = createUiBridge(tools, ui, () => "/tools");
  const old = bridge.inspect().revision;
  epoch++;
  const result = await bridge.execute({name: "ui_measure", revision: old, arguments: {targets: ["x"]}, expiresAtMs: Date.now() + 1000}, lifecycle.signal);
  assert.equal(result.error, "ui_context_changed");
  const removed = await bridge.execute({name: "voice_ui_measure", revision: bridge.inspect().revision, arguments: {}, expiresAtMs: Date.now() + 1000}, lifecycle.signal);
  assert.equal(removed.error, "ui_tool_not_allowed");
  lifecycle.abort();
});

test("unsupported text fragments never become complete text insets (R1 regression; synthetic DOM only)", () => {
  const rect = {left: 0, top: 0, width: 100, height: 40};
  const style = {display: "block", opacity: "1", transform: "none", zoom: "1", visibility: "visible", fontSize: "16px"};
  for (const side of ["Top", "Right", "Bottom", "Left"]) {style["padding" + side] = "0px";style["border" + side + "Width"] = "0px";}
  const button = {id: "sample", tagName: "BUTTON", parentElement: null, getClientRects: () => [rect],
    getBoundingClientRect: () => rect, querySelectorAll: () => [], clientLeft: 0, clientTop: 0, clientWidth: 100, clientHeight: 40};
  const plain = {parentElement: button, closest: () => null};
  const moved = {parentElement: button, closest: () => null};
  const view = {performance, innerWidth: 1000, innerHeight: 800, scrollX: 0, scrollY: 0, devicePixelRatio: 1,
    getComputedStyle: node => node === moved ? {...style, transform: "translateX(5px)"} : style};
  for (const includePlain of [true, false]) {
    const nodes = [...(includePlain ? [{nodeValue: "before", parentElement: plain}] : []), {nodeValue: "after", parentElement: moved}];
    const document = {defaultView: view, querySelectorAll: () => [button], getElementById: () => button,
      createTreeWalker: () => {let index = 0;return {nextNode: () => nodes[index++]};},
      createRange: () => ({selectNodeContents() {}, getClientRects: () => [{left: 10, top: 10, width: 20, height: 20}]})};
    const text = measureUi({targets: ["sample"]}, document).targets[0].text;
    assert.equal(text.status, includePlain ? "partial" : "unknown");
    assert(text.reasons.includes("transformed"));
    assert.equal(text.insetsPx, null);
    assert.equal(text.contentInsetsPx, null);
  }
});

test("measurement replies omit unrelated UI state on success and stale-context errors", async () => {
  const lifecycle = new AbortController();let epoch = 0;
  const ui = {inspect: () => ({privateDraft: "PRIVATE_DRAFT"}), transcript: () => null, contextRevision: () => epoch};
  const bridge = createUiBridge([{name: "ui_measure", inputSchema: {}, execute: async () => ({targets: []})}], ui, () => "/");
  const command = {name: "ui_measure", revision: bridge.inspect().revision, arguments: {}, expiresAtMs: Date.now() + 1000};
  const reply = await bridge.execute(command, lifecycle.signal);
  assert.equal(reply.status, "ok");assert.equal(reply.ui, undefined);assert(!JSON.stringify(reply).includes("PRIVATE_DRAFT"));
  epoch++;
  const stale = await bridge.execute(command, lifecycle.signal);
  assert.equal(stale.error, "ui_context_changed");assert.equal(stale.ui, undefined);
  lifecycle.abort();
});
