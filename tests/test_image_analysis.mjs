import test from "node:test";
import assert from "node:assert/strict";
import {imageBox, imageFileError, imageItems, isImageFile} from "../web/image-analysis-model.js";
import {mountImageAnalysis} from "../web/image-analysis.js";

test("image routing handles missing MIME and excludes audio/video", () => {
  assert.equal(isImageFile({name: "SCREEN.PNG", type: ""}), true);
  assert.equal(isImageFile({name: "frame", type: "image/jpeg"}), true);
  assert.equal(isImageFile({name: "clip.mp4", type: "video/mp4"}), false);
  assert.equal(isImageFile({name: "voice.wav", type: "audio/wav"}), false);
  assert.equal(imageFileError({name: "huge.png", size: 25 * 1024 * 1024}), "Картинка должна занимать не больше 24 МиБ.");
  assert.equal(imageFileError({name: "screen.webp", size: 50, type: "image/webp"}), "Выбери картинку PNG или JPEG.");
});

test("render proposals reject invalid coordinates and distinguish UI candidates from measured confidence", () => {
  for (const box of [[0, 0, NaN, 1], [0, 0, Infinity, 1], [-1, 0, .2, .2], [0, 0, 0, .1], [0, 0, 2, 2]])
    assert.equal(imageBox(box), null);
  const items = imageItems({objects: [{label: "cup", score: .81, box: [.1, .2, .2, .3]}],
    interface: {elements: [{role: "button_candidate", text: "<img onerror=alert(1)>", box: [.5, .1, .2, .1]},
      {role: "text_line", text: "already OCR", box: [.1, .1, .3, .1]}]},
    ocr: [{text: "hello", score: .9, box: [.1, .1, .3, .1]}]});
  assert.equal(items.length, 3);
  assert.deepEqual(items.map(row => row.layer), ["objects", "ui", "text"]);
  assert.equal(items[1].score, null);
  assert.equal(items[1].label, "<img onerror=alert(1)>");
});

/** Minimal stateful elements for exercising ownership/cancellation; actual rendering is checked in the browser. */
class Element {
  constructor() { this.children = []; this.attrs = {}; this.checked = true; this.value = "full"; this.hidden = false;
    this.classList = {toggle() {}}; this.dataset = {}; }
  setAttribute(name, value) { this.attrs[name] = value; }
  removeAttribute(name) { delete this.attrs[name]; }
  append(...nodes) { this.children.push(...nodes); }
  replaceChildren(...nodes) { this.children = nodes; }
}

test("closing an in-flight image fences stale model completion and releases its source URL", async t => {
  const previous = globalThis.document;
  globalThis.document = {createElement: () => new Element(), createElementNS: () => new Element()};
  t.after(() => { globalThis.document = previous; });
  const nodes = new Map();
  const find = id => { if (!nodes.has(id)) nodes.set(id, new Element()); return nodes.get(id); };
  const pending = [];
  const busy = [];
  const urls = [], revoked = [];
  t.mock.method(URL, "createObjectURL", () => { const url = `blob:${urls.length}`; urls.push(url); return url; });
  t.mock.method(URL, "revokeObjectURL", url => revoked.push(url));
  const view = mountImageAnalysis(find, (...args) => new Promise(resolve => pending.push({args, resolve})),
    {onOpen() { view.clear(); }, setBusy: value => busy.push(value), workbench: new Element()});
  const file = Object.assign(new Blob(["image"], {type: "image/png"}), {name: "source.png"});
  assert.equal(view.open(file), true);
  assert.equal(pending[0].args[0], "analyze-image");
  view.clear();
  assert.equal(pending[0].args[2].signal.aborted, true);
  pending[0].resolve({imageSize: [100, 100], objects: []});
  await new Promise(resolve => setImmediate(resolve));
  assert.equal(view.snapshot(), null);
  assert.equal(find("image-analysis").hidden, true);
  assert.deepEqual(revoked, urls);
  assert.equal(busy.at(-1), false);
});
