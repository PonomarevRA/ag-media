import test from "node:test";
import assert from "node:assert/strict";
import {mountMonitorActions} from "../web/editor-monitor-actions.js";

/** Minimal sibling-aware DOM fixture; adjacent controls deliberately have no whitespace nodes. */
class Node extends EventTarget {
  constructor(id) { super(); this.id = id; this.children = []; this.dataset = {}; this.attrs = new Map(); }
  get nextSibling() { return this.parentNode?.children[this.parentNode.children.indexOf(this) + 1] ?? null; }
  append(node) { this.insertBefore(node, null); }
  insertBefore(node, next) {
    if (next && next.parentNode !== this) throw Error("Invalid sibling anchor");
    if (node.parentNode) node.parentNode.children.splice(node.parentNode.children.indexOf(node), 1);
    const index = next ? this.children.indexOf(next) : this.children.length;
    this.children.splice(index, 0, node); node.parentNode = this;
  }
  querySelector() { return this.label; }
  setAttribute(key, value) { this.attrs.set(key, value); }
  removeAttribute(key) { this.attrs.delete(key); }
}

/** Expose readiness changes independently of the controller; a toggle event models native details. */
function fixture() {
  const nodes = new Map(["media-overlay-options", "media-video-export", "editor-clothing-toggle", "media-clothing-export", "status"].map(id => [id, new Node(id)]));
  const parent = new Node("preview"), actions = new Node("actions"), controls = new Node("transport");
  parent.append(nodes.get("media-overlay-options")); parent.append(actions);
  for (const id of ["media-video-export", "editor-clothing-toggle", "status"]) actions.append(nodes.get(id));
  nodes.get("media-overlay-options").label = new Node("checkbox-label");
  const details = nodes.get("media-clothing-export"); details.hidden = true; details.open = false;
  let observer;
  const document = {getElementById: id => nodes.get(id), defaultView: {MutationObserver: class {
    constructor(callback) { observer = this; this.callback = callback; }
    observe() {} disconnect() { this.disconnected = true; }
  }}};
  return {nodes, parent, actions, controls, details, document, observe: () => observer};
}

test("disposal restores adjacent siblings exactly and preserves export ownership across remount", () => {
  const env = fixture(), button = env.nodes.get("media-video-export");
  const original = [...env.actions.children]; let clicks = 0;
  button.addEventListener("click", () => clicks++);
  for (let attempt = 0; attempt < 2; attempt++) {
    const controller = mountMonitorActions(env.document, env.controls);
    assert.equal(env.controls.children.includes(button), true);
    button.dispatchEvent(new Event("click"));
    controller.dispose();
    assert.deepEqual(env.actions.children, original);
    assert.equal(env.observe().disconnected, true);
  }
  assert.equal(clicks, 2);
});

test("mask icon follows native readiness and disclosure state and stops reacting after disposal", () => {
  const env = fixture(), masks = env.nodes.get("editor-clothing-toggle");
  const controller = mountMonitorActions(env.document, env.controls);
  assert.equal(masks.hidden, true);
  masks.dispatchEvent(new Event("click"));
  assert.equal(env.details.open, false);
  env.details.hidden = false; env.observe().callback();
  assert.equal(masks.hidden, false);
  masks.dispatchEvent(new Event("click")); env.details.dispatchEvent(new Event("toggle"));
  assert.equal(env.details.open, true);
  assert.equal(masks.attrs.get("aria-expanded"), "true");
  env.details.hidden = true; env.observe().callback();
  assert.equal(masks.attrs.get("aria-expanded"), "false");
  controller.dispose(); env.details.open = false; env.details.hidden = false;
  masks.dispatchEvent(new Event("click"));
  assert.equal(env.details.open, false);
});
