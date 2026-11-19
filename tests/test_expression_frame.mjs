import assert from "node:assert/strict";
import test from "node:test";
import {expressionFaceBox, drawExpressionFrame, waitForExpressionFrame} from "../web/expression-frame-preview.js";
import {expressionMarker} from "../web/facial-expressions.js";
import {momentPage} from "../web/profile-moment-timeline.js";
import {layoutObjectAppearances} from "../web/video-object-appearance-timeline.js";

/** A controllable media boundary emits decoded-frame completion separately from requested currentTime. */
class Video extends EventTarget {
  hidden = false;
  error = null;
  readyState = 2;
  duration = 3;
  videoWidth = 1920;
  videoHeight = 1080;
  seeking = false;
  position = 0;
  paused = false;
  pause() { this.paused = true; }
  get currentTime() { return this.position; }
  set currentTime(value) { this.position = value; this.seeking = true; }
  decoded() { this.seeking = false; this.dispatchEvent(new Event("seeked")); }
}

const selection = {timeMs: 40, faceId: 3, track: 7};
const evidence = {timeline: {stepMs: 40, frames: [[], [
  [8, .7, .1, .1, .2, 0, 3], [7, .1, .2, .2, .3, 0, 3], [9, .4, .4, .1, .1, 0, 4],
], []]}};

test("head geometry requires the same source cell, face and track; a nearby or same-face other track is not reused", () => {
  assert.deepEqual(expressionFaceBox(evidence, selection), [.1, .2, .2, .3]);
  assert.equal(expressionFaceBox(evidence, {...selection, timeMs: 80}), null);
  assert.equal(expressionFaceBox(evidence, {...selection, track: 9}), null);
  assert.equal(expressionFaceBox(evidence, {...selection, faceId: 4}), null);
  assert.equal(expressionFaceBox(null, selection), null);
});

test("frame pixels and the chosen head use intrinsic source coordinates independent of a letterboxed player", () => {
  const calls = [];
  const ctx = {drawImage: (...args) => calls.push(["image", ...args]), strokeRect: (...args) => calls.push(["rect", ...args])};
  const canvas = {getContext: () => ctx};
  const video = new Video();
  drawExpressionFrame(canvas, video, [.1, .2, .2, .3]);
  assert.deepEqual([canvas.width, canvas.height], [1440, 810]);
  assert.deepEqual(calls[1], ["rect", 144, 162, 288, 243]);
  assert.deepEqual(calls[2], calls[1]);
  assert.equal(calls.filter(row => row[0] === "rect").length, 2, "only the selected head's two contrasting edges are painted");
});

test("a source-frame seek resolves only after the requested decoded frame, and pauses playback", async () => {
  const video = new Video();
  let completed = false;
  const pending = waitForExpressionFrame(video, 800, {timeoutMs: 100}).then(() => { completed = true; });
  await Promise.resolve();
  assert.equal(completed, false);
  assert.equal(video.paused, true);
  assert.equal(video.currentTime, .8);
  video.position = .7;
  video.decoded();
  await Promise.resolve();
  assert.equal(completed, false, "an unrelated completed seek is not the requested frame");
  video.position = .8;
  video.decoded();
  await pending;
  assert.equal(completed, true);
});

test("failed, cancelled and timed out seeks never resolve with a stale source frame", async () => {
  const video = new Video();
  const cancellation = new AbortController();
  const cancelled = waitForExpressionFrame(video, 500, {signal: cancellation.signal, timeoutMs: 100});
  cancellation.abort();
  await assert.rejects(cancelled, {name: "AbortError"});
  await assert.rejects(waitForExpressionFrame(video, 500, {timeoutMs: 5}), /недоступен/);
  const failed = waitForExpressionFrame(video, 500, {timeoutMs: 100});
  video.dispatchEvent(new Event("error"));
  await assert.rejects(failed, /недоступен/);
  await assert.rejects(waitForExpressionFrame(video, 4000), /недоступен/);
});

test("profile marker pages remain bounded on a full min/max source axis; expression markers are real emoji", () => {
  const entries = Array.from({length: 201}, (_, index) => ({timeMs: index * 100, startMs: index * 100, endMs: index * 100 + 80}));
  const first = momentPage(entries);
  const last = momentPage(entries, first.nextOffset);
  assert.equal(first.items.length, 200);
  assert.equal(last.items.length, 1);
  assert.deepEqual(first.domain, [0, 20080]);
  assert.deepEqual(last.domain, first.domain);
  const marks = layoutObjectAppearances(last.items, last.domain);
  assert(marks[0].dot > 99 && marks[0].dot < 100, "the final page retains its full-source position");
  assert.equal(expressionMarker("happy"), "🙂");
  assert.equal(expressionMarker("neutral"), "😐");
});
