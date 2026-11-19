import test from "node:test";
import assert from "node:assert/strict";
import {workspaceDimensions, minimumTimelineHeight} from "../web/editor-workspace.js";
import {fitTimelineScale} from "../web/video-timeline.js";
import {performVideoAction, videoActionEnabled, videoFrameStepMs, lastVideoFrameTime, mountVideoShortcuts} from "../web/video-shortcuts.js";
import {mountTranscriptPanel} from "../web/transcript-inspector.js";

/** Native-clock fixture records observable transport effects without downloading or decoding media. */
function sourceVideo(overrides = {}) {
  return {hidden: false, readyState: 1, videoWidth: 640, duration: 28.137, currentTime: 4.123,
    paused: true, pauseCalls: 0, playCalls: 0, src: "blob:original",
    pause() { this.paused = true; this.pauseCalls++; },
    play() { this.paused = false; this.playCalls++; return Promise.resolve(); }, ...overrides};
}

test("panel dimensions clamp corrupted preferences and reserve monitor space", () => {
  assert.deepEqual(workspaceDimensions({inspectorWidth: NaN, timelineHeight: Infinity}), {
    inspectorWidth: 320, timelineHeight: 320,
  });
  assert.deepEqual(workspaceDimensions({inspectorWidth: 9999, timelineHeight: 9999}, 1000, 600), {
    inspectorWidth: 460, timelineHeight: 360,
  });
  assert.deepEqual(workspaceDimensions({inspectorWidth: -1, timelineHeight: -1}), {
    inspectorWidth: 240, timelineHeight: 140,
  });
});

test("timeline minimum reserves real rows after actual toolbar/footer and sticky ruler", () => {
  assert.equal(minimumTimelineHeight(203, 38, 128), 369);
  assert.equal(minimumTimelineHeight(203, 38, 96), 337);
  assert.equal(minimumTimelineHeight(300, 70, 128), 498);
  assert.equal(minimumTimelineHeight(60, 38, 0), 194);
});

test("timeline fit is a viewport calculation for full durations without a minute cap", () => {
  for (const duration of [28_137, 65_000, 900_000, 7_200_000]) {
    const scale = fitTimelineScale(duration, 1000);
    assert.ok(Math.abs(duration * scale - 760) < .001);
  }
  assert.equal(fitTimelineScale(0, 1000), .11);
  assert.equal(fitTimelineScale(1000, 0), .11);
});

test("frame transport preserves fractional source time and clamps both boundaries", () => {
  const video = sourceVideo();
  assert.equal(performVideoAction(video, "next", 40), true);
  assert.ok(Math.abs(video.currentTime - 4.163) < 1e-9);
  performVideoAction(video, "previous", 40);
  assert.ok(Math.abs(video.currentTime - 4.123) < 1e-9);
  performVideoAction(video, "next", 40, true);
  assert.ok(Math.abs(video.currentTime - 5.123) < 1e-9);
  performVideoAction(video, "end");
  assert.equal(video.currentTime, 28.12);
  assert.equal(videoActionEnabled(video, "next"), false);
  performVideoAction(video, "start");
  assert.equal(video.currentTime, 0);
  assert.equal(videoActionEnabled(video, "previous"), false);
  assert.equal(video.src, "blob:original");
});

test("End recovers a valid image position after native playback has reached EOF", () => {
  const video = sourceVideo({currentTime: 28.137, ended: true});
  assert.equal(videoActionEnabled(video, "end"), true);
  assert.equal(performVideoAction(video, "end"), true);
  assert.equal(video.currentTime, 28.12);
  assert.equal(video.paused, true);
  assert.equal(videoActionEnabled(video, "end"), false);
});

test("declared source FPS wins over the analysis grid and End remains before EOF", () => {
  const video = sourceVideo({dataset: {sourceFrameRate: "30", editorFrameStepMs: "40"}});
  assert.ok(Math.abs(videoFrameStepMs(video) - 1000 / 30) < 1e-9);
  assert.ok(lastVideoFrameTime(video) < video.duration);
  assert.ok(video.duration - lastVideoFrameTime(video) <= 1 / 30);
  performVideoAction(video, "next", 40);
  assert.ok(Math.abs(video.currentTime - (4.123 + 1 / 30)) < 1e-9);
});

test("media controls wait for real metadata and preserve unavailable native fallbacks", () => {
  for (const overrides of [{readyState: 0}, {duration: Infinity}, {videoWidth: 0}, {hidden: true}]) {
    const video = sourceVideo(overrides);
    assert.equal(performVideoAction(video, "play"), false);
    assert.equal(performVideoAction(video, "end"), false);
    assert.equal(video.pauseCalls + video.playCalls, 0);
  }
  const video = sourceVideo();
  performVideoAction(video, "play");
  assert.equal(video.playCalls, 1);
  performVideoAction(video, "play");
  assert.equal(video.pauseCalls, 1);
});

test("keyboard transport excludes separators, editable forms and modal layers", () => {
  const previous = globalThis.document;
  let keydown;
  let dialogOpen = false;
  const panel = {hidden: false, contains: () => true};
  const document = {body: {}, documentElement: {}, addEventListener: (_, handler) => { keydown = handler; },
    querySelector: () => dialogOpen ? {} : null};
  globalThis.document = document;
  try {
    const video = sourceVideo();
    mountVideoShortcuts(video, panel, () => 40);
    const event = target => ({target, key: "ArrowRight", code: "ArrowRight", preventDefault() { this.defaultPrevented = true; }});
    keydown(event({closest: () => ({})}));
    assert.equal(video.currentTime, 4.123);
    dialogOpen = true;
    keydown(event(document.body));
    assert.equal(video.currentTime, 4.123);
    dialogOpen = false;
    keydown(event(document.body));
    assert.ok(Math.abs(video.currentTime - 4.163) < 1e-9);
  } finally { globalThis.document = previous; }
});

test("inspector collapse retains the legacy preference and never hides the source", () => {
  const previous = globalThis.localStorage;
  const saved = new Map([["voice-studio:transcript-panel-collapsed", "true"]]);
  globalThis.localStorage = {getItem: key => saved.get(key) ?? null, setItem: (key, value) => saved.set(key, value)};
  const panel = {dataset: {}, dispatchEvent() {}};
  const attributes = new Map();
  const toggle = {dataset: {}, setAttribute: (key, value) => attributes.set(key, value)};
  try {
    mountTranscriptPanel(panel, toggle);
    assert.equal(panel.dataset.collapsed, "true");
    assert.equal(attributes.get("aria-expanded"), "false");
    assert.equal(toggle.title, attributes.get("aria-label"));
    assert.equal(toggle.dataset.hint, attributes.get("aria-label"));
    toggle.onclick();
    assert.equal(panel.dataset.collapsed, "false");
    assert.equal(attributes.get("aria-expanded"), "true");
    assert.equal(toggle.title, attributes.get("aria-label"));
    assert.equal(toggle.dataset.hint, attributes.get("aria-label"));
    assert.equal(saved.get("voice-studio:transcript-panel-collapsed"), "false");
  } finally { globalThis.localStorage = previous; }
});
