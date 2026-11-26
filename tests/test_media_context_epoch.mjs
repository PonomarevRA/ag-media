import test from "node:test";
import assert from "node:assert/strict";
import {spawnSync} from "node:child_process";
import {readFile} from "node:fs/promises";
import {fileURLToPath} from "node:url";
import vm from "node:vm";
import {studioWebTools} from "../web/web-mcp.js";
import {createUiBridge} from "../web/mcp-bridge.js";
import * as words from "../web/i18n.js";
import * as speakers from "../web/speaker-label.js";
import {mergeAnalysisPartial} from "../web/video-analysis.js";
import {retainedSpeakerEdits, speakerEditKey} from "../web/analysis-stage-selection.js";
import {mountTranscriptOcrChecks} from "../web/transcript-ocr-checks.js";
import {mountTranscriptSrt} from "../web/transcript-srt.js";

if (!process.argv.includes("--host-fixture")) {
  test("the actual transcript host fences real WebMCP reads on report mutations only", () => {
    const result = spawnSync(process.execPath, ["--experimental-vm-modules", fileURLToPath(import.meta.url), "--host-fixture"],
      {encoding: "utf8", env: process.env, timeout: 10_000});
    assert.equal(result.status, 0, result.stdout + result.stderr);
  });
}

/** Keep a minimal DOM while executing the unmodified host and its real event handlers. */
class Element extends EventTarget {
  constructor(tag = "div") {
    super();
    this.tagName = tag.toUpperCase();
    this.children = []; this.dataset = {}; this.attributes = new Map();
    this.value = ""; this.checked = false; this.hidden = false; this.options = [];
    this.selectedOptions = [{parentElement: {tagName: "SELECT"}}];
    this.classList = {add() {}, remove() {}, toggle() {}};
    this.duration = 5; this.videoWidth = 0; this.readyState = 0; this.currentTime = 0; this.paused = true;
  }
  append(...children) { this.children.push(...children); }
  replaceChildren(...children) { this.children = children; }
  setAttribute(name, value) { this.attributes.set(name, value); }
  hasAttribute(name) { return this.attributes.has(name); }
  removeAttribute(name) { this.attributes.delete(name); }
  getAttribute(name) { return this.attributes.get(name); }
  closest() { return new Element(); }
  querySelector() { return new Element(); }
  querySelectorAll() { return []; }
  contains() { return false; }
  focus() {}
  load() {}
  pause() { this.paused = true; }
  play() { this.paused = false; return Promise.resolve(); }
}

/** Stub rendering collaborators, not the host's report state, mutation methods or WebMCP tools. */
async function mountedHost(upload = true) {
  const {createContext, SourceTextModule, SyntheticModule} = vm;
  const controls = new Map();
  const find = id => {
    if (!controls.has(id)) controls.set(id, new Element());
    return controls.get(id);
  };
  find("media-speech-language").value = "ru";
  find("media-audio-classifier").value = "apple";
  const noWork = () => {};
  const panel = () => new Proxy({}, {get: () => noWork});
  let emitChunk, stopLiveCapture;
  const known = {...words, ...speakers, mergeAnalysisPartial, retainedSpeakerEdits, speakerEditKey,
    mountTranscriptOcrChecks, mountTranscriptSrt,
    editorFaceBindings: () => null, speakerProfiles: () => [],
    captureAudio: async (_kind, _device, chunk, stopped) => {
      emitChunk = chunk;
      stopLiveCapture = stopped;
      return {stop: stopped};
    },
    mountAnalysisStageSelection: () => ({setSource() {}, setReport() {}, setBusy() {}, query: () => ({})})};
  let progress, finish;
  const request = (route, body, options) => {
    if (route === "transcribe-chunk") return Promise.resolve({language: "ru", segments: [
      {startMs: 0, endMs: 1000, text: "Synthetic live cue"}], dialoguePauses: []});
    assert.equal(route, "transcribe-media");
    progress = options.onProgress;
    return new Promise(resolve => { finish = resolve; });
  };
  const document = {getElementById: find, createElement: tag => new Element(tag),
    createTextNode: text => ({textContent: text}), activeElement: null};
  const context = createContext({document, console, Blob, URL, Event, AbortController,
    setTimeout, clearTimeout, performance, addEventListener() {},
    navigator: {mediaDevices: {getUserMedia() {}}}, window: {AudioWorkletNode: class {}},
    localStorage: {getItem: () => null, setItem() {}},
    Option: class extends Element { constructor(text, value) { super("option"); this.textContent = text; this.value = value; } }});
  const source = await readFile(new URL("../web/media-transcript.js", import.meta.url), "utf8");
  const hostModule = new SourceTextModule(source, {context});
  await hostModule.link((specifier, referencingModule) => {
    const declaration = [...source.matchAll(/import\s*\{([^}]+)\}\s*from\s*"([^"]+)"/g)]
      .find(match => match[2] === specifier);
    const names = declaration[1].split(",").map(name => name.trim());
    return new SyntheticModule(names, function() {
      for (const name of names) this.setExport(name, known[name] ?? (name.startsWith("mount") ? panel : noWork));
    }, {context: referencingModule.context});
  });
  await hostModule.evaluate();
  const host = hostModule.namespace.mountMediaTranscript(request, () => "ru");
  if (upload) {
    const file = Object.assign(new Blob(["synthetic"], {type: "audio/wav"}), {name: "synthetic.wav"});
    find("media-upload").files = [file];
    find("media-upload").onchange();
  }
  return {host, find, progress: update => progress({partial: update}), finish: report => finish(report),
    chunk: () => emitChunk(new Blob(["synthetic"]), 0, 1000), stop: () => stopLiveCapture()};
}

/** Exercise actual upload, speaker-edit and background-report paths against the bridge's real tools. */
async function checkHost() {
  const f = await mountedHost();
  const report = {revision: 7, language: "ru", durationMs: 5000, srt: "", segments: [
    {startMs: 0, endMs: 1000, text: "Synthetic cue", speaker: 1}], dialoguePauses: []};
  f.progress({phase: "base", report});
  const ui = {inspect: () => ({speechLanguage: "ru"}), transcript: f.host.currentTranscript,
    contextRevision: () => f.host.contextRevision?.(), seek: ms => f.host.seek(ms)};
  const lifecycle = new AbortController();
  const tools = studioWebTools({ui, request: async () => ({})}, lifecycle.signal);
  const bridge = createUiBridge(tools, ui, () => "/tools");
  const command = name => ({name, arguments: {}, revision: bridge.inspect().revision, expiresAtMs: Date.now() + 2000});
  const read = name => tools.find(tool => tool.name === name).execute();
  const initial = bridge.inspect().revision;
  assert.equal(bridge.inspect().revision, initial, "unchanged inspections must be stable");
  const staleEdit = command("ui_transcript");
  const original = f.host.currentTranscript();
  const cueCollection = original.segments;
  const speakerSelect = f.find("media-segments").children[0].children[1].children[0];
  speakerSelect.value = "2";
  speakerSelect.onchange();
  assert.equal((await read("ui_transcript")).items[0].speakerNumber, 2);
  assert.equal(f.host.currentTranscript(), original, "manual edit must preserve source identity for editor fencing");
  const editResult = await bridge.execute(staleEdit, lifecycle.signal);
  const staleEvidence = command("ui_facial_expressions");
  f.progress({phase: "detail", layer: "visual", report: {videoEvidence: {facialExpressions: {
    status: "ok", intervals: [{startMs: 0, endMs: 500, faceId: 1, track: 1, label: "happy", confidence: .9}]}}}});
  assert.equal(f.host.currentTranscript().segments, cueCollection);
  assert.equal((await read("ui_facial_expressions")).items.length, 1);
  const evidenceResult = await bridge.execute(staleEvidence, lifecycle.signal);
  const beforeAudio = bridge.inspect().revision;
  f.progress({phase: "detail", layer: "audio", report: {audioLayers: {music: []}}});
  const afterAudio = bridge.inspect().revision;
  const beforeProfiles = afterAudio;
  f.progress({phase: "detail", layer: "voice", report: {speakerProfileIndex: [{id: "synthetic-profile"}]}});
  const afterProfiles = bridge.inspect().revision;
  assert.equal(f.host.currentTranscript().revision, 7, "UI context must not modify persisted CAS revision");
  const beforeSeek = bridge.inspect().revision;
  f.host.seek(500);
  f.find("media-preview").play();
  assert.equal(bridge.inspect().revision, beforeSeek, "seek and playback must retain report context");
  const observations = {editStatus: editResult.status, editError: editResult.error,
    evidenceStatus: evidenceResult.status, evidenceError: evidenceResult.error,
    audioInvalidated: afterAudio !== beforeAudio, profilesInvalidated: afterProfiles !== beforeProfiles};
  console.log(JSON.stringify(observations));
  assert.equal(editResult.error, "ui_context_changed");
  assert.equal(evidenceResult.error, "ui_context_changed");
  assert.notEqual(afterAudio, beforeAudio);
  assert.notEqual(afterProfiles, beforeProfiles);
  f.finish(report);
  await new Promise(setImmediate);
  const beforeReset = f.host.contextRevision();
  f.find("media-upload").onchange();
  assert.notEqual(f.host.contextRevision(), beforeReset, "source reset must advance the host epoch");
  lifecycle.abort();
  await checkLiveHost();
}

/** Live cues advance the same host epoch while repeated fresh-wrapper observations remain stable. */
async function checkLiveHost() {
  const f = await mountedHost(false);
  f.find("capture-kind").value = "input";
  f.find("capture-transcribe").checked = true;
  const before = f.host.contextRevision();
  await f.find("capture-form").onsubmit({preventDefault() {}});
  assert.notEqual(f.host.contextRevision(), before, "starting a live source must reset context");
  const started = f.host.contextRevision();
  f.chunk();
  await new Promise(setImmediate);
  assert.notEqual(f.host.contextRevision(), started, "publishing live cues must advance context");
  const ui = {inspect: () => ({speechLanguage: "ru"}), transcript: f.host.currentTranscript,
    contextRevision: f.host.contextRevision};
  const lifecycle = new AbortController();
  const bridge = createUiBridge(studioWebTools({ui, request: async () => ({})}, lifecycle.signal), ui, () => "/tools");
  const observed = bridge.inspect().revision;
  assert.equal(bridge.inspect().revision, observed);
  assert.equal((await bridge.execute({name: "ui_transcript", arguments: {}, revision: observed,
    expiresAtMs: Date.now() + 2000}, lifecycle.signal)).result.items[0].text, "Synthetic live cue");
  await f.stop();
  lifecycle.abort();
}

if (process.argv.includes("--host-fixture")) await checkHost();
