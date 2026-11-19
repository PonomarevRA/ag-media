import test from "node:test";
import assert from "node:assert/strict";
import {createUiBridge, mountMcpBridge} from "../web/mcp-bridge.js";
import {studioWebTools} from "../web/web-mcp.js";

/** Use real semantic definitions with a small synthetic UI controller. */
function fixture() {
  const state = {page: "/", speechLanguage: "ru"};
  let report = null;
  const ui = {inspect: () => ({...state}), transcript: () => report,
    navigate: page => { state.page = "/" + page; return {...state}; }, peopleCount: value => ({people_count: value}),
    seek: time => ({time_ms: time})};
  const lifecycle = new AbortController();
  const tools = studioWebTools({ui, request: async () => ({})}, lifecycle.signal);
  const bridge = createUiBridge(tools, ui, () => state.page);
  const command = (name, args = {}, revision = bridge.inspect().revision) =>
    ({name, arguments: args, revision, expiresAtMs: Date.now() + 1000});
  return {state, bridge, tools, ui, command, lifecycle, replaceReport: value => { report = value; }};
}

test("shared schema rejects invalid navigation, scripts and unknown parameters", async () => {
  const f = fixture();
  assert.equal((await f.bridge.execute(f.command("ui_navigate", {page: "settings"}), f.lifecycle.signal)).status, "ok");
  assert.equal(f.state.page, "/settings");
  const invalid = await f.bridge.execute(f.command("ui_navigate", {page: "delete", script: "evil"}), f.lifecycle.signal);
  assert.equal(invalid.status, "error");
  assert.equal(f.state.page, "/settings");
  assert.equal((await f.bridge.execute(f.command("eval"), f.lifecycle.signal)).error, "ui_tool_not_allowed");
  assert(!f.bridge.catalogue.some(row => row.name === "voice_ui_transcribe_recording"));
});

test("context changes fence stale route and material commands", async () => {
  const f = fixture();
  const stale = f.command("ui_seek", {time_ms: 100});
  f.replaceReport({sourceDigest: "new-source", segments: []});
  assert.equal((await f.bridge.execute(stale, f.lifecycle.signal)).error, "ui_context_changed");
  const navigation = f.command("ui_navigate", {page: "history"});
  const result = await f.bridge.execute(navigation, f.lifecycle.signal);
  assert.equal(result.status, "ok");
  assert.notEqual(result.revision, navigation.revision);
});

test("live capture wrappers keep a stable context until the cue collection changes", async () => {
  const f = fixture();
  const segments = [{startMs: 0, endMs: 1000, text: "Synthetic capture"}];
  f.ui.transcript = () => ({segments, language: "ru"});
  const command = f.command("ui_transcript");
  assert.equal(f.bridge.inspect().revision, command.revision);
  const result = await f.bridge.execute(command, f.lifecycle.signal);
  assert.equal(result.status, "ok");
  assert.equal(result.result.items[0].text, "Synthetic capture");
  const stale = f.command("ui_transcript");
  segments.push({startMs: 1000, endMs: 2000, text: "Next chunk"});
  assert.equal((await f.bridge.execute(stale, f.lifecycle.signal)).error, "ui_context_changed");
  const replaced = f.command("ui_transcript");
  f.ui.transcript = () => ({segments: [...segments], language: "ru"});
  assert.equal((await f.bridge.execute(replaced, f.lifecycle.signal)).error, "ui_context_changed");
});

test("material identity and explicit revision fence reused cue collections", async () => {
  const f = fixture();
  const report = {id: "first", revision: 1, segments: [{startMs: 0, endMs: 1000, text: "Synthetic material"}]};
  f.replaceReport(report);
  const first = f.command("ui_transcript");
  report.id = "second";
  assert.equal((await f.bridge.execute(first, f.lifecycle.signal)).error, "ui_context_changed");
  const second = f.command("ui_transcript");
  report.segments[0].text = "Explicitly revised material";
  report.revision++;
  assert.equal((await f.bridge.execute(second, f.lifecycle.signal)).error, "ui_context_changed");
});

test("expired and revoked commands do not invoke an action", async () => {
  const f = fixture();
  const expired = {...f.command("ui_navigate", {page: "settings"}), expiresAtMs: Date.now() - 1};
  assert.equal((await f.bridge.execute(expired, f.lifecycle.signal)).error, "ui_command_expired");
  f.lifecycle.abort();
  assert.equal((await f.bridge.execute(f.command("ui_navigate", {page: "settings"}), f.lifecycle.signal)).error, "ui_bridge_disconnected");
  assert.equal(f.state.page, "/");
});

test("late disconnect from an obsolete lifecycle cannot erase the new connection status", async () => {
  const f = fixture();
  const before = {window: globalThis.window, location: globalThis.location,
    sessionStorage: globalThis.sessionStorage, fetch: globalThis.fetch};
  const enabled = Object.assign(new EventTarget(), {checked: true});
  const status = {textContent: ""};
  let count = 0, releaseOldDisconnect;
  const oldDisconnect = new Promise(resolve => { releaseOldDisconnect = resolve; });
  const first = new AbortController(), second = new AbortController();
  globalThis.window = {};
  globalThis.location = {pathname: "/", search: "", hash: ""};
  globalThis.sessionStorage = {getItem: () => null, setItem() {}};
  globalThis.fetch = async (url, options) => {
    if (url.endsWith("register")) return {ok: true, json: async () => ({windowId: String(++count).repeat(32), token: "token"})};
    if (url.endsWith("disconnect")) {
      if (JSON.parse(options.body).windowId.startsWith("1")) await oldDisconnect;
      return {ok: true, json: async () => ({ok: true})};
    }
    return new Promise((resolve, reject) => {
      options.signal.addEventListener("abort", () => reject(Error("aborted")), {once: true});
    });
  };
  const find = id => id === "mcp-bridge-enabled" ? enabled : status;
  try {
    mountMcpBridge(find, f.tools, f.ui, first.signal);
    await new Promise(setImmediate);
    first.abort();
    mountMcpBridge(find, f.tools, f.ui, second.signal);
    await new Promise(setImmediate);
    assert(status.textContent.includes("22222222"));
    releaseOldDisconnect();
    await new Promise(setImmediate);
    assert(status.textContent.includes("22222222"));
  } finally {
    first.abort(); second.abort(); releaseOldDisconnect();
    await new Promise(setImmediate);
    Object.assign(globalThis, before);
  }
});
