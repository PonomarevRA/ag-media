import test from "node:test";
import assert from "node:assert/strict";
import {mergeAnalysisPartial, readAnalysisStream, analysisStageStatus} from "../web/video-analysis.js";
import {studioWebTools} from "../web/web-mcp.js";

test("WebMCP accepts both boolean people count settings and rejects string flags", async () => {
  let selected = true;
  const ui = {peopleCount(value) {
    if (value !== undefined) selected = value;
    return {people_count: selected};
  }};
  const settings = studioWebTools({ui}, new AbortController().signal)
    .find(tool => tool.name === "ui_people_count");
  assert.equal((await settings.execute()).people_count, true);
  assert.equal((await settings.execute({people_count: false})).people_count, false);
  assert.ok((await settings.execute({people_count: "true"})).error);
  assert.equal((await settings.execute()).people_count, false);
  assert.equal((await settings.execute({people_count: true})).people_count, true);
});

function stream(...parts) {
  return new Response(new ReadableStream({
    start(controller) {
      for (const part of parts) controller.enqueue(part);
      controller.close();
    },
  }));
}

test("NDJSON progress survives splitting inside a UTF-8 label", async () => {
  const payload = new TextEncoder().encode('{"stage":"whisper","detail":"Речь"}\n{"result":{"segments":[]}}\n');
  const split = payload.indexOf(208) + 1;
  const events = [];
  const result = await readAnalysisStream(stream(payload.subarray(0, split), payload.subarray(split)),
                                          event => events.push(event));
  assert.deepEqual(events, [{stage: "whisper", detail: "Речь"}]);
  assert.deepEqual(result, {segments: []});
});

test("NDJSON propagates server errors and rejects truncated results", async () => {
  const bytes = message => new TextEncoder().encode(message);
  await assert.rejects(readAnalysisStream(stream(bytes('{"error":"Не удалось"}\n')), () => {}), /Не удалось/);
  await assert.rejects(readAnalysisStream(stream(bytes('{"result":')), () => {}), /прервался/);
});

test("base, audio and scene reports arrive separately without replacing unrelated layers", async () => {
  const cue = {startMs: 200, endMs: 600, text: "Кадр"};
  const frames = [
    {partial: {phase: "base", report: {segments: [cue], durationMs: 1000}}},
    {partial: {phase: "detail", layer: "audio", report: {audioLayers: [{kind: "music"}], segments: []}}},
    {partial: {phase: "detail", layer: "scene", report: {objectTimeline: [{label: "стол"}]}}},
    {result: {segments: [cue]}},
  ];
  const bytes = new TextEncoder().encode(frames.map(row => JSON.stringify(row)).join("\n") + "\n");
  let current = null;
  const updates = [];
  await readAnalysisStream(stream(bytes), event => {
    current = mergeAnalysisPartial(current, event.partial);
    updates.push(current);
  });
  assert.equal(updates[0].analysisComplete, false);
  assert.deepEqual(updates[0].segments, [cue]);
  assert.equal(updates[1].segments, updates[0].segments);
  assert.equal(updates[2].audioLayers, updates[1].audioLayers);
  assert.deepEqual(updates[2].objectTimeline, [{label: "стол"}]);
});

test("detail errors retain the prior base and malformed or duplicate patches are ignored", async () => {
  const report = {segments: [], durationMs: 1000};
  assert.equal(mergeAnalysisPartial(null, {phase: "detail", layer: "scene", report}), null);
  let current = null;
  const bytes = new TextEncoder().encode(JSON.stringify({partial: {phase: "base", report}})
    + '\n{"error":"Слой недоступен"}\n');
  await assert.rejects(readAnalysisStream(stream(bytes), event => {
    current = mergeAnalysisPartial(current, event.partial);
  }), /Слой недоступен/);
  assert.deepEqual(current, {...report, analysisComplete: false});
  assert.equal(mergeAnalysisPartial(current, {phase: "base", report}), current);
  assert.equal(mergeAnalysisPartial(current, {phase: "detail", layer: "visual", report: {segments: null}}), current);
});

test("WebMCP distinguishes preliminary transcript pages without changing completed pages", async () => {
  const report = {segments: [], language: "ru", analysisComplete: false};
  const tools = studioWebTools({ui: {transcript: () => report}}, new AbortController().signal);
  const transcript = tools.find(tool => tool.name === "ui_transcript");
  const preliminary = await transcript.execute();
  assert.equal(preliminary.analysisComplete, false);
  delete report.analysisComplete;
  assert.deepEqual(await transcript.execute(), {items: [], language: "ru", nextOffset: null, total: 0});
});


test("Turbo progress shows measured windows and a separate retry pass", async () => {
  const {recoveryProgress} = await import("../web/video-analysis.js");
  assert.equal(recoveryProgress({windowsDone: 5, windowsTotal: 11}), "фрагменты 5 из 11");
  assert.equal(recoveryProgress({windowsDone: 0, windowsTotal: 2, recoveryPass: "alternate"}),
    "другой язык · фрагменты 0 из 2");
  assert.equal(recoveryProgress({windowsDone: 2, windowsTotal: 1}), "");
});

test("checkpoint hits, shared inference and real recomputation have distinct truthful labels", () => {
  assert.equal(analysisStageStatus({state: "done", reused: true, cacheHit: true}), "Из кэша");
  assert.equal(analysisStageStatus({state: "partial", reused: true, cacheHit: true}), "Из кэша · частично");
  assert.equal(analysisStageStatus({state: "skipped", reused: true, cacheHit: true}), "Из кэша · пропущен");
  assert.equal(analysisStageStatus({state: "done", reused: true}, "recompute"), "Готово в общем процессе");
  assert.equal(analysisStageStatus({state: "running"}, "recompute"), "Пересчитывается");
  assert.equal(analysisStageStatus({state: "done", elapsedMs: 1250}, "recompute"), "Пересчитано · 1.3 с");
  assert.equal(analysisStageStatus({state: "failed"}, "recompute"), "Ошибка");
  assert.equal(analysisStageStatus({state: "skipped", reason: "Нет весов Light-ASD"}), "Пропущен · Нет весов Light-ASD");
  const reason = "Одна голосовая группа, коротких реплик для проверки нет";
  assert.equal(analysisStageStatus({state: "skipped", reason}), `Пропущен · ${reason}`);
  assert.equal(analysisStageStatus({state: "skipped", cacheHit: true, reason}), `Из кэша · пропущен · ${reason}`);
});
