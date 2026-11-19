import assert from "node:assert/strict";
import {resolveObjectURL} from "node:buffer";
import {test} from "node:test";
import {mountTranscriptJSONDownload} from "../web/media-transcript.js";

test("download serializes the latest visible transcript and releases the temporary URL", async context => {
  const downloaded = [];
  context.mock.timers.enable({apis: ["setTimeout"]});
  const originalDocument = globalThis.document;
  globalThis.document = {
    createElement(tag) {
      assert.equal(tag, "a");
      return {click() { downloaded.push(this); }};
    },
  };
  context.after(() => { globalThis.document = originalDocument; });

  const button = {};
  let reads = 0;
  let current;
  mountTranscriptJSONDownload(button, () => {
    reads++;
    return current;
  });
  assert.equal(reads, 0);
  button.onclick();
  assert.equal(downloaded.length, 0);

  current = {
    language: "ru",
    segments: [{startMs: 400, endMs: 1100, text: "Исходная реплика", speaker: 1}],
    srt: "1\n00:00:00,400 --> 00:00:01,100\nИсходная реплика\n",
  };
  current = {
    ...current,
    segments: [{...current.segments[0], text: "Исправленная реплика", speakerNumber: 2}],
    videoEvidence: {timeline: {stepMs: 40, frames: [[], [[1, .25, .3, .4, .5, 1, 2]]]}},
  };
  assert.equal(reads, 1);
  button.onclick();
  assert.equal(reads, 2);
  assert.equal(downloaded.length, 1);
  const [file] = downloaded;
  assert.equal(file.download, "transcript.json");
  const blob = resolveObjectURL(file.href);
  assert.equal(blob.type, "application/json");
  assert.equal(await blob.text(), JSON.stringify(current, null, 2));

  current = undefined;
  button.onclick();
  assert.equal(downloaded.length, 1);
  context.mock.timers.tick(30000);
  assert.equal(resolveObjectURL(file.href), undefined);
});
