import assert from "node:assert/strict";
import test from "node:test";
import {textFor} from "../web/i18n.js";
import {mixedSpeakerLabel} from "../web/speaker-label.js";

test("English labels format values without translating customer-provided text", () => {
  assert.equal(textFor("en", "Спикер {0}", 2), "Speaker 2");
  assert.equal(textFor("ru", "Спикер {0}", 2), "Спикер 2");
  assert.equal(textFor("en", "Готово · фраз: {0}", "Пример"), "Ready · Пример cues");
  assert.equal(textFor("en", "User-provided {0} text"), "User-provided {0} text");
});

test("overlap labels can follow the subtitle language independently of UI settings", () => {
  const cue = {overlap: true, overlaps: [{startMs: 0, endMs: 400, speakers: [2, 1]}]};
  assert.equal(mixedSpeakerLabel(cue, 200, "en"), "Overlapping speech · speakers 1, 2");
  assert.equal(mixedSpeakerLabel(cue, 200, "ru"), "Голоса пересекаются · спикеры 1, 2");
});
