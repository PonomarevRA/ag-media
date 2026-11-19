import test from "node:test";
import assert from "node:assert/strict";
import {filterMediaMods, mediaModProviders} from "../web/studio-mods.js";
import {modSettingValue, readModSetting, modSettingsForm} from "../web/studio-mod-fields.js";

const items = [
  {id: "builtin.whisper", origin: "builtin", enabled: true,
    manifest: {name: "Whisper Small", description: "Speech recognition", capabilities: [{id: "asr.decode", kind: "provider"}]}},
  {id: "my.words", origin: "user", enabled: false,
    manifest: {name: "Словарь", capabilities: [{id: "analysis.words", kind: "stage"}]}},
  {id: "my.asr", origin: "user", enabled: true,
    manifest: {name: "Alternative ASR", capabilities: [{id: "asr.decode", kind: "provider"}]}},
  {id: "broken", origin: "user", error: "Invalid manifest"},
];

test("mod catalog search includes capability IDs and safely lists invalid metadata without mutation", () => {
  assert.deepEqual(filterMediaMods(items, "ASR.DECODE").map(item => item.id), ["builtin.whisper", "my.asr"]);
  assert.deepEqual(filterMediaMods(items, "", "user").map(item => item.id), ["my.words", "my.asr", "broken"]);
  assert.deepEqual(filterMediaMods(items, "слОВарь").map(item => item.id), ["my.words"]);
  assert.equal(items.length, 4);
});

test("provider choices do not treat additional stages as replacements", () => {
  const providers = mediaModProviders(items);
  assert.deepEqual([...providers.keys()], ["asr.decode"]);
  assert.deepEqual(providers.get("asr.decode").map(item => item.id), ["builtin.whisper", "my.asr"]);
});

test("settings preserve typed enums, zero and false while rejecting empty or nonfinite numeric input", () => {
  assert.equal(modSettingValue("0", {type: "integer"}), 0);
  assert.equal(modSettingValue(false, {type: "boolean"}), false);
  assert.equal(modSettingValue('"0"', {type: "string", enum: ["0", "1"]}), "0");
  assert.deepEqual(modSettingValue('{"rate":2}', {type: "object"}), {rate: 2});
  for (const raw of ["", "  ", "Infinity", "NaN", "1.2"]) {
    assert.throws(() => modSettingValue(raw, {type: "integer"}));
  }
});

test("omitted optional settings stay absent, while explicit zero and false are retained", () => {
  const absent = {required: false, present: false, changed: false};
  for (const [raw, type] of [["", "integer"], ["", "object"], ["", "array"], ["", "string"], [false, "boolean"]]) {
    assert.deepEqual(readModSetting(raw, {type}, absent), {include: false});
  }
  assert.deepEqual(readModSetting("0", {type: "integer"}, {...absent, changed: true}), {include: true, value: 0});
  assert.deepEqual(readModSetting(false, {type: "boolean"}, {...absent, changed: true}), {include: true, value: false});
  assert.deepEqual(readModSetting("", {type: "integer"}, {...absent, present: true, changed: true}), {include: false});
  assert.throws(() => readModSetting("", {type: "integer"}, {...absent, required: true}));
});

test("a selected removed provider keeps its capability available for an explicit reset", () => {
  const providers = mediaModProviders([], {"asr.decode": "removed.mod"});
  assert.deepEqual([...providers.entries()], [["asr.decode", []]]);
});


test("production form preserves explicit and default null values on sibling edits", () => {
  // This DOM boundary drives the real form submit handler; it is not browser/AT evidence.
  class Field extends EventTarget {
    children = []; dataset = {}; value = "";
    append(...nodes) { this.children.push(...nodes); }
    setAttribute() {}
    setCustomValidity(message) { this.error = message; }
    reportValidity() { return !this.error; }
  }
  const previous = globalThis.document;
  globalThis.document = {createElement: () => new Field()};
  try {
    for (const required of [false, true]) {
      for (const variant of ["explicit", "enum", "default"]) {
        const definition = {type: "null"};
        if (variant === "enum") definition.enum = [null];
        if (variant === "default") definition.default = null;
        const settings = variant === "default" ? {count: 2} : {sentinel: null, count: 2};
        const item = {id: "null.settings", settings, manifest: {settingsSchema: {
          type: "object", properties: {sentinel: definition, count: {type: "integer"}},
          required: required ? ["sentinel"] : [],
        }}};
        let saved;
        const form = modSettingsForm(item, value => { saved = value; });
        const sentinel = form.children[0].children[1];
        const sibling = form.children[1].children[1];
        assert.equal(sentinel.value, "null", `${variant}, required=${required}`);
        sibling.value = "3";
        sibling.dispatchEvent(new Event("input"));
        form.dispatchEvent(new Event("submit", {cancelable: true}));
        assert.equal(sentinel.error, undefined);
        assert.deepEqual(saved, {sentinel: null, count: 3});
      }
    }
  } finally {
    if (previous === undefined) delete globalThis.document;
    else globalThis.document = previous;
  }
});
