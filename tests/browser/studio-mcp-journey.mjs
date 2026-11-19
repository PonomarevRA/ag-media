/** Exercise the browser-native tool contract against an isolated Studio with a saved speech fixture. */
import assert from "node:assert/strict";
import {readFile} from "node:fs/promises";

const [base, recordingId] = process.argv.slice(2);
const origin = new URL(base);
assert.equal(origin.hostname, "127.0.0.1");
assert.match(recordingId ?? "", /^[a-f0-9]{32}$/, "Pass the recording ID from the test workspace.");
const {chromium} = await import(process.env.VOICE_PLAYWRIGHT_MODULE || "playwright");
const executablePath = process.env.VOICE_CHROMIUM || undefined;

/** Invoke the actual Chromium API with the argument style negotiated by the client. */
function invoke(page, name, args = {}) {
  return page.evaluate(async ({name, args}) => {
    const context = document.modelContext;
    const tool = (await context.getTools()).find(item => item.name === name);
    if (!tool) throw Error("Missing tool " + name);
    const input = typeof tool.inputSchema === "string" ? JSON.stringify(args) : args;
    const result = await context.executeTool(tool, input);
    return typeof result === "string" ? JSON.parse(result) : result;
  }, {name, args});
}

/** Verify the user-facing journey, saved graph reads, and HTTP origin protection. */
async function checkJourney(browser) {
  const page = await browser.newPage({viewport: {width: 1050, height: 900}, acceptDownloads: true});
  const errors = [];
  page.on("pageerror", error => errors.push(error.message));
  await page.goto(new URL("/?lang=ru&ui=en", origin).href);
  await page.waitForFunction(async () => (await document.modelContext?.getTools())?.some(tool => tool.name === "ui_people_count"));
  await page.waitForFunction(() => !document.querySelector("#record").disabled);
  assert.equal((await invoke(page, "ui_inspect")).page, "/");
  const settings = await invoke(page, "ui_navigate", {page: "settings"});
  assert.equal(settings.page, "/settings");
  assert.equal(settings.designProfile, "retro");
  assert.equal(settings.colorMode, "dark");
  assert.equal(await page.locator("#settings-link").getAttribute("aria-current"), "page");
  assert.equal((await invoke(page, "ui_navigate", {page: "history"})).page, "/history");
  assert.equal(await page.locator("#history-link").getAttribute("aria-current"), "page");
  assert.equal((await invoke(page, "ui_navigate", {page: "transcript"})).section, "media-title");
  assert.equal((await invoke(page, "ui_people_count")).people_count, true);
  assert.equal((await invoke(page, "ui_people_count", {people_count: false})).people_count, false);
  assert.equal(await page.locator("#media-people-count").isChecked(), false);
  assert.equal((await invoke(page, "ui_people_count", {people_count: true})).people_count, true);

  const analyzed = await invoke(page, "voice_ui_transcribe_recording", {id: recordingId});
  assert(analyzed.cueCount > 0, JSON.stringify(analyzed));
  const view = await invoke(page, "ui_inspect");
  assert(view.controls["media-srt"].visible && view.controls["media-json"].visible);
  const pauses = await invoke(page, "ui_pauses", {limit: 1});
  assert.equal(pauses.measured, true);
  if (process.env.VOICE_TEST_EXPECT_PAUSE === "1") assert(pauses.items.length);
  for (const pause of pauses.items) {
    assert(pause.startMs < pause.cutMs && pause.cutMs < pause.endMs);
  }
  const report = await invoke(page, "ui_transcript", {limit: 1});
  assert(report.items[0].text && report.items[0].startMs < report.items[0].endMs);
  if (report.nextOffset !== null) {
    const next = await invoke(page, "ui_transcript", {offset: report.nextOffset, limit: 1});
    assert(next.items[0].startMs > report.items[0].startMs);
  }
  await page.waitForFunction(() => document.querySelector("#media-preview").duration > 0);
  if (pauses.items.length) {
    await page.locator(".editor-clip-pause").first().click();
    const time = await page.evaluate(async expected => {
      const video = document.querySelector("#media-preview");
      const deadline = performance.now() + 1500;
      while (Math.abs(video.currentTime * 1000 - expected) >= 12 && performance.now() < deadline) {
        await new Promise(requestAnimationFrame);
      }
      return Math.round(video.currentTime * 1000);
    }, pauses.items[0].cutMs);
    assert(Math.abs(time - pauses.items[0].cutMs) < 12);
  }
  const seek = await invoke(page, "ui_seek", {time_ms: 500});
  assert.deepEqual(seek, {timeMs: 500, paused: true});
  const download = page.waitForEvent("download");
  await page.locator("#media-srt").click();
  assert.equal((await download).suggestedFilename(), "subtitles.srt");
  const jsonControl = page.locator("#media-json");
  const jsonDownload = page.waitForEvent("download");
  await jsonControl.focus();
  await page.keyboard.press("Enter");
  const jsonFile = await jsonDownload;
  assert.equal(jsonFile.suggestedFilename(), "transcript.json");
  const savedReport = JSON.parse(await readFile(await jsonFile.path(), "utf8"));
  assert.equal(savedReport.segments.length, analyzed.cueCount);
  assert.equal(savedReport.segments[0].text, report.items[0].text);
  assert.equal(await jsonControl.evaluate(button => button === document.activeElement), true);

  assert((await invoke(page, "ui_seek", {time_ms: -1})).error);
  assert((await invoke(page, "voice_ui_transcribe_recording", {id: recordingId})).error);
  assert.equal((await invoke(page, "ui_inspect")).cueCount, analyzed.cueCount);

  const snapshots = (await invoke(page, "voice_graph_list", {limit: 1})).snapshots;
  if (snapshots.length) {
    const snapshot_id = snapshots[0].snapshotId;
    const overview = await invoke(page, "voice_graph_overview", {snapshot_id});
    const window = await invoke(page, "voice_graph_timeline", {
      snapshot_id, start_ms: 0, end_ms: Math.min(overview.durationMs, 300_000), limit: 1,
    });
    assert(window.nodes.length === 1);
    const node = await invoke(page, "voice_graph_node", {snapshot_id, node_id: window.nodes[0].id});
    assert.equal(node.id, window.nodes[0].id);
    const query = node.label.match(/[\p{L}\p{N}]+/u)?.[0];
    if (query) {
      const hits = await invoke(page, "voice_graph_search", {snapshot_id, query, kind: node.kind, limit: 1});
      assert(hits.nodes?.length);
    }
    console.log("Graph catalog, coverage, search, timeline and node verified.");
  }
  const blocked = await fetch(new URL("/api/graph-read", origin), {
    method: "POST", headers: {Origin: "https://example.org", "Content-Type": "application/json", "X-Voice-Studio": "1"},
    body: JSON.stringify({action: "list", args: {}}),
  });
  assert.equal(blocked.status, 403);
  assert.deepEqual(errors, []);
  console.log("Native WebMCP journey and cross-origin rejection passed.");
  await page.close();
}

/** Check that the inspector fits and restores keyboard focus, with or without native registration. */
async function checkPanel(browser, expectedNative) {
  for (const [width, scale, locale] of [[1050, 100, "ru"], [320, 100, "en"], [1050, 200, "en"], [320, 200, "ru"]]) {
    const page = await browser.newPage({viewport: {width, height: 900}});
    await page.goto(new URL("/tools?ui=" + locale, origin).href);
    await page.evaluate(size => document.documentElement.style.fontSize = size + "%", scale);
    const trigger = page.locator("#mcp-open");
    await trigger.focus();
    await page.keyboard.press("Enter");
    const dialog = page.locator("#mcp-dialog");
    await dialog.waitFor({state: "visible"});
    assert.equal(await page.locator("#mcp-tool-list li").count(), 11);
    await page.locator("#mcp-check").click();
    await page.locator("#mcp-check-result").filter({hasText: /Studio responded|Студия отвечает/}).waitFor();
    const dimensions = await dialog.evaluate(node => {
      const {left, right} = node.getBoundingClientRect();
      return {left, right, overflow: node.scrollWidth - node.clientWidth};
    });
    assert(dimensions.left >= 0 && dimensions.right <= width && dimensions.overflow <= 1, JSON.stringify(dimensions));
    assert((await trigger.boundingBox()).height >= 44);
    const status = await page.locator("#mcp-status").textContent();
    assert.match(status, expectedNative ? /enabled|включён/ : /unavailable|недоступен/);
    await page.keyboard.press("Escape");
    assert(await dialog.isHidden());
    assert(await trigger.evaluate(node => node === document.activeElement));
    await page.close();
  }
  console.log(`Inspector layout at 320 CSS px and 200% text passed (native: ${expectedNative}).`);
}

const browser = await chromium.launch({headless: true, executablePath, args: ["--enable-blink-features=WebMCP,WebMCPTesting"]});
try {
  await checkJourney(browser);
  await checkPanel(browser, true);
} finally {
  await browser.close();
}
const fallback = await chromium.launch({headless: true, executablePath, args: ["--disable-blink-features=WebMCP,WebMCPTesting"]});
try {
  await checkPanel(fallback, false);
} finally {
  await fallback.close();
}
