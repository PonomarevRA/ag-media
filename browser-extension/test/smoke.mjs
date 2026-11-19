// This test substitutes the browser's privileged capture API. The popup, offscreen
// recorder, worker messaging, HTTP, overlay and stop path run in actual Chromium.
import assert from "node:assert/strict";
import {promisify} from "node:util";
import {execFile} from "node:child_process";
import {createServer} from "node:http";
import {cp, mkdtemp, readFile, rm, writeFile} from "node:fs/promises";
import {tmpdir} from "node:os";
import {dirname, join, resolve} from "node:path";
import {fileURLToPath} from "node:url";

const executable = promisify(execFile);
const extension = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const {chromium} = await import(process.env.VOICE_PLAYWRIGHT || "playwright");
const integration = process.env.VOICE_LIVE_INTEGRATION === "1";
const integrationPort = Number(process.env.VOICE_LIVE_PORT || 8766);
assert(Number.isInteger(integrationPort) && integrationPort > 0 && integrationPort <= 65535, "Invalid VOICE_LIVE_PORT");
const folder = await mkdtemp(join(tmpdir(), "voice-live-sandbox-"));
const copy = join(folder, "extension");
const accepted = [];
let serverError;
let context;
let voice;
let rejectNextChunk = false;
const delay = ms => new Promise(resolve => setTimeout(resolve, ms));

const server = createServer(async (request, response) => {
  if (request.url === "/fixture") {
    response.end("<!doctype html><title>Voice Live synthetic fixture</title><p>Test tab</p>");
    return;
  }
  if (request.url === "/favicon.ico") {
    response.writeHead(204);
    response.end();
    return;
  }
  if (request.url === "/voice.wav" && voice) {
    response.writeHead(200, {"Content-Type": "audio/wav"});
    response.end(voice);
    return;
  }
  try {
    const url = new URL(request.url, "http://127.0.0.1");
    assert.equal(url.pathname, "/api/live-chunk");
    assert.equal(request.method, "POST");
    assert.equal(request.headers["x-voice-live-token"], "smoke-token");
    assert.equal(request.headers["content-type"], "video/webm");
    assert.match(url.searchParams.get("session"), /^[a-f0-9]{32}$/);
    assert.equal(url.searchParams.get("lang"), "en");
    const startMs = Number(url.searchParams.get("start_ms"));
    assert(Number.isInteger(startMs) && startMs >= 0);
    const chunks = [];
    for await (const chunk of request) chunks.push(chunk);
    if (rejectNextChunk) {
      response.writeHead(415, {"Content-Type": "application/json"});
      response.end(JSON.stringify({error: "Unsupported media."}));
      return;
    }
    const content = Buffer.concat(chunks);
    assert.equal(content.subarray(0, 4).toString("hex"), "1a45dfa3");
    const path = join(folder, "chunk-" + accepted.length + ".webm");
    await writeFile(path, content);
    accepted.push({path, startMs});
    const number = accepted.length;
    // Force the next encoded chunk to be dropped rather than queued.
    if (number === 1) await delay(3450);
    response.writeHead(200, {"Content-Type": "application/json"});
    response.end(JSON.stringify({
      durationMs: 3000,
      segments: [
        {startMs: startMs + 400, endMs: startMs + 1400, text: "Synthetic caption " + number},
        {startMs: startMs + 1450, endMs: startMs + 2300, text: "Synthetic answer " + number},
      ],
      faceFrames: [{timeMs: startMs + (number === 1 ? 1100 : 2850), faces: [{id: 1, box: [.2, .2, .25, .4]}]}],
    }));
  } catch (error) {
    serverError = error;
    response.writeHead(500);
    response.end();
  }
});
await new Promise(resolve => server.listen(0, "127.0.0.1", resolve));
const origin = `http://127.0.0.1:${server.address().port}`;

async function prependFile(name, prefix, replace = null) {
  const path = join(copy, name);
  const source = await readFile(path, "utf8");
  await writeFile(path, prefix + (replace ? source.replace(...replace) : source));
}

/** Poll the extension storage so asynchronous Chrome API calls finish before matching. */
async function waitForBrowserReport(popup, matches, timeoutMs) {
  const deadline = Date.now() + timeoutMs;
  let report;
  while (Date.now() < deadline) {
    const serialized = await popup.evaluate(async () => {
      const {smokeReport} = await chrome.storage.local.get("smokeReport");
      return JSON.stringify(smokeReport ?? null);
    });
    report = JSON.parse(serialized);
    if (report && matches(report)) return report;
    await delay(100);
  }
  assert.fail("Expected a browser report. Last warnings: " + JSON.stringify(report?.warnings ?? []));
}

try {
  let token = "smoke-token";
  if (integration) {
    token = process.env.VOICE_LIVE_TOKEN || (await readFile(join(extension, "../data/studio/live-token.txt"), "utf8")).trim();
    const fixture = join(folder, "test-speech.aiff");
    await executable("say", ["-v", "Samantha", "-o", fixture, "The morning train is leaving."]);
    const wav = join(folder, "test-speech.wav");
    await executable("ffmpeg", ["-v", "error", "-i", fixture, "-af", "apad", "-t", "3", "-ac", "1", "-ar", "16000", wav]);
    voice = await readFile(wav);
  }
  await cp(extension, copy, {recursive: true});
  await prependFile("worker.js", `
    const query = chrome.tabs.query.bind(chrome.tabs);
    chrome.tabs.query = async () => (await query({url: "${origin}/fixture"}));
    chrome.tabCapture.getMediaStreamId = async () => "synthetic-smoke-source";
    chrome.runtime.onMessage.addListener(message => {
      if (message.target === "worker" && message.type === "update") {
        void chrome.storage.local.set({smokeReport: message.report});
      }
    });
  `);
  await prependFile("offscreen.js", `
    navigator.mediaDevices.getUserMedia = async () => {
      const canvas = document.createElement("canvas");
      canvas.width = 320;
      canvas.height = 180;
      const ctx = canvas.getContext("2d");
      let toggle = false;
      const draw = () => {
        ctx.fillStyle = (toggle = !toggle) ? "#997342" : "#254444";
        ctx.fillRect(0, 0, canvas.width, canvas.height);
      };
      draw();
      setInterval(draw, 100);
      const audio = new AudioContext();
      const destination = audio.createMediaStreamDestination();
      if (${integration}) {
        const fixture = await fetch("${origin}/voice.wav");
        const speech = audio.createBufferSource();
        speech.buffer = await audio.decodeAudioData(await fixture.arrayBuffer());
        speech.loop = true;
        speech.loopEnd = 3;
        speech.connect(destination);
        speech.start();
      } else {
        const tone = audio.createOscillator();
        const gain = audio.createGain();
        gain.gain.value = 0.02;
        tone.connect(gain).connect(destination);
        tone.start();
      }
      await audio.resume();
      return new MediaStream([...canvas.captureStream(6).getVideoTracks(), ...destination.stream.getAudioTracks()]);
    };
  `, ["http://127.0.0.1:8766", integration ? `http://127.0.0.1:${integrationPort}` : origin]);
  context = await chromium.launchPersistentContext(join(folder, "profile"), {
    channel: "chromium",
    headless: true,
    ...(process.env.VOICE_CHROMIUM ? {executablePath: process.env.VOICE_CHROMIUM} : {}),
    args: [
      `--disable-extensions-except=${copy}`, `--load-extension=${copy}`,
      "--autoplay-policy=no-user-gesture-required",
    ],
  });
  const worker = context.serviceWorkers()[0] || await context.waitForEvent("serviceworker");
  const id = new URL(worker.url()).host;
  const tab = await context.newPage();
  const popup = await context.newPage();
  const errors = [];
  for (const page of [tab, popup]) page.on("pageerror", error => errors.push(error.message));
  await popup.goto(`chrome-extension://${id}/popup.html`);
  await popup.locator("#token").fill(token);
  await popup.locator("#language").selectOption("en");
  if (!integration) {
    await popup.locator("#start").click();
    await popup.getByText("Откройте обычную веб-страницу с видео.").waitFor();
  }
  await tab.goto(origin + "/fixture");
  await popup.locator("#start").click();
  await popup.locator("#stop").waitFor({timeout: 10000}).catch(async error => {
    throw new Error(await popup.locator("#status").textContent(), {cause: error});
  });
  await tab.locator("#voice-studio-live-overlay").waitFor();
  let integrationSummary;
  if (integration) {
    const smokeReport = await waitForBrowserReport(popup, report =>
      report.segments?.some(segment => /morning train|train is leaving/i.test(segment.text)), 20000);
    assert(smokeReport.durationMs > 0);
    assert(smokeReport.segments.every(segment => segment.startMs >= smokeReport.startMs));
    assert(smokeReport.faceFrames.length >= 1);
    assert(smokeReport.timingsMs.total > 0);
    integrationSummary = {
      durationMs: smokeReport.durationMs,
      speechSegments: smokeReport.segments.length,
      faceFrames: smokeReport.faceFrames.length,
      timingsMs: smokeReport.timingsMs,
      requestMs: smokeReport.requestMs,
      elapsedMs: smokeReport.elapsedMs,
    };
  } else {
    const cdp = await context.newCDPSession(tab);
    const outline = async () => JSON.stringify(await cdp.send("DOM.getDocument", {depth: -1, pierce: true}));
    for (let tries = 0; tries < 130 && !serverError; tries++) {
      if ((await outline()).includes("Synthetic caption 1")) break;
      await delay(100);
    }
    if (serverError) throw serverError;
    const staleOutline = await outline();
    assert(staleOutline.includes("Synthetic caption 1"));
    assert(!staleOutline.includes("Лицо 1 ·"));
    for (let tries = 0; tries < 100 && !serverError; tries++) {
      if ((await outline()).includes("Synthetic caption 2")) break;
      await delay(100);
    }
    if (serverError) throw serverError;
    const freshOutline = await outline();
    assert(freshOutline.includes("Synthetic caption 2"));
    assert(freshOutline.includes("Synthetic answer 2"));
    assert(freshOutline.includes("пропущено 1"));
    assert(freshOutline.includes("Лицо 1 ·"));
    await tab.evaluate(() => window.dispatchEvent(new Event("resize")));
    assert(!(await outline()).includes("Лицо 1 ·"));
    assert.equal(accepted.length, 2);
    assert(accepted[1].startMs - accepted[0].startMs >= 5000);
    for (const chunk of accepted) {
      const {stdout} = await executable("ffprobe", [
        "-v", "error", "-show_entries", "stream=codec_name", "-of", "csv=p=0", chunk.path,
      ]);
      assert.match(stdout, /^vp8\s+opus/m);
    }
  }
  await popup.locator("#stop").click();
  await tab.locator("#voice-studio-live-overlay").waitFor({state: "detached"});
  await popup.locator("#start").waitFor();
  await delay(3200);
  if (!integration) {
    assert.equal(accepted.length, 2);
    rejectNextChunk = true;
    await popup.locator("#start").click();
    await popup.locator("#stop").waitFor();
    await waitForBrowserReport(popup, report => report.warnings?.includes("Студия вернула код 415."), 8000);
    await tab.close();
    await popup.getByText("Вкладка закрыта.").waitFor();
  }
  assert.deepEqual(errors, []);
  if (integration) console.log("PASS Chromium MV3 with local Voice Studio:", JSON.stringify(integrationSummary));
  else console.log("PASS Chromium MV3: independent VP8/Opus, bounded queue, captions, face freshness, stop");
} finally {
  await context?.close();
  await new Promise(resolve => server.close(resolve));
  await rm(folder, {recursive: true, force: true});
}
