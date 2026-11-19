const ACTIVE_KEY = "voiceLiveActive";
const ERROR_KEY = "voiceLiveMessage";
const popupUrl = chrome.runtime.getURL("popup.html");
const offscreenUrl = chrome.runtime.getURL("offscreen.html");

let operation = false;
let closingSession;

/** Open exactly one DOM context for the tab media stream. */
async function ensureOffscreen() {
  const contexts = await chrome.runtime.getContexts({
    contextTypes: ["OFFSCREEN_DOCUMENT"], documentUrls: [offscreenUrl],
  });
  if (!contexts.length) {
    await chrome.offscreen.createDocument({
      url: "offscreen.html",
      reasons: ["USER_MEDIA"],
      justification: "Process the active tab audio and video explicitly selected by the user.",
    });
  }
}

async function notifyPopup() {
  await chrome.runtime.sendMessage({target: "popup", type: "state"}).catch(() => {});
}

async function activeSession() {
  const data = await chrome.storage.session.get(ACTIVE_KEY);
  return data[ACTIVE_KEY];
}

/** Finish the same session that initiated cleanup, retaining a stoppage reason. */
async function finish(active, message = "") {
  const current = await activeSession();
  if (!active || current?.session !== active.session || closingSession === active.session) return;
  closingSession = active.session;
  try {
    await Promise.allSettled([
      chrome.tabs.sendMessage(active.tabId, {type: "voice-live-stop", session: active.session}),
      chrome.action.setBadgeText({tabId: active.tabId, text: ""}),
      chrome.offscreen.closeDocument(),
    ]);
    await chrome.storage.session.set({[ERROR_KEY]: message});
    await chrome.storage.session.remove(ACTIVE_KEY);
    await notifyPopup();
  } finally {
    closingSession = undefined;
  }
}

/** Obtain the tab stream only after a user invokes the extension. */
async function startCapture(message) {
  if (await activeSession()) throw new Error("Остановите текущий анализ, чтобы открыть другую вкладку.");
  const [tab] = await chrome.tabs.query({active: true, currentWindow: true});
  if (!tab?.id || !/^https?:\/\//.test(tab.url || "")) {
    throw new Error("Откройте обычную веб-страницу с видео.");
  }
  if (!["ru", "en"].includes(message.language) || typeof message.token !== "string" || !message.token.trim() || message.token.length > 256) {
    throw new Error("Проверьте токен студии и язык речи.");
  }
  const session = crypto.randomUUID().replaceAll("-", "");
  const active = {tabId: tab.id, session};
  await chrome.storage.session.set({[ACTIVE_KEY]: active, [ERROR_KEY]: ""});
  try {
    const streamId = await chrome.tabCapture.getMediaStreamId({targetTabId: tab.id});
    await ensureOffscreen();
    await chrome.scripting.executeScript({target: {tabId: tab.id}, files: ["overlay.js"]});
    await chrome.tabs.sendMessage(tab.id, {type: "voice-live-start", session});
    const result = await chrome.runtime.sendMessage({
      target: "offscreen", type: "start", streamId, session, token: message.token, language: message.language,
    });
    if (!result?.ok) throw new Error(result?.error || "Не получилось запустить захват.");
    await chrome.storage.session.set({voiceLiveToken: message.token, voiceLiveLanguage: message.language});
    await Promise.allSettled([
      chrome.action.setBadgeBackgroundColor({tabId: tab.id, color: "#a23a27"}),
      chrome.action.setBadgeText({tabId: tab.id, text: "LIVE"}),
    ]);
    return {ok: true};
  } catch (error) {
    await finish(active, error.message);
    throw error;
  }
}

/** Stop tracks and requests before dismissing the session in the browser UI. */
async function stopCapture(message = "") {
  const active = await activeSession();
  if (!active) return {ok: true};
  await chrome.runtime.sendMessage({target: "offscreen", type: "stop", session: active.session}).catch(() => {});
  await finish(active, message);
  return {ok: true};
}

/** Reconcile the saved session state with the current offscreen document. */
async function status() {
  const [active, data] = await Promise.all([
    activeSession(), chrome.storage.session.get(ERROR_KEY),
  ]);
  if (active) {
    const contexts = await chrome.runtime.getContexts({
      contextTypes: ["OFFSCREEN_DOCUMENT"], documentUrls: [offscreenUrl],
    });
    if (!contexts.length && !operation) {
      await finish(active, "Захват завершился. Можно запустить его снова.");
      return {active: false, message: "Захват завершился. Можно запустить его снова."};
    }
  }
  return {active: Boolean(active), message: active ? "Анализ идёт." : data[ERROR_KEY]};
}

/** Serialize UI commands; tab media belongs only to the offscreen document. */
async function fromPopup(message) {
  if (message.type === "status") return status();
  if (operation) throw new Error("Дождитесь завершения текущего действия.");
  operation = true;
  try {
    if (message.type === "start") return await startCapture(message);
    if (message.type === "stop") return await stopCapture();
    throw new Error("Неизвестное действие.");
  } finally {
    operation = false;
  }
}

/** Route updates from the current capture to its original tab. */
async function fromOffscreen(message) {
  const active = await activeSession();
  if (!active || active.session !== message.session) return {ok: false};
  if (message.type === "ended") {
    await finish(active, message.message);
    return {ok: true};
  }
  if (message.type === "update") {
    await chrome.tabs.sendMessage(active.tabId, {
      type: "voice-live-update", session: active.session, report: message.report,
    }).catch(() => {});
    return {ok: true};
  }
  return {ok: false};
}

chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  if (message.target !== "worker" || sender.id !== chrome.runtime.id) return false;
  const handler = sender.url === popupUrl ? fromPopup : sender.url === offscreenUrl ? fromOffscreen : null;
  if (!handler) return false;
  handler(message).then(sendResponse, error => sendResponse({ok: false, error: error.message}));
  return true;
});

chrome.tabs.onRemoved.addListener(tabId => {
  activeSession().then(active => {
    if (active?.tabId === tabId) return stopCapture("Вкладка закрыта.");
  }).catch(() => {});
});

chrome.tabs.onUpdated.addListener((tabId, change) => {
  if (change.status !== "loading") return;
  activeSession().then(active => {
    if (active?.tabId === tabId) return stopCapture("Вкладка загрузила новую страницу.");
  }).catch(() => {});
});
