const form = document.querySelector("#capture-form");
const token = document.querySelector("#token");
const language = document.querySelector("#language");
const start = document.querySelector("#start");
const stop = document.querySelector("#stop");
const status = document.querySelector("#status");

/** Read the background session state when the popup is opened or notified. */
async function renderStatus(message = "") {
  const state = await chrome.runtime.sendMessage({target: "worker", type: "status"});
  form.hidden = Boolean(state.active);
  stop.hidden = !state.active;
  status.textContent = message || state.message || "";
}

const options = await chrome.storage.session.get(["voiceLiveToken", "voiceLiveLanguage"]);
token.value = options.voiceLiveToken ?? "";
language.value = options.voiceLiveLanguage ?? "ru";
await renderStatus();

form.addEventListener("submit", async event => {
  event.preventDefault();
  start.disabled = true;
  status.textContent = "Подключаем вкладку…";
  try {
    const value = token.value.trim();
    if (!value) throw new Error("Введите токен локальной студии.");
    const result = await chrome.runtime.sendMessage({
      target: "worker", type: "start", token: value, language: language.value,
    });
    if (!result.ok) throw new Error(result.error);
    await renderStatus();
  } catch (error) {
    status.textContent = error.message;
  } finally {
    start.disabled = false;
  }
});

stop.addEventListener("click", async () => {
  stop.disabled = true;
  try {
    const result = await chrome.runtime.sendMessage({target: "worker", type: "stop"});
    await renderStatus(result.ok ? "Анализ остановлен." : result.error);
  } catch (error) {
    status.textContent = error.message;
  } finally {
    stop.disabled = false;
  }
});

chrome.runtime.onMessage.addListener(message => {
  if (message.target === "popup" && message.type === "state") renderStatus().catch(() => {});
});
