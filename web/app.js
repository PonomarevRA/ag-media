import {apiText, installInterfaceLocale, uiLocale, uiText, whisperLanguageChoices, whisperLanguageName} from "./i18n.js";
import {monitorTape, TrackDeck} from "./deck.js";
import {ProcessingPanel} from "./processing.js";
import {mountPlayer} from "./player.js";
import {mountProsodyPilot, mountStressComparison, stressTakeLabel} from "./stress-comparison.js";
import {mountEditorWorkspace} from "./editor-workspace.js";
import {mountMediaTranscript} from "./media-transcript.js";
import {readAnalysisStream} from "./video-analysis.js";
import {mountStudioWebMcp} from "./web-mcp.js";
import {openMorphemeGraph, renderBlocks} from "./morpheme-graph.js";

import {attachStudioIcon} from "./studio-icons.js";
import {mountStudioNavigation} from "./studio-navigation.js";
import {installStudioDesign} from "./studio-design.js";
import {mountStudioSettings} from "./studio-settings.js";
import {mountStudioStorage} from "./studio-storage.js";
import {createAutoMap} from "./auto-synthesis.js";

const byId = id => document.getElementById(id);
const state = {data: null, language: "ru", refs: new Set(), models: new Set(),
  checkpoints: {}, assembly: [], blocks: [], profile: 0, audio: null, previewUrl: null,
  busy: false, toolsCaptureRevision: 0};
installInterfaceLocale(document, () => Boolean(state.audio || byId("media-preview").hasAttribute("src")));
installStudioDesign(document);
const navigation = mountStudioNavigation(byId);
const previewPlayer = mountPlayer(byId("preview"), uiText("Прослушивание записи"));
previewPlayer.root.hidden = true;
const deck = new TrackDeck(() => processing.syncTracks());
document.addEventListener("studio-design-change", () => deck.draw());
const processing = new ProcessingPanel(deck, () => state.language);
const jobNames = {queued: uiText("В очереди"), running: uiText("В работе"), done: uiText("Готово"), failed: uiText("Ошибка")};
const autoMap = createAutoMap(byId("auto-map"), audioPlayer, id => act(async () => {
  await request("auto-stop", {id});
  await refresh();
}));
const renderStress = mountStressComparison({request, refresh, act, compare: compareTracks});
const renderProsodyPilot = mountProsodyPilot(request, compareTracks);
const mediaTools = mountMediaTranscript(request, () => state.language);
mountEditorWorkspace(document);
const settings = mountStudioSettings(byId, request, {
  materialUrl, openMaterial: openSavedMaterial, dismissMaterial: mediaTools.dismissMaterial,
});
const storage = mountStudioStorage(byId, request);
const transcriptionLanguages = loadTranscriptionLanguages();
const engineDescriptions = {qwen: uiText("Первая выбранная запись с текстом"),
  chatterbox: uiText("Первая выбранная запись"), f5: uiText("Первая выбранная запись"), fish: uiText("Первые пять записей с текстом"), coqui: uiText("Первые пять выбранных записей")};

function element(tag, text, properties = {}) {
  const node = Object.assign(document.createElement(tag), properties);
  if (text) node.textContent = text;
  return node;
}

/** Populate both file-language pickers from the same catalogue the server validates. */
async function loadTranscriptionLanguages() {
  try {
    const {languages} = await request("transcription-languages");
    if (!Array.isArray(languages)) return;
    const other = whisperLanguageChoices(languages.filter(code => /^[a-z]{2,3}$/.test(code) && !["ru", "en"].includes(code)));
    for (const id of ["media-speech-language", "settings-material-language"]) {
      const select = byId(id);
      const selected = select.value;
      const group = select.querySelector("optgroup");
      group.replaceChildren(...other.map(code => element("option", whisperLanguageName(code), {value: code})));
      if ([...select.options].some(option => option.value === selected)) select.value = selected;
    }
  } catch {
    // The initial Russian, English and frequently tested choices are still available.
  }
}

function notice(message, error = false) {
  byId("status").textContent = message;
  byId("status").dataset.error = String(error);
}

async function request(path, body, extra = {}) {
  const {text, onProgress, signal, language = state.language, ...flags} = extra;
  const query = new URLSearchParams({lang: language, ...flags});
  const headers = {"X-Voice-Studio": "1"};
  if (text !== undefined) headers["X-Voice-Transcript"] = encodeURIComponent(text);
  const options = body === undefined ? {} : {method: "POST", headers};
  if (body instanceof Blob) options.body = body;
  else if (body !== undefined) {
    headers["Content-Type"] = "application/json";
    options.body = JSON.stringify(body);
  }
  if (onProgress) query.set("progress", "1");
  if (signal) options.signal = signal;
  const response = await fetch(`/api/${path}?${query}`, options);
  if (!response.ok) throw new Error(apiText((await response.json()).error) || uiText("Запрос не выполнен."));
  if (onProgress) return readAnalysisStream(response, onProgress);
  return ["export", "video-preview", "video-export", "video-dub", "person-masks"].includes(path) ? response.blob() : response.json();
}

/** Create a player for a stored WAV with its known server duration. */
function audioPlayer(id, name, seconds) {
  const audio = element("audio", "", {preload: "none", src: `/media/${id}.wav`});
  return mountPlayer(audio, name, seconds).root;
}

function choice(text, checked, disabled, callback) {
  const label = element("label");
  const input = element("input", "", {type: "checkbox", checked, disabled});
  input.onchange = () => callback(input);
  label.append(input, document.createTextNode(text));
  return label;
}

function fillSelect(select, options, value) {
  select.replaceChildren(...options.map(([key, text]) => element("option", text, {value: key})));
  select.value = value;
  if (select.selectedIndex < 0) select.selectedIndex = 0;
}

/** A position and duration distinguish recordings whose transcripts start the same way. */
function recordingName(item, index) {
  const kind = item.evaluation_only ? uiText("Проверка") : uiText("Запись");
  return uiText`${kind} ${index + 1} · ${item.seconds.toFixed(1)} с`;
}

function renderEngines() {
  const cards = [];
  for (const [id, model] of Object.entries(state.data.engines)) {
    const card = element("section", "", {className: "engine"});
    card.dataset.agWidth = "content";
    card.dataset.agWidthMinEm = "6";
    card.append(choice(model.name, state.models.has(id), !model.ready, input => {
      if (input.checked) state.models.add(id);
      else state.models.delete(id);
      syncButtons();
    }));
    card.append(element("p", apiText(model.reason), {className: model.ready ? "ready" : ""}));
    card.append(element("small", engineDescriptions[id]));
    if (id === "chatterbox") {
      card.append(element("small", uiText("Пилот · лицензия MLX-токенизатора требует подтверждения для поставки")));
    }
    const versions = state.data.checkpoints.filter(item => item.model === id);
    if (versions.length && id !== "fish") {
      const label = element("label", uiText("Веса модели"), {className: "select-field weights-field"});
      const select = element("select", "", {id: `weights-${id}`});
      label.htmlFor = select.id;
      select.setAttribute("aria-label", uiText`Веса ${model.name}`);
      fillSelect(select, [["", uiText("Базовые веса")], ...versions.map(item => [item.id, uiText`Профиль v${item.profile}`])],
        state.checkpoints[id] || "");
      select.onchange = () => { state.checkpoints[id] = select.value; };
      label.append(select);
      card.append(label);
    }
    cards.push(card);
  }
  byId("engines").replaceChildren(...cards);
}

/** Render the page of saved takes with its active filters and latest recordings first. */
function renderHistory() {
  const recordings = state.data.recordings;
  const search = byId("history-search").value.trim().toLocaleLowerCase().replaceAll("ё", "е");
  const kind = byId("history-kind").value;
  const takes = [];
  for (const [index, item] of recordings.entries()) {
    if (kind !== "all" && Boolean(item.evaluation_only) !== (kind === "evaluation")) continue;
    if (!item.text.toLocaleLowerCase().replaceAll("ё", "е").includes(search)) continue;
    const row = element("li", "", {className: "take"});
    const name = recordingName(item, index);
    const header = element("div", "", {className: "list-entry-heading"});
    header.dataset.agWidth = "content";
    header.dataset.agWidthMinEm = "7";
    header.append(element("strong", name));
    if (item.created) {
      const when = new Date(item.created.replace(" ", "T") + "Z");
      header.append(element("time", when.toLocaleString(uiLocale, {dateStyle: "medium", timeStyle: "short",
        timeZone: "Europe/Moscow"}), {dateTime: when.toISOString(), className: "muted"}));
    }
    const source = item.evaluation_only ? null :
      choice(uiText("В профиль"), state.refs.has(item.id), false, input => {
        if (input.checked && state.refs.size >= 500) {
          input.checked = false;
          notice(uiText("В один профиль можно включить до 500 записей."), true);
          return;
        }
        if (input.checked) state.refs.add(item.id);
        else state.refs.delete(item.id);
        state.profile = 0;
        byId("profiles").value = "";
        updateSampleCount();
        renderTraining();
        syncButtons();
      });
    if (source) {
      source.className = "profile-choice-toggle";
      source.querySelector("input").setAttribute("aria-label", uiText`В профиль — ${name}`);
      header.append(source);
    }
    row.append(header, element("p", item.text, {className: "take-transcript"}), audioPlayer(item.id, name, item.seconds));
    const actions = element("div", "", {className: "actions entry-actions"});
    actions.dataset.agWidth = "content";
    actions.dataset.agWidthMinEm = "7";
    const use = element("button", uiText("Использовать для синтеза"), {type: "button", className: "primary history-use"});
    use.onclick = () => {
      chooseComparison(item.id);
      showPage("/", "compare-title", true);
    };
    const compare = trackChoice(item.id, name + " · " + item.text.slice(0, 42));
    compare.addEventListener("change", event => {
      if (event.target.checked) showPage("/", "track-title", true);
    });
    const remove = removeButton("", "recording", item.id);
    remove.classList.add("studio-icon");
    remove.dataset.icon = "trash";
    remove.dataset.hint = uiText("Удалить запись");
    remove.setAttribute("aria-label", uiText`Удалить запись — ${name}`);
    attachStudioIcon(remove);
    actions.append(use, compare, remove);
    row.append(actions);
    takes.push(row);
  }
  byId("recordings").replaceChildren(...takes.reverse());
  const empty = byId("recordings-empty");
  empty.hidden = Boolean(takes.length);
  empty.textContent = recordings.length ? uiText("По этим фильтрам записей нет.") : uiText("Здесь появятся сохранённые записи.");
}

/** Mirror the active profile size in the studio. */
function updateSampleCount() {
  const count = state.data.recordings.filter(item => !item.evaluation_only).length;
  byId("sample-count").textContent = uiText`Выбрано ${state.refs.size} из ${count} записей корпуса`;
}

function renderRecordings() {
  const {profiles} = state.data;
  updateSampleCount();
  if (!byId("history-page").hidden) renderHistory();
  fillSelect(byId("profiles"), [["", uiText("Текущий набор")],
    ...profiles.map(item => [String(item.version), uiText`Версия ${item.version} · ${JSON.parse(item.refs).length} записей`])],
    String(state.profile || ""));
  renderReading();
  renderTraining();
}

/** Compare transcripts without treating capitalization or punctuation as content changes. */
function comparableReading(text) {
  return (text || "").toLowerCase().replaceAll("ё", "е").replace(/[^\p{L}\p{N}]+/gu, " ").trim();
}

/** Return the visible story text, or the legacy prompt when reading the corpus. */
function activeReading() {
  const story = state.data?.readingStory;
  const mode = byId("reading-mode").value;
  if (story && mode === "story-full") return story.text;
  if (story && mode === "story-parts") return story.parts[Number(byId("story-part").value)]?.text;
  return state.data?.prompt;
}

/** Keep the guided selector and history-derived completion separate from legacy prompt IDs. */
function renderReading(selectNext = false) {
  const {readingStory: story, prompt, promptNumber, promptCount, promptDone, promptBand} = state.data;
  const mode = byId("reading-mode").value;
  const useStory = Boolean(story && mode !== "corpus");
  const partsMode = useStory && mode === "story-parts";
  const partSelect = byId("story-part");
  byId("reading-mode-field").hidden = !story;
  byId("story-part-field").hidden = !partsMode;
  byId("story-hint").hidden = !useStory;
  if (useStory) {
    const recorded = new Set(state.data.recordings.filter(row => !row.evaluation_only)
      .map(row => comparableReading(row.text)));
    const done = text => recorded.has(comparableReading(text));
    const count = story.parts.filter(part => done(part.text)).length;
    const next = story.parts.findIndex(part => !done(part.text));
    const selected = partSelect.value;
    if (partsMode) {
      fillSelect(partSelect, story.parts.map(({label, text}, index) =>
        [String(index), `${index + 1}. ${label}${done(text) ? uiText(" · записан") : ""}`]),
        (selectNext || !selected) && next >= 0 ? String(next) : selected);
    }
    byId("prompt-progress").textContent = partsMode ? uiText`Записано отрывков: ${count} из ${story.parts.length}`
      : (done(story.text) ? uiText("История записана целиком") : uiText("Одна запись"));
    byId("story-hint").textContent = partsMode
      ? uiText("Сохраняй отрывки по одному. Для F5 выбери первым чистый дубль короче 12 секунд.")
      : uiText("Прочитай историю одним дублем, уложившись в 43 секунды. Для F5 лучше выбрать режим по частям.");
  } else {
    byId("prompt-progress").textContent = prompt
      ? uiText`Фраза ${promptNumber} из ${promptCount} · ${apiText(promptBand)} · записано ${promptDone}`
      : uiText`Записано ${promptDone} из ${promptCount}`;
  }
  byId("prompt-check").hidden = useStory || !prompt || location.pathname === "/tools";
  byId("next-prompt").textContent = activeReading() || uiText("Все подготовленные фразы записаны. Можно ввести свой текст.");
}

function renderTraining() {
  const trainers = state.data.trainers;
  byId("train-help").textContent = trainers.length
    ? uiText("Сохрани версию профиля с двумя или более записями. Каждая версия создаёт отдельный checkpoint.")
    : uiText("Для изменения весов подключи локальный рецепт. Записи и версии можно собирать уже сейчас.");
  byId("train-actions").replaceChildren(...trainers.map(model => {
    const button = element("button", uiText`Дообучить ${state.data.engines[model].name}`);
    button.disabled = state.busy || !state.profile;
    button.onclick = () => act(async () => {
      await request("train", {model, profile: state.profile});
      await refresh();
      notice(uiText("Дообучение поставлено в очередь."));
    });
    return button;
  }));
}

function assemblyChanged(id, selected) {
  if (selected && state.assembly.length < 12) state.assembly.push(id);
  else if (selected) notice(uiText("В дорожку можно включить до 12 дублей."), true);
  else state.assembly = state.assembly.filter(key => key !== id);
  renderAssemblyOrder();
  syncButtons();
}

/** Display the current assembly selection in playback order. */
function renderAssemblyOrder() {
  const labels = state.data.jobs.filter(job => job.kind === "synthesis");
  byId("assembly-order").textContent = state.assembly.map(key => {
    const model = labels.find(job => job.id === key)?.model;
    return state.data.modelNames?.[model] || state.data.engines[model]?.name || "";
  }).join(" → ");
}

/** Add the same item to the shared comparison transport. */
function trackChoice(id, name, caption = uiText("В сравнение")) {
  const label = choice(caption, deck.rows.has(id), false, input => {
    if (input.checked && deck.rows.size >= 6) {
      input.checked = false;
      notice(uiText("Для сравнения можно открыть до шести дорожек."), true);
      return;
    }
    deck.select(id, name, input.checked);
  });
  label.className = "track-choice";
  label.dataset.audioId = id;
  return label;
}

/** Place the accessible action on a labelled retro key with a visible focus hint. */
function resultKey(control, name, icon) {
  const key = element("span", "", {className: "result-key"});
  key.dataset.hint = name;
  const symbol = element("span", icon === "compare" ? "A/B" : "", {className: `result-icon icon-${icon}`});
  symbol.setAttribute("aria-hidden", "true");
  control.classList.add("result-key-control");
  const action = control.matches("label") ? control.querySelector("input") : control;
  action.setAttribute("aria-label", name);
  control.append(symbol);
  key.append(control);
  return key;
}

/** Keep the file actions next to their player and reuse the shared A/B checkbox. */
function resultActions(job, name) {
  const actions = element("div", "", {className: "result-keys"});
  actions.setAttribute("role", "group");
  actions.setAttribute("aria-label", uiText("Действия с WAV"));
  const download = element("a", "", {className: "download",
    href: `/media/${job.id}.wav`, download: `${job.model}-${job.id.slice(0, 8)}.wav`});
  const compare = trackChoice(job.id, name + " · " + job.text.slice(0, 42), "");
  const remove = removeButton("", "result", job.id);
  actions.append(resultKey(download, uiText("Скачать WAV"), "download"),
    resultKey(compare, uiText("В сравнение"), "compare"), resultKey(remove, uiText("Удалить WAV"), "delete"));
  return actions;
}

/** Remove an item only after the user confirms its scope. */
function removeButton(text, kind, id) {
  const button = element("button", text, {type: "button", className: "delete-item"});
  button.onclick = () => removeItem(kind, id, button);
  return button;
}

/** Confirm deletion and return keyboard focus to a surviving control. */
async function removeItem(kind, id, trigger) {
  const scope = {recording: uiText("эту запись и её расшифровку"), profile: uiText("выбранную версию профиля"), result: uiText("этот результат")};
  if (!confirm(uiText`Удалить ${scope[kind]}?`)) return;
  const restoreFocus = trigger && document.activeElement === trigger;
  await act(async () => {
    await request("delete", {kind, id});
    if (kind === "profile") state.profile = 0;
    if (kind === "recording" && state.comparison === id) chooseComparison("");
    state.refs.delete(id);
    if (state.lastSaved === id) state.lastSaved = "";
    state.assembly = state.assembly.filter(key => key !== id);
    state.blocks = state.blocks.filter(part => part.id !== id);
    renderBlockBin();
    renderAssemblyOrder();
    await deck.select(id, "", false);
    await refresh();
    notice(uiText("Удалено."));
  });
  if (restoreFocus && document.activeElement === document.body) {
    if (trigger.isConnected) trigger.focus({preventScroll: true});
    else if (kind === "result") byId("jobs-title").focus();
    else if (kind === "recording") byId("history-search").focus();
  }
}

function resultCard(job) {
  const card = element("li", "", {className: "candidate", tabIndex: -1});
  const name = stressTakeLabel(job.variant) || state.data.modelNames?.[job.model] || state.data.engines[job.model]?.name || uiText("Готовая дорожка");
  const heading = job.kind === "training" ? uiText`Дообучение · ${name}` : name;
  const header = element("div", "", {className: "list-entry-heading"});
  const variants = {baseline: "", stress: uiText(" · Ударения"), pauses: uiText(" · Паузы"),
    "tone-question": uiText(" · Вопрос"), "tone-exclamation": uiText(" · Восклицание"), "word-retry": uiText(" · Пересинтез")};
  const details = `${jobNames[job.status]}${variants[job.variant] || ""}${job.seconds ? uiText` · ${job.seconds.toFixed(1)} с` : ""}`;
  header.append(element("h4", heading), element("span", details, {className: "entry-state mono"}));
  card.append(header, element("p", job.text));
  if (job.error) card.append(element("p", apiText(job.error), {className: "muted"}));
  if (job.checkpoint_id) {
    card.append(element("p", `Checkpoint v${job.checkpoint_profile} · ${job.checkpoint_id.slice(0, 8)}`,
      {className: "muted"}));
  }
  const encoder = job.speaker_encoder === "eres2net" || (!job.speaker_encoder &&
    ["qwen", "chatterbox"].includes(job.model)) ? "ERes2Net" : "XTTS";
  if (job.comparison_id && job.status === "done" && job.variant !== "word-retry") {
    const voice = job.speaker_cosine == null ? uiText("нет оценки") : job.speaker_cosine.toFixed(3);
    const words = job.word_error_rate == null ? uiText("нет оценки") : `${(job.word_error_rate * 100).toFixed(1)}%`;
    card.append(element("p", uiText`Косинусная близость голоса: ${voice}${job.speaker_cosine == null ? "" : " · " + encoder}`),
      element("p", uiText`Ошибки слов, Whisper: ${words}`, {className: "muted"}),
      element("p", `Seed ${job.seed}`, {className: "muted mono"}));
  }
  if (job.variant === "word-retry" && job.status === "done") {
    if (job.word_error_rate !== null) {
      card.append(element("p", uiText`Ошибки слов, Whisper: ${(job.word_error_rate * 100).toFixed(1)}%`, {className: "muted"}));
    }
    if (job.comparison_id) {
      const score = job.speaker_cosine == null ? uiText("оценка недоступна") : job.speaker_cosine.toFixed(3).replace(".", ",");
      card.append(element("p", uiText`Близость голоса во фрагменте · ${score}${job.speaker_cosine == null ? "" : " · " + encoder}`, {className: "muted"}));
    }
  }
  if (job.status === "failed" && job.kind !== "training") {
    card.append(removeButton(uiText("Удалить результат"), "result", job.id));
  }
  if (job.status !== "done" || job.kind === "training") return card;
  card.append(audioPlayer(job.id, heading, job.seconds), resultActions(job, name));
  const baseline = state.data.jobs.find(other => other.id === job.baseline_id && other.status === "done");
  if (baseline) {
    const excerpt = job.variant === "word-retry";
    const baselineName = excerpt ? uiText("Полный дубль") : uiText("Базовый дубль");
    const label = excerpt ? uiText("A/B · фрагмент и полный дубль") : uiText("A/B с базовым дублем");
    const compare = element("button", label, {type: "button"});
    compare.onclick = () => compareTracks(baseline.id, baselineName, job.id, name + (variants[job.variant] || ""));
    card.append(compare);
  }
  if (job.kind === "synthesis") {
    const graph = element("button", uiText("Посмотреть граф морфем"), {type: "button", className: "graph-trigger"});
    graph.onclick = () => openMorphemeGraph(job, request, (part, label) => approveBlock(part, label, name),
      () => refresh(false, true), () => {
        if (graph.isConnected) graph.focus({preventScroll: true});
        else byId("jobs-title").focus();
      }, focusRetryBatch);
    card.append(graph);
    if (job.comparison_id && state.data.recordings.some(item => item.id === job.comparison_id)) {
      const label = job.variant === "word-retry" ? uiText("A/B · фрагмент и полная запись") : uiText("A/B с исходником");
      const compare = element("button", label, {type: "button"});
      compare.onclick = () => compareTake(job, name);
      card.append(compare);
    }
    const label = element("label", uiText("Сходство с голосом"), {className: "select-field"});
    const select = element("select", "", {className: "rating", id: `rating-${job.id}`});
    label.htmlFor = select.id;
    fillSelect(select, [["", uiText("Оценить")], ...[1, 2, 3, 4, 5].map(score => [String(score), uiText`${score} из 5`])],
      String(job.score || ""));
    select.options[0].disabled = true;
    select.onchange = () => saveRating(card, select, job);
    label.append(select);
    const include = choice(uiText("В дорожку"), state.assembly.includes(job.id), false, input => {
      assemblyChanged(job.id, input.checked);
      input.checked = state.assembly.includes(job.id);
    });
    include.className = "include";
    card.append(label, include);
  }
  return card;
}

/** Save the audition rating and retain keyboard focus across the polling refresh. */
async function saveRating(card, select, job) {
  const restoreFocus = document.activeElement === select;
  const previous = String(job.score || "");
  const score = Number(select.value);
  await act(async () => {
    try {
      await request("rate", {id: job.id, score});
      job.score = score;
      card.dataset.signature = jobSignature(job);
      notice(uiText("Оценка сохранена в MLflow."));
    } catch (error) {
      select.value = previous;
      throw error;
    }
  });
  if (restoreFocus && document.activeElement === document.body) {
    document.getElementById(select.id)?.focus({preventScroll: true});
  }
}

/** Identify meaningful job changes without replacing stable controls between polls. */
function jobSignature(job) {
  const originalExists = state.data.recordings.some(item => item.id === job.comparison_id);
  const baselineReady = state.data.jobs.some(other => other.id === job.baseline_id && other.status === "done");
  return JSON.stringify([job, originalExists, baselineReady]);
}

/** Put a candidate and its source on the shared transport, then move focus there. */
function compareTake(job, name) {
  compareTracks(job.comparison_id, uiText("Исходник"), job.id, name);
}

/** Set up the deck for an original or baseline paired with a new candidate. */
function compareTracks(left, leftName, right, rightName) {
  deck.clear();
  deck.select(left, leftName, true);
  deck.select(right, rightName, true);
  syncTrackChoices();
  byId("track-title").focus();
}

/** Keep the checkboxes aligned after replacing the deck selection. */
function syncTrackChoices() {
  for (const label of document.querySelectorAll(".track-choice")) {
    label.querySelector("input").checked = deck.rows.has(label.dataset.audioId);
  }
}

/** Jump to this batch without searching through unrelated results. */
function focusRetryBatch(ids) {
  const target = [...byId("jobs").children].find(card => ids.includes(card.dataset.key));
  (target || byId("jobs-title")).focus();
}

function renderJobs() {
  const current = new Map([...byId("jobs").children].map(card => [card.dataset.key, card]));
  for (const job of state.data.jobs) {
    const signature = jobSignature(job);
    const existing = current.get(job.id);
    if (existing?.dataset.signature === signature) {
      current.delete(job.id);
      continue;
    }
    const next = resultCard(job);
    next.dataset.key = job.id;
    next.dataset.signature = signature;
    if (existing) {
      const focusedCard = document.activeElement === existing;
      existing.replaceWith(next);
      if (focusedCard) next.focus({preventScroll: true});
      current.delete(job.id);
    } else {
      const position = state.data.jobs.indexOf(job);
      byId("jobs").insertBefore(next, byId("jobs").children[position] || null);
    }
  }
  for (const item of current.values()) item.remove();
  byId("jobs-empty").hidden = Boolean(state.data.jobs.length);
  const counts = {queued: 0, running: 0, done: 0, failed: 0};
  for (const job of state.data.jobs) counts[job.status]++;
  const waiting = counts.queued + counts.running;
  const summary = [
    waiting && uiText`В работе: ${waiting}`,
    counts.done && uiText`Готово: ${counts.done}`,
    counts.failed && uiText`Ошибки: ${counts.failed}`
  ].filter(Boolean).join(" · ");
  if (byId("jobs-summary").textContent !== summary) byId("jobs-summary").textContent = summary;
}

function syncButtons() {
  const working = state.busy || Boolean(state.recorder);
  const transcript = byId("transcript");
  const length = transcript.value.trim().length;
  byId("transcript-length").hidden = length <= transcript.maxLength;
  byId("save-recording").disabled = working || !state.audio || !byId("consent").checked ||
    !length || length > transcript.maxLength;
  byId("check-transcript").disabled = working || !length;
  byId("review-prosody").disabled = working || !byId("test-text").value.trim();
  byId("try-stress").disabled = working || state.language !== "ru" || !state.models.has("f5");
  byId("try-pauses").disabled = byId("try-tone").disabled = working || !["f5", "coqui"].some(model => state.models.has(model));
  byId("save-profile").disabled = working || !state.refs.size;
  byId("export").disabled = working || !state.refs.size;
  const missing = !state.refs.size ? uiText("Добавь запись в профиль.")
    : !state.models.size ? uiText("Выбери модель для синтеза.")
    : !byId("test-text").value.trim() ? uiText("Введи текст для синтеза.") : "";
  byId("synthesis-hint").textContent = state.data ? missing || uiText`Моделей выбрано: ${state.models.size}` : "";
  byId("synthesize").disabled = working || Boolean(missing);
  updateAutoControls(working);
  const next = byId("use-saved-take");
  next.hidden = !state.lastSaved || location.pathname === "/tools";
  next.disabled = working;
  byId("assemble").disabled = working || state.assembly.length < 2;
  byId("assemble-blocks").disabled = working || state.blocks.length < 2;
  byId("language").disabled = working;
  byId("interface-language").disabled = working;
  byId("record").disabled = state.busy || !state.data;
  byId("recognize").disabled = working || !state.audio || !state.data?.recognition;
  byId("delete-profile").disabled = working || !state.profile;
  for (const button of document.querySelectorAll(".delete-item")) button.disabled = working;
  for (const select of document.querySelectorAll(".rating")) select.disabled = working;
  byId("upload").disabled = working || !state.data;
  byId("profiles").disabled = working || !state.data?.profiles.length;
  for (const select of document.querySelectorAll(".weights-field select")) select.disabled = working;
  for (const input of byId("recordings").querySelectorAll("input")) input.disabled = working;
  byId("transcript").readOnly = Boolean(state.recorder || state.recognizing);
  byId("use-prompt").disabled = working || !state.audio || !activeReading() || byId("evaluation-only").checked;
  byId("reading-mode").disabled = byId("story-part").disabled = working;
  byId("held-out").disabled = working || !state.data;
  byId("clear-source").disabled = working;
  for (const button of document.querySelectorAll(".history-use")) button.disabled = working;
  byId("repeats").disabled = working || !state.comparison;
  byId("test-text").readOnly = Boolean(state.comparison);
  byId("comparison-help").hidden = !state.comparison;
  byId("mark-prompt").disabled = working || !state.audio || byId("evaluation-only").checked;
  byId("evaluation-only").disabled = working || !state.audio;
  for (const button of byId("train-actions").querySelectorAll("button")) {
    button.disabled = working || !state.profile;
  }
}

let refreshGeneration = 0;
let pollingNow = false;

/** Apply only the newest state response, and leave a foreground action in control. */
async function refresh(reset = false, polling = false) {
  const generation = ++refreshGeneration;
  const language = state.language;
  try {
    const data = await request("state");
    if (generation !== refreshGeneration || language !== state.language) return;
    applyState(data, language, reset, polling);
  } catch (error) {
    if (generation === refreshGeneration && language === state.language) throw error;
  }
}

/** Apply a current response to the active workspace. */
function applyState(data, language, reset, polling) {
  const previousEngines = state.data?.engines;
  const modelChanges = JSON.stringify([previousEngines, state.data?.checkpoints]) !==
    JSON.stringify([data.engines, data.checkpoints]);
  state.data = data;
  if (reset) {
    state.models = new Set(Object.keys(data.engines).filter(id => data.engines[id].ready));
    const latest = data.profiles[0];
    state.profile = latest?.version || 0;
    const corpus = data.recordings.filter(item => !item.evaluation_only);
    state.refs = new Set(latest ? JSON.parse(latest.refs) : corpus.slice(-5).reverse().map(item => item.id));
    byId("transcript").value = "";
    byId("mark-prompt").checked = false;
    byId("evaluation-only").checked = false;
    state.testIndex = -1;
    byId("test-text").value = data.testText;
  }
  for (const model of state.models) {
    if (!data.engines[model].ready) state.models.delete(model);
  }
  if (polling) {
    for (const [model, status] of Object.entries(data.engines)) {
      if (status.ready && !previousEngines?.[model]?.ready) state.models.add(model);
    }
  }
  byId("demo").hidden = !data.demo;
  if (reset || modelChanges || !polling) renderEngines();
  if (!polling) {
    renderRecordings();
    const select = byId("media-recording");
    fillSelect(select, [["", uiText("Выбери запись")], ...data.recordings.map((row, index) =>
      [row.id, recordingName(row, index) + " · " + row.text.slice(0, 60)])], select.value);
    select.dispatchEvent(new Event("change"));
  }
  renderJobs();
  renderStress(data, language);
  renderProsodyPilot(data.jobs);
  byId("stress-jump").hidden = byId("stress-panel").hidden;
  autoMap.render(data.autoRuns || [], data.engines);
  if (reset) chooseComparison(sessionStorage.getItem("voice-studio:source:" + language));
  syncButtons();
}

async function act(callback) {
  if (state.busy) return;
  state.busy = true;
  refreshGeneration++;
  syncButtons();
  try {
    await callback();
  } catch (error) {
    notice(error.message, true);
  } finally {
    state.busy = false;
    syncButtons();
  }
}

function setAudio(blob) {
  showPauses(null);
  byId("transcript-suggestions").replaceChildren();
  if (state.previewUrl) URL.revokeObjectURL(state.previewUrl);
  state.audio = blob;
  state.lastSaved = "";
  byId("record-feedback").textContent = "";
  byId("evaluation-only").checked = false;
  state.previewUrl = blob ? URL.createObjectURL(blob) : null;
  previewPlayer.setSource(state.previewUrl);
  syncButtons();
}

async function toggleRecording() {
  if (state.recorder) {
    if (state.recorder.state === "recording") state.recorder.stop();
    return;
  }
  if (!navigator.mediaDevices?.getUserMedia || !window.MediaRecorder) {
    notice(uiText("Микрофон недоступен в этом браузере. Загрузи аудиофайл."), true);
    return;
  }
  let stream;
  try {
    const tools = !byId("record-denoise-field").hidden;
    const revision = state.toolsCaptureRevision;
    const noiseSuppression = byId("record-denoise").checked;
    stream = await navigator.mediaDevices.getUserMedia({audio: tools ? {noiseSuppression} : true});
    if (tools && revision !== state.toolsCaptureRevision) {
      for (const track of stream.getTracks()) track.stop();
      return;
    }
    if (tools && noiseSuppression && stream.getAudioTracks()[0]?.getSettings?.().noiseSuppression !== true) {
      notice(uiText("Браузер не подтвердил шумоподавление. Запись продолжается."));
    }
    const recorder = new MediaRecorder(stream);
    const chunks = [];
    state.recorder = recorder;
    recorder.ondataavailable = event => chunks.push(event.data);
    recorder.onerror = () => notice(uiText("Запись прервана. Попробуй ещё раз."), true);
    recorder.start();
    const stopTape = monitorTape(stream);
    recorder.onstop = () => {
      stopTape();
      clearTimeout(state.stopTimer);
      stream.getTracks().forEach(track => track.stop());
      state.recorder = null;
      byId("record").lastChild.textContent = uiText(" Записать");
      const blob = new Blob(chunks, {type: recorder.mimeType});
      if (blob.size) {
        setAudio(blob);
        byId("transcript").value = "";
        byId("mark-prompt").checked = false;
        act(recognize);
      }
      syncButtons();
    };
    state.stopTimer = setTimeout(() => {
      if (recorder.state === "recording") recorder.stop();
    }, 43000);
    byId("record").lastChild.textContent = uiText(" Остановить запись");
  } catch {
    stream?.getTracks().forEach(track => track.stop());
    state.recorder = null;
    notice(uiText("Не удалось открыть микрофон. Проверь разрешения или загрузи файл."), true);
  }
  syncButtons();
}

byId("record").onclick = () => state.recorder ? toggleRecording() : act(toggleRecording);
byId("upload").onchange = event => {
  const file = event.target.files[0];
  if (file) {
    setAudio(file);
    byId("transcript").value = "";
    byId("mark-prompt").checked = false;
    act(recognize);
  }
  event.target.value = "";
};
/** Show measured pauses with seek controls and an optional punctuation draft. */
function showPauses(result) {
  const panel = byId("pause-review");
  panel.hidden = !result;
  const review = byId("apply-punctuation");
  const draft = byId("punctuation-draft");
  const suggestion = result?.suggestion;
  review.hidden = draft.hidden = !suggestion || suggestion === result.text || suggestion.length > 500;
  draft.textContent = review.hidden ? "" : suggestion;
  review.onclick = review.hidden ? null : () => {
    if (byId("transcript").value !== result.text) return;
    byId("transcript").value = suggestion;
    byId("transcript-suggestions").replaceChildren();
    review.hidden = draft.hidden = true;
    syncButtons();
  };
  const pauses = result?.pauses || [];
  byId("pause-summary").textContent = pauses.length ? uiText`Паузы от 0,35 с · ${pauses.length}` : uiText("Пауз от 0,35 с не найдено.");
  byId("pause-list").replaceChildren(...pauses.map(pause => {
    const duration = (pause.end - pause.start).toFixed(2).replace(".", ",");
    const start = pause.start.toFixed(2).replace(".", ",");
    const button = element("button", uiText`${start} с · ${duration} с`, {type: "button"});
    button.setAttribute("aria-label", uiText`Пауза на ${start} с, длительность ${duration} с. Перейти к паузе`);
    button.onclick = () => {
      const preview = byId("preview");
      preview.currentTime = pause.start;
      preview.play().catch(() => {});
    };
    return button;
  }));
}

/** Update the editable draft with the local ASR result. */
async function recognize() {
  const status = byId("recognition-status");
  if (!state.data?.recognition) {
    status.textContent = uiText("Whisper недоступен. Введи текст вручную.");
    return;
  }
  showPauses(null);
  status.textContent = uiText("Whisper распознаёт запись…");
  const transcript = byId("transcript");
  state.recognizing = true;
  transcript.readOnly = true;
  try {
    const result = await request("transcribe", state.audio);
    transcript.value = result.text;
    renderTranscriptSuggestions(result.text, result.corrections || []);
    showPauses(result);
    byId("mark-prompt").checked = !byId("evaluation-only").checked && byId("reading-mode").value === "corpus" &&
      Boolean(state.data.prompt) && comparableReading(result.text) === comparableReading(state.data.prompt);
    status.textContent = uiText("Проверь расшифровку перед сохранением.");
  } catch (error) {
    status.textContent = error.message;
  } finally {
    state.recognizing = false;
    transcript.readOnly = false;
  }
}

byId("recognize").onclick = () => act(recognize);
byId("delete-profile").onclick = () => removeItem("profile", state.profile);
/** Prepare a saved take as the source for synthesis and A/B. */
function chooseComparison(id) {
  const source = state.data.recordings.find(item => item.id === id);
  state.comparison = source?.id || "";
  byId("comparison-source").textContent = source ? uiText("Исходник: ") + source.text : uiText("Свободный текст");
  byId("clear-source").hidden = !source;
  sessionStorage.setItem("voice-studio:source:" + state.language, state.comparison);
  clearProsodyPreview();
  deck.clear();
  if (source) {
    const repeats = byId("repeats");
    if (!repeats.dataset.userChoice) repeats.value = "2";
    if (!state.refs.size && !source.evaluation_only) {
      state.refs.add(source.id);
      updateSampleCount();
    }
    byId("test-text").value = source.text;
    const index = state.data.recordings.indexOf(source);
    deck.select(source.id, recordingName(source, index) + " · " + source.text.slice(0, 42), true);
  }
  syncTrackChoices();
  syncButtons();
}

byId("repeats").onchange = event => { event.target.dataset.userChoice = "true"; };
byId("clear-source").onclick = () => {
  chooseComparison("");
  byId("test-text").focus();
};
byId("use-saved-take").onclick = () => {
  chooseComparison(state.lastSaved);
  byId("compare-title").focus();
};
byId("held-out").onclick = () => {
  chooseComparison("");
  const lines = state.data.heldOut;
  state.testIndex = (state.testIndex + 1) % lines.length;
  byId("test-text").value = lines[state.testIndex];
  syncButtons();
};
byId("consent").onchange = syncButtons;
byId("evaluation-only").onchange = () => {
  if (byId("evaluation-only").checked) byId("mark-prompt").checked = false;
  syncButtons();
};
byId("transcript").oninput = () => {
  byId("transcript-suggestions").replaceChildren();
  byId("apply-punctuation").hidden = byId("punctuation-draft").hidden = true;
  syncButtons();
};
/** Preview model annotations without replacing the user's text. */
byId("review-prosody").onclick = () => act(async () => {
  const original = byId("test-text").value;
  const draft = await request("prosody", {text: original});
  if (byId("test-text").value !== original) return;
  byId("prosody-marks").textContent = draft.markedText;
  const shadow = byId("prosody-shadow");
  const proposals = (draft.shadowStress || []).map(mark =>
    mark.proposal + (mark.source === "context-model" ? uiText(" (контекст)") : uiText(" (по буквам)")));
  shadow.hidden = proposals.length === 0;
  shadow.textContent = proposals.length ? uiText("Предложения мини-моделей для проверки · ") + proposals.join(", ") : "";
  const terms = draft.terms.map(term => term.canonical).join(", ");
  const tones = {statement: uiText("утверждение"), question: uiText("вопрос"), exclamation: uiText("восклицание")};
  const phrases = draft.phrases.map(phrase => tones[phrase.tone] +
    (phrase.pauseMs ? uiText` (+${phrase.pauseMs} мс)` : "")).join(" → ");
  byId("prosody-notes").textContent = [terms && uiText("Термины · ") + terms, phrases,
    proposals.length && uiText("Предложения мини-моделей не попадают в синтез."),
    draft.stress.some(mark => mark.source === "abstained") && uiText("В части слов ударение оставлено без поправки")]
    .filter(Boolean).join(" • ");
});

/** Render optional context-backed edits without changing the Whisper draft. */
function renderTranscriptSuggestions(original, fixes, showEmpty = false) {
  const field = byId("transcript");
  const suggestions = byId("transcript-suggestions");
  if (field.value !== original) return;
  suggestions.replaceChildren(...fixes.map(fix => {
    const button = element("button", `${fix.word} → ${fix.replacement}`, {type: "button"});
    if (fix.evidence?.length) button.title = uiText("Слова рядом: ") + fix.evidence.map(link => link.word).join(", ");
    button.onclick = () => {
      if (field.value !== original) return;
      const chars = Array.from(original);
      field.value = [...chars.slice(0, fix.start), fix.replacement, ...chars.slice(fix.end)].join("");
      field.dispatchEvent(new Event("input"));
      byId("mark-prompt").checked = false;
      field.focus();
    };
    return button;
  }));
  if (showEmpty && !fixes.length) suggestions.textContent = uiText("Подсказок по словам нет.");
}

byId("check-transcript").onclick = () => act(async () => {
  const field = byId("transcript");
  const original = field.value;
  const trimmed = original.trim();
  const draft = await request("prosody", {text: trimmed});
  const start = Array.from(original.trimEnd()).length - Array.from(trimmed).length;
  const edits = draft.corrections.map(fix => ({...fix, start: fix.start + start, end: fix.end + start}));
  renderTranscriptSuggestions(original, edits, true);
});

/** Clear annotations when the synthesis source changes. */
function clearProsodyPreview() {
  byId("prosody-marks").textContent = "";
  byId("prosody-shadow").hidden = true;
  byId("prosody-shadow").textContent = "";
  byId("prosody-notes").textContent = "";
}

byId("test-text").oninput = () => {
  clearProsodyPreview();
  syncButtons();
};
byId("language").onchange = event => act(async () => {
  state.comparison = "";
  deck.clear();
  state.language = event.target.value;
  byId("reading-mode").value = "corpus";
  clearProsodyPreview();
  state.lastSaved = "";
  updatePageLinks();
  const route = new URL(location.href);
  if (state.language === "en") route.searchParams.set("lang", "en");
  else route.searchParams.delete("lang");
  history.replaceState(null, "", route);
  byId("recognition-status").textContent = "";
  state.profile = 0;
  state.assembly = [];
  state.blocks = [];
  renderBlockBin();
  state.checkpoints = {};
  byId("assembly-order").textContent = "";
  setAudio(null);
  await refresh(true);
});
byId("save-recording").onclick = () => act(async () => {
  const tools = location.pathname === "/tools";
  const flags = {text: byId("transcript").value, consent: "1"};
  const evaluationOnly = tools || byId("evaluation-only").checked;
  const storyTake = !evaluationOnly && state.data.readingStory && byId("reading-mode").value !== "corpus";
  const advanceStory = storyTake && comparableReading(flags.text) === comparableReading(activeReading());
  if (evaluationOnly) flags.evaluation = "1";
  else if (!storyTake && byId("mark-prompt").checked && state.data.promptNumber) flags.prompt = state.data.promptNumber;
  const data = await request("record", state.audio, flags);
  if (!evaluationOnly && state.refs.size < 500) {
    state.refs.add(data.id);
    state.profile = 0;
  }
  setAudio(null);
  await refresh();
  if (advanceStory) renderReading(true);
  if (!tools) chooseComparison(data.id);
  byId("transcript").value = "";
  byId("recognition-status").textContent = "";
  byId("mark-prompt").checked = false;
  byId("evaluation-only").checked = false;
  const advanced = flags.prompt && Number(flags.prompt) !== state.data.promptNumber;
  const message = tools ? uiText("Запись сохранена в историю и доступна для транскрипции.")
    : evaluationOnly ? uiText("Проверочная запись сохранена.")
    : storyTake ? uiText("Запись истории сохранена в корпус.")
    : !state.data.prompt ? uiText("Запись сохранена. Корпус прочитан.")
    : advanced ? uiText("Запись сохранена. Следующая фраза на дисплее.") : uiText("Запись сохранена.");
  state.lastSaved = tools ? "" : data.id;
  byId("record-feedback").textContent = message;
}).then(() => {
  if (!state.lastSaved) return;
  byId("use-saved-take").focus();
});

byId("save-profile").onclick = () => act(async () => {
  const data = await request("profile", {refs: [...state.refs]});
  state.profile = data.version;
  await refresh();
  notice(uiText`Версия ${data.version} сохранена.`);
});
byId("profiles").onchange = () => {
  state.profile = Number(byId("profiles").value);
  const profile = state.data.profiles.find(item => item.version === state.profile);
  if (profile) state.refs = new Set(JSON.parse(profile.refs));
  renderRecordings();
  syncButtons();
};
byId("export").onclick = () => act(async () => {
  const blob = await request("export", {refs: [...state.refs]});
  const url = URL.createObjectURL(blob);
  element("a", "", {href: url, download: "voice-corpus.zip"}).click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
});
/** Auto-synthesis uses an independent evaluation recording and selected local engines. */
function updateAutoControls(working) {
  const models = [...state.models].filter(model => state.data?.engines[model]?.automatic);
  const references = [...state.refs].filter(id => id !== state.comparison).length;
  const active = state.data?.autoRuns?.some(run => run.status !== "stopped");
  const reason = active ? uiText("Автосинтез уже запущен.")
    : !state.comparison ? uiText("Выбери исходник для оценки в истории записей.")
    : references < 2 ? uiText("Добавь в профиль два образца отдельно от исходника.")
    : !models.length ? uiText("Выбери F5-TTS или Coqui.")
    : !state.data?.autoReady ? uiText("Для оценки нужны локальные Whisper и энкодер голоса.") : "";
  byId("auto-start").disabled = working || Boolean(reason);
  byId("auto-hint").textContent = reason || uiText`Образцов: ${references} · моделей: ${models.length}`;
}

byId("auto-form").onsubmit = event => {
  event.preventDefault();
  act(async () => {
    const {id} = await request("auto-synthesize", {
      comparison: state.comparison, refs: [...state.refs], checkpoints: state.checkpoints,
      models: [...state.models].filter(model => state.data?.engines[model]?.automatic),
      target: byId("auto-target").valueAsNumber, max_attempts: byId("auto-limit").valueAsNumber,
      minutes: byId("auto-minutes").valueAsNumber, stress: byId("try-stress").checked,
    });
    autoMap.select(id);
    await refresh();
    byId("auto-map-title").focus();
  });
};

byId("synthesize").onclick = () => act(async () => {
  await request("synthesize", {text: byId("test-text").value, models: [...state.models],
    refs: [...state.refs], profile: state.profile, checkpoints: state.checkpoints,
    comparison: state.comparison || null, prosody: {
      stress: byId("try-stress").checked && state.language === "ru" && state.models.has("f5"),
      pauses: byId("try-pauses").checked, tone: byId("try-tone").value
    }, repeats: state.comparison ? Number(byId("repeats").value) : 1});
  await refresh();
  byId("jobs-title").focus();
});
/** Keep approvals in order while moving between graphs and listening to other takes. */
function approveBlock(part, label, model) {
  if (state.blocks.length >= 12) return uiText("В сборке уже 12 фрагментов.");
  const duration = state.blocks.reduce((sum, item) => sum + item.endMs - item.startMs, part.endMs - part.startMs);
  if (duration > 90000) return uiText("Сборка должна быть не длиннее 90 секунд.");
  state.blocks.push({...part, label, model});
  renderBlockBin();
  return uiText`В сборке · ${state.blocks.length} фрагм.`;
}

/** Refresh the selected ranges and keep focus after keyboard edits. */
function renderBlockBin() {
  const list = byId("blocks");
  byId("block-title").parentElement.hidden = !state.blocks.length;
  renderBlocks(list, state.blocks, (index, delta) => {
    const [part] = state.blocks.splice(index, 1);
    if (delta) state.blocks.splice(index + delta, 0, part);
    renderBlockBin();
    const next = Math.min(index + delta, list.children.length - 1);
    (list.children[next]?.querySelector("button:not(:disabled)") || byId("jobs-title")).focus();
  });
  syncButtons();
}

byId("assemble-blocks").onclick = () => act(async () => {
  const {id} = await request("assemble-blocks", {blocks: state.blocks});
  await refresh();
  byId("jobs").querySelector(`[data-key="${id}"] .player-play`)?.focus();
  notice(uiText("Сборка сохранена. Прослушай стыки и скачай WAV."));
});

byId("assemble").onclick = () => act(async () => {
  await request("assemble", {ids: state.assembly});
  state.assembly = [];
  byId("assembly-order").textContent = "";
  byId("jobs").replaceChildren();
  await refresh();
  notice(uiText("Дорожка сохранена. Её можно скачать в списке результатов."));
});
byId("use-prompt").onclick = () => {
  byId("transcript").value = activeReading();
  byId("transcript").dispatchEvent(new Event("input"));
  byId("mark-prompt").checked = byId("reading-mode").value === "corpus";
  byId("transcript").focus();
  syncButtons();
};

for (const id of ["reading-mode", "story-part"]) {
  byId(id).onchange = () => {
    byId("mark-prompt").checked = false;
    renderReading(id === "reading-mode");
    syncButtons();
  };
}

/** Build a shareable local route with independent speech and interface settings. */
function pageUrl(path, anchor = "") {
  const query = new URLSearchParams();
  if (state.language === "en") query.set("lang", "en");
  if (uiLocale === "en") query.set("ui", "en");
  return path + (query.size ? "?" + query : "") + (anchor ? "#" + anchor : "");
}

/** Keep history links functional even when opened directly in another tab. */
function updatePageLinks() {
  document.querySelector(".brand").href = pageUrl("/");
  byId("studio-link").href = pageUrl("/");
  byId("wiki-link").href = uiLocale === "en" ? "/wiki/Home.html" : "/wiki/Home-ru.html";
  byId("tools-link").href = pageUrl("/tools");
  byId("storage-link").href = pageUrl("/storage");
  byId("settings-link").href = pageUrl("/settings");
  for (const link of document.querySelectorAll("#history-link, .history-jump")) link.href = pageUrl("/history");
}

const studioRoutes = {
  "/": ["studio", "title"], "/tools": ["tools", "tools-title"],
  "/storage": ["storage", "storage-title"],
  "/history": ["history", "history-title"], "/settings": ["settings", "settings-title"],
};

/** Include the saved material's language independently of the current speech workspace. */
function materialUrl(id, language) {
  const route = new URL(pageUrl("/tools", "media-title"), location.origin);
  route.searchParams.set("transcript", id);
  route.searchParams.set("material_lang", language);
  return route.pathname + route.search + route.hash;
}

/** Open a saved transcript through the same route used by direct links and browser history. */
function openSavedMaterial(url) {
  if (mediaTools.isBusy()) {
    notice(uiText("Дождись завершения текущей операции."), true);
    return;
  }
  showPage("/tools", "media-title", url);
  return loadRouteMaterial();
}

/** Load the material selected by the URL after the tools panel is mounted. */
async function loadRouteMaterial() {
  if (location.pathname !== "/tools") return;
  const query = new URLSearchParams(location.search);
  const id = query.get("transcript");
  if (!id) return;
  await transcriptionLanguages;
  if (location.pathname !== "/tools" || new URLSearchParams(location.search).toString() !== query.toString()) return;
  const tag = query.get("material_lang");
  const known = [...byId("settings-material-language").options].some(option => option.value === tag);
  const language = known ? tag : state.language;
  const current = mediaTools.currentTranscript();
  if (current?.id === id && current.language === language) return;
  await mediaTools.openMaterial(id, language);
}

/** Keep the recorder and each workspace mounted while switching pages. */
function showPage(path, anchor, navigate = false) {
  const [page, title] = studioRoutes[path] ?? studioRoutes["/"];
  const historyPage = path === "/history";
  const toolsPage = path === "/tools";
  if (!toolsPage && !byId("tools-page").hidden) {
    state.toolsCaptureRevision++;
    mediaTools.stopCapture();
    if (state.recorder?.state === "recording") state.recorder.stop();
  }
  for (const [name] of Object.values(studioRoutes)) {
    const mounted = byId(name + "-page");
    if (name === page || mounted.hidden) continue;
    for (const audio of mounted.querySelectorAll("audio")) audio.pause();
  }
  if (page !== "studio") deck.audio.pause();
  if (navigate) {
    const url = typeof navigate === "string" ? navigate : pageUrl(path, anchor);
    history.pushState(null, "", url);
  }
  moveAudioTools(toolsPage);
  for (const [name] of Object.values(studioRoutes)) byId(name + "-page").hidden = name !== page;
  navigation.select(page + "-link");
  document.title = page === "studio" ? "ag-media-studio" : byId("studio-current-page").textContent + " · ag-media-studio";
  if (historyPage && state.data) renderHistory();
  if (page === "settings") settings.open(state.language);
  if (page === "storage") storage.open();
  syncButtons();
  if (anchor || navigate) (byId(anchor) ?? byId(title)).focus();
}

/** Reuse the recorder and transcript panels so edits survive navigation. */
function moveAudioTools(toolsPage) {
  const recorder = byId("record-title").closest(".recorder");
  const media = byId("media-panel");
  if (toolsPage && recorder.parentElement.id !== "tools-recorder") {
    byId("tools-recorder").append(recorder);
    byId("tools-page").append(media);
  } else if (!toolsPage && recorder.parentElement.id === "tools-recorder") {
    byId("studio-work").prepend(recorder);
    byId("media-home").after(media);
  }
  media.hidden = !toolsPage;
  byId("corpus-reading").hidden = toolsPage;
  if (state.data) renderReading();
  else byId("prompt-check").hidden = toolsPage;
  byId("evaluation-only-field").hidden = toolsPage;
  byId("record-denoise-field").hidden = !toolsPage;
  byId("record-title").textContent = toolsPage ? uiText("Запись с шумоподавлением") : uiText("Запись образца");
}

byId("history-search").oninput = byId("history-kind").onchange = () => {
  if (state.data) renderHistory();
};
for (const link of document.querySelectorAll(".studio-nav a, .brand, .history-jump")) {
  link.onclick = event => {
    if (event.button || event.metaKey || event.ctrlKey || event.altKey || event.shiftKey) return;
    const path = new URL(link.href).pathname;
    // Documentation has its own document and must keep the active recording mounted.
    if (link.target === "_blank" || !Object.hasOwn(studioRoutes, path)) return;
    event.preventDefault();
    showPage(path, studioRoutes[path]?.[1], true);
  };
}
addEventListener("popstate", () => {
  const language = speechLanguage();
  const routeLocale = new URLSearchParams(location.search).get("ui");
  if (language !== state.language || (routeLocale && routeLocale !== uiLocale)) {
    location.reload();
    return;
  }
  showPage(location.pathname, location.hash.slice(1));
  loadRouteMaterial();
});

const journeySections = {
  studio: ["/", "title"], settings: ["/settings", "settings-title"], record: ["/tools", "record-title"], history: ["/history", "history-title"],
  storage: ["/storage", "storage-title"], transcript: ["/tools", "media-title"], synthesis: ["/", "compare-title"], results: ["/", "jobs-title"],
};

/** Report user-visible journey state using semantic control names and stable IDs. */
function inspectJourney() {
  const report = mediaTools.currentTranscript();
  const controls = {};
  for (const id of ["record", "media-upload", "media-recording-submit", "media-srt", "media-json",
                    "media-overlay-toggle", "media-editor-gaze", "save-recording", "synthesize"]) {
    const control = byId(id);
    if (!control) continue;
    controls[id] = {visible: !control.closest("[hidden]"), enabled: !control.disabled};
  }
  return {
    page: location.pathname, section: location.hash.slice(1) || null, speechLanguage: state.language,
    interfaceLanguage: uiLocale, designProfile: document.documentElement.dataset.designProfile,
    colorMode: document.documentElement.dataset.colorMode, transcriptionLanguage: byId("media-speech-language").value,
    ready: Boolean(state.data), analysisRunning: mediaTools.isBusy(),
    recording: state.recorder?.state === "recording", cueCount: report?.segments.length ?? 0, controls,
    recentRecordings: (state.data?.recordings ?? []).slice(-12).reverse().map(({id, seconds}) => ({id, seconds})),
    recentJobs: (state.data?.jobs ?? []).slice(0, 12).map(({id, kind, status, model}) => ({id, kind, status, model})),
  };
}

/** Use the existing router and preserve sources that are being recorded or analyzed. */
function navigateJourney(target) {
  if (state.recorder?.state === "recording" || mediaTools.isBusy()) {
    throw Error(uiText("Дождись завершения текущей операции."));
  }
  const mcpDialog = byId("mcp-dialog");
  if (mcpDialog.open) {
    mcpDialog.dataset.navigation = "true";
    mcpDialog.close();
  }
  const [path, anchor] = journeySections[target];
  showPage(path, anchor, true);
  return inspectJourney();
}

/** Interpret the local route's language query. */
function speechLanguage() {
  return new URLSearchParams(location.search).get("lang") === "en" ? "en" : "ru";
}
state.language = speechLanguage();
byId("language").value = state.language;
updatePageLinks();
showPage(location.pathname, location.hash.slice(1));
await loadRouteMaterial();
mountStudioWebMcp(byId, {
  request, ui: {
    inspect: inspectJourney, navigate: navigateJourney,
    transcript: () => location.pathname === "/tools" ? mediaTools.currentTranscript() : null,
    contextRevision: () => location.pathname === "/tools" ? mediaTools.contextRevision() : null,
    peopleCount: enabled => mediaTools.peopleCount(enabled),
    seek: ms => {
      if (location.pathname !== "/tools") throw Error(uiText("Открой раздел транскрипции."));
      return mediaTools.seek(ms);
    },
    transcribe: (id, signal) => {
      if (location.pathname !== "/tools") throw Error(uiText("Открой раздел транскрипции."));
      return mediaTools.transcribeRecording(id, signal);
    },
  },
});
await act(() => refresh(true));
for (const [anchor, panel] of [["stress-title", "stress-panel"], ["prosody-pilot-title", "prosody-pilot"]]) {
  if (location.hash === "#" + anchor && !byId(panel).hidden) byId(anchor).focus();
}
setInterval(async () => {
  if (state.busy || pollingNow || document.hidden) return;
  pollingNow = true;
  try {
    await refresh(false, true);
  } catch (error) {
    notice(error.message, true);
  } finally {
    pollingNow = false;
  }
}, 3000);
