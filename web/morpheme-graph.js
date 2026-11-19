import {uiLocale, uiText} from "./i18n.js";
import {mountPlayer} from "./player.js";

const labels = {match: uiText("Совпало"), substitution: uiText("Другое слово"), missing: uiText("Пропуск"), extra: uiText("Лишнее слово")};
const stressSources = {
  manual: uiText("метка в тексте"), dictionary: uiText("словарь"), yo: uiText("буква «ё»"), context: uiText("правило контекста"),
  model: uiText("текстовая модель")
};
const number = new Intl.NumberFormat(uiLocale, {minimumFractionDigits: 2, maximumFractionDigits: 2});
const seconds = milliseconds => number.format(milliseconds / 1000) + uiText(" с");
const pauseText = pause => uiText`${pause.durationMs} мс · ${seconds(pause.startMs)} – ${seconds(pause.endMs)}`;

function node(tag, text, className) {
  const element = document.createElement(tag);
  if (text) element.textContent = text;
  if (className) element.className = className;
  return element;
}

/** Show spelling-derived syllables and independent PCM pause measurements for a word. */
function renderWordProsody(details, row, pauses) {
  details.replaceChildren();
  const syllables = row?.syllables || [];
  if (syllables.length) {
    const spelling = node("dd", "", "graph-syllables");
    for (const syllable of syllables) {
      spelling.append(node(syllable.stressed ? "strong" : "span", syllable.text));
    }
    const source = stressSources[row.stress?.source];
    const stressed = syllables.find(syllable => syllable.stressed);
    details.append(node("dt", uiText("Слоги по тексту · приблизительно")), spelling,
      node("dt", uiText("Ожидаемое ударение")),
      node("dd", source && stressed ? `«${stressed.text}» · ${source}` : uiText("Надёжной метки в тексте нет")));
  }
  if (row?.index == null) return;
  for (const pause of pauses) {
    const boundAfter = pause.afterIndex === row.index;
    if (!boundAfter && pause.beforeIndex !== row.index) continue;
    details.append(node("dt", boundAfter ? uiText("Пауза после слова по WAV") : uiText("Пауза перед словом по WAV")),
      node("dd", pauseText(pause)));
  }
}

/** Show the local alignment, audition an editable range, and retry a selected word. */
export function openMorphemeGraph(job, request, approve, refresh, restoreFocus, showRetries) {
  const dialog = document.querySelector("#morpheme-graph");
  const find = selector => dialog.querySelector(selector);
  const status = find(".graph-status");
  const rows = find(".graph-rows");
  const selected = find(".graph-selected");
  const controls = find(".graph-controls");
  const parts = find(".graph-parts");
  find(".graph-prosody")?.remove();
  const details = node("dl", "", "graph-prosody");
  parts.after(details);
  find(".graph-pauses")?.remove();
  const pauseSummary = node("details", "", "graph-pauses");
  pauseSummary.hidden = true;
  status.after(pauseSummary);
  const score = find(".graph-score");
  const start = find(".graph-start");
  const end = find(".graph-end");
  const play = find(".graph-play");
  const add = find(".graph-add");
  const retry = find(".graph-retry");
  const count = find(".graph-count");
  const context = find(".graph-context");
  const show = find(".graph-show-retries");
  show.hidden = true;
  let batch = [];
  const audio = node("audio");
  audio.preload = "none";
  audio.src = `/media/${job.id}.wav`;
  const limit = Math.round(job.seconds * 1000);
  const player = mountPlayer(audio, uiText("Прослушивание дубля в графе"), job.seconds);
  find(".graph-player").replaceChildren(player.root);
  for (const track of document.querySelectorAll("audio")) track.pause();
  let current, choice, frame, hasOriginal, scoreTimer;
  let measuredPauses = [];
  let scoreQueue = Promise.resolve();
  const active = new AbortController();
  dialog.addEventListener("close", () => {
    active.abort();
    audio.pause();
    audio.removeAttribute("src");
    audio.load();
    cancelAnimationFrame(frame);
    clearTimeout(scoreTimer);
    if (dialog.returnValue === "retries") showRetries(batch);
    else restoreFocus();
  }, {once: true});
  dialog.returnValue = "";
  show.onclick = () => dialog.close("retries");
  find(".graph-close").onclick = () => dialog.close();
  selected.textContent = "";
  controls.hidden = true;
  rows.replaceChildren();
  status.textContent = uiText("Whisper сверяет слова с дублем…");
  dialog.showModal();

  /** Read precise user inputs in an integer unit accepted by the server. */
  function range() {
    return {id: job.id, startMs: Math.round(start.valueAsNumber * 1000), endMs: Math.round(end.valueAsNumber * 1000)};
  }

  /** Enable range actions only when both cut points are inside the take. */
  function updateRange() {
    const {startMs, endMs} = range();
    const valid = start.checkValidity() && end.checkValidity() && endMs - startMs >= 80;
    add.disabled = play.disabled = !valid;
  }

  /** Score the selected word automatically and ignore responses for older selections. */
  function updateScore(row) {
    clearTimeout(scoreTimer);
    score.hidden = row?.index == null;
    score.textContent = "";
    if (score.hidden) return;
    if (!hasOriginal) {
      score.textContent = uiText("Для оценки голоса выбери исходник перед синтезом.");
      return;
    }
    score.textContent = uiText("Сравниваем голос по фразе…");
    scoreTimer = setTimeout(() => {
      scoreQueue = scoreQueue.then(async () => {
        if (active.signal.aborted || current !== row) return;
        try {
          const result = await request("fragment-score", {id: job.id, index: row.index});
          if (active.signal.aborted || current !== row) return;
          score.textContent = result.cosine == null
            ? result.reason
            : uiText`Косинусная близость голоса · ${result.cosine.toLocaleString(uiLocale, {minimumFractionDigits: 3, maximumFractionDigits: 3})} · «${result.text}»`;
        } catch (error) {
          if (!active.signal.aborted && current === row) score.textContent = error.message;
        }
      });
    }, 220);
  }

  /** Open word details, or switch to a range independent of Whisper. */
  function select(row, button) {
    choice?.setAttribute("aria-pressed", "false");
    choice = button;
    choice.setAttribute("aria-pressed", "true");
    current = row;
    updateScore(row);
    selected.textContent = row?.expected || row?.heard?.text || uiText("Ручная нарезка");
    parts.replaceChildren();
    if (row?.morphemes.length) {
      for (const [part, role] of row.morphemes) {
        const item = node("li", "", "graph-part");
        item.append(node("b", part), node("span", role, "muted"));
        parts.append(item);
      }
    } else if (row) {
      parts.append(node("li", uiText("В словаре пока нет разбора."), "muted"));
    }
    renderWordProsody(details, row, measuredPauses);
    controls.hidden = false;
    find(".graph-repeat").hidden = row?.index == null;
    find(".graph-time-hint").textContent = row
      ? uiText("Время от Whisper приблизительное. Прослушай и уточни границы.")
      : uiText("Отметь начало и конец удачного фрагмента.");
    const hasTime = Number.isInteger(row?.heard?.startMs);
    start.value = hasTime ? (row.heard.startMs / 1000).toFixed(3) : "0";
    if (hasTime) end.value = (Math.min(limit, row.heard.endMs) / 1000).toFixed(3);
    else end.value = row ? "" : String(limit / 1000);
    updateRange();
  }

  const manual = find(".graph-manual");
  manual.setAttribute("aria-pressed", "false");
  manual.onclick = () => select(null, manual);
  count.value = "1";
  context.value = "short";
  rows.inert = manual.disabled = retry.disabled = context.disabled = count.disabled = false;
  start.max = end.max = String(limit / 1000);
  start.oninput = end.oninput = updateRange;
  /** Use the playhead to mark cuts without typing timestamps. */
  function usePlayhead(input) {
    audio.pause();
    input.value = Math.min(limit / 1000, audio.currentTime).toFixed(3);
    updateRange();
  }
  find(".graph-mark-start").onclick = () => usePlayhead(start);
  find(".graph-mark-end").onclick = () => usePlayhead(end);
  play.onclick = async () => {
    const {startMs, endMs} = range();
    cancelAnimationFrame(frame);
    try {
      audio.currentTime = startMs / 1000;
      await audio.play();
      function stopAtEnd() {
        if (audio.paused || active.signal.aborted) return;
        if (audio.currentTime >= endMs / 1000) audio.pause();
        else frame = requestAnimationFrame(stopAtEnd);
      }
      frame = requestAnimationFrame(stopAtEnd);
    } catch {
      if (!active.signal.aborted) status.textContent = uiText("Не удалось воспроизвести фрагмент.");
    }
  };
  add.onclick = () => {
    const label = current?.expected || current?.heard?.text || uiText("Ручной фрагмент");
    const message = approve(range(), label);
    status.textContent = message;
  };
  retry.onclick = async () => {
    const index = current.index;
    batch = [];
    show.hidden = true;
    rows.inert = true;
    manual.disabled = true;
    retry.disabled = context.disabled = count.disabled = true;
    status.textContent = uiText("Ставим повторы в очередь…");
    try {
      const {ids} = await request("retry-words", {id: job.id, index, count: Number(count.value), context: context.value});
      if (!active.signal.aborted) {
        batch = ids;
        show.hidden = false;
        status.textContent = uiText`Добавлено в «Результаты и A/B» · ${ids.length}`;
      }
      await refresh().catch(() => {});
    } catch (error) {
      if (!active.signal.aborted) status.textContent = error.message;
    } finally {
      if (!active.signal.aborted) rows.inert = manual.disabled = retry.disabled = context.disabled = count.disabled = false;
    }
  };

  request("word-graph", {id: job.id}).then(graph => {
    if (active.signal.aborted) return;
    hasOriginal = graph.hasOriginal;
    measuredPauses = graph.pauses || [];
    status.textContent = graph.rows.length
      ? uiText`Сверка готова · расхождений: ${graph.rows.filter(row => row.kind !== "match").length}`
      : uiText("Для этого дубля не найдено слов.");
    if (measuredPauses.length) {
      const unbound = measuredPauses.filter(pause => pause.afterIndex == null).length;
      const summary = uiText`Паузы по WAV · ${measuredPauses.length}` +
        (unbound ? uiText` (${unbound} без привязки к словам)` : "");
      const times = node("ul");
      for (const pause of measuredPauses) times.append(node("li", pauseText(pause)));
      pauseSummary.append(node("summary", summary), times);
      pauseSummary.hidden = false;
    }
    const options = graph.rows.map(row => {
      const item = node("li");
      const button = node("button", "", "graph-edge");
      button.type = "button";
      button.dataset.kind = row.kind;
      button.setAttribute("aria-pressed", "false");
      const edge = node("span", "", "graph-link");
      edge.setAttribute("aria-hidden", "true");
      const heard = row.heard?.text || "—";
      const match = node("span", heard, "graph-node graph-heard");
      match.append(node("small", labels[row.kind], "graph-kind"));
      button.append(node("span", row.expected || "—", "graph-node graph-expected"), edge, match);
      button.onclick = () => select(row, button);
      item.append(button);
      rows.append(item);
      return button;
    });
    let initial = graph.rows.findIndex(row => graph.focusIndex != null && row.index === graph.focusIndex);
    if (initial < 0) initial = graph.rows.findIndex(row => row.expected && row.kind !== "match");
    if (initial < 0) initial = graph.rows.findIndex(row => row.morphemes.length);
    if (!choice && options.length) options[Math.max(0, initial)].click();
  }).catch(error => {
    if (active.signal.aborted) return;
    status.textContent = error.message;
    if (!choice) manual.click();
  });
}

/** Render the exact order that will be sent to the WAV assembler. */
export function renderBlocks(list, blocks, change) {
  list.replaceChildren();
  for (const [index, block] of blocks.entries()) {
    const item = node("li");
    const description = node("span", `${block.label} · ${block.model} · ${seconds(block.startMs)}–${seconds(block.endMs)}`);
    const actions = node("div", "", "block-actions");
    for (const [delta, name, symbol] of [[-1, uiText("Выше"), "↑"], [1, uiText("Ниже"), "↓"], [0, uiText("Удалить"), "×"]]) {
      const button = node("button", symbol);
      button.type = "button";
      button.setAttribute("aria-label", uiText`${name} фрагмент ${index + 1}`);
      if (delta) button.disabled = !blocks[index + delta];
      button.onclick = () => change(index, delta);
      actions.append(button);
    }
    item.append(description, actions);
    list.append(item);
  }
}
