import {apiText, uiLocale, uiText} from "./i18n.js";
const reasons = {
  target: uiText("Цель достигнута"), attempt_limit: uiText("Лимит дублей исчерпан"), time_limit: uiText("Время истекло"),
  cancelled: uiText("Остановлено"), restart: uiText("Прервано перезапуском"), error: uiText("Прогон прервался с ошибкой"),
};
const outcomes = {
  promoted: uiText("Принят"), regression: uiText("Откат · предыдущий результат сохранён"),
  unchanged: uiText("Без улучшения"), unscored: uiText("Оценка недоступна"), failed: uiText("Ошибка синтеза"), interrupted: uiText("Прервано"),
};
const number = (value, digits = 1) => value == null ? "—" : value.toLocaleString(uiLocale, {maximumFractionDigits: digits});

function element(tag, text, properties = {}) {
  const node = Object.assign(document.createElement(tag), properties);
  if (text !== null) node.textContent = text;
  return node;
}

/** Format elapsed seconds without hiding a zero or implying an exact estimate. */
function duration(seconds) {
  const total = Math.max(0, Math.ceil(seconds));
  return uiText`${Math.floor(total / 60)} мин ${total % 60} с`;
}

/** Render ordered graph nodes and retain players across status polling. */
export function createAutoMap(root, player, stop) {
  const picker = root.querySelector("#auto-run-select");
  const graph = root.querySelector("#auto-graph");
  const summary = root.querySelector("#auto-run-summary");
  const timing = root.querySelector("#auto-timing");
  const script = root.querySelector("#auto-run-text");
  const halt = root.querySelector("#auto-stop");
  let latest = [];
  let selected;
  let pickerSignature;
  let graphRun;

  /** Build a playable trial card and its links to earlier decisions. */
  function card(attempt, run, engines) {
    const title = engines[attempt.model]?.name || attempt.model || uiText("Дубль");
    const node = element("li", null, {className: "auto-node", id: `auto-node-${attempt.job_id}`, tabIndex: -1});
    const heading = element("h3", uiText`Дубль ${attempt.position} · ${title}`);
    node.append(heading);
    const prior = run.attempts.find(row => row.job_id === attempt.incumbent_id);
    if (prior) {
      const label = attempt.outcome === "regression" ? uiText("Сохранён дубль") : uiText("Сравнение с дублем");
      const link = element("a", `${label} ${prior.position}`, {
        href: `#auto-node-${prior.job_id}`, className: "auto-backlink"
      });
      link.onclick = event => {
        event.preventDefault();
        document.getElementById(`auto-node-${prior.job_id}`)?.focus();
      };
      node.append(link);
    }
    const decision = attempt.outcome ? outcomes[attempt.outcome] : uiText("В работе");
    node.append(element("p", run.best_id === attempt.job_id ? uiText`${decision} · Выбранный результат` : decision));
    if (attempt.references != null) {
      const variant = attempt.variant === "auto-stress" ? uiText(" · с ударениями") : "";
      node.append(element("p", uiText`Образцов: ${attempt.references}${variant}`, {className: "muted"}));
    }
    if (attempt.finished) {
      const wer = attempt.wer == null ? "—" : `${number(attempt.wer * 100)}%`;
      node.append(element("p", uiText`Балл ${number(attempt.quality)} · ${run.scoring_version?.startsWith("eres2net") ? "ERes2Net" : "XTTS"} ${number(attempt.cosine, 3)} · WER ${wer}`),
        element("p", duration(attempt.finished - attempt.started), {className: "muted"}));
    }
    if (attempt.error) node.append(element("p", apiText(attempt.error), {className: "muted"}));
    if (attempt.audio_status === "done") {
      node.append(player(attempt.job_id, title, attempt.seconds),
        element("a", uiText("Скачать WAV"), {href: `/media/${attempt.job_id}.wav`, download: `take-${attempt.position}.wav`}));
    } else if (!attempt.audio_status) {
      node.append(element("p", uiText("Аудио удалено."), {className: "muted"}));
    }
    return node;
  }

  /** Update remaining budget and the estimate derived from completed attempts. */
  function updateTiming(run) {
    const end = run.finished || Date.now() / 1000;
    const elapsed = duration(Math.max(0, end - run.created));
    const budget = duration(run.deadline - run.created);
    let text = uiText`Прошло ${elapsed} из ${budget}`;
    if (!run.finished) {
      const completed = run.attempts.filter(row => row.finished);
      const remaining = Math.max(0, run.deadline - end);
      text += uiText` · осталось до лимита ${duration(remaining)}`;
      if (completed.length) {
        const average = completed.reduce((sum, row) => sum + row.finished - row.started, 0) / completed.length;
        text += uiText` · среднее на дубль ${duration(average)}`;
        const current = run.attempts.find(row => !row.finished);
        const estimate = average * (run.max_attempts - completed.length) -
          (current ? end - current.started : 0);
        if (estimate > 0) text += uiText` · до лимита дублей ≈ ${duration(Math.min(remaining, estimate))}`;
      }
    }
    if (timing.textContent !== text) timing.textContent = text;
  }

  /** Reconcile existing graph nodes by ID instead of interrupting playback each poll. */
  function render(runs, engines) {
    latest = runs;
    root.hidden = !runs.length;
    if (!runs.some(run => run.id === selected)) selected = runs[0]?.id;
    const signature = JSON.stringify(runs.map(run => [run.id, run.created, run.target]));
    if (signature !== pickerSignature) {
      pickerSignature = signature;
      picker.replaceChildren(...runs.map(run => new Option(
        uiText`${new Date(run.created * 1000).toLocaleString(uiLocale)} · цель ${run.target}`, run.id)));
    }
    if (!selected) return;
    picker.value = selected;
    const run = runs.find(item => item.id === selected);
    const active = run.status === "running" || run.status === "queued";
    if (script.textContent !== run.text) script.textContent = run.text;
    const best = run.attempts.find(row => row.job_id === run.best_id);
    const message = uiText`${active ? (run.status === "queued" ? uiText("В очереди") : uiText("В работе")) : reasons[run.reason]}
      · ${run.attempts.length} из ${run.max_attempts} · цель ${run.target} · лучший балл ${number(best?.quality)}`;
    const result = !active && !best ? message + uiText(" · нет оценённого дубля") : message;
    if (summary.textContent !== result) summary.textContent = result;
    halt.hidden = !active;
    halt.disabled = Boolean(run.cancelled);
    halt.textContent = run.cancelled ? uiText("Остановка запрошена") : uiText("Остановить после дубля");
    updateTiming(run);

    if (graphRun !== selected) {
      graphRun = selected;
      graph.replaceChildren();
    }
    const existing = new Map([...graph.children].map(node => [node.id, node]));
    for (const attempt of run.attempts) {
      const id = `auto-node-${attempt.job_id}`;
      const previous = existing.get(id);
      const signature = JSON.stringify(attempt);
      if (previous?.dataset.signature !== signature) {
        const node = card(attempt, run, engines);
        node.dataset.signature = signature;
        if (previous) {
          const focused = previous.contains(document.activeElement);
          previous.replaceWith(node);
          if (focused) node.focus({preventScroll: true});
        } else {
          graph.append(node);
        }
      }
      const node = document.getElementById(id);
      node.classList.toggle("auto-best", attempt.job_id === run.best_id);
      const decision = attempt.outcome ? outcomes[attempt.outcome] : uiText("В работе");
      const text = attempt.job_id === run.best_id ? uiText`${decision} · Выбранный результат` : decision;
      const status = node.querySelector("p");
      if (status.textContent !== text) status.textContent = text;
      existing.delete(id);
    }
    for (const node of existing.values()) node.remove();
  }

  let models = {};
  picker.onchange = () => {
    selected = picker.value;
    render(latest, models);
  };
  halt.onclick = () => stop(selected);
  return {
    render(runs, engines) {
      models = engines;
      render(runs, engines);
    },
    select(id) { selected = id; },
  };
}
