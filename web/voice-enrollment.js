import {uiText} from "./i18n.js";

const issues = {
  "short-speech": "Мало активного сигнала",
  "long-take": "Дольше 18 секунд",
  clipping: "Перегрузка",
  quiet: "Тихая запись",
  duplicate: "Повтор того же аудио",
};

function paragraph(text) {
  const node = document.createElement("p");
  node.textContent = text;
  return node;
}

/** Keep measured waveform controls separate from the encoder similarity score. */
export function showVoiceReferenceReport(find, report, recordings, prompts, {open = false} = {}) {
  const view = find("voice-reference-report");
  const output = find("voice-reference-results");
  const status = find("voice-reference-status");
  view.hidden = !report;
  if (!report) {
    output.replaceChildren();
    status.textContent = "";
    return;
  }
  const indices = new Map(report.takes.map(({id}, index) => [id, index + 1]));
  const used = new Set(report.referenceIds);
  const first = report.referenceIds[0];
  const guide = new Map(prompts.map(item => [item.id, item]));
  const titles = new Map(recordings.map(item => [item.id, item.text.slice(0, 65)]));
  const takes = document.createElement("ul");
  for (const take of report.takes) {
    const entry = document.createElement("li");
    const name = guide.get(take.prompt)?.label || titles.get(take.id) || "";
    const quality = [uiText`Активный сигнал ${(take.activeMs / 1000).toFixed(2)} с`,
      uiText`Уровень сигнала ${take.speechDbfs.toFixed(1)} дБFS`,
      uiText`Паузы ${take.pauseMs} мс`, uiText`${take.wordsPerMinute} слов/мин`,
      take.pitchHz && uiText`F0 ${take.pitchHz.join(" → ")} Гц`];
    for (const issue of take.issues) quality.push(uiText(issues[issue] || issue));
    if (take.id === first) quality.push(uiText("Основной образец для синтеза"));
    else if (used.has(take.id)) quality.push(uiText("Включён в профиль"));
    entry.append(paragraph(`${indices.get(take.id)}. ${uiText(name)}`),
      paragraph(quality.filter(Boolean).join(" · ")));
    takes.append(entry);
  }
  const scores = report.pairs.map(pair =>
    `${indices.get(pair.left)} ↔ ${indices.get(pair.right)} · ${pair.cosine.toFixed(3)}`);
  status.textContent = first
    ? uiText`Основной образец №${indices.get(first)} · в профиль входят ${used.size} записей`
    : uiText("Для сравнения нужны два разных чистых дубля и доступный энкодер.");
  output.replaceChildren(takes, paragraph(scores.length
    ? uiText`Косинусная близость от −1 до +1 · ${scores.join(" · ")}`
    : uiText("Сходство пока не измерено.")));
  if (open) view.open = true;
}

/** Match the live selection to the analyzed group regardless of checkbox order. */
export function reportMatches(report, ids) {
  return Boolean(report && report.takes.length === ids.size && report.takes.every(take => ids.has(take.id)));
}
