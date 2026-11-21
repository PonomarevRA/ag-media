import {uiText} from "./i18n.js";
import {packMedia} from "./person-layer.js";
import {effectiveSpeaker} from "./speaker-label.js";

/** Edit dubbing text for the transcript clock and request a local dub package. */
export function mountVideoDub(find, request) {
  const root = find("media-dub");
  const list = find("media-dub-cues");
  const status = find("media-dub-status");
  const submit = find("media-dub-submit");
  const translate = find("media-dub-translate");
  const dubLanguage = find("media-dub-language");
  const reduceAccent = find("media-dub-pronunciation");
  let source;
  let language;
  let target = dubLanguage.value;
  let rows = [];
  let active;

  function syncButton() {
    submit.disabled = !source || !rows.length || active === source;
    translate.disabled = submit.disabled;
    translate.hidden = target !== "en";
    dubLanguage.disabled = reduceAccent.disabled = Boolean(active && active === source);
  }

  /** Keep each language draft when switching dubbing language. */
  dubLanguage.onchange = () => {
    for (const {cue, input, drafts} of rows) {
      drafts[target] = input.value;
      input.value = drafts[dubLanguage.value] ?? (dubLanguage.value === language ? cue.sourceText : "");
    }
    target = dubLanguage.value;
    status.textContent = "";
    syncButton();
  };

  translate.onclick = async () => {
    const needed = rows.filter(row => !row.input.value.trim());
    if (!needed.length) {
      status.textContent = uiText("Английский текст уже заполнен.");
      return;
    }
    const file = source;
    active = file;
    syncButton();
    status.textContent = uiText("Готовим черновой перевод…");
    try {
      const result = await request("video-dub-translate",
        packMedia(file, {sourceLanguage: language, cues: needed.map(row => row.cue)}));
      if (source !== file) return;
      const texts = new Map(needed.map(({cue}, index) => [cue.startMs, result.translations[index]]));
      for (const {cue, input} of rows) {
        if (!input.value.trim()) input.value = texts.get(cue.startMs) ?? "";
      }
      status.textContent = uiText("Проверь перевод перед синтезом.");
    } catch (error) {
      if (source === file) status.textContent = error.message;
    } finally {
      if (active === file) active = undefined;
      syncButton();
    }
  };

  function download(blob) {
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = target === "en" ? "english-dub.zip" : "russian-dub.zip";
    link.click();
    setTimeout(() => URL.revokeObjectURL(url), 30000);
  }

  submit.onclick = async () => {
    if (submit.disabled) return;
    const invalid = rows.find(({pace}) => !pace.checkValidity());
    if (invalid) {
      invalid.pace.reportValidity();
      return;
    }
    const textKey = target === "en" ? "englishText" : "targetText";
    const cues = rows.map(({cue, input, pace}) => ({
      ...cue, [textKey]: input.value.trim(), tempo: pace.value ? pace.valueAsNumber : undefined,
    }));
    const missing = cues.findIndex(cue => !cue[textKey]);
    if (missing >= 0) {
      status.textContent = uiText("Заполни текст дубляжа для каждой реплики.");
      rows[missing].input.focus();
      return;
    }
    const file = source;
    active = file;
    syncButton();
    status.textContent = uiText("Отделяем фон и собираем дорожку дубляжа…");
    try {
      const blob = await request("video-dub", packMedia(file, {
        sourceLanguage: language, dubLanguage: target, cues,
        pronunciationMode: reduceAccent.checked ? "neutral" : "source",
      }));
      if (source !== file) return;
      download(blob);
      status.textContent = uiText("Дубляж сохранён.");
    } catch (error) {
      if (source === file) status.textContent = error.message;
    } finally {
      if (active === file) active = undefined;
      syncButton();
    }
  };

  return {setData(file, data) {
    const previous = new Map();
    if (source === file) {
      for (const {cue, input, pace, drafts} of rows) {
        previous.set(cue.startMs, {drafts: {...drafts, [target]: input.value}, tempo: pace.value});
      }
    } else {
      root.open = false;
      status.textContent = "";
    }
    source = file;
    language = data?.language;
    const canDub = file && data?.segments?.length && ["ru", "en"].includes(language);
    root.hidden = !canDub;
    list.replaceChildren();
    rows = [];
    for (const segment of canDub ? data.segments : []) {
      const {startMs, endMs, text} = segment;
      const speaker = effectiveSpeaker(segment);
      const row = document.createElement("li");
      const sourceLine = document.createElement("p");
      const seconds = (startMs / 1000).toFixed(2);
      const name = speaker ? uiText`Спикер ${speaker}` : uiText("Спикер не определён");
      sourceLine.textContent = `${seconds} · ${name} · ${text}`;
      const label = document.createElement("label");
      label.className = "select-field";
      label.textContent = uiText("Текст дубляжа");
      const input = document.createElement("textarea");
      input.rows = 2;
      input.maxLength = 1000;
      input.setAttribute("aria-label", uiText`Текст дубляжа реплики ${rows.length + 1}`);
      const saved = previous.get(startMs);
      const drafts = saved?.drafts ?? {};
      input.value = drafts[target] ?? (language === target ? text : "");
      label.append(input);
      const paceLabel = document.createElement("label");
      paceLabel.className = "select-field media-dub-pace";
      paceLabel.textContent = uiText("Темп (×)");
      const pace = document.createElement("input");
      pace.type = "number";
      pace.min = "0.8";
      pace.max = "1.8";
      pace.step = "0.01";
      pace.placeholder = uiText("Авто");
      pace.setAttribute("aria-label", uiText`Темп реплики ${rows.length + 1} (×, пустое поле — авто)`);
      pace.value = saved?.tempo ?? "";
      paceLabel.append(pace);
      row.append(sourceLine, label, paceLabel);
      list.append(row);
      rows.push({cue: {startMs, endMs, speaker, sourceText: text, overlap: Boolean(segment.overlap)}, input, pace, drafts});
    }
    syncButton();
  }};
}
