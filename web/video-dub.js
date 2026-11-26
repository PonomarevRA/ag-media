import {apiText, uiText} from "./i18n.js";
import {packMedia} from "./person-layer.js";
import {effectiveSpeaker} from "./speaker-label.js";
import {accentStyleLabel, voiceProfileLabel, voiceProfileSummary} from "./voice-catalog.js";

/** Edit dubbing text for the transcript clock and request a local dub package. */
export function mountVideoDub(find, request) {
  const root = find("media-dub");
  const list = find("media-dub-cues");
  const status = find("media-dub-status");
  const submit = find("media-dub-submit");
  const translate = find("media-dub-translate");
  const dubLanguage = find("media-dub-language");
  const translationField = find("media-dub-translation-field");
  const method = find("media-dub-translation-method");
  const translationStatus = find("media-dub-translation-status");
  const modOption = method.querySelector('[value="model"]');
  const whisperOption = method.querySelector('[value="whisper"]');
  const reduceAccent = find("media-dub-pronunciation");
  const profilePanel = find("media-dub-profiles");
  let source;
  let language;
  let target = dubLanguage.value;
  let rows = [];
  let active;
  let profileRequest = 0;
  let translationRequest = 0;
  let translationInfo;
  let preferredMethod;
  let profileLoading = false;
  let profileLoadFailed = false;
  const profileSelects = new Map();
  const profileDrafts = new Map();

  function syncButton() {
    const unavailableProfile = [...profileSelects.values()].some(select => select.selectedOptions[0]?.disabled);
    submit.disabled = !source || !rows.length || active === source || profileLoading || profileLoadFailed || unavailableProfile;
    const working = Boolean(active && active === source);
    const crossLanguage = Boolean(source && language !== target);
    translate.hidden = !crossLanguage || method.value === "manual";
    translate.disabled = working || !rows.length || (method.value === "model" && translationInfo?.status !== "ready");
    translationField.hidden = !crossLanguage;
    const message = crossLanguage && method.value !== "whisper" && translationInfo?.message;
    translationStatus.textContent = message ? apiText(message) : "";
    translationStatus.hidden = !message;
    dubLanguage.disabled = reduceAccent.disabled = method.disabled = working;
    for (const select of profileSelects.values()) select.disabled = active === source;
  }

  /** Load verified voices for the delivery language and explicitly marked accent profiles. */
  async function loadProfiles() {
    const revision = ++profileRequest;
    profileSelects.clear();
    profilePanel.replaceChildren();
    profilePanel.hidden = true;
    profileLoading = Boolean(root.open && source && rows.length);
    profileLoadFailed = false;
    syncButton();
    if (!profileLoading) return;
    let profiles;
    try {
      ({profiles} = await request("voice-catalog", undefined, {language: target, purpose: "dubbing"}));
      profiles = profiles.filter(profile => profile.dubbingReady);
    } catch {
      if (revision !== profileRequest) return;
      profileLoading = false;
      profileLoadFailed = [...(profileDrafts.get(target)?.values() || [])].some(Boolean);
      if (profileLoadFailed) {
        profilePanel.hidden = false;
        profilePanel.textContent = uiText("Профили не загрузились. Открой дубляж заново.");
      }
      syncButton();
      return;
    }
    if (revision !== profileRequest) return;
    profileLoading = false;
    const drafts = profileDrafts.get(target) || new Map();
    profileDrafts.set(target, drafts);
    if (!profiles?.length && ![...drafts.values()].some(Boolean)) {
      syncButton();
      return;
    }
    profiles ||= [];
    const speakers = [...new Set(rows.map(row => row.cue.speaker).filter(Number.isInteger))];
    const versions = new Map(profiles.map(row => [String(row.version), row]));
    for (const speaker of speakers.slice(0, 24)) {
      const field = document.createElement("div");
      field.className = "media-dub-voice";
      field.dataset.agWidth = "content";
      field.dataset.agWidthMinEm = "6";
      const label = document.createElement("label");
      label.className = "select-field";
      label.textContent = uiText`Голос · спикер ${speaker}`;
      const select = document.createElement("select");
      select.add(new Option(uiText("Голос из видео"), ""));
      for (const profile of profiles) {
        const accent = accentStyleLabel(profile.accentStyle);
        const label = accent ? `${voiceProfileLabel(profile)} · ${accent}` : voiceProfileLabel(profile);
        select.add(new Option(label, String(profile.version)));
      }
      const previous = drafts.get(speaker);
      if (previous && !versions.has(previous)) {
        const missing = new Option(uiText`Профиль ${previous} недоступен`, previous);
        missing.disabled = true;
        select.add(missing);
      }
      if (previous) select.value = previous;
      const evidence = document.createElement("p");
      evidence.className = "muted";
      evidence.id = `media-dub-voice-summary-${speaker}`;
      select.setAttribute("aria-describedby", evidence.id);
      const updateEvidence = () => {
        const chosen = versions.get(select.value);
        evidence.textContent = chosen ? voiceProfileSummary(chosen) : "";
        evidence.hidden = !chosen;
      };
      select.onchange = () => {
        drafts.set(speaker, select.value);
        updateEvidence();
        syncButton();
      };
      updateEvidence();
      profileSelects.set(speaker, select);
      label.append(select);
      field.append(label, evidence);
      profilePanel.append(field);
    }
    profilePanel.hidden = !profileSelects.size;
    syncButton();
  }

  /** Read the selected mod and whether the system has both languages, without sending cue text. */
  async function loadTranslator() {
    const revision = ++translationRequest;
    translationInfo = undefined;
    const canUseWhisper = target === "en" && ["ru", "zh"].includes(language);
    whisperOption.disabled = whisperOption.hidden = !canUseWhisper;
    modOption.disabled = true;
    method.value = preferredMethod === "manual" || !canUseWhisper ? "manual" : "whisper";
    syncButton();
    if (!root.open || !source || language === target) return;
    try {
      const info = await request("video-dub-text-translation", undefined, {source: language, target});
      if (revision !== translationRequest) return;
      translationInfo = info;
      const installedOrDownloadable = ["ready", "needs-languages"].includes(info.status);
      modOption.disabled = !installedOrDownloadable;
      modOption.textContent = info.provider === "builtin.apple-translation"
        ? uiText("Перевод Apple · офлайн") : uiText("Мод перевода");
      const usablePreference = preferredMethod === "manual" || preferredMethod === "whisper" && canUseWhisper;
      if (installedOrDownloadable && !usablePreference) method.value = "model";
      syncButton();
    } catch {
      if (revision === translationRequest) syncButton();
    }
  }

  method.onchange = () => {
    preferredMethod = method.value;
    status.textContent = "";
    syncButton();
  };
  root.ontoggle = () => {
    if (!root.open) return;
    void loadProfiles();
    void loadTranslator();
  };

  /** Keep each language draft when switching dubbing language. */
  dubLanguage.onchange = () => {
    for (const {cue, input, drafts} of rows) {
      drafts[target] = input.value;
      input.value = drafts[dubLanguage.value] ?? (dubLanguage.value === language ? cue.sourceText : "");
    }
    target = dubLanguage.value;
    status.textContent = "";
    void loadProfiles();
    void loadTranslator();
    syncButton();
  };

  /** Apply the ordered translation batch only to the exact still-empty text fields. */
  async function fillTranslation(file, outputLanguage, selectedMethod) {
    const visibleRows = rows;
    const needed = rows.filter(row => !row.input.value.trim());
    if (!needed.length) return true;
    const result = selectedMethod === "model"
      ? await request("video-dub-text-translation", {
        sourceLanguage: language, targetLanguage: outputLanguage, provider: translationInfo?.provider,
        texts: needed.map(row => row.cue.sourceText),
      })
      : await request("video-dub-translate", packMedia(file, {
        sourceLanguage: language, cues: needed.map(row => row.cue),
      }));
    if (source !== file || rows !== visibleRows || target !== outputLanguage) return false;
    for (const [index, {input}] of needed.entries()) {
      if (!input.value.trim()) input.value = result.translations[index] ?? "";
    }
    return true;
  }

  translate.onclick = async () => {
    if (translate.disabled) return;
    if (rows.every(row => row.input.value.trim())) {
      status.textContent = uiText("Текст перевода уже заполнен.");
      return;
    }
    const file = source;
    active = file;
    syncButton();
    status.textContent = uiText("Готовим черновой перевод…");
    try {
      if (await fillTranslation(file, target, method.value)) {
        status.textContent = uiText("Проверь перевод перед синтезом.");
      }
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
    const file = source;
    const outputLanguage = target;
    const selectedMethod = method.value;
    const speakerProfiles = Object.fromEntries([...profileSelects]
      .filter(([, select]) => select.value).map(([speaker, select]) => [speaker, Number(select.value)]));
    const pronunciationMode = reduceAccent.checked ? "neutral" : "source";
    active = file;
    syncButton();
    try {
      if (selectedMethod === "model" && language !== outputLanguage && rows.some(row => !row.input.value.trim())) {
        status.textContent = uiText("Готовим черновой перевод…");
        if (!await fillTranslation(file, outputLanguage, selectedMethod)) return;
      }
      const textKey = outputLanguage === "en" ? "englishText" : "targetText";
      const cues = rows.map(({cue, input, pace}) => ({
        ...cue, [textKey]: input.value.trim(), tempo: pace.value ? pace.valueAsNumber : undefined,
      }));
      const missing = cues.findIndex(cue => !cue[textKey]);
      if (missing >= 0) {
        status.textContent = uiText("Заполни текст дубляжа для каждой реплики.");
        rows[missing].input.focus();
        return;
      }
      status.textContent = uiText("Отделяем фон и собираем дорожку дубляжа…");
      const blob = await request("video-dub", packMedia(file, {
        sourceLanguage: language, dubLanguage: outputLanguage, cues,
        pronunciationMode,
        speakerProfiles,
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
      preferredMethod = undefined;
      status.textContent = "";
      profileDrafts.clear();
    }
    source = file;
    language = data?.language;
    const canDub = file && data?.segments?.length && ["ru", "en", "zh"].includes(language);
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
    void loadProfiles();
    void loadTranslator();
  }};
}
