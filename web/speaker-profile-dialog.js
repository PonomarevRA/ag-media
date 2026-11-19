import {createMomentTimeline, momentTimecode} from "./profile-moment-timeline.js";
import {mountExpressionFramePreview} from "./expression-frame-preview.js";
import {uiLocale, uiText} from "./i18n.js";
import {expressionSummary} from "./facial-expression-view.js";
import {gazeLabel} from "./speaker-gaze.js";

const durationNumber = new Intl.NumberFormat(uiLocale, {maximumFractionDigits: 2});
const percentage = new Intl.NumberFormat(uiLocale, {style: "percent", maximumFractionDigits: 1});
const duration = ms => uiText`${durationNumber.format(ms / 1000)} с`;

function element(tag, text, className = "") {
  const node = document.createElement(tag);
  node.textContent = text;
  if (className) node.className = className;
  return node;
}

/** Describe a stable clothing swatch with a broad color family in the card. */
function swatchColor(hex) {
  if (!hex) return "";
  const [red, green, blue] = hex.slice(1).match(/../g).map(part => Number.parseInt(part, 16));
  const high = Math.max(red, green, blue);
  const low = Math.min(red, green, blue);
  if (high < 65) return uiText("тёмный");
  if (high - low < 28) return uiText(low > 180 ? "белый" : "серый");
  if (high === red) return uiText(blue > green + 25 ? "розовый" : green > red * .62 ? "оранжевый" : "красный");
  return uiText(high === blue ? "синий" : "зелёный");
}

/** Use the known fields, preserving the difference between absent data and an observation. */
function appearance(profile) {
  const observations = {...profile.appearance, ...profile.face?.observations};
  const clothing = profile.clothing;
  const group = document.createElement("dl");
  group.className = "speaker-profile-facts";
  group.dataset.agWidth = "content";
  group.dataset.agWidthMinEm = "6";
  function fact(label, value) {
    const definition = element("dd", value);
    const term = element("dt", uiText(label));
    for (const column of [term, definition]) {
      column.dataset.agWidth = "content";
      column.dataset.agWidthMinEm = "2";
    }
    group.append(term, definition);
    return definition;
  }
  const age = observations.age ?? profile.age;
  const ageFact = fact("Возраст по внешности", Number.isInteger(age) && age > 0 && age <= 120
    ? uiText`Около ${age} лет` : uiText("Нет данных"));
  if (observations.ageSource) ageFact.title = `${observations.ageSource} · ${observations.ageSamples ?? "?"}`;
  const appearanceClass = observations.appearanceClass === "female" ? uiText("Женский")
    : observations.appearanceClass === "male" ? uiText("Мужской") : "";
  const classFact = fact("Пол по внешности", appearanceClass || uiText("Нет данных"));
  if (observations.appearanceSource) classFact.title = `${observations.appearanceSource} · ${observations.appearanceSamples ?? "?"}`;
  const hairColor = /^#[0-9a-f]{6}$/i.test(observations.hairSwatch) ? observations.hairSwatch : "";
  const hairFact = fact("Цвет волос в кадре", hairColor || uiText("Нет данных"));
  addSwatch(hairFact, hairColor);
  if (observations.hairSource) hairFact.title = `${observations.hairSource} · ${observations.hairSamples ?? "?"}`;
  const pattern = clothing?.pattern === "plain" ? uiText("Однотонная ткань")
    : clothing?.pattern === "patterned" ? uiText("Заметный рисунок или детали") : "";
  const color = observations?.clothingColor ? uiText(observations.clothingColor) : "";
  const swatchValue = /^#[0-9a-f]{6}$/i.test(clothing?.swatch) ? clothing.swatch : "";
  const outfit = fact("Одежда", [clothing?.kind, color || swatchColor(swatchValue), pattern]
    .filter(Boolean).join(" · ") || uiText("Нет данных"));
  addSwatch(outfit, swatchValue);
  if (observations?.headTurn) fact("Поворот головы в образце", uiText(observations.headTurn));
  if (profile.face) fact("Связь с лицом", profile.face.basis === "visual" ? uiText("Лицо в кадре · связь с речью не подтверждена")
    : profile.face.basis === "editorial" ? uiText("Ручная привязка") : "Light-ASD · SFace");
  if (profile.faceIds?.length > 1) fact("Подтверждённые лица", profile.faceIds.join(", "));
  if (profile.voice) {
    fact("Голосовой профиль", uiText("Предварительный · весь исходник"));
    fact("Чистые образцы голоса", String(profile.voice.sampleCount ?? 0));
    fact("Приоритет доказательств", uiText("Подтверждённая речь в видео → голос → неизвестно"));
  }
  return group;
}

/** Show the measured color next to its accessible text value. */
function addSwatch(definition, value) {
  if (!value) return;
  const swatch = element("span", "", "clothing-swatch");
  swatch.style.backgroundColor = value;
  swatch.setAttribute("aria-hidden", "true");
  definition.prepend(swatch);
}

/** Present only measured intervals and distinguish estimated frontal gaze from a confirmation. */
function gazeSummary(gaze, seek) {
  let release = () => {};
  const section = document.createElement("section");
  section.append(element("h4", uiText("Направление взгляда")));
  if (!gaze?.observedMs) {
    section.append(element("p", uiText("Взгляд не измерен. Включи оценку взгляда при анализе видео."), "muted"));
    return {element: section, dispose: release};
  }
  section.append(element("p", uiText`Измерено ${duration(gaze.observedMs)}. Доли рассчитаны по измеренным интервалам.`, "muted"));
  const list = document.createElement("dl");
  list.className = "speaker-profile-facts";
  for (const direction of ["camera", "left", "right", "center"]) {
    const ms = gaze.directionsMs[direction] ?? 0;
    list.append(element("dt", gazeLabel(direction)),
                element("dd", `${duration(ms)} · ${percentage.format(ms / gaze.observedMs)}`));
  }
  section.append(list, element("p", uiText("Направление к камере оценивается по положению глаз и головы."), "muted"));
  const moments = gaze.intervals.filter(interval => interval.direction === "camera");
  if (moments.length) {
    const group = document.createElement("div");
    group.className = "speaker-profile-moments";
    group.append(element("h4", uiText("Моменты взгляда в сторону камеры")));
    const timeline = createMomentTimeline(moments.map(interval => ({...interval, timeMs: interval.startMs})), {
      label: uiText("Моменты взгляда в сторону камеры"), className: "profile-gaze-timeline",
      describe: entry => `${momentTimecode(entry.startMs)}–${momentTimecode(entry.endMs)} · ${uiText("В сторону камеры")}`,
      onActivate: entry => seek(entry.timeMs),
    });
    group.append(timeline.element);
    release = timeline.dispose;
    section.append(group);
  }
  return {element: section, dispose: release};
}

/** Open one reusable native dialog with view state and explicit focus return. */
export function mountSpeakerProfileDialog(find, video) {
  const dialog = find("speaker-profile-dialog");
  const title = find("speaker-profile-title");
  const info = find("speaker-profile-info");
  const technical = find("speaker-profile-technical");
  const uiButton = find("speaker-profile-view");
  const jsonButton = find("speaker-profile-json");
  const closeButton = find("speaker-profile-close");
  let opener;
  let currentProfile;
  let contentDisposers = [];
  const expressionPreview = mountExpressionFramePreview(video);

  /** Disconnect observers when a profile is rebuilt or leaves the dialog. */
  function clearContent() {
    for (const dispose of contentDisposers) dispose();
    contentDisposers = [];
  }
  function setView(json) {
    if (json && !technical.value && currentProfile) technical.value = JSON.stringify(currentProfile, null, 2);
    info.hidden = json;
    technical.hidden = !json;
    uiButton.setAttribute("aria-pressed", String(!json));
    jsonButton.setAttribute("aria-pressed", String(json));
  }
  uiButton.onclick = () => setView(false);
  jsonButton.onclick = () => setView(true);
  closeButton.onclick = () => dialog.close();
  dialog.addEventListener("close", () => {
    clearContent();
    if (opener?.isConnected) opener.focus({preventScroll: true});
    opener = currentProfile = undefined;
    technical.value = "";
  });

  return {
    open(profile, button, evidence) {
      if (!profile) return;
      clearContent();
      opener = button;
      currentProfile = profile;
      technical.value = "";
      title.textContent = profile.name || uiText`Спикер ${profile.number}`;
      const seek = time => {
        if (!video.hidden && video.readyState > 0) {
          video.pause();
          video.currentTime = time / 1000;
          dialog.close();
        }
      };
      const facts = document.createElement("div");
      facts.className = "speaker-profile-appearance";
      const jpeg = evidence?.editorTimeline?.facePhotos?.[profile.face?.id];
      if (typeof jpeg === "string" && /^[A-Za-z0-9+/]+={0,2}$/.test(jpeg)) {
        const photo = document.createElement("img");
        photo.src = "data:image/jpeg;base64," + jpeg;
        photo.alt = title.textContent;
        facts.append(photo);
      }
      facts.append(appearance(profile));
      const gaze = gazeSummary(profile.gaze, seek);
      const expressions = expressionSummary(profile.facialExpressions, seek,
        (entry, button) => expressionPreview.open(entry, evidence, button));
      contentDisposers = [gaze.dispose, expressions.dispose];
      info.replaceChildren(facts, gaze.element, expressions.element);
      setView(false);
      if (!dialog.open) dialog.showModal();
      closeButton.focus();
    },
    reset() {
      expressionPreview.clear();
      clearContent();
      if (dialog.open) dialog.close();
    },
  };
}
