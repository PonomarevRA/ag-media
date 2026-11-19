import {uiText} from "./i18n.js";
import {effectiveSpeaker} from "./speaker-label.js";

/** Share the actual audio-visual state between detailed evidence and compact chat badges. */
export function activityStatus(activity) {
  if (!activity) return null;
  const active = activity.state === "visible-active";
  const label = active
    ? uiText`${activity.basis === "editorial-with-activity" ? uiText("Ручная метка + Light-ASD") : "Light-ASD"} · речь видна в ${activity.activeFrames} из ${activity.phraseFrames} кадров`
    : activity.state === "partial-or-ambiguous"
      ? uiText("Light-ASD · активность видна только на части фразы или у нескольких лиц")
      : uiText("Light-ASD · говорящий в кадре не подтверждён");
  return {kind: "activity", label, icon: active ? "face-check" : activity.state === "partial-or-ambiguous" ? "face-partial" : "face-question"};
}

/** Display the candidate faces for one timed phrase and save a photo on confirmation. */
export function mountVideoEvidence(evidence, segment, save) {
  const element = document.createElement("div");
  element.className = "video-evidence";
  const activity = evidence.activity;

  if (activity) {
    const note = document.createElement("p");
    note.className = "muted";
    note.dataset.transcriptStatus = "activity";
    note.textContent = activityStatus(activity).label;
    element.append(note);
  }

  const candidates = [];
  if (evidence.activePhoto) {
    candidates.push({jpeg: evidence.activePhoto, timeMs: activity.sample.time_ms, title: uiText("Лицо с подтверждённой активностью")});
  }
  for (const [index, jpeg] of (evidence.photos ?? []).entries()) {
    candidates.push({jpeg, timeMs: evidence.photoTimeMs, title: uiText`Лицо в кадре ${index + 1}`});
  }

  const controls = [];
  const status = document.createElement("span");
  status.dataset.transcriptStatus = "photo";
  status.setAttribute("role", "status");
  const remove = document.createElement("button");
  remove.type = "button";
  remove.textContent = uiText("Убрать фото");
  function update() {
    const chosen = segment.speakerPhoto;
    const speaker = segment.overlap && !segment.speakerNumber ? null : effectiveSpeaker(segment);
    for (const {button, jpeg} of controls) {
      button.disabled = speaker == null;
      button.setAttribute("aria-pressed", String(chosen?.jpeg === jpeg));
    }
    remove.hidden = !chosen;
    status.textContent = chosen ? uiText`Фото подтверждено · Спикер ${speaker}`
      : speaker == null && candidates.length ? uiText("Выбери метку спикера, чтобы прикрепить фото") : "";
  }

  if (candidates.length) {
    const details = document.createElement("details");
    const heading = document.createElement("summary");
    heading.textContent = uiText("Фото из видео");
    details.open = Boolean(evidence.activePhoto);
    const gallery = document.createElement("div");
    gallery.className = "video-photos";
    for (const {jpeg, timeMs, title} of candidates) {
      if (!/^[A-Za-z0-9+/]+={0,2}$/.test(jpeg)) continue;
      const card = document.createElement("div");
      const photo = document.createElement("img");
      photo.src = "data:image/jpeg;base64," + jpeg;
      photo.width = 64;
      photo.height = 64;
      photo.alt = title;
      photo.loading = "lazy";
      const button = document.createElement("button");
      button.type = "button";
      button.textContent = uiText("Прикрепить");
      button.setAttribute("aria-label", uiText`Прикрепить фото · ${title}`);
      button.onclick = () => {
        const speaker = effectiveSpeaker(segment);
        if (segment.overlap && !segment.speakerNumber || speaker == null) return;
        segment.speakerPhoto = {jpeg, timeMs, speaker};
        update();
        save();
      };
      controls.push({button, jpeg});
      card.append(photo, button);
      gallery.append(card);
    }
    remove.onclick = () => {
      delete segment.speakerPhoto;
      update();
      save();
    };
    details.append(heading, gallery, remove, status);
    element.append(details);
  }
  update();
  return {element, update};
}
