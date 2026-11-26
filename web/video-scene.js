import {uiLocale, uiText} from "./i18n.js";
import {colorPalette} from "./video-color-palette.js";

/** Render the measured regions and keep their percentages attached to a named denominator. */
function renderMoodbook(region, container, shots, video) {
  const rows = Array.isArray(shots) ? shots.filter(row => Number.isInteger(row?.startMs)) : [];
  const percent = new Intl.NumberFormat(uiLocale, {style: "percent", maximumFractionDigits: 1});
  container.replaceChildren();
  region.hidden = !rows.length;
  for (const [index, row] of rows.entries()) {
    const card = document.createElement("section");
    card.className = "moodbook-card";
    const seek = document.createElement("button");
    seek.type = "button";
    seek.textContent = `${uiText`План ${index + 1}`} · ${(row.startMs / 1000).toFixed(2)} ${uiText("с")}`;
    seek.addEventListener("click", () => {
      if (!video.videoWidth) return;
      video.pause();
      video.currentTime = (row.representativeMs ?? row.startMs) / 1000;
    });
    card.append(seek);
    for (const [title, samples] of [["Палитра плана", row.palette], ["Фон", row.backgroundPalette]]) {
      const caption = document.createElement("strong");
      caption.textContent = uiText(title);
      card.append(caption, colorPalette(samples, uiText(title)));
    }
    for (const person of row.characters ?? []) {
      if (!Number.isInteger(person.faceId) || !Number.isFinite(person.visibilityShare)) continue;
      const label = document.createElement("p");
      const visible = percent.format(person.visibilityShare);
      const area = Number.isFinite(person.frameAreaShare) ? percent.format(person.frameAreaShare) : null;
      label.textContent = `${uiText`Лицо ${person.faceId}`} · ${uiText`Видимость ${visible}`}`
        + (area ? " · " + uiText`Площадь рамки ${area}` : "");
      card.append(label, colorPalette(person.clothingPalette, uiText("Одежда")));
    }
    container.append(card);
  }
}
/** Show text and objects for the selected sampled frame and seek the player on demand. */
export function mountVideoScene(find, video, overlay) {
  const region = find("media-scene-result");
  const status = find("media-scene-status");
  const controls = find("media-scene-controls");
  const select = find("media-scene-time");
  const seek = find("media-scene-seek");
  const items = find("media-scene-items");
  const moodbook = find("media-moodbook");
  const moodbookShots = find("media-moodbook-shots");
  let frames = [];

  function updateAvailability() {
    seek.disabled = !video.videoWidth || !frames.length;
  }

  function showFrame() {
    items.replaceChildren();
    const frame = frames[Number(select.value)];
    if (!frame) return;
    const interfaces = frame.interfaces?.filter(panel => panel?.panelType).map(panel => panel?.panelType) ?? [];
    for (const [label, entries] of [[uiText("Интерфейс"), interfaces], [uiText("Текст"), frame.text.map(line => line.text)],
                                     [uiText("Предметы"), frame.objects.map(object => object.label)]]) {
      if (!entries.length) continue;
      const item = document.createElement("li");
      item.textContent = label + " — " + entries.join(" · ");
      items.append(item);
    }
  }

  select.addEventListener("change", showFrame);
  seek.addEventListener("click", () => {
    const frame = frames[Number(select.value)];
    if (!frame) return;
    video.pause();
    video.currentTime = frame.timeMs / 1000;
    overlay.setSceneFrame(frame);
    video.scrollIntoView({block: "nearest"});
    video.focus({preventScroll: true});
  });
  for (const event of ["loadedmetadata", "emptied", "error"]) video.addEventListener(event, updateAvailability);

  return {
    setData(evidence) {
      overlay.setSceneFrame(null);
      overlay.setSceneData(evidence);
      region.hidden = !evidence || evidence.status === "no-video";
      renderMoodbook(moodbook, moodbookShots, evidence?.moodbook, video);
      frames = (evidence?.frames ?? []).filter(frame => frame.text.length || frame.objects.length ||
        frame.interfaces?.some(panel => panel?.panelType));
      select.replaceChildren();
      for (const [index, frame] of frames.entries()) {
        const seconds = frame.timeMs / 1000;
        const time = `${Math.floor(seconds / 60)}:${(seconds % 60).toFixed(2).padStart(5, "0")}`;
        select.add(new Option(time, String(index)));
      }
      controls.hidden = !frames.length;
      const unavailable = Object.values(evidence?.stages ?? {}).some(stage => stage !== "ok");
      if (evidence?.status === "busy") status.textContent = uiText("Анализ кадров занят другим видео.");
      else if (evidence?.status === "failed") status.textContent = uiText("Анализ кадров недоступен.");
      else if (evidence?.status === "partial" && evidence.frameChanges) {
        status.textContent = uiText("Зоны изменений готовы. Распознавание текста и предметов продолжается.");
      }
      else {
        const count = evidence?.frames?.length ?? 0;
        const step = (evidence?.sampleMs ?? 3000) / 1000;
        status.textContent = "";
        if (count) status.textContent = evidence.ocrPasses === undefined
          ? uiText`Проверено кадров: ${count} · предметы каждые ${step} с. Сиреневые рамки выделяют текст, зелёные — предметы.`
          : uiText`Проверено кадров: ${count} · OCR: ${evidence.ocrPasses} областей · поиск предметов: ${evidence.objectFrames} кадров.`;
        const durations = [evidence?.decodeMs, evidence?.ocrMs, evidence?.objectMs];
        if (durations.every(Number.isFinite)) {
          const [decode, ocr, objects] = durations.map(ms => (ms / 1000).toFixed(1));
          status.textContent += uiText` Декодирование ${decode} с · OCR ${ocr} с · предметы ${objects} с.`;
        }
        const detail = evidence?.smallObjectPass;
        if (detail?.frames > 0 && Array.isArray(detail.shots) && Number.isFinite(detail.elapsedMs)) {
          status.textContent += uiText` Отдельный поиск мелких предметов: ${detail.shots.length} планов · ${(detail.elapsedMs / 1000).toFixed(1)} с.`;
        }
        if (!frames.length && count) status.textContent += uiText(" Уверенных находок нет.");
        if (unavailable) status.textContent += uiText(" Некоторые стадии недоступны.");
      }
      if (evidence) region.open = true;
      showFrame();
      updateAvailability();
    },
  };
}
