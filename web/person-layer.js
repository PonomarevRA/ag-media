import {uiText} from "./i18n.js";
/** Send bounded track metadata followed by the original local video bytes. */
export function packMedia(source, metadata) {
  const layer = new Blob([JSON.stringify(metadata)]);
  const header = new DataView(new ArrayBuffer(4));
  header.setUint32(0, layer.size);
  return new Blob([header.buffer, layer, source], {type: "application/octet-stream"});
}

/** Offer a native-frame clothing mask export for one stable face track. */
export function mountClothingExport(find, video, request) {
  const root = find("media-clothing-export");
  const target = find("media-clothing-face");
  const start = find("media-clothing-start");
  const duration = find("media-clothing-duration");
  const position = find("media-clothing-position");
  const submit = find("media-clothing-submit");
  const status = find("media-clothing-status");
  let active;
  let busy = false;

  function readPosition() {
    const seconds = Number.isFinite(video.currentTime) ? video.currentTime : 0;
    start.value = String(Math.round(seconds * 100) / 100);
  }

  position.addEventListener("click", readPosition);
  root.addEventListener("toggle", () => { if (root.open && start.value === "") readPosition(); });

  submit.addEventListener("click", async () => {
    if (!active || busy || !start.reportValidity() || !duration.reportValidity()) return;
    const job = active;
    const startMs = Math.round(start.valueAsNumber * 1000);
    const endMs = Math.min(job.endMs, startMs + Math.round(duration.valueAsNumber * 1000));
    if (startMs >= endMs) {
      status.textContent = uiText("Выбери начало в пределах обработанного трека.");
      return;
    }
    busy = submit.disabled = true;
    status.textContent = uiText("Готовим покадровые маски одежды…");
    try {
      const faceId = Number(target.value);
      const profile = job.profiles.get(faceId);
      const archive = await request("person-masks", packMedia(job.source, {
        faceId, startMs, endMs, sourceDigest: job.digest, timeline: job.timeline,
        speakerNumber: profile?.number, profileId: profile?.reference?.id,
      }));
      const url = URL.createObjectURL(archive);
      const link = document.createElement("a");
      link.href = url;
      link.download = "clothing-masks.zip";
      link.click();
      setTimeout(() => URL.revokeObjectURL(url), 30000);
      if (job === active) status.textContent = uiText("Архив масок скачан.");
    } catch (error) {
      if (job === active) status.textContent = error.message;
    } finally {
      busy = submit.disabled = false;
    }
  });

  return {
    update(source, digest, evidence, profiles = []) {
      const timeline = evidence?.timeline;
      const labels = evidence?.faceSpeakers;
      const ids = new Set(timeline?.frames.flatMap(frame => frame.map(hit => hit[6] ?? hit[0])) ?? []);
      const previous = target.value;
      if (source !== active?.source) {
        start.value = "";
        status.textContent = "";
        root.open = false;
      }
      active = source && digest && ids.size ? {source, digest, timeline,
        profiles: new Map(profiles.filter(profile => profile.face).map(profile => [profile.face.id, profile])),
        endMs: timeline.frames.length * timeline.stepMs} : null;
      root.hidden = !active;
      target.replaceChildren();
      for (const id of ids) {
        const binding = labels?.[id];
        const label = binding?.basis !== "visible-only" && binding?.speaker
          ? uiText("Спикер ") + binding.speaker + uiText(" · лицо ") + id : uiText("Лицо ") + id;
        target.add(new Option(label, String(id)));
      }
      if ([...ids].some(id => String(id) === previous)) target.value = previous;
    },
  };
}
