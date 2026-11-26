import {profileExpressions} from "./facial-expressions.js";
import {uiText} from "./i18n.js";
import {effectiveSpeaker, mixedSpeakerLabel} from "./speaker-label.js";
import {buildFaceGaze} from "./speaker-gaze.js";

/** Inspect one observed face without inventing a speaker or borrowing another face's traits. */
export function faceProfile(faceId, evidence, index = [], durationMs = Infinity) {
  if (!Number.isInteger(faceId) || faceId < 1 || !evidence) return undefined;
  const timeline = evidence.editorTimeline;
  const matches = index.filter(item => item.faceId === faceId || item.faceIds?.includes(faceId));
  const entry = matches.length === 1 ? matches[0] : undefined;
  const observations = {...(entry?.faceId === faceId ? entry?.appearance : null), ...timeline?.faceAttributes?.[faceId]};
  if (observations.age === undefined && entry?.faceId === faceId && entry?.age !== undefined) observations.age = entry.age;
  const profile = {id: `face-${faceId}`, kind: "face", name: uiText`Лицо ${faceId}`,
    face: {id: faceId, basis: entry?.faceBasis ?? "visual", observations},
    clothing: timeline?.clothingByFace?.[faceId] ?? (entry?.faceId === faceId ? entry?.clothing : undefined)};
  if (entry) {
    profile.reference = {id: entry.id, pinned: entry.pinned, reused: entry.reused};
    profile.faceIds = entry.faceIds;
    profile.voice = entry.voice;
  }
  profile.gaze = buildFaceGaze(evidence, durationMs).get(faceId);
  profile.facialExpressions = profileExpressions(profile, evidence);
  return profile;
}

/** Apply unique editorial assignments to the current workbench and MP4 export. */
export function editorFaceBindings(evidence, cues) {
  if (!evidence) return null;
  const manual = new Map();
  const owners = new Map();
  for (const cue of cues) {
    const face = cue.visualFace ?? evidence.phrases?.[cue.startMs]?.activity?.referenceCandidate?.face;
    if (!cue.speakerNumber || face == null || cue.overlap) continue;
    if (!manual.has(face)) manual.set(face, new Set());
    if (!owners.has(cue.speakerNumber)) owners.set(cue.speakerNumber, new Set());
    manual.get(face).add(cue.speakerNumber);
    owners.get(cue.speakerNumber).add(face);
  }
  const bindings = {};
  for (const [face, binding] of Object.entries(evidence.faceSpeakers ?? {})) {
    const edits = manual.get(Number(face));
    if (binding.confirmedIdentity && (!edits || edits.size === 1 && edits.has(binding.speaker))) {
      bindings[face] = binding;
    } else if (!edits && !owners.has(binding.speaker)) {
      bindings[face] = binding;
    }
  }
  for (const [face, numbers] of manual) {
    if (numbers.size !== 1) continue;
    const [speaker] = numbers;
    if (!bindings[face] && owners.get(speaker).size === 1) bindings[face] = {speaker, basis: "editorial"};
  }
  const bySpeaker = new Map();
  for (const [face, binding] of Object.entries(bindings)) {
    if (binding.basis === "visible-only") continue;
    if (!bySpeaker.has(binding.speaker)) bySpeaker.set(binding.speaker, []);
    bySpeaker.get(binding.speaker).push([face, binding]);
  }
  for (const group of bySpeaker.values()) {
    if (group.length < 2) continue;
    for (const [face, binding] of group) {
      if (!binding.confirmedIdentity) delete bindings[face];
    }
  }
  return {...evidence, faceSpeakers: bindings};
}

/** Collect speaker references from the final local audio and visual associations. */
export function speakerProfiles(cues, evidence, index = [], gazes = new Map()) {
  const profiles = new Map();
  const primary = new Map(index.filter(entry => entry.faceIds?.length).map(entry => [entry.number, entry.faceId]));
  for (const cue of cues) {
    const number = effectiveSpeaker(cue);
    if (number === null || mixedSpeakerLabel(cue)) continue;
    if (!profiles.has(number)) profiles.set(number, {number, cueCount: 0, audioClusters: [], voiceSamples: []});
    const profile = profiles.get(number);
    profile.cueCount++;
    if (cue.speakerName) profile.name = cue.speakerName;
    if (cue.speaker != null && !profile.audioClusters.includes(cue.speaker)) profile.audioClusters.push(cue.speaker);
    if (!cue.overlap && cue.endMs - cue.startMs >= 600) {
      profile.voiceSamples.push({startMs: cue.startMs, endMs: cue.endMs, text: cue.text.slice(0, 100)});
      profile.voiceSamples.sort((a, b) => (b.endMs - b.startMs) - (a.endMs - a.startMs));
      profile.voiceSamples.length = Math.min(3, profile.voiceSamples.length);
    }
  }
  for (const [id, binding] of Object.entries(evidence?.faceSpeakers ?? {})) {
    const profile = profiles.get(binding.speaker);
    if (!profile && binding.basis !== "visible-only") {
      profiles.set(binding.speaker, {number: binding.speaker, cueCount: 0, audioClusters: [], voiceSamples: []});
    }
    const confirmed = profiles.get(binding.speaker);
    if (confirmed && binding.basis !== "visible-only") {
      if (binding.confirmedIdentity) (confirmed.faceIds ??= []).push(Number(id));
      if (confirmed.face && primary.has(binding.speaker) && primary.get(binding.speaker) !== Number(id)) continue;
      confirmed.face = {id: Number(id), basis: binding.basis, observations: evidence.editorTimeline?.faceAttributes?.[id]};
      confirmed.clothing = evidence.editorTimeline?.clothingByFace?.[id];
    }
  }
  for (const entry of index) {
    const profile = profiles.get(entry.number);
    if (profile && entry.voice) profile.voice = entry.voice;
    if (!evidence && profile && entry.faceId != null && cues.some(cue =>
      effectiveSpeaker(cue) === entry.number && cue.speakerProfileId === entry.id)) {
      profile.face = {id: entry.faceId, basis: entry.faceBasis};
    }
    if (profile && (profile.face?.id ?? null) === entry.faceId) {
      profile.reference = {id: entry.id, pinned: entry.pinned, reused: entry.reused};
      profile.clothing ??= entry.clothing;
      profile.age ??= entry.age;
      profile.appearance ??= entry.appearance;
    }
  }
  for (const profile of profiles.values()) {
    if (gazes.has(profile.number)) profile.gaze = gazes.get(profile.number);
    const expressions = profileExpressions(profile, evidence);
    if (expressions) profile.facialExpressions = expressions;
  }
  return [...profiles.values()].sort((a, b) => a.number - b.number);
}

function appendText(parent, value, tag = "p") {
  const node = document.createElement(tag);
  node.textContent = value;
  parent.append(node);
  return node;
}

/** Show visual observations and navigation to voice samples on the existing video clock. */
export function renderSpeakerProfiles(root, list, video, profiles, evidence, openProfile) {
  list.replaceChildren();
  root.hidden = !profiles.length;
  for (const profile of profiles) {
    const item = document.createElement("li");
    const title = appendText(item, profile.name || (uiText("Спикер ") + profile.number), "strong");
    const jpeg = evidence?.editorTimeline?.facePhotos?.[profile.face?.id];
    if (jpeg && /^[A-Za-z0-9+/]+={0,2}$/.test(jpeg)) {
      const photo = document.createElement("img");
      photo.src = "data:image/jpeg;base64," + jpeg;
      photo.alt = title.textContent;
      photo.loading = "lazy";
      item.prepend(photo);
    }
    const details = [profile.cueCount + uiText(" репл."), profile.audioClusters.length ? "Sherpa " + profile.audioClusters.join(", ") : ""];
    if (profile.reference?.pinned) details.push(profile.reference.reused ? uiText("Закреплён · найден повторно") : uiText("Закреплён"));
    if (profile.face) {
      const {id, basis, observations} = profile.face;
      details.push(uiText("Лицо ") + id, basis === "editorial" ? uiText("Ручная привязка") : "Light-ASD · SFace");
      if (observations?.clothingColor && !profile.clothing) details.push(uiText("Одежда в образце · ") + uiText(observations.clothingColor));
      if (observations?.headTurn) details.push(uiText("Поворот головы в образце · ") + uiText(observations.headTurn));
    }
    appendText(item, details.filter(Boolean).join(" · ")).className = "muted";
    const clothing = profile.clothing;
    if (/^#[0-9a-f]{6}$/i.test(clothing?.swatch) && clothing.sampleTimesMs?.length) {
      const note = appendText(item, uiText`Одежда по маскам · кадров: ${clothing.sampleTimesMs.length}`);
      const pattern = clothing.pattern === "plain" ? uiText("Однотонная ткань")
        : clothing.pattern === "patterned" ? uiText("Заметный рисунок или детали") : "";
      if (pattern) note.append(" · " + pattern);
      note.className = "muted clothing-swatch-note";
      const swatch = document.createElement("span");
      swatch.className = "clothing-swatch";
      swatch.style.backgroundColor = clothing.swatch;
      swatch.setAttribute("aria-hidden", "true");
      note.prepend(swatch);
    }
    const inspection = appendText(item, uiText("Открыть цифровой профиль"), "button");
    inspection.type = "button";
    inspection.onclick = () => openProfile(profile, inspection, evidence);
    if (video.hasAttribute("src")) {
      for (const sample of profile.voiceSamples) {
        const button = appendText(item, Math.floor(sample.startMs / 1000) + uiText(" с · ") + sample.text, "button");
        button.type = "button";
        button.onclick = () => { video.currentTime = sample.startMs / 1000; };
      }
    }
    list.append(item);
  }
}
