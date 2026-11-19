import {uiText} from "./i18n.js";
import {effectiveSpeaker, mixedSpeakerLabel} from "./speaker-label.js";
import {attachStudioIcon} from "./studio-icons.js";
import {activityStatus} from "./video-evidence.js";

const controllers = new WeakMap();

/** Prefer an assigned portrait; a single cue photo stays a visibly unconfirmed local candidate. */
export function transcriptPortrait(segment, profile, evidence) {
  if (mixedSpeakerLabel(segment)) return null;
  const number = effectiveSpeaker(segment);
  if (number === null) return null;
  const valid = jpeg => typeof jpeg === "string" && /^[A-Za-z0-9+/]+={0,2}$/.test(jpeg);
  const chosen = segment.speakerPhoto;
  if (chosen?.speaker === number && valid(chosen.jpeg)) return {jpeg: chosen.jpeg, confirmed: true};
  const faceId = profile?.face?.id;
  const binding = evidence?.faceSpeakers?.[faceId];
  const portrait = evidence?.editorTimeline?.facePhotos?.[faceId];
  if (binding?.speaker === number && binding.basis !== "visible-only" && valid(portrait)) {
    return {jpeg: portrait, confirmed: true};
  }
  const phrase = evidence?.phrases?.[segment.startMs];
  if (phrase?.activity?.state === "visible-active" && valid(phrase.activePhoto)) {
    const activeBinding = evidence?.faceSpeakers?.[phrase.activity.face];
    return {jpeg: phrase.activePhoto,
      confirmed: activeBinding?.speaker === number && activeBinding.basis !== "visible-only"};
  }
  if (phrase?.photos?.length === 1 && valid(phrase.photos[0])) {
    return {jpeg: phrase.photos[0], confirmed: false};
  }
  return null;
}

/** Treat aliases as literal, short labels keyed by existing speaker IDs, never as identity evidence. */
export function cleanSpeakerNames(value) {
  const result = {};
  if (!value || typeof value !== "object" || Array.isArray(value)) return result;
  for (const [key, name] of Object.entries(value)) {
    if (!/^[1-9]\d{0,5}$/.test(key) || typeof name !== "string") continue;
    const text = name.replace(/[\p{Cc}\p{Cf}]/gu, "").trim().slice(0, 80);
    if (text) result[key] = text;
  }
  return result;
}

/** A digest owns names across reopenings; absent digests fall back to a stable saved-material ID only. */
export function speakerNamesKey(report) {
  const digest = typeof report?.sourceDigest === "string" ? report.sourceDigest.trim() : "";
  const id = typeof report?.id === "string" ? report.id.trim() : "";
  const key = digest ? "source:" + digest : id ? "material:" + id : "";
  return key ? "voice-studio:speaker-names:" + key.slice(0, 256) : null;
}

/** Consecutive turns keep their side; a new known speaker flips it. Mixed/unknown speech stays centred. */
export function transcriptSides(cues) {
  let previous = null, right = false;
  return cues.map(cue => {
    const number = mixedSpeakerLabel(cue) ? null : effectiveSpeaker(cue);
    if (number === null) return "center";
    if (previous !== null && previous !== number) right = !right;
    previous = number;
    return right ? "right" : "left";
  });
}

function iconButton(icon, label, extra = "") {
  const button = document.createElement("button");
  button.type = "button";
  button.className = "studio-icon transcript-tip " + extra;
  button.dataset.icon = icon;
  button.dataset.hint = label;
  button.setAttribute("aria-label", label);
  attachStudioIcon(button);
  installChatHint(button);
  return button;
}

/** Keep the hint inside the transcript scrollport rather than clipping it at either edge. */
export function positionChatHint(button) {
  const owner = button.closest(".editor-inspector-body");
  if (!owner) return;
  const area = owner.getBoundingClientRect();
  const box = button.getBoundingClientRect();
  const width = Math.min(224, owner.clientWidth - 16);
  const left = Math.max(area.left + 8, Math.min(box.left, area.left + owner.clientWidth - width - 8));
  button.style.setProperty("--chat-hint-width", `${width}px`);
  button.style.setProperty("--chat-hint-left", `${left - box.left}px`);
  const height = Number.parseFloat(getComputedStyle(button, "::after").height) || 100;
  button.dataset.hintSide = box.bottom + height + 8 > Math.min(area.bottom, innerHeight)
    && box.top - height - 8 >= Math.max(area.top, 0) ? "above" : "below";
}

/** Delay hover hints250ms without allocating invisible overflow; keyboard hints stay immediate. */
export function installChatHint(button) {
  let timer;
  const hide = () => { clearTimeout(timer); delete button.dataset.hintReady; };
  button.addEventListener("pointerenter", () => {
    hide();
    timer = setTimeout(() => {
      if (!button.isConnected || !button.matches(":hover")) return;
      button.dataset.hintReady = "";
      positionChatHint(button);
    }, 250);
  });
  button.addEventListener("pointerleave", hide);
  button.addEventListener("blur", hide);
  button.addEventListener("focus", () => positionChatHint(button));
}

/** Own source-scoped aliases and message layout without replacing native transcript/editor handlers. */
export function mountTranscriptChat(list, storage) {
  if (storage === undefined) {
    try { storage = window.localStorage; } catch { storage = null; }
  }
  let report = null, scope = null, aliases = {}, profiles = new Map(), evidence = null;
  const announcement = document.createElement("p");
  announcement.className = "visually-hidden";
  announcement.setAttribute("role", "status");
  list.after(announcement);

  function items() { return [...list.children].map(item => controllers.get(item)).filter(Boolean); }

  /** Update the visible run after a manual speaker edit or a streamed report patch. */
  function refresh() {
    const rows = items();
    const sides = transcriptSides(rows.map(row => row.segment));
    rows.forEach((row, i) => { row.item.dataset.side = sides[i]; row.update(); });
  }

  /** Persist a display alias in this source's browser slot/report metadata; canonical speaker IDs stay untouched. */
  function rename(number, name) {
    const clean = cleanSpeakerNames({[number]: name});
    if (clean[number]) aliases[number] = clean[number];
    else delete aliases[number];
    if (report) report.speakerNames = {...aliases};
    let durable = false;
    if (scope && storage) {
      try { storage.setItem(scope, JSON.stringify(aliases)); durable = true; } catch {}
    }
    refresh();
    announcement.textContent = uiText(durable ? "Имя сохранено для этого ролика" : "Имя изменено для текущей сессии");
  }

  /** Compose one bubble while retaining references to the original status nodes and controls. */
  function append(item, words, speech, segment) {
    item.classList.add("transcript-message");
    words.classList.add("transcript-message-body");
    const statusSources = [...words.querySelectorAll("[data-transcript-status]")];
    const heading = document.createElement("div");
    heading.className = "transcript-message-heading";
    const avatar = document.createElement("span");
    avatar.className = "transcript-avatar";
    avatar.setAttribute("aria-hidden", "true");
    const identity = document.createElement("div");
    identity.className = "transcript-identity";
    const name = document.createElement("button");
    name.type = "button";
    name.setAttribute("aria-expanded", "false");
    name.className = "transcript-name";
    name.dataset.icon = "edit";
    attachStudioIcon(name);
    const caption = document.createElement("span");
    name.append(caption);
    const badges = document.createElement("span");
    badges.className = "transcript-statuses";
    identity.append(name, badges);
    const stamp = item.firstElementChild;
    stamp.classList.add("transcript-message-time");
    heading.append(avatar, identity, stamp);

    const details = document.createElement("details");
    details.className = "transcript-message-details";
    const summary = document.createElement("summary");
    summary.textContent = uiText("Правка и данные анализа");
    details.append(summary);
    for (const node of [...words.childNodes]) {
      if (node === speech) continue;
      if (node.classList?.contains("transcript-speaker-label")) node.remove();
      else details.append(node);
    }
    const form = document.createElement("form");
    form.className = "transcript-rename";
    form.id = "transcript-rename-" + segment.startMs;
    name.setAttribute("aria-controls", form.id);
    form.hidden = true;
    const input = document.createElement("input");
    input.type = "text";
    input.maxLength = 80;
    input.setAttribute("aria-label", uiText("Имя спикера"));
    input.placeholder = uiText("Пустое имя — вернуть номер");
    const save = iconButton("check", uiText("Сохранить имя"));
    save.type = "submit";
    const cancel = iconButton("close", uiText("Отменить переименование"));
    form.append(input, save, cancel);
    let editingNumber = null;
    function closeRename(focus = true) {
      form.hidden = true;
      name.setAttribute("aria-expanded", "false");
      editingNumber = null;
      if (focus) name.focus({preventScroll: true});
    }
    name.onclick = () => {
      editingNumber = mixedSpeakerLabel(segment) ? null : effectiveSpeaker(segment);
      if (editingNumber === null) return;
      input.value = aliases[editingNumber] ?? segment.speakerName ?? profiles.get(editingNumber)?.name ?? "";
      form.hidden = false;
      name.setAttribute("aria-expanded", "true");
      input.focus();
      input.select();
    };
    cancel.onclick = () => closeRename();
    form.onkeydown = event => {
      if (event.key === "Escape") { event.preventDefault(); event.stopPropagation(); closeRename(); }
    };
    form.onsubmit = event => {
      event.preventDefault();
      if (editingNumber === null || editingNumber !== effectiveSpeaker(segment) || mixedSpeakerLabel(segment)) {
        closeRename();
        announcement.textContent = uiText("Спикер изменился. Открой имя ещё раз.");
        return;
      }
      rename(editingNumber, input.value);
      closeRename();
    };
    words.replaceChildren(heading, form, speech, details);
    speech.classList.add("transcript-message-speech");
    let avatarKey, statusKey;

    /** Project current names, eligible portraits and actual evidence badges without asserting a new identity. */
    function update() {
      const mixed = mixedSpeakerLabel(segment);
      const number = mixed ? null : effectiveSpeaker(segment);
      item.dataset.speaker = String(number ?? "unknown");
      const label = mixed ?? aliases[number] ?? segment.speakerName ?? profiles.get(number)?.name
        ?? (number ? uiText`Спикер ${number}` : uiText("Спикер не определён"));
      if (caption.textContent !== label) caption.textContent = label;
      name.disabled = number === null;
      name.setAttribute("aria-label", uiText`Переименовать: ${label}`);
      if (editingNumber !== null && editingNumber !== number) closeRename(false);
      const profile = profiles.get(number);
      const portrait = transcriptPortrait(segment, profile, evidence);
      avatar.dataset.portrait = portrait ? portrait.confirmed ? "confirmed" : "candidate" : "absent";
      const customName = aliases[number] ?? segment.speakerName ?? profile?.name;
      const initials = customName ? customName.split(/\s+/u).slice(0, 2).map(part => [...part][0] ?? "").join("").toLocaleUpperCase() : String(number ?? "?");
      const nextAvatar = portrait?.jpeg ?? (mixed ? "mixed" : "initials:" + initials);
      if (avatarKey !== nextAvatar) {
        avatarKey = nextAvatar;
        avatar.replaceChildren();
        if (portrait) {
          const img = document.createElement("img");
          img.src = "data:image/jpeg;base64," + portrait.jpeg;
          img.alt = "";
          img.loading = "lazy";
          avatar.append(img);
        } else avatar.textContent = mixed ? "⋯" : initials;
      }
      const statuses = [];
      if (portrait && !portrait.confirmed) statuses.push({kind: "photo-candidate", icon: "photo",
        label: uiText("Фото из кадра · связь с голосом не подтверждена")});
      const activity = activityStatus(evidence?.phrases?.[segment.startMs]?.activity);
      if (activity) statuses.push(activity);
      for (const source of statusSources) {
        const kind = source.dataset.transcriptStatus;
        if (kind === "activity" || source.hidden || !source.textContent.trim()) continue;
        statuses.push({kind, label: source.textContent.trim(), icon: {visual: "face", assignment: "pin", overlap: "voices", correction: "spell", acoustic: "mic", photo: "photo"}[kind] ?? "info"});
      }
      const key = JSON.stringify(statuses);
      if (key !== statusKey) {
        statusKey = key;
        badges.replaceChildren(...statuses.map(status => iconButton(status.icon, status.label, "transcript-status-icon")));
      }
    }
    controllers.set(item, {item, segment, update});
    words.addEventListener("speaker-edit", refresh);
    update();
  }

  return {append, refresh,
    reset() { report = null; scope = null; aliases = {}; profiles.clear(); evidence = null; announcement.textContent = ""; },
    /** Seed imported names, overlay saved browser names, then retain same-source session edits across patches. */
    setData(value, speakerProfiles, visualEvidence) {
      const key = speakerNamesKey(value);
      let saved = {};
      try { if (key && storage) saved = cleanSpeakerNames(JSON.parse(storage.getItem(key) ?? "{}")); } catch {}
      const retained = key === scope ? aliases : {};
      aliases = {...cleanSpeakerNames(value.speakerNames), ...saved, ...retained};
      scope = key;
      report = value;
      report.speakerNames = {...aliases};
      profiles = new Map(speakerProfiles.map(profile => [profile.number, profile]));
      evidence = visualEvidence;
      refresh();
    },
  };
}
