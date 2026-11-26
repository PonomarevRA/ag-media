import {uiText} from "./i18n.js";
import {mountEditorWindow} from "./editor-object-window.js";
import {effectiveSpeaker} from "./speaker-label.js";

/** Resolve multiple visible marks into explicit choices, with a pinned annotated frame. */
export function mountFrameInspection(find, video, {paint, openObject, openProfile}) {
  const prefix = "media-frame-inspection";
  const panel = find(prefix);
  const heading = find(prefix + "-title");
  const time = find(prefix + "-time");
  const canvas = find(prefix + "-image");
  const items = find(prefix + "-items");
  const detail = mountEditorWindow(find, {prefix, panel, heading, onClose: () => close(true)});
  let opener;
  let pending;

  /** Release the sampled image and return to the originating frame row. */
  function close(restoreFocus = false) {
    detail.clear();
    pending = undefined;
    canvas.width = canvas.height = 1;
    if (restoreFocus && opener?.isConnected) opener.focus({preventScroll: true});
    opener = undefined;
  }

  /** Keep playback paused on the source timestamp whose technical marks are shown. */
  function capture() {
    if (!pending || video.seeking || video.readyState < 2 || !video.videoWidth) return;
    if (Math.abs(video.currentTime * 1000 - pending.timeMs) >= 30) return;
    const scale = Math.min(1, 960 / Math.max(video.videoWidth, video.videoHeight));
    canvas.width = Math.round(video.videoWidth * scale);
    canvas.height = Math.round(video.videoHeight * scale);
    const context = canvas.getContext("2d");
    context.drawImage(video, 0, 0, canvas.width, canvas.height);
    paint(context, canvas.width, canvas.height, pending.timeMs);
    canvas.hidden = false;
    pending = undefined;
  }

  /** Display one measured entity directly, or open the frame window with all choices. */
  function open(kind, label, frame, trigger) {
    const choices = frameChoices(kind, frame, openObject, openProfile);
    if (choices.length === 1 && choices[0].action && kind !== "speech") {
      choices[0].action(trigger);
      return;
    }
    opener = trigger;
    heading.textContent = label;
    detail.setLabel(label);
    const seconds = frame.timeMs / 1000;
    const clock = `${Math.floor(seconds / 60)}:${(seconds % 60).toFixed(2).padStart(5, "0")}`;
    time.textContent = uiText`Кадр ${clock}`;
    const rows = choices.map(({text, action}) => {
      const node = document.createElement(action ? "button" : "p");
      node.textContent = text;
      if (action) {
        node.type = "button";
        node.onclick = () => action(node);
      }
      return node;
    });
    items.replaceChildren(...rows);
    canvas.hidden = true;
    pending = frame;
    video.pause();
    if (Math.abs(video.currentTime - seconds) >= .008) video.currentTime = seconds;
    capture();
    detail.show();
  }

  for (const event of ["seeked", "loadeddata"]) video.addEventListener(event, capture);
  return {open, reset: close};
}

/** Dispatch the exact marks already drawn in the monitor; text remains plain DOM text. */
export function frameChoices(kind, frame, openObject, openProfile) {
  if (kind === "object" || kind === "interface") {
    return ((kind === "object" ? frame.objects : frame.panels) ?? []).map(row => ({
      text: uiText(row.panelType ?? row.label), action: trigger => openObject(row, kind, frame, trigger),
    }));
  }
  if (kind === "face" || kind === "body" || kind === "body-count") {
    return characterChoices(kind, frame, openProfile);
  }
  if (kind === "speech" && frame.phrase) {
    const speaker = effectiveSpeaker(frame.phrase);
    return [{text: frame.phrase.text}, ...(speaker ? [{
      text: frame.speechLabel, action: trigger => openProfile(speaker, trigger),
    }] : [])];
  }
  if (kind === "shot") return [{text: uiText`План ${frame.shot}`}];
  if (kind === "change") return [{text: uiText`Зон изменений: ${frame.changes?.length ?? 0}`}];
  if (kind === "graphic") return [{text: uiText`Возможных вставок: ${frame.graphics?.length ?? 0}`},
    {text: uiText("Рамки построены по устойчивым изменениям изображения.")}];
  return (frame.text ?? []).filter(row => row.text).map(row => ({text: row.text}));
}

/** Associate figures only with face IDs measured on the exact displayed frame. */
function characterChoices(kind, frame, openProfile) {
  const faces = new Map((frame.faces ?? []).map(hit => [hit[0], hit[6] ?? hit[0]]));
  const choices = [], presented = new Set();
  if (kind === "body-count") choices.push({text: uiText`Оценка числа людей: ${frame.personCount}`});
  const tracks = kind === "body"
    ? [...(frame.marks ?? [])].filter(([, mark]) => mark.box).map(([track]) => track)
    : faces.keys();
  for (const track of tracks) {
    const faceId = faces.get(track);
    if (faceId === undefined) {
      choices.push({text: uiText`Трек ${track}`});
    } else if (!presented.has(faceId)) {
      presented.add(faceId);
      choices.push({text: uiText`Лицо ${faceId}`, action: trigger => openProfile({faceId}, trigger)});
    }
  }
  return choices;
}
