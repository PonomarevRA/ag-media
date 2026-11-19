import {createPlayerSurface} from "./studio-design.js";
import {attachStudioHint} from "./studio-icons.js";
import {uiText} from "./i18n.js";
/** Format media time as a compact clock without implying unknown duration is zero. */
function clock(seconds) {
  const total = Math.max(0, Math.floor(seconds));
  return `${String(Math.floor(total / 60)).padStart(2, "0")}:${String(total % 60).padStart(2, "0")}`;
}

/** Enhance a native audio element with the studio's transport controls. */
export function mountPlayer(audio, name, initialDuration = 0) {
  const template = document.querySelector("#audio-player-template");
  const root = createPlayerSurface(template);
  const play = root.querySelector(".player-play");
  const mute = root.querySelector(".player-mute");
  attachStudioHint(play);
  attachStudioHint(mute);
  const volume = root.querySelector(".player-volume");
  const seek = root.querySelector(".player-seek");
  const progress = root.querySelector(".player-progress");
  const elapsed = root.querySelector(".player-elapsed");
  const length = root.querySelector(".player-length");
  const status = root.querySelector(".player-status");
  root.setAttribute("aria-label", name);
  if (audio.parentNode) audio.before(root);
  root.append(audio);
  audio.controls = false;
  audio.hidden = true;

  /** Mirror browser events and the persisted length without eager media loading. */
  function update() {
    const hasSource = audio.hasAttribute("src");
    const knownDuration = audio.readyState ? audio.duration : initialDuration;
    const duration = Number.isFinite(knownDuration) && knownDuration > 0 ? knownDuration : 0;
    const position = Number.isFinite(audio.currentTime) ? audio.currentTime : 0;
    const playing = !audio.paused && !audio.ended;
    const silent = audio.muted || audio.volume === 0;
    play.disabled = !hasSource || Boolean(audio.error);
    play.dataset.playing = String(playing);
    const playLabel = playing ? uiText("Пауза") : uiText("Воспроизвести");
    play.setAttribute("aria-label", playLabel);
    play.dataset.hint = playLabel;
    mute.disabled = volume.disabled = !hasSource;
    mute.setAttribute("aria-pressed", String(silent));
    volume.value = String(audio.volume);
    elapsed.textContent = clock(position);
    length.textContent = duration > 0 ? clock(duration) : "—:—";
    seek.disabled = play.disabled || duration <= 0;
    seek.max = String(duration || 1);
    seek.value = String(Math.min(position, duration));
    seek.setAttribute("aria-valuetext", duration ? uiText`${clock(position)} из ${clock(duration)}` : uiText("Длительность неизвестна"));
    progress.value = duration ? Math.min(1, position / duration) : 0;
    if (audio.error) status.textContent = uiText("Аудио недоступно.");
  }

  play.addEventListener("click", async () => {
    status.textContent = "";
    if (!audio.paused && !audio.ended) {
      audio.pause();
      return;
    }
    try {
      await audio.play();
    } catch (error) {
      if (error.name !== "AbortError" && !audio.error) status.textContent = uiText("Не удалось начать воспроизведение.");
    }
    update();
  });
  seek.addEventListener("input", () => {
    audio.currentTime = Number(seek.value);
    update();
  });
  mute.addEventListener("click", () => {
    const wasSilent = audio.muted || audio.volume === 0;
    if (audio.volume === 0) audio.volume = 0.5;
    audio.muted = !wasSilent;
    update();
  });
  volume.addEventListener("input", () => {
    audio.volume = Number(volume.value);
    audio.muted = false;
    update();
  });
  for (const event of ["loadedmetadata", "durationchange", "timeupdate", "seeking", "seeked", "play", "pause", "ended", "volumechange", "error"]) {
    audio.addEventListener(event, update);
  }
  update();

  return {
    root,
    /** Set or clear a recorder preview while retaining the existing audio element. */
    setSource(source) {
      status.textContent = "";
      root.hidden = !source;
      if (source) audio.src = source;
      else audio.removeAttribute("src");
      audio.load();
      update();
    }
  };
}
