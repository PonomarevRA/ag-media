import {uiText} from "./i18n.js";
import {mountPlayer} from "./player.js";

/** Let the user audition and download a selected original and its removed signal. */
export function mountMusicSample(find, video, request) {
  const panel = find("media-music");
  const button = find("media-music-prepare");
  const result = find("media-music-result");
  const status = find("media-music-status");
  const loop = find("media-music-loop");
  const tracks = [["original", uiText("Исходный фрагмент")], ["removed", uiText("Убранный сигнал")]].map(([id, name]) => {
    const audio = find("media-music-" + id);
    return {
      id, audio, player: mountPlayer(audio, name),
      download: find("media-music-" + id + "-download"),
    };
  });
  let source;
  let controller;

  for (const track of tracks) {
    track.audio.addEventListener("play", () => {
      video.pause();
      for (const other of tracks) if (other !== track) other.audio.pause();
    });
  }
  video.addEventListener("play", () => { for (const track of tracks) track.audio.pause(); });
  loop.addEventListener("change", () => { for (const track of tracks) track.audio.loop = loop.checked; });

  function clearTracks() {
    for (const track of tracks) {
      if (track.url) URL.revokeObjectURL(track.url);
      track.url = null;
      track.player.setSource(null);
      track.download.removeAttribute("href");
    }
    result.hidden = true;
  }

  button.addEventListener("click", async () => {
    if (!source || controller) return;
    const pending = new AbortController();
    controller = pending;
    const start_ms = Math.round((Number.isFinite(video.currentTime) ? video.currentTime : 0) * 1000);
    video.pause();
    button.disabled = true;
    clearTracks();
    status.textContent = uiText("Готовим две дорожки…");
    try {
      const data = await request("music-sample", source, {start_ms, signal: pending.signal});
      if (pending.signal.aborted) return;
      for (const track of tracks) {
        const wav = Uint8Array.from(atob(data[track.id]), char => char.charCodeAt(0));
        track.url = URL.createObjectURL(new Blob([wav], {type: "audio/wav"}));
        track.player.setSource(track.url);
        track.download.href = track.url;
      }
      const method = data.method === "rnnoise" ? "RNNoise" : "FFT";
      status.textContent = uiText`Отметка ${(data.startMs / 1000).toFixed(1)} с · фрагмент ${(data.durationMs / 1000).toFixed(1)} с · ${method}`;
      result.hidden = false;
    } catch (error) {
      if (!pending.signal.aborted) status.textContent = error.message;
    } finally {
      if (controller === pending) {
        controller = null;
        button.disabled = false;
      }
    }
  });

  return {
    setSource(next) {
      controller?.abort();
      controller = null;
      source = next;
      button.disabled = false;
      clearTracks();
      status.textContent = "";
      panel.hidden = !source;
      panel.open = false;
    },
  };
}
