import {studioSignalPalette} from "./studio-design.js";
import {uiText} from "./i18n.js";
import {bandSpectrum, drawSpectrum} from "./spectrum.js";

/** Draw the recorder meter from microphone samples, and release it on stop. */
export function monitorTape(stream) {
  const tape = document.querySelector("#tape");
  const clock = document.querySelector("#tape-time");
  const canvas = document.querySelector("#meter");
  const ctx = canvas.getContext("2d");
  const started = performance.now();
  tape.dataset.mode = "recording";
  document.querySelector("#tape-mode").textContent = uiText("ЗАПИСЬ");
  let audio, analyser, samples, frame;
  try {
    audio = new AudioContext();
    analyser = audio.createAnalyser();
    analyser.fftSize = 256;
    samples = new Uint8Array(analyser.frequencyBinCount);
    audio.createMediaStreamSource(stream).connect(analyser);
    audio.resume().catch(() => {});
  } catch {
    // The recorder still works when audio visualization is unavailable.
  }
  function draw(now) {
    const seconds = Math.floor((now - started) / 1000);
    clock.textContent = "00:" + String(seconds).padStart(2, "0");
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    if (samples) analyser.getByteFrequencyData(samples);
    const bars = 28;
    const step = canvas.width / bars;
    ctx.fillStyle = studioSignalPalette(document).ink;
    for (let i = 0; i < bars; i++) {
      const level = samples ? samples[i * 3] / 255 : 0;
      const height = 2 + Math.round(level * 11) * 4;
      for (let y = canvas.height - 3; y > canvas.height - height; y -= 4) {
        ctx.fillRect(i * step, y, step - 2, 2);
      }
    }
    frame = requestAnimationFrame(draw);
  }
  frame = requestAnimationFrame(draw);
  return () => {
    cancelAnimationFrame(frame);
    audio?.close().catch(() => {});
    tape.dataset.mode = "idle";
    document.querySelector("#tape-mode").textContent = uiText("ГОТОВ");
  };
}

/** Compare real decoded peaks at one time scale with A/B playback and one seek position. */
export class TrackDeck {
  /** Bind the shared transport to the comparison panel. */
  constructor(onTracksChange) {
    this.onTracksChange = onTracksChange;
    this.list = document.querySelector("#track-list");
    this.playlist = this.list.closest(".deck-playlist");
    this.empty = document.querySelector("#track-empty");
    this.spectrumObserver = new ResizeObserver(entries => {
      const palette = studioSignalPalette(document);
      for (const {target, contentRect} of entries) {
        const row = this.rows.get(target.dataset.track);
        if (!row?.samples || !contentRect.width) continue;
        const width = Math.round(contentRect.width);
        if (target.width === width) continue;
        target.width = width;
        this.drawTrackSpectrum(row, palette);
      }
    });
    this.transport = document.querySelector("#transport");
    this.seek = document.querySelector("#track-seek");
    this.ruler = document.querySelector("#track-ruler");
    this.spectrumToggle = document.querySelector("#track-spectrum");
    this.spectrumToggle.onchange = () => this.draw();
    this.clock = document.querySelector("#track-time");
    this.status = document.querySelector("#track-status");
    this.audio = new Audio();
    this.rows = new Map();
    this.position = 0;
    this.duration = 0;
    this.seek.oninput = () => {
      this.seekTo(Number(this.seek.value));
    };
    this.audio.ontimeupdate = () => {
      if (this.active && !this.audio.paused) {
        this.position = this.audio.currentTime;
        this.draw();
      }
    };
    this.audio.onended = () => {
      if (this.active) this.position = this.audio.currentTime;
      this.draw();
      this.updateButtons();
    };
    this.audio.onpause = () => this.updateButtons();
    this.audio.onplay = () => this.updateButtons();
  }

  /** Toggle a track and decode its peaks and spectrum at a shared sample rate. */
  async select(id, name, enabled, preview) {
    if (!enabled) {
      const row = this.rows.get(id);
      if (this.active === row) this.stop();
      if (row) {
        this.spectrumObserver.unobserve(row.spectrum);
        row.node.remove();
      }
      this.rows.delete(id);
      this.draw();
      this.onTracksChange?.();
      return;
    }
    if (this.rows.has(id)) return;
    const node = document.createElement("li");
    node.className = "track";
    const button = document.createElement("button");
    button.type = "button";
    button.textContent = name;
    button.disabled = true;
    button.onclick = () => this.play(row);
    const canvas = document.createElement("canvas");
    canvas.width = 480;
    canvas.height = 42;
    canvas.className = "track-waveform";
    canvas.setAttribute("aria-hidden", "true");
    canvas.onclick = event => {
      if (!row.peaks || !canvas.clientWidth) return;
      const fraction = event.offsetX / canvas.clientWidth;
      this.seekTo(fraction * this.duration);
    };
    const info = document.createElement("span");
    info.textContent = uiText("Читаю аудио…");
    info.className = "track-duration";
    const visuals = document.createElement("div");
    visuals.className = "track-visuals";
    const spectrum = document.createElement("canvas");
    spectrum.className = "spectrum";
    spectrum.width = 480;
    spectrum.dataset.track = id;
    spectrum.height = 103;
    spectrum.setAttribute("role", "img");
    spectrum.setAttribute("aria-label", uiText("Частотный спектр в дБFS"));
    visuals.append(canvas, spectrum);
    const title = document.createElement("div");
    title.className = "track-titlebar";
    title.dataset.agWidth = "content";
    title.dataset.agWidthMinEm = "7";
    title.append(button, info);
    node.append(title, visuals);
    this.list.append(node);
    const row = {id, name, node, button, canvas, spectrum, info, url: preview?.url || `/media/${id}.wav`};
    this.rows.set(id, row);
    this.spectrumObserver.observe(spectrum);
    this.draw();
    let context;
    try {
      const audio = preview?.audio || await fetch(row.url);
      if (audio instanceof Response && !audio.ok) throw new Error("Audio unavailable");
      context = new AudioContext({sampleRate: 24000});
      const buffer = await context.decodeAudioData(await audio.arrayBuffer());
      const channel = buffer.getChannelData(0);
      const bins = Math.min(480, channel.length);
      const peaks = new Float32Array(bins);
      for (let i = 0; i < channel.length; i++) {
        const bin = Math.min(bins - 1, Math.floor(i * bins / channel.length));
        peaks[bin] = Math.max(peaks[bin], Math.abs(channel[i]));
      }
      if (this.rows.get(id) !== row) return;
      Object.assign(row, {duration: buffer.duration, peaks, samples: channel, rate: buffer.sampleRate});
      if (spectrum.clientWidth) spectrum.width = spectrum.clientWidth;
      info.textContent = buffer.duration.toFixed(1) + uiText(" с");
      button.disabled = false;
      this.updateButtons();
      this.draw();
      this.onTracksChange?.();
    } catch {
      info.textContent = uiText("Аудио недоступно");
    } finally {
      await context?.close();
    }
  }

  /** Move the shared cursor without changing the selected track or starting playback. */
  seekTo(seconds) {
    this.status.textContent = "";
    this.position = Math.min(Math.max(0, seconds), this.duration);
    if (this.active) this.audio.currentTime = Math.min(this.position, this.active.duration);
    this.draw();
  }

  /** Mark evenly spaced seconds when the common duration changes. */
  drawRuler() {
    if (this.ruler.dataset.duration === String(this.duration)) return;
    this.ruler.dataset.duration = String(this.duration);
    this.ruler.replaceChildren();
    if (!this.duration) return;
    for (let division = 0; division <= 4; division++) {
      const tick = document.createElement("span");
      tick.textContent = uiText`${(this.duration * division / 4).toFixed(1)} с`;
      this.ruler.append(tick);
    }
  }

  /** Switch to another take at the current time; pressing it again pauses. */
  async play(row) {
    this.status.textContent = "";
    const previous = this.active;
    if (previous && !this.audio.paused) this.position = this.audio.currentTime;
    if (previous === row && !this.audio.paused) {
      this.audio.pause();
      this.draw();
      return;
    }
    if (previous !== row && this.position >= row.duration && this.position < this.duration) {
      this.status.textContent = uiText`«${row.name}» заканчивается на ${row.duration.toFixed(1)} с. Передвинь курсор левее.`;
      this.draw();
      return;
    }
    this.audio.pause();
    this.active = row;
    if (previous !== row) this.audio.src = row.url;
    if (this.position >= row.duration) this.position = 0;
    this.audio.currentTime = this.position;
    this.draw();
    try { await this.audio.play(); }
    catch { this.updateButtons(); return; }
    this.updateButtons();
  }

  /** Stop playback and forget the selected track. */
  stop() {
    this.audio.pause();
    this.audio.removeAttribute("src");
    this.audio.load();
    this.active = null;
    this.updateButtons();
  }

  /** Reset the selection when changing workspaces or languages. */
  clear() {
    this.status.textContent = "";
    this.stop();
    this.spectrumObserver.disconnect();
    this.rows.clear();
    this.list.replaceChildren();
    this.position = 0;
    this.draw();
    this.onTracksChange?.();
  }

  /** Name playback actions explicitly for keyboard and screen reader users. */
  updateButtons() {
    for (const row of this.rows.values()) {
      const playing = this.active === row && !this.audio.paused;
      row.node.classList.toggle("is-playing", playing);
      row.button.textContent = (playing ? "Ⅱ " : "▶ ") + row.name;
      row.button.setAttribute("aria-label", (playing ? uiText("Пауза: ") : uiText("Воспроизвести: ")) + row.name);
    }
  }

  /** Render all envelopes on the same absolute time scale and amplitude range. */
  draw() {
    this.duration = Math.max(0, ...[...this.rows.values()].map(row => row.duration || 0));
    this.transport.hidden = !this.duration;
    this.empty.hidden = this.rows.size > 0;
    this.playlist.hidden = !this.rows.size;
    this.position = Math.min(this.position, this.duration);
    this.seek.max = String(this.duration || 1);
    this.seek.value = String(this.position);
    this.clock.textContent = uiText`${this.position.toFixed(1)} / ${this.duration.toFixed(1)} с`;
    this.drawRuler();
    this.playlist.dataset.showSpectrum = String(this.spectrumToggle.checked);
    const palette = studioSignalPalette(document);
    for (const row of this.rows.values()) {
      if (!row.peaks) continue;
      const {canvas, peaks, duration} = row;
      if (this.spectrumToggle.checked) this.drawTrackSpectrum(row, palette);
      const ctx = canvas.getContext("2d");
      const center = canvas.height / 2;
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.fillStyle = palette.peaks;
      const length = canvas.width * duration / this.duration;
      for (let i = 0; i < peaks.length; i++) {
        const height = Math.max(1, peaks[i] * center);
        ctx.fillRect(i * length / peaks.length, center - height, 1, 2 * height);
      }
      ctx.fillStyle = palette.cursor;
      ctx.fillRect(Math.min(canvas.width - 1, this.position / this.duration * canvas.width), 0, 1, canvas.height);
    }
  }
  /** Redraw the frequency panel when its cursor, width or display profile changes. */
  drawTrackSpectrum(row, palette) {
    const cursor = this.position >= row.duration ? this.position : Math.floor(this.position * 20) / 20;
    if (row.spectrumCursor !== cursor || !row.levels) {
      row.spectrumCursor = cursor;
      row.levels = bandSpectrum(row.samples, row.rate, cursor);
    } else if (row.spectrumWidth === row.spectrum.width && row.palette === palette) {
      return;
    }
    row.spectrumWidth = row.spectrum.width;
    row.palette = palette;
    drawSpectrum(row.spectrum, row.levels, palette, row.id === "processed");
  }
}
