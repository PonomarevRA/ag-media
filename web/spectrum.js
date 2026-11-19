const SIZE = 2048;
const BAND_EDGES = [60, 120, 250, 500, 1000, 2000, 4000, 8000, 12000];
const FLOOR = -72;
const WINDOW = Float64Array.from({length: SIZE}, (_, i) => 0.5 - 0.5 * Math.cos(2 * Math.PI * i / SIZE));
const WINDOW_POWER = WINDOW.reduce((total, value) => total + value * value, 0);

/** Transform a real window in place with a radix-2 FFT. */
function fft(real, imag) {
  for (let i = 1, j = 0; i < SIZE; i++) {
    let bit = SIZE >> 1;
    while (j & bit) {
      j ^= bit;
      bit >>= 1;
    }
    j |= bit;
    if (j > i) [real[i], real[j]] = [real[j], real[i]];
  }
  for (let width = 2; width <= SIZE; width *= 2) {
    const half = width / 2;
    for (let offset = 0; offset < half; offset++) {
      const angle = -2 * Math.PI * offset / width;
      const cosine = Math.cos(angle), sine = Math.sin(angle);
      for (let i = offset; i < SIZE; i += width) {
        const peer = i + half;
        const r = real[peer] * cosine - imag[peer] * sine;
        const im = real[peer] * sine + imag[peer] * cosine;
        real[peer] = real[i] - r;
        imag[peer] = imag[i] - im;
        real[i] += r;
        imag[i] += im;
      }
    }
  }
}

/** Compute band RMS in dBFS over a Hann window centered at the shared cursor. */
export function bandSpectrum(samples, rate, seconds) {
  const levels = new Array(BAND_EDGES.length - 1).fill(FLOOR);
  if (seconds < 0 || seconds >= samples.length / rate) return levels;
  const start = Math.round(seconds * rate) - SIZE / 2;
  const real = Float64Array.from(WINDOW, (weight, i) => weight * (samples[start + i] ?? 0));
  const imag = new Float64Array(SIZE);
  fft(real, imag);
  for (let band = 0; band < levels.length; band++) {
    const end = Math.min(SIZE / 2, Math.ceil(BAND_EDGES[band + 1] * SIZE / rate));
    let power = 0;
    for (let bin = Math.ceil(BAND_EDGES[band] * SIZE / rate); bin < end; bin++) {
      power += real[bin] ** 2 + imag[bin] ** 2;
    }
    const db = 10 * Math.log10(2 * power / (SIZE * WINDOW_POWER));
    levels[band] = Math.max(FLOOR, Math.min(0, db));
  }
  return levels;
}

/** Draw a spectrum against the same fixed scale for every track. */
export function drawSpectrum(canvas, levels, palette, processed) {
  const ctx = canvas.getContext("2d");
  const {width, height} = canvas;
  const left = 29, top = 9, bottom = height - 22;
  const step = (width - left - 8) / levels.length;
  ctx.clearRect(0, 0, width, height);
  ctx.font = "12px monospace";
  ctx.textAlign = "right";
  for (const db of [0, -24, -48, -72]) {
    const y = top + (bottom - top) * db / FLOOR;
    ctx.fillStyle = palette.ink;
    ctx.fillText(String(db), left - 5, y + 4);
    ctx.fillStyle = palette.grid;
    ctx.fillRect(left, y, width - left - 8, 1);
  }
  ctx.textAlign = "center";
  levels.forEach((db, i) => {
    const x = left + i * step;
    const label = BAND_EDGES[i];
    ctx.fillStyle = palette.ink;
    ctx.fillText(label >= 1000 ? label / 1000 + "k" : String(label), x + step / 2, height - 6);
    ctx.fillStyle = processed ? palette.processed : palette.ink;
    const bars = Math.floor((bottom - top) * (1 - db / FLOOR) / 4);
    for (let bar = 0; bar < bars; bar++) {
      ctx.fillRect(x + 5, bottom - 4 * (bar + 1), Math.max(1, step - 10), 2);
    }
  });
}

const eqCenters = [220, 1000, 3800];
const eqRate = 24000;
const eqWidthOctaves = 2;
const tau = 2 * Math.PI;

/** Predict the small-signal gain of the three peaking filters and output gain, in dB. */
export function eqResponse(hz, gains, outputGain) {
  const omega = tau * hz / eqRate;
  const sin = Math.sin(omega), cos = Math.cos(omega);
  let gain = outputGain;
  for (const [index, center] of eqCenters.entries()) {
    if (!gains[index]) continue;
    const angle = tau * center / eqRate;
    const sinCenter = Math.sin(angle);
    // RBJ's peaking biquad with the same two-octave width used by the WAV renderer.
    const alpha = sinCenter * Math.sinh(Math.LN2 * eqWidthOctaves * angle / (2 * sinCenter));
    const amplitude = 10 ** (gains[index] / 40);
    const real = (cos - Math.cos(angle)) ** 2;
    const imaginary = (alpha * sin) ** 2;
    gain += 10 * Math.log10((real + imaginary * amplitude ** 2) / (real + imaginary / amplitude ** 2));
  }
  return gain;
}

/** Draw the predicted EQ response on a logarithmic frequency axis. */
export function drawEqResponse(canvas, gains, outputGain) {
  const width = canvas.clientWidth, height = canvas.clientHeight;
  if (width < 90 || height < 64) return;
  const ratio = window.devicePixelRatio || 1;
  canvas.width = Math.round(width * ratio);
  canvas.height = Math.round(height * ratio);
  const ctx = canvas.getContext("2d");
  ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
  const styles = getComputedStyle(canvas);
  const ink = styles.color;
  const line = styles.getPropertyValue("--eq-plot-line");
  const accent = styles.getPropertyValue("--eq-plot-accent");
  const fill = styles.getPropertyValue("--eq-plot-fill");
  const peaks = Math.max(12, Math.abs(outputGain) + gains.reduce((sum, gain) => sum + Math.abs(gain), 0));
  const range = Math.ceil(peaks / 6) * 6;
  const left = 36, right = width - 14, top = 14, bottom = height - 24;
  const first = Math.log(40), span = Math.log(10000) - first;
  const x = hz => left + (Math.log(hz) - first) / span * (right - left);
  const y = db => top + (range - db) / (2 * range) * (bottom - top);
  let frequencyTicks = [40, 10000];
  if (width >= 300) frequencyTicks = [40, 100, 220, 1000, 3800, 10000];
  else if (width >= 150) frequencyTicks = [40, 220, 1000, 10000];
  ctx.font = Math.min(14, Number.parseFloat(styles.fontSize)) + "px ui-monospace, monospace";
  ctx.fillStyle = ink;
  ctx.textAlign = "right";
  for (const db of [range, 0, -range]) {
    const row = y(db);
    ctx.fillText(db > 0 ? "+" + db : String(db), left - 5, row + 3);
    ctx.strokeStyle = db === 0 ? ink : line;
    ctx.beginPath();
    ctx.moveTo(left, row);
    ctx.lineTo(right, row);
    ctx.stroke();
  }
  for (const hz of frequencyTicks) {
    const column = x(hz);
    ctx.textAlign = hz === 40 ? "left" : hz === 10000 ? "right" : "center";
    ctx.fillText(hz >= 1000 ? hz / 1000 + "k" : String(hz), column, height - 6);
    ctx.strokeStyle = line;
    ctx.beginPath();
    ctx.moveTo(column, top);
    ctx.lineTo(column, bottom);
    ctx.stroke();
  }
  const curve = new Path2D();
  const area = new Path2D();
  area.moveTo(left, y(0));
  for (let point = 0; point <= 120; point++) {
    const portion = point / 120;
    const column = left + portion * (right - left);
    const row = y(eqResponse(Math.exp(first + portion * span), gains, outputGain));
    if (point === 0) curve.moveTo(column, row);
    else curve.lineTo(column, row);
    area.lineTo(column, row);
  }
  area.lineTo(right, y(0));
  area.closePath();
  ctx.fillStyle = fill;
  ctx.fill(area);
  ctx.strokeStyle = accent;
  ctx.lineWidth = 2;
  ctx.stroke(curve);
}
