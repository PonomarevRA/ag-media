import {drawEqResponse} from "./spectrum.js";
import {apiText, uiText} from "./i18n.js";
/** Bind deck tracks, EQ controls and a downloadable processed copy. */
export class ProcessingPanel {
  /** Wire controls once and resolve track choices as audio becomes available. */
  constructor(deck, getLanguage) {
    this.deck = deck;
    this.getLanguage = getLanguage;
    this.panel = document.querySelector("#processing");
    this.source = this.panel.querySelector("#eq-source");
    this.reference = this.panel.querySelector("#eq-reference");
    this.controls = [...this.panel.querySelectorAll('input[type="range"]')];
    this.curve = this.panel.querySelector("#eq-response");
    this.curveObserver = new ResizeObserver(() => this.drawResponse());
    this.curveObserver.observe(this.curve);
    document.addEventListener("studio-design-change", () => this.drawResponse());
    this.denoise = this.panel.querySelector("#eq-denoise");
    this.apply = this.panel.querySelector("#eq-apply");
    this.auto = this.panel.querySelector("#eq-auto");
    this.status = this.panel.querySelector("#eq-status");
    this.result = this.panel.querySelector("#eq-result");
    this.download = this.panel.querySelector("#eq-download");
    this.pending = false;
    this.revision = 0;
    this.controls.forEach(control => {
      control.oninput = () => this.updateLevels();
    });
    this.source.onchange = () => {
      this.sourceChoice = this.source.value;
      this.dropResult();
      this.syncTracks();
    };
    this.reference.onchange = () => {
      this.referenceChoice = this.reference.value;
      if (this.appliedReference && this.appliedReference !== this.referenceChoice) this.dropResult();
      this.updateButtons();
    };
    this.apply.onclick = () => this.render(false);
    this.auto.onclick = () => this.render(true);
    this.panel.querySelector("#eq-remove").onclick = () => this.dropResult();
    this.updateLevels();
  }

  /** Keep selections stable as tracks finish decoding or are removed. */
  syncTracks() {
    const tracks = [...this.deck.rows.values()].filter(row => row.samples && row.id !== "processed");
    const oldSource = this.source.value, oldReference = this.reference.value;
    this.sourceChoice = tracks.some(row => row.id === this.sourceChoice) ? this.sourceChoice : "";
    const retainSource = (this.outputUrl || this.pending) && tracks.some(row => row.id === oldSource);
    const selected = this.sourceChoice || (retainSource ? oldSource : tracks.at(-1)?.id);
    this.fill(this.source, tracks, selected, uiText("Выбери дорожку"));
    const references = tracks.filter(row => row.id !== this.source.value);
    this.referenceChoice = references.some(row => row.id === this.referenceChoice) ? this.referenceChoice : "";
    this.fill(this.reference, references, this.referenceChoice || references[0]?.id, uiText("Выбери исходник"));
    this.panel.hidden = !tracks.length;
    if (this.outputUrl && (!this.deck.rows.has("processed") || oldSource !== this.source.value ||
      (this.appliedReference && oldReference !== this.reference.value))) this.dropResult();
    this.updateButtons();
  }

  /** Replace a select's choices without losing a track that still exists. */
  fill(select, tracks, selected, placeholder) {
    const options = tracks.map(row => new Option(row.name, row.id));
    const next = [new Option(placeholder, ""), ...options];
    const existing = [...select.options];
    if (next.length !== existing.length ||
      next.some((option, index) => option.value !== existing[index].value || option.text !== existing[index].text)) {
      select.replaceChildren(...next);
    }
    select.value = tracks.some(row => row.id === selected) ? selected : "";
  }

  /** Display slider values beside the names of the EQ bands. */
  updateLevels() {
    for (const control of this.controls) {
      control.labels[0].querySelector("output").value = uiText`${Number(control.value) > 0 ? "+" : ""}${control.value} дБ`;
    }
    this.drawResponse();
  }

  /** Plot the current settings on the visible modern EQ surface. */
  drawResponse() {
    if (!this.curve.clientWidth) return;
    const settings = this.controls.map(control => Number(control.value));
    drawEqResponse(this.curve, settings.slice(0, 3), settings[3]);
  }

  /** Keep actions available only when their input tracks are ready. */
  updateButtons() {
    this.apply.disabled = this.pending || !this.source.value;
    this.auto.disabled = this.apply.disabled || !this.reference.value || this.reference.value === this.source.value;
    this.source.disabled = this.pending;
    this.reference.disabled = this.pending || this.reference.options.length < 2;
    this.denoise.disabled = this.pending;
    for (const control of this.controls) control.disabled = this.pending;
  }

  /** Remove the temporary track and release its local audio URL. */
  dropResult() {
    this.revision++;
    this.status.textContent = "";
    this.result.hidden = true;
    if (!this.outputUrl) return;
    URL.revokeObjectURL(this.outputUrl);
    this.outputUrl = null;
    this.download.removeAttribute("href");
    this.deck.select("processed", "", false);
  }

  /** Process the selected WAV locally and add the copy to A/B playback. */
  async render(auto) {
    this.dropResult();
    this.pending = true;
    this.updateButtons();
    const revision = this.revision, language = this.getLanguage();
    const id = this.source.value, reference = this.reference.value;
    this.status.textContent = uiText("Обрабатываю дорожку…");
    try {
      const values = this.controls.map(control => Number(control.value));
      const body = {id, reference, auto, denoise: this.denoise.checked, gains: values.slice(0, 3), gain: values[3]};
      const response = await fetch(`/api/process?lang=${language}`, {
        method: "POST", headers: {"Content-Type": "application/json", "X-Voice-Studio": "1"},
        body: JSON.stringify(body)
      });
      if (!response.ok) throw new Error(apiText((await response.json()).error));
      const applied = JSON.parse(response.headers.get("X-Voice-Processing"));
      const audio = await response.blob();
      if (revision !== this.revision || language !== this.getLanguage() || id !== this.source.value || !this.deck.rows.has(id) ||
        (auto && reference !== this.reference.value)) {
        this.status.textContent = "";
        return;
      }
      this.controls.forEach((control, index) => {
        control.value = [...applied.gains, applied.gain][index];
      });
      this.updateLevels();
      this.appliedReference = auto ? reference : "";
      this.outputUrl = URL.createObjectURL(audio);
      this.download.href = this.outputUrl;
      this.result.hidden = false;
      const cleaner = {rnnoise: " • RNNoise", fft: uiText(" • FFT-шумоподавление"), none: ""}[applied.denoise];
      this.status.textContent = uiText("Копия готова") + cleaner;
      await this.deck.select("processed", uiText("После обработки"), true, {url: this.outputUrl, audio});
    } catch (error) {
      if (revision === this.revision) this.status.textContent = error.message || uiText("Обработка не удалась.");
    } finally {
      this.pending = false;
      this.updateButtons();
    }
  }
}
