class CaptureProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this.buffer = new Float32Array(sampleRate * 5);
    this.minFrames = Math.round(sampleRate * 2.4);
    this.quietFrames = Math.round(sampleRate * 0.28);
    this.quiet = 0;
    this.used = 0;
    this.stopped = false;
    this.port.onmessage = () => this.stop();
  }

  /** Emit every buffered sample before acknowledging the stop. */
  stop() {
    this.stopped = true;
    this.flush();
    this.port.postMessage({done: true});
  }

  /** Publish independent audio windows, including a final partial window. */
  flush() {
    if (!this.used) return;
    const audio = this.buffer.slice(0, this.used);
    this.port.postMessage({audio}, [audio.buffer]);
    this.used = 0;
    this.quiet = 0;
  }

  /** Prefer a pause after 2.4 seconds and bound a continuous-speech window to 5 seconds. */
  process(inputs) {
    const channels = inputs[0];
    if (this.stopped || !channels?.length) return true;
    let power = 0;
    const frames = channels[0].length;
    for (let frame = 0; frame < frames; frame++) {
      let sample = 0;
      for (const channel of channels) sample += channel[frame];
      sample /= channels.length;
      this.buffer[this.used++] = sample;
      power += sample * sample;
      if (this.used === this.buffer.length) this.flush();
    }
    // This threshold only chooses a boundary. Every sample is still sent to the recognizer.
    this.quiet = power / frames < 0.000036 ? this.quiet + frames : 0;
    if (this.used >= this.minFrames && this.quiet >= this.quietFrames) this.flush();
    return true;
  }
}

registerProcessor("voice-capture", CaptureProcessor);
