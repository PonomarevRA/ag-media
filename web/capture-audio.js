import {uiText} from "./i18n.js";
/** Encode an independent mono WAV chunk for the local recognizer. */
function asWav(samples, rate) {
  const view = new DataView(new ArrayBuffer(44 + samples.length * 2));
  const stamp = (offset, value) => {
    for (let index = 0; index < value.length; index++) view.setUint8(offset + index, value.charCodeAt(index));
  };
  stamp(0, "RIFF");
  view.setUint32(4, view.byteLength - 8, true);
  stamp(8, "WAVEfmt ");
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, 1, true);
  view.setUint32(24, rate, true);
  view.setUint32(28, rate * 2, true);
  view.setUint16(32, 2, true);
  view.setUint16(34, 16, true);
  stamp(36, "data");
  view.setUint32(40, samples.length * 2, true);
  for (let index = 0; index < samples.length; index++) {
    view.setInt16(44 + index * 2, Math.round(Math.max(-1, Math.min(1, samples[index])) * 32767), true);
  }
  return new Blob([view], {type: "audio/wav"});
}

/** Capture an explicitly chosen screen or audio input and emit independent timestamped chunks. */
export async function captureAudio(kind, device, onChunk, onStop, {denoise = false, monitor = null, signal} = {}) {
  const devices = navigator.mediaDevices;
  const stream = kind === "screen"
    ? await devices.getDisplayMedia({video: true, audio: true, systemAudio: "include"})
    : await devices.getUserMedia({audio: {
      ...(device ? {deviceId: {exact: device}} : {}),
      echoCancellation: false, noiseSuppression: denoise, autoGainControl: false
    }});
  let context;
  let monitorAudio;
  const tracks = stream.getTracks();
  /** Stop the monitor track and detach its audio element if it was opened. */
  function releaseMonitor() {
    if (!monitorAudio) return;
    monitorAudio.pause();
    for (const track of monitorAudio.srcObject?.getTracks() || []) track.stop();
    monitorAudio.srcObject = null;
    monitorAudio = null;
  }
  /** Release permissions and playback even if an asynchronous device setup is still pending. */
  function releaseInput() {
    for (const track of tracks) if (track.readyState !== "ended") track.stop();
    releaseMonitor();
    if (context && context.state !== "closed") context.close().catch(() => {});
  }
  signal?.addEventListener("abort", releaseInput, {once: true});
  try {
    signal?.throwIfAborted();
    const audio = stream.getAudioTracks();
    if (!audio.length) throw new Error(uiText("В выбранном источнике нет звука. Включи передачу звука в окне выбора."));
    context = onChunk ? new AudioContext({sampleRate: 16000}) : new AudioContext();
    if (onChunk) await context.audioWorklet.addModule("/capture-worklet.js");
    signal?.throwIfAborted();
    await context.resume();
    signal?.throwIfAborted();
    const node = onChunk ? new AudioWorkletNode(context, "voice-capture") : null;
    const source = context.createMediaStreamSource(new MediaStream(audio));
    let frames = 0;
    let stopping = false;
    let finished = false;
    let timeout;
    /** Release both media permissions and the audio graph after delivering the final chunk. */
    function cleanup() {
      clearTimeout(timeout);
      if (finished) return;
      finished = true;
      signal?.removeEventListener("abort", stop);
      source.disconnect();
      node?.disconnect();
      releaseMonitor();
      context.close().catch(() => {});
      onStop();
    }
    /** Stop the source immediately and ask the worklet for buffered samples. */
    function stop() {
      if (stopping) return;
      stopping = true;
      releaseMonitor();
      for (const track of tracks) track.stop();
      if (!node) {
        cleanup();
        return;
      }
      node.port.postMessage("flush");
      timeout = setTimeout(cleanup, 1500);
    }
    if (node) {
      node.port.onmessage = ({data}) => {
        if (data.done) {
          cleanup();
          return;
        }
        if (finished) return;
        const startMs = Math.round(frames * 1000 / context.sampleRate);
        frames += data.audio.length;
        const endMs = Math.round(frames * 1000 / context.sampleRate);
        onChunk(asWav(data.audio, context.sampleRate), startMs, endMs - startMs);
      };
    }
    if (monitor) {
      if (audio.some(track => track.label && track.label === monitor.label)) {
        throw new Error(uiText("Выбери другой аудиовыход, чтобы звук не вернулся на вход."));
      }
      const output = context.createMediaStreamDestination();
      source.connect(output);
      monitorAudio = new Audio();
      monitorAudio.srcObject = output.stream;
      await monitorAudio.setSinkId(monitor.deviceId);
      signal?.throwIfAborted();
      await monitorAudio.play();
    }
    signal?.throwIfAborted();
    if (audio.every(track => track.readyState === "ended")) {
      throw new Error(uiText("Источник звука закрыт."));
    }
    if (node) {
      source.connect(node);
      node.connect(context.destination); // The worklet's output is silent.
    }
    signal?.removeEventListener("abort", releaseInput);
    signal?.addEventListener("abort", stop, {once: true});
    for (const track of tracks) track.addEventListener("ended", stop, {once: true});
    return {stop, noiseSuppression: audio.every(track => track.getSettings().noiseSuppression === true)};
  } catch (error) {
    signal?.removeEventListener("abort", releaseInput);
    releaseInput();
    throw error;
  }
}

/** Join consecutive WAV chunks from one AudioContext without copying the PCM. */
export async function joinCapturedWav(chunks) {
  const header = new DataView(await chunks[0].slice(0, 44).arrayBuffer());
  const pcm = chunks.map(chunk => chunk.slice(44));
  const bytes = pcm.reduce((size, part) => size + part.size, 0);
  header.setUint32(4, bytes + 36, true);
  header.setUint32(40, bytes, true);
  return new Blob([header, ...pcm], {type: "audio/wav"});
}
