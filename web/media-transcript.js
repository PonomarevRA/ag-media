import {mountAnalysisStageSelection, stageSelectionQuery, retainedSpeakerEdits, speakerEditKey} from "./analysis-stage-selection.js";
import {buildSpeakerGaze} from "./speaker-gaze.js";
import {mountSpeakerProfileDialog} from "./speaker-profile-dialog.js";
import {mountTranscriptPanel} from "./transcript-inspector.js";
import {mountTranscriptOcrChecks} from "./transcript-ocr-checks.js";
import {mountTranscriptSrt} from "./transcript-srt.js";
import {mountVideoShortcuts} from "./video-shortcuts.js";
import {apiText, textFor, uiText} from "./i18n.js";
import {mountVideoTimeline} from "./video-timeline.js";
import {effectiveSpeaker, mixedSpeakerLabel} from "./speaker-label.js";
import {mountTranscriptEditor, mountCueEditor} from "./transcript-editor.js";
import {mergeAnalysisPartial, mountMediaAnalysis} from "./video-analysis.js";
import {mountMusicSample} from "./music-sample.js";
import {mountVideoScene} from "./video-scene.js";
import {mountFrameInspection} from "./video-frame-inspection.js";
import {mountFrameDetails} from "./video-frame-details.js";
import {mountVideoOverlay} from "./video-overlay.js";
import {mountVideoDub} from "./video-dub.js";
import {mountVideoEvidence} from "./video-evidence.js";
import {mountClothingExport} from "./person-layer.js";
import {editorFaceBindings, faceProfile, renderSpeakerProfiles, speakerProfiles} from "./speaker-profiles.js";
import {captureAudio, joinCapturedWav} from "./capture-audio.js";
import {mountTranscriptChat, installChatHint} from "./transcript-chat.js";
import {mountVideoStatistics} from "./video-statistics-view.js";
import {mountDialogueAudit} from "./dialogue-audit.js";
import {mountImageAnalysis} from "./image-analysis.js";
import {isImageFile} from "./image-analysis-model.js";

/** Render validated model offsets as inline marks without parsing transcript text as HTML. */
function appendPersonMentions(target, {text, personMentions}) {
  let cursor = 0;
  for (const span of Array.isArray(personMentions) ? personMentions : []) {
    if (span?.type !== "PERSON" || !Number.isInteger(span.start) || !Number.isInteger(span.end)
        || span.start < cursor || span.end <= span.start || span.end > text.length) continue;
    target.append(document.createTextNode(text.slice(cursor, span.start)));
    const mark = document.createElement("mark");
    mark.className = "transcript-person";
    mark.title = uiText("Возможное имя или фамилия");
    mark.textContent = text.slice(span.start, span.end);
    target.append(mark);
    cursor = span.end;
  }
  target.append(document.createTextNode(text.slice(cursor)));
}

/** Download the latest displayed report only when the user activates the export. */
export function mountTranscriptJSONDownload(button, currentReport) {
  button.onclick = () => {
    const report = currentReport();
    if (!report) return;
    // A saved report keeps the measured layout, while temporary images belong to the active session.
    const serialized = JSON.stringify(report, (key, value) =>
      key === "previewUrl" || key === "scrollPreviews" ? undefined : value, 2);
    const file = new Blob([serialized], {type: "application/json"});
    const url = URL.createObjectURL(file);
    const link = document.createElement("a");
    link.href = url;
    link.download = "transcript.json";
    try {
      link.click();
    } finally {
      // Give the browser time to start reading the file before releasing the URL.
      setTimeout(() => URL.revokeObjectURL(url), 30000);
    }
  };
}

/** Restore complete video/scene layers through the same setters for saved reports and live analysis. */
export function applyReportEvidenceLayers(report, current, setters) {
  for (const key of ["videoEvidence", "sceneEvidence"]) {
    const value = report[key] ?? null;
    if (current[key] !== value) {
      current[key] = value;
      setters[key](value);
    }
  }
  return current;
}

/** Connect file and live sources to the shared transcript and subtitle exports. */
export function mountMediaTranscript(request, currentLanguage) {
  const find = id => document.getElementById(id);
  const input = find("media-upload");
  const status = find("media-status");
  const playback = find("media-preview");
  const statistics = mountVideoStatistics(find, playback);
  const clothing = mountClothingExport(find, playback, request);
  const profileDialog = mountSpeakerProfileDialog(find, playback);
  let activeProfiles = [];
  const openProfile = (target, button) => profileDialog.open(typeof target === "object"
    ? faceProfile(target.faceId, displayedEvidence, exportData?.speakerProfileIndex, exportData?.durationMs)
    : activeProfiles.find(profile => profile.number === target), button, displayedEvidence);
  const overlay = mountVideoOverlay(playback, find("media-overlay"), find("media-overlay-options"),
    find("media-overlay-toggle"), mountFrameDetails(find("media-frame-details"),
      (...args) => frameInspection.open(...args)), find("media-change-toggle"));
  const editorTimeline = mountVideoTimeline(find, playback, openProfile, overlay.paintSnapshot);
  const frameInspection = mountFrameInspection(find, playback, {
    paint: overlay.paintSnapshot, openObject: editorTimeline.inspectFrameObject, openProfile,
  });
  mountVideoShortcuts(playback, find("media-panel"), () => displayedEvidence?.timeline?.stepMs ?? 40);
  const scene = mountVideoScene(find, playback, overlay);
  const analysis = mountMediaAnalysis(find, rerunMediaStages);
  const stageSelection = mountAnalysisStageSelection(find, rerunMediaStages);
  const music = mountMusicSample(find, playback, request);
  const dubbing = mountVideoDub(find, request);
  const previewStatus = find("media-preview-status");
  const videoExport = find("media-video-export");
  const videoExportStatus = find("media-video-export-status");
  const result = find("media-result");
  const images = mountImageAnalysis(find, request, {onOpen: reset, setBusy: busy,
    workbench: find("media-panel").querySelector(".editor-workbench")});
  mountTranscriptPanel(result, find("editor-inspector-toggle"));
  const sourceName = find("editor-source-name");
  const analysisOptions = find("media-analysis-options");
  const list = find("media-segments");
  const chat = mountTranscriptChat(list);
  for (const button of [find("media-srt"), find("media-json")]) {
    installChatHint(button);
  }
  const updateSrt = mountTranscriptSrt(find, request);
  const updateOcrChecks = mountTranscriptOcrChecks(find, playback);
  const json = find("media-json");
  const kind = find("capture-kind");
  const device = find("capture-device");
  const recording = find("media-recording");
  const transcribeRecording = find("media-recording-submit");
  const denoise = find("capture-denoise");
  const runTranscript = find("capture-transcribe");
  const monitorButton = find("capture-monitor");
  const monitorStatus = find("capture-monitor-status");
  const output = find("capture-output");
  const speechLanguage = find("media-speech-language");
  const audioClassifier = find("media-audio-classifier");
  const soundNames = {apple: "Apple SoundAnalysis", zipformer: "Zipformer Small INT8"};
  const audioPreference = "voice-studio.audio-classifier";
  try {
    const saved = localStorage.getItem(audioPreference);
    if (Object.hasOwn(soundNames, saved)) audioClassifier.value = saved;
  } catch {}
  audioClassifier.addEventListener("change", () => {
    try { localStorage.setItem(audioPreference, audioClassifier.value); } catch {}
  });
  const distinguish = find("media-speakers");
  const speakerCount = find("media-speaker-count");
  const mergeSpeakers = find("media-speaker-merge");
  const quality = find("media-quality");
  const visual = find("media-visual");
  const peopleCount = find("media-people-count");
  const peoplePreference = "voice-studio-people-count";
  try {
    peopleCount.checked = localStorage.getItem(peoplePreference) !== "0";
  } catch {}
  const bodyLayer = find("media-body");
  const gazeLayer = find("media-gaze");
  const sceneOption = find("media-scene");
  const layerCount = find("media-layer-count");
  const objectOptions = find("media-object-options");
  const objectSampling = find("media-object-sampling");
  const objectReconciliation = find("media-object-reconciliation");
  const evidenceStatus = find("media-evidence-status");
  let videoEvidence;
  let displayedEvidence;
  let sceneEvidence;
  const start = find("capture-start");
  const stopButton = find("capture-stop");
  let mediaUrl;
  let previewUpload;
  let originalUpload;
  let exportingVideo = false;
  let stopCapture;
  let captureController;
  let queue = Promise.resolve();
  let pending = 0;
  let elapsed = 0;
  let failures = 0;
  let captureLimit;
  let transcript;
  let captureLanguage;
  let captureLanguageMode;
  let dialoguePauses;
  let captured;
  let exportData;
  // UI observations change independently from a saved material's compare-and-swap revision.
  let uiContextRevision = 0;
  const dialogueAudit = mountDialogueAudit(find('media-dialogue-audit'), {
    currentReport: () => exportData, currentSource: () => originalUpload,
    seek: stamp => {playback.currentTime = stamp / 1000;},
  });
  mountTranscriptJSONDownload(json, () => exportData);
  let uploadSpeakerEdits;
  let draftTitle;
  const editing = mountTranscriptEditor(find, request, {
    current: () => exportData, title: () => draftTitle || uiText("Транскрипция"), apply: showMaterial,
  });
  let monitor = null;
  let outputDevices = new Map();
  let latencyMs = null;
  let working = false;
  /** An exact count takes precedence over the optional clustering threshold. */
  function speakerControls() {
    speakerCount.disabled = working || !distinguish.checked;
    mergeSpeakers.disabled = speakerCount.disabled || Boolean(speakerCount.value);
  }
  distinguish.onchange = speakerControls;
  speakerCount.oninput = speakerControls;
  speakerControls();

  /** Read the optional diarization controls once for each transcription. */
  function speakerOptions() {
    const count = distinguish.checked && speakerCount.value ? speakerCount.valueAsNumber : null;
    if (count !== null && (!Number.isInteger(count) || count < 1 || count > 12)) {
      throw new Error(uiText("Укажи от 1 до 12 говорящих или оставь поле пустым."));
    }
    return {speakers: distinguish.checked, speaker_count: count};
  }

  /** Translate enabled diarization options for requests carrying a WAV or video body. */
  function speakerQuery(options) {
    const query = options.speakers ? {speakers: "1"} : {};
    if (options.speaker_count !== null) query.speaker_count = options.speaker_count;
    return query;
  }

  /** Report speaker count and retain the transcript if the model could not tag it. */
  function speakerStatus(data) {
    if (data.speakerContextGraph?.decisions.some(row => row.state === "applied")) {
      const known = new Set(data.segments.map(effectiveSpeaker).filter(id => id !== null));
      return known.size ? uiText` · говорящих: ${known.size}` : "";
    }
    if (data.speakerWarning) return " · " + apiText(data.speakerWarning);
    return data.speakerCount ? uiText` · говорящих: ${data.speakerCount}` : "";
  }

  /** Replace an unsupported video with a temporary WebM on the same playback clock. */
  async function compatiblePreview() {
    const audioOnlyPlayback = playback.readyState >= 1 && !playback.videoWidth;
    const hasVideoAnalysis = videoEvidence?.timeline || sceneEvidence?.frames?.length;
    if ((!playback.error && !audioOnlyPlayback) || !previewUpload || !hasVideoAnalysis) return;
    const source = previewUpload;
    previewUpload = null;
    const originalUrl = mediaUrl;
    previewStatus.textContent = uiText("Подготавливаем совместимый предпросмотр…");
    previewStatus.hidden = false;
    try {
      const file = await request("video-preview", source);
      if (originalUrl !== mediaUrl) return;
      URL.revokeObjectURL(mediaUrl);
      mediaUrl = URL.createObjectURL(file);
      playback.src = mediaUrl;
      previewStatus.textContent = uiText("Совместимый предпросмотр готов");
    } catch (error) {
      if (originalUrl === mediaUrl) previewStatus.textContent = error.message;
    }
  }

  playback.addEventListener("error", compatiblePreview);
  playback.addEventListener("loadedmetadata", compatiblePreview);

  /** Free the last local media URL and reset the visible transcript. */
  function reset() {
    images.clear();
    uiContextRevision++;
    analysis.reset();
    music.setSource(null);
    dubbing.setData(null, null);
    scene.setData(null);
    overlay.setData(null, []);
    editorTimeline.setData(null);
    playback.pause();
    playback.removeAttribute("src");
    playback.load();
    playback.hidden = true;
    if (mediaUrl) URL.revokeObjectURL(mediaUrl);
    mediaUrl = null;
    previewUpload = null;
    originalUpload = null;
    stageSelection.setSource(null);
    stageSelection.setReport(null);
    sourceName.textContent = "";
    clothing.update(null);
    videoExport.hidden = true;
    videoExportStatus.textContent = "";
    previewStatus.hidden = true;
    updateSrt(null);
    updateOcrChecks(null);
    json.hidden = true;
    list.replaceChildren();
    videoEvidence = null;
    displayedEvidence = null;
    sceneEvidence = null;
    evidenceStatus.hidden = true;
    exportData = null;
    dialogueAudit.update(null);
    chat.reset();
    statistics.setData(null);
    transcript = null;
    captureLanguage = null;
    dialoguePauses = null;
    profileDialog.reset();
    frameInspection.reset();
    activeProfiles = [];
    renderSpeakerProfiles(find("media-speaker-profiles"), find("media-speaker-profile-list"), playback, [], null);
    draftTitle = null;
    editing.update();
    result.hidden = true;
  }

  const videoLayers = [visual, peopleCount, bodyLayer, gazeLayer, sceneOption];

  /** Reflect the enabled layers and enforce their shared face-tracking dependency. */
  function syncVideoLayers(event) {
    if (event?.target === visual && !visual.checked) bodyLayer.checked = gazeLayer.checked = false;
    if ((event?.target === bodyLayer || event?.target === gazeLayer) && event.target.checked) visual.checked = true;
    objectOptions.hidden = !sceneOption.checked;
    objectSampling.disabled = working || !sceneOption.checked;
    peopleCount.disabled = working || !visual.checked;
    const count = videoLayers.filter(option => option.checked && (option !== peopleCount || visual.checked)).length;
    layerCount.textContent = uiText`Слои видео: ${count} из ${videoLayers.length}`;
  }
  for (const option of videoLayers) option.addEventListener("change", syncVideoLayers);
  peopleCount.addEventListener("change", () => {
    try { localStorage.setItem(peoplePreference, peopleCount.checked ? "1" : "0"); } catch {}
  });
  syncVideoLayers();

  /** Disable competing sources while their audio is being captured or decoded. */
  function busy(active) {
    working = active;
    stageSelection.setBusy(active);
    analysis.setAvailable(!active && Boolean(originalUpload));
    objectReconciliation.disabled = active;
    editing.setBusy(active);
    input.disabled = kind.disabled = start.disabled = distinguish.disabled = active;
    speechLanguage.disabled = audioClassifier.disabled = quality.disabled = visual.disabled = sceneOption.disabled = active;
    syncVideoLayers();
    bodyLayer.disabled = gazeLayer.disabled = active;
    speakerControls();
    recording.disabled = denoise.disabled = runTranscript.disabled = monitorButton.disabled = output.disabled = active;
    transcribeRecording.disabled = active || !recording.value;
    device.disabled = find("capture-refresh").disabled = find("capture-url").disabled =
      find("capture-seconds").disabled = active;
  }

  /** Add speaker controls to transcripts retained after a diarization warning. */
  function prepareTranscript(data) {
    if (data.speakerWarning) {
      for (const row of data.segments) {
        if (!Object.hasOwn(row, "speaker")) row.speaker = null;
      }
      data.srt = subtitles(data.segments, data.language);
    }
    return data;
  }

  /** Render only newly arrived rows while keeping existing controls and focus. */
  function appendSegments(segments, material = null) {
    for (const segment of segments) {
      const item = document.createElement("li");
      item.dataset.cueStart = String(segment.startMs);
      const stamp = Math.floor(segment.startMs / 1000);
      const clock = `${Math.floor(stamp / 60)}:${String(stamp % 60).padStart(2, "0")}`;
      const label = document.createElement(playback.hidden ? "span" : "button");
      label.textContent = clock;
      if (!playback.hidden) {
        label.type = "button";
        label.setAttribute("aria-label", uiText("Перейти к ") + clock);
        label.onclick = () => {
          playback.currentTime = segment.startMs / 1000;
          playback.play().catch(() => {});
        };
      }
      const words = document.createElement("div");
      words.dataset.agWidth = "content";
      words.dataset.agWidthMinEm = "4";
      const speakerNumber = effectiveSpeaker(segment);
      if (!material && (Object.hasOwn(segment, "speaker") || speakerNumber !== null)) {
        const select = document.createElement("select");
        select.className = "media-speaker";
        select.setAttribute("aria-label", uiText`Кто говорит с ${clock}`);
        select.append(new Option(segment.overlap ? uiText("Смешанная речь") : uiText("Метка по анализу"), ""));
        for (let id = 1; id <= 12; id++) select.append(new Option(uiText`Спикер ${id}`, String(id)));
        const automatic = segment.linkedSpeakerNumber ?? segment.speaker ?? segment.visualSpeakerNumber;
        for (const id of new Set([speakerNumber, automatic])) {
          if (id > 12) select.append(new Option(uiText`Спикер ${id}`, String(id)));
        }
        select.value = segment.overlap && !segment.speakerNumber ? "" : (speakerNumber ?? "");
        select.onchange = () => {
          applySpeakerEdit(segment, select.value);
          select.value = segment.overlap && !segment.speakerNumber ? "" : (effectiveSpeaker(segment) ?? "");
          words.dispatchEvent(new Event("speaker-edit"));
        };
        words.append(select);
      }
      if (segment.overlap) {
        const overlap = document.createElement("small");
        overlap.className = "media-overlap";
        overlap.dataset.transcriptStatus = "overlap";
        const summary = document.createElement("span");
        const updateOverlap = () => {
          summary.textContent = mixedSpeakerLabel(segment) ?? uiText("Голоса пересекаются");
        };
        updateOverlap();
        overlap.append(summary);
        for (const {startMs, endMs} of segment.overlaps ?? []) {
          const interval = uiText`${(startMs / 1000).toFixed(2)}–${(endMs / 1000).toFixed(2)} с`;
          const marker = document.createElement(playback.hidden ? "span" : "button");
          marker.className = "media-overlap-time";
          marker.textContent = interval;
          if (!playback.hidden) {
            marker.type = "button";
            marker.setAttribute("aria-label", uiText`Перейти к пересечению ${interval}`);
            marker.onclick = () => {
              playback.currentTime = startMs / 1000;
              playback.play().catch(() => {});
            };
          }
          overlap.append(marker);
        }
        words.append(overlap);
        words.addEventListener("speaker-edit", updateOverlap);
      }
      if (material) {
        const identification = document.createElement("strong");
        identification.className = "transcript-speaker-label";
        identification.textContent = mixedSpeakerLabel(segment) ?? (speakerNumber ? uiText`Спикер ${speakerNumber}` : uiText("Спикер не определён"));
        words.append(identification);
      }
      if (segment.visualSpeaker) {
        const provenance = document.createElement("small");
        provenance.className = "media-overlap";
        provenance.dataset.transcriptStatus = "visual";
        let basis = "Light-ASD подтвердил речь";
        if (segment.visualSpeakerBasis === "editorial-with-activity") basis = "Ручная метка и Light-ASD";
        if (segment.visualSpeakerBasis === "short-reply-face-and-voice") basis = "Спикер уточнён по лицу и голосу";
        provenance.textContent = uiText`${uiText(basis)} · лицо №${segment.visualFace ?? segment.visualSpeaker}`;
        words.append(provenance);
      }
      const note = document.createElement("small");
      note.className = "media-overlap";
      note.dataset.transcriptStatus = "assignment";
      const updateSpeakerNote = () => {
        const evidence = segment.speakerNumberEvidence;
        note.hidden = !evidence;
        if (!evidence) return;
        const actor = evidence.replayed ? uiText("Правка из контекстного графа")
          : evidence.source === "agent" ? uiText("Назначено агентом") : uiText("Назначено вручную");
        const seconds = typeof evidence.contextStartMs === "number" ? (evidence.contextStartMs / 1000).toFixed(2) : null;
        note.textContent = seconds ? uiText`${actor} · основание в ${seconds} с` : actor;
      };
      updateSpeakerNote();
      words.append(note);
      words.addEventListener("speaker-edit", updateSpeakerNote);
      const speech = document.createElement("bdi");
      appendPersonMentions(speech, segment);
      words.append(speech);
      if (segment.asrCorrection?.basis === "repeated-title-card-ocr") {
        const correction = document.createElement("small");
        correction.className = "media-overlap";
        correction.dataset.transcriptStatus = "correction";
        const {word, replacement} = segment.asrCorrection;
        correction.textContent = uiText`Уточнено по титрам: ${word} → ${replacement}`;
        words.append(correction);
      }
      if (material) {
        if (segment.speaker != null && segment.speaker !== speakerNumber) {
          const acoustic = document.createElement("small");
          acoustic.className = "media-overlap";
          acoustic.dataset.transcriptStatus = "acoustic";
          acoustic.textContent = uiText`По звуку · Спикер ${segment.speaker}`;
          words.append(acoustic);
        }
        words.append(mountCueEditor(segment, material.segments, editing.patch));
      }
      const evidence = videoEvidence?.phrases?.[segment.startMs];
      if (evidence) {
        const panel = mountVideoEvidence(evidence, segment, () => exportTranscript(exportData));
        words.append(panel.element);
        words.addEventListener("speaker-edit", panel.update);
      }
      item.append(label, words);
      chat.append(item, words, speech, segment);
      list.append(item);
    }
    chat.refresh();
  }

  /** Apply a manual choice or return the cue to its current automatic assignment. */
  function setSpeakerSelection(segment, selection) {
    const previous = effectiveSpeaker(segment);
    if (selection) {
      segment.speakerNumber = Number(selection);
      segment.speakerNumberEvidence = {source: "user"};
    } else {
      delete segment.speakerNumber;
      delete segment.speakerNumberEvidence;
    }
    if (previous !== effectiveSpeaker(segment)) delete segment.speakerPhoto;
  }

  /** Regenerate speaker-dependent results after applying manual choices. */
  function refreshSpeakerExport(data) {
    const speakerIds = new Set(data.segments.map(effectiveSpeaker).filter(id => id !== null));
    data.speakerCount = speakerIds.size;
    data.srt = subtitles(data.segments, data.language);
  }

  /** Serialize a manual correction and keep it across server refinements of this upload. */
  function applySpeakerEdit(segment, selection) {
    if (!exportData) return;
    uploadSpeakerEdits?.set(speakerEditKey(segment), selection);
    setSpeakerSelection(segment, selection);
    refreshSpeakerExport(exportData);
    exportTranscript(exportData, true);
    status.textContent = uiText`Метка исправлена · говорящих: ${exportData.speakerCount}`;
  }

  /** Restore manual edits on refined cues using their unchanged source timecodes. */
  function restoreUploadSpeakerEdits(data) {
    if (!uploadSpeakerEdits?.size) return;
    for (const segment of data.segments) {
      if (uploadSpeakerEdits.has(speakerEditKey(segment))) {
        setSpeakerSelection(segment, uploadSpeakerEdits.get(speakerEditKey(segment)));
      }
    }
    refreshSpeakerExport(data);
  }

  /** Keep the active cue control focused when background evidence refreshes its row. */
  function replaceAnalysisCues(segments) {
    const focused = document.activeElement;
    const cue = focused?.closest("[data-cue-start]");
    const start = cue && list.contains(cue) ? cue.dataset.cueStart : null;
    const selector = focused?.matches(".media-speaker") ? ".media-speaker" : "button";
    list.replaceChildren();
    appendSegments(segments);
    if (start !== null) {
      list.querySelector(`[data-cue-start="${start}"] ${selector}`)?.focus({preventScroll: true});
    }
  }

  /** Reopen a saved document or keep the active playback on a local edit. */
  function showMaterial(data, keepMedia) {
    if (!keepMedia) reset();
    const base = keepMedia ? exportData : null;
    const photos = new Map((base?.segments ?? []).map(cue => [cue.startMs, cue.speakerPhoto]));
    for (const cue of data.segments) {
      const photo = photos.get(cue.startMs);
      if (photo && photo.speaker === effectiveSpeaker(cue)) cue.speakerPhoto = photo;
    }
    const material = {...base, ...data};
    applyVisualReport(material);
    list.replaceChildren();
    appendSegments(material.segments, material);
    exportTranscript(material);
    status.textContent = "";
  }

  /** Export the complete transcript with capture-relative offsets. */
  function exportTranscript(data, preserveTimeline = false) {
    // Every publisher has applied a report mutation, including metadata-only refinements.
    uiContextRevision++;
    exportData = data;
    stageSelection.setReport(data);
    dialogueAudit.update(data);
    displayedEvidence = editorFaceBindings(videoEvidence, data.segments);
    overlay.setData(displayedEvidence, data.segments, data.objectTimeline);
    const duration = Math.max(data.durationMs ?? 0, displayedEvidence?.editorTimeline?.durationMs ?? 0);
    const gazes = buildSpeakerGaze(displayedEvidence, duration);
    activeProfiles = speakerProfiles(data.segments, displayedEvidence, data.speakerProfileIndex, gazes);
    data.speakerProfiles = activeProfiles;
    editorTimeline.setData(data, displayedEvidence, gazes, preserveTimeline);
    if (data.analysisComplete !== false) {
      dubbing.setData(playback.videoWidth || displayedEvidence?.timeline || data.sceneEvidence?.frames?.length
        ? originalUpload : null, data);
      clothing.update(originalUpload, data.sourceDigest, displayedEvidence, data.speakerProfiles);
    }
    // Export the same evidence used by the workbench so newly added layers survive import.
    if (displayedEvidence) data.videoEvidence = displayedEvidence;
    chat.setData(data, activeProfiles, displayedEvidence);
    statistics.setData(data);
    renderSpeakerProfiles(find("media-speaker-profiles"), find("media-speaker-profile-list"),
      playback, data.speakerProfiles, displayedEvidence, profileDialog.open);
    editing.update();
    updateSrt(data);
    updateOcrChecks(data);
    json.hidden = false;
    videoExport.hidden = data.analysisComplete === false || !originalUpload || !(videoEvidence?.timeline || playback.videoWidth);
    result.hidden = false;
  }

  playback.addEventListener("loadedmetadata", () => {
    if (exportData && exportData.analysisComplete !== false && originalUpload && playback.videoWidth) {
      videoExport.hidden = false;
      dubbing.setData(originalUpload, exportData);
    }
  });

  videoExport.onclick = async () => {
    if (!originalUpload || !exportData || exportingVideo) return;
    const source = originalUpload;
    exportingVideo = videoExport.disabled = true;
    videoExportStatus.textContent = uiText("Собираем MP4 с техническим слоем…");
    try {
      const file = await request("video-export", overlay.exportBody(source),
                                 {language: exportData.language ?? currentLanguage()});
      const url = URL.createObjectURL(file);
      const link = document.createElement("a");
      link.href = url;
      link.download = source.name.replace(/\.[^.]+$/, "") + "-technical.mp4";
      link.click();
      setTimeout(() => URL.revokeObjectURL(url), 30000);
      if (source === originalUpload) videoExportStatus.textContent = uiText("MP4 готов");
    } catch (error) {
      if (source === originalUpload) videoExportStatus.textContent = error.message;
    } finally {
      exportingVideo = videoExport.disabled = false;
    }
  };

  /** Announce the detected language without changing the studio or interface locale. */
  function detectedLanguage(data) {
    if (data.languageMode !== "auto" || !data.segments.length) return "";
    const name = [...speechLanguage.options].find(option => option.value === data.language)?.textContent ?? data.language;
    return " · " + uiText`Определён язык: ${name}`;
  }

  /** Announce face and figure evidence as soon as that layer arrives. */
  function updateVideoEvidenceStatus() {
    if (!videoEvidence) {
      evidenceStatus.textContent = "";
      evidenceStatus.hidden = true;
      return;
    }
    if (videoEvidence) {
      const {stages = {}, sampledPhrases, faceTracking} = videoEvidence;
      const notes = [];
      if (stages.yunet === "ok") notes.push(uiText("Лица на кадрах · YuNet"));
      if (stages["light-asd"] === "ok") notes.push(uiText("Активность лица · Light-ASD"));
      else if (stages["light-asd"] === "partial" && videoEvidence.activityCoverage) {
        const windows = videoEvidence.activityCoverage.windows ?? [];
        const completed = windows.filter(row => row.status === "ok").length;
        notes.push(uiText`Light-ASD · обработано блоков: ${completed} из ${windows.length}`);
      }
      else if (stages["light-asd"] === "duration-limit") notes.push(uiText("Light-ASD доступен для речи в пределах первой минуты"));
      else if (stages["light-asd"] && stages["light-asd"] !== "running") {
        notes.push(videoEvidence.activityReason ? apiText(videoEvidence.activityReason)
          : uiText("Light-ASD недоступен для этого файла"));
      }
      if (stages.yunet === "unavailable" || stages.yunet === "failed") notes.push(uiText("Фото с YuNet недоступны"));
      if (stages.yunet === "busy") notes.push(uiText("Обработка лиц занята другим видео"));
      if (faceTracking?.status === "ok" && faceTracking.matches.length) {
        notes.push(uiText`Повторные появления лиц · ${faceTracking.matches.length}`);
      }
      if (faceTracking?.clothing?.matches?.length) {
        notes.push(uiText`Подтверждено по лицу и одежде: ${faceTracking.clothing.matches.length}`);
      }
      if (sampledPhrases) notes.push(uiText`Кадры для ${sampledPhrases} фраз`);
      const people = videoEvidence.personLayer;
      if (people?.reason) notes.push(apiText(people.reason));
      else if (people?.status === "unavailable" || people?.status === "failed") notes.push(uiText("Фигуры или оценка взгляда недоступны"));
      else if (people?.gazeUnavailable) notes.push(uiText("Оценка взгляда недоступна"));
      evidenceStatus.textContent = notes.join(" · ");
      evidenceStatus.hidden = !notes.length;
    }
  }

  /** Keep one canonical visual-layer application path before cues, profiles, overlays and exports. */
  function applyVisualReport(data) {
    applyReportEvidenceLayers(data, {videoEvidence, sceneEvidence}, {
      videoEvidence(value) {
        videoEvidence = value;
        updateVideoEvidenceStatus();
        compatiblePreview();
      },
      sceneEvidence(value) {
        sceneEvidence = value;
        scene.setData(value);
        compatiblePreview();
      },
    });
  }

  /** Apply one report view. Audio and scene patches leave the current cue controls intact. */
  function displayAnalysisReport(report, refreshCues, preserveTimeline) {
    const data = {...report};
    applyVisualReport(data);
    if (refreshCues) {
      data.segments = report.segments.map(segment => ({...segment}));
      prepareTranscript(data);
      restoreUploadSpeakerEdits(data);
      replaceAnalysisCues(data.segments);
    } else {
      data.segments = exportData.segments;
      if (uploadSpeakerEdits?.size) refreshSpeakerExport(data);
    }
    exportTranscript(data, preserveTimeline);
    return data;
  }

  /** Keep file language overrides outside the existing live capture contract. */
  function captureSpeechLanguage() {
    const language = speechLanguage.value;
    return ["default", "auto", "ru", "en"].includes(language) ? language : "auto";
  }

  /** Run the shared single-request path for uploads and fixed-length URL captures. */
  async function transcribe(route, payload, message, query = {}, retained = null) {
    busy(true);
    status.textContent = message;
    const tracked = route === "transcribe-media";
    let partial = null;
    if (tracked) {
      uploadSpeakerEdits = retainedSpeakerEdits(retained);
      analysis.start(query);
    }
    try {
      const options = {speech_lang: tracked ? speechLanguage.value : captureSpeechLanguage(), ...query};
      if (tracked) {
        options.onProgress = event => {
          if (!event.partial) {
            analysis.update(event);
            return;
          }
          const next = mergeAnalysisPartial(partial, event.partial);
          if (next === partial) return;
          displayAnalysisReport(next, next.segments !== partial?.segments, partial !== null);
          if (!partial && !["pending", "invalidated"].includes(next.analysisStages?.speech?.state)) {
            status.textContent = uiText("Основной анализ готов. Дополнительные слои загружаются…");
          }
          partial = next;
        };
      }
      const final = await request(route, payload, options);
      const data = displayAnalysisReport(final, true, partial !== null);
      if (tracked) {
        analysis.finish(undefined, data.analysisStages, data.pipeline);
        analysisOptions.open = false;
      }
      const soundStatus = ["unavailable", "failed"].includes(data.soundAnalysisStatus)
        ? uiText(" · классификация музыки и фоновых звуков недоступна") : "";
      const soundName = soundNames[data.soundClassifier];
      const graph = data.speakerContextGraph;
      const context = graph?.decisions.some(row => row.state === "conflicting") ? uiText(" · есть конфликт сохранённых меток") : "";
      const decisions = (data.speakerReview?.merges?.length ?? 0) + (data.speakerReview?.cueDecisions?.length ?? 0)
        + (data.speakerReview?.shortReplyDecisions?.length ?? 0);
      const reviewed = decisions ? uiText` · уточнений по профилям: ${decisions}` : "";
      status.textContent = uiText`Готово · фраз: ${data.segments.length}` + detectedLanguage(data) + speakerStatus(data)
        + (data.recognitionWarning ? " · " + apiText(data.recognitionWarning)
          : data.recognitionModel === "whisper-small" ? uiText(" · Whisper Small") : "") + context + reviewed
        + (soundName ? " · " + soundName : "") + soundStatus
        + (data.speakerProfileWarning ? " · " + apiText(data.speakerProfileWarning) : "");
      return data;
    } catch (error) {
      if (tracked) analysis.finish(error.message);
      status.textContent = partial ? uiText`Основной результат доступен. Анализ слоёв прерван: ${error.message}` : error.message;
    } finally {
      uploadSpeakerEdits = undefined;
      busy(false);
    }
  }

  input.onchange = () => {
    const file = input.files[0];
    if (!file) return;
    if (isImageFile(file)) {
      input.value = "";
      images.open(file);
      return;
    }
    let options;
    try {
      options = speakerOptions();
    } catch (error) {
      status.textContent = error.message;
      input.value = "";
      return;
    }
    reset();
    mediaUrl = URL.createObjectURL(file);
    draftTitle = file.name;
    sourceName.textContent = file.name;
    sourceName.title = file.name;
    previewUpload = originalUpload = file;
    music.setSource(file);
    playback.src = mediaUrl;
    playback.hidden = false;
    input.value = "";
    stageSelection.setSource(file);
    transcribe("transcribe-media", file,
      uiText(file.type.startsWith("audio/") ? "Распознаём речь…" : "Распознаём речь и проверяем видео…"),
      uploadQuery(file, options));
  };

  /** Read all current settings for either a new source or a later selected-stage run. */
  function uploadQuery(file, options = speakerOptions()) {
    const query = {...speakerQuery(options), ...stageSelection.query()};
    query.audio_classifier = audioClassifier.value;
    if (options.speakers && options.speaker_count === null && mergeSpeakers.checked) query.merge_similar = "1";
    if (quality.checked) query.quality = "1";
    if (!file.type.startsWith("audio/")) {
      if (visual.checked || bodyLayer.checked || gazeLayer.checked) query.visual = "1";
      query.people_count = peopleCount.checked && visual.checked ? "1" : "0";
      if (bodyLayer.checked) query.body = "1";
      if (gazeLayer.checked) query.gaze = "1";
      if (sceneOption.checked) {
        query.scene = "1";
        query.object_sampling = objectSampling.value;
      }
      if (objectReconciliation.checked) query.object_reconciliation = "1";
    }
    return query;
  }

  /** Reuse the selected original and its source-keyed checkpoints without resetting the workbench. */
  async function rerunMediaStages(ids, force) {
    if (working || !originalUpload) return;
    try {
      const query = {...uploadQuery(originalUpload), ...stageSelectionQuery(ids, force)};
      await transcribe("transcribe-media", originalUpload, uiText("Выполняем выбранные этапы…"), query, exportData);
    } catch (error) {
      status.textContent = error.message;
    }
  }

  recording.addEventListener("change", () => {
    transcribeRecording.disabled = working || !recording.value;
  });
  /** Use the visible history selection and the same analysis path for browser agents. */
  function transcribeSelectedRecording(signal) {
    if (working) return;
    const id = recording.value;
    if (!/^[a-f0-9]{32}$/.test(id)) return;
    let options;
    try {
      options = speakerOptions();
    } catch (error) {
      status.textContent = error.message;
      speakerCount.focus();
      return;
    }
    reset();
    playback.src = `/media/${id}.wav`;
    playback.hidden = false;
    return transcribe("transcribe-recording", {id, ...options, audio_classifier: audioClassifier.value},
      uiText("Распознаём сохранённую запись…"), {signal});
  }
  transcribeRecording.onclick = () => transcribeSelectedRecording();

  /** Update output selection while keeping audio disabled until capture starts. */
  function chooseMonitor(selection) {
    monitor = selection;
    monitorButton.textContent = selection ? uiText("Отключить аудиовыход") : uiText("Выбрать аудиовыход");
    monitorStatus.textContent = selection ? uiText`Вывод: ${selection.label || uiText("выбранное устройство")}` : "";
  }

  kind.onchange = () => {
    find("capture-device-fields").hidden = kind.value !== "input";
    find("capture-url-fields").hidden = kind.value !== "stream";
    monitorButton.hidden = monitorStatus.hidden = find("capture-monitor-help").hidden = kind.value === "stream";
    output.hidden = true;
    if (kind.value === "stream") chooseMonitor(null);
    runTranscript.closest("label").hidden = kind.value === "stream";
  };
  monitorButton.onclick = async () => {
    if (monitor) {
      output.value = "";
      chooseMonitor(null);
      return;
    }
    const devices = navigator.mediaDevices;
    if (!devices || typeof HTMLMediaElement.prototype.setSinkId !== "function") {
      monitorStatus.textContent = uiText("Этот браузер не поддерживает выбор аудиовыхода.");
      return;
    }
    busy(true);
    try {
      if (typeof devices.selectAudioOutput === "function") {
        chooseMonitor(await devices.selectAudioOutput());
      } else {
        const outputs = (await devices.enumerateDevices()).filter(item => item.kind === "audiooutput" && item.deviceId);
        const namedOutputs = outputs.filter(item => item.deviceId !== "default");
        outputDevices = new Map(outputs.map(item => [item.deviceId, item]));
        output.replaceChildren(new Option(uiText("Без вывода"), ""), new Option(uiText("Системный выход по умолчанию"), "default"),
          ...namedOutputs.map((item, index) =>
            new Option(item.label || uiText`Аудиовыход ${index + 1}`, item.deviceId)));
        output.hidden = false;
        monitorStatus.textContent = namedOutputs.length
          ? uiText("Выбери аудиовыход.")
          : uiText("Доступен системный выход. Для списка устройств разреши доступ к аудиовходам.");
      }
    } catch (error) {
      monitorStatus.textContent = error.name === "NotAllowedError" ? uiText("Аудиовыход не выбран.") : error.message;
    } finally {
      busy(false);
      if (!output.hidden) output.focus();
    }
  };
  output.onchange = () => {
    const selected = output.selectedOptions[0];
    const available = outputDevices.get(output.value);
    chooseMonitor(output.value ? {deviceId: output.value, label: available?.label || selected.textContent} : null);
  };
  find("capture-refresh").onclick = async () => {
    busy(true);
    try {
      const probe = await navigator.mediaDevices.getUserMedia({audio: true});
      try {
        const inputs = (await navigator.mediaDevices.enumerateDevices()).filter(item => item.kind === "audioinput");
        const previous = device.value;
        device.replaceChildren(new Option(uiText("Системный по умолчанию"), ""),
          ...inputs.filter(item => item.deviceId !== "default").map((item, index) =>
            new Option(item.label || uiText`Аудиовход ${index + 1}`, item.deviceId)));
        if ([...device.options].some(option => option.value === previous)) device.value = previous;
        status.textContent = uiText("Выбери аудиовход.");
      } finally {
        for (const track of probe.getTracks()) track.stop();
      }
    } catch {
      status.textContent = uiText("Браузер не получил доступ к аудиовходам.");
    } finally {
      busy(false);
    }
  };

  /** Convert global capture timestamps to SubRip time. */
  function stamp(ms) {
    return new Date(ms).toISOString().slice(11, 23).replace(".", ",");
  }

  /** Build SubRip cues with the current speaker labels. */
  function subtitles(segments, language = currentLanguage()) {
    return segments.map((cue, index) => {
      const {startMs, endMs, text, speaker, speakerName} = cue;
      const number = effectiveSpeaker(cue);
      const label = mixedSpeakerLabel(cue, undefined, language) ?? speakerName;
      let prefix = label ? `${label}: ` : "";
      if (!prefix && number) prefix = textFor(language, "Спикер {0}: ", number);
      if (!prefix && speaker !== undefined) prefix = textFor(language, "Спикер не определён: ");
      return `${index + 1}\n${stamp(startMs)} --> ${stamp(endMs)}\n${prefix}${text}\n`;
    }).join("\n");
  }

  /** Process each audio window once, in order, and make gaps explicit. */
  function enqueue(blob, offset, duration, language) {
    elapsed = offset + duration;
    if (pending >= 8) {
      failures++;
      stopCapture?.();
      return;
    }
    pending++;
    const readyAt = performance.now();
    queue = queue.then(async () => {
      try {
        const data = await request("transcribe-chunk", blob, {language, speech_lang: captureLanguage});
        if (["auto", "default"].includes(captureLanguage) && data.segments.length) {
          captureLanguage = data.language;
          captureLanguageMode = data.languageMode;
        }
        const next = data.segments.map(segment => ({
          ...segment, startMs: segment.startMs + offset, endMs: segment.endMs + offset
        }));
        transcript.push(...next);
        for (const pause of data.dialoguePauses ?? []) {
          dialoguePauses.push({...pause, startMs: pause.startMs + offset, endMs: pause.endMs + offset,
            cutMs: pause.cutMs + offset});
        }
        if (next.length || data.dialoguePauses?.length) uiContextRevision++;
        appendSegments(next);
        result.hidden = false;
        if (next.length) latencyMs = Math.round(performance.now() - readyAt);
      } catch {
        failures++;
      } finally {
        pending--;
        status.textContent = uiText`Фраз: ${transcript.length} · в очереди: ${pending}` +
          (latencyMs === null ? "" : uiText` · ответ после фрагмента: ${(latencyMs / 1000).toFixed(1)} с`) +
          (failures ? uiText` · пропусков: ${failures}` : "");
      }
    });
  }

  find("capture-form").onsubmit = async event => {
    event.preventDefault();
    const captureKind = kind.value;
    const transcribing = captureKind === "stream" || runTranscript.checked;
    let speakerSettings = {speakers: false, speaker_count: null};
    if (transcribing) {
      try {
        speakerSettings = speakerOptions();
      } catch (error) {
        status.textContent = error.message;
        speakerCount.focus();
        return;
      }
    }
    if (!transcribing && !monitor) {
      status.textContent = uiText("Выбери аудиовыход или включи транскрипцию.");
      monitorButton.focus();
      return;
    }
    if (kind.value === "stream") {
      const url = find("capture-url").value.trim();
      if (!url) {
        status.textContent = uiText("Введи ссылку на поток.");
        find("capture-url").focus();
        return;
      }
      reset();
      transcribe("transcribe-stream", {
        url, seconds: find("capture-seconds").valueAsNumber, ...speakerSettings, audio_classifier: audioClassifier.value,
      }, uiText("Захватываем отрезок потока…"));
      return;
    }
    reset();
    busy(true);
    transcript = [];
    captureLanguage = captureSpeechLanguage();
    captureLanguageMode = captureLanguage;
    dialoguePauses = [];
    captured = [];
    pending = elapsed = failures = 0;
    latencyMs = null;
    const language = currentLanguage();
    if (!navigator.mediaDevices?.[captureKind === "screen" ? "getDisplayMedia" : "getUserMedia"] ||
        transcribing && !window.AudioWorkletNode) {
      status.textContent = uiText("Захват выбранного источника недоступен в этом браузере.");
      busy(false);
      return;
    }
    status.textContent = uiText("Открываем источник…");
    const controller = new AbortController();
    captureController = controller;
    try {
      const capture = await captureAudio(captureKind, device.value, transcribing ? (blob, offset, duration) => {
        if (speakerSettings.speakers) captured.push(blob);
        enqueue(blob, offset, duration, language);
      } : null, async () => {
        clearTimeout(captureLimit);
        stopCapture = null;
        captureController = null;
        const restoreFocus = document.activeElement === stopButton;
        stopButton.hidden = true;
        if (!transcribing) {
          status.textContent = uiText("Вывод звука остановлен.");
          busy(false);
          if (restoreFocus) start.focus();
          return;
        }
        status.textContent = uiText("Обрабатываем оставшиеся фрагменты…");
        await queue;
        const resolved = ["auto", "default"].includes(captureLanguage) ? language : captureLanguage;
        let data = {segments: transcript, text: transcript.map(row => row.text).join(" "),
          srt: subtitles(transcript, resolved), language: resolved, durationMs: elapsed, skippedChunks: failures,
          dialoguePauses};
        if (captured.length) {
          status.textContent = uiText("Сопоставляем голоса по всей записи…");
          try {
            const full = await joinCapturedWav(captured);
            const labelled = await request("transcribe-capture", full,
              {language, speech_lang: captureLanguage, ...speakerQuery(speakerSettings)});
            if (labelled.segments.some(segment => Object.hasOwn(segment, "speaker"))) {
              data = labelled;
              list.replaceChildren();
              appendSegments(data.segments);
            } else {
              data.speakerWarning = labelled.speakerWarning;
            }
          } catch (error) {
            data.speakerWarning = error.message;
          } finally {
            captured = [];
          }
        }
        if (data.speakerWarning) {
          prepareTranscript(data);
          list.replaceChildren();
          appendSegments(data.segments);
        }
        if (captureLanguageMode === "auto" && data.segments.length) data.languageMode = "auto";
        exportTranscript(data);
        status.textContent = uiText`Захват завершён · фраз: ${data.segments.length}` + detectedLanguage(data) + speakerStatus(data) +
          (data.skippedChunks ? uiText` · пропусков: ${data.skippedChunks}` : "");
        busy(false);
        if (restoreFocus) start.focus();
      }, {denoise: denoise.checked, monitor, signal: controller.signal});
      stopCapture = capture.stop;
      if (transcribing) captureLimit = setTimeout(() => stopCapture?.(), 15 * 60 * 1000);
      stopButton.hidden = false;
      stopButton.focus();
      const noiseStatus = captureKind === "input" && denoise.checked
        ? (capture.noiseSuppression ? uiText(" Шумоподавление включено.") : uiText(" Вход не подтвердил шумоподавление."))
        : "";
      status.textContent = (transcribing
        ? uiText("Захват идёт. Первые фразы появятся после короткой паузы.")
        : uiText("Звук передаётся на выбранный выход.")) + noiseStatus;
    } catch (error) {
      captureController = null;
      status.textContent = controller.signal.aborted ? uiText("Захват отменён.") : error.message;
      busy(false);
    }
  };
  stopButton.onclick = () => stopCapture?.();
  /** Stop active capture or discard an input opened after leaving the tools page. */
  function closeCapture() {
    if (stopCapture) stopCapture();
    else captureController?.abort();
  }
  addEventListener("pagehide", closeCapture);
  return {
    openMaterial: editing.open, stopCapture: closeCapture,
    /** Read or update the saved preference for future video uploads. */
    peopleCount(enabled) {
      if (enabled === undefined) return {people_count: peopleCount.checked, available: visual.checked};
      if (working) throw Error(uiText("Дождись завершения текущей операции."));
      peopleCount.checked = enabled;
      peopleCount.dispatchEvent(new Event("change"));
      return {people_count: peopleCount.checked, available: visual.checked};
    },
    dismissMaterial(id, language) {
      if (exportData?.id === id && exportData.language === language) reset();
    },
    currentTranscript: () => exportData ?? (transcript?.length
      ? {segments: transcript, language: captureLanguage, dialoguePauses} : null),
    /** Read the host-owned report epoch without mutating material data or playback state. */
    contextRevision: () => uiContextRevision,
    isBusy: () => working,
    async transcribeRecording(id, signal) {
      if (working || exportData?.segments.length || transcript?.length) {
        throw Error(uiText("Дождись завершения операции или открой пустую вкладку транскрипции."));
      }
      if (![...recording.options].some(option => option.value === id)) {
        throw Error(uiText("Выбранная запись недоступна."));
      }
      recording.value = id;
      recording.dispatchEvent(new Event("change"));
      const data = await transcribeSelectedRecording(signal);
      if (!data) throw Error(status.textContent);
      return {cueCount: data.segments.length};
    },
    seek(ms) {
      if (playback.hidden || !Number.isFinite(playback.duration) || ms > playback.duration * 1000) {
        throw Error(uiText("Открой аудио или видео и укажи таймкод в его пределах."));
      }
      playback.pause();
      playback.currentTime = ms / 1000;
      return {timeMs: Math.round(playback.currentTime * 1000), paused: playback.paused};
    },
  };
}
