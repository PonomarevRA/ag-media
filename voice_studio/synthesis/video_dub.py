"""Render a dub from a reviewed transcript on the source video's clock."""

import json
import logging
from pathlib import Path
import subprocess
from threading import BoundedSemaphore
from time import monotonic
import wave
from zipfile import ZIP_STORED, ZipFile

from voice_studio.synthesis.commercial_voice import ready
from voice_studio.audio.voice_quality import measure_dialogue, restore_dialogue
from voice_studio.synthesis.dub_background import BACKGROUND_WAV, MODEL_NAME, mix_english, prepare_background
from voice_studio.synthesis.dub_timing import source_speech_window, speech_edges
from voice_studio.synthesis.dub_voice_profiles import RATE, reference_report, render_voice_profiles
from voice_studio.synthesis.engines import ffmpeg, generate
from voice_studio.synthesis.voice_controls import pronunciation_mode
from voice_studio.speech.media_transcript import FORMATS, MAX_MEDIA_SECONDS, audio_offset, srt_time
from voice_studio.speech.recognition import translate_clips

MAX_DUB_CUES = 400
MAX_SPEED = 1.8
MIN_SPEED = .8
SLOT = BoundedSemaphore(1)
ENGLISH_WAV = "english.wav"
DIALOGUE_WAV = "english-dialogue.wav"
DUAL_AUDIO_MP4 = "original-and-english.mp4"
DUB_FILES = {
    "en": (ENGLISH_WAV, DIALOGUE_WAV, DUAL_AUDIO_MP4, "english.srt"),
    "ru": ("russian.wav", "russian-dialogue.wav", "original-and-russian.mp4", "russian.srt"),
}
AUDIO_LANGUAGES = {"ru": "rus", "en": "eng", "zh": "zho"}
DUB_TRACK_NAMES = {"ru": "Russian dub", "en": "English dub"}


def source_streams(source):
    """Read the duration and primary audio/video codecs from a local media file."""
    try:
        raw = subprocess.check_output([
            "ffprobe", "-v", "error", "-protocol_whitelist", "file,pipe", "-format_whitelist", FORMATS,
            "-show_entries", "format=duration:stream=codec_type,codec_name:stream_disposition=attached_pic",
            "-of", "json", str(source),
        ], timeout=20)
        data = json.loads(raw)
        streams = data["streams"]
        video = next(row for row in streams if row["codec_type"] == "video" and
                     not row.get("disposition", {}).get("attached_pic"))
        audio = next(row for row in streams if row["codec_type"] == "audio")
        duration = float(data["format"]["duration"])
        if not 0 < duration <= MAX_MEDIA_SECONDS:
            raise ValueError
    except (ValueError, OSError, LookupError, StopIteration, subprocess.SubprocessError) as error:
        raise ValueError("Для дубляжа нужно видео со звуком длительностью до 15 минут.") from error
    return round(duration * 1000), video.get("codec_name"), audio.get("codec_name")


def clean_cues(metadata, duration_ms, *, require_text=True):
    """Validate ordered non-overlapping slots and translations before allocating audio."""
    rows = metadata.get("cues")
    dub_language = metadata.get("dubLanguage", "en") if require_text else "en"
    if dub_language not in ("ru", "en"):
        raise ValueError("Выбери русский или английский язык дубляжа.")
    mode = pronunciation_mode(metadata.get("pronunciationMode", "source"), dub_language)
    text_key = "englishText" if dub_language == "en" else "targetText"
    language = metadata.get("sourceLanguage")
    if not isinstance(language, str) or language not in AUDIO_LANGUAGES:
        raise ValueError("Нужна расшифровка на русском, английском или китайском.")
    if not isinstance(rows, list) or not 1 <= len(rows) <= MAX_DUB_CUES:
        raise ValueError("Для дубляжа нужно от 1 до 400 реплик.")
    cues = []
    last_end = 0
    for index, row in enumerate(rows, 1):
        if not isinstance(row, dict):
            raise ValueError(f"Некорректные данные реплики №{index}.")
        start, end, number = (row.get(key) for key in ("startMs", "endMs", "speaker"))
        if any(type(value) is not int for value in (start, end, number)) or not 1 <= number <= 999:
            raise ValueError(f"Укажи спикера и таймкод для реплики №{index}.")
        if not last_end <= start < end <= duration_ms:
            raise ValueError(f"Исправь таймкод или пересечение в реплике №{index}.")
        original, translation = (row.get(key) for key in ("sourceText", text_key))
        values = (original, translation) if require_text else (original,)
        if any(not isinstance(value, str) or not 1 <= len(value.strip()) <= 1000 for value in values):
            raise ValueError(f"Проверь исходный текст и текст дубляжа реплики №{index}.")
        if row.get("overlap"):
            raise ValueError(f"Раздели перекрывающуюся речь в реплике №{index} перед дубляжом.")
        cue = {"startMs": start, "endMs": end, "speaker": number, "sourceText": " ".join(original.split()),
               text_key: " ".join(translation.split()) if require_text else ""}
        if dub_language != "en":
            cue["dubLanguage"] = dub_language
        if mode != "source":
            cue["pronunciationMode"] = mode
        if require_text and (speed := row.get("tempo")) is not None:
            if type(speed) not in (int, float) or not MIN_SPEED <= speed <= MAX_SPEED:
                raise ValueError(f"Укажи темп от 0,8 до 1,8 для реплики №{index}.")
            cue["tempo"] = speed
        cues.append(cue)
        last_end = end
    return cues, language


def sample_index(ms):
    """Convert a source-clock timestamp to a sample index at 24 kHz."""
    return ms * RATE // 1000


def open_pcm(path, rate=RATE):
    """Create a mono 24 kHz PCM output used by the synthesis and mix stages."""
    output = wave.open(str(path), "wb")
    output.setparams((1, 2, rate, 0, "NONE", "not compressed"))
    return output


def write_silence(output, frames):
    """Write zero samples in bounded blocks for long video gaps."""
    zeros = bytes(RATE * 2)
    while frames > 0:
        size = min(frames, RATE)
        output.writeframesraw(zeros[:size * 2])
        frames -= size


def decode_original(source, pcm, rate):
    """Decode the selected audio stream at a known PCM sample rate."""
    ffmpeg(["-protocol_whitelist", "file,pipe", "-format_whitelist", FORMATS, "-i", str(source),
            "-map", "0:a:0", "-af", "asetpts=PTS-STARTPTS,aresample=async=1:first_pts=0",
            "-ac", "1", "-ar", str(rate), "-t", str(MAX_MEDIA_SECONDS), "-c:a", "pcm_s16le", str(pcm)])


def suggest_english(source, metadata, folder):
    """Translate each timed speech slot locally with multilingual Whisper Small."""
    duration, _, _ = source_streams(source)
    cues, language = clean_cues(metadata, duration, require_text=False)
    if language == "en":
        return [cue["sourceText"] for cue in cues]
    rate = 16_000
    pcm = folder / "translation-source.wav"
    decode_original(source, pcm, rate)
    offset = audio_offset(source)
    translations = [""] * len(cues)
    clips = {}
    with wave.open(str(pcm), "rb") as audio:
        for index, cue in enumerate(cues):
            start = max(0, (cue["startMs"] - offset) * rate // 1000)
            end = min(audio.getnframes(), (cue["endMs"] - offset) * rate // 1000)
            if end <= start:
                continue
            audio.setpos(start)
            chunk = folder / f"translation-cue-{index}.wav"
            with open_pcm(chunk, rate) as output:
                output.writeframes(audio.readframes(end - start))
            clips[index] = chunk
    for index, text in zip(clips, translate_clips(list(clips.values()), language), strict=True):
        translations[index] = " ".join(text.split())[:1000]
    return translations


def render_reference_tracks(source, cues, folder, voice_prompts=None):
    """Cut original references and measure the speech edges on the same decoded clock."""
    pcm = folder / "reference-source.wav"
    decode_original(source, pcm, RATE)
    start = audio_offset(source)
    references = dict(voice_prompts or {})
    unassigned = [cue for cue in cues if cue["speaker"] not in references]
    if unassigned:
        references.update(render_voice_profiles(pcm, folder / BACKGROUND_WAV, unassigned, start, folder))
    with wave.open(str(pcm), "rb") as original:
        windows = [source_speech_window(original, cue["startMs"], cue["endMs"], start, RATE) for cue in cues]
    return references, windows


def read_voice_pcm(raw, folder, index):
    """Read generated speech and normalize engines with a different WAV layout."""
    try:
        with wave.open(str(raw), "rb") as audio:
            if (audio.getframerate(), audio.getnchannels(), audio.getsampwidth()) == (RATE, 1, 2):
                return audio.readframes(audio.getnframes())
    except wave.Error as error:
        raise ValueError(f"Модель вернула повреждённый WAV для реплики №{index}.") from error
    normalized = folder / "normalized.wav"
    ffmpeg(["-i", str(raw), "-ac", "1", "-ar", str(RATE), "-c:a", "pcm_s16le", str(normalized)])
    with wave.open(str(normalized), "rb") as audio:
        return audio.readframes(audio.getnframes())


def change_tempo(pcm, speed, folder):
    """Adjust duration with FFmpeg atempo while preserving pitch."""
    if abs(speed - 1) < .002:
        return pcm
    voice, aligned = folder / "voice.wav", folder / "aligned.wav"
    with open_pcm(voice) as audio:
        audio.writeframes(pcm)
    ffmpeg(["-i", str(voice), "-af", f"atempo={speed:.6f}", "-c:a", "pcm_s16le", str(aligned)])
    with wave.open(str(aligned), "rb") as audio:
        return audio.readframes(audio.getnframes())


def fit_voice(pcm, cue, window, folder):
    """Align speech to the original edges and increase tempo only as needed to fit."""
    first, last = speech_edges(pcm, RATE)
    voice = pcm[first * 2:last * 2]
    frames = len(voice) // 2
    if not frames or not any(voice):
        return None
    start, target_end = window
    capacity = sample_index(cue["endMs"] - start)
    target = sample_index(target_end - start)
    requested = cue.get("tempo")
    speed = requested if requested is not None else max(MIN_SPEED, min(MAX_SPEED, frames / max(1, target)))
    fitted = change_tempo(voice, speed, folder)
    if len(fitted) // 2 > capacity and requested is None:
        # atempo can vary slightly from the requested duration, so leave up to 20 ms.
        safe_capacity = capacity - min(sample_index(20), capacity // 20)
        speed = min(MAX_SPEED, speed * (len(fitted) // 2) / max(1, safe_capacity))
        fitted = change_tempo(voice, speed, folder)
    length = len(fitted) // 2
    if not 0 < length <= capacity:
        return None
    return fitted, start, start + round(length * 1000 / RATE), round(speed, 3)


def render_cue(cue, index, references, window, folder, deadline):
    """Try independent seeds when Qwen overruns the slot or stalls during decoding."""
    raw = folder / "raw.wav"
    allowed = max(45, min(180, (cue["endMs"] - cue["startMs"]) / 250))
    has_take = False
    primary = references[cue["speaker"]]
    language = cue.get("dubLanguage", "en")
    text = cue["englishText"] if language == "en" else cue["targetText"]
    fallback = primary.get("fallback")
    prompts = (primary, primary, fallback) if fallback else (primary,) * 3
    if fallback and "tempo" in cue:
        prompts = (primary, fallback)
    for attempt, prompt in enumerate(prompts):
        # Reproduce the baseline seed when returning to the original prompt.
        seed = 419 + index + (0 if prompt is fallback else attempt * 1009)
        try:
            generate("qwen", {"output": str(raw), "text": text, "language": language,
                              "seed": seed, "references": [prompt],
                              "pronunciationMode": cue.get("pronunciationMode", "source"),
                              "deadline": min(deadline, monotonic() + allowed)}, False)
        except subprocess.TimeoutExpired:
            if monotonic() >= deadline:
                break
            continue
        has_take = True
        pcm = read_voice_pcm(raw, folder, index)
        if fit := fit_voice(pcm, cue, window, folder):
            return (*fit, round(len(pcm) * 500 / RATE), prompt is fallback)
        if "tempo" in cue and not fallback:
            break
    if not has_take:
        raise ValueError(f"Qwen3-TTS не успела синтезировать реплику №{index}.")
    raise ValueError(f"Реплика №{index} выходит за свой интервал. Сократи текст дубляжа или увеличь темп.")


def create_dialogue_track(cues, references, windows, duration_ms, folder, filename):
    """Place every rendered cue and silence at exact sample boundaries."""
    deadline = monotonic() + 1200
    position, audit = 0, []
    with open_pcm(folder / filename) as dialogue:
        for index, (cue, window) in enumerate(zip(cues, windows, strict=True), 1):
            pcm, rendered_start, rendered_end, speed, raw_ms, used_fallback = render_cue(
                cue, index, references, window, folder, deadline,
            )
            start = sample_index(rendered_start)
            write_silence(dialogue, start - position)
            dialogue.writeframesraw(pcm)
            position = start + len(pcm) // 2
            audit.append({**cue, "rawDurationMs": raw_ms, "tempo": speed, "voiceReferenceFallback": used_fallback,
                          "sourceSpeechStartMs": window[0], "sourceSpeechEndMs": window[1],
                          "renderedStartMs": rendered_start, "renderedEndMs": rendered_end})
        write_silence(dialogue, sample_index(duration_ms) - position)
    return audit


def mux_tracks(source, folder, source_lang, dub_language, video_codec, audio_codec, clean_mix=None):
    """Tag both tracks and preserve compatible bitstreams from the original."""
    video = ["-c:v", "copy"] if video_codec in {"h264", "hevc", "av1"} else [
        "-c:v", "libx264", "-preset", "veryfast", "-pix_fmt", "yuv420p"]
    mixed, _, video_file, _ = DUB_FILES[dub_language]
    inputs = ["-i", str(folder / mixed)]
    clean_track_args = []
    if clean_mix is not None:
        inputs.extend(("-i", str(clean_mix)))
        clean_track_args = ["-map", "2:a:0", "-c:a:2", "aac", "-b:a:2", "192k",
                            "-metadata:s:a:2", f"language={AUDIO_LANGUAGES[dub_language]}",
                            "-metadata:s:a:2", f"handler_name={DUB_TRACK_NAMES[dub_language]} (clean)",
                            "-disposition:a:2", "0"]
    ffmpeg(["-protocol_whitelist", "file,pipe", "-format_whitelist", FORMATS, "-i", str(source),
            *inputs, "-map", "0:V:0", "-map", "0:a:0", "-map", "1:a:0", *clean_track_args,
            *video, "-c:a:0", "copy" if audio_codec == "aac" else "aac", "-c:a:1", "aac", "-b:a:1", "192k",
            "-metadata:s:a:0", f"language={AUDIO_LANGUAGES[source_lang]}", "-metadata:s:a:0", "handler_name=Original",
            "-metadata:s:a:1", f"language={AUDIO_LANGUAGES[dub_language]}",
            "-metadata:s:a:1", f"handler_name={DUB_TRACK_NAMES[dub_language]}",
            "-disposition:a:0", "default", "-disposition:a:1", "0", "-movflags", "+faststart",
            str(folder / video_file)], timeout=360)


def prepare_clean_mix(source, folder, report, dialogue, clean_mix):
    """Add optional dialogue treatment while reusing the exact original bed."""
    cues = report["cues"]
    original = folder / "reference-source.wav"
    profiles = {}
    try:
        if original.is_file():
            spans = [(cue["sourceSpeechStartMs"], cue["sourceSpeechEndMs"]) for cue in cues]
            profiles["source"] = measure_dialogue(original, spans, offset_ms=audio_offset(source))
    except (OSError, ValueError, subprocess.SubprocessError):
        profiles["source"] = {"status": "unavailable"}
    try:
        cleaned = folder / "clean-dialogue.wav"
        spans = [(cue["renderedStartMs"], cue["renderedEndMs"]) for cue in cues]
        profiles.update(restore_dialogue(folder / dialogue, cleaned, spans))
        if profiles["status"] == "enhanced":
            mix_english(cleaned, folder / BACKGROUND_WAV, report["durationMs"], clean_mix)
            profiles["file"] = clean_mix.name
    except (OSError, ValueError, subprocess.SubprocessError, wave.Error) as error:
        logging.getLogger(__name__).warning("Optional clean dub track failed: %s", error)
        profiles["status"] = "unavailable"
    return profiles


def finish_video_dub(source, folder, archive, report, video_codec, audio_codec):
    """Mix the aligned dialogue with the source bed and package the final media."""
    report = {**report, "background": {"model": MODEL_NAME, "file": BACKGROUND_WAV}}
    language = report["dubLanguage"]
    mixed, dialogue, video_file, subtitle_file = DUB_FILES[language]
    text_key = "englishText" if language == "en" else "targetText"
    mix_english(folder / dialogue, folder / BACKGROUND_WAV, report["durationMs"], folder / mixed)
    clean_mix = folder / f"{Path(mixed).stem}-clean.wav"
    quality = prepare_clean_mix(source, folder, report, dialogue, clean_mix)
    report["audioQuality"] = quality
    clean_track = clean_mix if quality["status"] == "enhanced" else None
    mux_tracks(source, folder, report["sourceLanguage"], language, video_codec, audio_codec, clean_track)
    subtitles = "\n\n".join(
        f"{index}\n{srt_time(cue['renderedStartMs'])} --> {srt_time(cue['renderedEndMs'])}\n{cue[text_key]}"
        for index, cue in enumerate(report["cues"], 1)) + "\n"
    with ZipFile(archive, "w", compression=ZIP_STORED) as output:
        for filename in (mixed, dialogue, BACKGROUND_WAV, video_file):
            output.write(folder / filename, filename)
        if clean_track:
            output.write(clean_track, clean_track.name)
        output.writestr(subtitle_file, subtitles)
        output.writestr("dub-report.json", json.dumps(report, ensure_ascii=False, indent=2))
    return report


def build_video_dub(source, metadata, folder, archive, *, voice_prompts=None):
    """Build a dub with the original music and effects, then publish the ZIP."""
    if not SLOT.acquire(blocking=False):
        raise ValueError("Сейчас создаётся другая дорожка. Дождись завершения.")
    try:
        duration, video_codec, audio_codec = source_streams(source)
        cues, language = clean_cues(metadata, duration)
        if voice_prompts and set(voice_prompts) - {cue["speaker"] for cue in cues}:
            raise ValueError("Профиль назначен спикеру, которого нет в дубляже.")
        if not ready("qwen"):
            raise ValueError("Локальная модель Qwen3-TTS недоступна.")
        prepare_background(source, duration, folder)
        references, windows = (render_reference_tracks(source, cues, folder, voice_prompts)
                               if voice_prompts else render_reference_tracks(source, cues, folder))
        dub_language = metadata.get("dubLanguage", "en")
        _, dialogue, _, _ = DUB_FILES[dub_language]
        audit = create_dialogue_track(cues, references, windows, duration, folder, dialogue)
        report = {"durationMs": duration, "sourceLanguage": language, "dubLanguage": dub_language, "model": "qwen",
                  "pronunciationMode": metadata.get("pronunciationMode", "source"),
                  "references": reference_report(references),
                  "cues": audit}
        return finish_video_dub(source, folder, archive, report, video_codec, audio_codec)
    finally:
        SLOT.release()
