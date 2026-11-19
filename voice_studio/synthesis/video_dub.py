"""Render an English dub from a reviewed transcript on the source video's clock."""

import json
from pathlib import Path
import subprocess
from threading import BoundedSemaphore
from time import monotonic
import wave
from zipfile import ZIP_STORED, ZipFile

from voice_studio.synthesis.commercial_voice import ready
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


def clean_cues(metadata, duration_ms, *, require_english=True):
    """Validate ordered non-overlapping slots and translations before allocating audio."""
    rows = metadata.get("cues")
    mode = pronunciation_mode(metadata.get("pronunciationMode", "source"), "en")
    language = metadata.get("sourceLanguage")
    if language not in ("ru", "en") or not isinstance(rows, list) or not 1 <= len(rows) <= MAX_DUB_CUES:
        raise ValueError("Нужна расшифровка на русском или английском, до 400 реплик.")
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
        original, english = (row.get(key) for key in ("sourceText", "englishText"))
        values = (original, english) if require_english else (original,)
        if any(not isinstance(value, str) or not 1 <= len(value.strip()) <= 1000 for value in values):
            raise ValueError(f"Добавь исходный текст и английский перевод реплики №{index}.")
        if row.get("overlap"):
            raise ValueError(f"Раздели перекрывающуюся речь в реплике №{index} перед дубляжом.")
        cue = {"startMs": start, "endMs": end, "speaker": number, "sourceText": " ".join(original.split()),
               "englishText": " ".join(english.split()) if require_english else ""}
        if mode != "source":
            cue["pronunciationMode"] = mode
        if require_english and (speed := row.get("tempo")) is not None:
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
    cues, language = clean_cues(metadata, duration, require_english=False)
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


def render_reference_tracks(source, cues, folder):
    """Cut original references and measure the speech edges on the same decoded clock."""
    pcm = folder / "reference-source.wav"
    decode_original(source, pcm, RATE)
    start = audio_offset(source)
    references = render_voice_profiles(pcm, folder / BACKGROUND_WAV, cues, start, folder)
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
    fallback = primary.get("fallback")
    prompts = (primary, primary, fallback) if fallback else (primary,) * 3
    if fallback and "tempo" in cue:
        prompts = (primary, fallback)
    for attempt, prompt in enumerate(prompts):
        # Reproduce the baseline seed when returning to the original prompt.
        seed = 419 + index + (0 if prompt is fallback else attempt * 1009)
        try:
            generate("qwen", {"output": str(raw), "text": cue["englishText"], "language": "en",
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
    raise ValueError(f"Английская реплика №{index} выходит за свой интервал. Сократи перевод или увеличь темп.")


def create_english_track(cues, references, windows, duration_ms, folder):
    """Place every rendered cue and silence at exact sample boundaries."""
    deadline = monotonic() + 1200
    position, audit = 0, []
    with open_pcm(folder / DIALOGUE_WAV) as english:
        for index, (cue, window) in enumerate(zip(cues, windows, strict=True), 1):
            pcm, rendered_start, rendered_end, speed, raw_ms, used_fallback = render_cue(
                cue, index, references, window, folder, deadline,
            )
            start = sample_index(rendered_start)
            write_silence(english, start - position)
            english.writeframesraw(pcm)
            position = start + len(pcm) // 2
            audit.append({**cue, "rawDurationMs": raw_ms, "tempo": speed, "voiceReferenceFallback": used_fallback,
                          "sourceSpeechStartMs": window[0], "sourceSpeechEndMs": window[1],
                          "renderedStartMs": rendered_start, "renderedEndMs": rendered_end})
        write_silence(english, sample_index(duration_ms) - position)
    return audit


def mux_tracks(source, folder, source_lang, video_codec, audio_codec):
    """Tag both tracks and preserve compatible bitstreams from the original."""
    video = ["-c:v", "copy"] if video_codec in {"h264", "hevc", "av1"} else [
        "-c:v", "libx264", "-preset", "veryfast", "-pix_fmt", "yuv420p"]
    default_language = {"ru": "rus", "en": "eng"}[source_lang]
    ffmpeg(["-protocol_whitelist", "file,pipe", "-format_whitelist", FORMATS, "-i", str(source),
            "-i", str(folder / ENGLISH_WAV), "-map", "0:V:0", "-map", "0:a:0", "-map", "1:a:0",
            *video, "-c:a:0", "copy" if audio_codec == "aac" else "aac", "-c:a:1", "aac", "-b:a:1", "192k",
            "-metadata:s:a:0", f"language={default_language}", "-metadata:s:a:0", "handler_name=Original",
            "-metadata:s:a:1", "language=eng", "-metadata:s:a:1", "handler_name=English dub",
            "-disposition:a:0", "default", "-disposition:a:1", "0", "-movflags", "+faststart",
            str(folder / DUAL_AUDIO_MP4)], timeout=360)


def finish_video_dub(source, folder, archive, report, video_codec, audio_codec):
    """Mix an aligned English voice with the source bed and package the final media."""
    report = {**report, "background": {"model": MODEL_NAME, "file": BACKGROUND_WAV}}
    mix_english(folder / DIALOGUE_WAV, folder / BACKGROUND_WAV, report["durationMs"], folder / ENGLISH_WAV)
    mux_tracks(source, folder, report["sourceLanguage"], video_codec, audio_codec)
    subtitles = "\n\n".join(
        f"{index}\n{srt_time(cue['renderedStartMs'])} --> {srt_time(cue['renderedEndMs'])}\n{cue['englishText']}"
        for index, cue in enumerate(report["cues"], 1)) + "\n"
    with ZipFile(archive, "w", compression=ZIP_STORED) as output:
        for filename in (ENGLISH_WAV, DIALOGUE_WAV, BACKGROUND_WAV, DUAL_AUDIO_MP4):
            output.write(folder / filename, filename)
        output.writestr("english.srt", subtitles)
        output.writestr("dub-report.json", json.dumps(report, ensure_ascii=False, indent=2))
    return report


def build_video_dub(source, metadata, folder, archive):
    """Build a dub with the original music and effects, then publish the ZIP."""
    if not SLOT.acquire(blocking=False):
        raise ValueError("Сейчас создаётся другая дорожка. Дождись завершения.")
    try:
        duration, video_codec, audio_codec = source_streams(source)
        cues, language = clean_cues(metadata, duration)
        if not ready("qwen"):
            raise ValueError("Локальная модель Qwen3-TTS недоступна.")
        prepare_background(source, duration, folder)
        references, windows = render_reference_tracks(source, cues, folder)
        audit = create_english_track(cues, references, windows, duration, folder)
        report = {"durationMs": duration, "sourceLanguage": language, "dubLanguage": "en", "model": "qwen",
                  "references": reference_report(references),
                  "cues": audit}
        return finish_video_dub(source, folder, archive, report, video_codec, audio_codec)
    finally:
        SLOT.release()
