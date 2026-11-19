"""Classify independent music and sound cues on prepared audio and attach pitch evidence."""

from voice_studio.mod_runtime import model_provider

from hashlib import sha256
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory
from typing import Literal
import wave

from voice_studio.core.analysis_stages import stage
from voice_studio.audio.speech_intonation import measure_intonation
from voice_studio.core.storage_cache import use_cache
from voice_studio.audio.zipformer_audio import BASIS as ZIPFORMER_BASIS, classify_zipformer

from voice_paths import RUNTIME_DIR, ensure_temporary_root

LABELS = {
    "music": ("Музыка", "Music"),
    "rising": ("Восходящая", "Rising"),
    "falling": ("Нисходящая", "Falling"),
    "steady": ("Ровная", "Steady"),
    "rise-fall": ("Подъём и спад", "Rise and fall"),
    "fall-rise": ("Спад и подъём", "Fall and rise"),
    "applause": ("Аплодисменты", "Applause"),
    "laughter": ("Смех", "Laughter"),
    "crowd": ("Толпа", "Crowd"),
    "cricket_chirp": ("Сверчок", "Cricket"),
    "bird": ("Птицы", "Birds"),
    "dog_bark": ("Лай", "Barking"),
    "rain": ("Дождь", "Rain"),
    "wind": ("Ветер", "Wind"),
    "traffic_noise": ("Дорожный шум", "Traffic"),
    "door_slam": ("Хлопок двери", "Door slam"),
    "knock": ("Стук", "Knock"),
    "typing_computer_keyboard": ("Клавиатура", "Keyboard"),
    "glass_breaking": ("Звон разбитого стекла", "Glass breaking"),
    "siren": ("Сирена", "Siren"),
    "cough": ("Кашель", "Cough"),
}
INTONATION = {"rising", "falling", "steady", "rise-fall", "fall-rise"}
BACKGROUND = set(LABELS) - INTONATION - {"music"}
MUSIC_MIN = .32
MUSIC_SEED = .45
WINDOW_HOP_MS = 750
APPLE_BASIS = "apple-sound-analysis-v1"
SOUND_BASES = {"apple": APPLE_BASIS, "zipformer": ZIPFORMER_BASIS}
SoundClassifier = Literal["auto", "apple", "zipformer"]


def select_sound_classifier(requested="auto"):
    """Resolve the saved choice or the platform default for sound tagging."""
    if requested == "auto":
        requested = os.environ.get("VOICE_AUDIO_CLASSIFIER", "auto")
    if requested == "auto":
        return "apple" if sys.platform == "darwin" else "zipformer"
    if not isinstance(requested, str) or requested not in SOUND_BASES:
        raise ValueError("Классификатор звука должен быть auto, apple или zipformer.")
    return requested


def sound_binary():
    """Build the small Apple SoundAnalysis helper once per source revision."""
    swift = shutil.which("swiftc") if sys.platform == "darwin" else None
    if not swift:
        return None
    source = Path(__file__).with_name("native") / "audio_scene.swift"
    ensure_temporary_root()
    cache = RUNTIME_DIR / "audio-scene"
    cache.mkdir(mode=0o700, parents=True, exist_ok=True)
    key = sha256(source.read_bytes()).hexdigest()[:16]
    target = cache / f"sound-{key}"
    if not target.is_file():
        with use_cache(cache, cache / "swift-modules"), TemporaryDirectory(dir=cache) as folder:
            artifact = Path(folder) / "sound"
            subprocess.run([swift, "-O", "-module-cache-path", str(cache / "swift-modules"),
                            str(source), "-o", str(artifact)], check=True, timeout=40, capture_output=True)
            artifact.replace(target)
    return target


@model_provider('audio.classification', 'builtin.apple-sound')
def classify(wav, duration_ms):
    """Run local inference and report an unavailable backend without blocking transcription."""
    if duration_ms < 1500:
        return [], "too-short"
    try:
        binary = sound_binary()
        if binary is None:
            return [], "unavailable"
        chunk = os.environ.get("VOICE_SOUND_STREAM_CHUNK_MS", "")
        stream = chunk.isascii() and chunk.isdecimal() and len(chunk) <= 4 and 100 <= int(chunk) <= 6000
        options = ["--stream", str(wav), chunk] if stream else [str(wav)]
        raw = subprocess.check_output([str(binary), *options], timeout=max(15, duration_ms / 10000),
                                      stderr=subprocess.DEVNULL)
        return json.loads(raw), "done"
    except (OSError, ValueError, subprocess.SubprocessError):
        return [], "failed"


def observations(rows, duration_ms):
    """Map a 1.5 s model window to its central hop and keep independent class scores."""
    result = []
    for row in rows:
        start, end = round(1000 * row["start"]), round(1000 * row["end"])
        center = (start + end) // 2
        begin = start if start == 0 else center - WINDOW_HOP_MS // 2
        stop = min(duration_ms, center + (WINDOW_HOP_MS + 1) // 2)
        if end + WINDOW_HOP_MS > duration_ms:
            stop = min(end, duration_ms)
        if stop <= begin:
            continue
        scores = {item["label"]: item["score"] for item in row["scores"]}
        result.append((begin, stop, scores))
    return result


def join_observations(observations, kind, basis):
    """Merge contiguous same-label evidence and keep the strongest model score."""
    result = []
    for begin, end, label, score in observations:
        last = result[-1] if result else None
        if last and last["label"] == label and begin <= last["endMs"]:
            last["endMs"] = max(last["endMs"], end)
            last["confidence"] = max(last["confidence"], round(score, 3))
            last["windows"] += 1
        else:
            result.append({"kind": kind, "label": label, "startMs": begin, "endMs": end,
                           "confidence": round(score, 3), "windows": 1, "basis": basis})
    return result


def sound_spans(rows, duration_ms, basis=APPLE_BASIS):
    """Detect sustained music and independently select recognizable non-speech sounds."""
    music, background = [], []
    cricket_min = .25 if basis == ZIPFORMER_BASIS else .48
    music_seed, music_single = (.60, .90) if basis == ZIPFORMER_BASIS else (MUSIC_SEED, .78)
    for begin, end, scores in observations(rows, duration_ms):
        score = scores.get("music", 0)
        if score >= MUSIC_MIN:
            music.append((begin, end, "music", score))
        candidates = [(score, label) for label, score in scores.items()
                      if label in BACKGROUND and score >= (cricket_min if label == "cricket_chirp" else .62)]
        if candidates:
            score, label = max(candidates)
            background.append((begin, end, label, score))
    music = [row for row in join_observations(music, "music", basis)
             if row["confidence"] >= music_seed and (row["windows"] >= 2 or row["confidence"] >= music_single)]
    return [*music, *join_observations(background, "background", basis)]


def analyze_audio_scene(wav, cues, offset_ms, overlaps, emit, classifier="auto"):
    """Reuse decoded PCM for both layers and translate the sound timestamps to media time."""
    backend = select_sound_classifier(classifier)
    tag = classify if backend == "apple" else classify_zipformer
    with stage(emit, "audio-layers") as progress:
        try:
            with wave.open(str(wav)) as audio:
                duration_ms = audio.getnframes() * 1000 // audio.getframerate()
            pitch = measure_intonation(wav, cues, offset_ms, overlaps)
        except (OSError, EOFError, wave.Error):
            progress["state"] = "failed"
            return {"audioLayers": [], "soundAnalysisStatus": "failed", "soundClassifier": backend}
        rows, status = tag(wav, duration_ms)
        sounds = sound_spans(rows, duration_ms, SOUND_BASES[backend])
        if status == "failed":
            progress["state"] = "failed"
        elif status != "done" and not cues:
            progress["state"] = "skipped"
        for sound in sounds:
            sound["startMs"] += offset_ms
            sound["endMs"] += offset_ms
        return {"audioLayers": sorted([*sounds, *pitch], key=lambda row: (row["startMs"], row["kind"])),
                "soundAnalysisStatus": status, "soundClassifier": backend}


def clean_audio_layers(rows, duration_ms):
    """Whitelist the fields used by UI/graph and reject invalid imported intervals."""
    if not isinstance(rows, list):
        return []
    clean = []
    for item in rows:
        if not isinstance(item, dict):
            continue
        kind, label = item.get("kind"), item.get("label")
        if not isinstance(label, str):
            continue
        allowed = (kind == "intonation" and label in INTONATION or kind == "music" and label == "music"
                   or kind == "background" and label in BACKGROUND)
        start, end = item.get("startMs"), item.get("endMs")
        if not allowed or type(start) is not int or type(end) is not int or not 0 <= start < end <= duration_ms:
            continue
        keys = ("pitchHz", "semitones") if kind == "intonation" else ("confidence", "windows")
        basis = ZIPFORMER_BASIS if item.get("basis") == ZIPFORMER_BASIS else APPLE_BASIS
        row = {"kind": kind, "label": label, "startMs": start, "endMs": end,
               "basis": "speech-f0" if kind == "intonation" else basis}
        for key in keys:
            value = item.get(key)
            if key == "pitchHz":
                if isinstance(value, list) and len(value) == 3 and all(type(hz) in (float, int) and 60 <= hz <= 480 for hz in value):
                    row[key] = value
            elif key == "windows" and type(value) is int and value >= 1:
                row[key] = value
            elif key == "confidence" and type(value) in (float, int) and 0 <= value <= 1:
                row[key] = value
            elif key == "semitones" and type(value) in (float, int) and -9 <= value <= 9:
                row[key] = value
        clean.append(row)
    return sorted(clean, key=lambda row: (row["startMs"], row["endMs"], row["kind"]))
