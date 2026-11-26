"""Build bounded speaker prompts for cross-language dubbing."""

from collections import defaultdict
from itertools import combinations
import logging
import subprocess
import wave

from voice_studio.synthesis.dub_background import RATE as BACKGROUND_RATE
from voice_studio.synthesis.engines import ffmpeg
from voice_studio.vision.face_reid import cosine
from voice_studio.speech.speaker_review import MAX_CLUSTERS, MODEL_PATH, extract_embeddings

RATE = 24_000
MIN_CLIP_MS = 800
MAX_CLIP_MS = 16_000
MAX_PROFILE_MS = 18_000
MAX_CANDIDATES = 6
MIN_AGREEMENT = .70
JOIN_MS = 120


def duration(cue):
    """Return the exact duration of a source cue in milliseconds."""
    return cue["endMs"] - cue["startMs"]


def reference_candidates(cues, offset, audio_ms):
    """Collect up to six fully available, unambiguous cues per reviewed speaker."""
    groups = defaultdict(list)
    for cue in cues:
        if (not cue.get("overlap") and offset <= cue["startMs"] and cue["endMs"] <= offset + audio_ms
                and MIN_CLIP_MS <= duration(cue) <= MAX_CLIP_MS):
            groups[cue["speaker"]].append(cue)
    for cue in cues:
        if not groups[cue["speaker"]]:
            raise ValueError(f"Для спикера {cue['speaker']} нужен чистый образец голоса от 0,8 до 16 секунд.")
    return {number: sorted(rows, key=duration, reverse=True)[:MAX_CANDIDATES] for number, rows in groups.items()}


def verified_clips(candidates, embeddings):
    """Pick the strongest same-label pair, using duration as a tiebreaker."""
    best_pair, best_score = None, None
    for left, right in combinations(candidates, 2):
        length = duration(left) + duration(right) + JOIN_MS
        first = embeddings.get(left["startMs"])
        second = embeddings.get(right["startMs"])
        if length > MAX_PROFILE_MS or not first or not second:
            continue
        agreement = cosine(first, second)
        score = (agreement, length)
        if agreement >= MIN_AGREEMENT and (best_score is None or score > best_score):
            best_pair, best_score = (left, right), score
    if best_pair is not None:
        return sorted(best_pair, key=lambda cue: cue["startMs"]), round(best_score[0], 3)
    return candidates[:1], None


def collect_embeddings(path, groups, offset):
    """Verify candidates only inside reviewed labels with multiple usable cues."""
    spans = {}
    for group in groups.values():
        if len(group) > 1:
            for cue in group:
                start = cue["startMs"]
                spans[start] = [(start - offset, cue["endMs"] - offset)]
    if not spans or not (MODEL_PATH / "embedding.onnx").is_file():
        return {}
    entries = list(spans.items())
    verified = {}
    for first in range(0, len(entries), MAX_CLUSTERS):
        batch = dict(entries[first:first + MAX_CLUSTERS])
        try:
            verified.update({stamp: vectors[0] for stamp, vectors in extract_embeddings(path, batch).items() if vectors})
        except (OSError, ValueError, subprocess.SubprocessError) as error:
            logging.getLogger(__name__).warning("Speaker reference verification unavailable: %s", error)
            break
    return verified


def isolate_reference_audio(original, background, output, offset):
    """Subtract the source-aligned accompaniment before extracting voice samples."""
    # The dialogue PCM starts at the first audio packet, while the bed starts at video zero.
    trim = f"atrim=start_sample={offset * BACKGROUND_RATE // 1000},asetpts=PTS-STARTPTS"
    filters = (f"[1:a]{trim},aresample={RATE},aformat=channel_layouts=mono,volume=-1[bed];"
               "[0:a][bed]amix=inputs=2:duration=first:normalize=0")
    ffmpeg(["-i", str(original), "-i", str(background), "-filter_complex", filters,
            "-c:a", "pcm_s16le", str(output)])


def write_prompt(audio, clips, offset, path):
    """Concatenate exact cue samples and their transcripts with a short quiet join."""
    with wave.open(str(path), "wb") as output:
        output.setparams((1, 2, RATE, 0, "NONE", "not compressed"))
        for index, cue in enumerate(clips):
            if index:
                output.writeframesraw(bytes(JOIN_MS * RATE // 1000 * 2))
            audio.setpos((cue["startMs"] - offset) * RATE // 1000)
            output.writeframesraw(audio.readframes(duration(cue) * RATE // 1000))
    phrases = []
    for cue in clips[:-1]:
        text = cue["sourceText"]
        phrases.append(text if text.endswith((".", "!", "?", "…")) else text + ".")
    return " ".join([*phrases, clips[-1]["sourceText"]])


def render_voice_profiles(original, background, cues, offset, folder):
    """Reuse the separated source dialogue for identity-checked Qwen prompts."""
    isolated = folder / "reference-dialogue.wav"
    isolate_reference_audio(original, background, isolated, offset)
    profiles = {}
    with wave.open(str(isolated), "rb") as audio, wave.open(str(original), "rb") as original_audio:
        groups = reference_candidates(cues, offset, audio.getnframes() * 1000 // RATE)
        vectors = collect_embeddings(isolated, groups, offset)
        for number, candidates in groups.items():
            clips, agreement = verified_clips(candidates, vectors)
            path = folder / f"speaker-{number}.wav"
            transcript = write_prompt(audio if agreement is not None else original_audio, clips, offset, path)
            profiles[number] = {"path": str(path), "text": transcript,
                                "startMs": clips[0]["startMs"], "endMs": clips[-1]["endMs"],
                                "clips": [{key: cue[key] for key in ("startMs", "endMs")} for cue in clips],
                                "voiceAgreement": agreement}
            if agreement is not None:
                fallback = folder / f"speaker-{number}-original.wav"
                text = write_prompt(original_audio, [candidates[0]], offset, fallback)
                profiles[number]["fallback"] = {"path": str(fallback), "text": text,
                                                "startMs": candidates[0]["startMs"], "endMs": candidates[0]["endMs"]}
    return profiles


def reference_report(profiles):
    """Expose the source spans and agreement in the export without reference file paths."""
    summary = {}
    for number, profile in profiles.items():
        if "profileVersion" in profile:
            summary[number] = {"profileVersion": profile["profileVersion"],
                               "recordedLanguage": profile["recordedLanguage"]}
            continue
        row = {key: profile[key] for key in ("startMs", "endMs", "clips", "voiceAgreement")}
        fallback = profile.get("fallback")
        row["fallback"] = {key: fallback[key] for key in ("startMs", "endMs")} if fallback else None
        summary[number] = row
    return summary
