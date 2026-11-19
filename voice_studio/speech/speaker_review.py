"""Review provisional Sherpa clusters against local voice and active-face references."""

from voice_studio.mod_runtime import current_snapshot, model_resource, worker_environment

from voice_studio.mod_runtime import model_provider

from voice_paths import PROJECT_ROOT

from array import array
from collections import Counter, defaultdict
from dataclasses import dataclass
import json
import os
from pathlib import Path
import subprocess
import sys
import wave

from voice_studio.vision.face_reid import cosine, unit_vector
from voice_studio.vision.face_speakers import active_face
from voice_studio.speech.short_reply_review import find_replies, review_short_replies
from voice_studio.speech.speaker_diarization import MODEL_PATH, detect_turns, speaker_for
from voice_studio.speech.voiced_speaker_review import review_voiced_windows

MAX_SAMPLES = 72
MAX_CLUSTERS = 24
REVIEW_VERSION = "av-profile-1"


@dataclass
class VoiceProfile:
    number: int
    face: int | None
    time_ms: int
    samples: list
    conflicts: bool


def anchor_spans(cues, audio_start_ms):
    """Select at most three clean non-overlapping examples per provisional label."""
    candidates = defaultdict(list)
    for cue in cues:
        number = cue.get("speaker")
        start, end = cue["startMs"] - audio_start_ms, cue["endMs"] - audio_start_ms
        if number is not None and not cue.get("overlap") and start >= 0 and end - start >= 950:
            candidates[number].append((min(end - start, 3600), start))
    best = sorted(candidates, key=lambda number: -sum(duration for duration, _ in candidates[number]))
    return {number: [[start, start + duration] for duration, start in sorted(candidates[number], reverse=True)[:3]]
            for number in best[:MAX_CLUSTERS]}


@model_provider('speech.embedding', 'builtin.sherpa-embedding')
def extract_embeddings(wav, spans):
    """Compute references with the selected provider and bounded native worker mode."""
    from voice_studio.speech.embedding_protocol import file_signature, validate_spans
    validate_spans(spans)
    mode = os.environ.get('VOICE_SPEAKER_EMBEDDING_MODE', 'resident')
    if mode not in ('oneshot', 'resident'):
        raise ValueError('VOICE_SPEAKER_EMBEDDING_MODE must be oneshot or resident')
    if mode == 'resident':
        from voice_studio.speech.embedding_worker import get_worker
        weights = model_resource('speech.embedding-resource', 'builtin.sherpa-embedding', MODEL_PATH / 'embedding.onnx').resolve()
        identity = {'signature': file_signature(weights), 'threads': 2,
                    'configuration': current_snapshot().fingerprint(['speech.embedding', 'speech.embedding-resource'])}
        timeout = float(os.environ.get('VOICE_SPEAKER_EMBEDDING_TIMEOUT_SECONDS', '35'))
        return get_worker().extract(wav, spans, weights, identity, timeout=timeout)
    proc = subprocess.run([sys.executable, str(PROJECT_ROOT / "speaker_review.py"), str(wav)], input=json.dumps(spans),
                          env=worker_environment(), stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, timeout=35, check=True)
    return {int(number): vectors for number, vectors in json.loads(proc.stdout).items()}


def embedding_extractor(weights):
    """Resolve the legacy caller's frozen model resource, then open its native encoder."""
    weights = model_resource('speech.embedding-resource', 'builtin.sherpa-embedding', weights)
    return native_embedding_extractor(weights)


def native_embedding_extractor(weights):
    """Open already-resolved weights on two CPU threads without consulting child mods."""
    import sherpa_onnx

    return sherpa_onnx.SpeakerEmbeddingExtractor(sherpa_onnx.SpeakerEmbeddingExtractorConfig(
        str(weights), num_threads=2))


def infer_embeddings(wav, spans, extractor=None):
    """Read only requested PCM windows and reuse one 3D-Speaker extractor."""
    model = extractor if extractor is not None else embedding_extractor(MODEL_PATH / "embedding.onnx")
    output = {}
    with wave.open(str(wav)) as audio:
        rate = audio.getframerate()
        for number, windows in spans.items():
            features = []
            for start, end in windows:
                left = min(audio.getnframes(), start * rate // 1000)
                right = min(audio.getnframes(), end * rate // 1000)
                audio.setpos(left)
                pcm = array("h", audio.readframes(right - left))
                if sys.byteorder != "little":
                    pcm.byteswap()
                stream = model.create_stream()
                stream.accept_waveform(rate, array("f", (sample / 32768 for sample in pcm)))
                stream.input_finished()
                if model.is_ready(stream):
                    features.append(unit_vector([float(value) for value in model.compute(stream)]))
            output[number] = [feature for feature in features if feature]
    return output


def build_profiles(cues, timeline, embeddings):
    """Find recurring independent visual evidence and retain conflicting active faces."""
    face_votes = defaultdict(Counter)
    first = {}
    for cue in cues:
        number = cue.get("speaker")
        if number is None:
            continue
        first.setdefault(number, cue["startMs"])
        face = active_face(cue, timeline)
        if face is not None:
            duration = min(2200, cue["endMs"] - cue["startMs"])
            face_votes[number][face] += duration if cue.get("visualFace") is not None else duration // 2
    profiles = {}
    for number, first_ms in first.items():
        supported = [face for face, duration in face_votes[number].items() if duration >= 600]
        profiles[number] = VoiceProfile(number, supported[0] if len(supported) == 1 else None,
                                        first_ms, embeddings.get(number, []), len(supported) > 1)
    return profiles


def likelihood(left, right):
    """Measure robust cross-reference agreement instead of relying on a single pair."""
    pairs = sorted(cosine(a, b) for a in left.samples for b in right.samples)
    if not pairs:
        return None
    return pairs[len(pairs) // 2]


def validate_pairs(profiles, overlaps):
    """Accept disjoint mutual matches with independent visual or repeated voice support."""
    from itertools import combinations

    # Whole-cluster reassignment must preserve the published source labels of mixed speech.
    concurrent = {number for span in overlaps for number in span["speakers"]}
    suggestions = []
    candidates = defaultdict(list)
    for left, right in combinations(profiles.values(), 2):
        if left.conflicts or right.conflicts or left.number in concurrent or right.number in concurrent:
            continue
        if left.face is not None and right.face is not None and left.face != right.face:
            continue
        score = likelihood(left, right)
        if score is None:
            continue
        same_face = left.face is not None and left.face == right.face
        samples = min(len(left.samples), len(right.samples))
        # Unseen short fragments need stronger corroboration than two anchored views of one face.
        if (same_face and score >= .72) or (samples >= 2 and score >= .88):
            candidates[left.number].append((score, right.number, same_face))
            candidates[right.number].append((score, left.number, same_face))
    leaders = {}
    for number, rows in candidates.items():
        rows.sort(reverse=True)
        if len(rows) == 1 or rows[0][0] - rows[1][0] >= .055:
            leaders[number] = rows[0]
    for number, (score, other, same_face) in leaders.items():
        if number >= other or leaders.get(other, (None, None))[1] != number:
            continue
        first, second = sorted((number, other), key=lambda value: (profiles[value].time_ms, value))
        suggestions.append({"from": second, "to": first, "voiceCosine": round(score, 3),
                            "basis": "same-active-face-and-voice" if same_face else "repeated-voice"})
    return suggestions


def review_visible_cues(cues, profiles, turns, audio_start_ms, merged):
    """Use the second clustering pass only inside individually corroborated visual cues."""
    groups = defaultdict(list)
    for cue in cues:
        raw, face = cue.get("speaker"), cue.get("visualFace")
        if (raw in merged or raw not in profiles or profiles[raw].face != face or face is None
                or cue.get("overlap") or cue["endMs"] - cue["startMs"] < 950):
            continue
        number, mixed = speaker_for(turns, cue["startMs"] - audio_start_ms, cue["endMs"] - audio_start_ms)
        if number is not None and not mixed:
            groups[face, number].append(cue)
    decisions = []
    for (face, _), rows in groups.items():
        numbers = list(dict.fromkeys(cue["speaker"] for cue in rows))
        if len(numbers) != 2:
            continue
        owner, other = numbers
        score = likelihood(profiles[owner], profiles[other])
        if score is None or score < .47:
            continue
        for cue in rows:
            if cue["speaker"] == other:
                cue["profileSpeakerNumber"] = owner
                decisions.append({"startMs": cue["startMs"], "from": other, "to": owner,
                                  "face": face, "voiceCosine": round(score, 3)})
    return decisions


def review_speakers(result, wav, audio_start_ms, *, alternate=True):
    """Run the independent evidence pass without changing raw Sherpa cluster labels."""
    evidence = result.get("videoEvidence", {})
    timeline = evidence.get("timeline")
    if not timeline:
        return None
    cues = result["segments"]
    replies = find_replies(cues, evidence, audio_start_ms)
    spans = anchor_spans(cues, audio_start_ms)
    enough_clusters = len(spans) >= 2
    if not enough_clusters and not replies:
        return None
    embeddings = extract_embeddings(wav, spans) if enough_clusters else {}
    profiles = build_profiles(cues, timeline, embeddings) if enough_clusters else {}
    merges = validate_pairs(profiles, result.get("speechOverlaps", []))
    aliases = {row["from"]: row["to"] for row in merges}
    for cue in result["segments"]:
        if cue.get("speaker") in aliases:
            cue["profileSpeakerNumber"] = aliases[cue["speaker"]]
    known_faces = [profile.face for profile in profiles.values() if profile.face is not None]
    needs_alternate = alternate and len(set(known_faces)) < len(known_faces)
    reviewed = []
    if needs_alternate:
        try:
            turns = detect_turns(wav, None, merge_similar=True)
            reviewed = review_visible_cues(result["segments"], profiles, turns, audio_start_ms, aliases)
        except ValueError:
            needs_alternate = False
    restored = review_short_replies(cues, replies, wav, audio_start_ms, extract_embeddings)
    try:
        voiced = review_voiced_windows(cues, timeline, result.get("speechOverlaps", []),
                                      wav, audio_start_ms, extract_embeddings)
    except (OSError, ValueError, subprocess.SubprocessError):
        voiced = []
    if not enough_clusters and not restored and not voiced:
        return None
    report = {"version": REVIEW_VERSION, "analyzedClusters": len(profiles),
              "voiceSamples": sum(len(profile.samples) for profile in profiles.values()), "merges": merges,
              "secondClustering": needs_alternate, "cueDecisions": reviewed, "shortReplyDecisions": restored, "voicedWindowDecisions": voiced}
    result["speakerReview"] = report
    return report


if __name__ == "__main__":
    if sys.argv[1:] == ['--embedding-worker']:
        from voice_studio.speech.embedding_worker_server import serve
        serve()
    else:
        spans = json.load(sys.stdin)
        if len(spans) > MAX_CLUSTERS or sum(map(len, spans.values())) > MAX_SAMPLES:
            raise ValueError("Too many profile references")
        print(json.dumps(infer_embeddings(Path(sys.argv[1]), spans), allow_nan=False))
