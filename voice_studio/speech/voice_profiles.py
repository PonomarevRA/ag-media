"""Bounded, source-scoped acoustic identities with explicit speaking-video priority.

Embeddings remain in the caller's local registry/cache. Public reports expose only
provenance and matching scores; raw diarization labels are never rewritten.
"""

from voice_studio.mod_runtime import model_resource

from voice_studio.core.voice_registry_lock import serialize_voice_file
from array import array
from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from math import isfinite, sqrt
import os
from pathlib import Path
from statistics import median
import sys
from uuid import uuid4
import wave

from voice_studio.speech.voice_profile_evidence import audio_interval, overlaps_audio, speaking_evidence
from voice_studio.speech.voice_profile_public import ID as PUBLIC_PROFILE_ID

VERSION = "source-voice-profiles-1"


@dataclass(frozen=True)
class VoiceProfileConfig:
    """Conservative heuristics for local Sherpa embeddings, not calibrated probabilities."""

    max_profiles: int = 64
    max_exemplars: int = 8
    max_windows: int = 96
    min_duration_ms: int = 1200
    max_duration_ms: int = 4000
    min_match: float = .58
    min_margin: float = .10
    min_learn: float = .68
    new_identity_below: float = .38
    min_rms: float = .0005
    max_clipped_fraction: float = .02

    def __post_init__(self):
        """Reject policies that disable bounds or reverse the abstention interval."""
        if not (1 <= self.max_profiles <= 256 and 1 <= self.max_exemplars <= 32 and 1 <= self.max_windows <= 512):
            raise ValueError("Voice profile budgets are outside supported bounds")
        if not (1000 <= self.min_duration_ms <= self.max_duration_ms <= 10000):
            raise ValueError("Invalid voice sample duration policy")
        if not (-1 <= self.new_identity_below < self.min_match <= self.min_learn <= 1 and 0 <= self.min_margin <= 2):
            raise ValueError("Invalid voice matching policy")
        if not (0 <= self.min_rms <= 1 and 0 <= self.max_clipped_fraction <= 1):
            raise ValueError("Invalid voice quality policy")


def _fingerprint(value):
    """Hash deterministic JSON for cache policy and local model provenance."""
    return sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def local_encoder_id():
    """Identify existing weights by content; never fetch a model."""
    from voice_studio.speech.speaker_diarization import MODEL_PATH
    path = model_resource('speech.embedding-resource', 'builtin.sherpa-embedding', MODEL_PATH / 'embedding.onnx')
    digest = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return f"sherpa-3d-speaker:{digest.hexdigest()}"


def _unit(vector):
    """Validate finite nonzero features and normalize without silent dimension truncation."""
    if not isinstance(vector, (list, tuple)) or not all(type(value) in (int, float) for value in vector):
        return None
    values = [float(value) for value in vector]
    if not values or not all(isfinite(value) for value in values):
        return None
    norm = sqrt(sum(value * value for value in values))
    return [value / norm for value in values] if norm > 0 else None


def _cosine(left, right):
    """Compare only features of the same encoder dimension."""
    if len(left) != len(right):
        return -1.0
    return sum(a * b for a, b in zip(left, right))


class VoiceProfileRegistry:
    """Own bounded source-local identities; one writer must serialize cache updates."""

    def __init__(self, source_id, *, encoder_id=None, config=None):
        """Initialize a registry for one immutable source/upload identity."""
        if not source_id:
            raise ValueError("Voice profiles require an immutable source_id")
        self.source_id = str(source_id)
        self.encoder_id = encoder_id or local_encoder_id()
        self.config = config or VoiceProfileConfig()
        self.profiles = []
        self.dimension = None

    def _rank(self, vector):
        """Use median of best three references, penalizing a disagreeing lower quartile."""
        ranks = []
        for profile in self.profiles:
            scores = sorted((_cosine(vector, sample["vector"]) for sample in profile["samples"]), reverse=True)
            if scores:
                score = median(scores[:3])
                if len(scores) >= 3:
                    score = min(score, scores[(len(scores) * 3) // 4] + .15)
                ranks.append((score, profile))
        ranks.sort(key=lambda row: (-row[0], row[1]["id"]))
        return ranks

    def _new(self, subject=None):
        """Create one tentative identity within the explicit registry bound."""
        if len(self.profiles) >= self.config.max_profiles:
            return None
        profile = {"id": uuid4().hex, "number": len(self.profiles) + 1, "subjectId": subject, "visualSubjects": [], "samples": []}
        self.profiles.append(profile)
        return profile

    def _learn(self, profile, vector, provenance, explicit=False):
        """Retain clean diverse samples; exact or near-duplicate excerpts add no weight."""
        samples = profile["samples"]
        for sample in samples:
            if sample["key"] == provenance["key"]:
                if explicit:
                    sample["explicitVideo"] = True
                    sample["videoEvidence"] = provenance.get("videoEvidence")
                return False
            # Adjacent same-source spans and nearly identical vectors are not independent proof.
            gap = max(sample["startMs"], provenance["startMs"]) - min(sample["endMs"], provenance["endMs"])
            if gap < 500 or _cosine(vector, sample["vector"]) > .985:
                return False
        candidate = {**provenance, "vector": vector, "explicitVideo": explicit}
        if len(samples) < self.config.max_exemplars:
            samples.append(candidate)
            return True
        # Keep visually grounded examples; replace only a redundant ungrounded sample.
        replacements = []
        for index, sample in enumerate(samples):
            if sample["explicitVideo"]:
                continue
            others = [other for i, other in enumerate(samples) if i != index]
            redundancy = max((_cosine(sample["vector"], other["vector"]) for other in others), default=-1)
            novelty = max((_cosine(vector, other["vector"]) for other in others), default=-1)
            if explicit or novelty < redundancy - .02:
                replacements.append((redundancy, index))
        if replacements:
            samples[max(replacements)[1]] = candidate
            return True
        return False

    def observe(self, vector, provenance, *, video=None, quality=None):
        """Resolve one clean cue, prioritizing explicit talker evidence over acoustic clusters.

        provenance contains source-clock startMs/endMs, key, windowId and rawSpeaker.
        Passing quality.accepted=False can still attach a visual identity, but never
        learns or acoustically matches. Ambiguous audio neither learns nor creates.
        """
        if provenance.get("sourceId") != self.source_id:
            raise ValueError("Voice observation belongs to a different source")
        quality = quality or {"accepted": True, "reasons": []}
        unit = _unit(vector) if vector is not None and quality.get("accepted") else None
        if vector is not None and quality.get("accepted") and unit is None:
            quality = {**quality, "accepted": False, "reasons": ["invalid-embedding"]}
        if unit and self.dimension is not None and len(unit) != self.dimension:
            raise ValueError("Voice encoder dimension changed")
        if unit:
            self.dimension = len(unit)
        ranks = self._rank(unit) if unit else []
        score = ranks[0][0] if ranks else None
        margin = score - ranks[1][0] if len(ranks) > 1 else None
        unique = len(ranks) == 1 or (margin is not None and margin >= self.config.min_margin)
        match = ranks[0][1] if ranks and score >= self.config.min_match and unique else None
        basis, profile = "audio-abstain", None
        reasons = list(quality.get("reasons", []))
        if len(ranks) == 1:
            reasons.append("single-profile-no-runner-up")
        if video:
            subject = video["subjectId"]
            scope = video.get("visualScope") if video.get("identityScope") == "window-local" else None
            profile = next((p for p in self.profiles if p["subjectId"] == subject
                            or any(s["id"] == subject for s in p["visualSubjects"])), None)
            if profile is None and match:
                # A fresh window's face ID is not global identity. Strong acoustic
                # agreement can connect it unless this window already proves another face.
                conflicting = any(s["scope"] == scope and s["id"] != subject for s in match["visualSubjects"])
                has_global_subject = any(s["scope"] is None for s in match["visualSubjects"])
                if (not match["visualSubjects"] or (scope is not None and not conflicting)
                        or (scope is None and not has_global_subject)):
                    profile = match
            if profile is None:
                profile = self._new(subject)
            if profile is not None:
                profile["subjectId"] = profile["subjectId"] or subject
                observation = {"id": subject, "scope": scope}
                if observation not in profile["visualSubjects"]:
                    # Visual alias history is bounded independently from acoustic samples.
                    profile["visualSubjects"].append(observation)
                    if len(profile["visualSubjects"]) > 32:
                        oldest_local = next((i for i, s in enumerate(profile["visualSubjects"]) if s["scope"] is not None), 0)
                        del profile["visualSubjects"][oldest_local]
            basis = "explicit-video"
            if match and profile and match["id"] != profile["id"]:
                reasons.append("explicit-video-overrides-audio")
        elif match:
            profile, basis = match, "robust-audio-match"
        elif unit and (score is None or score < self.config.new_identity_below):
            profile, basis = self._new(), "new-audio-profile"
        elif unit:
            reasons.append("audio-score-or-margin-insufficient")
        learned = False
        if profile and unit and (video or basis == "new-audio-profile" or score >= self.config.min_learn):
            sample_provenance = {**provenance, "videoEvidence": video} if video else provenance
            learned = self._learn(profile, unit, sample_provenance, explicit=bool(video))
        if profile is None and len(self.profiles) >= self.config.max_profiles:
            reasons.append("profile-budget-exhausted")
        return {"profileId": profile["id"] if profile else None, "basis": basis,
                "state": "matched" if profile else "abstained", "score": round(score, 5) if score is not None else None,
                "margin": round(margin, 5) if margin is not None else None, "learned": learned,
                "audioCandidateProfileId": ranks[0][1]["id"] if ranks else None,
                "quality": quality, "reasons": reasons, "video": video, **provenance}

    def index(self):
        """Return a vector-free public identity index with source-clock reference intervals."""
        profiles = []
        for profile in self.profiles:
            refs = [{key: value for key, value in sample.items() if key != "vector"} for sample in profile["samples"]]
            profiles.append({"id": profile["id"], "number": profile["number"], "subjectId": profile["subjectId"],
                             "visualSubjects": [dict(subject) for subject in profile["visualSubjects"]],
                             "sampleCount": len(refs), "references": refs,
                             "status": "video-grounded" if profile["subjectId"] else "audio-provisional"})
        from voice_studio.speech.voice_profile_public import public_voice_profile
        return [public_voice_profile(row) for row in profiles]

    def save(self, path):
        """Atomically replace one local source-scoped cache; cache contains acoustic features."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"version": VERSION, "sourceId": self.source_id, "encoderId": self.encoder_id,
                   "config": asdict(self.config), "dimension": self.dimension, "profiles": self.profiles}
        temporary = path.with_name(path.name + "." + uuid4().hex + ".tmp")
        try:
            with temporary.open("x", encoding="utf-8") as stream:
                os.chmod(temporary, 0o600)
                json.dump(payload, stream, allow_nan=False)
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)

    @classmethod
    def load(cls, path, source_id, *, encoder_id=None, config=None):
        """Load bounded validated cache state, rejecting changed source, encoder or policy."""
        cache = Path(path)
        if cache.stat().st_size > 16 * 1024 * 1024:
            raise ValueError("Voice profile cache exceeds byte budget")
        payload = json.loads(cache.read_text())
        if not isinstance(payload, dict):
            raise ValueError("Invalid voice profile cache document")
        registry = cls(source_id, encoder_id=encoder_id, config=config)
        if (payload.get("version") != VERSION or payload.get("sourceId") != registry.source_id
                or payload.get("encoderId") != registry.encoder_id or payload.get("config") != asdict(registry.config)):
            raise ValueError("Voice profile cache source, encoder or policy mismatch")
        profiles = payload.get("profiles", [])
        if not isinstance(profiles, list) or len(profiles) > registry.config.max_profiles:
            raise ValueError("Voice profile cache exceeds profile budget")
        ids, subjects, aliases = set(), set(), set()
        dimension = payload.get("dimension")
        for profile in profiles:
            if (not isinstance(profile, dict) or not isinstance(profile.get("id"), str)
                    or not PUBLIC_PROFILE_ID.fullmatch(profile['id'])
                    or not isinstance(profile.get("visualSubjects"), list)
                    or not isinstance(profile.get("samples"), list)
                    or type(profile.get("number")) is not int):
                raise ValueError("Invalid voice profile cache fields")
            if (profile["id"] in ids or profile["number"] != len(ids) + 1
                    or len(profile["visualSubjects"]) > 32 or len(profile["samples"]) > registry.config.max_exemplars):
                raise ValueError("Invalid voice profile cache identity or sample count")
            ids.add(profile["id"])
            subject = profile.get("subjectId")
            if subject is not None and not isinstance(subject, str):
                raise ValueError("Invalid cached visual identity")
            if subject is not None and subject in subjects:
                raise ValueError("Duplicate visual identity in voice profile cache")
            subjects.add(subject)
            for alias in profile["visualSubjects"]:
                if not isinstance(alias, dict) or not isinstance(alias.get("id"), str):
                    raise ValueError("Invalid cached visual alias")
                if alias["id"] in aliases:
                    raise ValueError("Duplicate cached visual alias")
                aliases.add(alias["id"])
            for sample in profile["samples"]:
                if not isinstance(sample, dict) or not isinstance(sample.get("vector"), list):
                    raise ValueError("Invalid cached voice sample")
                if (not isinstance(sample.get('key'), str) or not sample['key']
                        or type(sample.get('startMs')) is not int or type(sample.get('endMs')) is not int
                        or not 0 <= sample['startMs'] < sample['endMs']
                        or type(sample.get('explicitVideo')) is not bool
                        or type(dimension) is not int or not 1 <= dimension <= 4096):
                    raise ValueError('Invalid cached reference provenance')
                vector = _unit(sample["vector"])
                if vector is None or len(vector) != dimension:
                    raise ValueError("Invalid cached voice vector")
                sample["vector"] = vector
        registry.profiles, registry.dimension = profiles, dimension
        return registry


def _pcm_quality(pcm, config):
    """Gate silence and gross clipping; this is not a learned speech/music classifier."""
    if not pcm:
        return {"accepted": False, "reasons": ["empty-audio"]}
    rms = sqrt(sum((value / 32768) ** 2 for value in pcm) / len(pcm))
    clipped = sum(abs(value) >= 32700 for value in pcm) / len(pcm)
    reasons = []
    if rms < config.min_rms:
        reasons.append("low-energy")
    if clipped > config.max_clipped_fraction:
        reasons.append("clipped-audio")
    return {"accepted": not reasons, "reasons": reasons, "rms": round(rms, 6), "clippedFraction": round(clipped, 6)}


def _extract(audio, start, end, config, extractor):
    """Read bounded mono PCM and infer only usable local windows using existing weights."""
    rate = audio.getframerate()
    first, last = round(start * rate / 1000), round(end * rate / 1000)
    if first < 0 or last > audio.getnframes():
        return None, {"accepted": False, "reasons": ["outside-audio-window"]}
    audio.setpos(first)
    pcm = array("h", audio.readframes(last - first))
    if sys.byteorder != "little":
        pcm.byteswap()
    quality = _pcm_quality(pcm, config)
    if not quality["accepted"]:
        return None, quality
    stream = extractor.create_stream()
    stream.accept_waveform(rate, array("f", (value / 32768 for value in pcm)))
    stream.input_finished()
    if not extractor.is_ready(stream):
        return None, {**quality, "accepted": False, "reasons": ["encoder-not-ready"]}
    vector = [float(value) for value in extractor.compute(stream)]
    if _unit(vector) is None:
        return None, {**quality, "accepted": False, "reasons": ["invalid-embedding"]}
    return vector, quality


@serialize_voice_file
def apply_voice_profiles(result, wav, audio_start_ms, *, source_id, registry=None,
                         cache_path=None, window_id=None, visual_scope=None, extractor=None):
    """Attach source-stable IDs and explicit evidence to audio or audiovisual cues.

    Times in cues use source milliseconds; WAV zero corresponds to audio_start_ms.
    Caller owns immutable source_id, serial registry/cache access and stable visual
    subject scope. Errors remain visible to the caller; no weights are downloaded.
    """
    if registry is None:
        if cache_path and Path(cache_path).is_file():
            registry = VoiceProfileRegistry.load(cache_path, source_id)
        else:
            registry = VoiceProfileRegistry(source_id)
    if registry.source_id != str(source_id):
        raise ValueError("Cannot reuse a voice registry across sources")
    cues = result.get("segments", [])
    candidates = []
    for index, cue in enumerate(cues):
        start, end = audio_interval(cue)
        if not overlaps_audio(cue, result) and end - start >= registry.config.min_duration_ms:
            candidates.append(index)
    # Sampling across the whole window avoids exhausting the budget on its first speaker.
    budget = registry.config.max_windows
    if len(candidates) > budget:
        candidates = [candidates[i * len(candidates) // budget] for i in range(budget)]
    selected = set(candidates)
    decisions = []
    with wave.open(str(wav)) as audio:
        if audio.getnchannels() != 1 or audio.getsampwidth() != 2:
            raise ValueError("Voice profiling requires mono 16-bit PCM WAV")
        for index, cue in enumerate(cues):
            start, end = audio_interval(cue)
            end = min(end, start + registry.config.max_duration_ms)
            provenance = {"key": _fingerprint([source_id, start, end]), "sourceId": str(source_id),
                          "windowId": window_id, "startMs": start, "endMs": end, "rawSpeaker": cue.get("speaker")}
            video = speaking_evidence(cue, result, visual_scope)
            if overlaps_audio(cue, result):
                decision = {"profileId": None, "state": "abstained", "basis": "overlap", "learned": False,
                            "reasons": ["overlap"], "quality": {"accepted": False, "reasons": ["overlap"]}, **provenance}
            else:
                vector = None
                excluded_reason = "short-audio"
                if end - start >= registry.config.min_duration_ms:
                    excluded_reason = "sample-budget"
                quality = {"accepted": False, "reasons": [excluded_reason]}
                if index in selected:
                    if extractor is None:
                        from voice_studio.speech.speaker_review import embedding_extractor
                        from voice_studio.speech.speaker_diarization import MODEL_PATH
                        extractor = embedding_extractor(MODEL_PATH / "embedding.onnx")
                    vector, quality = _extract(audio, start - audio_start_ms, end - audio_start_ms, registry.config, extractor)
                decision = registry.observe(vector, provenance, video=video, quality=quality)
            cue.pop("voiceProfileId", None)
            if decision["profileId"]:
                cue["voiceProfileId"] = decision["profileId"]
            cue["voiceProfileEvidence"] = decision
            decisions.append(decision)
    if cache_path:
        registry.save(cache_path)
    report = {"version": VERSION, "sourceId": str(source_id), "encoderId": registry.encoder_id,
              "config": asdict(registry.config), "profiles": registry.index(), "analyzedWindows": len(selected),
              "matchedCues": sum(row["state"] == "matched" for row in decisions),
              "abstainedCues": sum(row["state"] == "abstained" for row in decisions),
              "limits": ["heuristic thresholds; audio identities remain provisional", "no speech/music or denoising model"]}
    result["voiceProfiles"] = report
    return report
