"""Conservative static/frozen face-crop veto for unverified visual inferences.

Exact repeated normalized pixels are a risk signal, not an avatar detector or
proof that a person is absent. Motion, crop jitter or missing signatures abstain.
"""

from bisect import bisect_right
from collections import defaultdict
from hashlib import sha256
import re

SIGNATURE_VERSION = 'face-crop-rgb32-v1'
MIN_SPAN_MS = 10000
MIN_SAMPLES = 10
MAX_SAMPLE_GAP_MS = 1500
MAX_TRACKS = 256
MAX_SAMPLES = 50000


def face_crop_signature(frame, detection):
    """Hash a small face crop from an already decoded RGB/BGR frame; store no pixels.

    No landmarks or expression predictions affect the signature. Exact equality
    is deliberately strict: noisy live video and one-pixel crop jitter abstain.
    """
    import cv2
    import numpy as np

    if (not isinstance(frame, np.ndarray) or frame.dtype != np.uint8
            or frame.ndim != 3 or frame.shape[2] != 3):
        return None
    box = np.asarray(detection[:4], dtype=np.float64)
    if box.shape != (4,) or not np.isfinite(box).all() or min(box[2:]) < 32:
        return None
    x, y, width, height = (int(round(value)) for value in box)
    frame_height, frame_width = frame.shape[:2]
    if x < 0 or y < 0 or x + width > frame_width or y + height > frame_height:
        return None
    crop = frame[y:y + height, x:x + width]
    normalized = cv2.resize(crop, (32, 32), interpolation=cv2.INTER_AREA)
    return {'version': SIGNATURE_VERSION, 'sha256': sha256(normalized.tobytes()).hexdigest(),
            'basis': 'decoded-face-crop', 'normalizedWidth': 32, 'normalizedHeight': 32,
            'channels': 3, 'comparison': 'exact-normalized-pixels'}


def valid_signature(sample):
    """Accept only the bounded worker descriptor; missing/legacy evidence abstains."""
    value = sample.get('faceCropSignature')
    if (not isinstance(value, dict) or value.get('version') != SIGNATURE_VERSION
            or value.get('basis') != 'decoded-face-crop' or value.get('comparison') != 'exact-normalized-pixels'
            or value.get('normalizedWidth') != 32 or value.get('normalizedHeight') != 32
            or value.get('channels') != 3 or not isinstance(value.get('sha256'), str)
            or not re.fullmatch(r'[a-f0-9]{64}', value['sha256'])):
        return None
    return value['sha256']


def annotate_static_faces(samples, duration, cuts=()):
    """Retain model candidates and flag only sustained exact face-crop repeats.

    Samples must already be validated and time-sorted. Group by both track and
    face ID; gaps, source cuts, missing signatures and content changes reset a
    run. Exclusions are retrospective sample cells, clipped to source duration.
    """
    cut_times = sorted({value for value in cuts if type(value) is int and 0 <= value <= duration})
    groups = defaultdict(list)
    skipped = 0
    signatures = 0
    for sample in samples[:MAX_SAMPLES]:
        key = sample['track'], sample['faceId']
        if key not in groups and len(groups) >= MAX_TRACKS:
            skipped += 1
            continue
        groups[key].append(sample)
        signatures += valid_signature(sample) is not None
    skipped += max(0, len(samples) - MAX_SAMPLES)
    exclusions = []

    def save_run(run, track, face):
        """Publish a veto only after both independent count and elapsed-time gates."""
        if len(run) < MIN_SAMPLES:
            return
        start, last = run[0]['timeMs'], run[-1]['timeMs']
        if last - start < MIN_SPAN_MS:
            return
        # The same forward sample-cell convention is used by expression_intervals.
        end = min(duration, last + 480)
        next_cut = bisect_right(cut_times, last)
        if next_cut < len(cut_times):
            end = min(end, cut_times[next_cut])
        exclusions.append({'startMs': start, 'endMs': end, 'track': track, 'faceId': face,
                           'status': 'suspected-static-or-frozen', 'basis': 'exact-normalized-face-crop-repeat',
                           'sampleCount': len(run), 'spanMs': last - start,
                           'reactionAggregateEligible': False, 'strongAVSpeakerAnchorEligible': False,
                           'avatarClassification': 'unknown', 'livePersonPresence': 'unknown'})

    for (track, face), rows in groups.items():
        run, previous_hash, previous_cut = [], None, None
        for sample in rows:
            signature = valid_signature(sample)
            cut = bisect_right(cut_times, sample['timeMs'])
            continuous = bool(run) and 0 < sample['timeMs'] - run[-1]['timeMs'] <= MAX_SAMPLE_GAP_MS
            if not signature or signature != previous_hash or not continuous or cut != previous_cut:
                save_run(run, track, face)
                run = []
            if signature:
                run.append(sample)
            previous_hash, previous_cut = signature, cut
        save_run(run, track, face)
    exclusions.sort(key=lambda row: (row['startMs'], row['track'], row['faceId']))
    by_key = defaultdict(list)
    for span in exclusions:
        by_key[span['track'], span['faceId']].append(span)
    annotated = []
    for sample in samples:
        static = any(span['startMs'] <= sample['timeMs'] < span['endMs']
                     for span in by_key.get((sample['track'], sample['faceId']), ()))
        annotated.append({**sample, 'evidenceStatus': 'model-candidate', 'verificationStatus': 'unverified',
                          'internalStateInference': False, 'reactionEligible': not static,
                          'staticFaceStatus': 'suspected-static-or-frozen' if static else 'unknown'})
    summary = {'version': 'static-face-evidence-v1', 'status': 'partial' if skipped else 'ok',
               'basis': 'exact-normalized-face-crop-repeat', 'minSpanMs': MIN_SPAN_MS,
               'minSamples': MIN_SAMPLES, 'maxSampleGapMs': MAX_SAMPLE_GAP_MS,
               'maxTracks': MAX_TRACKS, 'maxSamples': MAX_SAMPLES, 'skippedSamples': skipped,
               'samplesWithSignature': signatures, 'exclusions': exclusions,
               'risk': 'Static or frozen pixels can be an avatar, a paused feed, or a still live person; identity and liveness remain unknown.',
               'missingSignaturePolicy': 'unknown-no-veto',
               'verificationStatus': 'unverified', 'internalStateInference': False}
    return annotated, summary


def strong_av_anchor_eligible(layer, face_id, time_ms):
    """Return whether the static guard permits an inferred AV anchor at source time.

    True means absence of this veto, never positive speech evidence. Explicit
    human/editorial speech confirmation is a separate higher-priority fact.
    """
    if not isinstance(layer, dict) or type(time_ms) is not int:
        return True
    guard = layer.get('staticFaceEvidence') or {}
    for span in guard.get('exclusions', []):
        if (span.get('faceId') == face_id and span.get('strongAVSpeakerAnchorEligible') is False
                and type(span.get('startMs')) is int and type(span.get('endMs')) is int
                and span['startMs'] <= time_ms < span['endMs']):
            return False
    return True
