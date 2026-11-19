"""Find reviewable dialogue defects without editing ASR or inferring identities.

All intervals are half-open milliseconds on the original media clock. Speaker
IDs are opaque acoustic labels, never people. This module performs no I/O.
"""

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
import math
import re

SCHEMA_VERSION = 'dialogue-audit-v1'
ALGORITHM_VERSION = '1.1.0'


@dataclass(frozen=True)
class AuditOptions:
    """Thresholds for candidate discovery, rather than guarantees of speech quality."""
    max_input_rows: int = 50000
    max_candidates: int = 2000
    max_turns_per_cue: int = 256
    min_speaker_ms: int = 400
    min_missing_ms: int = 900
    sparse_min_voiced_ms: int = 4000
    sparse_words_per_second: float = .85
    edge_drift_ms: int = 1600
    max_snippet_ms: int = 6500
    min_snippet_ms: int = 650

    def __post_init__(self):
        """Reject unbounded or contradictory thresholds before discovery."""
        values = asdict(self)
        for key, value in values.items():
            if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value) or value <= 0:
                raise ValueError(f'{key} must be positive and finite')
        for key in values:
            if key != 'sparse_words_per_second' and type(values[key]) is not int:
                raise ValueError(f'{key} must be an integer')
        if self.min_snippet_ms > self.max_snippet_ms or self.max_snippet_ms > 30000:
            raise ValueError('snippet bounds must satisfy 0 < min <= max <= 30000 ms')


def stable_hash(value):
    """Hash JSON by value so formatting and dictionary insertion order do not matter."""
    return sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':'),
                             allow_nan=False).encode()).hexdigest()


def interval(row, duration):
    """Validate one media-clock interval; malformed or out-of-bounds facts fail closed."""
    start, end = row.get('startMs'), row.get('endMs')
    if type(start) is not int or type(end) is not int or not 0 <= start < end <= duration:
        raise ValueError(f'invalid source-clock interval: {start!r}..{end!r}')
    return start, end


def union(spans):
    """Merge duplicates and overlapping intervals before measuring elapsed time."""
    merged = []
    for start, end in sorted(spans):
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return merged


def subtract(start, end, covered):
    """Return uncovered pieces of one interval, retaining the original time clock."""
    cursor = start
    result = []
    for left, right in union(covered):
        if right <= cursor:
            continue
        if left >= end:
            break
        if left > cursor:
            result.append([cursor, min(left, end)])
        cursor = max(cursor, right)
    if cursor < end:
        result.append([cursor, end])
    return result


def clipped_turns(turns, start, end):
    """Read overlapping turns; adjacency alone never counts as simultaneous voices."""
    rows = []
    for turn in turns:
        left, right = max(start, turn['startMs']), min(end, turn['endMs'])
        if right > left:
            rows.append({'startMs': left, 'endMs': right, 'speaker': turn.get('speaker')})
    return rows


def turn_evidence(turns):
    """Measure each opaque speaker and actual overlap independently of transitions."""
    groups = {}
    for row in turns:
        speaker = row.get('speaker')
        if speaker is not None:
            groups.setdefault(speaker, []).append([row['startMs'], row['endMs']])
    spans = {speaker: union(rows) for speaker, rows in groups.items()}
    concurrent = []
    speakers = list(spans)
    for index, speaker in enumerate(speakers):
        for other in speakers[index + 1:]:
            for left, right in spans[speaker]:
                for begin, stop in spans[other]:
                    if min(right, stop) > max(left, begin):
                        concurrent.append([max(left, begin), min(right, stop)])
    transitions = []
    known = sorted((row for row in turns if row.get('speaker') is not None),
                   key=lambda row: (row['startMs'], row['endMs'], str(row['speaker'])))
    for previous, current in zip(known, known[1:]):
        if previous['speaker'] != current['speaker'] and previous['endMs'] <= current['startMs']:
            transitions.append((previous['endMs'] + current['startMs']) // 2)
    return spans, union(concurrent), sorted(set(transitions))


def clean_windows(turns, start, end, options):
    """Plan only single-label snippets, removing intervals with concurrent labels."""
    rows = clipped_turns(turns, start, end)
    grouped = {}
    for row in rows:
        grouped.setdefault(row.get('speaker'), []).append([row['startMs'], row['endMs']])
    windows = []
    for speaker, spans in grouped.items():
        others = []
        for other, other_spans in grouped.items():
            if other != speaker:
                others.extend(other_spans)
        for left, right in union(spans):
            for begin, stop in subtract(left, right, others):
                while stop - begin >= options.min_snippet_ms:
                    finish = min(stop, begin + options.max_snippet_ms)
                    windows.append({'startMs': begin, 'endMs': finish, 'speaker': speaker,
                                    'audioBasis': 'non-overlapping-turn', 'identity': 'unresolved'})
                    begin = finish
    return sorted(windows, key=lambda row: (row['startMs'], row['endMs'], str(row['speaker'])))


def visual_evidence(facts, start, end):
    """Keep bounded AV provenance without treating sampled activity as an identity."""
    if not facts:
        return {'status': 'not-supplied', 'identity': 'unresolved', 'authority': 'audio-fallback'}
    samples = facts.get('samples', [])
    matched = [row for row in samples if start <= row.get('timeMs', -1) < end]
    explicit = [row for row in matched if row.get('evidenceStrength') == 'explicit'
                and row.get('active') is True]
    return {'status': 'available', 'sampleCount': len(matched), 'explicitSampleCount': len(explicit),
            'meaning': facts.get('meaning', 'unspecified'), 'identity': 'unresolved',
            'authority': 'explicit-video-first' if explicit else 'audio-fallback'}


def audit_dialogue(report, turns, *, source, visual_facts=None, options=None):
    """Return deterministic candidate inventory; inputs are never mutated.

    ``source`` must contain sha256 and durationMs. A report source hash, when
    present, must agree. Coverage is acoustic evidence, not proof that words were
    lost: sparse text and boundaries become suggestions requiring review.
    """
    options = options or AuditOptions()
    duration = report.get('durationMs')
    if type(duration) is not int or duration <= 0:
        raise ValueError('report durationMs must be a positive integer')
    if not isinstance(source.get('sha256'), str) or not re.fullmatch(r'[a-f0-9]{64}', source['sha256']):
        raise ValueError('source sha256 is required')
    if source.get('durationMs') != duration:
        raise ValueError('source/report duration mismatch')
    expected_hashes = [report.get('sourceSha256'), report.get('sourceDigest'),
                       (report.get('source') or {}).get('sha256')]
    if any(expected and expected != source['sha256'] for expected in expected_hashes):
        raise ValueError('source/report hash mismatch')
    cues = report.get('segments', [])
    raw_turns = turns.get('turns', []) if isinstance(turns, dict) else turns
    sample_count = len((visual_facts or {}).get('samples', []))
    if len(cues) + len(raw_turns) + sample_count > options.max_input_rows:
        raise ValueError('audit input row budget exceeded')
    unique = {}
    for row in raw_turns:
        left, right = interval(row, duration)
        speaker = row.get('speaker')
        if speaker is not None and not isinstance(speaker, (int, str)):
            raise ValueError('speaker must be an opaque integer/string or null')
        unique[(left, right, speaker)] = {'startMs': left, 'endMs': right, 'speaker': speaker}
    normalized = sorted(unique.values(), key=lambda row: (row['startMs'], row['endMs'], str(row['speaker'])))
    for cue in cues:
        interval(cue, duration)
    candidates = []
    for index, cue in enumerate(cues):
        left, right = cue['startMs'], cue['endMs']
        local = clipped_turns(normalized, left, right)
        if len(local) > options.max_turns_per_cue:
            raise ValueError('per-cue turn budget exceeded')
        voiced = union([[row['startMs'], row['endMs']] for row in local])
        speech_ms = sum(end - start for start, end in voiced)
        if not speech_ms:
            continue  # No acoustic support: text alone must not schedule a silence decode.
        spans, concurrent, transitions = turn_evidence(local)
        speakers = [key for key, rows in spans.items()
                    if sum(end - start for start, end in rows) >= options.min_speaker_ms]
        simultaneous_ms = sum(end - start for start, end in concurrent)
        word_count = len(re.findall(r'\w+', cue.get('text', ''), re.UNICODE))
        density = word_count * 1000 / speech_ms
        reasons = []
        if len(speakers) >= 2:
            reasons.append('multiple-acoustic-labels')
        if transitions and len(speakers) >= 2:
            reasons.append('speaker-transition-inside-cue')
        if simultaneous_ms >= options.min_speaker_ms:
            reasons.append('simultaneous-voices')
        if speech_ms >= options.sparse_min_voiced_ms and density < options.sparse_words_per_second:
            reasons.append('sparse-text-over-voiced-audio')
        edge_ms = max(voiced[0][0] - left, right - voiced[-1][1])
        if edge_ms >= options.edge_drift_ms:
            reasons.append('cue-boundary-drift')
        if not spans and speech_ms >= 1000:
            reasons.append('unknown-acoustic-label')
        if not reasons:
            continue
        evidence = {'cueIndices': [index], 'turns': local, 'speechMs': speech_ms,
                    'textWords': word_count, 'wordsPerVoicedSecond': round(density, 4),
                    'transitionMs': transitions, 'simultaneousMs': simultaneous_ms,
                    'edgeDriftMs': edge_ms, 'visual': visual_evidence(visual_facts, left, right)}
        priority = 'high' if 'sparse-text-over-voiced-audio' in reasons else 'medium'
        if len(candidates) >= options.max_candidates:
            raise ValueError('audit candidate budget exceeded')
        candidates.append({'startMs': left, 'endMs': right, 'priority': priority,
                           'reasons': reasons, 'evidence': evidence,
                           'reviewWindows': clean_windows(normalized, left, right, options)})
    text_coverage = union([[row['startMs'], row['endMs']] for row in cues if row.get('text', '').strip()])
    uncovered = []
    for left, right in union([[row['startMs'], row['endMs']] for row in normalized]):
        uncovered.extend(subtract(left, right, text_coverage))
    for left, right in union(uncovered):
        if right - left < options.min_missing_ms:
            continue
        if len(candidates) >= options.max_candidates:
            raise ValueError('audit candidate budget exceeded')
        candidates.append({'startMs': left, 'endMs': right, 'priority': 'high',
                           'reasons': ['voiced-audio-without-text'],
                           'evidence': {'cueIndices': [], 'speechMs': right - left,
                                        'visual': visual_evidence(visual_facts, left, right)},
                           'reviewWindows': clean_windows(normalized, left, right, options)})
    for candidate in candidates:
        candidate['id'] = stable_hash(candidate)[:20]
    candidates.sort(key=lambda row: (row['startMs'], row['endMs'], row['id']))
    hashes = {'report': stable_hash(report), 'turns': stable_hash(turns),
              'visual': stable_hash(visual_facts)}
    identity = {'algorithmVersion': ALGORITHM_VERSION, 'source': source, 'inputHashes': hashes,
                'options': asdict(options)}
    return {'schemaVersion': SCHEMA_VERSION, **identity, 'cacheKey': stable_hash(identity),
            'candidates': candidates,
            'summary': {'candidateCount': len(candidates), 'cueCount': len(cues),
                        'turnCount': len(normalized), 'duplicateTurnsRemoved': len(raw_turns) - len(normalized),
                        'reasonCounts': {reason: sum(reason in row['reasons'] for row in candidates)
                                         for reason in sorted({r for row in candidates for r in row['reasons']})}},
            'limits': ['Candidates are structural evidence, not confirmed transcript errors.',
                       'Acoustic labels and sampled visual activity do not establish identity.',
                       'ASR offsets are not forced alignment.'],
            'repairs': {'status': 'not-requested', 'suggestions': []}}
