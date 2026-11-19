"""Adapt bounded native EOF rounding for dialogue audits without changing raw facts.

The tolerance is an application policy, not a backend timing guarantee. External
turns which differ from the report's native measurements retain strict validation.
"""

from copy import deepcopy

from voice_studio.speech.dialogue_audit import AuditOptions, audit_dialogue, stable_hash

POLICY_VERSION = 'native-eof-copy-v1'
MAX_EOF_OVERSHOOT_MS = 40
MAX_ADJUSTMENT_DETAILS = 64


def _audit_turns(report, turns, max_rows):
    """Copy only matching native evidence and trim small, otherwise valid EOF overruns."""
    rows = turns.get('turns') if isinstance(turns, dict) else turns
    duration = report.get('durationMs')
    if (type(duration) is not int or duration <= 0 or not isinstance(rows, list)
            or len(rows) > max_rows or rows != report.get('speechTurns')):
        return turns, []
    copied = deepcopy(turns)
    copied_rows = copied['turns'] if isinstance(copied, dict) else copied
    adjustments = []
    for index, row in enumerate(copied_rows):
        if not isinstance(row, dict):
            continue
        start, end = row.get('startMs'), row.get('endMs')
        if (type(start) is int and type(end) is int and 0 <= start < duration
                and duration < end <= duration + MAX_EOF_OVERSHOOT_MS):
            row['endMs'] = duration
            adjustments.append({'turnIndex': index, 'originalEndMs': end, 'usedEndMs': duration,
                                'basis': 'bounded-native-eof-overrun'})
    return copied, adjustments


def audit_native_dialogue(report, turns, *, source, visual_facts=None, options=None):
    """Run the strict detector on an audit copy, retaining raw and normalized hash provenance.

    The public detector remains strict. Its inputHashes.turns describes the exact
    normalized argument; rawTurnsHash separately binds the untouched native input.
    The resulting cache key also binds this policy and all adjustment evidence.
    """
    options = options or AuditOptions()
    copied, adjustments = _audit_turns(report, turns, options.max_input_rows)
    inventory = audit_dialogue(report, copied, source=source, visual_facts=visual_facts, options=options)
    if not adjustments:
        return inventory
    provenance = {'policyVersion': POLICY_VERSION, 'maxEofOvershootMs': MAX_EOF_OVERSHOOT_MS,
                  'basis': 'matching-report-native-turns', 'rawTurnsHash': stable_hash(turns),
                  'auditTurnsHash': inventory['inputHashes']['turns'],
                  'durationMs': report['durationMs'], 'adjustmentCount': len(adjustments),
                  'adjustments': adjustments[:MAX_ADJUSTMENT_DETAILS],
                  'truncated': len(adjustments) > MAX_ADJUSTMENT_DETAILS}
    inventory['nativeTurnBounds'] = provenance
    inventory['cacheKey'] = stable_hash({'strictAuditKey': inventory['cacheKey'],
                                         'nativeTurnBounds': provenance})
    return inventory
