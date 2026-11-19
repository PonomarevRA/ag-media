"""Conservative experiment-only promotion rules; scores are proxies, never perceptual truth."""

from math import isfinite


def pitch_promotion(previous, candidate):
    """Keep the previous phrase when F0 refinement lacks evidence or regresses text/voice proxies."""
    names = ['whisper_word_error_rate', 'eres2net_encoder_cosine']
    if any(not isinstance(row.get(name), (int, float)) or not isfinite(row[name])
           for row in (previous, candidate) for name in names):
        return {'decision': 'rollback', 'reason': 'missing ASR or voice evidence'}
    before, after = (candidate.get('f0_shape_' + key + '_semitones') for key in ('before', 'after'))
    if not all(isinstance(value, (int, float)) and isfinite(value) for value in (before, after)):
        return {'decision': 'rollback', 'reason': 'missing final F0 evidence'}
    if candidate['whisper_word_error_rate'] > previous['whisper_word_error_rate'] + .01:
        return {'decision': 'rollback', 'reason': 'ASR regression'}
    if candidate['eres2net_encoder_cosine'] < previous['eres2net_encoder_cosine'] - .03:
        return {'decision': 'rollback', 'reason': 'voice similarity regression'}
    if after >= before:
        return {'decision': 'rollback', 'reason': 'F0 shape did not improve'}
    return {'decision': 'candidate', 'reason': 'proxy guards passed; audition still required'}
