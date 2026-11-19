"""Measure conservative speech edges in 16-bit mono PCM."""

from array import array
import sys


def speech_edges(pcm, rate):
    """Return sample offsets around sustained signal, with one window of padding."""
    samples = array("h", pcm)
    if sys.byteorder != "little":
        samples.byteswap()
    width = rate // 50
    powers = []
    for start in range(0, len(samples), width):
        window = samples[start:start + width]
        powers.append(sum(value * value for value in window) / len(window))
    if len(powers) < 2:
        return 0, len(samples)
    sorted_power = sorted(powers)
    reference = sorted_power[int((len(sorted_power) - 1) * .9)]
    threshold = max(30 ** 2, reference * .02)
    active = [index for index in range(len(powers) - 1)
              if powers[index] >= threshold and powers[index + 1] >= threshold]
    if not active:
        return 0, len(samples)
    return max(0, (active[0] - 1) * width), min(len(samples), (active[-1] + 3) * width)


def source_speech_window(audio, cue_start, cue_end, audio_start, rate):
    """Refine a cue only within its original timestamp range and available PCM."""
    first = max(0, cue_start - audio_start)
    last = max(0, cue_end - audio_start)
    left = min(audio.getnframes(), first * rate // 1000)
    right = min(audio.getnframes(), last * rate // 1000)
    if right <= left:
        return cue_start, cue_end
    audio.setpos(left)
    rise, fall = speech_edges(audio.readframes(right - left), rate)
    # Background sound or a quiet word can make envelope detection conservative.
    maximum_trim = min(450, (cue_end - cue_start) // 5)
    start = max(cue_start, audio_start + first + rise * 1000 // rate)
    end = min(cue_end, audio_start + first + (fall * 1000 + rate - 1) // rate)
    return min(start, cue_start + maximum_trim), max(end, cue_end - maximum_trim)
