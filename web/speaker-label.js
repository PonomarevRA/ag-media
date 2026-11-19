import {textFor, uiLocale} from "./i18n.js";
/** Resolve the one displayed speaker while keeping the model labels available as evidence. */
export function effectiveSpeaker(cue) {
  if (cue.speakerNumber ?? cue.linkedSpeakerNumber) return cue.speakerNumber ?? cue.linkedSpeakerNumber;
  if (cue.voiceProfileEvidence) return cue.voiceProfileId ? cue.profileSpeakerNumber ?? null : null;
  return cue.speakerNumber ?? cue.linkedSpeakerNumber ?? cue.profileSpeakerNumber ?? cue.speaker ?? cue.visualSpeakerNumber ?? null;
}

/** Format detected overlap without assigning the mixed text to either speaker. */
export function mixedSpeakerLabel(cue, at, locale = uiLocale) {
  if (!cue.overlap || cue.speakerNumber || cue.speakerName) return null;
  if (cue.voiceProfileEvidence) return textFor(locale, "Голоса пересекаются");
  const intervals = cue.overlaps ?? [];
  const matching = at === undefined ? intervals : intervals.filter(span => span.startMs <= at && at < span.endMs);
  if (at !== undefined && intervals.length && !matching.length) return null;
  const ids = [...new Set(matching.flatMap(span => span.speakers))].sort((a, b) => a - b);
  return ids.length ? textFor(locale, "Голоса пересекаются · спикеры {0}", ids.join(", "))
    : textFor(locale, "Голоса пересекаются");
}
