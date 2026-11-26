"""Keep short audiovisual confirmations on the measured part of a speech cue."""

from voice_studio.vision.video_activity_chunks import STEP

MIN_POSITIVE_FRAMES = 8
MAX_INTERVALS_PER_CUE = 8
BASIS = "light-asd-local"


def visible_runs(frames, start, end, cuts):
    """Yield uninterrupted runs of one active track and face in a cue."""
    first = (start + STEP - 1) // STEP
    last = min(len(frames), (end + STEP - 1) // STEP)
    active = None
    began = first
    for index in range(first, last):
        hits = [face for face in frames[index] if face[5] == 1]
        witness = (hits[0][0], hits[0][6]) if len(hits) == 1 else None
        if witness != active or index in cuts:
            if active is not None and index - began >= MIN_POSITIVE_FRAMES:
                yield began, index, active
            began, active = index, witness
    if active is not None and last - began >= MIN_POSITIVE_FRAMES:
        yield began, last, active


def local_visible_speech(phrases, visual, observations):
    """Publish local model evidence when the whole cue lacks a visual assignment."""
    timeline = visual["timeline"]
    if timeline.get("stepMs") != STEP:
        return []
    frames = timeline["frames"]
    shots = visual["editorTimeline"]["shots"]
    cut_times = {row["startMs"] for row in shots}
    cut_times.update(visual["faceTracking"]["identityCutTimesMs"])
    cuts = {(time + STEP - 1) // STEP for time in cut_times}
    intervals = []
    for start, end in phrases:
        observation = observations.get(str(start), {})
        if observation.get("state") == "visible-active":
            continue
        for count, (left, right, (track, face)) in enumerate(visible_runs(frames, start, end, cuts)):
            if count >= MAX_INTERVALS_PER_CUE:
                break
            intervals.append({"startMs": left * STEP, "endMs": min(end, right * STEP),
                              "track": track, "faceId": face, "cueStartMs": start,
                              "basis": BASIS, "activeFrames": right - left})
    return sorted(intervals, key=lambda item: item["startMs"])
