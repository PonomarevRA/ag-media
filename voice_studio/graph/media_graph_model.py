"""Build a typed, source-clock graph from local media analysis observations."""

from bisect import bisect_left, bisect_right
from collections import Counter, defaultdict
from hashlib import sha256
from math import isfinite
import re

from voice_studio.audio.audio_scene import LABELS as AUDIO_LABELS, clean_audio_layers
from voice_studio.audio.dialogue_pauses import clean_dialogue_pauses
from voice_studio.graph.object_screens import VerifiedPanels, panel_scroll_previews, screen_evidence
from voice_studio.graph.media_graph_interfaces import PANEL_TYPES, add_interface_observations
from voice_studio.graph.frame_change_graph import add_frame_changes, change_coverage
from voice_studio.graph.media_graph_expressions import add_facial_expressions
from voice_studio.graph.scene_hierarchy import add_scene_hierarchy
from voice_studio.graph.ui_activity_graph import add_ui_activity
from voice_studio.graph.dialogue_audit_graph import add_dialogue_review
from voice_studio.graph.media_graph_mod_extensions import add_mod_extensions
from voice_studio.graph.media_graph_objects import DISPLAY_LABELS, add_object_tracks, add_speaker_context, box_coverage
from voice_studio.graph.media_object_profiles import add_object_profiles
from voice_studio.graph.speech_screen_evidence import add_speech_screen_evidence
from voice_studio.speech.person_mentions import clean_mentions
from voice_studio.speech.speaker_profiles import APPEARANCE_FIELDS, cue_number
from voice_studio.vision.frame_changes import BASIS as FRAME_CHANGE_DETAIL
from voice_studio.vision.scene_sampling import SMALL_OBJECT_DETAIL
from voice_studio.vision.video_colors import hex_rgb
from voice_studio.vision.vehicle_evidence import validated_body_color

KIND_NAMES = {
    "ru": {"audio_track": "Звуковая дорожка", "video_track": "Видеодорожка", "shot": "План",
           "speaker": "Спикер", "face": "Лицо", "frame": "Кадр", "overlap": "Пересечение голосов",
           "object": "Предметы", "text": "Надписи", "speech": "Реплики", "faces": "Лица",
           "frame_change": "Изменение в кадре", "camera_motion": "Движение камеры", "person_motion": "Движение персонажа"},
    "en": {"audio_track": "Audio track", "video_track": "Video track", "shot": "Shot",
           "speaker": "Speaker", "face": "Face", "frame": "Frame", "overlap": "Overlapping speech",
           "object": "Objects", "text": "Screen text", "speech": "Utterances", "faces": "Faces",
           "frame_change": "Frame change", "camera_motion": "Camera motion", "person_motion": "Character motion"},
}
MAX_TEXT = 4096
COLOR_HEX = re.compile(r"#[0-9a-fA-F]{6}")


def source_interval(row, duration):
    """Return a bounded half-open interval on the source clock, in milliseconds."""
    start, end = row.get("startMs"), row.get("endMs")
    if type(start) is int and type(end) is int and 0 <= start < min(end, duration):
        return start, min(end, duration)
    return None


def box_coordinates(value):
    """Keep finite detections wholly contained in normalized source coordinates."""
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    if not all(type(item) in (int, float) and isfinite(item) and 0 <= item <= 1 for item in value):
        return None
    left, top, width, height = value
    if not width or not height or left >= 1 or top >= 1 or left + width > 1.002 or top + height > 1.002:
        return None
    return [round(item, 4) for item in (left, top, min(width, 1 - left), min(height, 1 - top))]


def positive_id(value):
    """Accept integer identities and the serialized integer keys used by the vision report."""
    if isinstance(value, str) and value.isascii() and value.isdecimal() and len(value) <= 9:
        value = int(value)
    return value if type(value) is int and value > 0 else None


def compact_source_data(row, keys):
    """Copy only short scalar facts from an analysis record."""
    result = {}
    for key in keys:
        value = row.get(key)
        if type(value) in (str, bool, int):
            result[key] = value[:256] if isinstance(value, str) else value
        elif type(value) is float and isfinite(value):
            result[key] = value
    return result


def source_key(label):
    """Intern repeated names and screen readings without embedding them in edge IDs."""
    normalized = " ".join(label.casefold().split())
    return sha256(normalized.encode()).hexdigest()[:20]


def source_coverage(report, duration):
    """Preserve analyzed windows, including gaps outside the visual sampling budget."""
    declared = report.get("coverage")
    if isinstance(declared, dict):
        coverage = {}
        for modality, rows in declared.items():
            if not isinstance(rows, list):
                continue
            coverage[modality] = []
            for row in rows:
                bounds = source_interval(row, duration)
                if bounds:
                    item = {"startMs": bounds[0], "endMs": bounds[1],
                            **compact_source_data(row, ("status", "reason", "method", "windowStartMs",
                                                        "samples", "recognitionModel", "speakerWarning",
                                                        "objectSampling", "objectFrames", "interfaceFrames",
                                                        "interfaceOcrCrops", "interfaceStatus"))}
                    for key in ("statuses", "interfaceOcrStatuses"):
                        if isinstance(statuses := row.get(key), dict):
                            item[key] = compact_source_data(statuses, tuple(statuses))
                    if modality == "scene" and (detail := change_coverage(row)):
                        item["frameChanges"] = detail
                    coverage[modality].append(item)
        return coverage
    evidence = report.get("videoEvidence") or {}
    timeline = evidence.get("timeline") or {}
    editor = evidence.get("editorTimeline") or {}
    scene = report.get("sceneEvidence") or {}
    coverage = {}

    def mark(modality, end, status="sampled"):
        """Record one measured prefix when the analyzer did return observations."""
        if type(end) is int and end > 0:
            coverage[modality] = [{"startMs": 0, "endMs": min(end, duration), "status": status}]

    mark("speech", report.get("durationMs"), "analyzed")
    if timeline.get("frames"):
        mark("face", len(timeline["frames"]) * timeline.get("stepMs", 40))
    if shot_color_rows(report, duration):
        mark("shots", duration)
    elif editor.get("shots"):
        mark("shots", editor.get("durationMs") or duration)
    elif scene.get("status") == "ok" and isinstance(scene.get("shotStartsMs"), list) and scene.get("frames"):
        mark("shots", max(row["timeMs"] for row in scene["frames"]) + 1)
    if scene.get("status") == "ok" and scene.get("frames"):
        stages = scene.get("stages") or {}
        complete = all(value == "ok" for value in stages.values()) and scene.get("interfaceStatus", "ok") == "ok"
        status = "sampled" if complete else "partial"
        mark("scene", max(row["timeMs"] for row in scene["frames"]) + 1, status)
        coverage["scene"][0].update(compact_source_data(
            scene, ("objectSampling", "objectFrames", "interfaceFrames", "interfaceOcrCrops", "interfaceStatus")),
            statuses=compact_source_data(stages, tuple(stages)))
        if detail := change_coverage(scene):
            coverage["scene"][0]["frameChanges"] = detail
        if isinstance(screen_ocr := scene.get("interfaceOcrStatuses"), dict):
            coverage["scene"][0]["interfaceOcrStatuses"] = compact_source_data(screen_ocr, tuple(screen_ocr))
    return coverage


def shot_color_rows(report, duration):
    """Accept only bounded color records, including data imported from Studio JSON."""
    rows = report.get("shotColors")
    if not isinstance(rows, list):
        return []
    valid = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        start, average, count = row.get("startMs"), row.get("average"), row.get("sampleCount")
        if (type(start) is int and 0 <= start < duration and isinstance(average, str)
                and COLOR_HEX.fullmatch(average) and type(count) is int and 1 <= count <= 1_000_000):
            valid.append((start, average, count))
    return valid


def assign_shot_colors(graph, rows):
    """Weight partition averages by frame count when a shot crosses a graph window."""
    totals = {}
    for start, average, count in rows:
        key = next(graph.intersecting_shots(start, start + 1), None)
        if key is None:
            continue
        total = totals.setdefault(key, [0, 0, 0, 0])
        for channel, offset in enumerate((1, 3, 5)):
            total[channel] += count * int(average[offset:offset + 2], 16)
        total[3] += count
    for key, (red, green, blue, count) in totals.items():
        graph.nodes[key]["data"]["color"] = {
            "average": hex_rgb((red / count, green / count, blue / count)),
            "sampleCount": count,
        }


def covers_all(interval, overlaps):
    """Check whether explicit overlap windows leave any unmixed time on this cue."""
    position, end = interval
    for left, right in sorted(overlaps):
        if left > position:
            return False
        position = max(position, right)
        if position >= end:
            return True
    return False


class MediaGraph:
    """Own stable snapshot IDs, interval lookup and referentially complete directed edges."""

    def __init__(self, duration, language):
        self.duration = duration
        self.language = language
        self.names = KIND_NAMES.get(language, KIND_NAMES["ru"])
        self.nodes = {}
        self.edges = set()
        self.shot_ids = []
        self.shot_starts = []
        self.shot_facts = defaultdict(lambda: {"object": Counter(), "text": Counter(), "speech": 0, "faces": set()})

    def node(self, key, kind, label, interval=(None, None), data=None):
        """Intern one node by its stable snapshot identity."""
        return self.nodes.setdefault(key, {"id": key, "kind": kind, "startMs": interval[0], "endMs": interval[1],
                                           "label": label, "data": data or {}})

    def connect(self, source, relation, target):
        """Insert one directed, deduplicated relation between existing dictionary entries."""
        if source in self.nodes and target in self.nodes:
            self.edges.add((source, relation, target))

    def track(self, media, stream=None):
        """Register an audio or video track, optionally linked to its container stream index."""
        kind = f"{media}_track"
        index = (stream or {}).get("index")
        suffix = str(index) if type(index) is int else "0"
        key = f"{kind}:{suffix}"
        data = compact_source_data(stream or {}, ("index", "codec", "codec_name", "language", "channels",
                                                    "sample_rate", "width", "height"))
        label = f"{self.names[kind]} {index}" if type(index) is int else self.names[kind]
        self.node(key, kind, label, (0, self.duration), data)
        return key

    def speaker(self, number, profile=None, scope=None):
        """Return a speaker by stable ID with local numbering and exact-source profile links."""
        key = f"speaker:{number}"
        data = {"number": number}
        if profile:
            facts = compact_source_data(profile, ("id", "faceId", "faceBasis", "pinned", "scope", "windowStartMs"))
            if profile_id := facts.pop("id", None):
                facts["profileId"] = profile_id
            data.update(facts)
            if profile.get("faceIds"):
                data["faceIds"] = [value for raw in profile["faceIds"] if (value := positive_id(raw))]
            outfit = profile.get("clothing")
            if isinstance(outfit, dict):
                data["clothing"] = compact_source_data(outfit, ("swatch", "pattern", "source"))
            if isinstance(appearance := profile.get("appearance"), dict):
                data["appearance"] = compact_source_data(appearance, APPEARANCE_FIELDS)
        label = f"{self.names['speaker']} {number}"
        if scope is not None:
            local = positive_id(scope.get("speakerLocalNumber"))
            window = scope.get("speakerWindowStartMs")
            if local is not None and type(window) is int and window >= 0:
                minutes, seconds = divmod(window // 1000, 60)
                hours, minutes = divmod(minutes, 60)
                label = f"{self.names['speaker']} {local} · {hours:02d}:{minutes:02d}:{seconds:02d}"
                data.update({"localNumber": local, "windowStartMs": window})
        node = self.node(key, "speaker", label, data=data)
        if profile or scope is not None:
            node["data"].update(data)
        if "localNumber" in data:
            node["label"] = label
        return key

    def face(self, identity, outfit=None, appearance=None):
        """Intern a visible face without confusing it with a confirmed audible speaker."""
        key = f"face:{identity}"
        node = self.node(key, "face", f"{self.names['face']} {identity}", data={"faceId": identity})
        if isinstance(outfit, dict):
            node["data"]["clothing"] = compact_source_data(outfit, ("swatch", "pattern", "source"))
        if isinstance(appearance, dict) and (facts := compact_source_data(appearance, APPEARANCE_FIELDS)):
            node["data"]["appearance"] = facts
        return key

    def shots(self, cuts, coverage, video_track):
        """Create contiguous shots only within explicitly analyzed video windows."""
        cut_times = sorted({item["startMs"] for item in cuts
                            if type(item.get("startMs")) is int and 0 <= item["startMs"] < self.duration})
        detected = set(cut_times)
        ranges = []
        for row in coverage:
            bounds = source_interval(row, self.duration)
            if bounds and row.get("status") not in ("failed", "unavailable", "skipped", "unknown", "uncovered"):
                ranges.append(bounds)
        merged = []
        for begin, end in sorted(ranges):
            # Adjacent scan windows share one continuous source interval, even without a cut.
            if merged and begin <= merged[-1][1]:
                merged[-1][1] = max(end, merged[-1][1])
            else:
                merged.append([begin, end])
        previous = None
        for begin, end in merged:
            left = bisect_right(cut_times, begin)
            right = bisect_right(cut_times, end - 1)
            marks = [begin, *cut_times[left:right], end]
            for start, stop in zip(marks, marks[1:]):
                key = f"shot:{start}"
                self.shot_ids.append(key)
                self.shot_starts.append(start)
                self.node(key, "shot", f"{self.names['shot']} {len(self.shot_ids)}", (start, stop),
                          {"boundaryBasis": "detected-cut" if start in detected else "analysis-window"})
                self.connect(key, "on_track", video_track)
                if previous and self.nodes[previous]["endMs"] == start:
                    self.connect(previous, "next", key)
                previous = key

    def intersecting_shots(self, start, end):
        """Yield the few plans intersecting an interval without a full graph scan."""
        index = max(0, bisect_right(self.shot_starts, start) - 1)
        while index < len(self.shot_ids) and self.shot_starts[index] < end:
            key = self.shot_ids[index]
            if self.nodes[key]["endMs"] > start:
                yield key
            index += 1

    def link_shots(self, node, start, end, evidence_kind=None):
        """Attach events to every plan they actually intersect."""
        for shot in self.intersecting_shots(start, end):
            self.connect(node, "in_shot", shot)
            if evidence_kind == "speech":
                self.shot_facts[shot]["speech"] += 1

    def frame(self, stamp, source):
        """Register a sampled instant once, even when several detectors saw the same frame."""
        key = f"frame:{stamp}"
        frame = self.node(key, "frame", f"{self.names['frame']} {stamp} ms", (stamp, stamp + 1), {"sources": []})
        if source not in frame["data"]["sources"]:
            frame["data"]["sources"].append(source)
        self.link_shots(key, stamp, stamp + 1)
        return key

    def observations_summary(self):
        """Describe plans using only detected entities and indexed speech."""
        for shot, facts in self.shot_facts.items():
            description = []
            for kind in ("object", "text"):
                entries = [label for label, _ in facts[kind].most_common(3)]
                if entries:
                    description.append(f"{self.names[kind]} — {', '.join(entries)}")
            if facts["speech"]:
                description.append(f"{self.names['speech']} — {facts['speech']}")
            if facts["faces"]:
                description.append(f"{self.names['faces']} — {len(facts['faces'])}")
            self.nodes[shot]["data"]["description"] = " · ".join(description)


def stream_tracks(graph, report, kind):
    """Link observations to the first applicable stream while cataloguing known secondary tracks."""
    streams = defaultdict(list)
    for row in report.get("streams") or report.get("mediaStreams") or []:
        media = row.get("kind") or row.get("codec_type")
        if media in ("audio", "video"):
            streams[media].append(graph.track(media, row))
    audio = streams["audio"][0] if streams["audio"] else graph.track("audio") if report.get("segments") or kind == "audio" else None
    video = streams["video"][0] if streams["video"] else graph.track("video") if kind == "video" else None
    return audio, video


def bind_profiles(graph, report):
    """Normalize confirmed face-to-speaker links and digital profile anchors."""
    evidence = report.get("videoEvidence") or {}
    editor = evidence.get("editorTimeline") or {}
    outfits = editor.get("clothingByFace") or {}
    attributes = editor.get("faceAttributes") or {}
    bindings = {}
    # A visible, silent face owns its observations even when no speech binding was produced.
    observed = set(attributes) | set(outfits) | set(editor.get("facePhotos") or {})
    identities = {identity for raw_id in observed if (identity := positive_id(raw_id))}
    for identity in sorted(identities):
        graph.face(identity, outfits.get(str(identity), outfits.get(identity)),
                   attributes.get(str(identity), attributes.get(identity)))
    for profile in report.get("speakerProfileIndex") or []:
        if number := positive_id(profile.get("number")):
            speaker = graph.speaker(number, profile)
            if identity := positive_id(profile.get("faceId")):
                face = graph.face(identity, outfits.get(str(identity), outfits.get(identity)),
                                  attributes.get(str(identity), attributes.get(identity)))
                if profile.get("faceBasis") in ("audio-visual", "editorial"):
                    graph.connect(speaker, "profile_face", face)
    for raw_id, binding in (evidence.get("faceSpeakers") or {}).items():
        identity, number = positive_id(raw_id), positive_id(binding.get("speaker"))
        if identity is None:
            continue
        face = graph.face(identity, outfits.get(str(identity), outfits.get(identity)),
                          attributes.get(str(identity), attributes.get(identity)))
        if number and binding.get("basis") in ("audio-visual", "editorial"):
            speaker = graph.speaker(number)
            bindings[identity] = number
            graph.connect(speaker, "profile_face", face)
    return bindings


def add_utterances(graph, report, track):
    """Add each transcript cue and preserve overlap participants as timed edges."""
    cues = sorted((row for row in report.get("segments") or [] if source_interval(row, graph.duration)),
                  key=lambda row: (row["startMs"], row["endMs"]))
    at_time = Counter()
    previous = None
    for cue in cues:
        interval = source_interval(cue, graph.duration)
        start, end = interval
        index = at_time[start]
        at_time[start] += 1
        key = f"utterance:{start}:{index}"
        text = str(cue.get("text") or "")[:MAX_TEXT]
        editorial = bool(cue.get("speakerNumber") or cue.get("speakerName"))
        is_mixed = bool(cue.get("overlap") and not editorial)
        overlaps = []
        if is_mixed:
            for span in cue.get("overlaps") or []:
                bounds = source_interval(span, graph.duration)
                if bounds:
                    left, right = max(bounds[0], start), min(bounds[1], end)
                    if left < right:
                        overlaps.append((left, right, span))
        data = compact_source_data(cue, ("speakerName", "asrModel", "visualSpeakerBasis", "speakerBasis"))
        window = cue.get("speakerAudioWindow")
        if isinstance(window, dict) and (bounds := source_interval(window, graph.duration)):
            if start <= bounds[0] < bounds[1] <= end:
                data["speakerAudioWindow"] = {"startMs": bounds[0], "endMs": bounds[1]}
        if mentions := clean_mentions(text, cue.get("personMentions")):
            data["personMentions"] = mentions
        data["overlapUnresolved"] = is_mixed and not overlaps
        if is_mixed and overlaps and not covers_all(interval, [(left, right) for left, right, _ in overlaps]):
            data["speakerAssignment"] = "outside-overlap"
        graph.node(key, "utterance", text, interval, data)
        graph.connect(key, "on_track", track)
        graph.link_shots(key, start, end, "speech")
        if previous is not None:
            graph.connect(previous, "next", key)
        previous = key
        number = positive_id(cue_number(cue))
        if number and (not is_mixed or data.get("speakerAssignment") == "outside-overlap"):
            indexed = graph.nodes.get(f"speaker:{number}", {}).get("data", {}).get("profileId")
            cue_profile = cue.get("speakerProfileId")
            profile = {"id": cue_profile} if cue_profile and not indexed else None
            graph.connect(key, "spoken_by", graph.speaker(number, profile, cue))
        for offset, (left, right, span) in enumerate(overlaps):
            mixed_key = f"overlap:{start}:{index}:{offset}"
            graph.node(mixed_key, "overlap", graph.names["overlap"], (left, right),
                       compact_source_data(span, ("basis", "source")))
            graph.connect(mixed_key, "overlap_of", key)
            graph.link_shots(mixed_key, left, right)
            for number in {positive_id(number) for number in span.get("speakers") or []} - {None}:
                if 'voiceProfileEvidence' in cue:
                    raw_key = f"raw_audio_cluster:{cue.get('speakerWindowStartMs', 0)}:{number}"
                    graph.node(raw_key, 'raw_audio_cluster', f"Audio cluster {number}",
                               data={'number': number, 'identityScope': 'raw-cluster-not-source-profile'})
                    graph.connect(mixed_key, 'raw_participant', raw_key)
                else:
                    graph.connect(mixed_key, "participant", graph.speaker(number))
    return cues


def add_audio_layers(graph, report, track):
    """Expose model and pitch observations on the audio track and matching shots."""
    if track is None:
        return
    language = int(graph.language == "en")
    utterances = {(node["startMs"], node["endMs"]): node["id"] for node in graph.nodes.values()
                  if node["kind"] == "utterance"}
    for index, row in enumerate(clean_audio_layers(report.get("audioLayers"), graph.duration)):
        start, end = row["startMs"], row["endMs"]
        kind, label = row["kind"], row["label"]
        key = f"audio_{kind}:{start}:{index}"
        graph.node(key, f"audio_{kind}", AUDIO_LABELS[label][language], (start, end), row)
        graph.connect(key, "on_track", track)
        graph.link_shots(key, start, end)
        cue = utterances.get((start, end))
        if kind == "intonation" and cue:
            graph.connect(key, "intonation_of", cue)


def add_dialogue_pauses(graph, report, track):
    """Expose candidate points and nearby utterances on the original timeline."""
    pauses = clean_dialogue_pauses(report.get("dialoguePauses"), graph.duration)
    if not pauses or track is None:
        return
    utterances = [node for node in graph.nodes.values() if node["kind"] == "utterance"]
    by_end = sorted((node["endMs"], node["id"]) for node in utterances)
    starts = [node["startMs"] for node in utterances]
    ends = [end for end, _ in by_end]
    label = "Dialogue pause" if graph.language == "en" else "Пауза в диалоге"
    for pause in pauses:
        start, end = pause["startMs"], pause["endMs"]
        key = f"pause:{start}:{end}"
        data = {**pause, "durationMs": end - start}
        graph.node(key, "pause", label, (start, end), data)
        graph.connect(key, "on_track", track)
        graph.link_shots(key, start, end)
        before = bisect_right(ends, start) - 1
        after = bisect_left(starts, end)
        if before >= 0:
            graph.connect(key, "after_utterance", by_end[before][1])
        if after < len(utterances):
            graph.connect(key, "before_utterance", utterances[after]["id"])


def scene_observations(graph, report, *, keep_display_aliases=False):
    """Record timed objects, OCR and any visual screen structure."""
    frames = (report.get("sceneEvidence") or {}).get("frames") or []
    panels = VerifiedPanels(graph, frames, box_coordinates)
    display_aliases, verified = [], {}
    for row in frames:
        stamp = row.get("timeMs")
        if type(stamp) is not int or not 0 <= stamp < graph.duration:
            continue
        if keep_display_aliases and row.get("panelChecked") is True:
            frame_id = graph.frame(stamp, "interface-layout")
            graph.nodes[frame_id]["data"]["panelChecked"] = True
        panel_sample = None
        for kind, source, inputs, field in (("object", "nanodet", row.get("objects"), "label"),
                                              ("screen_text", "ocr", row.get("text"), "text")):
            for index, item in enumerate(inputs or []):
                label = str(item.get(field) or "").strip()[:512]
                box = box_coordinates(item.get("box"))
                if label and box:
                    if kind == "object" and label.casefold() in DISPLAY_LABELS:
                        if panel_sample is None:
                            panel_sample = panels.measurement_at(stamp)
                        layout_stamp, bounds = panel_sample
                        owners = [panel for panel in bounds if box_coverage(box, panel) >= .9]
                        if owners:
                            if keep_display_aliases and len(owners) == 1:
                                display_aliases.append((stamp, index, label, box, item, layout_stamp, owners[0]))
                            continue
                    screen = screen_evidence(label, box, row, box_coordinates) if kind == "object" else None
                    add_scene_observation(graph, stamp, kind, source, index, label, box, item, screen=screen)
        interfaces = row.get("interfaces") or []
        # Appended crop detections keep the baseline indexes, including the synthetic panel.
        panel_offset = sum(item.get("detailBasis") not in (SMALL_OBJECT_DETAIL, FRAME_CHANGE_DETAIL) for item in row.get("objects") or ())
        for index, panel in enumerate(interfaces, start=panel_offset):
            if not isinstance(panel, dict) or panel.get("panelType") not in PANEL_TYPES:
                continue
            if not (box := box_coordinates(panel.get("box"))):
                continue
            panel_type = panel["panelType"]
            screen = {"type": panel_type, "basis": "verified-panel"}
            node_id = add_scene_observation(graph, stamp, "object", "interface-layout", index, panel_type, box, {}, screen=screen)
            if keep_display_aliases:
                verified[stamp, tuple(box)] = node_id
        add_interface_observations(graph, stamp, interfaces, box_coordinates)
    for stamp, index, label, box, item, layout_stamp, panel in display_aliases:
        owner = verified.get((layout_stamp, tuple(panel)))
        if owner:
            alias_id = f"object_alias:{stamp}:{index}:{source_key(label)}"
            info = {"box": box, "source": "nanodet", "sourceIndex": index,
                    **compact_source_data(item, ("score", "basis", "detailBasis"))}
            graph.node(alias_id, "object_alias", label, (stamp, stamp + 1), info)
            graph.connect(alias_id, "alias_of", owner)
            graph.connect(alias_id, "at_frame", graph.frame(stamp, "nanodet"))


def object_matches(item, stamp):
    """Validate optional comparison data from new and imported scene reports."""
    matches = []
    for entry in item.get("objectMatches") or []:
        if not isinstance(entry, dict):
            continue
        time, label, scale = entry.get("timeMs"), entry.get("label"), entry.get("scaleRatio")
        box = box_coordinates(entry.get("box"))
        if (type(time) is not int or not 0 <= time <= stamp
                or not isinstance(label, str) or not label.strip() or not box
                or type(scale) not in (float, int) or not isfinite(scale) or scale <= 0):
            continue
        metrics = compact_source_data(entry, ("objectSimilarity", "edgeSimilarity", "backgroundSimilarity"))
        matches.append({"timeMs": time, "label": label.strip()[:512], "box": box, "scaleRatio": scale,
                        **({"comparisonBasis": "frame-change-anchor"} if entry.get("comparisonBasis") == "frame-change-anchor" else {}),
                        **{key: value for key, value in metrics.items()
                           if type(value) in (int, float) and -1 <= value <= 1}})
    return matches[:3]


def add_scene_observation(graph, stamp, kind, source, index, label, box, item, *, screen=None):
    """Keep the detector box and confidence at the measured instant."""
    entity = f"{kind}:{source_key(label)}"
    graph.node(entity, kind, label)
    node_kind = "object_observation" if kind == "object" else "text_observation"
    observation = f"{node_kind}:{stamp}:{index}:{source_key(label)}"
    data = {"box": box, "source": source, **compact_source_data(item, ("score", "basis", "detailBasis"))}
    if kind == "object":
        if source == "nanodet":
            data["rawLabel"] = label
        if screen:
            data["interfaceEvidence"] = screen
        if palette := item.get("objectPalette"):
            data["objectPalette"] = palette
        if matches := object_matches(item, stamp):
            data["objectMatches"] = matches
        # The reconciliation module validates these bounded measured-pair metrics.
        pairs = item.get("objectPartMatches")
        if isinstance(pairs, list):
            data["objectPartMatches"] = [dict(pair) for pair in pairs[:3] if isinstance(pair, dict)]
        if body_color := validated_body_color(item.get("vehicleBodyColor")):
            data["vehicleBodyColor"] = body_color
    graph.node(observation, node_kind, label, (stamp, stamp + 1), data)
    graph.connect(observation, "at_frame", graph.frame(stamp, source))
    graph.connect(observation, "instance_of", entity)
    for shot in graph.intersecting_shots(stamp, stamp + 1):
        graph.shot_facts[shot]["object" if kind == "object" else "text"][label] += 1
    return observation


def add_person_counts(graph, report):
    """Publish estimated counts and measured figure boxes on their source clock."""
    editor = (report.get("videoEvidence") or {}).get("editorTimeline") or {}
    pilot = editor.get("personPilot")
    samples = pilot.get("samples") if isinstance(pilot, dict) else None
    if not isinstance(samples, list):
        return
    for row in samples:
        if not isinstance(row, dict):
            continue
        stamp, count = row.get("timeMs"), row.get("count")
        if type(stamp) is not int or not 0 <= stamp < graph.duration or type(count) is not int or not 0 <= count <= 20:
            continue
        figures = []
        people = row.get("people")
        for item in people[:20] if isinstance(people, list) else ():
            if isinstance(item, dict) and (box := box_coordinates(item.get("box"))):
                figures.append({"box": box, **compact_source_data(item, ("score",))})
        key = f"person_count:{stamp}"
        graph.node(key, "person_count", "People estimate", (stamp, stamp + 1),
                   {"count": count, "source": "nanodet", "people": figures})
        graph.connect(key, "at_frame", graph.frame(stamp, "nanodet"))
        graph.link_shots(key, stamp, stamp + 1)


def person_observations(graph, report):
    """Expose sampled body geometry, head pose, and gaze through links to visible face IDs."""
    evidence = report.get("videoEvidence") or {}
    for row in (evidence.get("personLayer") or {}).get("frames") or []:
        stamp = row.get("timeMs")
        if type(stamp) is not int or not 0 <= stamp < graph.duration:
            continue
        for person in row.get("people") or []:
            identity, track = positive_id(person.get("faceId")), positive_id(person.get("track"))
            if not identity or not track:
                continue
            data = {"source": "person-layer", "faceId": identity, "track": track}
            if box := box_coordinates(person.get("box")):
                data["box"] = box
            for field, keys in (("headPose", ("yawDeg", "pitchDeg", "horizontal", "vertical")),
                                ("gaze", ("horizontal", "irisFraction"))):
                facts = compact_source_data(person.get(field) or {}, keys)
                if facts:
                    data[field] = facts
            if len(data) == 3:
                continue
            face = graph.face(identity)
            key = f"person_observation:{stamp}:{track}"
            graph.node(key, "person_observation", graph.nodes[face]["label"], (stamp, stamp + 1), data)
            graph.connect(key, "instance_of", face)
            graph.connect(key, "at_frame", graph.frame(stamp, "person-layer"))
            for shot in graph.intersecting_shots(stamp, stamp + 1):
                graph.shot_facts[shot]["faces"].add(face)
                graph.connect(shot, "visible_face", face)


def add_motion(graph, report, video_track, bindings):
    """Keep motion observations queryable on the original clock and attach known identities."""
    evidence = report.get("videoEvidence") or {}
    editor = evidence.get("editorTimeline") or {}
    scene = report.get("sceneEvidence") or {}
    layers = (
        ("camera_motion", scene.get("cameraMotion")),
        ("person_motion", editor.get("personMotion")),
    )
    for kind, rows in layers:
        if not isinstance(rows, list):
            continue
        for row in rows:
            if not isinstance(row, dict) or not (bounds := source_interval(row, graph.duration)):
                continue
            start, end = bounds
            shots = list(graph.intersecting_shots(start, end))
            if len(shots) > 1 or end != row["endMs"]:
                continue
            facts = compact_source_data(row, ("horizontal", "vertical", "zoom", "shiftX", "shiftY", "scale",
                                              "deltaX", "deltaY", "depth", "basis", "depthBasis",
                                              "confidence", "depthConfidence"))
            face = positive_id(row.get("faceId"))
            track = positive_id(row.get("track"))
            if kind == "person_motion":
                if face is None or track is None:
                    continue
                facts.update(faceId=face, track=track)
            key = f"{kind}:{start}:{end}" if kind == "camera_motion" else f"{kind}:{face}:{track}:{start}:{end}"
            graph.node(key, kind, graph.names[kind], bounds, facts)
            graph.connect(key, "on_track", video_track)
            if shots:
                graph.connect(key, "in_shot", shots[0])
            if kind == "person_motion":
                graph.connect(key, "instance_of", graph.face(face))
                if face in bindings:
                    graph.connect(key, "describes", graph.speaker(bindings[face]))


def current_interval(rows, index, stamp):
    """Advance a cursor over sorted intervals while preserving the first matching interval."""
    while index < len(rows) and rows[index]["endMs"] <= stamp:
        index += 1
    row = rows[index] if index < len(rows) and rows[index]["startMs"] <= stamp else None
    return row, index


def selected_speakers(cue, stamp):
    """Use final cue assignment outside overlap and explicit participants inside it."""
    if not cue:
        return set(), False
    if cue.get("overlap") and not (cue.get("speakerName") or cue.get("speakerNumber")):
        windows = cue.get("overlaps") or []
        matching = [row for row in windows if row["startMs"] <= stamp < row["endMs"]]
        if matching or not windows:
            speakers = {positive_id(number) for row in matching for number in (row.get("speakers") or [])}
            return speakers - {None}, True
    speaker = positive_id(cue_number(cue))
    return ({speaker} if speaker else set()), False


def sustained_face(frames, index, track):
    """Require one visible active track on two samples either side of the sample."""
    if index < 2 or index + 2 >= len(frames):
        return False
    for frame in frames[index - 2:index + 3]:
        candidates = [hit[0] for hit in frame if len(hit) >= 6 and hit[5] == 1]
        if candidates != [track]:
            return False
    return True


def active_face_tracks(evidence, frames, index, cue, stamp, rows, speakers, mixed):
    """Highlight only raw activity consistent with an explicit final speaker or overlap."""
    candidates = [row["track"] for row, speaker, raw in rows if raw and speaker in speakers]
    if not candidates:
        return set()
    spans = evidence.get("simultaneousFaces") or []
    simultaneous = next((span for span in spans if span["startMs"] <= stamp < span["endMs"]), None)
    if simultaneous:
        return set(candidates).intersection(simultaneous.get("tracks") or [])
    if sum(raw for _, _, raw in rows) != 1:
        return set()
    track = candidates[0]
    records = evidence.get("phrases") or {}
    span = records.get(str(cue["startMs"]), records.get(cue["startMs"], {}))
    activity = span.get("activity") or {}
    if activity.get("state") == "visible-active" and activity.get("track") == track:
        return {track}
    return {track} if not mixed and sustained_face(frames, index, track) else set()


def face_samples(graph, report, cues, bindings):
    """Keep exact-clock nonempty face geometry outside the graph's relational node table."""
    evidence = report.get("videoEvidence") or {}
    timeline = evidence.get("timeline") or {}
    frames = timeline.get("frames") or []
    step = timeline.get("stepMs") or 40
    if type(step) is not int or step < 1:
        return [], None
    if not frames:
        return [], None
    raw_windows = (report.get("coverage") or {}).get("face")
    windows = None if raw_windows is None else sorted(
        (span for span in raw_windows if span.get("status") not in ("failed", "unavailable", "skipped", "uncovered")
         and source_interval(span, graph.duration)), key=lambda span: span["startMs"])
    output = []
    seen_shot_faces = set()
    cue_index = window_index = 0
    for index, frame in enumerate(frames):
        stamp = index * step
        if stamp >= graph.duration:
            break
        if not frame:
            continue
        if windows is not None:
            window, window_index = current_interval(windows, window_index, stamp)
            if window is None:
                continue
        shot_ids = tuple(graph.intersecting_shots(stamp, stamp + 1))
        cue, cue_index = current_interval(cues, cue_index, stamp)
        allowed, mixed = selected_speakers(cue, stamp)
        rows = []
        for hit in frame:
            if len(hit) < 6:
                continue
            track = positive_id(hit[0])
            identity = positive_id(hit[6] if len(hit) >= 7 else hit[0])
            box = box_coordinates(hit[1:5])
            if not track or not identity or not box:
                continue
            face_id = graph.face(identity)
            speaker = bindings.get(identity)
            speaker_id = graph.speaker(speaker) if speaker else None
            rows.append(({"track": track, "faceId": identity, "speakerId": speaker_id, "box": box}, speaker, hit[5] == 1))
            for shot in shot_ids:
                if (shot, face_id) not in seen_shot_faces:
                    seen_shot_faces.add((shot, face_id))
                    graph.shot_facts[shot]["faces"].add(face_id)
                    graph.connect(shot, "visible_face", face_id)
        if rows:
            active = active_face_tracks(evidence, frames, index, cue, stamp, rows, allowed, mixed) if cue else set()
            output.append((stamp, [{**row, "active": row["track"] in active} for row, _, _ in rows]))
    return output, step


def video_shot_cuts(report, duration):
    """Resolve the same authoritative native/legacy boundaries before coverage is materialised."""
    editor = (report.get("videoEvidence") or {}).get("editorTimeline") or {}
    colors = shot_color_rows(report, duration)
    cuts = editor.get("shots") or []
    if not isinstance(report.get("coverage"), dict):
        scene_cuts = (report.get("sceneEvidence") or {}).get("shotStartsMs")
        if colors:
            cuts = [{"startMs": start} for start, _, _ in colors]
        elif scene_cuts:
            cuts = [{"startMs": stamp} for stamp in scene_cuts]
    return cuts


def add_video_shots(graph, report, coverage, video_track):
    """Use the same shot boundaries for graph snapshots and timeline object tracks."""
    colors = shot_color_rows(report, graph.duration)
    cuts = video_shot_cuts(report, graph.duration)
    graph.shots(cuts, coverage.get("shots") or [], video_track)
    assign_shot_colors(graph, colors)
    for mood in (report.get("sceneEvidence") or {}).get("moodbook") or []:
        if not isinstance(mood, dict) or not (bounds := source_interval(mood, graph.duration)):
            continue
        shot = graph.nodes.get(f"shot:{bounds[0]}")
        if shot is not None and bounds[1] <= shot["endMs"]:
            shot["data"]["moodbook"] = mood


def object_layers(report):
    """Project shot-local tracks, measured box samples and profiles onto the editor source clock."""
    scene = report["sceneEvidence"]
    if not any(row.get("objects") or any(isinstance(panel, dict) and panel.get("panelType") in PANEL_TYPES
               for panel in row.get("interfaces") or ()) for row in scene.get("frames", [])):
        return {"objectTimeline": [], "objectProfiles": []}
    editor = (report.get("videoEvidence") or {}).get("editorTimeline") or {}
    duration = max(report["durationMs"], editor.get("durationMs", 0),
                   max(row["timeMs"] + 1 for row in report["sceneEvidence"]["frames"]))
    graph = MediaGraph(duration, report.get("language"))
    _, video_track = stream_tracks(graph, report, "video")
    add_video_shots(graph, report, source_coverage(report, duration), video_track)
    scene_observations(graph, report)
    review = bool(report.get("objectReconciliationEnabled"))
    tracks = add_object_tracks(graph, object_reconciliation=review)
    profiles = add_object_profiles(graph, tracks, object_reconciliation=review)
    previews = panel_scroll_previews(graph, tracks, scene.get("interfaceScrolls", ()), box_coordinates)
    clips = []
    for key, shot, samples in tracks:
        node = graph.nodes[key]
        clips.append({field: node[field] for field in ("id", "label", "startMs", "endMs")}
                     | {"shotId": shot, "sampleCount": len(samples),
                        "representative": node["data"]["representative"], "profileId": node["data"]["profileId"],
                        "boxSamples": [[sample["startMs"], sample["data"]["box"], sample["data"].get("score")]
                                       for sample in samples]})
    return {"objectTimeline": sorted(clips, key=lambda clip: (clip["startMs"], clip["endMs"], clip["id"])),
            "objectProfiles": [{"id": node["id"], "label": node["label"], **node["data"],
                                **({"scrollPreviews": previews[node["id"]]} if node["id"] in previews else {})}
                               for node in profiles]}


def build_graph(report, *, kind, duration_ms, object_reconciliation=False):
    """Create one media graph snapshot and return logical nodes, edges, geometry and coverage."""
    if kind not in ("video", "audio") or type(duration_ms) is not int or duration_ms <= 0:
        raise ValueError("A media graph requires a kind and positive duration in milliseconds.")
    graph = MediaGraph(duration_ms, report.get("language"))
    coverage = source_coverage(report, duration_ms)
    audio_track, video_track = stream_tracks(graph, report, kind)
    if video_track:
        add_video_shots(graph, report, coverage, video_track)
    bindings = bind_profiles(graph, report)
    cues = add_utterances(graph, report, audio_track)
    add_dialogue_pauses(graph, report, audio_track)
    add_audio_layers(graph, report, audio_track)
    if kind == "video":
        scene_observations(graph, report, keep_display_aliases=True)
        add_ui_activity(graph, report, box_coordinates)
        person_observations(graph, report)
        add_person_counts(graph, report)
        add_motion(graph, report, video_track, bindings)
        add_frame_changes(graph, report.get("sceneEvidence") or {}, video_track, box_coordinates)
        faces, step = face_samples(graph, report, cues, bindings)
        add_facial_expressions(graph, report, video_track)
        review = object_reconciliation or bool(report.get("objectReconciliationEnabled"))
        tracks = add_object_tracks(graph, object_reconciliation=review)
        add_object_profiles(graph, tracks, object_reconciliation=review)
        add_scene_hierarchy(graph)
        add_speaker_context(graph, tracks, faces, step)
        graph.observations_summary()
    else:
        faces, step = [], None
    add_dialogue_review(graph, report, source_interval)
    if video_track:
        add_speech_screen_evidence(graph)
    extensions = add_mod_extensions(graph, report)
    activity = (report.get("sceneEvidence") or {}).get("uiActivity")
    if not isinstance(activity, dict):
        activity = {}
    return {"nodes": list(graph.nodes.values()), "edges": sorted(graph.edges), "faceFrames": faces,
            "faceSampleMs": step, "coverage": coverage, "modExtensions": extensions,
            "speakerScope": report.get("speakerScope") or "source",
            "uiActivity": {"version": activity.get('version', 1),
                           "geometryDiscoveryIntervalMs": activity.get('geometryDiscoveryIntervalMs'),
                           "meaning": activity.get('meaning', 'visual-activity-candidate'),
                           "truncated": activity.get("truncated") is True
                           or len(activity.get('panels') or []) > 64 or len(activity.get('samples') or []) > 100000}}
