"""Score time-aligned local audio and tracked faces with the pinned Light-ASD model."""

from voice_studio.mod_runtime import model_resource, is_replacement

from argparse import ArgumentParser
from base64 import b64encode
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from dataclasses import dataclass
from itertools import islice
from functools import lru_cache
import os
import sys
from pathlib import Path
from tempfile import TemporaryFile
from threading import local
from typing import BinaryIO
from time import perf_counter
import json

import cv2
import numpy as np
from voice_studio.vision.adaptive_faces import ANCHOR_STEP, ScanFrame, detect_adaptively
from voice_studio.core.analysis_stages import MAX_ACTIVITY_CUES
from voice_studio.vision.clothing_review import review_clothing
from voice_studio.speech.speaker_photos import collect as collect_photos, thumbnail
from voice_studio.vision.face_reid import Signatures, distinct_faces, face_references, group_tracks
from voice_studio.core.media_resources import stage_threads
from voice_studio.vision.visual_attributes import visible_attributes
from voice_studio.vision.facial_expression_model import ExpressionClassifier

cv2.setNumThreads(1)

def overlap(left, right):
    """Measure IoU between two face boxes."""
    lx, ly, lw, lh = left
    rx, ry, rw, rh = right
    width = max(0, min(lx + lw, rx + rw) - max(lx, rx))
    height = max(0, min(ly + lh, ry + rh) - max(ly, ry))
    intersection = width * height
    return intersection / (lw * lh + rw * rh - intersection)

class GrayFrames:
    """Keep decoded image bytes on a temporary file rather than in the worker heap."""

    def __init__(self, storage: BinaryIO, width: int, height: int):
        self.storage = storage
        self.shape = (height, width)
        self.frame_size = width * height
        self.count = 0

    def append(self, frame):
        """Write one packed grayscale frame during the sequential scan."""
        self.storage.write(memoryview(frame))
        self.count += 1

    def __len__(self):
        return self.count

    def __getitem__(self, index):
        if not 0 <= index < self.count:
            raise IndexError(index)
        self.storage.seek(index * self.frame_size)
        raw = self.storage.read(self.frame_size)
        return np.frombuffer(raw, dtype=np.uint8).reshape(self.shape)


@dataclass
class FaceScan:
    """Decoded activity input plus bounded previews for the editor timeline."""

    frames: GrayFrames
    tracks: list
    width: int
    height: int
    shots: list
    previews: dict
    identity_cuts: list
    estimated_frames: list
    person_pilot: dict | None = None
    expressions: object = None


def capture_shot(frame, frame_number, shots):
    """Record each cut and attach a small color preview for the first 60 shots."""
    shot = {"startMs": frame_number * 40}
    if len(shots) < 60:
        height, width = frame.shape[:2]
        scale = 120 / max(width, height)
        size = (max(1, round(width * scale)), max(1, round(height * scale)))
        ok, encoded = cv2.imencode(".jpg", cv2.resize(frame, size), [cv2.IMWRITE_JPEG_QUALITY, 72])
        if ok:
            shot["jpeg"] = b64encode(encoded).decode("ascii")
    shots.append(shot)


def capture_face(frame, detection, track, previews, time_ms):
    """Keep a good color crop per candidate track without encoding every frame."""
    size = min(detection[2:4])
    quality = float(size * detection[-1])
    prior = previews.get(track)
    if size < 28 or (prior is None and len(previews) >= 48) or (prior and quality < prior[0] * 1.2):
        return
    crop = thumbnail(frame, detection)
    if crop:
        previews[track] = (quality, crop, visible_attributes(frame, detection, time_ms))


def group_face_previews(previews, aliases):
    """Choose a representative crop and its visual observations per stable face."""
    chosen = {}
    for track, (quality, crop, attributes) in previews.items():
        face = aliases[track]
        if face not in chosen or quality > chosen[face][0]:
            chosen[face] = (quality, crop, attributes)
    photos = {face: crop for face, (_, crop, _) in chosen.items()}
    attributes = {face: data for face, (_, _, data) in chosen.items() if data}
    return photos, attributes


def source_frames(video, frames, height):
    """Decode the original frame clock and check cuts even when skipping YuNet."""
    previous = None
    while True:
        success, frame = video.read()
        if not success:
            return
        number = len(frames)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frames.append(gray)
        # On a vertical repost the UI can stay static while the central shot changes.
        reduced = cv2.resize(gray[height // 4:height * 7 // 10], (32, 32), interpolation=cv2.INTER_AREA)
        cut, suspect = number == 0, False
        if previous is not None:
            difference = cv2.absdiff(reduced, previous)
            large_change = np.mean(difference > 45)
            mean_change = np.mean(difference)
            cut = large_change >= .39 and mean_change >= 34
            suspect = not cut and large_change >= .22 and mean_change >= 24
        previous = reduced
        yield ScanFrame(number, frame, cut, suspect, gray)


def load_frames(video_path, yunet_path, signatures, storage, sampling="dense", pilot=None, expressions=None):
    """Spool grayscale frames and match detections against recent same-shot tracks."""
    video = cv2.VideoCapture(str(video_path))
    try:
        if not video.isOpened():
            raise ValueError('Cannot open the converted video')
        if abs(video.get(cv2.CAP_PROP_FPS) - 25) > 0.01:
            raise ValueError('Video must be prepared at 25 FPS')
        yunet_path = model_resource('vision.faces', 'builtin.yunet', yunet_path)
        width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        def create_detector():
            """Use identical model parameters for every independent detector."""
            return cv2.FaceDetectorYN.create(str(yunet_path), '', (width, height), 0.78, 0.30, 500)

        frames = GrayFrames(storage, width, height)
        source = source_frames(video, frames, height)
        with closing(detect_source_frames(source, create_detector, sampling)) as detections:
            return scan_tracks(detections, signatures, frames, width, height, pilot, expressions)
    finally:
        video.release()


def detect_source_frames(source, create_detector, sampling):
    """Select dense or adaptive detection while preserving source frame order."""
    cores = stage_threads(os.cpu_count() or 1)
    workers = os.getenv("VOICE_YUNET_WORKERS", "2" if cores >= 8 else "1")
    if sampling == "dense" and workers == "2" and cores >= 6:
        yield from parallel_detections(source, create_detector)
        return
    detector = create_detector()
    if sampling == "adaptive":
        yield from detect_adaptively(source, lambda frame: detector.detect(frame)[1])
    else:
        for sample in source:
            yield sample, detector.detect(sample.color)[1], True


def parallel_detections(source, create_detector):
    """Use two separate YuNet instances with at most four decoded frames queued."""
    state = local()

    def initialize():
        """Each OpenCV worker owns its detector and input workspace."""
        state.detector = create_detector()

    def detect(sample):
        """Return measurements linked to the exact input frame."""
        return sample, state.detector.detect(sample.color)[1], True

    with ThreadPoolExecutor(max_workers=2, initializer=initialize, thread_name_prefix="yunet") as workers:
        while batch := list(islice(source, 4)):
            yield from workers.map(detect, batch)


def scan_tracks(detections_by_frame, signatures, frames, width, height, pilot=None, expressions=None):
    """Build tracks in source order and use measured faces for identity and photos."""
    tracks = []
    recent = {}
    shots, previews = [], {}
    identity_cuts, estimated_frames = [], []
    for sample, detections, measured in detections_by_frame:
        if pilot is not None:
            detections = pilot.augment(sample, detections, measured)
        frame_number, frame = sample.number, sample.color
        if sample.cut:
            recent.clear()
            capture_shot(frame, frame_number, shots)
        if not measured:
            estimated_frames.append(frame_number)
        if detections is None:
            continue
        recent = {index: seen for index, seen in recent.items() if frame_number - seen <= 5}
        matches = []
        for track_index in recent:
            track = tracks[track_index]
            for detection_index, detection in enumerate(detections):
                iou = overlap(track[-1][1], detection[:4])
                if iou >= 0.3:
                    matches.append((iou, track_index, detection_index))
        assigned_tracks, assigned_detections = set(), set()
        rejected = {}
        has_identity_cut = False
        for _, track_index, detection_index in sorted(matches, reverse=True):
            if track_index in assigned_tracks or detection_index in assigned_detections or detection_index in rejected:
                continue
            detection = detections[detection_index]
            vector = ()
            if sample.suspect_cut:
                different, vector = signatures.check_cut(track_index + 1, frame, detection)
                if different:
                    recent.pop(track_index, None)
                    rejected[detection_index] = vector
                    assigned_tracks.add(track_index)
                    has_identity_cut = True
                    continue
            tracks[track_index].append((frame_number, detection[:4]))
            recent[track_index] = frame_number
            if measured:
                signatures.observe(track_index + 1, frame_number, frame, detection, vector)
                capture_face(frame, detection, track_index + 1, previews, frame_number * 40)
                if expressions is not None:
                    expressions.observe(frame_number, frame, detection, track_index + 1)
            assigned_tracks.add(track_index)
            assigned_detections.add(detection_index)
        for detection_index, detection in enumerate(detections):
            if measured and detection_index not in assigned_detections:
                tracks.append([(frame_number, detection[:4])])
                recent[len(tracks) - 1] = frame_number
                signatures.observe(len(tracks), frame_number, frame, detection, rejected.get(detection_index, ()))
                capture_face(frame, detection, len(tracks), previews, frame_number * 40)
                if expressions is not None:
                    expressions.observe(frame_number, frame, detection, len(tracks))
        if has_identity_cut:
            identity_cuts.append(frame_number * 40)
            capture_shot(frame, frame_number, shots)
    return FaceScan(frames, tracks, width, height, shots, previews, identity_cuts, estimated_frames,
                    pilot.data() if pilot is not None else None, expressions)

def mouth_crop(frame, size, x_center, y_center):
    """Extract the model crop with the same constant border as a padded full frame."""
    padding = int(size * 1.8)
    x_center += padding
    y_center += padding
    height, width = frame.shape
    top, bottom, _ = slice(int(y_center - size), int(y_center + size * 1.8)).indices(height + 2 * padding)
    left, right, _ = slice(int(x_center - size * 1.4), int(x_center + size * 1.4)).indices(width + 2 * padding)
    top, bottom, left, right = (value - padding for value in (top, bottom, left, right))
    crop = frame[max(0, top):max(0, bottom), max(0, left):max(0, right)]
    if not crop.size:
        return np.full((max(0, bottom - top), max(0, right - left)), 110, dtype=frame.dtype)
    border = (max(0, -top), max(0, bottom - height), max(0, -left), max(0, right - width))
    if any(border):
        crop = np.pad(crop, (border[:2], border[2:]), mode="constant", constant_values=110)
    return crop


def build_features(frames, track, samples):
    """Align normalized face crops with 16 kHz MFCC features."""
    from voice_studio.vendor.speech_features.base import mfcc as speech_mfcc
    from scipy.ndimage import median_filter

    source_frames, boxes = zip(*track)
    box_matrix = np.array(boxes)
    frame_indices = np.arange(source_frames[0], source_frames[-1] + 1)
    values = np.array([np.interp(frame_indices, source_frames, column) for column in box_matrix.T])
    x, y, width, height = values
    sizes, centers_x, centers_y = [
        median_filter(value, size=13, mode='nearest')
        for value in [np.maximum(width, height) / 2, x + width / 2, y + height / 2]
    ]
    audio = samples[source_frames[0] * 640:(source_frames[-1] + 1) * 640]
    mfcc = speech_mfcc(audio, 16000, numcep=13, winlen=.025, winstep=.01)
    count = min(len(frame_indices), len(mfcc) // 4)
    # Only retain the model's 112px crop and reuse one resize buffer.
    crops = np.empty((count, 112, 112), dtype=np.uint8)
    resized = np.empty((224, 224), dtype=np.uint8)
    for index in range(count):
        crop = mouth_crop(frames[frame_indices[index]], sizes[index], centers_x[index], centers_y[index])
        cv2.resize(crop, (224, 224), dst=resized)
        crops[index] = resized[56:168, 56:168]
    return crops, mfcc[:count * 4]

@lru_cache(maxsize=1)
def load_model(weights_path):
    """Load the pinned CPU checkpoint into its network and classifier."""
    import torch
    from voice_studio.vendor.light_asd.Model import ASD_Model
    from voice_studio.vendor.light_asd.loss import lossAV

    torch.set_num_threads(stage_threads(4))
    weights = torch.load(weights_path, map_location='cpu', weights_only=True)
    network = ASD_Model()
    classifier = lossAV()
    for prefix, module in [('model.', network), ('lossAV.', classifier)]:
        module.load_state_dict({k.removeprefix(prefix): v for k, v in weights.items() if k.startswith(prefix)})
        module.eval()
    return network, classifier

def audio_visual_scores(network, classifier, video, audio):
    """Average the same six votes, computing identical clipped windows only once."""
    import torch

    votes = []
    windows = {}
    for duration in (1, 2, 3, 4, 5, 6):
        window_scores = []
        for begin in range(0, len(video), duration * 25):
            end = min(len(video), begin + duration * 25)
            bounds = (begin, end)
            if bounds not in windows:
                a = torch.as_tensor(audio[None, begin * 4:end * 4], dtype=torch.float32)
                v = torch.as_tensor(video[None, begin:end], dtype=torch.float32)
                embed_a = network.forward_audio_frontend(a)
                embed_v = network.forward_visual_frontend(v)
                scores = classifier(network.forward_audio_visual_backend(embed_a, embed_v), labels=None)
                windows[bounds] = scores.tolist()
            window_scores.extend(windows[bounds])
        votes.append(window_scores)
    return np.mean(votes, axis=0)



def ceil_frame(milliseconds):
    """Convert a millisecond boundary to the first frame starting at or after it."""
    return (milliseconds + 39) // 40

def summarize(scores, track, span, frames):
    """Aggregate activity inside a phrase and crop the exact highest-scoring frame."""
    track_start = track[0][0]
    start = max(track_start, ceil_frame(span['start_ms']))
    end = min(track_start + len(scores), ceil_frame(span['end_ms']))
    if start >= end:
        return None
    segment = scores[start - track_start:end - track_start]
    matched_detections = [(frame, bbox) for frame, bbox in track if start <= frame < end]
    representative = None
    if matched_detections:
        frame, box = max(matched_detections, key=lambda result: scores[result[0] - track_start])
        if scores[frame - track_start] >= 0:
            representative = {
                "time_ms": frame * 40,
                "jpeg": thumbnail(frames[frame], box),
            }
    return {
        'frames': len(segment),
        'positive_frames': int(np.sum(segment >= 0)),
        'mean_logit': round(float(np.mean(segment)), 3),
        'representative': representative,
    }

def run(args, *, include_identity=False):
    """Score tracked faces, with private descriptors only for the window protocol."""
    start = perf_counter()
    signatures = Signatures(args.sface)
    pilot = None
    if args.person_pilot_source:
        from voice_studio.vision.person_face_pilot import PersonFacePilot

        pilot = PersonFacePilot(args.person_pilot_source, args.person_pilot_formats,
                                args.person_pilot_offset, args.person_pilot_model, args.yunet,
                                count_only=args.person_pilot_count_only)
    with TemporaryFile() as storage:
        expressions = ExpressionClassifier(getattr(args, "expressions_model", None))
        scan = load_frames(args.video, args.yunet, signatures, storage, args.yunet_sampling, pilot, expressions)
        result = score_scan(args, signatures, scan, start, include_identity=include_identity)
    if args.photo_source:
        try:
            result["phrasePhotos"] = collect_photos(str(args.photo_source), str(args.yunet),
                                                   args.photo_formats, args.photo_phrases)
        except (OSError, cv2.error, ValueError):
            # Keep completed activity results. The parent can retry photos independently.
            pass
    return result


def score_spans(args, frames, tracks, aliases, spans):
    """Load Light-ASD dependencies only for windows with audio evidence to score."""
    import torch
    from scipy.io import wavfile

    sample_rate, samples = wavfile.read(args.audio)
    if sample_rate != 16000 or samples.ndim != 1 or samples.dtype != np.int16:
        raise ValueError('Audio must be 16 kHz mono 16-bit PCM WAV')
    network, classifier = load_model(model_resource('vision.active-speaker', 'builtin.light-asd', args.weights))
    activity = {}
    with torch.inference_mode():
        for number, track in enumerate(tracks, 1):
            if len(track) < 12:
                continue
            first_frame, last_frame = track[0][0], track[-1][0] + 1
            overlapping = [
                span for span in spans
                if first_frame < ceil_frame(span['end_ms']) and last_frame > ceil_frame(span['start_ms'])
            ]
            if not overlapping or first_frame * 640 >= len(samples):
                continue
            video, audio = build_features(frames, track, samples)
            if len(video) < 2:
                continue
            scores = audio_visual_scores(network, classifier, video, audio)
            activity[number] = scores
            for span in overlapping:
                result = summarize(scores, track, span, frames)
                if result:
                    result.update(track=number, face=aliases[number],
                                  track_start_ms=first_frame * 40, track_end_ms=last_frame * 40)
                    span['candidates'].append(result)
    return activity


def score_scan(args, signatures, scan, start, *, include_identity=False):
    """Package a scan, optionally adding descriptors for private window transport."""
    frames, tracks = scan.frames, scan.tracks
    qualities = {track: preview[0] for track, preview in scan.previews.items()}
    aliases, matches = group_tracks(tracks, signatures.samples, qualities=qualities)
    tracked = perf_counter()
    spans = [dict(span, candidates=[]) for span in args.spans_ms]
    activity = score_spans(args, frames, tracks, aliases, spans) if spans else {}
    scored = perf_counter()
    active_faces = appearance_priorities(spans)
    aliases, outfits, clothing = review_clothing(args.video, tracks, signatures.samples, aliases,
                                                scan.width, scan.height, active_faces)
    for span in spans:
        for candidate in span["candidates"]:
            candidate["face"] = aliases[candidate["track"]]
    photos, attributes = group_face_previews(scan.previews, aliases)
    for face, observation in clothing.pop("appearance", {}).items():
        attributes.setdefault(face, {}).update(observation)
    result = {
        'frame_count': len(frames),
        'frame_width': scan.width,
        'frame_height': scan.height,
        'fps': 25,
        'seconds': {
            'detect_and_track': tracked - start,
            'crop_and_score': scored - tracked,
        },
        'positive_logit_threshold': 0,
        'faceTracking': {"status": signatures.status, "tracks": len(tracks),
                         "faces": len(set(aliases.values())), "matches": matches, "samples": signatures.count,
                         "identityCuts": len(scan.identity_cuts), "identityCutTimesMs": scan.identity_cuts,
                         "distinctFaces": distinct_faces(aliases, signatures.samples),
                         "clothing": clothing},
        'timeline': pack_timeline(tracks, activity, aliases, len(frames), scan.width, scan.height),
        'editorTimeline': {
            'durationMs': len(frames) * 40, 'shots': scan.shots,
            'facePhotos': photos, 'faceAttributes': attributes, 'clothingByFace': outfits,
        },
        'spans': spans,
    }
    if scan.expressions is not None:
        result["facialExpressions"] = scan.expressions.data(aliases)
    if args.yunet_sampling == "adaptive":
        result["timeline"]["estimatedFrames"] = scan.estimated_frames
        result["faceTracking"]["sampling"] = {
            "mode": "adaptive", "anchorStep": ANCHOR_STEP,
            "detectorFrames": len(frames) - len(scan.estimated_frames),
            "estimatedFrames": len(scan.estimated_frames),
        }
    if scan.person_pilot is not None:
        result["editorTimeline"]["personPilot"] = scan.person_pilot
    if include_identity:
        # ActivityTimeline consumes these source-local vectors before publishing
        # or caching results. Standalone worker output never contains them.
        result["_faceSignatures"] = face_references(aliases, signatures.samples, qualities)
    return result


def appearance_priorities(spans):
    """Prioritize one dominant visible talker per cue within the existing sample budget."""
    faces = []
    for span in spans:
        if not span["candidates"]:
            continue
        candidate = max(span["candidates"], key=lambda row: row["positive_frames"])
        if candidate["positive_frames"] >= 10 and candidate["positive_frames"] * 4 >= candidate["frames"] * 3:
            faces.append(candidate["face"])
    return faces

def pack_timeline(tracks, activity, aliases, count, width, height):
    """Return track samples at 40 ms intervals without extending beyond their bounds."""
    rows = [[] for _ in range(count)]
    for number, track in enumerate(tracks, 1):
        if len(track) < 3:
            continue
        scores = activity.get(number, ())
        origin = track[0][0]
        for index, (x, y, w, h) in track:
            offset = index - origin
            speaking = offset < len(scores) and scores[offset] >= 0
            left, top = max(0, x), max(0, y)
            w, h = min(width, x + w) - left, min(height, y + h) - top
            if w > 0 and h > 0:
                box = [round(float(value / size), 4)
                       for value, size in ((left, width), (top, height), (w, width), (h, height))]
                rows[index].append([number, *box, int(speaking), aliases[number]])
    return {"stepMs": 40, "frames": rows}


def parse_spans(value):
    """Check the bounded list of millisecond intervals."""
    data = json.loads(value)
    if not isinstance(data, list) or len(data) > MAX_ACTIVITY_CUES:
        raise ValueError(f'Provide up to {MAX_ACTIVITY_CUES} spans')
    for span in data:
        if not isinstance(span, dict):
            raise ValueError('Each span must have start_ms and end_ms')
        start = span.get('start_ms')
        end = span.get('end_ms')
        if type(start) is not int or type(end) is not int or not 0 <= start < end:
            raise ValueError('Each span must satisfy 0 <= start_ms < end_ms with integer times')
    return [{'start_ms': span['start_ms'], 'end_ms': span['end_ms']} for span in data]

def parse_arguments(arguments=None):
    """Read paths to the prepared video, audio, detection model and checkpoint."""
    parser = ArgumentParser(description='Run Light-ASD on YuNet tracks locally.')
    parser.add_argument('--video', type=Path, required=True)
    parser.add_argument('--audio', type=Path, required=True)
    parser.add_argument('--yunet', type=Path, required=True)
    parser.add_argument('--yunet-sampling', choices=('dense', 'adaptive'), default='dense')
    parser.add_argument('--sface', type=Path)
    parser.add_argument('--expressions-model', type=Path)
    parser.add_argument('--weights', type=Path, required=True)
    parser.add_argument('--spans-ms', type=parse_spans, required=True)
    parser.add_argument('--photo-source', type=Path)
    parser.add_argument('--photo-formats')
    parser.add_argument('--photo-phrases', type=json.loads)
    parser.add_argument('--person-pilot-source', type=Path)
    parser.add_argument('--person-pilot-model', type=Path)
    parser.add_argument('--person-pilot-formats')
    parser.add_argument('--person-pilot-offset', type=int, default=0)
    parser.add_argument('--person-pilot-count-only', action='store_true')
    return parser.parse_args(arguments)

def serve_windows():
    """Accept one internal request at a time and keep model weights between windows."""
    from contextlib import redirect_stdout

    output = sys.stdout
    for line in sys.stdin:
        try:
            with redirect_stdout(sys.stderr):
                result = run(parse_arguments(json.loads(line)), include_identity=True)
            reply = json.dumps({"result": result}, separators=(",", ":"), allow_nan=False)
        except (Exception, SystemExit):
            reply = '{"error":"window-failed"}'
        output.write(reply + "\n")
        output.flush()


if __name__ == '__main__':
    if sys.argv[1:] == ["--window-server"]:
        serve_windows()
    else:
        print(json.dumps(run(parse_arguments()), separators=(',', ':'), allow_nan=False))
