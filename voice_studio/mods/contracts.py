"""Host-owned versioned JSON contracts, units and semantic media validation.

These contracts cover custom JSON provider boundaries. They do not describe the
opaque Python objects passed between existing native builtin implementations.
"""

from copy import deepcopy
import math
from pathlib import Path

from .errors import ModError

RESOURCE_CAPABILITIES = (
    "speech.embedding-resource", "speech.metric-resource", "vision.faces", "vision.face-embedding",
    "vision.active-speaker", "vision.expressions", "vision.objects", "vision.gaze",
    "vision.segmentation", "vision.appearance", "synthesis.accent-resource",
    "synthesis.vocoder-resource", "synthesis.tokenizer-resource", "prosody.context", "prosody.lexical",
)
OUTPUT_IDS = {"speech.asr": "studio.asr", "speech.diarization": "studio.diarization",
              "speech.embedding": "studio.embedding", "audio.classification": "studio.audio-classification",
              "translation.text": "studio.translations",
              "vision.visual-batch": "studio.visual", "vision.scene-batch": "studio.scene"}
for _capability in RESOURCE_CAPABILITIES:
    OUTPUT_IDS[_capability] = "studio.model-resource"


def _object_schema(properties=None, required=()):
    """Describe a JSON object while retaining native extension metadata fields."""
    return {"type": "object", "properties": properties or {}, "required": list(required), "additionalProperties": True}


def _array_schema(items):
    """Describe a bounded collection used by all published canonical contracts."""
    return {"type": "array", "items": items, "maxItems": 100000}


_TIME = {"type": "integer", "minimum": 0, "maximum": 9007199254740991}
_SCORE = {"type": "number", "minimum": 0, "maximum": 1}
_BOX = {"type": "array", "items": _SCORE, "minItems": 4, "maxItems": 4,
        "description": "Normalized [x, y, width, height]; x+width and y+height must not exceed 1."}
_SEGMENT = _object_schema({"text": {"type": "string"},
                           "offsets": _object_schema({"from": _TIME, "to": _TIME}, ("from", "to"))},
                          ("text", "offsets"))
_DEFINITIONS = {
    "studio.asr": {
        "schema": _array_schema(_object_schema({"transcription": _array_schema(_SEGMENT)}, ("transcription",))),
        "units": {"transcription[].offsets.from": "milliseconds", "transcription[].offsets.to": "milliseconds"},
        "supportLimits": ["One result per input job when jobs are supplied.", "Segment starts are nondecreasing; end >= start; zero durations are permitted for untimed translation.", "Native optional callback/partial-result Python contracts are not covered."]},
    "studio.diarization": {
        "schema": _array_schema(_object_schema({"startMs": _TIME, "endMs": _TIME,
                                                 "speaker": {"type": "integer", "minimum": 1}},
                                                ("startMs", "endMs", "speaker"))),
        "units": {"startMs": "milliseconds", "endMs": "milliseconds", "speaker": "one-based integer ID"},
        "supportLimits": ["Turns are sorted by startMs; endMs > startMs. Overlapping speakers are permitted."]},
    "studio.model-resource": {
        "schema": _object_schema({"modelPath": {"type": "string", "minLength": 1}}, ("modelPath",)),
        "units": {"modelPath": "absolute existing local file path"},
        "supportLimits": ["A compatible existing artifact path only; format/runtime compatibility remains the native loader's responsibility.", "No weights or dependency installation; external artifact bytes are not package-content pinned."]},
    "studio.embedding": {
        "schema": {"type": "object", "additionalProperties": {"type": "array", "items": {"type": "array", "items": {"type": "number"}, "minItems": 1}}},
        "units": {"values": "finite dimensionless vector components"},
        "supportLimits": ["Speaker keys are strings. Nonempty vectors within a result have one common dimension; empty speaker sample lists are permitted."]},
    "studio.audio-classification": {
        "schema": {"type": "array", "minItems": 2, "maxItems": 2,
                   "prefixItems": [_array_schema(_object_schema({"start": {"type": "number", "minimum": 0}, "end": {"type": "number", "minimum": 0},
                       "scores": _array_schema(_object_schema({"label": {"type": "string"}, "score": _SCORE}, ("label", "score")))}, ("start", "end", "scores"))), {"type": "string"}]},
        "units": {"observations[].start": "seconds", "observations[].end": "seconds", "scores[].score": "probability in [0,1]"},
        "supportLimits": ["JSON pair [observations, status]; observations ordered by start, end >= start."]},
    "studio.visual": {
        "schema": _object_schema({"stages": {"type": "object"}, "phrases": {"type": "object"}}, ("stages", "phrases")),
        "units": {},
        "supportLimits": ["Pilot validates report envelope and finite JSON only; opaque stage/phrase payload internals and optional timelines have no canonical guarantees yet."]},
    "studio.scene": {
        "schema": _object_schema({"status": {"type": "string"}, "frames": _array_schema(_object_schema({"timeMs": _TIME,
            "objects": _array_schema(_object_schema({"label": {"type": "string"}, "score": _SCORE, "box": _BOX}, ("label", "score", "box"))),
            "text": _array_schema(_object_schema({"text": {"type": "string"}, "box": _BOX}, ("text", "box")))}, ("timeMs", "objects", "text")))}, ("status", "frames")),
        "units": {"frames[].timeMs": "milliseconds", "box": "normalized [x,y,width,height]", "score": "probability in [0,1]"},
        "supportLimits": ["Frames sorted by timeMs; all normalized boxes fit inside the image. Optional motion, color, shot and interface metadata is finite JSON but otherwise not standardized."]},
    "studio.translations": {
        "schema": _object_schema({"translations": {"type": "array", "minItems": 1, "maxItems": 400,
                                    "items": {"type": "string", "minLength": 1, "maxLength": 1000}}}, ("translations",)),
        "units": {},
        "supportLimits": ["Input uses sourceLanguage, targetLanguage and texts. Return one translation per text in input order."]},
    "studio.named-input": {
        "schema": {"type": "object", "additionalProperties": True}, "units": {},
        "supportLimits": ["Finite JSON object containing named arguments from the documented host batch operation.", "Operation-specific required fields, units and opaque builtin inputs are not standardized by this generic input envelope."]},
}


def get_contracts():
    """Return detached metadata, machine-readable schemas, units and explicit pilot limits."""
    records = []
    for contract_id, definition in _DEFINITIONS.items():
        input_contract = contract_id == "studio.named-input"
        capabilities = sorted(OUTPUT_IDS) if input_contract else sorted(cap for cap, name in OUTPUT_IDS.items() if name == contract_id)
        records.append({"id": contract_id, "version": 1, "direction": "input" if input_contract else "output",
                        "capabilities": capabilities, **deepcopy(definition)})
    return {"apiVersion": 1, "contracts": records}


def contract_for_capability(capability, direction="output"):
    """Return a supported default contract reference, or None for an unsupported pilot boundary."""
    if direction not in {"input", "output"}:
        raise ModError("Contract direction must be input or output")
    contract_id = "studio.named-input" if direction == "input" and capability in OUTPUT_IDS else OUTPUT_IDS.get(capability)
    return {"id": contract_id, "version": 1} if contract_id else None


def resolve_contract(reference, *, capability=None, direction="output"):
    """Reject unknown versions, wrong directions and capability-incompatible declarations without imports."""
    if (not isinstance(reference, dict) or set(reference) != {"id", "version"}
            or not isinstance(reference["id"], str) or type(reference["version"]) is not int
            or reference["version"] != 1 or reference["id"] not in _DEFINITIONS):
        raise ModError("Unsupported host contract reference; use a published exact id and integer version")
    expected_direction = "input" if reference["id"] == "studio.named-input" else "output"
    if direction != expected_direction:
        raise ModError(f"Contract {reference['id']} is not an {direction} contract")
    if capability is not None and contract_for_capability(capability, direction) != reference:
        raise ModError(f"Contract {reference['id']}:v1 is incompatible with capability {capability}")
    return reference["id"]


def _fail(path, message):
    """Raise an actionable error anchored at the exact canonical JSON path."""
    raise ModError(f"{path}: {message}")


def _finite(value, path, depth=0):
    """Reject non-JSON values, booleans as numbers, nonfinite extras and pathological nesting."""
    if depth > 64:
        _fail(path, "JSON nesting exceeds 64 levels")
    if value is None or type(value) in (str, bool, int):
        return
    if type(value) is float:
        if not math.isfinite(value):
            _fail(path, "number must be finite")
        return
    if isinstance(value, list):
        for index, child in enumerate(value):
            _finite(child, f"{path}[{index}]", depth + 1)
    elif isinstance(value, dict):
        for key, child in value.items():
            if not isinstance(key, str):
                _fail(path, "object keys must be strings")
            _finite(child, f"{path}.{key}", depth + 1)
    else:
        _fail(path, "expected finite JSON")


def _object(value, path, required=()):
    """Require a JSON object with path-aware errors for absent fields."""
    if not isinstance(value, dict):
        _fail(path, "expected object")
    for key in required:
        if key not in value:
            _fail(f"{path}.{key}", "required field is missing")


def _rows(value, path):
    """Require a bounded array and return its path-indexed members."""
    if not isinstance(value, list) or len(value) > 100000:
        _fail(path, "expected array with at most 100000 items")
    return enumerate(value)


def _number(value, path, minimum=None, maximum=None, integer=False):
    """Validate finite numeric values without coercing booleans, strings or fractional timestamps."""
    try:
        finite = type(value) in ((int,) if integer else (int, float)) and math.isfinite(value)
    except OverflowError:
        finite = False
    if not finite:
        _fail(path, "expected finite integer" if integer else "expected finite number")
    if minimum is not None and value < minimum or maximum is not None and value > maximum:
        _fail(path, f"number must be in [{minimum}, {maximum}]")


def _text(value, path):
    """Require text while allowing empty transcript text and extension metadata."""
    if not isinstance(value, str):
        _fail(path, "expected string")


def _interval(row, path, left, right, previous, *, milliseconds=True, positive=False):
    """Validate one ordered interval and return its start, allowing overlaps between rows."""
    _number(row[left], f"{path}.{left}", 0, 9007199254740991, integer=milliseconds)
    _number(row[right], f"{path}.{right}", 0, 9007199254740991, integer=milliseconds)
    if row[right] < row[left] or positive and row[right] == row[left]:
        _fail(f"{path}.{right}", f"must be {'greater than' if positive else 'at least'} {left}")
    if row[left] < previous:
        _fail(f"{path}.{left}", "interval starts must be nondecreasing")
    return row[left]


def _box(value, path):
    """Validate normalized x/y/width/height and reject boxes crossing image bounds."""
    if not isinstance(value, list) or len(value) != 4:
        _fail(path, "expected normalized [x,y,width,height]")
    for index, number in enumerate(value):
        _number(number, f"{path}[{index}]", 0, 1)
    if value[0] + value[2] > 1 + 1e-9 or value[1] + value[3] > 1 + 1e-9:
        _fail(path, "box extends beyond normalized image bounds")


def _asr(value, path):
    """Validate ordered millisecond Whisper-compatible batch transcription results."""
    for index, result in _rows(value, path):
        result_path = f"{path}[{index}]"
        _object(result, result_path, ("transcription",))
        previous = -1
        for segment_index, segment in _rows(result["transcription"], result_path + ".transcription"):
            segment_path = f"{result_path}.transcription[{segment_index}]"
            _object(segment, segment_path, ("text", "offsets"))
            _text(segment["text"], segment_path + ".text")
            _object(segment["offsets"], segment_path + ".offsets", ("from", "to"))
            previous = _interval(segment["offsets"], segment_path + ".offsets", "from", "to", previous)


def _diarization(value, path):
    """Validate sorted, positive-duration turns with one-based speaker IDs."""
    previous = -1
    for index, row in _rows(value, path):
        row_path = f"{path}[{index}]"
        _object(row, row_path, ("startMs", "endMs", "speaker"))
        previous = _interval(row, row_path, "startMs", "endMs", previous, positive=True)
        _number(row["speaker"], row_path + ".speaker", 1, integer=True)


def _scene(value, path):
    """Validate canonical scene frames, probabilities and normalized object/OCR boxes."""
    _object(value, path, ("status", "frames"))
    _text(value["status"], path + ".status")
    previous = -1
    for index, frame in _rows(value["frames"], path + ".frames"):
        frame_path = f"{path}.frames[{index}]"
        _object(frame, frame_path, ("timeMs", "objects", "text"))
        _number(frame["timeMs"], frame_path + ".timeMs", 0, 9007199254740991, integer=True)
        if frame["timeMs"] < previous:
            _fail(frame_path + ".timeMs", "frame times must be nondecreasing")
        previous = frame["timeMs"]
        for field in ("objects", "text"):
            for row_index, row in _rows(frame[field], frame_path + "." + field):
                row_path = f"{frame_path}.{field}[{row_index}]"
                required = ("label", "score", "box") if field == "objects" else ("text", "box")
                _object(row, row_path, required)
                _text(row[required[0]], row_path + "." + required[0])
                if field == "objects":
                    _number(row["score"], row_path + ".score", 0, 1)
                _box(row["box"], row_path + ".box")


def _embedding(value, path):
    """Validate finite speaker embeddings and a consistent nonempty vector dimension."""
    _object(value, path)
    dimension = None
    for speaker, samples in value.items():
        for index, vector in _rows(samples, f"{path}.{speaker}"):
            vector_path = f"{path}.{speaker}[{index}]"
            _rows(vector, vector_path)
            if not vector or dimension is not None and len(vector) != dimension:
                _fail(vector_path, "embedding vectors must share a nonzero dimension")
            dimension = len(vector)
            for offset, number in enumerate(vector):
                _number(number, f"{vector_path}[{offset}]")


def _audio(value, path):
    """Validate seconds-based audio observations and their classification probabilities."""
    if not isinstance(value, list) or len(value) != 2:
        _fail(path, "expected [observations, status]")
    _text(value[1], path + "[1]")
    previous = -1
    for index, row in _rows(value[0], path + "[0]"):
        row_path = f"{path}[0][{index}]"
        _object(row, row_path, ("start", "end", "scores"))
        previous = _interval(row, row_path, "start", "end", previous, milliseconds=False)
        for score_index, score in _rows(row["scores"], row_path + ".scores"):
            score_path = f"{row_path}.scores[{score_index}]"
            _object(score, score_path, ("label", "score"))
            _text(score["label"], score_path + ".label")
            _number(score["score"], score_path + ".score", 0, 1)


def _translations(value, path):
    """Validate nonblank bounded translations before replacing any cue text."""
    _object(value, path, ("translations",))
    rows = value["translations"]
    if not isinstance(rows, list) or not 1 <= len(rows) <= 400:
        _fail(path + ".translations", "expected 1 to 400 translations")
    for index, text in enumerate(rows):
        if not isinstance(text, str) or not text.strip() or len(text) > 1000:
            _fail(f"{path}.translations[{index}]", "expected nonblank text of up to 1000 characters")


def validate_contract(value, reference, path="result", *, capability=None, direction="output"):
    """Validate finite JSON and semantic invariants, raising ModError at the failing JSON path.

    reference selects an exact published id/version. capability, when supplied,
    also enforces compatibility; direction separates model inputs from host results.
    The value is checked without coercion, mutation, code imports or downloads.
    """
    contract_id = resolve_contract(reference, capability=capability, direction=direction)
    _finite(value, path)
    validators = {"studio.asr": _asr, "studio.diarization": _diarization,
                  "studio.scene": _scene, "studio.embedding": _embedding,
                  "studio.audio-classification": _audio, "studio.translations": _translations}
    if contract_id in validators:
        validators[contract_id](value, path)
    elif contract_id == "studio.model-resource":
        _object(value, path, ("modelPath",))
        _text(value["modelPath"], path + ".modelPath")
        filename = Path(value["modelPath"])
        if not filename.is_absolute() or not filename.is_file():
            _fail(path + ".modelPath", "expected an absolute existing local file")
    elif contract_id == "studio.visual":
        _object(value, path, ("stages", "phrases"))
        _object(value["stages"], path + ".stages")
        _object(value["phrases"], path + ".phrases")
    else:
        _object(value, path)
