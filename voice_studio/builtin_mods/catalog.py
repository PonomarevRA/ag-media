"""Metadata-only inventory of the existing installed inference adapters.

This inventory checks paths only: it never imports native inference libraries,
loads weights, compiles helpers, contacts a server, or installs dependencies.
"""
from pathlib import Path
from hashlib import sha256
import os
import shutil
import sys

from voice_paths import PROJECT_ROOT, model_path, dependency_path
from voice_studio.speech.transcription_languages import WHISPER_LANGUAGES

# Keep each checkpoint a separate mod even when several share a runtime.
MODELS = (
    ('whisper-live', 'Whisper Base Q5_1 live', 'speech.asr', 'whisper/ggml-base-q5_1.bin', 'MIT', 'speech/recognition.py:run_whisper'),
    ('whisper-base', 'Whisper Base', 'speech.asr', 'whisper/ggml-base.bin', 'MIT', 'speech/recognition.py:run_whisper'),
    ('whisper-small', 'Whisper Small Q5_1', 'speech.asr', 'whisper/ggml-small-q5_1.bin', 'MIT', 'speech/recognition.py:run_whisper'),
    ('whisper-recovery', 'Whisper Large V3 Turbo Q5_0', 'speech.asr', 'whisper/ggml-large-v3-turbo-q5_0.bin', 'MIT', 'speech/recognition.py:run_whisper'),
    ('pyannote-segmentation', 'Pyannote Segmentation ONNX', 'speech.diarization', 'diarization/segmentation.onnx', 'MIT upstream; exact ONNX redistribution unconfirmed', 'speech/speaker_diarization.py:detect_turns'),
    ('sherpa-embedding', '3D Speaker ERes2Net ONNX', 'speech.embedding', 'diarization/embedding.onnx', 'Apache-2.0 upstream; exact ONNX redistribution unconfirmed', 'speech/speaker_review.py:extract_embeddings'),
    ('apple-sound', 'Apple SoundAnalysis', 'audio.classification', None, 'Apple system framework', 'audio/audio_scene.py:classify'),
    ('zipformer', 'Zipformer Small INT8 2024-04-15', 'audio.classification', 'audio-tagging/zipformer-small-int8/model.int8.onnx', 'Apache-2.0', 'audio/zipformer_audio.py:classify_zipformer'),
    ('rnnoise', 'RNNoise cb.rnnn', 'audio.denoise', 'denoise/cb.rnnn', 'BSD-3-Clause code; cb.rnnn weights unconfirmed', 'audio/audio_processing.py:denoise_wave'),
    ('fft-denoise', 'FFmpeg FFT denoiser', 'audio.denoise', None, 'Installed FFmpeg license; see THIRD-PARTY.md', 'audio/audio_processing.py:denoise_wave'),
    ('mrx', 'MERL MRX SNR', 'audio.separation', 'separation/mrx/default_mrx_pre_trained_weights.pth', 'See THIRD-PARTY.md exact checkpoint terms', 'audio/agent_audio.py:run_separator'),
    ('hdemucs', 'HDemucs high MUSDB', 'audio.background', 'separation/hdemucs_high_trained.pt', 'See THIRD-PARTY.md exact checkpoint terms', 'synthesis/dub_background.py:prepare_background'),
    ('yunet', 'YuNet 2023mar', 'vision.faces', 'vision/yunet.onnx', 'MIT', 'workers/light_asd_worker.py:load_frames'),
    ('sface', 'SFace INT8 2021dec', 'vision.face-embedding', 'vision/sface-int8.onnx', 'Apache-2.0', 'vision/face_reid.py:Signatures'),
    ('nanodet', 'NanoDet Plus 2022nov', 'vision.objects', 'vision/nanodet.onnx', 'Apache-2.0', 'vision/video_scene.py:analyze_frames'),
    ('light-asd', 'Light ASD AVA', 'vision.active-speaker', 'vision/light-asd/pretrain_AVA_CVPR.model', 'MIT code; exact weights redistribution unconfirmed', 'workers/light_asd_worker.py:score_scan'),
    ('mobilefacenet-expression', 'MobileFaceNet FER 2022july', 'vision.expressions', 'vision/facial_expression_recognition_mobilefacenet_2022july.onnx', 'Apache-2.0', 'vision/facial_expression_model.py:ExpressionClassifier'),
    ('apple-ocr', 'Apple Vision OCR', 'vision.ocr', None, 'Apple system framework', 'vision/scene_ocr.py:recognize'),
    ('tesseract', 'Tesseract Russian + English', 'vision.ocr', 'vision/tessdata/rus.traineddata', 'Apache-2.0', 'vision/scene_ocr.py:recognize'),
    ('mediapipe-face', 'MediaPipe Face Landmarker', 'vision.gaze', 'mediapipe/face_landmarker.task', 'Apache-2.0', 'vision/person_layers.py:dispatch_layers'),
    ('mediapipe-selfie', 'MediaPipe Selfie Multiclass', 'vision.segmentation', 'mediapipe/selfie_multiclass_256x256.tflite', 'Apache-2.0', 'vision/person_layers.py:dispatch_layers'),
    ('qwen', 'Qwen3 TTS 0.6B MLX', 'synthesis.tts', 'commercial/qwen/config.json', 'See THIRD-PARTY.md pinned MLX checkpoint', 'synthesis/engines.py:generate'),
    ('chatterbox', 'Chatterbox Multilingual V3 MLX', 'synthesis.tts', 'commercial/chatterbox/config.json', 'See THIRD-PARTY.md pinned MLX checkpoint', 'synthesis/engines.py:generate'),
    ('f5', 'F5 TTS', 'synthesis.tts', 'f5/model.safetensors', 'Noncommercial checkpoint; see THIRD-PARTY.md', 'synthesis/engines.py:generate'),
    ('fish', 'Fish Speech local server', 'synthesis.tts', None, 'Server/checkpoint license operator supplied; see THIRD-PARTY.md', 'synthesis/engines.py:generate'),
    ('f5-accent', 'F5 Russian accent tune', 'synthesis.accent-resource', 'f5/accent-tune.safetensors', 'CC BY-NC 4.0 checkpoint; see THIRD-PARTY.md', 'synthesis/engines.py:generate'),
    ('vocos', 'Vocos vocoder', 'synthesis.vocoder-resource', 'vocos/pytorch_model.bin', 'See THIRD-PARTY.md exact weights', 'synthesis/engines.py:generate'),
    ('s3tokenizer', 'S3Tokenizer V2', 'synthesis.tokenizer-resource', 'commercial/hub/models--mlx-community--S3TokenizerV2/refs/main', 'See THIRD-PARTY.md pinned checkpoint', 'synthesis/commercial_voice.py:generate'),
    ('xtts-speaker-encoder', 'XTTS ResNet speaker encoder', 'speech.metric-resource', 'xtts/model.pth', 'Coqui Public Model License 1.0.0', 'synthesis/speaker_metric.py:cosine'),
    ('age-gender', 'OpenVINO age gender', 'vision.appearance', 'vision/openvino-age-gender/model.xml', 'See THIRD-PARTY.md exact checkpoint', 'vision/face_appearance.py:appearance_model'),
    ('stress-context', 'Stress context model', 'prosody.context', 'prosody/context.json', 'Project trained model', 'prosody/stress_context.py:predict'),
    ('stress-lexical', 'Stress lexical model', 'prosody.lexical', 'prosody/selected.json.packed', 'Project trained model', 'prosody/stress_char_runtime.py:predict'),
    ('visual-batch', 'Existing visual batch pipeline', 'vision.visual-batch', None, 'See constituent model licenses', 'vision/video_revolver.py:review_video'),
    ('frame-changes', 'Frame change regions', 'vision.frame-changes', None, 'MIT; uses installed FFmpeg and OpenCV', 'vision/frame_changes.py:run_scan'),
    ('scene-batch', 'Existing scene batch pipeline', 'vision.scene-batch', None, 'See constituent model licenses', 'vision/video_scene.py:review_scene'),
    ('person-batch', 'Existing person batch pipeline', 'vision.person-batch', None, 'See constituent model licenses', 'vision/person_layers.py:dispatch_layers'),
    ('coqui', 'Coqui XTTS v2', 'synthesis.tts', 'xtts/model.pth', 'Coqui Public Model License 1.0.0', 'synthesis/engines.py:generate'),
)


def resource_protocol(capability):
    """Separate compatible local artifacts from architecture-independent batch outputs."""
    batch_vision = {'vision.ocr', 'vision.visual-batch', 'vision.scene-batch', 'vision.person-batch', 'vision.frame-changes'}
    return capability.endswith('-resource') or capability.startswith('prosody.') or (
        capability.startswith('vision.') and capability not in batch_vision)


def installed_path(key, relative):
    """Mirror existing operator overrides and legacy fallback paths without opening weights."""
    overrides = {'whisper-base': 'VOICE_WHISPER_MODEL', 'whisper-small': 'VOICE_WHISPER_SMALL_MODEL',
                 'whisper-recovery': 'VOICE_WHISPER_RECOVERY_MODEL', 'f5': 'VOICE_F5_CHECKPOINT',
                 'coqui': 'VOICE_COQUI_CHECKPOINT'}
    if key in overrides and os.environ.get(overrides[key]):
        return Path(os.environ[overrides[key]]).expanduser()
    path = model_path(*relative.split('/')) if relative else None
    if key == 'light-asd' and not path.is_file():
        path = dependency_path('light-asd') / 'upstream/weight/pretrain_AVA_CVPR.model'
    return path


def manifests():
    """Return builtin manifests without opening weights or probing inference runtimes."""
    from voice_studio.mods.contracts import contract_for_capability
    result = []
    for key, name, capability, relative, license_name, entry in MODELS:
        path = installed_path(key, relative)
        ready = path.is_file() if path else bool(shutil.which('ffmpeg'))
        if key.startswith('apple-'):
            ready = sys.platform == 'darwin'
        if key == 'fish':
            ready = False  # Discovery must never contact the local server.
        adapter_file = PROJECT_ROOT / 'voice_studio' / entry.split(':', 1)[0]
        adapter_hash = sha256(adapter_file.read_bytes()).hexdigest() if adapter_file.is_file() else None
        result.append({'apiVersion': 1, 'id': 'builtin.' + key, 'name': name, 'version': '1.0.0',
                       'description': ('Compatible local model resource; see local-model-resource-v1. ' if resource_protocol(capability) else 'Complete batch result provider. ') + entry,
                       'capabilities': [{'id': capability, 'kind': 'provider', 'default': True}],
                       'settingsSchema': {'type': 'object', 'properties': {}, 'additionalProperties': False},
                       'settings': {}, 'license': license_name,
                       'models': [{'path': str(path) if path else None, 'entrypoint': entry}],
                       'availability': {'ready': ready, 'reason': 'Local artifact/platform present; runtime not probed' if ready else 'Artifact unavailable or service unprobed',
                                        'files': [str(path)] if path else []},
                       'provenance': {'source': 'THIRD-PARTY.md', 'adapter': entry,
                                      'weightsMoved': False, 'downloads': False, 'adapterCodeSha256': adapter_hash,
                                      'protocol': 'local-model-resource-v1' if resource_protocol(capability) else 'named-batch-result-v1'}})
    for item in result:
        key = item['id']
        if key == 'builtin.frame-changes':
            item['defaultEnabled'] = False
            item['description'] = 'Local pixel changes. Assist mode adds split-screen OCR and targeted object checks.'
            item['settings'] = {'mode': 'measure', 'pixelThreshold': 24,
                                'adaptiveThreshold': False, 'compensateCamera': False}
            item['settingsSchema']['properties'].update({
                'mode': {'type': 'string', 'enum': ['measure', 'objects', 'assist'], 'title': 'Mode'},
                'pixelThreshold': {'type': 'integer', 'minimum': 8, 'maximum': 96,
                                   'title': 'Pixel threshold (8-bit channel)'},
                'adaptiveThreshold': {'type': 'boolean', 'title': 'Адаптивный порог изменений'},
                'compensateCamera': {'type': 'boolean', 'title': 'Компенсация движения камеры'},
            })
        if key.startswith('builtin.whisper-'):
            maximum = min(32, max(1, os.cpu_count() or 1))
            languages = ('ru', 'en') if key == 'builtin.whisper-live' else WHISPER_LANGUAGES
            item['settings']['language'] = 'auto'
            item['settingsSchema']['required'] = ['language']
            item['settingsSchema']['properties']['language'] = {
                'type': 'string', 'enum': ['auto', *languages],
                'title': 'Язык распознавания',
                'description': 'Применяется, когда язык не задан для отдельной транскрипции.'}
            item['settingsSchema']['properties']['threads'] = {
                'type': 'integer', 'minimum': 1, 'maximum': maximum,
                'title': 'CPU threads',
                'description': 'Optional Whisper CPU thread limit; omit to retain the existing stage default.'}
        if key == 'builtin.sherpa-embedding':
            item['capabilities'].append({'id': 'speech.embedding-resource', 'kind': 'provider', 'default': True})
        if key == 'builtin.chatterbox':
            item['requires'] = ['builtin.s3tokenizer']
        if key == 'builtin.f5':
            item['provenance']['optionalSubmodels'] = ['builtin.f5-accent', 'builtin.vocos']
        if key == 'builtin.pyannote-segmentation':
            item['provenance']['submodels'] = ['builtin.sherpa-embedding']
    for item in result:
        for declaration in item['capabilities']:
            reference = contract_for_capability(declaration['id'])
            if reference is not None:
                declaration['contract'] = reference
            input_reference = contract_for_capability(declaration['id'], direction='input')
            if input_reference is not None:
                declaration['inputContract'] = input_reference
    return result
