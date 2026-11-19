"""Connect immutable mod selections to existing native model entrypoints.

Builtin calls retain Python objects and their original execution path. User mods
receive JSON with local paths at a bounded batch boundary, never one process per
pixel, word, or frame. A run captures selections once, including worker children.
"""
from contextlib import contextmanager
from contextvars import ContextVar, copy_context
from functools import wraps
import inspect
import json
import os
from pathlib import Path
from threading import Lock

from voice_studio.builtin_mods.catalog import manifests

CURRENT = ContextVar('media_mod_execution', default=None)
REGISTRY = None
REGISTRY_LOCK = Lock()
OPERATIONS = {}
LAYER_CAPABILITIES = {
    'speech': ('speech.asr', 'speech.diarization', 'speech.embedding-resource'),
    'voice': ('speech.embedding', 'speech.embedding-resource'),
    'audio': ('audio.classification',),
    'visual': ('vision.visual-batch', 'vision.faces', 'vision.face-embedding', 'vision.active-speaker', 'vision.expressions', 'vision.objects', 'vision.appearance'),
    'people': ('vision.person-batch', 'vision.objects', 'vision.gaze', 'vision.segmentation'),
    'scene': ('vision.scene-batch', 'vision.objects', 'vision.ocr'),
    'object_review': ('vision.objects',),
}


def _builtin(mod_id, capability, payload, settings):
    """Invoke the exact selected checkpoint, preserving the previous native runner."""
    if '_call' in payload:
        return payload['_call']()
    operation = payload['_operation']
    if operation not in OPERATIONS:
        from voice_studio.mods import ModError
        raise ModError('Builtin providers are invoked through their documented Studio operation.')
    function = OPERATIONS[operation]
    args, kwargs = payload['_args'], dict(payload['_kwargs'])
    key = mod_id.removeprefix('builtin.')
    if capability == 'speech.asr':
        from voice_studio.speech.recognition import MODEL, SMALL_MODEL, RECOVERY_MODEL
        bound = inspect.signature(function).bind(*args, **kwargs)
        if payload.get('_explicit'):
            bound.arguments['model'] = {'whisper-base': MODEL, 'whisper-small': SMALL_MODEL,
                                        'whisper-recovery': RECOVERY_MODEL,
                                        'whisper-live': MODEL.with_name('ggml-base-q5_1.bin')}[key]
        if 'threads' in settings:
            from voice_studio.core.media_resources import stage_threads
            bound.arguments['threads'] = stage_threads(settings['threads'])
        return function(*bound.args, **bound.kwargs)
    if capability == 'audio.classification':
        if key == 'apple-sound':
            from voice_studio.audio.audio_scene import classify
            function = classify.__wrapped__
        else:
            from voice_studio.audio.zipformer_audio import classify_zipformer
            function = classify_zipformer.__wrapped__
    elif capability == 'audio.denoise':
        from voice_studio.audio.audio_processing import denoise_native
        return denoise_native(*args, **kwargs, provider=key if payload.get('_explicit') else None)
    elif capability == 'vision.ocr':
        kwargs['backend'] = key if payload.get('_explicit') else None
    elif capability == 'synthesis.tts':
        bound = inspect.signature(function).bind(*args, **kwargs)
        bound.arguments['model'] = key
        args, kwargs = bound.args, bound.kwargs
    return function(*args, **kwargs)


def _callback(mod_id, capability):
    """Bind one builtin mod identity without importing its optional native runtime."""
    def invoke(payload, settings):
        return _builtin(mod_id, capability, payload, settings)
    return invoke


def registry():
    """Use one process-local registry; each snapshot rereads durable configuration."""
    global REGISTRY
    with REGISTRY_LOCK:
        if REGISTRY is None:
            from voice_studio.mods import ModRegistry
            inventory = manifests()
            callbacks = {}
            for item in inventory:
                for declaration in item['capabilities']:
                    capability = declaration['id']
                    callbacks[(item['id'], capability)] = _callback(item['id'], capability)
            REGISTRY = ModRegistry(builtin_manifests=inventory, builtin_callbacks=callbacks)
        return REGISTRY


def current_snapshot():
    """Resolve a frozen parent-worker snapshot before considering current settings."""
    snapshot = CURRENT.get()
    if snapshot is not None:
        return snapshot
    wire = os.environ.get('VOICE_MOD_SNAPSHOT')
    return registry().snapshot_from_wire(json.loads(wire)) if wire else registry().snapshot()


@contextmanager
def execution_scope(snapshot=None):
    """Retain the same selection revision through nested calls and child threads."""
    previous = CURRENT.get()
    token = CURRENT.set(snapshot or previous or current_snapshot())
    try:
        yield CURRENT.get()
    finally:
        CURRENT.reset(token)


def worker_environment(environment=None):
    """Pass the exact run revision to a worker without mutating process environment."""
    result = dict(os.environ if environment is None else environment)
    snapshot = current_snapshot()
    wire = snapshot.to_wire()
    for capability in wire['selections']:
        snapshot.selected(capability)
    result['VOICE_MOD_SNAPSHOT'] = json.dumps(wire, separators=(',', ':'))
    return result


def submit_with_context(executor, operation, *args, **kwargs):
    """Copy the current immutable run selection into an existing thread pool task."""
    return executor.submit(copy_context().run, operation, *args, **kwargs)


def json_payload(value):
    """Encode paths and bounded ordinary containers; reject opaque model objects."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): json_payload(item) for key, item in value.items() if not callable(item)}
    if isinstance(value, (list, tuple)):
        return [json_payload(item) for item in value]
    if value is None or type(value) in (str, bool, int, float):
        return value
    raise TypeError('Custom model providers require file/batch inputs, not ' + type(value).__name__)


def builtin_provider(snapshot, capability, mod_id):
    """Use captured package origin; user IDs never determine executable privileges."""
    for record in snapshot.provenance([capability]):
        if record['modId'] == mod_id:
            return record.get('origin') == 'builtin'
    return False


def model_provider(capability, default):
    """Route one existing batch operation through its selected provider and builtin."""
    def decorate(operation):
        """Register the original callable before replacing its public entrypoint."""
        identity = operation.__module__ + ':' + operation.__qualname__
        OPERATIONS[identity] = operation
        signature = inspect.signature(operation)

        @wraps(operation)
        def dispatch(*args, **kwargs):
            """Keep builtin inputs untouched and expose named JSON inputs to replacements."""
            with execution_scope() as snapshot:
                mod_id = default(*args, **kwargs) if callable(default) else default
                selected = snapshot.selected(capability, default=mod_id)
                if builtin_provider(snapshot, capability, selected):
                    payload = {'_operation': identity, '_args': args, '_kwargs': kwargs,
                               '_explicit': capability in snapshot.to_wire()['selections']}
                else:
                    bound = signature.bind(*args, **kwargs)
                    bound.apply_defaults()
                    payload = json_payload(dict(bound.arguments))
                result = snapshot.invoke(capability, payload, default=mod_id)
                if not builtin_provider(snapshot, capability, selected):
                    from voice_studio.mod_contract_bridge import validate_provider_output
                    validate_provider_output(capability, payload, result)
                    if capability == 'vision.ocr':
                        return {int(stamp): rows for stamp, rows in result[0].items()}, result[1]
                    if capability == 'speech.embedding':
                        return {int(number): vectors for number, vectors in result.items()}
                return result
        return dispatch
    return decorate


def native_batch(capability, default, payload, operation):
    """Route one complete model batch while retaining native objects in process."""
    snapshot = current_snapshot()
    selected = snapshot.selected(capability, default=default)
    request = {'_call': operation} if builtin_provider(snapshot, capability, selected) else json_payload(payload)
    result = snapshot.invoke(capability, request, default=default)
    if not builtin_provider(snapshot, capability, selected):
        from voice_studio.mod_contract_bridge import validate_provider_output
        validate_provider_output(capability, request, result)
    return result


def is_replacement(capability, default):
    """Check the frozen provider choice before allocating an optional native model."""
    snapshot = current_snapshot()
    selected = snapshot.selected(capability, default=default)
    return not builtin_provider(snapshot, capability, selected)


def mod_scope(operation):
    """Capture settings for a non-media operation without running additional stages."""
    @wraps(operation)
    def run(*args, **kwargs):
        with execution_scope():
            return operation(*args, **kwargs)
    return run


def analysis_run(operation):
    """Capture one run, execute enabled additional stages, and append provenance."""
    signature = inspect.signature(operation)

    @wraps(operation)
    def run(*args, **kwargs):
        """Reuse nested snapshots and enrich only the complete outer report."""
        if CURRENT.get() is not None:
            return operation(*args, **kwargs)
        with execution_scope() as snapshot:
            from voice_studio.mod_pipeline import captured_pipeline
            pipeline = captured_pipeline(snapshot)
            result = operation(*args, **kwargs)
            if isinstance(result, dict):
                bound = signature.bind(*args, **kwargs)
                source = next((bound.arguments[key] for key in ('source', 'file_path', 'url', 'path') if key in bound.arguments), None)
                from voice_studio.mod_stage_results import additional_stages
                selected_stages = bound.arguments.get('selected_stages')
                workspace = bound.arguments.get('root') or bound.arguments.get('studio_root')
                outputs = additional_stages(snapshot, result, source, selected_stages, workspace, pipeline)
                provider = snapshot.to_wire()['selections'].get('speech.asr')
                if provider and 'segments' in result:
                    actual = {'builtin.whisper-base': 'whisper-base', 'builtin.whisper-small': 'whisper-small',
                              'builtin.whisper-recovery': 'whisper-large-v3-turbo',
                              'builtin.whisper-live': 'whisper-base-q5_1'}.get(provider, provider)
                    if result.get('recognitionModel') and result['recognitionModel'] != actual:
                        result['recognitionRequestedModel'] = result['recognitionModel']
                    result['recognitionProvider'] = provider
                    result['recognitionModel'] = actual
                    if result.get('recognitionWarning') == 'Whisper Small не установлен. Использован Whisper Base.':
                        result.pop('recognitionWarning')
                if outputs:
                    result['modStages'] = outputs
                result['modPipeline'] = {'apiVersion': 1, 'revision': pipeline['revision'],
                                         'fingerprint': pipeline['fingerprint'],
                                         'order': [item['nodeId'] for item in pipeline['stages']],
                                         'dependencies': {item['nodeId']: item['dependsOn'] for item in pipeline['stages']}}
                result['modProvenance'] = run_provenance(snapshot)
            return result
    return run


def run_provenance(snapshot):
    """Export captured settings with candidate provider configurations and exact hashes."""
    settings = snapshot.to_wire()['mods']
    return [{**record, 'settings': settings.get(record['modId'], {}).get('settings', {})}
            for record in snapshot.provenance()]


def layer_fingerprint(name):
    """Invalidate only the native layer whose provider selection/settings changed."""
    return current_snapshot().fingerprint(LAYER_CAPABILITIES.get(name, ()))

RESOURCE_CACHE = {}
RESOURCE_LOCK = Lock()


def model_resource(capability, default, path):
    """Resolve one local compatible weight artifact once per captured worker batch.

    Resource providers return {modelPath: absolute path}. The existing native
    inference runtime loads that artifact; no per-frame process is introduced.
    """
    snapshot = current_snapshot()
    selected = snapshot.selected(capability, default=default)
    if builtin_provider(snapshot, capability, selected):
        value = snapshot.invoke(capability, {'_call': lambda: str(path)}, default=default)
        return Path(value)
    key = (snapshot.fingerprint([capability]), str(path))
    with RESOURCE_LOCK:
        if key not in RESOURCE_CACHE:
            payload = {'modelPath': str(path), 'protocol': 'local-model-resource-v1'}
            value = snapshot.invoke(capability, payload, default=default)
            from voice_studio.mod_contract_bridge import validate_provider_output
            validate_provider_output(capability, payload, value)
            from voice_studio.mods import ModExecutionError
            if not isinstance(value, dict) or not isinstance(value.get('modelPath'), str):
                raise ModExecutionError('Model resource provider must return {modelPath: absolute local path}.')
            target = Path(value['modelPath'])
            if not target.is_absolute() or not target.is_file():
                raise ModExecutionError('Selected model resource does not exist as an absolute local file.')
            if len(RESOURCE_CACHE) >= 128:
                RESOURCE_CACHE.clear()
            RESOURCE_CACHE[key] = target
        target = RESOURCE_CACHE[key]
        if not target.is_file():
            from voice_studio.mods import ModExecutionError
            raise ModExecutionError('Previously selected model resource is no longer available.')
        return target
