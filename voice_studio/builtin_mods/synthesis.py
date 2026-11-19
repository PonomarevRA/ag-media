"""Keep queued multi-phrase synthesis on one mod revision and publish provenance."""
from functools import wraps
import json

from voice_studio.mod_runtime import execution_scope, run_provenance


def capture_synthesis(operation):
    """Save actual provider identity beside a completed WAV without changing job schema."""
    @wraps(operation)
    def run(studio, key, model, *args, **kwargs):
        """Capture once for all phrases and record the provider of the published WAV."""
        with execution_scope() as snapshot:
            result = operation(studio, key, model, *args, **kwargs)
            if (studio.media / (key + '.wav')).is_file():
                actual = snapshot.selected('synthesis.tts', default='builtin.' + model)
                relevant = snapshot.provenance(('synthesis.tts', 'synthesis.accent-resource',
                                                'synthesis.vocoder-resource', 'synthesis.tokenizer-resource'))
                identifiers = {item['modId'] for item in relevant}
                records = [item for item in run_provenance(snapshot) if item['modId'] in identifiers]
                provenance = {'requestedModel': model, 'actualProvider': actual,
                              'modProvenance': records}
                target = studio.media / (key + '.mods.json')
                temporary = target.with_suffix('.tmp')
                temporary.write_text(json.dumps(provenance, ensure_ascii=False), encoding='utf-8')
                temporary.replace(target)
            return result
    return run


def synthesis_provenance(media, key):
    """Read completed job provenance, preserving compatibility with historical jobs."""
    try:
        value = json.loads((media / (key + '.mods.json')).read_bytes())
    except (OSError, ValueError):
        return {}
    return value if isinstance(value, dict) else {}
