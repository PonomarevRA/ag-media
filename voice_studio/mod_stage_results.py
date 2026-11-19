"""Checkpoint additional-stage outputs independently of the seven native layers."""
from hashlib import sha256
from pathlib import Path
from time import perf_counter

from voice_paths import workspace_temp
from voice_studio.mods.schema import json_bytes

METADATA = {'modStages', 'modProvenance', 'analysisStages', 'checkpoints', 'sourceDigest',
            'recognitionProvider', 'recognitionRequestedModel', 'modPipeline'}


def additional_stages(snapshot, result, source, selected_stages, workspace=None, pipeline=None):
    """Run or restore enabled user stages; an explicit empty selection never executes code."""
    if pipeline is None:
        from voice_studio.mod_pipeline import captured_pipeline
        pipeline = captured_pipeline(snapshot)
    declarations = pipeline['stages']
    if not declarations:
        return []
    from voice_studio.graph.media_analysis_revolver import AnalysisRevolver
    from voice_studio.graph.media_stage_cache import source_digest
    source_path = Path(source) if source is not None else None
    cache = AnalysisRevolver()
    if source_path is not None and source_path.is_file() and workspace is not None:
        directory = workspace_temp(workspace) / 'mod-stage-results' / source_digest(source_path)
        cache = AnalysisRevolver(directory)
    evidence = {key: value for key, value in result.items() if key not in METADATA}
    report_hash = sha256(json_bytes([evidence, str(source_path) if source_path is not None else None])).hexdigest()
    outputs = []
    completed = {}
    for declaration in declarations:
        capability, mod_id = declaration['capability'], declaration['modId']
        node_id = declaration['nodeId']
        parents = declaration['dependsOn']
        upstream = {parent: completed[parent] for parent in parents if parent in completed}
        missing_parent = len(upstream) != len(parents)
        key = sha256(json_bytes([snapshot.fingerprint([capability]), report_hash, parents, upstream])).hexdigest()
        checkpoint = cache.child(mod_id).child(key)
        saved = None if missing_parent else checkpoint.load('complete', required=('result',))
        started = perf_counter()
        if saved is not None:
            value, state = saved['result'], 'reused'
        elif missing_parent or selected_stages == [] or selected_stages == ():
            outputs.append({'modId': mod_id, 'capability': capability, 'nodeId': node_id, 'dependsOn': parents, 'state': 'pending'})
            continue
        else:
            transcript = result.get('text', ' '.join(row.get('text', '') for row in result.get('segments', [])))
            payload = {'report': evidence, 'transcript': transcript,
                       'sourcePath': str(source_path) if source_path is not None else None, 'upstream': upstream}
            value = snapshot.invoke_mod(mod_id, capability, payload)
            checkpoint.save('complete', {'result': value})
            state = 'done'
        completed[node_id] = value
        outputs.append({'modId': mod_id, 'capability': capability, 'nodeId': node_id, 'dependsOn': parents, 'state': state,
                        'elapsedMs': round((perf_counter() - started) * 1000), 'result': value})
    return outputs
