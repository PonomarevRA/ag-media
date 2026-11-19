"""Apply the application's canonical contracts at existing provider boundaries."""
from voice_studio.mods import ModError, ModExecutionError


def validate_provider_output(capability, payload, result):
    """Validate known user-provider results even when their legacy manifest omits a contract.

    The worker has already converted a declared model-specific output format. This
    host check protects native consumers and cannot be disabled by a user schema.
    Builtin opaque Python callbacks do not use this JSON boundary.
    """
    from voice_studio.mods.contracts import contract_for_capability, validate_contract
    reference = contract_for_capability(capability)
    if reference is None:
        return result
    try:
        validate_contract(result, reference, 'provider output', capability=capability)
    except ModError as error:
        raise ModExecutionError(str(error)) from error
    if capability == 'speech.asr' and isinstance(payload, dict):
        jobs = payload.get('jobs')
        if isinstance(jobs, list) and len(result) != len(jobs):
            raise ModExecutionError('ASR provider must return exactly one canonical result per input job.')
    return result
