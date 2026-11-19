"""Shared local HTTP/MCP operations for mod settings and explicit test runs."""
from voice_studio.mod_runtime import registry, run_provenance
from voice_studio.mods import ModError
from voice_studio.core.mcp_lanes import AsyncWorkLane


def mod_operation(operation, data=None):
    """Apply one explicit settings action; discovery and templates never execute code."""
    data = {} if data is None else data
    if not isinstance(data, dict):
        raise ModError('Mod request must be an object.')
    service = registry()
    if operation == 'pipeline':
        from voice_studio.mod_pipeline import pipeline_catalog, save_pipeline
        if not data:
            return pipeline_catalog(service)
        return save_pipeline(service, data.get('dependencies'), data.get('expectedRevision'))
    if operation == 'contracts':
        from voice_studio.mods.contracts import get_contracts
        return get_contracts()
    if operation == 'list':
        return service.catalog()
    if operation == 'rescan':
        return service.catalog()
    elif operation == 'enable':
        service.enable(data.get('id'), data.get('fingerprint'), data.get('enabled', True))
    elif operation == 'select':
        service.select(data.get('capability'), data.get('id'))
    elif operation == 'configure':
        service.configure(data.get('id'), data.get('settings'))
    elif operation == 'template':
        return service.create_template(data.get('id'), template=data.get('template', 'word-count'))
    elif operation == 'export-test':
        snapshot = service.snapshot()
        value = snapshot.invoke_mod(data.get('id'), data.get('capability'), data.get('payload', {}))
        from voice_studio.mod_contract_bridge import validate_provider_output
        validate_provider_output(data.get('capability'), data.get('payload', {}), value)
        return {'result': value, 'modProvenance': run_provenance(snapshot)}
    else:
        raise ModError('Unknown mod operation: ' + operation)
    return service.catalog()


def register_mod_tools(mcp):
    """Register shared settings operations and return their IO lane for session cleanup."""
    lane = AsyncWorkLane('mod-settings', capacity=2)
    @mcp.tool()
    @lane.offload
    def media_mods_pipeline(dependencies: dict | None = None, expectedRevision: int | None = None) -> dict:
        """Read the processing graph or save custom-stage dependencies with revision CAS; native stage order remains locked."""
        if dependencies is None:
            return mod_operation('pipeline')
        return mod_operation('pipeline', {'dependencies': dependencies, 'expectedRevision': expectedRevision})

    @mcp.tool()
    @lane.offload
    def media_mods_contracts() -> dict:
        """Discover canonical input/output schemas, versions, units and conversion support for provider authors."""
        return mod_operation('contracts')

    @mcp.tool()
    @lane.offload
    def media_mods_list() -> dict:
        """List metadata, exact package hashes, availability, settings and selections; run no mod code."""
        return mod_operation('list')

    @mcp.tool()
    @lane.offload
    def media_mods_rescan() -> dict:
        """Discover local mod manifests without executing or enabling newly found code."""
        return mod_operation('rescan')

    @mcp.tool()
    @lane.offload
    def media_mods_enable(id: str, fingerprint: str, enabled: bool = True) -> dict:
        """Enable an exact reviewed package hash or disable a mod; install no model or dependency."""
        return mod_operation('enable', {'id': id, 'fingerprint': fingerprint, 'enabled': enabled})

    @mcp.tool()
    @lane.offload
    def media_mods_select(capability: str, id: str | None = None) -> dict:
        """Select the provider for a capability; null restores the existing stage default."""
        return mod_operation('select', {'capability': capability, 'id': id})

    @mcp.tool()
    @lane.offload
    def media_mods_configure(id: str, settings: dict) -> dict:
        """Validate and save a mod's JSON settings; a running analysis retains its captured revision."""
        return mod_operation('configure', {'id': id, 'settings': settings})

    @mcp.tool()
    @lane.offload
    def media_mods_template(id: str, template: str = 'word-count') -> dict:
        """Create a disabled editable local mod with an executable additional-stage example and tests."""
        return mod_operation('template', {'id': id, 'template': template})

    @mcp.tool()
    @lane.offload
    def media_mods_test(id: str, capability: str, payload: dict) -> dict:
        """Explicitly run one enabled mod with supplied fixture input in its bounded local worker."""
        return mod_operation('export-test', {'id': id, 'capability': capability, 'payload': payload})

    return lane
