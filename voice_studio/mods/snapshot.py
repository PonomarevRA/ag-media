"""Request-scoped provider selection and reproducible media mod execution."""

from copy import deepcopy
from hashlib import sha256

from .errors import ModError, ModExecutionError
from .execution import execute
from .schema import json_bytes, validate_value
from .contracts import validate_contract


class ExecutionSnapshot:
    """An isolated captured configuration; later settings edits affect only future snapshots."""

    def __init__(self, records, selections, callbacks, extensions=None):
        """Capture catalog records, selections and host callbacks by value except callable identities."""
        self._records = deepcopy(records)
        self._selections = deepcopy(selections)
        self._callbacks = dict(callbacks)
        self._extensions = deepcopy(extensions or {})

    def extension(self, name, default=None):
        """Return detached host metadata captured atomically with selection/settings for this run."""
        return deepcopy(self._extensions.get(name, default))

    def _record(self, mod_id, capability):
        """Resolve an enabled exact package and capability, reporting failure without fallback."""
        record = self._records.get(mod_id)
        if record is None or not record.get("enabled"):
            reason = record.get("error", record.get("status")) if record else "not discovered"
            raise ModError(f"Selected mod {mod_id} is unavailable: {reason}")
        for declaration in record["package"].manifest["capabilities"]:
            if declaration["id"] == capability:
                return record, declaration
        raise ModError(f"Mod {mod_id} does not provide {capability}")

    def selected(self, capability, default=None):
        """Return the explicit selection or a host default; an invalid selection never falls back."""
        mod_id = self._selections.get(capability)
        if mod_id is None:
            mod_id = default
        if mod_id is None:
            candidates = []
            for identifier, record in self._records.items():
                if record["package"].origin != "builtin":
                    continue
                for item in record["package"].manifest["capabilities"]:
                    if item["id"] == capability and item.get("default"):
                        candidates.append(identifier)
            if len(candidates) != 1:
                raise ModError(f"No unique default provider for {capability}")
            mod_id = candidates[0]
        _, declaration = self._record(mod_id, capability)
        if declaration["kind"] != "provider":
            raise ModError("A selected replacement must declare a provider capability")
        return mod_id

    def invoke(self, capability, payload, default=None, **budgets):
        """Invoke the selected provider, using a builtin mod ID default only when none is selected."""
        return self.invoke_mod(self.selected(capability, default), capability, payload, **budgets)

    def invoke_mod(self, mod_id, capability, payload, **budgets):
        """Execute an enabled provider/stage once; builtins run directly, user code runs one batch."""
        record, declaration = self._record(mod_id, capability)
        package = record["package"]
        if "inputSchema" in declaration:
            validate_value(payload, declaration["inputSchema"], "provider input")
        if package.origin == "builtin":
            callback = self._callbacks.get((mod_id, capability))
            if callback is None:
                raise ModError(f"Builtin callback is not registered for {mod_id}/{capability}")
            # Opaque model objects stay in-process; no JSON round-trip or process per frame.
            result = callback(payload, deepcopy(record["settings"]))
        else:
            if "inputContract" in declaration:
                validate_contract(payload, declaration["inputContract"], "provider input", capability=capability, direction="input")
            result = execute(package, declaration, payload, deepcopy(record["settings"]), **budgets)
            if "contract" in declaration:
                try:
                    validate_contract(result, declaration["contract"], "provider output", capability=capability)
                except ModError as error:
                    raise ModExecutionError(str(error)) from error
        if "outputSchema" in declaration:
            try:
                validate_value(result, declaration["outputSchema"], "provider output")
            except ModError as error:
                raise ModExecutionError(str(error)) from error
        return result

    def custom_stages(self):
        """List enabled user stages deterministically, without executing or importing any code."""
        stages = []
        for mod_id, record in sorted(self._records.items()):
            if not record["enabled"] or record["package"].origin != "user":
                continue
            for item in record["package"].manifest["capabilities"]:
                if item["kind"] == "stage":
                    stages.append({**deepcopy(item), "modId": mod_id, "capability": item["id"]})
        return stages

    def _relevant(self, capabilities):
        """Find selected/default providers, enabled additive stages and their transitive dependencies."""
        if isinstance(capabilities, str):
            capabilities = [capabilities]
        wanted = set(capabilities) if capabilities is not None else None
        relevant = set()
        for mod_id, record in self._records.items():
            for declaration in record["package"].manifest["capabilities"]:
                capability = declaration["id"]
                if wanted is not None and capability not in wanted:
                    continue
                selected = self._selections.get(capability)
                # Some legacy entry points choose their default by a model argument.
                # Include that capability's enabled builtin alternatives conservatively.
                is_default = (selected is None and record["enabled"]
                              and record["package"].origin == "builtin")
                if selected == mod_id or is_default or declaration["kind"] == "stage" and record["enabled"]:
                    relevant.add(mod_id)
        # Keep explicit missing selections in the fingerprint instead of dropping evidence.
        for capability, mod_id in self._selections.items():
            if wanted is None or capability in wanted:
                relevant.add(mod_id)
        pending = list(relevant)
        while pending:
            record = self._records.get(pending.pop())
            if record is None:
                continue
            for dependency in record["package"].manifest["requires"]:
                if dependency not in relevant:
                    relevant.add(dependency)
                    pending.append(dependency)
        return relevant

    def provenance(self, capabilities=None):
        """Return JSON provenance for effective relevant packages, including disabled-selection evidence."""
        result = []
        for mod_id in sorted(self._relevant(capabilities)):
            record = self._records.get(mod_id)
            if record is None:
                result.append({"modId": mod_id, "enabled": False, "status": "missing"})
                continue
            package = record["package"]
            result.append({"modId": mod_id, "version": package.manifest["version"],
                           "fingerprint": package.fingerprint, "origin": package.origin,
                           "settingsHash": sha256(json_bytes(record["settings"])).hexdigest(),
                           "enabled": record["enabled"], "status": record["status"]})
        return result

    def fingerprint(self, capabilities):
        """Hash relevant package/version/code/settings and selections for only the requested capabilities."""
        if isinstance(capabilities, str):
            capabilities = [capabilities]
        capabilities = sorted(set(capabilities))
        selections = {key: value for key, value in self._selections.items() if key in capabilities}
        value = {"capabilities": capabilities, "selections": selections,
                 "mods": self.provenance(capabilities)}
        return sha256(json_bytes(value)).hexdigest()

    def to_wire(self):
        """Export a JSON config for owned model workers; exact content hashes are verified on restore."""
        mods = {}
        for mod_id, record in self._records.items():
            mods[mod_id] = {"fingerprint": record["package"].fingerprint,
                            "enabled": record["enabled"], "settings": deepcopy(record["settings"])}
        return {"apiVersion": 1, "mods": mods, "selections": deepcopy(self._selections),
                "extensions": deepcopy(self._extensions)}
