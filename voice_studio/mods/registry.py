"""External media mod catalog, explicit activation and durable provider selection."""

from copy import deepcopy
from hashlib import sha256
import inspect
from itertools import islice
import json
import os
from pathlib import Path
import shutil
import tempfile

from .discovery import MAX_PACKAGES, Package, builtin_package, read_package, unmet_requirements
from .errors import ModError
from .schema import identifier, json_bytes, validate_value
from .snapshot import ExecutionSnapshot
from .state import StateStore

NOTICE = ("Executable mods are trusted local Python running with your user permissions, including "
          "filesystem and network access. Worker limits are not an OS sandbox. No dependencies or "
          "model weights are installed automatically.")


class ModRegistry:
    """Discover metadata safely and bind every opt-in executable package to its displayed hash."""

    def __init__(self, data_dir=None, builtin_manifests=(), builtin_callbacks=None):
        """Bind external state and host-owned providers; construction does not import user code."""
        if data_dir is None:
            from voice_paths import application_data
            data_dir = os.environ.get("VOICE_MEDIA_MODS_DIR") or application_data() / "MediaMods"
        self.data_dir = Path(data_dir).expanduser().resolve()
        self.directory = self.data_dir / "packages"
        self.store = StateStore(self.data_dir / "state.json")
        self._callbacks = dict(builtin_callbacks or {})
        self._builtins = {}
        for value in builtin_manifests:
            package = builtin_package(value)
            mod_id = package.manifest["id"]
            if mod_id in self._builtins:
                raise ModError(f"Duplicate builtin mod id {mod_id}")
            digest = sha256(package.fingerprint.encode())
            for declaration in package.manifest["capabilities"]:
                callback = self._callbacks.get((mod_id, declaration["id"]))
                if callback is not None:
                    try:
                        digest.update(inspect.getsource(callback).encode("utf-8"))
                        filename = inspect.getsourcefile(callback)
                        if filename is not None:
                            digest.update(Path(filename).read_bytes())
                    except (OSError, TypeError):
                        digest.update(f"{callback.__module__}.{callback.__qualname__}".encode())
            self._builtins[mod_id] = Package(package.manifest, digest.hexdigest(), "builtin", None, {})

    def _discover(self):
        """Return valid packages plus independent diagnostic rows, keeping duplicate IDs unavailable."""
        packages = dict(self._builtins)
        errors = []
        truncated = False
        if self.directory.is_symlink():
            errors.append({"id": "user-directory", "enabled": False, "status": "error", "origin": "user",
                           "error": "User package directory cannot be a symlink"})
            return packages, errors, False
        if self.directory.is_dir():
            roots = sorted(islice(self.directory.iterdir(), MAX_PACKAGES + 1))
            truncated = len(roots) > MAX_PACKAGES
            for root in roots[:MAX_PACKAGES]:
                try:
                    if not root.is_dir() and not root.is_symlink():
                        continue
                    package = read_package(root)
                    mod_id = package.manifest["id"]
                    if mod_id in packages:
                        raise ModError("Duplicate mod id; user packages cannot shadow bundled providers")
                    packages[mod_id] = package
                except (ModError, OSError) as error:
                    errors.append({"id": root.name, "enabled": False, "status": "error", "origin": "user", "error": str(error)[:2000]})
        return packages, errors, truncated

    def _records(self, state):
        """Compute activation, settings, prerequisites and dependencies without code execution."""
        packages, errors, truncated = self._discover()
        records = {}
        for mod_id, package in packages.items():
            pinned = state["enabled"].get(mod_id)
            if package.origin == "builtin":
                # The installed application vouches for builtin code across updates.
                # A saved hash records opt-in; worker snapshots still require exact code.
                enabled = pinned is not False and (isinstance(pinned, str) or package.manifest.get("defaultEnabled", True))
            else:
                enabled = pinned == package.fingerprint
            status = "enabled" if enabled else "changed" if isinstance(pinned, str) else "disabled"
            settings = deepcopy(package.manifest["settings"])
            settings.update(deepcopy(state["settings"].get(mod_id, {})))
            record = {"package": package, "enabled": enabled, "status": status, "settings": settings,
                      "requirementsMissing": unmet_requirements(package)}
            try:
                validate_value(settings, package.manifest["settingsSchema"], "settings")
            except ModError as error:
                record.update(enabled=False, status="invalid-settings", error=str(error))
            if record["requirementsMissing"]:
                record.update(enabled=False, status="missing-requirements",
                              error="; ".join(record["requirementsMissing"]))
            records[mod_id] = record
        cycles = self._cycles(packages)
        for mod_id in cycles:
            records[mod_id].update(enabled=False, status="dependency-cycle", error="Cyclic mod dependencies")
        # Iteration propagates a disabled prerequisite through the full dependency chain.
        while True:
            changed = False
            for record in records.values():
                if not record["enabled"]:
                    continue
                required = record["package"].manifest["requires"]
                missing = [item for item in required if item not in records or not records[item]["enabled"]]
                if missing:
                    record.update(enabled=False, status="missing-dependency", error="Enable prerequisites first: " + ", ".join(missing))
                    changed = True
            if not changed:
                break
        return records, errors, truncated

    @staticmethod
    def _cycles(packages):
        """Identify cycle members independently of activation state so enabling cannot create one."""
        visiting, visited, cycles = [], set(), set()

        def visit(mod_id):
            """Visit one bounded dependency graph node and record its cycle path."""
            if mod_id in visiting:
                cycles.update(visiting[visiting.index(mod_id):])
                return
            if mod_id in visited or mod_id not in packages:
                return
            visiting.append(mod_id)
            for dependency in packages[mod_id].manifest["requires"]:
                visit(dependency)
            visiting.pop()
            visited.add(mod_id)

        for mod_id in packages:
            visit(mod_id)
        return cycles

    def catalog(self):
        """Return safe metadata and effective settings, never importing or executing package code."""
        state = self.store.read()
        records, errors, truncated = self._records(state)
        rows = []
        for mod_id, record in sorted(records.items()):
            package = record["package"]
            row = {key: deepcopy(value) for key, value in record.items() if key != "package"}
            row.update(id=mod_id, manifest=deepcopy(package.manifest), origin=package.origin,
                       fingerprint=package.fingerprint)
            if package.origin == "builtin":
                availability = row["manifest"].get("availability", {})
                files = availability.get("files", [])
                if files:
                    ready = all(Path(path).is_file() for path in files)
                    availability.update(ready=ready, reason="Local artifact/platform present; runtime not probed" if ready
                                        else "Artifact unavailable or service unprobed")
            rows.append(row)
        return {"apiVersion": 1, "mods": rows + errors, "selections": deepcopy(state["selections"]),
                "revision": state["revision"], "directory": str(self.directory), "notice": NOTICE,
                "truncated": truncated}

    def enable(self, mod_id, fingerprint, enabled=True):
        """Explicitly activate exact displayed content, or disable without altering selection.

        Args:
            mod_id: Discovered package ID.
            fingerprint: Displayed SHA-256; required when enabling, ignored when disabling.
            enabled: Strict boolean activation choice.

        Returns:
            Refreshed metadata catalog after an atomic state update.

        Raises:
            ModError: Content changed, prerequisites are inactive, or state is invalid.
        """
        identifier(mod_id, "mod id")
        if type(enabled) is not bool:
            raise ModError("enabled must be boolean")
        with self.store.transaction() as state:
            records, _, _ = self._records(state)
            record = records.get(mod_id)
            if record is None:
                raise ModError("Mod not found or incompatible")
            if enabled:
                if fingerprint != record["package"].fingerprint:
                    raise ModError("Mod changed; refresh and explicitly enable the displayed fingerprint")
                if record["status"] in {"dependency-cycle", "invalid-settings", "missing-requirements"}:
                    raise ModError(record.get("error", "Mod is incompatible"))
                required = record["package"].manifest["requires"]
                if any(item not in records or not records[item]["enabled"] for item in required):
                    raise ModError("Enable declared dependencies first")
                state["enabled"][mod_id] = fingerprint
            else:
                state["enabled"][mod_id] = False
        return self.catalog()

    def select(self, capability, mod_id):
        """Persist an enabled replacement provider, or reset one capability to its host default.

        Args:
            capability: Exact declared provider capability ID.
            mod_id: Enabled package ID, or None to clear only this selection.

        Returns:
            Refreshed metadata catalog. Package activation and other selections are preserved.

        Raises:
            ModError: The package is inactive, incompatible, or declares only an additive stage.
        """
        identifier(capability, "capability")
        if mod_id is not None:
            identifier(mod_id, "mod id")
        with self.store.transaction() as state:
            if mod_id is None:
                state["selections"].pop(capability, None)
            else:
                records, _, _ = self._records(state)
                record = records.get(mod_id)
                if record is None or not record["enabled"]:
                    raise ModError("Select requires an enabled compatible mod")
                declarations = record["package"].manifest["capabilities"]
                if not any(item["id"] == capability and item["kind"] == "provider" for item in declarations):
                    raise ModError("Mod does not declare this provider capability")
                state["selections"][capability] = mod_id
        return self.catalog()

    def configure(self, mod_id, settings):
        """Replace overrides with validated settings; omitted keys revert to manifest defaults.

        Args:
            mod_id: Discovered package ID, which may be disabled.
            settings: Finite JSON object, bounded to 64 KiB, validated with manifest defaults.

        Returns:
            Refreshed catalog. Existing request snapshots retain their previous settings.

        Raises:
            ModError: Settings violate the declared schema or the durable state is unreadable.
        """
        identifier(mod_id, "mod id")
        if not isinstance(settings, dict) or len(json_bytes(settings)) > 65536:
            raise ModError("Settings must be a JSON object no larger than 64 KiB")
        with self.store.transaction() as state:
            records, _, _ = self._records(state)
            if mod_id not in records:
                raise ModError("Mod not found or incompatible")
            declared = records[mod_id]["package"].manifest
            effective = {**deepcopy(declared["settings"]), **deepcopy(settings)}
            validate_value(effective, declared["settingsSchema"], "settings")
            state["settings"][mod_id] = deepcopy(settings)
        return self.catalog()

    def snapshot(self):
        """Capture exact code bytes, enabled dependencies, selection and settings once per request."""
        state = self.store.read()
        records, _, _ = self._records(state)
        return ExecutionSnapshot(records, state["selections"], self._callbacks, state.get("extensions", {}))

    def snapshot_from_wire(self, wire):
        """Restore frozen worker config without writing or using newly changed current selections.

        wire is the finite JSON object produced by ExecutionSnapshot.to_wire(). Captured
        settings and activation survive later UI changes. Changed or missing enabled code
        raises ModError; prerequisite and settings checks are repeated in this interpreter.
        The returned snapshot does not enable anything in the durable configuration.
        """
        if (not isinstance(wire, dict) or set(wire) - {"apiVersion", "mods", "selections", "extensions"}
                or not {"apiVersion", "mods", "selections"} <= set(wire)
                or type(wire["apiVersion"]) is not int or wire["apiVersion"] != 1
                or not isinstance(wire["mods"], dict) or len(wire["mods"]) > 128
                or not isinstance(wire["selections"], dict) or len(json_bytes(wire)) > 262144):
            raise ModError("Invalid worker mod snapshot")
        extensions = wire.get("extensions", {})
        if not isinstance(extensions, dict) or len(json_bytes(extensions)) > 65536:
            raise ModError("Worker snapshot extensions must be a JSON object no larger than 64 KiB")
        state = {"enabled": {}, "settings": {}, "selections": deepcopy(wire["selections"])}
        packages, _, _ = self._discover()
        for mod_id in packages:
            state["enabled"][mod_id] = False
        for capability, mod_id in state["selections"].items():
            identifier(capability)
            identifier(mod_id)
        for mod_id, item in wire["mods"].items():
            identifier(mod_id)
            if (not isinstance(item, dict) or set(item) != {"fingerprint", "enabled", "settings"}
                    or type(item["enabled"]) is not bool or not isinstance(item["settings"], dict)):
                raise ModError("Invalid captured package configuration")
            if item["enabled"]:
                if mod_id not in packages or packages[mod_id].fingerprint != item["fingerprint"]:
                    raise ModError(f"Captured mod {mod_id} changed before worker execution")
                state["enabled"][mod_id] = item["fingerprint"]
            state["settings"][mod_id] = deepcopy(item["settings"])
        records, _, _ = self._records(state)
        return ExecutionSnapshot(records, state["selections"], self._callbacks, extensions)

    def create_template(self, mod_id, template="word-count"):
        """Create an inert, disabled custom analysis-stage example in the external package directory."""
        identifier(mod_id, "mod id")
        if template not in {"word-count", "asr-seconds"}:
            raise ModError("Unknown bundled template; choose word-count or asr-seconds")
        # Use the same cross-process lock as mutations so concurrent creations cannot
        # replace each other's package. State revision also records this catalog change.
        with self.store.transaction():
            self.directory.mkdir(mode=0o700, parents=True, exist_ok=True)
            if self.directory.is_symlink():
                raise ModError("User package directory cannot be a symlink")
            if mod_id in self._builtins or (self.directory / mod_id).exists():
                raise ModError("A package with this ID already exists")
            template_root = Path(__file__).with_name("templates") / template
            with tempfile.TemporaryDirectory(dir=self.directory, prefix=".new-mod-") as temporary:
                staged = Path(temporary) / mod_id
                shutil.copytree(template_root, staged, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
                path = staged / "mod.json"
                value = json.loads(path.read_text(encoding="utf-8"))
                value["id"] = mod_id
                value["name"] = mod_id
                path.write_bytes(json_bytes(value))
                # Rename only after validating the whole package; this never activates code.
                read_package(staged)
                staged.rename(self.directory / mod_id)
        result = self.catalog()
        result["createdPath"] = str(self.directory / mod_id)
        return result
