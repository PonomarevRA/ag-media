"""Exercise media mod trust, selection, batch execution and request isolation contracts."""

from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from voice_studio.mods import ModError, ModExecutionError, ModRegistry
from voice_studio.mods.schema import manifest


def builtin(mod_id="builtin-asr", capability="speech.asr"):
    """Build a small host-owned provider with one validated setting."""
    return {"apiVersion": 1, "id": mod_id, "name": mod_id, "version": "1.0.0",
            "capabilities": [{"id": capability, "kind": "provider", "default": True}],
            "settingsSchema": {"type": "object", "properties": {"gain": {"type": "number", "minimum": 0, "maximum": 2}}},
            "settings": {"gain": 1}, "license": "MIT", "provenance": {"author": "test"}}


class MediaModRegistryTests(unittest.TestCase):
    """Use disposable external directories; no real models, downloads or app state are touched."""

    def setUp(self):
        self.temporary = TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.registry = ModRegistry(self.root)

    def package(self, mod_id="custom-provider", capability="speech.asr", code=None, **changes):
        """Write an inert example package and return its displayed metadata row."""
        declared = builtin(mod_id, capability)
        declared["capabilities"][0].pop("default")
        declared["capabilities"][0]["entrypoint"] = "worker.py:run"
        declared.update(changes)
        root = self.registry.directory / mod_id
        root.mkdir(parents=True, exist_ok=True)
        (root / "mod.json").write_text(json.dumps(declared), encoding="utf-8")
        (root / "worker.py").write_text(code or "def run(payload, settings):\n    return {'text': payload['text'].upper(), 'gain': settings['gain']}\n", encoding="utf-8")
        return self.row(mod_id)

    def row(self, mod_id):
        """Find a single catalog row by opaque package ID."""
        return next(item for item in self.registry.catalog()["mods"] if item["id"] == mod_id)

    def enable(self, mod_id):
        """Activate exactly the package version currently displayed by the catalog."""
        self.registry.enable(mod_id, self.row(mod_id)["fingerprint"])

    def test_discovery_never_imports_and_activation_requires_exact_hash(self):
        sentinel = self.root / "imported"
        row = self.package(code=f"from pathlib import Path\nPath({str(sentinel)!r}).touch()\ndef run(payload, settings): return payload\n")
        self.assertFalse(sentinel.exists())
        self.assertFalse(row["enabled"])
        with self.assertRaises(ModError):
            self.registry.enable(row["id"], "0" * 64)
        self.enable(row["id"])
        self.assertFalse(sentinel.exists())
        self.registry.snapshot().invoke_mod(row["id"], "speech.asr", {"text": "hello"})
        self.assertTrue(sentinel.exists())

    def test_actual_replacement_routes_payload_and_settings_and_is_durable(self):
        self.package()
        self.enable("custom-provider")
        self.registry.select("speech.asr", "custom-provider")
        self.registry.configure("custom-provider", {"gain": 1.5})
        fresh = ModRegistry(self.root)
        self.assertEqual(fresh.snapshot().invoke("speech.asr", {"text": "hello"}), {"text": "HELLO", "gain": 1.5})
        self.assertEqual(fresh.catalog()["selections"], {"speech.asr": "custom-provider"})

    def test_changed_sibling_code_is_disabled_and_never_falls_back(self):
        self.package(code="from helper import transform\ndef run(payload, settings): return transform(payload)\n")
        helper = self.registry.directory / "custom-provider" / "helper.py"
        helper.write_text("def transform(payload): return {'v': 1}\n")
        self.enable("custom-provider")
        self.registry.select("speech.asr", "custom-provider")
        captured = self.registry.snapshot()
        helper.write_text("def transform(payload): return {'v': 2}\n")
        self.assertEqual(self.row("custom-provider")["status"], "changed")
        with self.assertRaises(ModError):
            self.registry.snapshot().invoke("speech.asr", {}, default="other-builtin")
        self.assertEqual(captured.invoke("speech.asr", {}), {"v": 1})
        self.enable("custom-provider")
        self.assertEqual(self.registry.snapshot().invoke("speech.asr", {}), {"v": 2})

    def test_new_stage_template_is_inert_and_actually_executes(self):
        result = self.registry.create_template("count-words")
        self.assertTrue(Path(result["createdPath"]).is_dir())
        self.assertEqual(self.registry.snapshot().custom_stages(), [])
        self.enable("count-words")
        snapshot = self.registry.snapshot()
        stage = snapshot.custom_stages()[0]
        self.assertEqual(stage["capability"], "analysis.word-count")
        payload = {"transcript": {"text": "I love local tools"}, "raw": {"unchanged": True}}
        before = deepcopy(payload)
        self.assertEqual(snapshot.invoke_mod(stage["modId"], stage["capability"], payload)["wordCount"], 4)
        self.assertEqual(payload, before)
        with self.assertRaises(ModError):
            self.registry.select("analysis.word-count", "count-words")
        with self.assertRaises(ModError):
            self.registry.create_template("count-words")

    def test_snapshot_and_wire_ignore_later_selection_settings_disable(self):
        self.package()
        self.enable("custom-provider")
        self.registry.select("speech.asr", "custom-provider")
        captured = self.registry.snapshot()
        wire = json.loads(json.dumps(captured.to_wire()))
        self.registry.configure("custom-provider", {"gain": 2})
        self.registry.select("speech.asr", None)
        self.registry.enable("custom-provider", None, False)
        self.assertEqual(captured.invoke("speech.asr", {"text": "hello"})["gain"], 1)
        restored = ModRegistry(self.root).snapshot_from_wire(wire)
        self.assertEqual(restored.invoke("speech.asr", {"text": "hello"})["gain"], 1)
        self.assertEqual(self.registry.catalog()["selections"], {})
        self.assertFalse(self.row("custom-provider")["enabled"])

    def test_wire_rejects_changed_captured_code(self):
        self.package()
        self.enable("custom-provider")
        wire = self.registry.snapshot().to_wire()
        (self.registry.directory / "custom-provider" / "worker.py").write_text("raise RuntimeError('changed')")
        with self.assertRaisesRegex(ModError, "changed before worker"):
            self.registry.snapshot_from_wire(wire)

    def test_builtin_callbacks_keep_opaque_objects_and_default_behavior(self):
        observed = []
        token = object()

        def direct(payload, settings):
            """Return an opaque token unchanged and record exactly one direct call."""
            observed.append((payload, settings))
            return payload

        self.registry = ModRegistry(self.root, [builtin()], {("builtin-asr", "speech.asr"): direct})
        snapshot = self.registry.snapshot()
        self.assertIs(snapshot.invoke("speech.asr", token), token)
        self.assertEqual(len(observed), 1)
        self.registry.enable("builtin-asr", None, False)
        with self.assertRaises(ModError):
            self.registry.snapshot().invoke("speech.asr", token)

    def test_builtin_upgrade_preserves_opt_in_and_selection_but_rejects_old_worker_snapshot(self):
        initial = builtin()
        initial["defaultEnabled"] = False
        self.registry = ModRegistry(self.root, [initial])
        self.enable("builtin-asr")
        self.registry.select("speech.asr", "builtin-asr")
        self.registry.configure("builtin-asr", {"gain": 1.5})
        previous = self.registry.snapshot()
        previous_digest = self.row("builtin-asr")["fingerprint"]

        upgraded = {**initial, "version": "1.1.0"}
        self.registry = ModRegistry(self.root, [upgraded])
        current = self.registry.snapshot()
        self.assertEqual(current.selected("speech.asr"), "builtin-asr")
        self.assertEqual(self.row("builtin-asr")["settings"], {"gain": 1.5})
        self.assertNotEqual(current.fingerprint(["speech.asr"]), previous.fingerprint(["speech.asr"]))
        with self.assertRaisesRegex(ModError, "changed before worker"):
            self.registry.snapshot_from_wire(previous.to_wire())
        with self.assertRaisesRegex(ModError, "refresh"):
            self.registry.enable("builtin-asr", previous_digest)

        self.registry.enable("builtin-asr", None, False)
        fresh = ModRegistry(self.root, [{**initial, "version": "1.2.0", "defaultEnabled": True}])
        with self.assertRaisesRegex(ModError, "disabled"):
            fresh.snapshot().selected("speech.asr")

    def test_builtin_availability_does_not_break_worker_snapshot_after_download(self):
        model = str(self.root / "whisper.bin")
        declared = {**builtin(), "models": [{"path": model}],
                    "availability": {"files": [model], "ready": False, "reason": "missing"}}
        self.registry = ModRegistry(self.root, [declared])
        captured = self.registry.snapshot()
        available = {**declared, "availability": {"files": [model], "ready": True, "reason": "present"}}
        worker = ModRegistry(self.root, [available])
        restored = worker.snapshot_from_wire(captured.to_wire())
        self.assertEqual(restored.selected("speech.asr"), "builtin-asr")
        self.assertEqual(restored.fingerprint(["speech.asr"]), captured.fingerprint(["speech.asr"]))

    def test_capability_fingerprint_excludes_unrelated_settings_and_tracks_dynamic_defaults(self):
        asr = builtin()
        asr["capabilities"][0].pop("default")
        other = builtin("builtin-ocr", "vision.ocr")
        self.registry = ModRegistry(self.root, [asr, other])
        snapshot = self.registry.snapshot()
        self.registry.configure("builtin-ocr", {"gain": 2})
        changed_ocr = self.registry.snapshot()
        self.assertEqual(snapshot.fingerprint(["speech.asr"]), changed_ocr.fingerprint(["speech.asr"]))
        self.assertNotEqual(snapshot.fingerprint(["vision.ocr"]), changed_ocr.fingerprint(["vision.ocr"]))
        self.registry.configure("builtin-asr", {"gain": 2})
        self.assertNotEqual(snapshot.fingerprint(["speech.asr"]), self.registry.snapshot().fingerprint(["speech.asr"]))

    def test_missing_dependencies_cycle_and_transitive_disable(self):
        self.package("dependency")
        self.package("dependent", requires=["dependency"])
        with self.assertRaisesRegex(ModError, "dependencies"):
            self.enable("dependent")
        self.enable("dependency")
        self.enable("dependent")
        self.registry.enable("dependency", None, False)
        self.assertEqual(self.row("dependent")["status"], "missing-dependency")
        self.package("cycle-a", requires=["cycle-b"])
        self.package("cycle-b", requires=["cycle-a"])
        self.assertEqual(self.row("cycle-a")["status"], "dependency-cycle")
        with self.assertRaises(ModError):
            self.enable("cycle-a")

    def test_dependency_settings_affect_dependent_capability_fingerprint(self):
        self.package("dependency", capability="audio.support")
        self.package("dependent", requires=["dependency"])
        self.enable("dependency")
        self.enable("dependent")
        self.registry.select("speech.asr", "dependent")
        before = self.registry.snapshot().fingerprint(["speech.asr"])
        self.registry.configure("dependency", {"gain": 2})
        self.assertNotEqual(before, self.registry.snapshot().fingerprint(["speech.asr"]))

    def test_missing_requirements_do_not_import_or_install(self):
        self.package(requirements={"python": ">=99.0", "packages": ["voice-nonexistent-distribution-1973"], "files": ["absent.bin"]})
        row = self.row("custom-provider")
        self.assertEqual(row["status"], "missing-requirements")
        self.assertEqual(len(row["requirementsMissing"]), 3)
        with self.assertRaises(ModError):
            self.enable("custom-provider")

    def test_duplicate_builtin_id_cannot_be_shadowed(self):
        self.package("builtin-asr")
        self.registry = ModRegistry(self.root, [builtin()])
        rows = self.registry.catalog()["mods"]
        self.assertTrue(any(row["origin"] == "builtin" and row["enabled"] for row in rows))
        self.assertTrue(any("Duplicate" in row.get("error", "") for row in rows))

    def test_symlinks_and_escaped_entrypoints_are_unavailable(self):
        self.package()
        root = self.registry.directory / "custom-provider"
        (root / "escape.py").symlink_to(self.root / "outside.py")
        self.assertIn("symlink", self.row("custom-provider")["error"])
        (root / "escape.py").unlink()
        path = root / "mod.json"
        declared = json.loads(path.read_text())
        declared["capabilities"][0]["entrypoint"] = "../outside.py:run"
        path.write_text(json.dumps(declared))
        self.assertIn("relative", self.row("custom-provider")["error"])

    def test_schema_rejects_unknown_fields_bool_versions_and_invalid_settings(self):
        for changes in ({"apiVersion": True}, {"apiVersion": 2}, {"version": "latest"}, {"unknown": True}):
            declared = builtin()
            declared.update(changes)
            with self.assertRaises(ModError):
                manifest(declared, builtin=True)
        self.package()
        for settings in ({"gain": True}, {"gain": 3}, {"unexpected": 1}, {"gain": float("nan")}):
            with self.assertRaises(ModError):
                self.registry.configure("custom-provider", settings)
        self.assertEqual(self.row("custom-provider")["settings"], {"gain": 1})

    def test_corrupt_state_fails_closed_and_does_not_reset_defaults(self):
        self.root.mkdir(exist_ok=True)
        (self.root / "state.json").write_text("{broken")
        self.registry = ModRegistry(self.root, [builtin()])
        with self.assertRaisesRegex(ModError, "state"):
            self.registry.snapshot()
        self.assertEqual((self.root / "state.json").read_text(), "{broken")

    def test_custom_worker_errors_timeouts_output_bounds_and_dataclasses(self):
        cases = [("def run(payload, settings): raise ValueError('bad model')\n", "failed"),
                 ("import time\ndef run(payload, settings): time.sleep(10)\n", "exceeded"),
                 ("import os\ndef run(payload, settings): os.write(2, b'x'*100000); return {}\n", "output limit")]
        for code, message in cases:
            self.package(code=code)
            self.enable("custom-provider")
            with self.assertRaisesRegex(ModExecutionError, message):
                self.registry.snapshot().invoke_mod("custom-provider", "speech.asr", {}, timeout=.3, output_limit=1024)
        self.package(code="from __future__ import annotations\nfrom dataclasses import dataclass\n@dataclass\nclass Value:\n    number: int\ndef run(payload, settings): return {'value': Value(7).number}\n")
        self.enable("custom-provider")
        self.assertEqual(self.registry.snapshot().invoke_mod("custom-provider", "speech.asr", {}), {"value": 7})

    def test_output_schema_validation_is_explicit(self):
        declared = builtin("custom-provider")
        declared["capabilities"] = [{"id": "speech.asr", "kind": "provider", "entrypoint": "worker.py:run",
                                     "outputSchema": {"type": "integer"}}]
        self.package(capabilities=declared["capabilities"])
        self.enable("custom-provider")
        with self.assertRaisesRegex(ModExecutionError, "provider output"):
            self.registry.snapshot().invoke_mod("custom-provider", "speech.asr", {"text": "hello"})

    def test_denied_group_signal_still_kills_direct_child_without_thread_error(self):
        self.package(code="import os\ndef run(payload, settings):\n    while True: os.write(2, b'x'*100000)\n")
        self.enable("custom-provider")
        spawned = []
        original = subprocess.Popen

        def spawn(*args, **kwargs):
            """Retain the real owned process to verify cleanup after the failure."""
            process = original(*args, **kwargs)
            spawned.append(process)
            return process

        with patch("voice_studio.mods.execution.subprocess.Popen", side_effect=spawn), \
                patch("voice_studio.mods.execution.os.killpg", side_effect=PermissionError("denied"), create=True), \
                patch("threading.excepthook") as background_error:
            with self.assertRaisesRegex(ModExecutionError, "output limit"):
                self.registry.snapshot().invoke_mod("custom-provider", "speech.asr", {}, timeout=2, output_limit=1024)
        self.assertIsNotNone(spawned[0].poll())
        background_error.assert_not_called()

    def test_cross_process_atomic_updates_preserve_each_writer(self):
        code = ("import sys; from pathlib import Path; from voice_studio.mods.state import StateStore; "
                "s=StateStore(Path(sys.argv[1])); "
                "exec(\"with s.transaction() as state:\\n state['settings'][sys.argv[2]]={'v':1}\")")
        workers = [subprocess.Popen([sys.executable, "-c", code, str(self.root / "state.json"), f"writer-{i}"]) for i in range(5)]
        for worker in workers:
            self.assertEqual(worker.wait(timeout=10), 0)
        state = self.registry.store.read()
        self.assertEqual(state["revision"], 5)
        self.assertEqual(len(state["settings"]), 5)


if __name__ == "__main__":
    unittest.main()
