"""Exercise host contracts and real same-worker model-format adapters without dependencies."""

from copy import deepcopy
import json
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from voice_studio.mods import ModError, ModExecutionError, ModRegistry
from voice_studio.mod_contracts import contract_for_capability, get_contracts, validate_contract


class MediaModContractTests(unittest.TestCase):
    """Use synthetic vendor responses; no real model, weight or installed state is changed."""

    def setUp(self):
        self.temporary = TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.registry = ModRegistry(self.temporary.name)
        self.registry.create_template("vendor-asr", template="asr-seconds")
        self.root = self.registry.directory / "vendor-asr"

    def declared(self):
        """Read the example declaration for a focused fixture change."""
        return json.loads((self.root / "mod.json").read_text())

    def write_manifest(self, value):
        """Publish edited test metadata while leaving activation explicitly disabled/changed."""
        (self.root / "mod.json").write_text(json.dumps(value))

    def row(self):
        """Return the example catalog row including independent validation errors."""
        return next(item for item in self.registry.catalog()["mods"] if item["id"] == "vendor-asr")

    def invoke(self, payload=None):
        """Enable the current displayed bytes and invoke the replacement through selection."""
        self.registry.enable("vendor-asr", self.row()["fingerprint"])
        self.registry.select("speech.asr", "vendor-asr")
        return self.registry.snapshot().invoke("speech.asr", payload or {"jobs": [["/local/audio.wav", "/local/result"]], "language": "en"})

    def test_contract_catalog_is_machine_readable_detached_and_capability_bound(self):
        catalog = get_contracts()
        self.assertEqual(catalog["apiVersion"], 1)
        asr = next(item for item in catalog["contracts"] if item["id"] == "studio.asr")
        self.assertEqual(asr["version"], 1)
        self.assertEqual(asr["units"]["transcription[].offsets.from"], "milliseconds")
        self.assertIn("schema", asr)
        self.assertTrue(asr["supportLimits"])
        asr["schema"].clear()
        self.assertTrue(next(item for item in get_contracts()["contracts"] if item["id"] == "studio.asr")["schema"])
        self.assertEqual(contract_for_capability("vision.faces"), {"id": "studio.model-resource", "version": 1})
        self.assertIsNone(contract_for_capability("vision.person-batch"))

    def test_unknown_versions_incompatible_refs_and_missing_hooks_fail_before_import(self):
        sentinel = Path(self.temporary.name) / "imported"
        (self.root / "model.py").write_text(f"from pathlib import Path\nPath({str(sentinel)!r}).touch()\ndef run(payload, settings): return []\n")
        original = self.declared()
        for reference in ({"id": "studio.unknown", "version": 1}, {"id": "studio.asr", "version": 2},
                          {"id": "studio.asr", "version": True}, {"id": "studio.scene", "version": 1}):
            value = deepcopy(original)
            value["capabilities"][0]["contract"] = reference
            self.write_manifest(value)
            self.assertEqual(self.row()["status"], "error")
            self.assertFalse(sentinel.exists())
        for hook in ("../escape.py:run", "missing.py:run"):
            value = deepcopy(original)
            value["capabilities"][0]["convert_output"] = hook
            self.write_manifest(value)
            self.assertEqual(self.row()["status"], "error")
            self.assertFalse(sentinel.exists())

    def test_real_vendor_seconds_convert_to_studio_millisecond_offsets(self):
        result = self.invoke()
        self.assertEqual(result[0]["transcription"], [{"text": "Example vendor transcript", "offsets": {"from": 125, "to": 1375}}])
        self.assertEqual(result[0]["result"]["language"], "en")

    def test_prepare_model_convert_share_one_module_state_and_one_process(self):
        value = self.declared()
        cap = value["capabilities"][0]
        cap.update(entrypoint="combined.py:run", prepare_input="combined.py:prepare", convert_output="combined.py:convert")
        cap.pop("rawOutputSchema")
        self.write_manifest(value)
        (self.root / "combined.py").write_text(
            "import os\npids=[]\ndef prepare(payload, settings):\n pids.append(os.getpid()); return {'prepared':True}\n"
            "def run(payload, settings):\n assert payload['prepared']; pids.append(os.getpid()); return {'text':'same worker'}\n"
            "def convert(raw, settings):\n pids.append(os.getpid()); return [{'transcription':[{'text':raw['text'],'offsets':{'from':0,'to':1}}], 'workerPids':pids}]\n")
        original = subprocess.Popen
        with patch("voice_studio.mods.execution.subprocess.Popen", wraps=original) as spawn:
            result = self.invoke()
        self.assertEqual(spawn.call_count, 1)
        self.assertEqual(len(result[0]["workerPids"]), 3)
        self.assertEqual(len(set(result[0]["workerPids"])), 1)

    def test_raw_schema_is_checked_before_converter_can_hide_invalid_vendor_data(self):
        sentinel = Path(self.temporary.name) / "converted"
        (self.root / "model.py").write_text("def run(payload, settings): return [{'language':'en','utterances':'wrong type'}]\n")
        (self.root / "adapter.py").write_text(
            f"from pathlib import Path\ndef prepare_input(payload, settings): return payload\ndef convert_output(raw, settings):\n Path({str(sentinel)!r}).touch(); return []\n")
        with self.assertRaisesRegex(ModExecutionError, r"model raw output\[0\].utterances"):
            self.invoke()
        self.assertFalse(sentinel.exists())

    def test_bad_conversion_is_rejected_before_returning_to_consumer(self):
        (self.root / "model.py").write_text("def run(payload, settings): return []\n")
        for stop in (-1, .5, True, "20"):
            (self.root / "adapter.py").write_text(
                "def prepare_input(payload, settings): return payload\n"
                f"def convert_output(raw, settings): return [{{'transcription':[{{'text':'bad','offsets':{{'from':10,'to':{stop!r}}}}}]}}]\n")
            with self.assertRaisesRegex(ModExecutionError, r"provider output\[0\].transcription\[0\].offsets.to"):
                self.invoke()

    def test_input_contract_is_validated_before_model_import(self):
        with self.assertRaisesRegex(ModError, "provider input"):
            self.invoke(["not a named object"])

    def test_adapter_bytes_are_pinned_and_running_snapshot_uses_captured_conversion(self):
        self.registry.enable("vendor-asr", self.row()["fingerprint"])
        self.registry.select("speech.asr", "vendor-asr")
        snapshot = self.registry.snapshot()
        before = snapshot.fingerprint(["speech.asr"])
        (self.root / "adapter.py").write_text("def prepare_input(payload, settings): return payload\ndef convert_output(raw, settings): raise RuntimeError('changed adapter')\n")
        self.assertEqual(self.row()["status"], "changed")
        self.assertNotEqual(before, self.registry.snapshot().fingerprint(["speech.asr"]))
        result = snapshot.invoke("speech.asr", {"jobs": [["/audio.wav", "/result"]], "language": "en"})
        self.assertEqual(result[0]["transcription"][0]["offsets"]["from"], 125)

    def test_existing_output_schema_still_validates_converted_final_result(self):
        value = self.declared()
        value["capabilities"][0]["outputSchema"] = {"type": "integer"}
        self.write_manifest(value)
        with self.assertRaisesRegex(ModExecutionError, "provider output must be integer"):
            self.invoke()

    def test_semantic_asr_turns_vectors_boxes_and_finite_extra_fields(self):
        cases = [
            ("speech.asr", [{"transcription": [{"text": "x", "offsets": {"from": 20, "to": 10}}]}], "offsets.to"),
            ("speech.diarization", [{"startMs": 20, "endMs": 30, "speaker": 1}, {"startMs": 10, "endMs": 40, "speaker": 2}], "startMs"),
            ("speech.diarization", [{"startMs": 0, "endMs": 0, "speaker": 1}], "endMs"),
            ("speech.embedding", {"1": [[1, 2], [1]]}, "embedding vectors"),
            ("audio.classification", [[{"start": 0, "end": 1, "scores": [{"label": "x", "score": 1.1}]}], "ready"], "score"),
            ("vision.visual-batch", {"stages": {}, "phrases": {}, "extra": float("nan")}, "extra"),
            ("vision.scene-batch", {"status": "ready", "frames": [{"timeMs": 0, "objects": [{"label": "x", "score": .5, "box": [.8, .1, .3, .2]}], "text": []}]}, "box"),
        ]
        for capability, result, error in cases:
            with self.assertRaisesRegex(ModError, error):
                validate_contract(result, contract_for_capability(capability), "output", capability=capability)
        validate_contract([{"startMs": 0, "endMs": 10, "speaker": 1}, {"startMs": 5, "endMs": 15, "speaker": 2}],
                          {"id": "studio.diarization", "version": 1})

    def test_resource_contract_checks_absolute_existing_artifact(self):
        reference = {"id": "studio.model-resource", "version": 1}
        filename = Path(self.temporary.name) / "fixture.onnx"
        filename.write_bytes(b"fixture; loader compatibility is outside the contract")
        validate_contract({"modelPath": str(filename)}, reference)
        for value in ("relative.onnx", str(filename.with_name("missing.onnx"))):
            with self.assertRaisesRegex(ModError, "modelPath"):
                validate_contract({"modelPath": value}, reference)

    def test_pipeline_extension_is_atomic_frozen_wire_and_legacy_compatible(self):
        with self.registry.store.transaction() as state:
            state["extensions"]["pipeline"] = {"dependencies": {"mod:a:b": ["builtin:speech"]}}
        snapshot = self.registry.snapshot()
        wire = snapshot.to_wire()
        with self.registry.store.transaction() as state:
            state["extensions"]["pipeline"] = {"dependencies": {}}
        self.assertEqual(snapshot.extension("pipeline"), {"dependencies": {"mod:a:b": ["builtin:speech"]}})
        copy = snapshot.extension("pipeline")
        copy.clear()
        self.assertTrue(snapshot.extension("pipeline"))
        self.assertEqual(self.registry.snapshot_from_wire(wire).extension("pipeline"), snapshot.extension("pipeline"))
        wire.pop("extensions")
        self.assertIsNone(self.registry.snapshot_from_wire(wire).extension("pipeline"))
        legacy = self.registry.store.read()
        legacy.pop("extensions")
        self.registry.store.path.write_text(json.dumps(legacy))
        self.assertEqual(self.registry.store.read()["extensions"], {})

    def test_oversized_or_nonobject_extensions_do_not_corrupt_state(self):
        before = self.registry.store.read()
        for extensions in (["wrong"], {"large": "x" * 65536}, {"bad": float("nan")}):
            with self.assertRaises(ModError):
                with self.registry.store.transaction() as state:
                    state["extensions"] = extensions
            self.assertEqual(self.registry.store.read(), before)
            wire = self.registry.snapshot().to_wire()
            wire["extensions"] = extensions
            with self.assertRaises(ModError):
                self.registry.snapshot_from_wire(wire)


if __name__ == "__main__":
    unittest.main()
