"""
Verify that interval transport changes preserve every JSON distinction and reject bad wire.
"""

from copy import deepcopy
import json
from unittest import TestCase

from voice_studio.graph.media_graph_interval_wire import pack_interval, unpack_interval


class IntervalWireTests(TestCase):
    """
    Exercise the codec contract independently of graph selection and frame extraction.
    """

    def packet(self):
        """
        Return sparse, typed and colliding future fields alongside repeated evidence.
        """
        identity = "utterance:1000:2500:long-exact-source-id"
        policy = "Whole ASR cue; not word-aligned to actual pixels or selected interval."
        return {"schemaVersion": "media-interval/1", "snapshotId": "frozen", "sourceSha256": "abc",
                "future": [{"n": None}, {"n": False}, {}, {"n": 0}, {"n": "0"}],
                "sections": {"speech": {"total": 9, "omitted": 7, "items": [
                    {"id": identity, "label": "Привет 🌍", "policy": policy, "score": .123456789,
                     "startMs": 999, "endMs": 2500, "false": False, "nullable": None},
                    {"id": identity, "label": "Different facts", "policy": policy, "score": 0,
                     "startMs": 1001, "endMs": 2501, "false": True, "nullable": None}]}},
                "markers": [{"stringRef": 0}, {"columns": ["x"], "rows": [[None]]},
                            {"literal": [["x", 1]]}, {"stringRef": False, "other": True}],
                "frames": {"samples": [{"sourcePts": 3003, "actualMs": 100.1, "sourceTimeBase": "1/30000"}],
                           "context": {"cueId": identity, "textSpan": policy}},
                "coverage": {"declaredStartMs": 0, "samples": 100, "countScope": "declared-window"}}

    def test_round_trip_all_values_and_no_mutation(self):
        """
        Recover complete JSON including unknown fields, duplicate IDs and marker collisions.
        """
        original = self.packet()
        before = deepcopy(original)
        for dictionary in (False, True):
            with self.subTest(dictionary=dictionary):
                wire = pack_interval(original, dictionary=dictionary)
                frozen = deepcopy(wire)
                decoded = unpack_interval(json.loads(json.dumps(wire, ensure_ascii=False)))
                self.assertEqual(decoded, original)
                self.assertEqual(json.dumps(decoded, sort_keys=True), json.dumps(original, sort_keys=True))
                self.assertEqual(original, before)
                self.assertEqual(wire, frozen)
                self.assertIsNot(decoded, original)
        self.assertTrue(pack_interval(original, dictionary=True)["strings"])

    def test_literal_shapes_inside_uniform_tables(self):
        """
        Table expansion does not reinterpret original rows that match reserved markers.
        """
        for rows in ([{"stringRef": 0}, {"stringRef": False}],
                     [{"literal": []}, {"literal": [["z", None]]}],
                     [{"columns": [], "rows": []}, {"columns": ["a"], "rows": [[]]}]):
            original = {"schemaVersion": "media-interval/1", "future": rows}
            self.assertEqual(unpack_interval(pack_interval(original, dictionary=True)), original)

    def test_invalid_tables_and_references(self):
        """
        Reject invalid encoded structures instead of silently overwriting or coercing data.
        """
        for invalid in ({"stringRef": False}, {"stringRef": -1}, {"stringRef": 999},
                        {"columns": ["x", "x"], "rows": [[1, 2]]},
                        {"columns": [0], "rows": [[1]]},
                        {"columns": ["x"], "rows": [[]]},
                        {"literal": [["x", 1], ["x", 2]]}, {"literal": [[False, 1]]}):
            wire = pack_interval(self.packet(), dictionary=True)
            wire["payload"]["future"] = invalid
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                unpack_interval(wire)

    def test_invalid_input_schemas_and_nonfinite_json(self):
        """
        Reject unknown envelopes, non-JSON values and nonfinite numbers explicitly.
        """
        for value in (float("nan"), float("inf"), float("-inf"), (1, 2), {1: "bad"}):
            with self.subTest(value=value), self.assertRaises(ValueError):
                pack_interval({"schemaVersion": "media-interval/1", "future": value})
        for changed in ({"schemaVersion": "media-interval/3"}, {"encoding": "opaque"},
                        {"legend": {}}, {"newEnvelopeField": True}, {"strings": [False]}):
            wire = pack_interval(self.packet(), dictionary=True)
            wire.update(changed)
            with self.subTest(changed=changed), self.assertRaises(ValueError):
                unpack_interval(wire)

    def test_expansion_and_recursion_limits(self):
        """
        Bound expanded repeated strings and nested payloads, before oversized restoration.
        """
        wire = pack_interval({"schemaVersion": "media-interval/1"}, dictionary=True)
        wire["strings"] = ["x" * 100_000]
        wire["payload"]["future"] = [{"stringRef": 0} for _ in range(50)]
        with self.assertRaisesRegex(ValueError, "byte limit"):
            unpack_interval(wire)
        nested = None
        for _ in range(70):
            nested = [nested]
        with self.assertRaisesRegex(ValueError, "depth"):
            pack_interval({"schemaVersion": "media-interval/1", "future": nested})
