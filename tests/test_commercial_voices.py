"""Cover catalog gating and the offline MLX dispatch contract."""

import json
from unittest import TestCase
from unittest.mock import patch

from voice_studio.synthesis.engines import availability, generate


class CommercialVoiceTests(TestCase):
    def test_default_catalog_hides_research_weights_without_probing_fish(self):
        with patch('voice_studio.synthesis.engines.research_enabled', return_value=False), \
             patch('voice_studio.synthesis.engines.commercial_voice.ready', return_value=True), \
             patch('voice_studio.synthesis.engines.fish_request') as fish:
            status = availability(False, "ru")
        self.assertEqual(set(status), {"qwen", "chatterbox"})
        self.assertTrue(all(info["ready"] and info["automatic"] for info in status.values()))
        fish.assert_not_called()

    def test_reference_and_seed_are_passed_to_offline_worker(self):
        source = {"path": "/local/reference.wav", "text": "Source"}
        with patch('voice_studio.synthesis.engines.commercial_voice.generate') as local:
            generate("qwen", {"output": "/local/result.wav", "references": [source], "seed": 17,
                              "language": "en", "text": "New line", "deadline": None}, False)
        model, payload, timeout = local.call_args.args
        self.assertEqual((model, timeout), ("qwen", 900))
        self.assertEqual(json.loads(payload)["references"], [source])
        self.assertEqual(json.loads(payload)["seed"], 17)
        self.assertNotIn("deadline", json.loads(payload))
