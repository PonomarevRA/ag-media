"""Regression for exporting the model updated by the optimizer."""

from unittest import TestCase

from trainers.f5 import inference_weights


class F5ExportTests(TestCase):
    def test_export_uses_online_tensors_and_keeps_loader_metadata(self):
        """Use the changed tensor even when the saved EMA tensor is still stale."""
        trained = {"model_state_dict": {"transformer.weight": "updated"},
                   "ema_model_state_dict": {"ema_model.transformer.weight": "stale",
                                            "initted": True, "step": 3}}
        self.assertEqual(inference_weights(trained), {"ema_model.transformer.weight": "updated",
                                                      "initted": True, "step": 3})
