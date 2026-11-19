"""Check runtime diagnostic boundaries without restarting or mutating a running server."""

from unittest import TestCase
from unittest.mock import patch

from voice_studio.mcp import runtime_info


class McpRuntimeTests(TestCase):
    def test_runtime_reports_restart_after_source_change_and_unknown_after_read_failure(self):
        baseline = runtime_info.STARTUP_SOURCE_REVISION
        with patch.object(runtime_info, "source_revision", return_value=baseline):
            self.assertFalse(runtime_info.runtime_info()["restartRequired"])
        with patch.object(runtime_info, "source_revision", return_value="changed"):
            self.assertTrue(runtime_info.runtime_info()["restartRequired"])
        with patch.object(runtime_info, "source_revision", side_effect=OSError):
            self.assertIsNone(runtime_info.runtime_info()["restartRequired"])
