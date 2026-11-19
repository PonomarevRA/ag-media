"""Compatibility launcher for voice_studio.mcp.voice_mcp."""

import sys

from voice_studio import launch

launch(__name__, "voice_studio.mcp.voice_mcp", sys.modules)
