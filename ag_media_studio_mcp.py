"""Launch the ag-media-studio MCP server with the existing tool contracts."""

import sys

from voice_studio import launch

launch(__name__, "voice_studio.mcp.voice_mcp", sys.modules)
