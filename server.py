"""Compatibility launcher for voice_studio.app.server."""

import sys

from voice_studio import launch

launch(__name__, "voice_studio.app.server", sys.modules)
