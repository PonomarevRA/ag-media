"""Compatibility launcher for voice_studio.synthesis.setup_commercial_voices."""

import sys

from voice_studio import launch

launch(__name__, "voice_studio.synthesis.setup_commercial_voices", sys.modules)
