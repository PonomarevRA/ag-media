"""Compatibility launcher for voice_studio.workers.commercial_voice_worker."""

import sys

from voice_studio import launch

launch(__name__, "voice_studio.workers.commercial_voice_worker", sys.modules)
