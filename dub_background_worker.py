"""Compatibility launcher for voice_studio.workers.dub_background_worker."""

import sys

from voice_studio import launch

launch(__name__, "voice_studio.workers.dub_background_worker", sys.modules)
