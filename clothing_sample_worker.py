"""Compatibility launcher for voice_studio.workers.clothing_sample_worker."""

import sys

from voice_studio import launch

launch(__name__, "voice_studio.workers.clothing_sample_worker", sys.modules)
