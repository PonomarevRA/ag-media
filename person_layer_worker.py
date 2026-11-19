"""Compatibility launcher for voice_studio.workers.person_layer_worker."""

import sys

from voice_studio import launch

launch(__name__, "voice_studio.workers.person_layer_worker", sys.modules)
