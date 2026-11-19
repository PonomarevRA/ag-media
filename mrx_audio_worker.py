"""Compatibility launcher for voice_studio.workers.mrx_audio_worker."""

import sys

from voice_studio import launch

launch(__name__, "voice_studio.workers.mrx_audio_worker", sys.modules)
