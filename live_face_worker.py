"""Compatibility launcher for voice_studio.workers.live_face_worker."""

import sys

from voice_studio import launch

launch(__name__, "voice_studio.workers.live_face_worker", sys.modules)
