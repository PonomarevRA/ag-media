"""Launch ag-media-studio; the voice_studio package remains import-compatible."""

import sys

from voice_studio import launch

launch(__name__, "voice_studio.app.server", sys.modules)
