"""Find optional inference wheels installed outside the application bundle."""

from importlib.util import find_spec
import os
import platform
import sys

from voice_paths import dependency_path


def runtime_path(group):
    """Keep compiled wheels isolated by platform, architecture and Python ABI."""
    tag = f"{sys.implementation.cache_tag}-{sys.platform}-{platform.machine()}"
    return dependency_path("model-runtimes", tag, group)


def runtime_ready(group):
    """Return whether the installer verified and published this runtime."""
    return (runtime_path(group) / ".studio-ready").is_file()


def runtime_environment(group, environment=None):
    """Prepend the managed runtime for a worker without changing the host's imports."""
    env = dict(os.environ if environment is None else environment)
    env.pop("VOICE_WORKER_RUNTIME", None)
    if runtime_ready(group):
        env["VOICE_WORKER_RUNTIME"] = str(runtime_path(group))
    return env


def sherpa_ready():
    """Recognize the managed Sherpa runtime and previous source installations."""
    return runtime_ready("sherpa") or find_spec("sherpa_onnx") is not None
