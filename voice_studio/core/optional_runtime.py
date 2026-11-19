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


def runtime_environment(group, environment=None, *, extra_groups=()):
    """Prepend installed worker runtimes in priority order without changing host imports."""
    env = dict(os.environ if environment is None else environment)
    env.pop("VOICE_WORKER_RUNTIME", None)
    paths = [str(runtime_path(name)) for name in (group, *extra_groups) if runtime_ready(name)]
    if paths:
        env["VOICE_WORKER_RUNTIME"] = os.pathsep.join(paths)
    return env


def sherpa_ready():
    """Recognize the managed Sherpa runtime and previous source installations."""
    return runtime_ready("sherpa") or find_spec("sherpa_onnx") is not None
