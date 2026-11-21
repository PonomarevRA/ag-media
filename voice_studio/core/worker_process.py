"""Build child commands for modules owned by the installed Studio package."""

from voice_paths import PROJECT_ROOT


_BOOTSTRAP = """
import os
import runpy
import sys

source = sys.argv.pop(1)
module = sys.argv.pop(1)
if sys.path and sys.path[0] == "":
    sys.path[0] = source
else:
    sys.path.insert(0, source)
runtime = os.environ.get("VOICE_WORKER_RUNTIME")
if runtime:
    sys.path[1:1] = runtime.split(os.pathsep)
runpy.run_module(module, run_name="__main__", alter_sys=True)
"""


def worker_command(python, module, *arguments, unbuffered=False):
    """Run a package module with the selected interpreter and the caller's environment.

    Import the matching Studio sources independently of cwd and PYTHONPATH.
    The caller can still supply optional native dependencies through PYTHONPATH.
    """
    # Bytecode written by a worker would invalidate a signed application bundle.
    command = [str(python), "-B"]
    if unbuffered:
        command.append("-u")
    return [*command, "-c", _BOOTSTRAP, str(PROJECT_ROOT), module, *map(str, arguments)]
