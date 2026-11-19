"""Local Voice Studio media analysis, synthesis and application entry points."""

from importlib import import_module
from runpy import run_module


def launch(name, module, modules):
    """Run a configured script or share its canonical module with older callers."""
    if name == "__main__":
        run_module(module, run_name="__main__", alter_sys=True)
    else:
        modules[name] = import_module(module)
