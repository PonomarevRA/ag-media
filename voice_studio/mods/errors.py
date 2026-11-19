"""Public failures for media package discovery, activation and execution."""


class ModError(RuntimeError):
    """A mod is invalid, unavailable, changed, or failed; callers must expose the error."""


class ModExecutionError(ModError):
    """An enabled provider exceeded its budget or returned an invalid response."""
