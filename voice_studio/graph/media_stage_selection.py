"""Validate selective media runs and expose pending work without starting a worker."""

STAGE_NAMES = ("speech", "audio", "visual", "people", "scene", "voice", "object_review")
DEPENDENCIES = {"audio": ("speech",), "visual": ("speech",), "people": ("visual",), "voice": ("speech",)}


def stage_names(value, label):
    """Accept explicit lists only; reject unknown stages before any expensive analysis."""
    if value is None:
        return None
    if not isinstance(value, (list, tuple)) or any(not isinstance(name, str) or name not in STAGE_NAMES for name in value):
        raise ValueError(f"{label}: допустимы только {', '.join(STAGE_NAMES)}.")
    return set(value)


def query_stage_names(query, key):
    """Parse one comma-separated query value, including an explicitly empty selection."""
    if key not in query:
        return None
    values = query[key]
    if len(values) != 1:
        raise ValueError(f"Параметр {key} должен быть указан один раз.")
    names = values[0].split(",") if values[0] else []
    stage_names(names, key)
    return names


class StageSelection:
    """Expand required inputs, leaving unrelated stages eligible for cache-only reuse."""

    def __init__(self, selected=None, forced=None):
        self.selected = stage_names(selected, "selected_stages")
        self.forced = stage_names(forced, "force_stages") or set()
        self.requested = None if self.selected is None else self.selected | self.forced
        self.enabled = None if self.requested is None else set(self.requested)
        if self.enabled is not None:
            pending = list(self.enabled)
            while pending:
                for dependency in DEPENDENCIES.get(pending.pop(), ()):
                    if dependency not in self.enabled:
                        self.enabled.add(dependency)
                        pending.append(dependency)
        self.states = {}

    @property
    def selective(self):
        """Default calls keep their historical flags and output format."""
        return self.selected is not None or bool(self.forced)

    def runs(self, name):
        """Dependencies may read checkpoints and compute only when those are missing."""
        return self.enabled is None or name in self.enabled

    def mark(self, name, state):
        """Record one layer's actual action; never label a placeholder completed."""
        self.states[name] = {"state": state, "selected": self.requested is None or name in self.requested,
                             "forced": name in self.forced,
                             "dependency": self.enabled is not None and name in self.enabled and name not in self.requested}

    def attach(self, report):
        """Expose all stages, including disabled or not-yet-measured work."""
        if self.selective:
            for name in STAGE_NAMES:
                if name not in self.states:
                    self.mark(name, "pending")
            report["analysisStages"] = dict(self.states)
        return report
