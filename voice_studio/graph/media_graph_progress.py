"""Bounded progress state for immutable snapshots created by one graph build."""

from threading import Lock


class GraphPublication:
    """Expose a coherent revision only after its snapshot is available on disk."""

    def __init__(self, job_id):
        self.job_id = job_id
        self.lock = Lock()
        self.latest = {}
        self.activities = {}

    def state(self):
        """Read the latest immutable envelope under the publication lock."""
        with self.lock:
            state = dict(self.latest)
            if self.activities:
                state['workingStages'] = list(self.activities.values())
                state['snapshotPhase'] = state.get('phase')
                state['phase'] = next(reversed(self.activities.values()))['stage']
            return state

    def activity(self, event):
        """Expose actual running workers separately from the last readable snapshot phase."""
        name = event.get('stage')
        if not isinstance(name, str):
            return
        with self.lock:
            if event.get('state') == 'running':
                self.activities[name] = {key: event[key] for key in ('stage', 'state', 'detail', 'windowStartMs') if key in event}
            else:
                self.activities.pop(name, None)

    def publish(self, phase, operation):
        """Build a snapshot with its revision metadata, then advertise it atomically."""
        previous = self.state()
        revision = previous.get("revision", 0) + 1
        marker = {"jobId": self.job_id, "phase": phase, "revision": revision, "partial": phase != "complete"}
        if previous.get('snapshot'):
            marker["previousSnapshotId"] = previous["snapshot"]["snapshotId"]
        snapshot = operation(marker)
        with self.lock:
            self.latest = {"phase": phase, "revision": revision, "snapshot": snapshot}
            if phase == 'complete':
                self.activities.clear()
        return snapshot
