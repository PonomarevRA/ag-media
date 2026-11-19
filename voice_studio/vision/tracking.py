"""Record experiment metadata in a local MLflow SQLite store."""

from mlflow import MlflowClient
from mlflow.exceptions import MlflowException


class Tracking:
    """Track timings and listener feedback without logging recordings or text."""

    def __init__(self, root):
        self.client = MlflowClient(tracking_uri=f"sqlite:///{root / 'mlflow.db'}")
        experiment = self.client.get_experiment_by_name("Voice Studio")
        self.experiment = experiment.experiment_id if experiment else self.client.create_experiment(
            "Voice Studio", artifact_location=(root / "mlflow-artifacts").as_uri()
        )

    def start(self, model, language, references, demo, mode, profile, checkpoint_id=None):
        """Create a run for one candidate or training job and record its provenance."""
        run = self.client.create_run(self.experiment, tags={"mlflow.runName": model})
        run_id = run.info.run_id
        for key, value in {"model": model, "language": language, "references": references,
                           "demo": demo, "mode": mode, "profile": profile}.items():
            self.client.log_param(run_id, key, str(value))
        if checkpoint_id:
            self.client.log_param(run_id, "checkpoint_id", checkpoint_id)
        return run_id

    def finish(self, run_id, elapsed, duration=None, failed=False):
        """Close the run after recording wall time and optional audio duration."""
        self.client.log_metric(run_id, "elapsed_seconds", elapsed)
        if duration is not None:
            self.client.log_metric(run_id, "audio_seconds", duration)
            self.client.log_metric(run_id, "real_time_factor", elapsed / duration)
        self.client.set_terminated(run_id, "FAILED" if failed else "FINISHED")

    def rate(self, run_id, score):
        """Store a listener's likeness rating on the corresponding model run."""
        if run_id:
            self.client.log_metric(run_id, "listener_similarity", score)

    def interrupt(self, run_id):
        """Close a tracked job that was interrupted by a process restart."""
        if run_id:
            try:
                self.client.set_terminated(run_id, "KILLED")
            except MlflowException as exc:
                print(f"MLflow interruption: {exc}", flush=True)
