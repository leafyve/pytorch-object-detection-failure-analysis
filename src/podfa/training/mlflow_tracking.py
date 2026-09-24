"""Small local-file MLflow adapter used by the training pipeline."""

from __future__ import annotations

import os
from pathlib import Path
from types import TracebackType
from typing import Any

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

import mlflow
from mlflow import MlflowClient


class LocalMLflowTracker:
    """Track a run without requiring a hosted account or remote service."""

    def __init__(self, tracking_dir: Path | str, experiment_name: str, run_name: str) -> None:
        self.tracking_dir = Path(tracking_dir)
        self.experiment_name = experiment_name
        self.run_name = run_name
        self._active_run: mlflow.ActiveRun | None = None
        self.tracking_uri = ""

    def __enter__(self) -> LocalMLflowTracker:
        self.tracking_dir.mkdir(parents=True, exist_ok=True)
        database = (self.tracking_dir / "mlflow.db").resolve().as_posix()
        self.tracking_uri = f"sqlite:///{database}"
        mlflow.set_tracking_uri(self.tracking_uri)
        client = MlflowClient(tracking_uri=self.tracking_uri)
        experiment = client.get_experiment_by_name(self.experiment_name)
        if experiment is None:
            experiment_id = client.create_experiment(
                self.experiment_name,
                artifact_location=(self.tracking_dir / "artifacts").resolve().as_uri(),
            )
        else:
            experiment_id = experiment.experiment_id
        self._active_run = mlflow.start_run(
            experiment_id=experiment_id, run_name=self.run_name
        )
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        status = "FAILED" if exc_type is not None else "FINISHED"
        mlflow.end_run(status=status)
        self._active_run = None

    @property
    def run_id(self) -> str:
        if self._active_run is None:
            raise RuntimeError("MLflow run is not active")
        return self._active_run.info.run_id

    def log_params(self, values: dict[str, Any]) -> None:
        mlflow.log_params({key: str(value) for key, value in values.items()})

    def log_metrics(self, values: dict[str, float], step: int | None = None) -> None:
        mlflow.log_metrics(values, step=step)

    def log_artifact(self, path: Path | str, artifact_path: str | None = None) -> None:
        mlflow.log_artifact(str(path), artifact_path=artifact_path)
