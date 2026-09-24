from __future__ import annotations

from pathlib import Path

from mlflow import MlflowClient

from podfa.training.mlflow_tracking import LocalMLflowTracker


def test_local_mlflow_tracker_records_params_metrics_and_artifact(tmp_path: Path) -> None:
    tracking_dir = tmp_path / "mlruns"
    artifact = tmp_path / "environment.json"
    artifact.write_text('{"device": "cpu"}\n', encoding="utf-8")

    with LocalMLflowTracker(tracking_dir, "fixture-experiment", "fixture-run") as tracker:
        tracker.log_params({"architecture": "fixture", "seed": 42})
        tracker.log_metrics({"train_loss": 1.25, "map_50": 0.5}, step=1)
        tracker.log_artifact(artifact)
        run_id = tracker.run_id

    client = MlflowClient(tracking_uri=tracker.tracking_uri)
    run = client.get_run(run_id)
    artifacts = client.list_artifacts(run_id)

    assert run.data.params["architecture"] == "fixture"
    assert run.data.metrics["map_50"] == 0.5
    assert [item.path for item in artifacts] == ["environment.json"]
