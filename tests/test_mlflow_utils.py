from unittest.mock import MagicMock, patch

from src.mlflow_utils import register_model_if_improved


@patch("src.mlflow_utils.mlflow")
def test_registers_when_no_production_model_exists(mock_mlflow):
    mock_client = MagicMock()
    mock_client.get_latest_versions.return_value = []
    mock_mlflow.tracking.MlflowClient.return_value = mock_client
    mock_mlflow.register_model.return_value = MagicMock(version="1")

    result = register_model_if_improved(
        model_uri="runs:/abc/model",
        model_name="event_classifier",
        metric_name="accuracy",
        metric_value=0.9,
    )

    assert result is True
    mock_mlflow.register_model.assert_called_once()


@patch("src.mlflow_utils.mlflow")
def test_skips_registration_when_incumbent_is_better(mock_mlflow):
    mock_client = MagicMock()
    mock_version = MagicMock(run_id="run-1")
    mock_client.get_latest_versions.return_value = [mock_version]
    mock_run = MagicMock()
    mock_run.data.metrics = {"accuracy": 0.95}
    mock_client.get_run.return_value = mock_run
    mock_mlflow.tracking.MlflowClient.return_value = mock_client

    result = register_model_if_improved(
        model_uri="runs:/abc/model",
        model_name="event_classifier",
        metric_name="accuracy",
        metric_value=0.90,
    )

    assert result is False
    mock_mlflow.register_model.assert_not_called()


@patch("src.mlflow_utils.mlflow")
def test_registers_when_new_model_beats_incumbent(mock_mlflow):
    mock_client = MagicMock()
    mock_version = MagicMock(run_id="run-1")
    mock_client.get_latest_versions.return_value = [mock_version]
    mock_run = MagicMock()
    mock_run.data.metrics = {"accuracy": 0.85}
    mock_client.get_run.return_value = mock_run
    mock_mlflow.tracking.MlflowClient.return_value = mock_client
    mock_mlflow.register_model.return_value = MagicMock(version="2")

    result = register_model_if_improved(
        model_uri="runs:/abc/model",
        model_name="event_classifier",
        metric_name="accuracy",
        metric_value=0.90,
    )

    assert result is True
    mock_client.transition_model_version_stage.assert_called_once_with("event_classifier", "2", "Production")
