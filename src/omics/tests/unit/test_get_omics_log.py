from unittest.mock import Mock, patch

import pytest
from ugbio_omics.get_omics_log import OMICS_LOG_GROUP, fetch_save_log, get_log_for_task


@pytest.fixture
def mock_boto3_client():
    with patch("boto3.client") as mock_client:
        # Create separate mock clients for "logs" and "omics"
        mock_logs_client = Mock()
        mock_omics_client = Mock()

        # Configure the side_effect to return the appropriate mock client based on the service name
        def client_side_effect(service_name, *args, **kwargs):
            if service_name == "logs":
                return mock_logs_client
            elif service_name == "omics":
                return mock_omics_client
            else:
                raise ValueError(f"Unexpected service name: {service_name}")

        mock_boto3_client.side_effect = client_side_effect

        # Configure the mock clients
        mock_logs_client.get_log_events.return_value = {
            "events": [{"message": "log message 1"}, {"message": "log message 2"}]
        }
        yield mock_client


@pytest.fixture
def mock_get_run_info():
    with patch("ugbio_omics.get_omics_log.get_run_info") as mock_get_run_info:
        mock_get_run_info.return_value = {
            "tasks": [
                {"taskId": "taskID1", "name": "taskName1", "status": "SUCCEEDED"},
                {"taskId": "taskID2", "name": "taskName2", "status": "FAILED"},
            ],
            "status": "SUCCEEDED",
        }
        yield mock_get_run_info


def test_get_log_for_task_all_tasks(mock_boto3_client, mock_get_run_info, tmpdir):
    get_log_for_task("runID", output_path=tmpdir, output_prefix="test_", failed=False)

    mock_get_run_info.assert_called_once()
    mock_boto3_client.assert_called_with("logs")
    assert len(tmpdir.listdir()) == 2  # check that 2 log files created for all tasks
    assert any(f.basename.endswith("taskID1_taskName1.log") for f in tmpdir.listdir())
    assert any(f.basename.endswith("taskID2_taskName2.log") for f in tmpdir.listdir())


def test_get_log_for_task_failed_tasks(mock_boto3_client, mock_get_run_info, tmpdir):
    get_log_for_task("runID", output_path=tmpdir, output_prefix="test_")

    mock_get_run_info.assert_called_once()
    mock_boto3_client.assert_called_with("logs")
    assert len(tmpdir.listdir()) == 1  # check that one log file created for the failed task
    assert any(f.basename.endswith("taskID2_taskName2.log") for f in tmpdir.listdir())


def test_get_log_for_task_get_engine_log(mock_boto3_client, tmpdir):
    with patch("ugbio_omics.get_omics_log.get_run_info") as mock_get_run_info:
        mock_get_run_info.return_value = {
            "tasks": [
                {"taskId": "taskID1", "name": "taskName1", "status": "SUCCEEDED"},
                {"taskId": "taskID2", "name": "taskName2", "status": "SUCCEEDED"},
            ],
            "status": "FAILED",
        }

        get_log_for_task("runID", output_path=tmpdir, output_prefix="test_")

        mock_get_run_info.assert_called_once()
    mock_boto3_client.assert_called_with("logs")
    assert len(tmpdir.listdir()) == 1  # check that engine log created
    assert any(f.basename.endswith("engine.log") for f in tmpdir.listdir())


def test_get_log_for_task_get_engine_log(mock_boto3_client, tmpdir):
    with patch("ugbio_omics.get_omics_log.get_run_info") as mock_get_run_info:
        mock_get_run_info.return_value = {
            "tasks": [
                {"taskId": "task1", "name": "Task_1", "status": "SUCCEEDED"},
                {"taskId": "task2", "name": "Task_2", "status": "SUCCEEDED"},
            ],
            "status": "FAILED",
        }

        get_log_for_task("run1", output_path=tmpdir, output_prefix="test_", exclude_failed=True)

        mock_get_run_info.assert_called_once()
    mock_boto3_client.assert_called_with("logs")
    assert len(tmpdir.listdir()) == 3  # check that 2 log files created for all tasks + engine log


@patch("ugbio_omics.get_omics_log.boto3")
def test_fetch_save_log_with_events(mock_boto3, tmpdir):
    mock_boto3.client.return_value.get_log_events.side_effect = [
        {"events": [{"message": "log message 1"}, {"message": "log message 2"}], "nextForwardToken": None},
        {"events": [{"message": "log message 3"}], "nextForwardToken": None},
    ]

    fetch_save_log(log_stream_name="log_stream", output="output.log", output_path=tmpdir)

    mock_boto3.client.assert_called_with("logs")
    mock_boto3.client.return_value.get_log_events.assert_called_once_with(
        logGroupName=OMICS_LOG_GROUP, logStreamName="log_stream", startFromHead=True
    )
    output_file = tmpdir / "output.log"
    assert output_file.exists()
    assert len(tmpdir.listdir()) == 1  # only one log file should be created
    with open(output_file) as f:
        content = f.read()
        assert "log message 1" in content
        assert "log message 2" in content


@patch("ugbio_omics.get_omics_log.boto3")
def test_fetch_save_log_with_events_and_pagination(mock_boto3, tmpdir):
    mock_boto3.client.return_value.get_log_events.side_effect = [
        {"events": [{"message": "log message 1"}, {"message": "log message 2"}], "nextForwardToken": "token"},
        {"events": [{"message": "log message 3"}], "nextForwardToken": None},
    ]

    fetch_save_log(log_stream_name="log_stream", output="output.log", output_path=tmpdir)

    mock_boto3.client.assert_called_with("logs")
    mock_boto3.client.return_value.get_log_events.assert_any_call(
        logGroupName=OMICS_LOG_GROUP, logStreamName="log_stream", nextToken="token", startFromHead=True
    )
    mock_boto3.client.return_value.get_log_events.assert_any_call(
        logGroupName=OMICS_LOG_GROUP, logStreamName="log_stream", startFromHead=True
    )
    output_file = tmpdir / "output.log"
    assert output_file.exists()
    assert len(tmpdir.listdir()) == 1  # only one log file should be created
    with open(output_file) as f:
        content = f.read()
        assert "log message 1" in content
        assert "log message 2" in content
        assert "log message 3" in content


@patch("ugbio_omics.get_omics_log.boto3")
def test_fetch_save_log_no_events(mock_boto3, tmpdir):
    mock_boto3.client.return_value.get_log_events.return_value = {"events": [], "nextForwardToken": None}

    fetch_save_log(log_stream_name="log_stream", output="output.log", output_path=tmpdir)

    mock_boto3.client.assert_called_with("logs")
    mock_boto3.client.return_value.get_log_events.assert_called_once_with(
        logGroupName=OMICS_LOG_GROUP, logStreamName="log_stream", startFromHead=True
    )
    assert not tmpdir.listdir()
