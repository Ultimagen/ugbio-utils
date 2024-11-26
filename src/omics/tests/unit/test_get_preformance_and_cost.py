import json
from datetime import timedelta
from pathlib import Path
from unittest.mock import Mock, patch

import pandas as pd
import pytest
from ugbio_omics.get_preformance import (
    MonitorLog,
    RunCost,
    performance,
    process_monitor_log,
)


@pytest.fixture
def resources_dir():
    inputs_dir = Path(__file__).parent.parent / "resources"
    return inputs_dir


@pytest.fixture
def mock_boto3_client():
    with patch("boto3.client") as mock_client:
        yield mock_client


@pytest.fixture
def mock_log_events(resources_dir):
    with open(resources_dir / "monitor_log.json") as f:
        return json.load(f)


@pytest.fixture
def mock_log_events_with_next_token(resources_dir):
    with open(resources_dir / "monitor_log_next_token.json") as f:
        return json.load(f)


@pytest.fixture
def mock_run_info():
    return {
        "tasks": [
            {"taskId": "test_task_id_1", "name": "test_task_name_1"},
            {"taskId": "test_task_id_2", "name": "test_task_name_2"},
        ]
    }


@pytest.fixture
def mock_run_cost():
    mock_run_cost_instance = Mock(spec=RunCost)
    mock_run_cost_instance.get_tasks_cost.return_value = pd.DataFrame(
        {"task": ["test_task_name_1", "test_task_name_2"], "cost": [10.0, 20.0], "instance": ["m5.large", "m5.xlarge"]}
    )
    mock_run_cost_instance.get_storage_cost.return_value = 5.0
    return mock_run_cost_instance


def test_process_monitor_log(mock_boto3_client, mock_log_events_with_next_token, mock_log_events):
    mock_client_instance = Mock()
    mock_boto3_client.return_value = mock_client_instance
    mock_client_instance.filter_log_events.side_effect = [mock_log_events_with_next_token, mock_log_events]

    run_id = "test_run_id"
    task = {"taskId": "test_task_id", "name": "test_task_name"}

    monitor_log = process_monitor_log(run_id, task, client=mock_client_instance)

    assert mock_client_instance.filter_log_events.call_count == 2
    assert isinstance(monitor_log, MonitorLog)
    assert monitor_log.task_name == "test_task_name"
    assert monitor_log.task_id == "test_task_id"
    assert not monitor_log.df.empty
    assert monitor_log.df.shape[0] == 36
    assert monitor_log.df["CPU"].iloc[0] == 52.10
    assert monitor_log.df["Memory"].iloc[0] == 21.00
    assert monitor_log.df["IO_rKb/s"].iloc[0] == 41.00
    assert monitor_log.df["IO_wKb/s"].iloc[0] == 367.00
    assert monitor_log.df["IOWait"].iloc[0] == 0.00


@patch("ugbio_omics.get_preformance_and_cost.get_run_info")
@patch("ugbio_omics.get_preformance_and_cost.RunCost", autospec=True)
def test_performance(
    mock_run_cost_class, mock_get_run_info, mock_boto3_client, mock_run_info, mock_run_cost, mock_log_events, tmpdir
):
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
    mock_logs_client.filter_log_events.return_value = mock_log_events
    mock_get_run_info.return_value = mock_run_info
    mock_run_cost_class.return_value = mock_run_cost

    run_id = "test_run_id"
    output_dir = tmpdir
    output_prefix = "test_prefix_"

    total_performance_df, run_cost = performance(run_id, output_dir=output_dir, output_prefix=output_prefix)

    assert isinstance(total_performance_df, pd.DataFrame)
    assert isinstance(run_cost, RunCost)
    assert total_performance_df.shape[0] == 2
    assert total_performance_df["task"].iloc[0] == "test_task_name_1"
    assert total_performance_df["cost"].iloc[0] == 10.0
    assert total_performance_df["instance"].iloc[0] == "m5.large"
    assert total_performance_df["total_storage_cost"].iloc[0] == 5.0


def test_monitor_log_initialization():
    monitor_log = MonitorLog()
    assert monitor_log.df_columns == ["time", "CPU", "Memory", "IO_rKb/s", "IO_wKb/s", "IOWait"]
    assert monitor_log.df.empty
    assert monitor_log.start_time is None
    assert monitor_log.run_time == timedelta(0)
    assert monitor_log.total_cpu is None
    assert monitor_log.total_memory is None
    assert monitor_log.task_name is None
    assert monitor_log.task_id is None


def test_monitor_log_process_line_general_info():
    monitor_log = MonitorLog()
    line = "MONITORING, [Tue Feb 13 15:33:19 IST 2024], General Information, CPU: 20, Memory(GiB): 7"
    monitor_log.process_line(line)
    assert monitor_log.total_cpu == 20
    assert monitor_log.total_memory == 7.0
    assert monitor_log.start_time is not None


def test_monitor_log_process_line_monitoring_info():
    monitor_log = MonitorLog()
    line = "MONITORING, [Sun Mar 24 21:35:30 UTC 2024], %CPU: 52.10, %Memory: 21.00, IO_rKb/s: 41.00, IO_wKb/s: 367.00, %IOWait: 0.00"  # noqa: E501
    monitor_log.process_line(line)
    assert not monitor_log.df.empty
    assert monitor_log.df.shape[0] == 1
    assert monitor_log.df["CPU"].iloc[0] == 52.10
    assert monitor_log.df["Memory"].iloc[0] == 21.00
    assert monitor_log.df["IO_rKb/s"].iloc[0] == 41.00
    assert monitor_log.df["IO_wKb/s"].iloc[0] == 367.00
    assert monitor_log.df["IOWait"].iloc[0] == 0.00


def test_monitor_log_process_line_invalid_data():
    monitor_log = MonitorLog()
    line = "MONITORING, [Sun Mar 24 21:35:30 UTC 2024], %CPU: invalid, %Memory: invalid, IO_rKb/s: invalid, IO_wKb/s: invalid, %IOWait: invalid"  # noqa: E501
    monitor_log.process_line(line)
    assert not monitor_log.df.empty
    assert monitor_log.df.shape[0] == 1
    assert monitor_log.df["CPU"].iloc[0] == 0.0
    assert monitor_log.df["Memory"].iloc[0] == 0.0
    assert monitor_log.df["IO_rKb/s"].iloc[0] == 0.0
    assert monitor_log.df["IO_wKb/s"].iloc[0] == 0.0
    assert monitor_log.df["IOWait"].iloc[0] == 0.0


def test_monitor_log_process_line_no_monitoring():
    monitor_log = MonitorLog()
    line = "This line does not contain monitoring data"
    monitor_log.process_line(line)
    assert monitor_log.df.empty
