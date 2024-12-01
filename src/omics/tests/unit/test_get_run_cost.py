from unittest import mock
from unittest.mock import Mock, patch

import pandas as pd
import pytest
from ugbio_omics.get_run_cost import Columns, RunCost


@pytest.fixture
def run_cost(tmpdir):
    return RunCost(run_id="test_run", output_dir=tmpdir)


@pytest.fixture
def mock_get_run_info():
    with patch("ugbio_omics.get_run_cost.get_run_info") as mock_get_run_info:
        mock_get_run_info.return_value = {
            "outputUri": "s3://test-bucket/test-pipeline",
            "tasks": [
                {"taskId": "taskID1", "name": "taskName1", "status": "SUCCEEDED"},
                {"taskId": "taskID2", "name": "taskName2", "status": "SUCCEEDED"},
            ],
            "status": "SUCCEEDED",
        }
        yield mock_get_run_info


def test_calculate_run_cost(run_cost, mock_get_run_info):
    with patch("boto3.client") as mock_boto3_client, patch("pandas.read_csv") as mock_read_csv:
        # Create separate mock clients for "s3" and "omics"
        mock_s3_client = Mock()
        mock_omics_client = Mock()

        # Configure the side_effect to return the appropriate mock client based on the service name
        def client_side_effect(service_name, *args, **kwargs):
            if service_name == "s3":
                return mock_s3_client
            elif service_name == "omics":
                return mock_omics_client
            else:
                raise ValueError(f"Unexpected service name: {service_name}")

        mock_boto3_client.side_effect = client_side_effect

        mock_read_csv.return_value = pd.DataFrame(
            {
                Columns.ESTIMATED_USD_COLUMN.value: [10.0, 20.0],
                Columns.TYPE_COLUMN.value: ["run", "task"],
                Columns.NAME_COLUMN.value: ["task1", "task2"],
                Columns.OMICS_INSTANCE_TYPE_RESERVED.value: ["type1", "type2"],
                Columns.CPU_REQUESTED.value: [2, 4],
                Columns.MEMORY_REQUESTED_GIB.value: [8, 16],
                Columns.GPUS_REQUESTED.value: [1, 2],
                Columns.RUNNING_SECONDS.value: [100, 200],
            }
        )

        run_cost.calculate_run_cost()

        assert "omics_test_run.cost.csv" in run_cost.cost_csv
        assert run_cost.cost_df is not None
        mock_get_run_info.assert_called_once_with("test_run", client=mock_omics_client)
        mock_s3_client.download_fileobj.assert_any_call(
            "test-bucket", "test-pipeline/test_run/logs/run-test_run.csv", mock.ANY
        )
        mock_s3_client.download_fileobj.assert_any_call(
            "test-bucket", "test-pipeline/test_run/logs/plots/test_run_timeline.html", mock.ANY
        )


def test_get_total_cost(run_cost):
    with patch.object(run_cost, "calculate_run_cost") as mock_calculate_run_cost:
        run_cost.cost_df = pd.DataFrame({Columns.ESTIMATED_USD_COLUMN.value: [10.0, 20.0]})

        total_cost = run_cost.get_total_cost()

        assert total_cost == 30.0
        mock_calculate_run_cost.assert_not_called()


def test_get_total_cost_no_cost_df(run_cost):
    with patch.object(run_cost, "calculate_run_cost") as mock_calculate_run_cost:
        mock_calculate_run_cost.side_effect = lambda: setattr(
            run_cost, "cost_df", pd.DataFrame({Columns.ESTIMATED_USD_COLUMN.value: [10.0, 20.0]})
        )

        total_cost = run_cost.get_total_cost()

        assert total_cost == 30.0
        mock_calculate_run_cost.assert_called_once()


def test_get_storage_cost(run_cost):
    with patch.object(run_cost, "calculate_run_cost") as mock_calculate_run_cost:
        run_cost.cost_df = pd.DataFrame(
            {Columns.ESTIMATED_USD_COLUMN.value: [10.0, 20.0], Columns.TYPE_COLUMN.value: ["run", "task"]}
        )

        storage_cost = run_cost.get_storage_cost()

        assert storage_cost == 10.0
        mock_calculate_run_cost.assert_not_called()


def test_get_storage_cost_no_cost_df(run_cost):
    with patch.object(run_cost, "calculate_run_cost") as mock_calculate_run_cost:
        mock_calculate_run_cost.side_effect = lambda: setattr(
            run_cost,
            "cost_df",
            pd.DataFrame(
                {Columns.ESTIMATED_USD_COLUMN.value: [10.0, 20.0], Columns.TYPE_COLUMN.value: ["run", "task"]}
            ),
        )

        storage_cost = run_cost.get_storage_cost()

        assert storage_cost == 10.0
        mock_calculate_run_cost.assert_called_once()


def test_get_tasks_cost(run_cost):
    with patch.object(run_cost, "calculate_run_cost") as mock_calculate_run_cost:
        run_cost.cost_df = pd.DataFrame(
            {
                Columns.ESTIMATED_USD_COLUMN.value: [10.0, 20.0],
                Columns.TYPE_COLUMN.value: ["run", "task"],
                Columns.NAME_COLUMN.value: ["task1", "task2"],
                Columns.OMICS_INSTANCE_TYPE_RESERVED.value: ["type1", "type2"],
                Columns.RUNNING_SECONDS.value: [100, 200],
            }
        )

        tasks_cost = run_cost.get_tasks_cost()

        assert not tasks_cost.empty
        assert len(tasks_cost) == 1
        assert tasks_cost.iloc[0][Columns.NAME_COLUMN.value] == "task2"
        mock_calculate_run_cost.assert_not_called()


def test_get_tasks_cost_no_cost_df(run_cost):
    with patch.object(run_cost, "calculate_run_cost") as mock_calculate_run_cost:
        mock_calculate_run_cost.side_effect = lambda: setattr(
            run_cost,
            "cost_df",
            pd.DataFrame(
                {
                    Columns.ESTIMATED_USD_COLUMN.value: [10.0, 20.0],
                    Columns.TYPE_COLUMN.value: ["run", "task"],
                    Columns.NAME_COLUMN.value: ["task1", "task2"],
                    Columns.OMICS_INSTANCE_TYPE_RESERVED.value: ["type1", "type2"],
                    Columns.RUNNING_SECONDS.value: [100, 200],
                }
            ),
        )

        tasks_cost = run_cost.get_tasks_cost()

        assert not tasks_cost.empty
        assert len(tasks_cost) == 1
        assert tasks_cost.iloc[0][Columns.NAME_COLUMN.value] == "task2"
        mock_calculate_run_cost.assert_called_once()


def test_get_tasks_resources(run_cost):
    with patch.object(run_cost, "calculate_run_cost") as mock_calculate_run_cost:
        run_cost.cost_df = pd.DataFrame(
            {
                Columns.ESTIMATED_USD_COLUMN.value: [10.0, 20.0],
                Columns.TYPE_COLUMN.value: ["run", "task"],
                Columns.NAME_COLUMN.value: ["task1", "task2"],
                Columns.OMICS_INSTANCE_TYPE_RESERVED.value: ["type1", "type2"],
                Columns.CPU_REQUESTED.value: [2, 4],
                Columns.MEMORY_REQUESTED_GIB.value: [8, 16],
                Columns.GPUS_REQUESTED.value: [1, 2],
            }
        )

        tasks_resources = run_cost.get_tasks_resources()

        assert not tasks_resources.empty
        assert len(tasks_resources) == 1
        assert tasks_resources.iloc[0][Columns.NAME_COLUMN.value] == "task2"
        mock_calculate_run_cost.assert_not_called()


def test_get_tasks_resources_no_cost_df(run_cost):
    with patch.object(run_cost, "calculate_run_cost") as mock_calculate_run_cost:
        mock_calculate_run_cost.side_effect = lambda: setattr(
            run_cost,
            "cost_df",
            pd.DataFrame(
                {
                    Columns.ESTIMATED_USD_COLUMN.value: [10.0, 20.0],
                    Columns.TYPE_COLUMN.value: ["run", "task"],
                    Columns.NAME_COLUMN.value: ["task1", "task2"],
                    Columns.OMICS_INSTANCE_TYPE_RESERVED.value: ["type1", "type2"],
                    Columns.CPU_REQUESTED.value: [2, 4],
                    Columns.MEMORY_REQUESTED_GIB.value: [8, 16],
                    Columns.GPUS_REQUESTED.value: [1, 2],
                }
            ),
        )

        tasks_resources = run_cost.get_tasks_resources()

        assert not tasks_resources.empty
        assert len(tasks_resources) == 1
        assert tasks_resources.iloc[0][Columns.NAME_COLUMN.value] == "task2"
        mock_calculate_run_cost.assert_called_once()
