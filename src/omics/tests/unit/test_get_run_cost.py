from unittest.mock import patch

import pandas as pd
import pytest
from ugbio_omics.get_run_cost import Columns, RunCost


@pytest.fixture
def run_cost():
    return RunCost(run_id="test_run")


def test_calculate_run_cost(run_cost):
    with (
        patch("ugbio_omics.get_run_cost.RunCost.run_analyzer") as mock_run_analyzer,
        patch("pandas.read_csv") as mock_read_csv,
    ):
        mock_read_csv.return_value = pd.DataFrame(
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

        run_cost.calculate_run_cost()

        mock_run_analyzer.assert_called_once_with("test_run", "omics_test_run.cost.csv", "omics_test_run_plots")
        assert run_cost.cost_csv == "omics_test_run.cost.csv"
        assert run_cost.cost_df is not None


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
