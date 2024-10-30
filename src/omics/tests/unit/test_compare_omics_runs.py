from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from ugbio_omics.compare_omics_runs import compare_omics_runs, single_run


@pytest.fixture
def mock_omics_session():
    return MagicMock()


@pytest.fixture
def mock_output_path(tmp_path):
    return tmp_path


@pytest.fixture
def mock_omics_run_id():
    return "test_run_id"


@pytest.fixture
def mock_run_ids():
    return ["run_id_1", "run_id_2"]


@patch("ugbio_omics.compare_omics_runs.get_omics_performance_cost")
@patch("ugbio_omics.compare_omics_runs.get_omics_total_duration")
@patch("ugbio_omics.compare_omics_runs.extract_omics_resources")
@patch("ugbio_omics.compare_cromwell_omics.RunCost")
def test_single_run(
    mock_run_cost,
    mock_extract_omics_resources,
    mock_get_omics_total_duration,
    mock_get_omics_performance_cost,
    mock_omics_session,
    mock_output_path,
):
    mock_run_cost.return_value = MagicMock()
    mock_get_omics_performance_cost.return_value = (
        pd.DataFrame({"task": ["task1", "task2", "total"], "cost_SUM": [10, 20, 35], "run_time (hours)": [1, 3, None]}),
        5,
        mock_run_cost,
    )
    mock_get_omics_total_duration.return_value = 4
    mock_extract_omics_resources.return_value = pd.DataFrame(
        {
            "task": ["task1", "task2"],
            "omics_resources": ["{'cpus': 1, 'memory_gib': 14, 'gpus': 0}", "{'cpus': 2, 'memory_gib': 36, 'gpus': 0}"],
            "omics_instance": ["omics.c.large", "omics.c.xlarge"],
        }
    )

    run_id = "run_id_1"
    result_df = single_run(run_id, mock_omics_session, mock_output_path)

    assert not result_df.empty
    assert "task" in result_df.columns
    assert f"{run_id}_cost" in result_df.columns
    assert f"{run_id}_duration" in result_df.columns
    assert f"{run_id}_resources" in result_df.columns
    assert f"{run_id}_instance" in result_df.columns


@patch("ugbio_omics.compare_omics_runs.single_run")
def test_compare_omics_runs(mock_single_run, mock_omics_session, mock_output_path, mock_run_ids):
    mock_single_run.side_effect = [
        pd.DataFrame(
            {
                "task": ["task1", "total"],
                "run_id_1_cost": [10, 20],
                "run_id_1_duration": [1, 2],
                "run_id_1_resources": ["res1", "res2"],
                "run_id_1_instance": ["inst1", "inst2"],
            }
        ),
        pd.DataFrame(
            {
                "task": ["task1", "total"],
                "run_id_2_cost": [15, 25],
                "run_id_2_duration": [1.5, 2.5],
                "run_id_2_resources": ["res3", "res4"],
                "run_id_2_instance": ["inst3", "inst4"],
            }
        ),
    ]

    compare_omics_runs(mock_run_ids, mock_omics_session, mock_output_path)

    compare_file = f"{mock_output_path}/compare_omics_runs.csv"
    result_df = pd.read_csv(compare_file)

    assert not result_df.empty
    assert "task" in result_df.columns
    assert "run_id_1_cost" in result_df.columns
    assert "run_id_2_cost" in result_df.columns
    assert "run_id_1_duration" in result_df.columns
    assert "run_id_2_duration" in result_df.columns
    assert "run_id_1_resources" in result_df.columns
    assert "run_id_2_resources" in result_df.columns
    assert "run_id_1_instance" in result_df.columns
    assert "run_id_2_instance" in result_df.columns
