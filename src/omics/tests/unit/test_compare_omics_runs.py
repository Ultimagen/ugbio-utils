from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from ugbio_omics.compare_omics_runs import compare_omics_runs, single_run


@pytest.fixture
def resources_dir():
    inputs_dir = Path(__file__).parent.parent / "resources"
    return inputs_dir


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


@patch("ugbio_omics.compare_omics_runs.get_omics_cost_perfromance")
@patch("ugbio_omics.compare_omics_runs.extract_omics_resources")
@patch("ugbio_omics.compare_cromwell_omics.RunCost")
def test_single_run(
    mock_run_cost,
    mock_extract_omics_resources,
    mock_get_omics_cost_perfromance,
    mock_omics_session,
    mock_output_path,
    resources_dir,
):
    mock_run_cost.return_value = MagicMock()
    mock_get_omics_cost_perfromance.return_value = (
        pd.DataFrame(
            {
                "task": ["task1", "task2", "total"],
                "cost": [10, 20, 35],
                "run_time (hours)": [1, 3, None],
                "instance": ["omics.c.large", "omics.c.xlarge", None],
            }
        ),
        mock_run_cost,
    )
    mock_extract_omics_resources.return_value = pd.DataFrame(
        {
            "task": ["task1", "task2"],
            "omics_resources": ["{'cpus': 1, 'memory_gib': 14, 'gpus': 0}", "{'cpus': 2, 'memory_gib': 36, 'gpus': 0}"],
            "omics_instance": ["omics.c.large", "omics.c.xlarge"],
        }
    )

    run_id = "run_id_1"
    result_df = single_run(run_id, mock_omics_session, mock_output_path)

    expected_df = pd.read_csv(resources_dir / "expected_compare_omics_single_run.csv")
    pd.testing.assert_frame_equal(result_df, expected_df)


@patch("ugbio_omics.compare_omics_runs.single_run")
def test_compare_omics_runs(mock_single_run, mock_omics_session, mock_output_path, mock_run_ids, resources_dir):
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

    output_file = Path(f"{mock_output_path}/compare_omics_runs.csv")
    assert output_file.is_file()
    expected_df = pd.read_csv(resources_dir / "expected_compare_omics_runs.csv")
    result_df = pd.read_csv(output_file)
    pd.testing.assert_frame_equal(result_df, expected_df)
