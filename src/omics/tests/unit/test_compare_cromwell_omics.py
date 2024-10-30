from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from ugbio_omics.compare_cromwell_omics import (
    compare_cromwell_omics,
    copy_cromwell_data,
    cromwell_cost,
    cromwell_performance,
    extract_omics_resources,
    get_cromwell_total_duration,
    get_omics_performance_cost,
    get_omics_total_duration,
)
from ugbio_omics.get_run_cost import RunCost


@pytest.fixture
def resources_dir():
    inputs_dir = Path(__file__).parent.parent / "resources"
    return inputs_dir


@pytest.fixture
def mock_session():
    return MagicMock()


@pytest.fixture
def mock_run_cost():
    return MagicMock(spec=RunCost)


# TODO: go over this test
def test_compare_cromwell_omics(mock_session, tmpdir):
    cromwell_wid = "test_cromwell_wid"
    omics_run_id = "test_omics_run_id"
    workflow_name = "test_workflow"
    overwrite = False

    with (
        patch("ugbio_omics.compare_cromwell_omics.copy_cromwell_data") as mock_copy_cromwell_data,
        patch("ugbio_omics.compare_cromwell_omics.cromwell_cost") as mock_cromwell_cost,
        patch("ugbio_omics.compare_cromwell_omics.cromwell_performance") as mock_cromwell_performance,
        patch("ugbio_omics.compare_cromwell_omics.get_omics_performance_cost") as mock_get_omics_performance_cost,
        patch("ugbio_omics.compare_cromwell_omics.get_cromwell_total_duration") as mock_get_cromwell_total_duration,
        patch("ugbio_omics.compare_cromwell_omics.get_omics_total_duration") as mock_get_omics_total_duration,
        patch("ugbio_omics.compare_cromwell_omics.extract_omics_resources") as mock_extract_omics_resources,
        patch("pandas.DataFrame.to_csv") as mock_to_csv,
    ):
        mock_copy_cromwell_data.return_value = ("performance_file.csv", "metadata_file.json")
        mock_cromwell_cost.return_value = (pd.DataFrame({"task": ["task1"], "compute_cost": [10]}), 5)
        mock_cromwell_performance.return_value = pd.DataFrame({"task": ["task1"], "run_time (hours)": [1]})
        mock_get_omics_performance_cost.return_value = (
            pd.DataFrame({"task": ["task1"], "cost_SUM": [15]}),
            3,
            mock_run_cost,
        )
        mock_get_cromwell_total_duration.return_value = 2
        mock_get_omics_total_duration.return_value = 3
        mock_extract_omics_resources.return_value = pd.DataFrame(
            {"task": ["task1"], "omics_resources": [{"cpus": 2, "mem_gbs": 4}]}
        )

        compare_cromwell_omics(cromwell_wid, omics_run_id, mock_session, workflow_name, tmpdir, overwrite=overwrite)

        mock_copy_cromwell_data.assert_called_once_with(workflow_name, cromwell_wid, tmpdir, overwrite=overwrite)
        mock_cromwell_cost.assert_called_once_with("metadata_file.json", tmpdir, workflow_name, cromwell_wid)
        mock_cromwell_performance.assert_called_once_with("performance_file.csv")
        mock_get_omics_performance_cost.assert_called_once_with(omics_run_id, mock_session, tmpdir)
        mock_get_cromwell_total_duration.assert_called_once_with("metadata_file.json")
        mock_get_omics_total_duration.assert_called_once_with(omics_run_id, mock_session)
        mock_extract_omics_resources.assert_called_once_with(mock_run_cost)
        mock_to_csv.assert_called_once()


@patch("ugbio_omics.compare_cromwell_omics.storage.Client")
def test_copy_cromwell_data(mock_storage_client, tmpdir, resources_dir):
    workflow_name = "test_workflow"
    cromwell_wid = "test_cromwell_wid"

    with patch("builtins.open", new_callable=MagicMock) as mock_open:
        mock_bucket = MagicMock()
        mock_blob = MagicMock()
        with (
            open(resources_dir / "cromwell.performance.csv") as performance_file,
            open(resources_dir / "cromwell.metadata.json") as metadata_file,
        ):
            mock_blob.download_as_text.side_effect = [performance_file.read(), metadata_file.read()]
        mock_bucket.get_blob.return_value = mock_blob
        mock_storage_client.return_value.get_bucket.return_value = mock_bucket

        performance_file, metadata_file = copy_cromwell_data(workflow_name, cromwell_wid, tmpdir)

        assert performance_file == f"{tmpdir}/cromwell_{cromwell_wid}.performance.csv"
        assert metadata_file == f"{tmpdir}/cromwell_{cromwell_wid}.metadata.json"
        mock_storage_client.return_value.get_bucket.assert_called_once_with("cromwell-backend-ultima-data-307918")
        mock_bucket.get_blob.assert_any_call(f"cromwell-execution/{workflow_name}/{cromwell_wid}/performance.csv")
        mock_bucket.get_blob.assert_any_call(f"cromwell-execution/{workflow_name}/{cromwell_wid}/metadata.json")
        mock_open.assert_any_call(performance_file, "w")
        mock_open.assert_any_call(metadata_file, "w")


@patch("ugbio_omics.compare_cromwell_omics.calculate_cost")
def test_cromwell_cost(mock_calculate_cost, tmpdir, resources_dir):
    metadata_file = resources_dir / "cromwell.metadata.json"
    workflow_name = "test_workflow"
    cromwell_wid = "test_cromwell_wid"

    with open(resources_dir / "cromwell.cost.csv") as cost_file:
        mock_returned_cost_df = pd.read_csv(cost_file)
    with patch("pandas.read_csv") as mock_read_csv:
        mock_read_csv.return_value = mock_returned_cost_df

        cromwell_cost_df, cromwell_disk_cost = cromwell_cost(metadata_file, tmpdir, workflow_name, cromwell_wid)

        mock_calculate_cost.assert_called_once()
        mock_read_csv.assert_called_once()
        assert cromwell_cost_df["compute_cost"].iloc[0] == 0.003142125
        assert cromwell_disk_cost == 3.7991075799086755
        assert cromwell_cost_df.shape == (9, 30)
        assert "compute_cost" in cromwell_cost_df.columns
        assert "task" in cromwell_cost_df.columns
        assert "total" in cromwell_cost_df["task"].to_numpy()
        total_row = cromwell_cost_df[cromwell_cost_df["task"] == "total"]
        assert not pd.isna(total_row["compute_cost"].to_numpy()[0])
        for col in cromwell_cost_df.columns:
            if col != "task" and col != "compute_cost":
                assert pd.isna(total_row[col].to_numpy()[0])


@patch("ugbio_omics.compare_cromwell_omics.calculate_cost")
def test_cromwell_cost_exception(mock_calculate_cost, tmpdir, resources_dir):
    metadata_file = resources_dir / "cromwell.metadata.json"
    workflow_name = "test_workflow"
    cromwell_wid = "test_cromwell_wid"

    mock_calculate_cost.side_effect = ValueError("Error")

    with pytest.raises(ValueError):
        cromwell_cost(metadata_file, tmpdir, workflow_name, cromwell_wid)


def test_cromwell_performance(resources_dir):
    performance_file = resources_dir / "cromwell.performance.csv"

    with open(performance_file) as file:
        mock_returned_df = pd.read_csv(file)
    with patch("pandas.read_csv") as mock_read_csv:
        mock_read_csv.return_value = mock_returned_df

        performance_df = cromwell_performance(performance_file)

        mock_read_csv.assert_called_once_with(performance_file)
        assert performance_df.shape == (8, 10)
        assert all("attempt" not in task for task in performance_df["task"].to_numpy())
        assert all("call-" not in task for task in performance_df["task"].to_numpy())
        for col in performance_df.columns:
            if col != "task":
                assert (
                    pd.to_numeric(performance_df[col].dropna(), errors="coerce").notna().all()
                ), f"Column {col} contains non-numeric values"


def test_get_cromwell_total_duration(resources_dir):
    metadata_file = resources_dir / "cromwell.metadata.json"

    total_duration = get_cromwell_total_duration(metadata_file)

    assert total_duration == 19.405835


@patch("ugbio_omics.compare_cromwell_omics.omics_performance")
def test_get_omics_performance_cost(mock_omics_performance, mock_session, mock_run_cost, tmpdir):
    omics_run_id = "test_omics_run_id"
    mock_value_performance_df = pd.DataFrame(
        {
            "task": ["task1", "task2"],
            "total_CPU": [1, 2],
            "mean_%_CPU": [10, 20],
            "max_%_CPU": [20, 30],
            "total_Memory(GB)": [2, 4],
            "mean_%_Memory": [10, 20],
            "max_%_Memory": [20, 30],
            "run_time (hours)": [1, 2],
            "count_entries": [10, 12],
            "cost": [5, 10],
            "instance": ["m5.large", "m5.xlarge"],
            "total_storage_cost": [5, 5],
        }
    )

    mock_omics_performance.return_value = (mock_value_performance_df, mock_run_cost)
    mock_run_cost.get_storage_cost.return_value = 5
    mock_run_cost.get_total_cost.return_value = 20

    performance_df, omics_disk_cost, run_cost = get_omics_performance_cost(omics_run_id, mock_session, tmpdir)

    mock_omics_performance.assert_called_once_with(omics_run_id, session=mock_session, output_dir=tmpdir)
    assert omics_disk_cost == 5
    assert performance_df.shape == (3, 11)
    assert "total" in performance_df["task"].to_numpy()
    assert performance_df[performance_df["task"] == "total"]["cost_SUM"].iloc[0] == 20
    assert run_cost == mock_run_cost
    for col in performance_df.columns:
        if col not in ["task", "instance"]:
            assert (
                pd.to_numeric(performance_df[col].dropna(), errors="coerce").notna().all()
            ), f"Column {col} contains non-numeric values"


def test_get_omics_performance_file_exists(mock_session, mock_output_path):
    pass  # TODO: add tests and all tests below


def test_get_omics_total_duration(mock_session):
    omics_run_id = "test_omics_run_id"

    with patch("ugbio_omics.compare_cromwell_omics.get_run_info") as mock_get_run_info:
        mock_get_run_info.return_value = {"duration": pd.Timedelta(hours=1)}

        total_duration = get_omics_total_duration(omics_run_id, mock_session)

        assert total_duration == 1


def test_extract_omics_resources(mock_run_cost):
    with patch.object(mock_run_cost, "get_tasks_resources") as mock_get_tasks_resources:
        mock_get_tasks_resources.return_value = pd.DataFrame(
            {
                "task_name": ["task1"],
                "cpu_requested": [2],
                "memory_requested_gib": [4],
                "gpus_requested": [1],
                "omics_instance": ["instance1"],
            }
        )

        resources_df = extract_omics_resources(mock_run_cost)

        assert resources_df["task"].iloc[0] == "task1"
        assert resources_df["omics_resources"].iloc[0] == {"cpus": 2, "mem_gbs": 4, "gpus": 1}
