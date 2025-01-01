from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from ugbio_omics.compare_cromwell_omics import (
    compare_cromwell_omics,
    copy_cromwell_data,
    cromwell_cost,
    cromwell_performance,
    get_cromwell_total_duration,
    get_omics_cost_perfromance,
)


@pytest.fixture
def resources_dir():
    inputs_dir = Path(__file__).parent.parent / "resources"
    return inputs_dir


@pytest.fixture
def mock_session():
    return MagicMock()


@pytest.fixture
def mock_run_cost():
    with patch("ugbio_omics.compare_cromwell_omics.RunCost") as mock_run_cost:
        mock_run_cost.return_value.get_run_cost.return_value = pd.DataFrame(
            {
                "task": ["task1", "task2", "storage", "total"],
                "cost": [10, 5, 5, 20],
                "run_time (hours)": [1, 3, None, None],
                "instance": ["omics.c.large", "omics.c.xlarge", None, None],
            }
        )

        mock_run_cost.return_value.get_tasks_resources.return_value = pd.DataFrame(
            {
                "name": ["task1", "task2"],
                "cpusRequested": [2, 6],
                "memoryRequestedGiB": [4, 8],
                "gpusRequested": [1, 0],
                "omicsInstanceTypeReserved": ["instance1", "instance2"],
            }
        )
        yield mock_run_cost


def test_compare_cromwell_omics(mock_run_cost, resources_dir, mock_session, tmpdir):
    cromwell_wid = "test_cromwell_wid"
    omics_run_id = "test_omics_run_id"
    workflow_name = "test_workflow"
    overwrite = False

    with (
        patch("ugbio_omics.compare_cromwell_omics.storage.Client") as mock_storage_client,
        patch("ugbio_omics.cromwell_calculate_cost.urlopen") as mock_urlopen,
    ):
        # cromwell mocks
        mock_bucket = MagicMock()
        mock_blob = MagicMock()
        with (
            open(resources_dir / "cromwell.performance.csv") as performance_file,
            open(resources_dir / "cromwell.metadata.json") as metadata_file,
        ):
            mock_blob.download_as_text.side_effect = [performance_file.read(), metadata_file.read()]
        mock_bucket.get_blob.return_value = mock_blob
        mock_storage_client.return_value.get_bucket.return_value = mock_bucket

        with open(resources_dir / "cromwell_pricelist.json") as cromwell_pricelist_file:
            mock_urlopen.return_value.read.return_value = cromwell_pricelist_file.read().encode("utf-8")
            mock_urlopen.return_value.info.return_value.get.return_value = None  # No gzip encoding

        compare_cromwell_omics(cromwell_wid, omics_run_id, mock_session, workflow_name, tmpdir, overwrite=overwrite)

    output_file = tmpdir / f"compare_omics_{omics_run_id}_cromwell_{cromwell_wid}.csv"
    assert output_file.isfile()
    expected_df = pd.read_csv(resources_dir / "expected_compare_cromwell_omics.csv").drop(
        columns=["cromwell_resources"]
    )
    result_df = pd.read_csv(output_file).drop(columns=["cromwell_resources"])
    pd.testing.assert_frame_equal(result_df, expected_df, check_dtype=False)


@patch("ugbio_omics.compare_cromwell_omics.omics_performance")
def test_compare_cromwell_omics_with_performance(
    mock_omics_performance, mock_run_cost, resources_dir, mock_session, tmpdir
):
    cromwell_wid = "test_cromwell_wid"
    omics_run_id = "test_omics_run_id"
    workflow_name = "test_workflow"
    overwrite = False
    get_performance = True

    with (
        patch("ugbio_omics.compare_cromwell_omics.storage.Client") as mock_storage_client,
        patch("ugbio_omics.cromwell_calculate_cost.urlopen") as mock_urlopen,
    ):
        # cromwell mocks
        mock_bucket = MagicMock()
        mock_blob = MagicMock()
        with (
            open(resources_dir / "cromwell.performance.csv") as performance_file,
            open(resources_dir / "cromwell.metadata.json") as metadata_file,
        ):
            mock_blob.download_as_text.side_effect = [performance_file.read(), metadata_file.read()]
        mock_bucket.get_blob.return_value = mock_blob
        mock_storage_client.return_value.get_bucket.return_value = mock_bucket

        with open(resources_dir / "cromwell_pricelist.json") as cromwell_pricelist_file:
            mock_urlopen.return_value.read.return_value = cromwell_pricelist_file.read().encode("utf-8")
            mock_urlopen.return_value.info.return_value.get.return_value = None  # No gzip encoding

        compare_cromwell_omics(
            cromwell_wid,
            omics_run_id,
            mock_session,
            workflow_name,
            tmpdir,
            overwrite=overwrite,
            get_performance=get_performance,
        )

    output_file = tmpdir / f"compare_omics_{omics_run_id}_cromwell_{cromwell_wid}.csv"
    assert output_file.isfile()
    expected_df = pd.read_csv(resources_dir / "expected_compare_cromwell_omics.csv").drop(
        columns=["cromwell_resources"]
    )
    result_df = pd.read_csv(output_file).drop(columns=["cromwell_resources"])
    pd.testing.assert_frame_equal(result_df, expected_df, check_dtype=False)
    mock_omics_performance.assert_called_once_with(omics_run_id, session=mock_session, output_dir=tmpdir)


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

        cromwell_cost_df = cromwell_cost(metadata_file, tmpdir, workflow_name, cromwell_wid)

        mock_calculate_cost.assert_called_once()
        mock_read_csv.assert_called_once()
        # expected_df = pd.read_csv(resources_dir / "expected_cromwell_cost_df.csv")
        # pd.testing.assert_frame_equal(cromwell_cost_df, expected_df)
        assert cromwell_cost_df["compute_cost"].iloc[0] == 17.418350291666663
        assert cromwell_cost_df.shape == (5, 30)
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
        assert performance_df.shape == (3, 10)
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
def test_get_omics_cost_perfromance(mock_omics_performance, mock_session, mock_run_cost, tmpdir, resources_dir):
    omics_run_id = "test_omics_run_id"

    cost_df, run_cost = get_omics_cost_perfromance(omics_run_id, mock_session, tmpdir, get_performance=True)

    mock_omics_performance.assert_called_once_with(omics_run_id, session=mock_session, output_dir=tmpdir)
    expected_df = pd.read_csv(resources_dir / "expected_omics_cost_df.csv")
    pd.testing.assert_frame_equal(cost_df, expected_df)
