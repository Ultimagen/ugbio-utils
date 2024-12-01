from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest
from ugbio_omics.parse_manifest_log import parse_manifest_log


@pytest.fixture
def resources_dir():
    inputs_dir = Path(__file__).parent.parent / "resources"
    return inputs_dir


@pytest.fixture
def output_path(tmpdir):
    return Path(tmpdir)


@pytest.fixture
def mock_boto3_client():
    with patch("ugbio_omics.parse_manifest_log.boto3.client") as mock_client:
        yield mock_client


@pytest.fixture
def mock_fetch_save_log():
    with patch("ugbio_omics.parse_manifest_log.fetch_save_log") as mock_fetch:
        yield mock_fetch


def test_parse_manifest_log_no_log_stream(mock_boto3_client):
    mock_client_instance = mock_boto3_client.return_value
    mock_client_instance.describe_log_streams.return_value = {"logStreams": []}

    with patch("builtins.print") as mock_print:
        parse_manifest_log("test_run_id")
        mock_print.assert_called_with("No manifest log stream found for run id 'test_run_id'")


def test_parse_manifest_log_success(mock_boto3_client, mock_fetch_save_log, resources_dir, tmpdir):
    mock_client_instance = mock_boto3_client.return_value
    mock_client_instance.describe_log_streams.return_value = {"logStreams": [{"logStreamName": "test_log_stream"}]}
    mock_fetch_save_log.return_value = resources_dir / "manifest.json"

    parse_manifest_log("test_run_id", output_path=tmpdir)

    # Check that the general run info file was created and has the expected content
    assert (tmpdir / "omics_test_run_id_general_run_info.json").exists()

    with open(tmpdir / "omics_test_run_id_general_run_info.json") as output_file:
        output_data = output_file.read()
    with open(resources_dir / "expected_manifest_general_run_info.json") as expected_file:
        expected_data = expected_file.read()

    assert output_data == expected_data, "The output general run info does not match the expected data"

    # Check that the task manifest csv file was created and has the expected content
    assert (tmpdir / "omics_test_run_id_task_manifests.csv").exists()

    output_df = pd.read_csv(tmpdir / "omics_test_run_id_task_manifests.csv")
    expected_df = pd.read_csv(resources_dir / "expected_manifest_task_manifests.csv")

    pd.testing.assert_frame_equal(output_df, expected_df, check_like=True)
