from argparse import Namespace
from unittest.mock import MagicMock, patch

import pytest
from ugbio_omics.retrieve_from_glacier import main

# filepath: src/omics/ugbio_omics/test_retrieve_from_glacier.py


@pytest.fixture
def mock_cloud_validator():
    with patch("ugbio_omics.retrieve_from_glacier.cfv.CloudFilesValidator") as mock_validator:
        yield mock_validator


@pytest.fixture
def mock_boto3_client():
    with patch("ugbio_omics.retrieve_from_glacier.boto3.client") as mock_client:
        yield mock_client


@pytest.fixture
def mock_logger():
    with patch("ugbio_omics.retrieve_from_glacier.logging") as mock_logger:
        yield mock_logger


def test_all_files_valid(mock_cloud_validator, mock_logger):
    # Mock CloudFilesValidator behavior
    mock_instance = MagicMock()
    mock_instance.validate.return_value = True
    mock_instance.non_validated_files = []
    mock_instance.glacier_files = []
    mock_cloud_validator.return_value = mock_instance

    # Mock arguments
    with patch(
        "ugbio_omics.retrieve_from_glacier.parse_arguments",
        return_value=Namespace(wdl="test.wdl", param_json="test.json", retrieve=False, n_days=30),
    ):
        main()

    # Assertions
    mock_logger.info.assert_any_call("The WDL and JSON files are valid, no GLACIER  files found.")


def test_missing_files(mock_cloud_validator, mock_logger):
    # Mock CloudFilesValidator behavior
    mock_instance = MagicMock()
    mock_instance.validate.return_value = False
    mock_instance.non_validated_files = ["file1", "file2"]
    mock_instance.glacier_files = []
    mock_cloud_validator.return_value = mock_instance

    # Mock arguments
    with patch(
        "ugbio_omics.retrieve_from_glacier.parse_arguments",
        return_value=Namespace(wdl="test.wdl", param_json="test.json", retrieve=False, n_days=30),
    ):
        main()

    # Assertions
    mock_logger.info.assert_any_call("Missing files found, correct first")


def test_glacier_files_no_retrieve(mock_cloud_validator, mock_logger):
    # Mock CloudFilesValidator behavior
    mock_instance = MagicMock()
    mock_instance.validate.return_value = False
    mock_instance.non_validated_files = []
    mock_instance.glacier_files = ["s3://bucket/file1", "s3://bucket/file2"]
    mock_cloud_validator.return_value = mock_instance

    # Mock arguments
    with patch(
        "ugbio_omics.retrieve_from_glacier.parse_arguments",
        return_value=Namespace(wdl="test.wdl", param_json="test.json", retrieve=False, n_days=30),
    ):
        main()

    # Assertions
    mock_logger.info.assert_any_call("The WDL and JSON files are valid, but the following files are in GLACIER:")
    mock_logger.info.assert_any_call("s3://bucket/file1")
    mock_logger.info.assert_any_call("s3://bucket/file2")
    mock_logger.info.assert_any_call("Use --retrieve to start retrieval of the files from GLACIER.")


def test_glacier_files_with_retrieve(mock_cloud_validator, mock_boto3_client, mock_logger):
    # Mock CloudFilesValidator behavior
    mock_instance = MagicMock()
    mock_instance.validate.return_value = False
    mock_instance.non_validated_files = []
    mock_instance.glacier_files = ["s3://bucket/file1", "s3://bucket/file2"]
    mock_cloud_validator.return_value = mock_instance

    # Mock boto3 client
    mock_s3 = MagicMock()
    mock_boto3_client.return_value = mock_s3

    # Mock arguments
    with patch(
        "ugbio_omics.retrieve_from_glacier.parse_arguments",
        return_value=Namespace(wdl="test.wdl", param_json="test.json", retrieve=True, n_days=30),
    ):
        main()

    # Assertions
    mock_logger.info.assert_any_call("Starting retrieval of the files from GLACIER.")
    mock_s3.restore_object.assert_any_call(Bucket="bucket", Key="file1", RestoreRequest={"Days": 30})
    mock_s3.restore_object.assert_any_call(Bucket="bucket", Key="file2", RestoreRequest={"Days": 30})
    mock_logger.info.assert_any_call("Retrieval started.")
