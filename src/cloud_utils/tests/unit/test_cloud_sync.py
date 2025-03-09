import os
from unittest import mock
from unittest.mock import MagicMock, patch

import pytest
from ugbio_cloud_utils.cloud_sync import (
    cloud_sync,
    dir_path,
    download_from_gs,
    download_from_s3,
    optional_cloud_sync,
)


def test_dir_path_local():
    """Test dir_path with a valid local directory."""
    with patch("os.path.isdir", return_value=True):
        assert dir_path("/some/local/path") == "/some/local/path"


def test_dir_path_cloud():
    """Test dir_path with a valid cloud path."""
    assert dir_path("gs://gs-bucket/object", check_cloud_path=True) == "gs://gs-bucket/object"
    assert dir_path("s3://s3-bucket/object", check_cloud_path=True) == "s3://s3-bucket/object"


def test_dir_path_invalid_cloud():
    """Test dir_path with an invalid cloud path."""
    with pytest.raises(ValueError):
        dir_path("invalid-cloud-path", check_cloud_path=True)


def test_dir_path_invalid_local():
    """Test dir_path with an invalid local path."""
    with pytest.raises(NotADirectoryError):
        dir_path("/invalid/local/path")


@patch("ugbio_cloud_utils.cloud_sync.download_from_gs")
@patch("ugbio_cloud_utils.cloud_sync.download_from_s3")
def test_cloud_sync_gs(mock_download_from_s3, mock_download_from_gs, tmp_path):
    """Test cloud_sync for a Google Cloud Storage path."""
    local_dir = tmp_path / "data"
    cloud_path = "gs://gs-bucket/obj1"
    os.makedirs(local_dir)

    with patch("os.path.isdir", return_value=True):
        result = cloud_sync(cloud_path, str(local_dir), dry_run=True)
        assert "cloud_sync/gs/gs-bucket/obj1" in result

    mock_download_from_gs.assert_not_called()
    mock_download_from_s3.assert_not_called()

    with patch("os.path.isdir", return_value=True):
        cloud_sync(cloud_path, str(local_dir), print_output=True)

    mock_download_from_gs.assert_called_once_with("gs-bucket", "obj1", mock.ANY)
    mock_download_from_s3.assert_not_called()


@patch("ugbio_cloud_utils.cloud_sync.download_from_gs")
@patch("ugbio_cloud_utils.cloud_sync.download_from_s3")
def test_cloud_sync_s3(mock_download_from_s3, mock_download_from_gs, tmp_path):
    """Test cloud_sync for an S3 path."""
    local_dir = tmp_path / "data"
    cloud_path = "s3://s3-bucket/obj1"
    os.makedirs(local_dir)

    with patch("os.path.isdir", return_value=True):
        result = cloud_sync(cloud_path, str(local_dir), dry_run=True)
        assert "cloud_sync/s3/s3-bucket/obj1" in result

    mock_download_from_gs.assert_not_called()
    mock_download_from_s3.assert_not_called()

    with patch("os.path.isdir", return_value=True):
        cloud_sync(cloud_path, str(local_dir), print_output=True)

    mock_download_from_gs.assert_not_called()
    mock_download_from_s3.assert_called_once_with("s3-bucket", "obj1", mock.ANY)


def test_optional_cloud_sync(tmp_path):
    """Test optional_cloud_sync."""
    local_dir = tmp_path / "data"
    os.makedirs(local_dir)

    with patch("ugbio_cloud_utils.cloud_sync.cloud_sync") as mock_cloud_sync:
        optional_cloud_sync(
            "gs://gs-bucket/obj1",
            str(local_dir),
            print_output=False,
            force_download=False,
            raise_error_is_file_exists=False,
            dry_run=False,
        )
        mock_cloud_sync.assert_called_once_with(
            "gs://gs-bucket/obj1",
            str(local_dir),
            print_output=False,
            force_download=False,
            raise_error_is_file_exists=False,
            dry_run=False,
        )

    with patch("ugbio_cloud_utils.cloud_sync.cloud_sync") as mock_cloud_sync:
        result = optional_cloud_sync(str(local_dir), str(local_dir))
        assert result == str(local_dir)
        mock_cloud_sync.assert_not_called()


@patch("google.cloud.storage.Client")
def test_download_from_gs(mock_storage_client):
    """Test download_from_gs."""
    mock_bucket = MagicMock()
    mock_blob = MagicMock()
    mock_storage_client.return_value.bucket.return_value = mock_bucket
    mock_bucket.blob.return_value = mock_blob

    download_from_gs("gs-bucket", "obj1", "file1")

    mock_storage_client.assert_called_once()
    mock_bucket.blob.assert_called_once_with("obj1")
    mock_blob.download_to_filename.assert_called_once_with("file1")


@patch("boto3.Session")
def test_download_from_s3(mock_boto3_session):
    """Test download_from_s3."""
    mock_client = MagicMock()
    mock_boto3_session.return_value.client.return_value = mock_client

    download_from_s3("s3-bucket", "obj1", "file1")

    mock_boto3_session.assert_called_once_with(profile_name="default")
    mock_client.download_file.assert_called_once_with("s3-bucket", "obj1", "file1")
