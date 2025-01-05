from unittest.mock import MagicMock, patch

from ugbio_omics.omics_cache_path import (
    build_cache_full_uri,
    copy_omics_cached_indexes,
    get_run_cache_path,
    process_s3_uri,
)


@patch("ugbio_omics.omics_cache_path.get_aws_client")
def test_get_run_cache_path(mock_get_aws_client):
    # Mock the Omics client response
    mock_aws_client = MagicMock()
    mock_aws_client.get_run.return_value = {"cacheId": "test-cache-id"}
    mock_aws_client.get_run_cache.return_value = {"cacheS3Uri": "s3://bucket1/cache_always/"}
    mock_aws_client.list_objects_v2.return_value = {
        "CommonPrefixes": [{"Prefix": "cache_always/test-run-id/test-task-id/task-uuid"}]
    }

    mock_get_aws_client.return_value = mock_aws_client

    run_id = "test-run-id"
    task_id = "test-task-id"

    # Test without task_id
    result = get_run_cache_path(run_id)
    assert result == "s3://bucket1/cache_always/test-run-id"

    # Test with task_id
    result = get_run_cache_path(run_id, task_id)
    assert result == "s3://bucket1/cache_always/test-run-id/test-task-id/task-uuid"


@patch("ugbio_omics.omics_cache_path.get_aws_client")
def test_build_cache_full_uri(mock_get_aws_client):
    # Mock the S3 client response
    mock_s3_client = MagicMock()
    mock_s3_client.list_objects_v2.return_value = {"CommonPrefixes": [{"Prefix": "cache_always/subfolder1/"}]}

    mock_get_aws_client.return_value = mock_s3_client

    s3_uri = "s3://bucket1/cache_always/"
    result = build_cache_full_uri(s3_uri)
    assert result == "s3://bucket1/cache_always/subfolder1/"


@patch("ugbio_omics.omics_cache_path.get_aws_client")
def test_copy_omics_cached_indexes(mock_get_aws_client):
    mock_aws_client = MagicMock()
    mock_aws_client.get_paginator = MagicMock()
    mock_paginator = MagicMock(return_value=None)
    mock_paginator.paginate = MagicMock()
    mock_paginator.paginate.return_value = iter(
        [{"Contents": [{"Key": "path/to/vcf_index/file1"}, {"Key": "path/to/cram_index/file2"}]}]
    )

    mock_aws_client.get_paginator.return_value = mock_paginator

    mock_aws_client.copy = MagicMock()

    mock_get_aws_client.return_value = mock_aws_client

    cache_s3_uri = "s3://bucket1/cache_always/my-run"
    copy_omics_cached_indexes(cache_s3_uri)

    # Assert the copy method is called with the expected arguments
    mock_aws_client.copy.assert_any_call(
        {"Bucket": "bucket1", "Key": "path/to/vcf_index/file1"},
        "bucket1",
        "path/to/vcf/file1",
    )
    mock_aws_client.copy.assert_any_call(
        {"Bucket": "bucket1", "Key": "path/to/cram_index/file2"},
        "bucket1",
        "path/to/cram/file2",
    )


def test_process_s3_uri():
    s3_uri = "s3://bucket1/cache_always/"
    bucket, prefix = process_s3_uri(s3_uri)
    assert bucket == "bucket1"
    assert prefix == "cache_always/"
