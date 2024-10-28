from datetime import UTC, datetime
from unittest.mock import MagicMock, patch

import pytest
from ugbio_omics.get_run_info import get_run_info


@pytest.fixture
def mock_boto_client():
    with patch("ugbio_omics.get_run_info.boto3.client") as mock_client:
        yield mock_client


def test_get_run_info_no_client(mock_boto_client):
    mock_client_instance = mock_boto_client.return_value
    mock_client_instance.get_run.return_value = {
        "id": "test_run_id",
        "startTime": datetime.now(UTC),
        "stopTime": datetime.now(UTC),
        "ResponseMetadata": {},
    }
    mock_client_instance.list_run_tasks.side_effect = [
        {
            "items": [
                {
                    "taskId": "5484191",
                    "status": "COMPLETED",
                    "name": "MrdDataAnalysis",
                    "cpus": 4,
                    "memory": 8,
                    "creationTime": datetime.now(UTC),
                    "startTime": datetime.now(UTC),
                    "stopTime": datetime.now(UTC),
                    "gpus": 0,
                    "instanceType": "omics.c.xlarge",
                }
            ],
            "nextToken": "token1",
        },
        {
            "items": [
                {
                    "taskId": "2036762",
                    "status": "COMPLETED",
                    "name": "MrdDataAnalysis2",
                    "cpus": 4,
                    "memory": 8,
                    "creationTime": datetime.now(UTC),
                    "startTime": datetime.now(UTC),
                    "stopTime": datetime.now(UTC),
                    "gpus": 0,
                    "instanceType": "omics.c.xlarge",
                }
            ],
            "nextToken": None,
        },
    ]

    run_info = get_run_info("test_run_id")

    assert run_info["id"] == "test_run_id"
    assert run_info["tasks"][0]["taskId"] == "5484191"
    assert run_info["tasks"][1]["taskId"] == "2036762"
    assert run_info["tasks"][0].get("duration") is not None
    assert run_info["tasks"][1].get("duration") is not None


def test_get_run_info_with_client():
    mock_client_instance = MagicMock()
    mock_client_instance.get_run.return_value = {
        "id": "test_run_id",
        "startTime": datetime.now(UTC),
        "stopTime": datetime.now(UTC),
        "ResponseMetadata": {},
    }
    mock_client_instance.list_run_tasks.return_value = {"items": [], "nextToken": None}

    run_info = get_run_info("test_run_id", client=mock_client_instance)

    assert run_info["id"] == "test_run_id"
    assert run_info["tasks"] == []
