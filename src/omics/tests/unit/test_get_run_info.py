import json
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from dateutil import parser
from ugbio_omics.get_run_info import get_run_info


@pytest.fixture
def resources_dir():
    inputs_dir = Path(__file__).parent.parent / "resources"
    return inputs_dir


@pytest.fixture
def mock_boto_client():
    with patch("ugbio_omics.get_run_info.boto3.client") as mock_client:
        yield mock_client


@pytest.fixture
def mock_get_run_return_value():
    return {
        "id": "test_run_id",
        "startTime": parser.parse("2024-11-27T12:50:06.336648+00:00"),
        "stopTime": parser.parse("2024-11-27T12:59:56.336648+00:00"),
        "ResponseMetadata": {},
    }


@pytest.fixture
def mock_list_run_tasks_return_values():
    return [
        {
            "items": [
                {
                    "taskId": "5484191",
                    "status": "COMPLETED",
                    "name": "MrdDataAnalysis",
                    "cpus": 4,
                    "memory": 8,
                    "creationTime": parser.parse("2024-11-27T12:53:06.336648+00:00"),
                    "startTime": parser.parse("2024-11-27T12:55:06.336648+00:00"),
                    "stopTime": parser.parse("2024-11-27T12:59:06.336648+00:00"),
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
                    "creationTime": parser.parse("2024-11-27T12:53:06.336648+00:00"),
                    "startTime": parser.parse("2024-11-27T12:54:06.336648+00:00"),
                    "stopTime": parser.parse("2024-11-27T12:56:06.336648+00:00"),
                    "gpus": 0,
                    "instanceType": "omics.c.xlarge",
                }
            ],
            "nextToken": None,
        },
    ]


# Custom JSON decoder for datetime and timedelta objects
def json_deserial(obj):
    """JSON deserializer for objects not serializable by default json code"""
    for key, value in obj.items():
        if isinstance(value, str):
            try:
                obj[key] = datetime.fromisoformat(value)
            except ValueError:
                try:
                    # Assuming the timedelta is stored as "HH:MM:SS"
                    hours, minutes, seconds = map(int, value.split(":"))
                    obj[key] = timedelta(hours=hours, minutes=minutes, seconds=seconds)
                except ValueError:
                    pass
    return obj


def test_get_run_info_no_client(
    mock_boto_client, resources_dir, mock_get_run_return_value, mock_list_run_tasks_return_values
):
    mock_client_instance = mock_boto_client.return_value
    mock_client_instance.get_run.return_value = mock_get_run_return_value
    mock_client_instance.list_run_tasks.side_effect = mock_list_run_tasks_return_values

    run_info = get_run_info("test_run_id")

    with open(resources_dir / "expected_run_info.json") as f:
        expected_run_info = json.load(f, object_hook=json_deserial)

    assert run_info == expected_run_info


def test_get_run_info_with_client(mock_get_run_return_value):
    mock_client_instance = MagicMock()
    mock_client_instance.get_run.return_value = mock_get_run_return_value
    mock_client_instance.list_run_tasks.return_value = {"items": [], "nextToken": None}

    run_info = get_run_info("test_run_id", client=mock_client_instance)

    assert run_info["id"] == "test_run_id"
    assert run_info["tasks"] == []


def test_get_run_info_no_stoptime(mock_get_run_return_value, mock_list_run_tasks_return_values):
    mock_client_instance = MagicMock()
    mock_get_run_return_value.pop("stopTime")
    mock_client_instance.get_run.return_value = mock_get_run_return_value
    mock_list_run_tasks_return_values[1]["items"][0].pop("stopTime")
    mock_client_instance.list_run_tasks.side_effect = mock_list_run_tasks_return_values

    run_info = get_run_info("test_run_id", client=mock_client_instance)

    assert run_info["duration"]
    assert run_info["tasks"][0]["duration"]


def test_get_run_info_tasks_pagination(mock_get_run_return_value, mock_list_run_tasks_return_values):
    mock_client_instance = MagicMock()
    mock_client_instance.get_run.return_value = mock_get_run_return_value
    mock_client_instance.list_run_tasks.side_effect = mock_list_run_tasks_return_values

    run_info = get_run_info("test_run_id", client=mock_client_instance)

    assert len(run_info["tasks"]) == 2
    mock_client_instance.list_run_tasks.assert_called_with(id="test_run_id", startingToken="token1")
