from datetime import UTC, datetime

import boto3


def get_run_info(run_id, client=None):
    if not client:
        client = boto3.client("omics")

    run = client.get_run(id=run_id)
    run.update({"duration": (run["stopTime"] if "stopTime" in run else datetime.now(UTC)) - run["startTime"]})

    response = client.list_run_tasks(id=run_id)
    tasks = response["items"]
    while response.get("nextToken"):
        response = client.list_run_tasks(id=run_id, startingToken=response.get("nextToken"))
        tasks += response["items"]

    def calc_task_duration(task):
        stop_time = task["stopTime"] if "stopTime" in task else datetime.now(UTC)
        start_time = (
            task["startTime"] if "startTime" in task else stop_time
        )  # when task is CANCELLED sometime there's no startTime
        return stop_time - start_time

    tasks = [{**task, "duration": calc_task_duration(task)} for task in tasks]

    del run["ResponseMetadata"]

    run["tasks"] = tasks
    return run
