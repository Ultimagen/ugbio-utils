import os
from argparse import ArgumentParser

import boto3

from ugbio_omics.get_run_info import get_run_info

FAILED_STATUS = "FAILED"
OMICS_LOG_GROUP = "/aws/omics/WorkflowLog"


def get_log_for_task(run_id, task_id=None, session=None, output_path=None, output_prefix="", *, failed: bool = True):
    # Get omics client to retrieve the run and tasks information
    if session:
        omics_client = session.client("omics")
    else:
        omics_client = boto3.client("omics")
    run = get_run_info(run_id, client=omics_client)

    # If no given a specific task_id, get the logs for all tasks or failed tasks
    if not task_id:
        task_ids = [task["taskId"] for task in run["tasks"] if (not failed or task["status"] == FAILED_STATUS)]
        task_names = {
            task["taskId"]: task["name"] for task in run["tasks"] if (not failed or task["status"] == FAILED_STATUS)
        }

    else:
        task_ids = [task_id]
        task_names = {task["taskId"]: task["name"] for task in run["tasks"] if task["taskId"] == task_id}

    # Get the logs for each task
    for task_id_val in task_ids:
        task_name = task_names[task_id_val]
        print("------------------------------------------")
        print(f"Getting log for task {task_name} (taskId: {task_id_val})")
        log_stream_name = f"run/{run_id}/task/{task_id_val}"

        output = f"{output_prefix}run_{run_id}_task_{task_id_val}_{task_name}.log"
        fetch_save_log(log_stream_name, output, output_path, session)

    # in case that the run failed but there're no failed tasks the run's engine log should include the error
    if run["status"] == FAILED_STATUS and FAILED_STATUS not in [task["status"] for task in run["tasks"]]:
        print(f"Run status is {FAILED_STATUS} but no failed tasks. Getting run's engine log")
        log_stream_name = f"run/{run_id}/engine"
        output = f"{output_prefix}run_{run_id}_engine.log"
        fetch_save_log(log_stream_name, output, output_path, session)


def fetch_save_log(log_stream_name, output, output_path, session=None):
    # Get logs client
    if session:
        client = session.client("logs")
    else:
        client = boto3.client("logs")

    print(f"Getting log events for log group '{OMICS_LOG_GROUP}' and log stream '{log_stream_name}'")

    # get first page of log events
    response = client.get_log_events(logGroupName=OMICS_LOG_GROUP, logStreamName=log_stream_name, startFromHead=True)

    # check if events is not empty
    if not response.get("events"):
        print(f"No events found for log group '{OMICS_LOG_GROUP}' and log stream '{log_stream_name}'")
        return

    if output_path:
        os.makedirs(output_path, exist_ok=True)
        output = f"{output_path}/{output}"

    # write event message to output file
    with open(output, "w") as file:
        for event in response["events"]:
            file.write(f"{event['message']}\n")

        # get next page of log events
        while len(response.get("events")) > 0 and response.get("nextForwardToken"):
            response = client.get_log_events(
                logGroupName=OMICS_LOG_GROUP,
                logStreamName=log_stream_name,
                nextToken=response.get("nextForwardToken"),
                startFromHead=True,
            )
            for event in response["events"]:
                file.write(f"{event['message']}\n")

    print(f"Log file saved to: {output}")
    return output


def main():
    parser = ArgumentParser()
    parser.add_argument("--region", type=str, help="AWS region to use", default="us-east-1")
    parser.add_argument("--run-id", type=str, help="HealthOmics workflow run-id to analyze")
    parser.add_argument(
        "--task-id",
        type=str,
        help="HealthOmics workflow task-id to analyze. Leave empty to get the logs for all tasks",
        default=None,
    )
    parser.add_argument(
        "--failed", dest="failed", action="store_true", help="Set to true to get logs for failed tasks only"
    )
    parser.add_argument("--no-failed", dest="failed", action="store_false")
    parser.set_defaults(failed=False)
    parser.add_argument("--output", type=str, help="Output dir to save log events", default=None)
    parser.add_argument("--output-prefix", type=str, help="File name prefix for the output", required=False, default="")

    args = parser.parse_args()
    session = boto3.Session(region_name=args.region)

    get_log_for_task(
        args.run_id,
        args.task_id,
        session=session,
        output_path=args.output,
        output_prefix=args.output_prefix,
        failed=args.failed,
    )


if __name__ == "__main__":
    main()
