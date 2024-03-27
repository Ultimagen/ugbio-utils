from argparse import ArgumentParser
import os

import boto3

from compute_pricing import get_run_info


def get_log_for_task(run_id, task_id=None, session=None, output_path=None, output_prefix=''):
    # Get omics client to retrieve the run and tasks information
    if session:
        omics_client = session.client('omics')
    else:
        omics_client = boto3.client('omics')
    run = get_run_info(run_id, client=omics_client)

    # If no given a specific task_id, get the logs for all tasks
    if not task_id:
        task_ids = [task['taskId'] for task in run['tasks']]
        task_names = {task['taskId']: task['name'] for task in run['tasks']}
    else:
        task_ids = [task_id]
        task_names = {task['taskId']: task['name'] for task in run['tasks'] if task['taskId'] == task_id}

    # Get the logs for each task
    for task_id in task_ids:
        task_name = task_names[task_id]
        print("------------------------------------------")
        print(f"Getting log for task {task_name} (taskId: {task_id})")
        log_stream_name = f'run/{run_id}/task/{task_id}'
        log_group_name = '/aws/omics/WorkflowLog'
        
        output = f'{output_prefix}run_{run_id}_task_{task_id}_{task_name}.log'
        if output_path:
            os.makedirs(output_path, exist_ok=True)
            output = f"{output_path}/{output}"

        fetch_save_log(log_stream_name, log_group_name, output, session)

def fetch_save_log(log_stream_name, log_group_name, output, session=None):
    # Get logs client
    if session:
        client = session.client('logs')
    else:
        client = boto3.client('logs')

    print(f"Getting log events for log group '{log_group_name}' and log stream '{log_stream_name}'")

    # get first page of log events
    response = client.get_log_events( 
        logGroupName=log_group_name,
        logStreamName=log_stream_name,
        startFromHead=True
    )

    # check if events is not empty
    if not response.get('events'):
        print(f"No events found for log group '{log_group_name}' and log stream '{log_stream_name}'")
        return

    # write event message to output file
    with open(output, 'w') as file:
        for event in response['events']:
            file.write(f"{event['message']}\n")

        # get next page of log events
        while len(response.get('events')) > 0 and response.get('nextForwardToken'):
            response = client.get_log_events(
                logGroupName=log_group_name,
                logStreamName=log_stream_name,
                nextToken=response.get('nextForwardToken'),
                startFromHead=True
            )
            for event in response['events']:
                file.write(f"{event['message']}\n")
    
    print(f"Log file saved to: {output}")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--profile', type=str, help="AWS profile name to use from local profiles listed in ~/.aws/config")
    parser.add_argument('--region', type=str, help="AWS region to use", default='us-east-1')
    parser.add_argument('--run-id', type=str, help="HealthOmics workflow run-id to analyze")
    parser.add_argument('--task-id', type=str, help="HealthOmics workflow task-id to analyze. Leave empty to get the logs for all tasks", default=None)
    parser.add_argument('--output', type=str, help="Output dir to save log events", default=None)
    parser.add_argument('--output-prefix', type=str, help="File name prefix for the output", required=False)

    args = parser.parse_args()
    session = boto3.Session(region_name=args.region, profile_name=args.profile)

    get_log_for_task(args.run_id, args.task_id, session=session, output_path=args.output, output_prefix=args.output_prefix)