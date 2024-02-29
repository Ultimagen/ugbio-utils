from argparse import ArgumentParser
import boto3

def get_log_for_task(run_id, task_id, client=None, output=None):
    if not client:
        client = boto3.client('logs')
    
    log_stream_name = f'run/{run_id}/task/{task_id}'
    log_group_name = '/aws/omics/WorkflowLog'
    if not output:
        output = f'run_{run_id}_task_{task_id}.log'
    
    print(f"Getting log events for log group '{log_group_name}' and log stream '{log_stream_name}'")

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
    parser.add_argument('--task-id', type=str, help="HealthOmics workflow task-id to analyze")
    parser.add_argument('--output', type=str, help="Output file to save log events", default=None)

    args = parser.parse_args()
    session = boto3.Session(region_name=args.region, profile_name=args.profile)

    get_log_for_task(args.run_id, args.task_id, client=session.client('logs'), output=args.output)