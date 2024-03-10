from argparse import ArgumentParser
from datetime import timedelta, datetime
import os
import boto3
import pandas as pd

from compute_pricing import get_run_cost, get_run_info
import json

class MonitorLog:
    def __init__(self):
        self.df = pd.DataFrame(columns=["CPU","Memory"])
        self.start_time = None
        self.run_time = timedelta(0)
        self.total_cpu = None
        self.total_memory = None
    
    def process_line(self, line):
        if 'MONITORING' not in line:
            return
        
        # Get general information
        if "General Information" in line:
            #line=MONITORING, [Tue Feb 13 15:33:19 IST 2024], General Information, CPU: 20, Memory(GiB): 7
            split_line = line.split("MONITORING")[1].split(",")
            time = datetime.strptime(
                    " ".join(split_line[1].split("[")[1].split("]")[0].split()[:-2]),
                      '%a %b %d %H:%M:%S')  # Ignore time zone and convert the string: Tue Feb 13 15:33:19
            if self.start_time is None:
                self.start_time = time
            try:
                self.total_cpu = int(split_line[3].split(":")[1] or 0)
            except ValueError:
                self.total_cpu = 0
            try:
                self.total_memory = float(split_line[4].split(":")[1].split("GiB")[0] or 0)
            except ValueError:
                self.total_memory = 0.0

        # Get monitoring information
        else:
            #line=MONITORING, [Tue Feb 13 15:33:28 IST 2024], %CPU: 0.30, %Memory: 51.80
            split_line = line.split("MONITORING")[1].split(",")
            time = datetime.strptime(
                    " ".join(split_line[1].split("[")[1].split("]")[0].split()[:-2]),
                      '%a %b %d %H:%M:%S')  # Ignore time zone and convert the string: Tue Feb 13 15:33:28
            try:
                cpu = float(split_line[2].split(":")[1] or 0)
            except ValueError:
                cpu = 0.0
            try:
                memory = float(split_line[3].split(":")[1] or 0)
            except ValueError:
                memory = 0.0

            if self.start_time is None:
                self.start_time = time

            self.run_time = time - self.start_time
            self.df = pd.concat([self.df, pd.DataFrame({"CPU": [cpu], "Memory": [memory]})], ignore_index=True)

def performance(run_id, session=None, output_prefix=None):
    os.makedirs(output_prefix, exist_ok=True)

    columns=[   "task",
                "total_CPU", "mean_%_CPU","max_%_CPU","count_entries_CPU",
                "total_Memory(GB)", "mean_%_Memory","max_%_Memory","count_entries_Memory",
                "run_time (hours)"]
    total_performance_df = pd.DataFrame(columns=columns)

    # Get run info from omics
    omics_client = session.client('omics')
    run = get_run_info(run_id, client=omics_client)

    # Process monitor log for each task
    for task in run['tasks']:
        print("------------------------------------------")
        print(f"Process monitor log for task {task['name']} (taskId: {task['taskId']})")
        monitor_log = process_monitor_log(run_id, task['taskId'], client=session.client('logs'))

        new_row = pd.DataFrame({
            "task": [task['name']],
            "total_CPU": [monitor_log.total_cpu],
            "mean_%_CPU": [monitor_log.df['CPU'].mean()],
            "max_%_CPU": [monitor_log.df['CPU'].max()],
            "count_entries_CPU": [monitor_log.df['CPU'].count()],
            "total_Memory(GB)": [monitor_log.total_memory],
            "mean_%_Memory": [monitor_log.df['Memory'].mean()],
            "max_%_Memory": [monitor_log.df['Memory'].max()],
            "count_entries_Memory": [monitor_log.df['Memory'].count()],
            "run_time (hours)": [monitor_log.run_time.total_seconds()/3600]
        })
        total_performance_df = pd.concat([total_performance_df, new_row], ignore_index=True)
    
    # Process cost and add to the performance data
    print("Add cost per task to performance data")
    cost = get_run_cost(run_id, client=omics_client)
    storage_cost = cost['cost_detail']['storage_cost']['cost']
    tasks_costs = cost['cost_detail']['task_costs']
    cost_df = pd.DataFrame({
        "task": [task['name'] for task in tasks_costs],
        "cost": [task['cost'] for task in tasks_costs],
        "instance": [task['instance'] for task in tasks_costs],
        "total_storage_cost": storage_cost})
    total_performance_df = total_performance_df.merge(cost_df, on="task", how="left")

    # Save cost information
    cost_output = f"omics_{run_id}.cost.json"
    if output_prefix is not None:
        cost_output = f"{output_prefix}/{cost_output}"
    print(f"Saving cost data to: {cost_output}")
    with open(cost_output, 'w') as f:
        json.dump(cost, f, indent=4)

    # Save performance data
    output = f"omics_{run_id}.performance.csv"
    if output_prefix is not None:
        output = f"{output_prefix}/{output}"
    print(f"Saving performance data to: {output}")
    total_performance_df.to_csv(output, index=False)

def process_monitor_log(run_id, task_id, client=None) -> MonitorLog:
    if not client:
        client = boto3.client('logs')
    
    log_stream_name = f'run/{run_id}/task/{task_id}'
    log_group_name = '/aws/omics/WorkflowLog'

    monitor_log = MonitorLog()

    print(f"Get log events for log group '{log_group_name}' and log stream '{log_stream_name}'")

    # Get log events of specific task from CloudWatch that contain the word 'MONITORING'
    response = client.filter_log_events(
        logGroupName=log_group_name,
        logStreamNames=[log_stream_name],
        filterPattern='MONITORING'
    )

    # check that log is not empty
    if not response.get('events'):
        print(f"No events found for log group '{log_group_name}' and log stream '{log_stream_name}'")
        return monitor_log
    
    # process monitoring log events
    for event in response['events']:
        monitor_log.process_line(event['message'])

    # get next page of log events
    while len(response.get('events')) > 0 and response.get('nextForwardToken'):
        response = client.filter_log_events(
            logGroupName=log_group_name,
            logStreamNames=[log_stream_name],
            filterPattern='MONITORING',
            nextToken=response.get('nextForwardToken')
        )

        for event in response['events']:
            monitor_log.process_line(event['message'])
    
    print(f"Done processing monitor log for task {task_id}")
    return monitor_log


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--profile', type=str, help="AWS profile to use")
    parser.add_argument('--region', type=str, help="AWS region to use", default='us-east-1')
    parser.add_argument('--run-id', type=str, help="HealthOmics workflow run-id to analyze")
    parser.add_argument('--output-path', type=str, help="Output path to save performance data", required=False)

    args = parser.parse_args()
    session = boto3.Session(region_name=args.region, profile_name=args.profile)

    performance(args.run_id, session=session, output_prefix=args.output_path)