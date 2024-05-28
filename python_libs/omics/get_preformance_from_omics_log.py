import json
import os
from argparse import ArgumentParser
from datetime import timedelta

import boto3
import dateutil.parser
import pandas as pd
from compute_pricing import get_run_cost, get_run_info
from typing import Tuple


class MonitorLog:
    def __init__(self):
        self.df = pd.DataFrame(columns=["CPU","Memory","IO_rKb/s","IO_wKb/s","IOWait"])
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
            # Convert the date string: Tue Feb 13 15:33:19 IST 2024
            time = dateutil.parser.parse(split_line[1].split("[")[1].split("]")[0])  
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
            #line=MONITORING, [Sun Mar 24 21:35:30 UTC 2024], %CPU: 52.10, %Memory: 21.00, IO_rKb/s: 41.00, IO_wKb/s: 367.00, %IOWait: 0.00
            split_line = line.split("MONITORING")[1].split(",")
            # Convert the date string: Tue Feb 13 15:33:28 IST 2024
            time = dateutil.parser.parse(split_line[1].split("[")[1].split("]")[0])
            try:
                cpu = float(split_line[2].split(":")[1] or 0)
            except ValueError:
                cpu = 0.0
            try:
                memory = float(split_line[3].split(":")[1] or 0)
            except ValueError:
                memory = 0.0
            try:
                io_rkb = float(split_line[4].split(":")[1] or 0)
            except:
                io_rkb = 0.0
            try:
                io_wkb = float(split_line[5].split(":")[1] or 0)
            except:
                io_wkb = 0.0
            try:
                iowait = float(split_line[6].split(":")[1] or 0)
            except:
                iowait = 0.0

            if self.start_time is None:
                self.start_time = time

            self.run_time = time - self.start_time
            self.df = pd.concat([self.df, pd.DataFrame({   
                "CPU": [cpu], 
                "Memory": [memory], 
                "IO_rKb/s":[io_rkb], 
                "IO_wKb/s": [io_wkb], 
                "IOWait": [iowait]
                })], ignore_index=True)

def performance(run_id, session=None, output_dir=None, output_prefix='') -> Tuple[pd.DataFrame, dict]:
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    total_performance_df = pd.DataFrame()

    # Get run info from omics
    if session:
        omics_client = session.client('omics')
    else:
        omics_client = boto3.client('omics')
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
            "total_Memory(GB)": [monitor_log.total_memory],
            "mean_%_Memory": [monitor_log.df['Memory'].mean()],
            "max_%_Memory": [monitor_log.df['Memory'].max()],
            "run_time (hours)": [monitor_log.run_time.total_seconds()/3600],
            "mean_IO_rKb/s": [monitor_log.df['IO_rKb/s'].mean()],
            "max_IO_rKb/s": [monitor_log.df['IO_rKb/s'].max()],
            "mean_IO_wKb/s": [monitor_log.df['IO_wKb/s'].mean()],
            "max_IO_wKb/s": [monitor_log.df['IO_wKb/s'].max()],
            "mean_IOWait": [monitor_log.df['IOWait'].mean()],
            "max_IOWait": [monitor_log.df['IOWait'].max()],
            "count_entries": [monitor_log.df['CPU'].count()],
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
    cost_output = f"{output_prefix}omics_{run_id}.cost.json"
    if output_dir is not None:
        cost_output = f"{output_dir}/{cost_output}"
    print(f"Saving cost data to: {cost_output}")
    with open(cost_output, 'w') as f:
        json.dump(cost, f, indent=4)

    # Save performance data
    output = f"{output_prefix}omics_{run_id}.performance.csv"
    if output_dir is not None:
        output = f"{output_dir}/{output}"
    print(f"Saving performance data to: {output}")
    total_performance_df.to_csv(output, index=False)

    return total_performance_df, cost

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
    while len(response.get('events')) > 0 and response.get('nextToken'):
        response = client.filter_log_events(
            logGroupName=log_group_name,
            logStreamNames=[log_stream_name],
            filterPattern='MONITORING',
            nextToken=response.get('nextToken')
        )

        for event in response['events']:
            monitor_log.process_line(event['message'])
    
    print(f"Done processing monitor log for task {task_id}")
    return monitor_log


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--region', type=str, help="AWS region to use", default='us-east-1')
    parser.add_argument('--run-id', type=str, help="HealthOmics workflow run-id to analyze")
    parser.add_argument('--output-path', type=str, help="Output dir to save performance data", required=False)
    parser.add_argument('--output-prefix', type=str, help="File name prefix for the output", required=False)

    args = parser.parse_args()
    session = boto3.Session(region_name=args.region)

    performance(args.run_id, session=session, output_dir=args.output_path, output_prefix=args.output_prefix)