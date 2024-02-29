from argparse import ArgumentParser
from datetime import timedelta, datetime
import boto3
import pandas as pd

from compute_pricing import get_run_cost, get_run_info

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
        
        if "General Information" in line:
            # Get general information
            #line=MONITORING, [Tue Feb 13 15:33:19 IST 2024], General Information, CPU: 20, Memory(GiB): 7
            split_line = line.split("MONITORING")[1].split(",")
            time = datetime.strptime(
                    " ".join(split_line[1].split("[")[1].split("]")[0].split()[:-2]),
                      '%a %b %d %H:%M:%S')  # Ignore time zone and convert the string: Tue Feb 13 15:33:19
            if self.start_time is None:
                self.start_time = time
            self.total_cpu = int(split_line[3].split(":")[1])
            self.total_memory = float(split_line[4].split(":")[1].split("GiB")[0])

        else:
            #line=MONITORING, [Tue Feb 13 15:33:28 IST 2024], %CPU: 0.30, %Memory: 51.80
            split_line = line.split("MONITORING")[1].split(",")
            time = datetime.strptime(
                    " ".join(split_line[1].split("[")[1].split("]")[0].split()[:-2]),
                      '%a %b %d %H:%M:%S')  # Ignore time zone and convert the string: Tue Feb 13 15:33:28
            cpu = float(split_line[2].split(":")[1])
            memory = float(split_line[3].split(":")[1])

            if self.start_time is None:
                self.start_time = time

            self.run_time = time - self.start_time
            self.df = self.df.append({"CPU": cpu, "Memory": memory}, ignore_index=True)

def performance(run_id, session=None, output=None):
    omics_client = session.client('omics')
    columns=[   "task",
                "total_CPU", "mean_%_CPU","max_%_CPU","count_entries_CPU",
                "total_Memory(GB)", "mean_%_Memory","max_%_Memory","count_entries_Memory",
                "run_time (hours)"]
             
    total_performance_df = pd.DataFrame(columns=columns)
    run = get_run_info(run_id, client=omics_client)

    for task in run['tasks']:
        print("------------------------------------------")
        print(f"Process monitor log for task {task['name']} (taskId: {task['taskId']})")
        monitor_log = process_monitor_log(run_id, task['taskId'], client=session.client('logs'))

        total_performance_df = total_performance_df.append({
            "task": task['name'],
            "total_CPU": monitor_log.total_cpu,
            "mean_%_CPU": monitor_log.df['CPU'].mean(),
            "max_%_CPU": monitor_log.df['CPU'].max(),
            "count_entries_CPU": monitor_log.df['CPU'].count(),
            "total_Memory(GB)": monitor_log.total_memory,
            "mean_%_Memory": monitor_log.df['Memory'].mean(),
            "max_%_Memory": monitor_log.df['Memory'].max(),
            "count_entries_Memory": monitor_log.df['Memory'].count(),
            "run_time (hours)": monitor_log.run_time.total_seconds()/3600
        }, ignore_index=True)

    if output is None:
        output = f"performance_run_{run_id}.csv"
    
    print("Add cost per task to performance data")
    cost = get_run_cost(run_id, client=omics_client)
    tasks_costs = cost['cost_detail']['task_costs']
    cost_df = pd.DataFrame({
        "task": [task['name'] for task in tasks_costs],
        "cost": [task['cost'] for task in tasks_costs],
        "instance": [task['instance'] for task in tasks_costs]})
    total_performance_df = total_performance_df.merge(cost_df, on="task", how="left")

    print(f"Saving performance data to {output}")
    total_performance_df.to_csv(output, index=False)

def process_monitor_log(run_id, task_id, client=None) -> MonitorLog:
    if not client:
        client = boto3.client('logs')
    
    log_stream_name = f'run/{run_id}/task/{task_id}'
    log_group_name = '/aws/omics/WorkflowLog'

    monitor_log = MonitorLog()

    print(f"Get log events for log group '{log_group_name}' and log stream '{log_stream_name}'")

    response = client.filter_log_events(
        logGroupName=log_group_name,
        logStreamNames=[log_stream_name],
        filterPattern='MONITORING'
    )

    # check that log is not empty
    if not response.get('events'):
        print(f"No events found for log group '{log_group_name}' and log stream '{log_stream_name}'")
        return monitor_log
    
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
    parser.add_argument('--output', type=str, help="Output file path to save performance data", required=False)

    args = parser.parse_args()
    session = boto3.Session(region_name=args.region, profile_name=args.profile)

    performance(args.run_id, session=session, output=args.output)