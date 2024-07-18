import json
import os
from argparse import ArgumentParser
from datetime import timedelta
from typing import Tuple

import boto3
import dateutil.parser
import pandas as pd
import plotly.graph_objects as go
from compute_pricing import get_run_cost, get_run_info
from plotly.subplots import make_subplots
from enum import Enum


class MonitorColumns(Enum):
    TIME = "time"
    CPU = "CPU"
    MEMORY = "Memory"
    IO_RKB = "IO_rKb/s"
    IO_WKB = "IO_wKb/s"
    IOWAIT = "IOWait"

class MonitorLog:
    def __init__(self):
        self.df_columns = [column.value for column in MonitorColumns]
        self.df = pd.DataFrame(columns=self.df_columns)
        self.start_time = None
        self.run_time = timedelta(0)
        self.total_cpu = None
        self.total_memory = None
        self.task_name = None
        self.task_id = None
        # create a dict where key is the column name and value is the color
        self.colors = dict(zip(self.df_columns, ['black','blue', 'green', 'red', 'purple', 'orange']))
    
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
                "time": [time], 
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

    monitor_logs = [] 

    # Process monitor log for each task
    for task in run['tasks']:
        print("------------------------------------------")
        print(f"Process monitor log for task {task['name']} (taskId: {task['taskId']})")
        monitor_log = process_monitor_log(run_id, task, client=session.client('logs'))
        monitor_logs.append(monitor_log)

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

    # Save figures to HTML
    save_figures_to_html(monitor_logs, run_id, output_dir, output_prefix)

    return total_performance_df, cost

def process_monitor_log(run_id, task, client=None) -> MonitorLog:
    if not client:
        client = boto3.client('logs')
    
    task_id = task['taskId']
    
    log_stream_name = f'run/{run_id}/task/{task_id}'
    log_group_name = '/aws/omics/WorkflowLog'

    monitor_log = MonitorLog()
    monitor_log.task_name = task['name']
    monitor_log.task_id = task_id

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

def save_figures_to_html(monitor_logs, run_id, output_dir=None, output_prefix=''):
    print("Generate and save performance plots to HTML...")
    # TODO: combine scttared tasks into one plot
    # Plot only tasks that ran for more than 5 minutes.
    dont_plot_tasks = []
    plot_tasks = []
    
    for monitor_log in monitor_logs:

        # Monitor log script prints every 10 seconds.
        # If monitor_log.df has less than 30 entires it means it worked less than 5 minutes and we don't need the plot for this task.
        if monitor_log.df.shape[0] < 30: 
            print(f"Task: {monitor_log.task_name} (id:{monitor_log.task_id}) run less than 5 minutes and will not be plotted")
            dont_plot_tasks.append(monitor_log)
        else:
            plot_tasks.append(monitor_log)

    # Combine all figures into one HTML file
    num_rows = len(plot_tasks)  # Each figure will be in its own row
    combined_fig = make_subplots(rows=num_rows, cols=1, 
            subplot_titles=[f"Task: {monitor_log.task_name} (id:{monitor_log.task_id})" for monitor_log in plot_tasks])

    add_to_legend = set()

    for i, monitor_log in enumerate(plot_tasks, start=1):
        print(f"Plotting task: {monitor_log.task_name} (id:{monitor_log.task_id})")
        # Add traces for each metric
        add_trace(combined_fig, monitor_log, MonitorColumns.CPU.value, i)
        add_trace(combined_fig, monitor_log, MonitorColumns.MEMORY.value, i)
        # Add IO metrics only if the coresepong coulmns are not empty and sum is not 0
        if monitor_log.df[MonitorColumns.IO_RKB.value].notna().any() and monitor_log.df[MonitorColumns.IO_RKB.value].sum() != 0:
            add_trace(combined_fig, monitor_log, MonitorColumns.IO_RKB.value, i, normelize=True)
            add_to_legend.add(MonitorColumns.IO_RKB.value)
        if monitor_log.df[MonitorColumns.IO_WKB.value].notna().any() and monitor_log.df[MonitorColumns.IO_WKB.value].sum() != 0:
            add_trace(combined_fig, monitor_log, MonitorColumns.IO_WKB.value, i, normelize=True)
            add_to_legend.add(MonitorColumns.IO_WKB.value)
        if monitor_log.df[MonitorColumns.IOWAIT.value].notna().any() and monitor_log.df[MonitorColumns.IOWAIT.value].sum() != 0:
            add_trace(combined_fig, monitor_log, MonitorColumns.IOWAIT.value, i)
            add_to_legend.add(MonitorColumns.IOWAIT.value)
        
        # y-axis settings
        combined_fig.update_yaxes(title="%", range=[0, 110], row=i, col=1)
        
        # Add caption for each subplot
        combined_fig.add_annotation(
            text=f"#CPU: {monitor_log.total_cpu}, Memory(Gib): {monitor_log.total_memory}, runtime(H): {monitor_log.run_time.total_seconds()/3600:.2f}", 
            xref=f"x{i if i != 1 else ''} domain",  # Reference the x-axis of the ith subplot
            yref=f"y{i if i != 1 else ''} domain",  # Reference the y-axis of the ith subplot
            x=0.5,  # Position the caption in the middle of the subplot horizontally
            y=-0.15,  # Position the caption just below the subplot
            showarrow=False,
            font=dict(size=12),
            align="center",
            xanchor="center",
            yanchor="top"
        )
    
    # Add missing lables in the legend by adding invisible traces
    for label in add_to_legend:
        combined_fig.add_trace(
            go.Scatter(x=[None], y=[None], mode='lines', name=label),
            row=1, col=1  # Add it to the first subplot where the leggend appears
        )

    # touch up the layout and add title and subtext
    combined_fig.update_layout(height=300*num_rows+50, title_text=f"Omics {run_id} Performance Plots", title_font=dict(size=20))
    if len(dont_plot_tasks) > 0:
        combined_fig.add_annotation(
            text=f"Short tasks without a plot: {[f"{monitor_log.task_name} (runtime (H): {monitor_log.run_time.total_seconds()/3600:.2f})" for monitor_log in dont_plot_tasks]}",  # Your subtitle text
            xref="paper",  # Position relative to the entire plotting area
            yref="paper",  # Position relative to the entire plotting area
            x=0,  # Center the text horizontally
            y=1,  # Adjust this value as needed to position below the title
            showarrow=False,
            font=dict(size=12),
            yanchor="bottom",
            yshift=25  # Shift down by 25 pixels to add more "padding"
        )

    # Save the combined figure to an HTML file
    combined_fig_output = f"{output_prefix}omics_{run_id}_performance_plots.html"
    if output_dir is not None:
        combined_fig_output = f"{output_dir}/{combined_fig_output}"
    print(f"Saving performance plots report to: {combined_fig_output}")
    combined_fig.write_html(combined_fig_output)

def add_trace(combined_fig, monitor_log, col_name, row, normelize=False):
    show_legend = row == 1

    if normelize:
        monitor_log.df[col_name] = monitor_log.df[col_name] / monitor_log.df[col_name].max() * 100

    combined_fig.add_trace(go.Scatter(
            x=monitor_log.df[MonitorColumns.TIME.value],
            y=monitor_log.df[col_name],
            mode='lines',
            name=col_name,
            line=dict(color=monitor_log.colors[col_name]),
            showlegend=show_legend), row=row, col=1)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--region', type=str, help="AWS region to use", default='us-east-1')
    parser.add_argument('--run-id', type=str, help="HealthOmics workflow run-id to analyze")
    parser.add_argument('--output-path', type=str, help="Output dir to save performance data", required=False)
    parser.add_argument('--output-prefix', type=str, help="File name prefix for the output", required=False, default='')

    args = parser.parse_args()
    session = boto3.Session(region_name=args.region)

    performance(args.run_id, session=session, output_dir=args.output_path, output_prefix=args.output_prefix)