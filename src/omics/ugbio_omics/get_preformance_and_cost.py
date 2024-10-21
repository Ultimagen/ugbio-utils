import os
from argparse import ArgumentParser
from datetime import timedelta, tzinfo
from enum import Enum

import boto3
import dateutil.parser
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ugbio_omics.get_omics_log import OMICS_LOG_GROUP
from ugbio_omics.get_run_cost import PLOTS_DIR, RUN_ID_PLACEHOLDER, Columns, RunCost
from ugbio_omics.get_run_info import get_run_info


class EDT(tzinfo):
    def utcoffset(self, dt):
        return timedelta(hours=-4)

    def dst(self, dt):
        return timedelta(0)

    def tzname(self, dt):
        return "EDT"


tzinfos = {
    "EDT": EDT(),
}


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
        self.colors = dict(zip(self.df_columns, ["black", "blue", "green", "red", "purple", "orange"], strict=False))

    def process_line(self, line):  # noqa: C901, PLR0912
        if "MONITORING" not in line:
            return

        # Get general information
        if "General Information" in line:
            # line=MONITORING, [Tue Feb 13 15:33:19 IST 2024], General Information, CPU: 20, Memory(GiB): 7
            split_line = line.split("MONITORING")[1].split(",")
            # Convert the date string: Tue Feb 13 15:33:19 IST 2024
            date_str = split_line[1].split("[")[1].split("]")[0]
            time = dateutil.parser.parse(date_str, tzinfos=tzinfos)
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
            # line=MONITORING, [Sun Mar 24 21:35:30 UTC 2024], %CPU: 52.10, %Memory: 21.00, IO_rKb/s: 41.00, IO_wKb/s: 367.00, %IOWait: 0.00  # noqa: E501
            split_line = line.split("MONITORING")[1].split(",")
            # Convert the date string: Tue Feb 13 15:33:28 IST 2024
            date_str = split_line[1].split("[")[1].split("]")[0]
            time = dateutil.parser.parse(date_str, tzinfos=tzinfos)
            try:
                cpu = float(split_line[2].split(":")[1] or 0)
            except (ValueError, IndexError):
                cpu = 0.0
            try:
                memory = float(split_line[3].split(":")[1] or 0)
            except (ValueError, IndexError):
                memory = 0.0
            try:
                io_rkb = float(split_line[4].split(":")[1] or 0)
            except (ValueError, IndexError):
                io_rkb = 0.0
            try:
                io_wkb = float(split_line[5].split(":")[1] or 0)
            except (ValueError, IndexError):
                io_wkb = 0.0
            try:
                iowait = float(split_line[6].split(":")[1] or 0)
            except (ValueError, IndexError):
                iowait = 0.0

            if self.start_time is None:
                self.start_time = time

            self.run_time = time - self.start_time

            # Create a new DataFrame
            new_data = pd.DataFrame(
                {
                    MonitorColumns.TIME.value: [time],
                    MonitorColumns.CPU.value: [cpu],
                    MonitorColumns.MEMORY.value: [memory],
                    MonitorColumns.IO_RKB.value: [io_rkb],
                    MonitorColumns.IO_WKB.value: [io_wkb],
                    MonitorColumns.IOWAIT.value: [iowait],
                }
            )

            # Drop empty or all-NA columns
            new_data = new_data.dropna(axis=1, how="all")

            # Ensure the DataFrame is not empty after dropping columns
            if not new_data.empty:
                if self.df.empty:
                    self.df = new_data
                else:
                    self.df = pd.concat([self.df, new_data], ignore_index=True)


def performance(run_id, session=None, output_dir=None, output_prefix="") -> tuple[pd.DataFrame, RunCost]:
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    total_performance_df = pd.DataFrame()

    # Get run info from omics
    if session:
        omics_client = session.client("omics")
    else:
        omics_client = boto3.client("omics")
    run = get_run_info(run_id, client=omics_client)

    scattered_tasks = set()
    monitor_logs = []

    # Process monitor log for each task
    for task in run["tasks"]:
        print("------------------------------------------")
        print(f"Process monitor log for task {task['name']} (taskId: {task['taskId']})")
        monitor_log = process_monitor_log(run_id, task, client=session.client("logs"))
        monitor_logs.append(monitor_log)

        new_row = pd.DataFrame(
            {
                "task": [task["name"]],
                "total_CPU": [monitor_log.total_cpu],
                "mean_%_CPU": [monitor_log.df[MonitorColumns.CPU.value].mean()],
                "max_%_CPU": [monitor_log.df[MonitorColumns.CPU.value].max()],
                "total_Memory(GB)": [monitor_log.total_memory],
                "mean_%_Memory": [monitor_log.df[MonitorColumns.MEMORY.value].mean()],
                "max_%_Memory": [monitor_log.df[MonitorColumns.MEMORY.value].max()],
                "run_time (hours)": [monitor_log.run_time.total_seconds() / 3600],
                "mean_IO_rKb/s": [monitor_log.df[MonitorColumns.IO_RKB.value].mean()],
                "max_IO_rKb/s": [monitor_log.df[MonitorColumns.IO_RKB.value].max()],
                "mean_IO_wKb/s": [monitor_log.df[MonitorColumns.IO_WKB.value].mean()],
                "max_IO_wKb/s": [monitor_log.df[MonitorColumns.IO_WKB.value].max()],
                "mean_IOWait": [monitor_log.df[MonitorColumns.IOWAIT.value].mean()],
                "max_IOWait": [monitor_log.df[MonitorColumns.IOWAIT.value].max()],
                "count_entries": [monitor_log.df[MonitorColumns.CPU.value].count()],
            }
        )
        total_performance_df = pd.concat([total_performance_df, new_row], ignore_index=True)

        # check if the task is a scattered task
        if "-" in task["name"]:
            scattered_tasks.add(task["name"].split("-")[0])

    # Process cost and add to the performance data
    print("Add cost per task to performance data")
    run_cost = RunCost(run_id, output_dir=output_dir, output_prefix=output_prefix)
    cost_df = run_cost.get_tasks_cost()
    cost_df = cost_df.rename(
        columns={
            Columns.NAME_COLUMN.value: "task",
            Columns.ESTIMATED_USD_COLUMN.value: "cost",
            Columns.OMICS_INSTANCE_TYPE_RESERVED.value: "instance",
        }
    )
    cost_df["total_storage_cost"] = run_cost.get_storage_cost()
    total_performance_df = total_performance_df.merge(cost_df, on="task", how="left")

    # Save performance data
    output = f"{output_prefix}omics_{run_id}.performance.csv"
    if output_dir is not None:
        output = f"{output_dir}/{output}"
    print(f"Saving performance data to: {output}")
    total_performance_df.to_csv(output, index=False)

    # Save figures to HTML
    save_figures_to_html(monitor_logs, run_id, scattered_tasks, output_dir, output_prefix)

    return total_performance_df, run_cost


def process_monitor_log(run_id, task, client=None) -> MonitorLog:
    if not client:
        client = boto3.client("logs")

    task_id = task["taskId"]

    log_stream_name = f"run/{run_id}/task/{task_id}"
    log_group_name = OMICS_LOG_GROUP

    monitor_log = MonitorLog()
    monitor_log.task_name = task["name"]
    monitor_log.task_id = task_id

    print(f"Get log events for log group '{log_group_name}' and log stream '{log_stream_name}'")

    # Get log events of specific task from CloudWatch that contain the word 'MONITORING'
    response = client.filter_log_events(
        logGroupName=log_group_name, logStreamNames=[log_stream_name], filterPattern="MONITORING"
    )

    # check that log is not empty
    if not response.get("events"):
        print(f"No events found for log group '{log_group_name}' and log stream '{log_stream_name}'")
        return monitor_log

    # process monitoring log events
    for event in response["events"]:
        monitor_log.process_line(event["message"])

    # get next page of log events
    while len(response.get("events")) > 0 and response.get("nextToken"):
        response = client.filter_log_events(
            logGroupName=log_group_name,
            logStreamNames=[log_stream_name],
            filterPattern="MONITORING",
            nextToken=response.get("nextToken"),
        )

        for event in response["events"]:
            monitor_log.process_line(event["message"])

    print(f"Done processing monitor log for task {task_id}")
    return monitor_log


def save_figures_to_html(monitor_logs, run_id, scattered_tasks, output_dir=None, output_prefix=""):  # noqa: C901, PLR0912, PLR0915 #TODO: refactor this function
    print("Generate and save performance plots to HTML...")
    dont_plot_tasks = []
    plot_tasks = []

    for monitor_log in monitor_logs:
        # Monitor log script prints every 10 seconds.
        # If monitor_log.df has less than 30 entires it means it worked less than 5 minutes and we don't need
        # the plot for this task.
        if monitor_log.df.shape[0] < 30:  # noqa: PLR2004
            print(
                f"Task: {monitor_log.task_name} (id:{monitor_log.task_id}) run less than 5 minutes "
                "and will not be plotted"
            )
            dont_plot_tasks.append(monitor_log)
        elif not any(
            task in monitor_log.task_name for task in scattered_tasks
        ):  # check if the task name is not a subtring of a scattered task
            plot_tasks.append(monitor_log)

    # TODO: fix this logic to use 5 columns only if scattered tasks have IO metrics
    if len(scattered_tasks) > 0:
        if add_io_plots(plot_tasks):
            cols = 5  # for scattered tasks plot cpu, memory, IO_read, IO_write, IO_wait
        else:
            cols = 2  # for scattered tasks plot cpu and memory
    else:
        cols = 1  # no scattered tasks, each row will contain one plot

    # Combine all figures into one HTML file
    rows = len(plot_tasks) + len(scattered_tasks)  # Each figure will be in its own row

    # regular tasks span over all columns, for the scattered tasks, each coulmn will have a plot
    specs = []
    for _ in range(len(plot_tasks)):
        # E.g., if col=5, it will result to [{'colspan': 5}, None, None, None, None] for each row,
        # meaning the first column will span all 5 columns
        specs.append([{"colspan": cols}] + [None] * (cols - 1))
    for _ in range(len(scattered_tasks)):
        # E.g., if col=5, it will result to [{}, {}, {}, {}, {}] for each row, meaning each column will have a plot
        specs.append([{} for _ in range(cols)])

    subplot_titles = [f"Task: {monitor_log.task_name} (id:{monitor_log.task_id})" for monitor_log in plot_tasks]
    combined_fig = make_subplots(rows=rows, cols=cols, specs=specs, subplot_titles=subplot_titles)

    add_to_legend = set()

    for i, monitor_log in enumerate(plot_tasks, start=1):
        print(f"Plotting task: {monitor_log.task_name} (id:{monitor_log.task_id})")
        # Add traces for each metric
        add_trace(combined_fig, monitor_log, MonitorColumns.CPU.value, i)
        add_trace(combined_fig, monitor_log, MonitorColumns.MEMORY.value, i)
        # Add IO metrics only if the coresepong coulmns are not empty and sum is not 0
        if (
            monitor_log.df[MonitorColumns.IO_RKB.value].notna().any()
            and monitor_log.df[MonitorColumns.IO_RKB.value].sum() != 0
        ):
            add_trace(combined_fig, monitor_log, MonitorColumns.IO_RKB.value, i, normelize=True)
            add_to_legend.add(MonitorColumns.IO_RKB.value)
        if (
            monitor_log.df[MonitorColumns.IO_WKB.value].notna().any()
            and monitor_log.df[MonitorColumns.IO_WKB.value].sum() != 0
        ):
            add_trace(combined_fig, monitor_log, MonitorColumns.IO_WKB.value, i, normelize=True)
            add_to_legend.add(MonitorColumns.IO_WKB.value)
        if (
            monitor_log.df[MonitorColumns.IOWAIT.value].notna().any()
            and monitor_log.df[MonitorColumns.IOWAIT.value].sum() != 0
        ):
            add_trace(combined_fig, monitor_log, MonitorColumns.IOWAIT.value, i)
            add_to_legend.add(MonitorColumns.IOWAIT.value)

        # Add caption for each subplot
        combined_fig.add_annotation(
            text=(
                f"#CPU: {monitor_log.total_cpu}, Memory(Gib): {monitor_log.total_memory}, "
                f"runtime(H): {monitor_log.run_time.total_seconds()/3600:.2f}"
            ),
            xref=f"x{i if i != 1 else ''} domain",  # Reference the x-axis of the ith subplot
            yref=f"y{i if i != 1 else ''} domain",  # Reference the y-axis of the ith subplot
            x=0.5,  # Position the caption in the middle of the subplot horizontally
            y=-0.15,  # Position the caption just below the subplot
            showarrow=False,
            font={"size": 12},
            align="center",
            xanchor="center",
            yanchor="top",
        )

    # Add missing lables in the legend by adding invisible traces
    for label in add_to_legend:
        combined_fig.add_trace(
            go.Scatter(x=[None], y=[None], mode="lines", name=label),
            row=1,
            col=1,  # Add it to the first subplot where the leggend appears
        )

    # Add plots for scattered tasks
    for i, task in enumerate(scattered_tasks, start=len(plot_tasks) + 1):
        print(f"Plotting scattered task: {task}")
        scattered_task_monitor_logs = [log for log in monitor_logs if task in log.task_name]

        # Itereate over tasks within a single scattered task and add metrics to the plots
        for _, monitor_log in enumerate(scattered_task_monitor_logs, start=1):
            add_trace(
                combined_fig,
                monitor_log,
                MonitorColumns.CPU.value,
                row=i,
                col=1,
                normelize_time=True,
                name=monitor_log.task_name,
            )
            add_trace(
                combined_fig,
                monitor_log,
                MonitorColumns.MEMORY.value,
                row=i,
                col=2,
                normelize_time=True,
                name=monitor_log.task_name,
            )

            if (
                monitor_log.df[MonitorColumns.IO_RKB.value].notna().any()
                and monitor_log.df[MonitorColumns.IO_RKB.value].sum() != 0
            ):
                add_trace(
                    combined_fig,
                    monitor_log,
                    MonitorColumns.IO_RKB.value,
                    row=i,
                    col=3,
                    normelize=True,
                    normelize_time=True,
                    name=monitor_log.task_name,
                )

            if (
                monitor_log.df[MonitorColumns.IO_WKB.value].notna().any()
                and monitor_log.df[MonitorColumns.IO_WKB.value].sum() != 0
            ):
                add_trace(
                    combined_fig,
                    monitor_log,
                    MonitorColumns.IO_WKB.value,
                    row=i,
                    col=4,
                    normelize=True,
                    normelize_time=True,
                    name=monitor_log.task_name,
                )

            if (
                monitor_log.df[MonitorColumns.IOWAIT.value].notna().any()
                and monitor_log.df[MonitorColumns.IOWAIT.value].sum() != 0
            ):
                add_trace(
                    combined_fig,
                    monitor_log,
                    MonitorColumns.IOWAIT.value,
                    row=i,
                    col=5,
                    normelize=False,
                    normelize_time=True,
                    name=monitor_log.task_name,
                )

        # Add caption for each subplot
        cpu = scattered_task_monitor_logs[0].total_cpu
        memory = scattered_task_monitor_logs[0].total_memory
        average_runtime = (
            sum([monitor_log.run_time.total_seconds() for monitor_log in scattered_task_monitor_logs])
            / len(scattered_task_monitor_logs)
            / 3600
        )
        combined_fig.add_annotation(
            text=(
                f"Scattered Task: {task} #CPU: {cpu}, Memory(Gib): {memory}, "
                f"average runtime(H): {average_runtime:.2f}"
            ),
            xref="paper",
            yref="paper",
            x=0.5,  # Middle of the row
            y=(1 - (i - 0.8) / rows),  # Top of the row
            showarrow=False,
            font={"size": 12},
            align="center",
            xanchor="center",
            yanchor="auto",
        )

    # touch up the layout and add title and subtext
    combined_fig.update_yaxes(title="%", range=[0, 110])
    combined_fig.update_layout(
        height=300 * rows + 50, title_text=f"Omics {run_id} Performance Plots", title_font={"size": 20}
    )
    if len(dont_plot_tasks) > 0:
        task_runtimes = [
            f"{monitor_log.task_name} (runtime (H): {monitor_log.run_time.total_seconds()/3600:.2f})"
            for monitor_log in dont_plot_tasks
        ]
        combined_fig.add_annotation(
            text=(f"Short tasks without a plot: {task_runtimes}"),
            xref="paper",  # Position relative to the entire plotting area
            yref="paper",  # Position relative to the entire plotting area
            x=0,  # Center the text horizontally
            y=1,  # Adjust this value as needed to position below the title
            showarrow=False,
            font={"size": 12},
            yanchor="bottom",
            yshift=25,  # Shift down by 25 pixels to add more "padding"
        )

    # Save the combined figure to an HTML file
    plots_dir = PLOTS_DIR.replace(RUN_ID_PLACEHOLDER, run_id)
    combined_fig_output = f"{plots_dir}/{output_prefix}omics_{run_id}_performance_plots.html"
    if output_dir is not None:
        combined_fig_output = f"{output_dir}/{combined_fig_output}"
    print(f"Saving performance plots report to: {combined_fig_output}")
    combined_fig.write_html(combined_fig_output)


def add_io_plots(monitor_logs) -> bool:
    for monitor_log in monitor_logs:
        if (
            monitor_log.df[MonitorColumns.IO_RKB.value].notna().any()
            and monitor_log.df[MonitorColumns.IO_RKB.value].sum() != 0
        ):
            return True
        if (
            monitor_log.df[MonitorColumns.IO_WKB.value].notna().any()
            and monitor_log.df[MonitorColumns.IO_WKB.value].sum() != 0
        ):
            return True
        if (
            monitor_log.df[MonitorColumns.IOWAIT.value].notna().any()
            and monitor_log.df[MonitorColumns.IOWAIT.value].sum() != 0
        ):
            return True
    return False


def add_trace(combined_fig, monitor_log, col_name, row, col=1, name=None, *, normelize=False, normelize_time=False):
    show_legend = row == 1

    if normelize:
        monitor_log.df[col_name] = monitor_log.df[col_name] / monitor_log.df[col_name].max() * 100
    if normelize_time:
        monitor_log.df[MonitorColumns.TIME.value] = (
            monitor_log.df[MonitorColumns.TIME.value] - monitor_log.df[MonitorColumns.TIME.value].min()
        )

    if not name:
        name = col_name

    combined_fig.add_trace(
        go.Scatter(
            x=monitor_log.df[MonitorColumns.TIME.value],
            y=monitor_log.df[col_name],
            mode="lines",
            name=name,
            line={"color": monitor_log.colors[col_name]},
            showlegend=show_legend,
        ),
        row=row,
        col=col,
    )


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--region", type=str, help="AWS region to use", default="us-east-1")
    parser.add_argument("--run-id", type=str, help="HealthOmics workflow run-id to analyze")
    parser.add_argument("--output-path", type=str, help="Output dir to save performance data", required=False)
    parser.add_argument("--output-prefix", type=str, help="File name prefix for the output", required=False, default="")

    args = parser.parse_args()
    session = boto3.Session(region_name=args.region)

    performance(args.run_id, session=session, output_dir=args.output_path, output_prefix=args.output_prefix)
