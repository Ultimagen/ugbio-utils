import os
from enum import Enum

import boto3
import botocore
import pandas as pd

from ugbio_omics.get_run_info import get_run_info

RUN_ID_PLACEHOLDER = "<RUN_ID>"
PLOTS_DIR = f"omics_{RUN_ID_PLACEHOLDER}_plots"


class Columns(Enum):
    ESTIMATED_USD_COLUMN = "estimatedUSD"
    TYPE_COLUMN = "type"
    NAME_COLUMN = "name"
    OMICS_INSTANCE_TYPE_RESERVED = "omicsInstanceTypeReserved"
    CPU_REQUESTED = "cpusRequested"
    MEMORY_REQUESTED_GIB = "memoryRequestedGiB"
    GPUS_REQUESTED = "gpusRequested"
    RUNNING_SECONDS = "runningSeconds"


class RunCost:
    def __init__(self, run_id, output_dir=None, output_prefix="", session=None):
        self.run_id = run_id
        self.output_dir = output_dir
        if self.output_dir:
            os.makedirs(output_dir, exist_ok=True)
        self.output_prefix = output_prefix
        self.session = session
        self.cost_csv = None
        self.cost_df = None

    def calculate_run_cost(self) -> None:
        cost_output = f"{self.output_prefix}omics_{self.run_id}.cost.csv"
        plots_dir = PLOTS_DIR.replace(RUN_ID_PLACEHOLDER, self.run_id)

        if self.output_dir:
            cost_output = f"{self.output_dir}/{cost_output}"
            plots_dir = f"{self.output_dir}/{plots_dir}"

        os.makedirs(plots_dir, exist_ok=True)

        if self.session:
            omics_client = self.session.client("omics")
            s3_client = self.session.client("s3")
        else:
            omics_client = boto3.client("omics")
            s3_client = boto3.client("s3")

        # Get bucket name and path for run outputs
        run = get_run_info(self.run_id, client=omics_client, get_tasks=False)
        bucket_name, path = self.get_logs_key(run)

        # Download cost.csv from s3 run outputs
        key = f"{path}/run-{self.run_id}.csv"
        cost_csv_uri = f"s3://{bucket_name}/{key}"
        print(f"Run cost downaloaded from {cost_csv_uri} and saved to {cost_output}")
        with open(cost_output, "wb") as f:
            s3_client.download_fileobj(bucket_name, key, f)

        # Download timeline plot from s3 run outputs (if avaliable)
        key = f"{path}/plots/{self.run_id}_timeline.html"
        timeline_plots_uri = f"s3://{bucket_name}/{key}"
        plots_output = f"{plots_dir}/{self.run_id}_timeline.html"
        print(f"Timeline plot downaloaded from {timeline_plots_uri} and saved to {plots_output}")
        with open(plots_output, "wb") as f:
            try:
                s3_client.download_fileobj(bucket_name, key, f)
            except botocore.exceptions.ClientError as e:
                print(f"Error downloading timeline plot: {e}")
                print(f"Skipping timeline plot download for run {self.run_id}")

        self.cost_csv = cost_output
        self.cost_df = pd.read_csv(self.cost_csv)
        # make sure cost and duration are numeric
        for col in [Columns.ESTIMATED_USD_COLUMN.value, Columns.RUNNING_SECONDS.value]:
            self.cost_df[col] = pd.to_numeric(self.cost_df[col], errors="coerce")

    def get_logs_key(self, run):
        run_output_uri = run["outputUri"]
        bucket_name, pipeline = run_output_uri.replace("s3://", "").split("/", 1)
        path = f"{pipeline}/{self.run_id}/logs"
        return bucket_name, path

    def get_total_cost(self) -> float:
        if self.cost_df is None:
            self.calculate_run_cost()

        return self.cost_df[Columns.ESTIMATED_USD_COLUMN.value].sum()

    def get_total_runtime(self) -> float:
        if self.cost_df is None:
            self.calculate_run_cost()

        return self.cost_df[self.cost_df[Columns.TYPE_COLUMN.value] == "run"][Columns.RUNNING_SECONDS.value].to_numpy()[
            0
        ]

    def get_storage_cost(self) -> float:
        if self.cost_df is None:
            self.calculate_run_cost()

        return self.cost_df[self.cost_df[Columns.TYPE_COLUMN.value] == "run"][
            Columns.ESTIMATED_USD_COLUMN.value
        ].to_numpy()[0]

    def get_tasks_cost(self) -> pd.DataFrame:
        if self.cost_df is None:
            self.calculate_run_cost()

        return self.cost_df[self.cost_df[Columns.TYPE_COLUMN.value] == "task"][
            [
                Columns.NAME_COLUMN.value,
                Columns.ESTIMATED_USD_COLUMN.value,
                Columns.OMICS_INSTANCE_TYPE_RESERVED.value,
                Columns.RUNNING_SECONDS.value,
            ]
        ]

    def get_tasks_resources(self) -> pd.DataFrame:
        if self.cost_df is None:
            self.calculate_run_cost()

        return self.cost_df[self.cost_df[Columns.TYPE_COLUMN.value] == "task"][
            [
                Columns.NAME_COLUMN.value,
                Columns.OMICS_INSTANCE_TYPE_RESERVED.value,
                Columns.CPU_REQUESTED.value,
                Columns.MEMORY_REQUESTED_GIB.value,
                Columns.GPUS_REQUESTED.value,
            ]
        ]

    def get_run_cost(self) -> pd.DataFrame:
        tasks_cost_df = self.get_tasks_cost()

        # group by task name, calculate sum of cost and mean of runtime
        tasks_cost_df[Columns.NAME_COLUMN.value] = tasks_cost_df[Columns.NAME_COLUMN.value].str.split("-").str[0]
        grouped_df = (
            tasks_cost_df.groupby(Columns.NAME_COLUMN.value)
            .agg(
                {
                    Columns.ESTIMATED_USD_COLUMN.value: "sum",
                    Columns.RUNNING_SECONDS.value: "mean",
                    Columns.OMICS_INSTANCE_TYPE_RESERVED.value: "first",
                }
            )
            .reset_index()
        )

        # add storage and total cost
        storage_total_df = pd.DataFrame(
            {
                Columns.NAME_COLUMN.value: ["storage", "total"],
                Columns.ESTIMATED_USD_COLUMN.value: [self.get_storage_cost(), self.get_total_cost()],
                Columns.RUNNING_SECONDS.value: [None, self.get_total_runtime()],
            }
        )

        grouped_df = pd.concat([grouped_df, storage_total_df], ignore_index=True)

        # convert runtime in seconds to hours
        grouped_df[Columns.RUNNING_SECONDS.value] = (
            grouped_df[Columns.RUNNING_SECONDS.value] / 60 / 60
        )  # change runtime to hours

        # Rename columns
        grouped_df = grouped_df.rename(
            columns={
                Columns.NAME_COLUMN.value: "task",
                Columns.ESTIMATED_USD_COLUMN.value: "cost",
                Columns.OMICS_INSTANCE_TYPE_RESERVED.value: "instance",
                Columns.RUNNING_SECONDS.value: "run_time (hours)",
            }
        )
        return grouped_df
