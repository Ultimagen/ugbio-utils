import os
import subprocess
from enum import Enum

import pandas as pd

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


class RunCost:
    def __init__(self, run_id, output_dir=None, output_prefix=""):
        self.run_id = run_id
        self.output_dir = output_dir
        if self.output_dir:
            os.makedirs(output_dir, exist_ok=True)
        self.output_prefix = output_prefix
        self.cost_csv = None
        self.cost_df = None

    def calculate_run_cost(self) -> None:
        cost_output = f"{self.output_prefix}omics_{self.run_id}.cost.csv"
        plots_dir = PLOTS_DIR.replace(RUN_ID_PLACEHOLDER, self.run_id)

        if self.output_dir:
            cost_output = f"{self.output_dir}/{cost_output}"
            plots_dir = f"{self.output_dir}/{plots_dir}"

        self.run_analyzer(self.run_id, cost_output, plots_dir)
        print(f"Run cost calculated and saved to {cost_output}")
        self.cost_csv = cost_output
        self.cost_df = pd.read_csv(self.cost_csv)

    def get_total_cost(self) -> float:
        if self.cost_df is None:
            self.calculate_run_cost()

        return self.cost_df[Columns.ESTIMATED_USD_COLUMN.value].sum()

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
            [Columns.NAME_COLUMN.value, Columns.ESTIMATED_USD_COLUMN.value, Columns.OMICS_INSTANCE_TYPE_RESERVED.value]
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

    @staticmethod
    def run_analyzer(run_id, run_csv_name, plots_dir):
        analyzer_command = ["python", "-m", "omics.cli.run_analyzer", run_id, "-o", run_csv_name, "-P", plots_dir]

        subprocess.run(
            analyzer_command,
            check=True,
        )
