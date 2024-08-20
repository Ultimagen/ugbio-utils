import os
import subprocess

import pandas as pd

ESTIMATED_USD_COLUMN = "estimatedUSD"
TYPE_COLUMN = "type"
NAME_COLUMN = "name"
SIZE_RESERVED_COLUMN = "sizeReserved"

class RunCost:
    def __init__(self, run_id, output_dir=None, output_prefix=''):
        self.run_id = run_id
        self.output_dir = output_dir
        if self.output_dir:
            os.makedirs(output_dir, exist_ok=True)
        self.output_prefix = output_prefix
        self.cost_csv = None
        self.cost_df = None

    def calculate_run_cost(self) -> None:
        cost_output = f"{self.output_prefix}omics_{self.run_id}.cost.csv"

        if self.output_dir:
            cost_output = f"{self.output_dir}/{cost_output}"
        
        self.run_analyzer(self.run_id, cost_output)
        print(f"Run cost calculated and saved to {cost_output}")
        self.cost_csv = cost_output
        self.cost_df = pd.read_csv(self.cost_csv)
    
    def get_total_cost(self) -> float:
        if self.cost_df is None:
            self.calculate_run_cost()

        return self.cost_df[ESTIMATED_USD_COLUMN].sum()
    
    def get_storage_cost(self) -> float:
        if self.cost_df is None:
            self.calculate_run_cost()

        return self.cost_df[self.cost_df[TYPE_COLUMN] == 'run'][ESTIMATED_USD_COLUMN].values[0]
    
    def get_tasks_cost(self) -> pd.DataFrame:
        if self.cost_df is None:
            self.calculate_run_cost()
        
        return self.cost_df[self.cost_df[TYPE_COLUMN] == 'task'][[NAME_COLUMN, ESTIMATED_USD_COLUMN, SIZE_RESERVED_COLUMN]]

    @staticmethod
    def run_analyzer(run_id, run_csv_name):
        analyzer_command = [
            'python', '-m', 'omics.cli.run_analyzer',
            run_id,
            "-o", run_csv_name
        ]

        subprocess.run(
            analyzer_command,
            check=True,
        )
