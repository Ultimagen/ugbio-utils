import os
from argparse import ArgumentParser

import boto3
import pandas as pd

from ugbio_omics.compare_cromwell_omics import (
    extract_omics_resources,
    get_omics_cost_perfromance,
)


def single_run(omics_run_id, omics_session, output_path, *, get_performance=False):
    # Calculate omics performance and cost
    omics_cost_df, omics_run_cost = get_omics_cost_perfromance(
        omics_run_id, omics_session, output_path, get_performance=get_performance
    )
    # Rename columns with run_id
    omics_cost_df = omics_cost_df.rename(
        columns={"cost": f"{omics_run_id}_cost", "run_time (hours)": f"{omics_run_id}_duration(H)"}
    )

    # Drop the 'instance' column from omics_cost_df
    if "instance" in omics_cost_df.columns:
        omics_cost_df = omics_cost_df.drop(columns=["instance"])

    # move total cost to the end
    omics_cost_df = pd.concat(
        [omics_cost_df[omics_cost_df.task != "total"], omics_cost_df[omics_cost_df.task == "total"]], ignore_index=True
    )

    # Add resources
    resources_df = extract_omics_resources(omics_run_cost)
    resources_df = resources_df.rename(
        columns={"omics_resources": f"{omics_run_id}_resources", "omics_instance": f"{omics_run_id}_instance"}
    )

    # Merge cost, duration and resources
    final_df = omics_cost_df.merge(resources_df, on="task", how="outer")
    return final_df


def compare_omics_runs(run_ids, session, output_path, *, get_performance=False):
    # Create the output directory if it doesn't exist
    if output_path:
        os.makedirs(output_path, exist_ok=True)
    else:
        output_path = os.getcwd()

    all_df = []
    for run_id in run_ids:
        single_df = single_run(run_id, session, output_path, get_performance=get_performance)
        all_df.append(single_df)

    final_df = all_df[0]
    for single_df in all_df[1:]:
        final_df = final_df.merge(single_df, on="task", how="outer")

    compare_file = f"{output_path}/compare_omics_runs.csv"
    final_df.to_csv(compare_file, index=False, float_format="%.3f")
    print(f"Comparison saved in: {compare_file}")


def main():
    parser = ArgumentParser()
    parser.add_argument("--region", type=str, help="AWS region to use", default="us-east-1")
    parser.add_argument(
        "--output-path",
        type=str,
        help="Output path for all files copied and generated during the analysis",
        default=None,
    )
    parser.add_argument("--run-ids", type=str, help="Omics run ids to compare (seprated by comma)", required=True)
    parser.add_argument(
        "--performance",
        type=bool,
        help="Get CPU and memory performance from the monitor log (work only if task logs still accessbile)",
        default=False,
    )

    args = parser.parse_args()
    session = boto3.Session(region_name=args.region)
    run_ids = args.run_ids.split(",")

    compare_omics_runs(run_ids, session, args.output_path, get_performance=args.performance)


if __name__ == "__main__":
    main()
