import os
from argparse import ArgumentParser

import boto3
import pandas as pd

from ugbio_omics.compare_cromwell_omics import (
    extract_omics_resources,
    get_omics_performance_cost,
    get_omics_total_duration,
)


def single_run(omics_run_id, omics_session, output_path):
    # Calculate omics performance and cost
    omics_performance_df, omics_disk_cost, omics_run_cost = get_omics_performance_cost(
        omics_run_id, omics_session, output_path
    )

    # fix up cost data
    cost_df = pd.DataFrame(omics_performance_df, columns=["task", "cost_SUM"])
    cost_df = cost_df.rename(columns={"cost_SUM": f"{omics_run_id}_cost"})
    # add disk cost
    cost_df = pd.concat(
        [cost_df, pd.DataFrame({"task": ["disk"], f"{omics_run_id}_cost": [omics_disk_cost]})], ignore_index=True
    )
    # move total cost to the end
    cost_df = pd.concat([cost_df[cost_df.task != "total"], cost_df[cost_df.task == "total"]], ignore_index=True)

    # run duration
    omics_total_duration = get_omics_total_duration(omics_run_id, omics_session)
    omics_performance_df.loc[omics_performance_df["task"] == "total", "run_time (hours)"] = omics_total_duration
    duration_df = pd.DataFrame(omics_performance_df, columns=["task", "run_time (hours)"])
    duration_df = duration_df.rename(columns={"run_time (hours)": f"{omics_run_id}_duration"})

    # Add resources
    resources_df = extract_omics_resources(omics_run_cost)
    resources_df = resources_df.rename(
        columns={"omics_resources": f"{omics_run_id}_resources", "omics_instance": f"{omics_run_id}_instance"}
    )

    # Merge cost, duration and resources
    final_df = cost_df.merge(duration_df, on="task", how="outer")
    final_df = final_df.merge(resources_df, on="task", how="outer")
    return final_df


def compare_omics_runs(run_ids, session, output_path):
    # Create the output directory if it doesn't exist
    if output_path:
        os.makedirs(output_path, exist_ok=True)
    else:
        output_path = os.getcwd()

    all_df = []
    for run_id in run_ids:
        single_df = single_run(run_id, session, output_path)
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

    args = parser.parse_args()
    session = boto3.Session(region_name=args.region)
    run_ids = args.run_ids.split(",")

    compare_omics_runs(run_ids, session, args.output_path)


if __name__ == "__main__":
    main()
