import json
import os
from argparse import ArgumentParser

import boto3
import dateutil.parser
import pandas as pd
from google.cloud import storage

from ugbio_omics.cromwell_calculate_cost import calculate_cost
from ugbio_omics.get_performance import performance as omics_performance
from ugbio_omics.get_run_cost import Columns, RunCost


def compare_cromwell_omics(
    cromwell_wid, omics_run_id, omics_session, workflow_name, output_path, *, overwrite=False, get_performance=False
):
    # Create the output directory if it doesn't exist
    if output_path:
        os.makedirs(output_path, exist_ok=True)
    else:
        output_path = os.getcwd()

    # Get cromwell performance and metadata files
    cromwell_performance_file, cromwell_metadata_file = copy_cromwell_data(
        workflow_name, cromwell_wid, output_path, overwrite=overwrite
    )
    # Calculate cromwell cost
    cromwell_cost_df = cromwell_cost(cromwell_metadata_file, output_path, workflow_name, cromwell_wid)

    # Calculate cromwell performance
    cromwell_performance_df = cromwell_performance(cromwell_performance_file)

    # Calculate omics cost (and performance if requested)
    omics_cost_df, omics_run_cost = get_omics_cost_perfromance(
        omics_run_id, omics_session, output_path, get_performance=get_performance
    )

    # Compare cromwell and omics cost
    print("Comparing cromwell and omics...")
    cost_df = pd.DataFrame(omics_cost_df, columns=["task", "cost"])
    cost_df = cost_df.merge(cromwell_cost_df[["task", "compute_cost"]], on="task", how="outer")
    cost_df = cost_df.rename(columns={"cost": "omics_cost", "compute_cost": "cromwell_cost"})

    # add cost difference
    cost_df["cost_diff"] = cost_df["omics_cost"] - cost_df["cromwell_cost"]
    # move total cost to the end
    cost_df = pd.concat([cost_df[cost_df.task != "total"], cost_df[cost_df.task == "total"]], ignore_index=True)

    # Compare cromwell and omics run duration
    cromwell_total_duration = get_cromwell_total_duration(cromwell_metadata_file)
    cromwell_performance_df = pd.concat(
        [cromwell_performance_df, pd.DataFrame({"task": ["total"], "run_time (hours)": [cromwell_total_duration]})],
        ignore_index=True,
    )

    duration_df = pd.DataFrame(omics_cost_df, columns=["task", "run_time (hours)"])
    duration_df = duration_df.merge(cromwell_performance_df[["task", "run_time (hours)"]], on="task", how="outer")
    duration_df = duration_df.rename(
        columns={"run_time (hours)_x": "omics_duration", "run_time (hours)_y": "cromwell_duration"}
    )
    # add duration difference
    duration_df["duration_diff"] = duration_df["omics_duration"] - duration_df["cromwell_duration"]

    # Add resources for omics and cromwell
    resources_df = extract_omics_resources(omics_run_cost)

    # Merge cromwell_cost_df['cpus','mem_gbs'] into one column as a dictionary
    cromwell_resources = cromwell_cost_df[["cpus", "mem_gbs"]].apply(
        lambda x: {"cpus": x["cpus"], "mem_gbs": x["mem_gbs"]}, axis=1
    )
    cromwell_cost_df["cromwell_resources"] = cromwell_resources
    resources_df = resources_df.merge(cromwell_cost_df[["task", "cromwell_resources"]], on="task", how="outer")

    # Merge cost, duration and resources and save to file
    compare_df = cost_df.merge(duration_df, on="task", how="outer")
    compare_df = compare_df.merge(resources_df, on="task", how="outer")
    compare_file = f"{output_path}/compare_omics_{omics_run_id}_cromwell_{cromwell_wid}.csv"
    compare_df.to_csv(compare_file, index=False, float_format="%.3f")
    print(f"Comparison saved in: {compare_file}")


def extract_omics_resources(omics_run_cost: RunCost) -> pd.DataFrame:
    resources_df = omics_run_cost.get_tasks_resources()
    # Merge cpus, mem_gbs, gpus into one column as a dictionary
    resources_df["omics_resources"] = resources_df.apply(
        lambda row: {
            "cpus": row[Columns.CPU_REQUESTED.value],
            "mem_gbs": row[Columns.MEMORY_REQUESTED_GIB.value],
            "gpus": row[Columns.GPUS_REQUESTED.value],
        },
        axis=1,
    )
    resources_df = resources_df.drop(
        columns=[Columns.CPU_REQUESTED.value, Columns.MEMORY_REQUESTED_GIB.value, Columns.GPUS_REQUESTED.value],
    )
    resources_df = resources_df.rename(
        columns={Columns.NAME_COLUMN.value: "task", Columns.OMICS_INSTANCE_TYPE_RESERVED.value: "omics_instance"},
    )
    resources_df["task"] = resources_df["task"].str.split("-").str[0]
    resources_df = resources_df.groupby("task").first().reset_index()
    return resources_df


def copy_cromwell_data(workflow_name, cromwell_wid, output_path, *, overwrite=False):
    # Get cromwell bucket for the workflow
    print(f"Get cromwell performance file for workflow: {workflow_name}/{cromwell_wid}")
    cromwell_client = storage.Client()
    bucket_name = "cromwell-backend-ultima-data-307918"
    bucket = cromwell_client.get_bucket(bucket_name)

    # save performance.csv as local file
    performance_file = f"{output_path}/cromwell_{cromwell_wid}.performance.csv"
    if os.path.exists(performance_file) and not overwrite:
        print(f"Skipping download. Cromwell performance file already exists: {performance_file}")
    else:
        blob = bucket.get_blob(f"cromwell-execution/{workflow_name}/{cromwell_wid}/performance.csv")
        if not blob:
            raise Exception(f"Performance file not found for workflow: {workflow_name}/{cromwell_wid}")
        with open(performance_file, "w") as f:
            f.write(blob.download_as_text(encoding="utf-8"))
        print(f"Cromwell performance saved in: {performance_file}")

    # save metadata.json as local file
    print(f"Get cromwell metadata file for workflow: {workflow_name}/{cromwell_wid}")
    metadata_file = f"{output_path}/cromwell_{cromwell_wid}.metadata.json"
    if os.path.exists(metadata_file) and not overwrite:
        print(f"Skipping download. Cromwell metadata file already exists: {metadata_file}")
    else:
        blob = bucket.get_blob(f"cromwell-execution/{workflow_name}/{cromwell_wid}/metadata.json")
        if not blob:
            raise Exception(f"Metadata file not found for workflow: {workflow_name}/{cromwell_wid}")
        with open(metadata_file, "w") as f:
            f.write(blob.download_as_text(encoding="utf-8"))
        print(f"Cromwell metadata saved in: {metadata_file}")

    return performance_file, metadata_file


def cromwell_cost(metadata_file, output_path, workflow_name, cromwell_wid):
    # run the calculate_cost.py script from terra_pipeline
    print(f"Calculating cromwell cost for workflow: {workflow_name}/{cromwell_wid}")

    cromwell_cost_file = f"{output_path}/cromwell_{cromwell_wid}.cost.csv"

    with open(metadata_file) as data_file:
        metadata = json.load(data_file)

    try:
        calculate_cost(
            metadata=metadata,
            ignore_preempted=False,
            only_total_cost=False,
            print_header=True,
            output_file=cromwell_cost_file,
        )
    except Exception as e:
        print(f"Error occurred while calculating cromwell cost: {e}")
        raise e

    print(f"Cromwell cost saved in: {cromwell_cost_file}")

    # load cromwell cost into a dataframe and clean up the data
    cromwell_cost_df = pd.read_csv(cromwell_cost_file)
    cromwell_cost_df = cromwell_cost_df.rename(columns=lambda x: x.strip())
    # if there is a any row similiar to the column names- delete that row from df and re-save the file
    cromwell_cost_df = cromwell_cost_df[~cromwell_cost_df["task_name"].str.contains("task_name")]
    # convert all numeric columns to numeric
    for col in cromwell_cost_df.columns:
        if "cost" in col:
            cromwell_cost_df[col] = pd.to_numeric(cromwell_cost_df[col], errors="coerce")
    # numeric_columns = ["cpu_cost", "pe_cpu_cost", "mem_cost", "pe_mem_cost", "gpu_cost", "pe_gpu_cost", "disk_cost"]
    # cromwell_cost_df[numeric_columns] = cromwell_cost_df[numeric_columns].apply(pd.to_numeric, errors='coerce')

    # add compute cost as an additional column
    cromwell_cost_df["compute_cost"] = (
        cromwell_cost_df["cpu_cost"]
        + cromwell_cost_df["pe_cpu_cost"]
        + cromwell_cost_df["mem_cost"]
        + cromwell_cost_df["pe_mem_cost"]
        + cromwell_cost_df["gpu_cost"]
        + cromwell_cost_df["pe_gpu_cost"]
    )
    cromwell_cost_df["task"] = cromwell_cost_df["task_name"].str.split(".").str[1]

    cromwell_disk_cost = cromwell_cost_df["disk_cost"].sum()

    # add storage and total cost as an additional row
    cromwell_total_cost = cromwell_cost_df["total_cost"].sum()
    cromwell_cost_df = pd.concat(
        [
            cromwell_cost_df,
            pd.DataFrame({"task": ["storage", "total"], "compute_cost": [cromwell_disk_cost, cromwell_total_cost]}),
        ],
        ignore_index=True,
    )

    return cromwell_cost_df


def cromwell_performance(cromwell_performance_file):
    # load and clean cromwell performance into a dataframe
    performance_df = pd.read_csv(cromwell_performance_file)
    performance_df = performance_df.rename(columns=lambda x: x.strip())
    performance_df["task"] = performance_df["task"].str.strip()
    numeric_columns = performance_df.columns.drop(["task"])
    performance_df[numeric_columns] = performance_df[numeric_columns].apply(pd.to_numeric, errors="coerce")

    # Drop multiple preemtible attempts and keep the last one
    performance_df["attempt"] = performance_df["task"].str.extract(r"attempt-(\d+)", expand=False).astype(float)
    performance_df["task"] = performance_df["task"].str.split("/attempt-", expand=True)[0]
    performance_df = performance_df.sort_values("attempt", ascending=False).drop_duplicates(subset="task", keep="first")
    performance_df = performance_df.drop(columns="attempt")

    # group by task name and calculate mean of performance metrics
    performance_df["task"] = performance_df["task"].str.split("/").str[0]
    grouped_df = performance_df.groupby("task").mean().reset_index()

    # remove prefix "call-" from task name
    grouped_df["task"] = grouped_df["task"].str.split("call-").str[1]

    return grouped_df


def get_cromwell_total_duration(cromwell_metadata_file):
    with open(cromwell_metadata_file) as f:
        metadata = json.load(f)
    end_time = dateutil.parser.parse(metadata["end"])
    start_time = dateutil.parser.parse(metadata["start"])
    total_duration = end_time - start_time
    return total_duration.total_seconds() / 3600


def get_omics_cost_perfromance(
    omics_run_id, session, output_path, *, get_performance=False
) -> tuple[pd.DataFrame, float, RunCost]:
    print(f"Calculating omics cost for run: {omics_run_id}")

    # Get run cost
    run_cost = RunCost(omics_run_id, output_dir=output_path)
    cost_df = run_cost.get_run_cost()

    # Calculate performance
    if get_performance:
        print(f"Calculating omics performance for run: {omics_run_id}")
        try:
            omics_performance(omics_run_id, session=session, output_dir=output_path)
        except Exception as e:
            print(f"Error when trying to calculate omics perfroamnce. Details: {e}")
            raise

    return cost_df, run_cost


def main():
    parser = ArgumentParser()
    # cromwell args
    parser.add_argument("--cromwell-wid", type=str, help="Cromwell workflow id to analyze")

    # omics args
    parser.add_argument("--region", type=str, help="AWS region to use", default="us-east-1")
    parser.add_argument("--omics-run-id", type=str, help="HealthOmics workflow run-id to analyze")

    # general args
    parser.add_argument("--workflow-name", type=str, help="Workflow name e.g. EfficientDV")
    parser.add_argument(
        "--output-path",
        type=str,
        help="Output path for all files copied and generated during the analysis",
        default=None,
    )
    parser.add_argument(
        "--performance",
        type=bool,
        help="Get CPU and memory performance from the monitor log (work only if task logs still accessbile)",
        default=False,
    )
    parser.add_argument("--overwrite", type=bool, help="Overwrite downloaded files from cloud", default=False)

    args = parser.parse_args()
    session = boto3.Session(region_name=args.region)

    compare_cromwell_omics(
        args.cromwell_wid,
        args.omics_run_id,
        session,
        args.workflow_name,
        args.output_path,
        overwrite=args.overwrite,
        get_performance=args.performance,
    )


if __name__ == "__main__":
    main()

# output_path = "/home/inbalzelig/data/omics/somatic_efficientdv"
# workflow_name = "EfficientDV"
# cromwell_wid = "31b29c63-f7b2-4a7c-986a-64171126d9c4"
# omics_run_id = "2068853"
