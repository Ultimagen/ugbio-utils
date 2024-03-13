from argparse import ArgumentParser
import json
from google.cloud import storage
import subprocess
import boto3

import pandas as pd
import os
from get_preformance_from_omics_log import performance as omics_performance


def compare_cromwell_omics(cromwell_wid, omics_run_id, omics_session, workflow_name, output_path, overwrite=False):
    # Create the output directory if it doesn't exist
    if output_path:
        os.makedirs(output_path, exist_ok=True)
    else:
        output_path = os.getcwd()

    # Get cromwell performance and metadata files
    cromwell_performance_file, cromwell_metadata_file = copy_cromwell_data(workflow_name, cromwell_wid, output_path, overwrite=overwrite)

    # Calculate cromwell cost
    cromwell_cost_df, cromwell_disk_cost = cromwell_cost(cromwell_metadata_file, output_path, workflow_name, cromwell_wid)

    # Calculate omics cost
    omics_cost_df, omics_disk_cost = omics_cost(omics_run_id, omics_session, output_path)

    # Compare cromwell and omics cost
    print(f"Comparing cromwell and omics cost")
    cost_df = pd.DataFrame(omics_cost_df, columns=["task", "cost"])
    cost_df = cost_df.merge(cromwell_cost_df[['task','compute_cost']], on="task", how="left")
    cost_df = cost_df.rename(columns={"cost": "omics", "compute_cost": "cromwell"})
    # add disk cost
    cost_df = pd.concat([cost_df, pd.DataFrame({"task": ["disk"], "cromwell": [cromwell_disk_cost], "omics":[omics_disk_cost]})], ignore_index=True)
    # add cost difference
    cost_df["cost_diff"] = cost_df["omics"] - cost_df["cromwell"]

    # save the cost comparison to a file
    cost_comparison_file = f"{output_path}/cost_omics_{omics_run_id}_cromwell_{cromwell_wid}.csv"
    cost_df.to_csv(cost_comparison_file, index=False, float_format = '%.3f')
    print(f"Cost comparison saved in: {cost_comparison_file}")



def copy_cromwell_data(workflow_name, cromwell_wid, output_path, overwrite=False):
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
        blob = bucket.get_blob(f'cromwell-execution/{workflow_name}/{cromwell_wid}/performance.csv')
        with open(performance_file, "w") as f:
            f.write(blob.download_as_text(encoding="utf-8"))

    # save metadata.json as local file
    print(f"Get cromwell metadata file for workflow: {workflow_name}/{cromwell_wid}")
    metadata_file = f"{output_path}/cromwell_{cromwell_wid}.metadata.json"
    if os.path.exists(metadata_file) and not overwrite:
        print(f"Skipping download. Cromwell metadata file already exists: {metadata_file}")
    else:
        blob = bucket.get_blob(f'cromwell-execution/{workflow_name}/{cromwell_wid}/metadata.json')
        with open(metadata_file, "w") as f:
            f.write(blob.download_as_text(encoding="utf-8"))
    
    return performance_file, metadata_file

def cromwell_cost(metadata_file, output_path, workflow_name, cromwell_wid):
    # run the calculate_cost.py script from terra_pipeline
    print(f"Calculating cromwell cost for workflow: {workflow_name}/{cromwell_wid}")
    cromwell_cost_file = f"{output_path}/cromwell_{cromwell_wid}.cost.csv"
    command = f"python3 ~/workspace/terra_pipeline/python_libs/utilities/calculate_cost.py -m {metadata_file} > {cromwell_cost_file}"
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    # Print process output in realtime
    while True:
        output = process.stdout.readline()
        if process.poll() is not None:
            break
        if output:
            print(output.strip().decode('utf-8'))
    if process.returncode != 0:
        print(process.stderr.read().decode('utf-8'))
        raise Exception(f"Error occurred while calculating cromwell cost")
    print(f"Cromwell cost saved in: {cromwell_cost_file}")

    # load cromwell cost into a dataframe
    cromwell_cost_df = pd.read_csv(cromwell_cost_file)
    cromwell_cost_df["compute_cost"] = cromwell_cost_df["cpu_cost"] + cromwell_cost_df["pe_cpu_cost"] + \
                                        cromwell_cost_df["mem_cost"] + cromwell_cost_df["pe_mem_cost"] + \
                                        cromwell_cost_df["gpu_cost"] + cromwell_cost_df["pe_gpu_cost"]
    cromwell_cost_df["task"] = cromwell_cost_df["task_name"].str.split(".").str[1]
    cromwell_disk_cost = cromwell_cost_df["disk_cost"].sum()
    cromwell_cost_df = cromwell_cost_df.rename(columns=lambda x: x.strip())
    return cromwell_cost_df, cromwell_disk_cost

def omics_cost(omics_run_id, session, output_path, overwrite=False):
    print(f"Calculating omics cost for run: {omics_run_id}")
    performance_file = f'{output_path}/omics_{omics_run_id}.performance.csv'
    if os.path.exists(performance_file) and not overwrite:
        print(f"Skipping download. Omics performance file already exists: {performance_file}")
        performance_df = pd.read_csv(performance_file)
    else:
        performance_df, _ = omics_performance(omics_run_id, session=session, output_prefix=output_path)
    performance_df = performance_df.rename(columns=lambda x: x.strip())
    performance_df['task'] = performance_df['task'].str.strip()
    omics_disk_cost = performance_df["total_storage_cost"].values[0]
    grouped_omics_df = performance_df.groupby(performance_df['task'].str.split("-").str[0]).sum().reset_index()
    return grouped_omics_df, omics_disk_cost



if __name__ == "__main__":
    parser = ArgumentParser()
    # cromwell args
    parser.add_argument('--cromwell-wid', type=str, help="Cromwell workflow id to analyze")

    # omics args
    parser.add_argument('--omics-profile', type=str, help="AWS profile to use")
    parser.add_argument('--region', type=str, help="AWS region to use", default='us-east-1')
    parser.add_argument('--omics-run-id', type=str, help="HealthOmics workflow run-id to analyze")

    # general args
    parser.add_argument('--workflow-name', type=str, help="Workflow name e.g. EfficientDV")
    parser.add_argument('--output-path', type=str, help="Output path for all files copied and generated during the analysis", default=None)
    parser.add_argument('--overwrite', type=bool, help="Overwrite downloaded files from cloud", default=False)

    args = parser.parse_args()
    session = boto3.Session(region_name=args.region, profile_name=args.omics_profile)

    compare_cromwell_omics(args.cromwell_wid, args.omics_run_id, session, args.workflow_name, args.output_path, args.overwrite)

# output_path = "/home/inbalzelig/data/omics/somatic_efficientdv"
# workflow_name = "EfficientDV"
# cromwell_wid = "31b29c63-f7b2-4a7c-986a-64171126d9c4"
# omics_run_id = "2068853"
# omics_profile = "omics-dev"