import json
import os
import subprocess
from argparse import ArgumentParser

import boto3
import dateutil.parser
import pandas as pd
from compute_pricing import get_run_cost, get_run_info
from get_preformance_from_omics_log import performance as omics_performance
from google.cloud import storage


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
    # Calculate cromwell performance
    cromwell_performance_df = cromwell_performance(cromwell_performance_file)

    # Calculate omics performance and cost
    omics_performance_df, omics_disk_cost = get_omics_performance_cost(omics_run_id, omics_session, output_path)

    # Compare cromwell and omics cost
    print(f"Comparing cromwell and omics...")
    cost_df = pd.DataFrame(omics_performance_df, columns=["task", "cost_SUM"])
    cost_df = cost_df.merge(cromwell_cost_df[['task','compute_cost']], on="task", how="left")
    cost_df = cost_df.rename(columns={"cost_SUM": "omics_cost", "compute_cost": "cromwell_cost"})
    # add disk cost
    cost_df = pd.concat([cost_df, pd.DataFrame({"task": ["disk"], "cromwell_cost": [cromwell_disk_cost], "omics_cost":[omics_disk_cost]})], ignore_index=True)
    # add cost difference
    cost_df["cost_diff"] = cost_df["omics_cost"] - cost_df["cromwell_cost"]
    #move total cost to the end
    cost_df = cost_df[cost_df.task != 'total'].append(cost_df[cost_df.task == 'total'])

    # Compare cromwell and omics run duration
    cromwell_total_duration = get_cromwell_total_duration(cromwell_metadata_file)
    cromwell_performance_df = pd.concat([cromwell_performance_df, pd.DataFrame({"task": ["total"], "run_time (hours)": [cromwell_total_duration]})], ignore_index=True)
    omics_total_duration = get_omics_total_duration(omics_run_id, omics_session)
    omics_performance_df.loc[omics_performance_df['task'] == 'total', 'run_time (hours)'] = omics_total_duration
    duration_df = pd.DataFrame(omics_performance_df, columns=["task", "run_time (hours)"])
    duration_df = duration_df.merge(cromwell_performance_df[['task','run_time (hours)']], on="task", how="left")
    duration_df = duration_df.rename(columns={"run_time (hours)_x": "omics_duration", "run_time (hours)_y": "cromwell_duration"})
    # add duration difference
    duration_df["duration_diff"] = duration_df["omics_duration"] - duration_df["cromwell_duration"]

    # Merge cost and duration and save to file
    compare_df = cost_df.merge(duration_df, on="task", how="left")
    compare_file = f"{output_path}/compare_omics_{omics_run_id}_cromwell_{cromwell_wid}.csv"
    compare_df.to_csv(compare_file, index=False, float_format = '%.3f')
    print(f"Comparison saved in: {compare_file}")


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
        blob = bucket.get_blob(f'cromwell-execution/{workflow_name}/{cromwell_wid}/metadata.json')
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

    # add total cost as an additional row
    cromwell_total_cost = cromwell_cost_df["total_cost"].sum()
    cromwell_cost_df = pd.concat([cromwell_cost_df, pd.DataFrame({"task": ["total"], "compute_cost": [cromwell_total_cost]})], ignore_index=True)

    cromwell_disk_cost = cromwell_cost_df["disk_cost"].sum()

    cromwell_cost_df = cromwell_cost_df.rename(columns=lambda x: x.strip())
    return cromwell_cost_df, cromwell_disk_cost

def cromwell_performance(cromwell_performance_file):
    # load and clean cromwell performance into a dataframe
    performance_df = pd.read_csv(cromwell_performance_file)
    performance_df = performance_df.rename(columns=lambda x: x.strip())
    performance_df['task'] = performance_df['task'].str.strip()
    numeric_columns = performance_df.columns.drop(['task'])
    performance_df[numeric_columns] = performance_df[numeric_columns].apply(pd.to_numeric, errors='coerce')

    # Drop multiple preemtible attempts and keep the last one
    performance_df['attempt'] = performance_df['task'].str.extract('attempt-(\d+)', expand=False).astype(float)
    performance_df['task'] = performance_df['task'].str.split('/attempt-', expand=True)[0]
    performance_df = performance_df.sort_values('attempt', ascending=False).drop_duplicates(subset='task', keep='first')
    performance_df = performance_df.drop(columns='attempt')

    # group by task name and calculate mean of performance metrics
    performance_df['task'] = performance_df['task'].str.split('/').str[0]
    grouped_df = performance_df.groupby('task').mean().reset_index()

    # remove prefix "call-" from task name
    grouped_df['task'] = grouped_df['task'].str.split('call-').str[1]

    return grouped_df

def get_cromwell_total_duration(cromwell_metadata_file):
    with open(cromwell_metadata_file, "r") as f:
        metadata = json.load(f)
    end_time = dateutil.parser.parse(metadata["end"])
    start_time = dateutil.parser.parse(metadata["start"])
    total_duration = end_time - start_time
    return total_duration.total_seconds()/3600

def get_omics_performance_cost(omics_run_id, session, output_path, overwrite=False):
    print(f"Calculating omics performance and cost for run: {omics_run_id}")
    performance_file = f'{output_path}/omics_{omics_run_id}.performance.csv'
    if os.path.exists(performance_file) and not overwrite:
        print(f"Skipping download. Omics performance file already exists: {performance_file}")
        performance_df = pd.read_csv(performance_file)
        if session:
            omics_client = session.client('omics')
        else:
            omics_client = boto3.client('omics')
        cost = get_run_cost(omics_run_id, client=omics_client)
    else:
        performance_df, cost = omics_performance(omics_run_id, session=session, output_dir=output_path)
    performance_df = performance_df.rename(columns=lambda x: x.strip())
    performance_df['task'] = performance_df['task'].str.strip()

    omics_disk_cost = performance_df["total_storage_cost"].values[0]

    # group by task name, calculate mean of performance metrics and sum of cost
    performance_df['task'] = performance_df['task'].str.split('-').str[0]
    grouped_df = performance_df.groupby('task').mean().drop('cost', axis=1)
    grouped_df['cost_SUM'] = performance_df.groupby('task')['cost'].sum()
    grouped_df = grouped_df.reset_index()
    # add total cost as an additional row
    grouped_df = pd.concat([grouped_df, pd.DataFrame({"task": ["total"], "cost_SUM": [cost['total']]})], ignore_index=True)

    return grouped_df, omics_disk_cost

def get_omics_total_duration(omics_run_id, session=None):
    if session:
        omics_client = session.client('omics')
    else:
        omics_client = boto3.client('omics')

    run = get_run_info(omics_run_id, client=omics_client)
    total_duration = run['duration'].total_seconds() / 3600
    return total_duration


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