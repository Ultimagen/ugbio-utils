from argparse import ArgumentParser
import os

import boto3
import pandas as pd

from get_omics_log import fetch_save_log

def parse_manifest_log(run_id, output_path=None, session=None, output_prefix=''):
    manifest_json_file = f'{output_prefix}omics_{run_id}_manifest.json'
    if output_path:
        os.makedirs(output_path, exist_ok=True)
        manifest_json_file = f"{output_path}/{manifest_json_file}"

    # Get logs client
    if session:
        client = session.client('logs')
    else:
        client = boto3.client('logs')

    log_group_name = '/aws/omics/WorkflowLog'

    # get manifest log stream name
    res = client.describe_log_streams(
        logGroupName=log_group_name,
        logStreamNamePrefix=f'manifest/run/{run_id}'
    )

    if not res.get('logStreams'):
        print(f"No manifest log stream found for run id '{run_id}'")
        return
    
    log_stream_name = res['logStreams'][0]['logStreamName']
    
    fetch_save_log(log_stream_name, log_group_name, manifest_json_file, session)

    # parse manifest log from json into df
    df = pd.read_json(manifest_json_file, lines=True)
    
    # Save general run info and print out storage info
    run_df = df.head(1).dropna(axis=1, how='all')

    print('---------Storage info---------')
    storage_df = run_df['metrics'].apply(pd.Series)
    print(storage_df.to_json(orient='records', lines=True, default_handler=str, indent=4))

    general_run_info_file = f'{output_prefix}omics_{run_id}_general_run_info.json'
    if output_path:
        general_run_info_file = f"{output_path}/{general_run_info_file}"
    print(f"Saving general run info to: {general_run_info_file}")
    run_df.to_json(general_run_info_file, orient='records', lines=True, default_handler=str, indent=4)

    # Save tasks info to csv
    tasks_df = df.drop(0).dropna(axis=1, how='all') # remove first line that contains general run info
    metrics_df = tasks_df['metrics'].apply(pd.Series) #expand metrics column into separate columns
    tasks_df = pd.concat([tasks_df.drop(['metrics'], axis=1), metrics_df], axis=1)

    tasks_manifest_csv_file = f'{output_prefix}omics_{run_id}_task_manifests.csv'
    if output_path:
        tasks_manifest_csv_file = f"{output_path}/{tasks_manifest_csv_file}"
    print(f"Saving task manifests to: {tasks_manifest_csv_file}")
    tasks_df.to_csv(tasks_manifest_csv_file, index=False)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--profile', type=str, help="AWS profile name to use from local profiles listed in ~/.aws/config")
    parser.add_argument('--region', type=str, help="AWS region to use", default='us-east-1')
    parser.add_argument('--run-id', type=str, help="HealthOmics workflow run-id to analyze")
    parser.add_argument('--output', type=str, help="Output dir to save log events", default=None)
    parser.add_argument('--output-prefix', type=str, help="File name prefix for the output", required=False)

    args = parser.parse_args()
    session = boto3.Session(region_name=args.region, profile_name=args.profile)

    parse_manifest_log(args.run_id, output_path=args.output, session=session, output_prefix=args.output_prefix)