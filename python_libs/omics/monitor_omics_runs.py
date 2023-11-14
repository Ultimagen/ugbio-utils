import json
import sys
import time
from argparse import ArgumentParser
import boto3
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO)

NON_COMPLETED_STATUSES = ['PENDING', 'STARTING', 'RUNNING', 'STOPPING']
MIN_BETWEEN_CHECKS = 5
SUCCESS_STATUS = 'COMPLETED'


def poll_omics_runs(run_id, poll_until_done, interval, report_markdown, aws_region):
    omics_client = boto3.client('omics', region_name=aws_region)
    runs_res = omics_client.list_runs(name=run_id)['items']

    logging.info(f'{len(runs_res)} runs were found for run_id: {run_id}')

    if poll_until_done:
        non_completed_runs = list(filter(lambda p: p['status'] in NON_COMPLETED_STATUSES, runs_res))
        while non_completed_runs:
            logging.info(f'{len(non_completed_runs)} are still active, sleeping for {interval} minutes...')
            time.sleep(interval * 60)
            omics_client = boto3.client('omics', region_name=aws_region)
            runs_res = omics_client.list_runs(name=run_id)['items']
            non_completed_runs = list(filter(lambda p: p['status'] in NON_COMPLETED_STATUSES, runs_res))

    summarize_run(run_id, omics_client, report_markdown, poll_until_done)


def get_workflow_name(workflow_id, omics_client):
    workflow = omics_client.get_workflow(id=workflow_id)
    return workflow["name"]


def get_run_name(run_id, omics_client):
    run = omics_client.get_run(id=run_id)
    return run["tags"].get("run_name", run["name"])


def summarize_run(run_id, omics_client, report_markdown, poll_until_done):
    report_path = report_markdown if report_markdown else f'/tmp/{run_id}_workflows_report.md'
    runs_res = omics_client.list_runs(name=run_id)
    omics_runs = runs_res['items']
    if omics_runs:
        logging.debug(omics_runs)
        tests_df = pd.DataFrame.from_records(omics_runs)
        tests_df['workflowName'] = tests_df.apply(lambda x: get_workflow_name(x['workflowId'], omics_client), axis=1)
        tests_df['runName'] = tests_df.apply(lambda x: get_run_name(x['id'], omics_client), axis=1)
        logging.debug(tests_df)
        tests_df = tests_df.loc[:, tests_df.columns.isin(
            ['name', 'id', 'runName', 'workflowName', 'status', 'creationTime', 'startTime', 'stopTime'])]
        with open(report_path, 'w') as fp:
            fp.write(tests_df.to_markdown())
        logging.info(f"{run_id} report was generated and saved in: {report_path}")
        if poll_until_done:
            failed_filter = tests_df['status'] != SUCCESS_STATUS
            failed_tests = len(tests_df[failed_filter])
            if failed_tests > 0:
                sys.exit(f"{len(tests_df[failed_filter])} unsuccessful tests")
    else:
        sys.exit("no running tests")


def main(raw_args=None):
    parser = ArgumentParser()
    parser.add_argument("run_id", help="Run ID to query")
    parser.add_argument("-p", "--poll_until_done", help="Poll jobs until done", action="store_true")
    parser.add_argument("-rm", "--report_markdown", type=str, help="full path of generated tests report markdown",
                        required=False)
    parser.add_argument("-i", "--interval", help="minutes to wait between checking  workflow status", type=int,
                        default=MIN_BETWEEN_CHECKS)
    parser.add_argument("-r", "--aws_region", type=str, help="aws region", default="us-east-1")
    args = parser.parse_args(raw_args)
    poll_omics_runs(args.run_id, args.poll_until_done, args.interval, args.report_markdown, args.aws_region)


if __name__ == '__main__':
    main()
