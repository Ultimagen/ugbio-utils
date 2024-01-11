from argparse import ArgumentParser
import boto3
import logging
from datetime import datetime, timedelta

logging.basicConfig(level=logging.INFO)


# run example: python cleanup_old_omics_runs.py 30
def cleanup_old_runs(days_ago, aws_region):
    omics_client = boto3.client('omics', region_name=aws_region)
    threshold = datetime.now().replace(tzinfo=None) - timedelta(days=days_ago)

    logging.info(f"going to delete runs created before {threshold}")
    deleted_runs = 0
    res = omics_client.list_runs()
    while True:
        logging.info(f"going over {len(res['items'])} runs....")
        for run in res['items']:
            creation_time = run['creationTime'].replace(tzinfo=None)
            if creation_time < threshold:
                logging.info(f"deleting run {run['id']} created on {run['creationTime']}")
                deleted_runs += 1
                omics_client.delete_run(id=run['id'])
        next_token = res['nextToken'] if 'nextToken' in res else None
        if not next_token:
            break
        res = omics_client.list_runs(startingToken=next_token)

    logging.info(f"old runs cleanup completed: {deleted_runs} runs were deleted")


def main(raw_args=None):
    parser = ArgumentParser()
    parser.add_argument("days_ago", help="cleanup runs that created more than X days ago")
    parser.add_argument("-r", "--aws_region", type=str, help="aws region", default="us-east-1")
    args = parser.parse_args(raw_args)
    cleanup_old_runs(int(args.days_ago), args.aws_region)


if __name__ == '__main__':
    main()
