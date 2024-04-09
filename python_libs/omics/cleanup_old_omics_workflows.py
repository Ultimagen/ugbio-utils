from argparse import ArgumentParser
import boto3
import logging
from datetime import datetime, timedelta
from re import match

logging.basicConfig(level=logging.INFO)

DEFAULT_KEEP_VERSION_PATTERN = r"^v\d+\.\d+\.\d+(\.\d+)?$"

# run example: python cleanup_old_omics_workflows.py 30
def cleanup_old_workflows(days_ago, aws_region, keep_version_pattern=DEFAULT_KEEP_VERSION_PATTERN):
    omics_client = boto3.client("omics", region_name=aws_region)
    threshold = datetime.now().replace(tzinfo=None) - timedelta(days=days_ago)

    logging.info(f"going to delete workflows created before {threshold}")
    deleted_workflows = 0
    res = omics_client.list_workflows()
    while True:
        logging.info(f"going over {len(res['items'])} workflows....")
        for workflow in res["items"]:
            creation_time = workflow["creationTime"].replace(tzinfo=None)
            if creation_time < threshold:
                wid = workflow["id"]
                can_delete = True
                if keep_version_pattern:
                    workflow_version = omics_client.get_workflow(id=wid)["tags"].get("pipeline_version")
                    logging.info(f"found version: {workflow_version}")
                    if workflow_version and match(keep_version_pattern, workflow_version):
                        logging.warning(f"won't delete workflow {wid} created on {workflow['creationTime']}")
                        can_delete = False
                if can_delete:
                    logging.info(f"deleting workflow {wid} created on {workflow['creationTime']}")
                    deleted_workflows += 1
                    omics_client.delete_workflow(id=wid)
        next_token = res["nextToken"] if "nextToken" in res else None
        if not next_token:
            break
        res = omics_client.list_workflows(startingToken=next_token)

    logging.info(f"old workflows cleanup completed: {deleted_workflows} workflows were deleted")


def main(raw_args=None):
    parser = ArgumentParser()
    parser.add_argument("days_ago", help="cleanup workflows that created more than X days ago")
    parser.add_argument("-k", "--keep_version_pattern", help="exclude workflows tagged with this version (match regex)", default=DEFAULT_KEEP_VERSION_PATTERN)
    parser.add_argument("-r", "--aws_region", type=str, help="aws region", default="us-east-1")
    args = parser.parse_args(raw_args)
    cleanup_old_workflows(int(args.days_ago), args.aws_region, args.keep_version_pattern)


if __name__ == "__main__":
    main()
