import json
from argparse import ArgumentParser
import boto3
import logging

logging.basicConfig(level=logging.INFO)


def delete_omics_workflows(omics_workflows_json, aws_region):
    with open(omics_workflows_json) as omics_workflows_file:
        omics_workflows = json.load(omics_workflows_file)
    omics_client = boto3.client('omics', region_name=aws_region)
    for wf_id in omics_workflows.values():
        omics_client.delete_workflow(id=wf_id)
        logging.info(f"omics workflow {wf_id} was deleted successfully")


def main(raw_args=None):
    parser = ArgumentParser()
    parser.add_argument("omics_workflows", help="Path to omics_workflows json file")
    parser.add_argument("-r", "--aws_region", type=str, help="aws region", default="us-east-1")
    args = parser.parse_args(raw_args)
    delete_omics_workflows(args.omics_workflows, args.aws_region)


if __name__ == '__main__':
    main()
