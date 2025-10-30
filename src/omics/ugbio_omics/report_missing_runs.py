import json
import logging
import time
from argparse import ArgumentParser

import boto3

from ugbio_omics.report_run_completion import report_run_completion

logging.basicConfig(level=logging.INFO)

LAMBDA_FUNCTION_NAME = "omicsWorkflowStartHandler"
SUCCESS_STATUS_CODE = 200


def report_run_start(run_ids: list, aws_region):
    account_id = boto3.client("sts").get_caller_identity().get("Account")
    lambda_client = boto3.client("lambda", region_name=aws_region)
    for run_id in run_ids:
        lambda_input = {
            "detail-type": "Run Status Change (Manual)",
            "source": "ugbio_omics.report_run_start",
            "account": account_id,
            "region": aws_region,
            "resources": [f"arn:aws:omics:{aws_region}:{account_id}:run/{run_id}"],
        }
        response = lambda_client.invoke(
            FunctionName=LAMBDA_FUNCTION_NAME, InvocationType="RequestResponse", Payload=json.dumps(lambda_input)
        )
        logging.info(f"{LAMBDA_FUNCTION_NAME} was triggered manually with event: {lambda_input}")

        if response["StatusCode"] != SUCCESS_STATUS_CODE:
            raise Exception("Error. StatusCode: " + response["StatusCode"])


def main(raw_args=None):
    parser = ArgumentParser(description="Reports start and End of Omics Runs that are missing from the database")
    parser.add_argument("run_ids", help="Omics run ids (space separated)", nargs="+")
    parser.add_argument("-r", "--aws_region", type=str, help="aws region", default="us-east-1")
    args = parser.parse_args(raw_args)
    report_run_start(args.run_ids, args.aws_region)
    time.sleep(30)  # sleep for 30 seconds to allow the start event to be processed
    report_run_completion(args.run_ids, args.aws_region)


if __name__ == "__main__":
    main()
