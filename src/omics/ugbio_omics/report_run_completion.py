import json
import logging
from argparse import ArgumentParser

import boto3

logging.basicConfig(level=logging.INFO)

LAMBDA_FUNCTION_NAME = "omicsEndRunHandler"
SUCCESS_STATUS_CODE = 200


def report_run_completion(run_ids: list, aws_region):
    account_id = boto3.client("sts").get_caller_identity().get("Account")
    lambda_client = boto3.client("lambda", region_name=aws_region)
    for run_id in run_ids:
        lambda_input = {
            "detail-type": "Run Status Change (Manual)",
            "source": "ugbio_omics.report_run_completion",
            "account": account_id,
            "region": aws_region,
            "runAnalyzerResult": {"Payload": {"run_id": run_id}},
        }
        response = lambda_client.invoke(
            FunctionName=LAMBDA_FUNCTION_NAME, InvocationType="RequestResponse", Payload=json.dumps(lambda_input)
        )
        logging.info(f"{LAMBDA_FUNCTION_NAME} was triggered manually with event: {lambda_input}")

        if response["StatusCode"] != SUCCESS_STATUS_CODE:
            raise Exception("Error. StatusCode: " + response["StatusCode"])


def main(raw_args=None):
    parser = ArgumentParser(description="Reports completion of Omics Run by invoking omicsEndRunHandler lambda")
    parser.add_argument("run_ids", help="Omics run ids (space separated)", nargs="+")
    parser.add_argument("-r", "--aws_region", type=str, help="aws region", default="us-east-1")
    args = parser.parse_args(raw_args)
    report_run_completion(args.run_ids, args.aws_region)


if __name__ == "__main__":
    main()
