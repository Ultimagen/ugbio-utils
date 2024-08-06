import json
from argparse import ArgumentParser
import boto3
import logging

logging.basicConfig(level=logging.INFO)

LAMBDA_FUNCTION_NAME = "omicsEndRunHandler"


def report_run_completion(run_id, aws_region):
    account_id = boto3.client('sts').get_caller_identity().get('Account')
    assert account_id
    lambda_client = boto3.client('lambda', region_name=aws_region)
    lambda_input = {
        'detail-type': 'Run Status Change (Manual)',
        'source': 'terra_pipeline.omics.scripts',
        'account': account_id,
        'region': aws_region,
        'resources': [f'arn:aws:omics:{aws_region}:{account_id}:run/{run_id}']
    }
    response = lambda_client.invoke(FunctionName=LAMBDA_FUNCTION_NAME,
                                    InvocationType='RequestResponse',
                                    Payload=json.dumps(lambda_input))
    logging.info(
        f"{LAMBDA_FUNCTION_NAME} was triggered manually with event: {lambda_input}")

    if response["StatusCode"] != 200:
        raise Exception("Error. StatusCode: " + response["StatusCode"])


def main(raw_args=None):
    parser = ArgumentParser(description="Reports completion of Omics Run by invoking omicsEndRunHandler lambda")
    parser.add_argument("run_id", help="Omics run id")
    parser.add_argument("-r", "--aws_region", type=str, help="aws region", default="us-east-1")
    args = parser.parse_args(raw_args)
    report_run_completion(args.run_id, args.aws_region)


if __name__ == '__main__':
    main()
