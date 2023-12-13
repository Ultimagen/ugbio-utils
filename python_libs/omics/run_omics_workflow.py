import json
from argparse import ArgumentParser
import boto3
import logging

logging.basicConfig(level=logging.INFO)

OMICS_WDLS = {
    "ug_deepvariant_variant_calling.wdl": "UGDVVariantCallingPipeline",
    "single_sample_qc.wdl": "SingleSampleQC",
    "cohort_panel_cnmops_CNV_calling.wdl": "CohortCnmopsCNVCalling",
    "efficient_dv.wdl": "EfficientDV",
    "trim_align_sort.wdl": "TrimAlignSort",
    "single_read_snv.wdl": "SingleReadSNV",
    "balanced_strand.wdl": "BalancedStrand"
}


def run_omics_pipeline(run_id, wdl_name, params_file, omics_workflows_json, run_group, aws_account, aws_region):
    with open(omics_workflows_json) as omics_workflows_file:
        omics_workflows = json.load(omics_workflows_file)
    with open(params_file) as params:
        params_json = json.load(params)
    workflow_name = OMICS_WDLS[wdl_name]
    workflow_id = omics_workflows[workflow_name]
    lambda_input = {
        "WorkflowId": workflow_id,
        "WorkflowName": workflow_name,
        "RunId": run_id,
        "InputParams": params_json,
        "RunGroupName": run_group
    }
    lambda_run_pipeline = f"arn:aws:lambda:{aws_region}:{aws_account}:function:startOmicsWorkflow"
    logging.debug(f"going to run invoke start run lamda with args: {lambda_input}")
    res = invoke_lambda_sync(lambda_run_pipeline, lambda_input, aws_region)
    logging.info(res)
    return res["WorkflowRunId"]


def invoke_lambda_sync(lambda_function_name, input_json, aws_region):
    lambda_client = boto3.client('lambda', region_name=aws_region)

    response = lambda_client.invoke(FunctionName=lambda_function_name,
                                    Payload=json.dumps(input_json))
    payload_response_string = response["Payload"].read()

    payload_response_object = json.loads(payload_response_string)
    if payload_response_object is not None and payload_response_object.get("errorMessage") is not None:
        error_message = payload_response_object["errorMessage"]
        stack_trace = payload_response_object["stackTrace"]
        raise Exception(f"{error_message}: {str(stack_trace)}")

    return payload_response_object


def main(raw_args=None):
    parser = ArgumentParser()
    parser.add_argument("run_id", help="Run ID of the data set to analyze")
    parser.add_argument("wdl_name", help="wdl name to run")
    parser.add_argument("params", help="Path to json parameters file")
    parser.add_argument("omics_workflows", help="Path to omics_workflows json file")
    parser.add_argument("-g", "--run_group", help="name of run group", required=False)
    parser.add_argument("-a", "--aws_account", type=str, help="aws account id", default="380827583499")
    parser.add_argument("-r", "--aws_region", type=str, help="aws region", default="us-east-1")
    args = parser.parse_args(raw_args)
    run_omics_pipeline(args.run_id, args.wdl_name, args.params, args.omics_workflows, args.run_group, args.aws_account,
                       args.aws_region)


if __name__ == '__main__':
    main()
