from argparse import ArgumentParser
from decimal import Decimal
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import boto3

OMICS_WORKFLOWS_FILE_NAME = "omics_workflows.json"
OMICS_DDB_TABLE_REGION = "us-east-1"
OMICS_WORKFLOW_DDB_TABLE = "OmicsWorkflows"
OMICS_DDB_VERSION_KEY = "version"
OMICS_DDB_WORKFLOW_KEY = "workflow"
OMICS_DDB_WORKFLOW_ID_KEY = "workflow_id"
OMICS_DDB_INPUT_TEMPLATES_KEY = "input_templates"

s3_client = boto3.client("s3")
s3_resource = boto3.resource('s3')
db_table_resource = boto3.resource("dynamodb", region_name=OMICS_DDB_TABLE_REGION).Table(OMICS_WORKFLOW_DDB_TABLE)


def main(terra_pipeline_version, wdls_bucket):
    print(f"Get workflows map for version {terra_pipeline_version}")
    workflows_map = get_workflows_json_from_s3(terra_pipeline_version, wdls_bucket)
    version_dict = {}
    for w, wid in workflows_map.items():
        version_dict[w] = {
            OMICS_DDB_VERSION_KEY: terra_pipeline_version,
            OMICS_DDB_WORKFLOW_KEY: w,
            OMICS_DDB_WORKFLOW_ID_KEY: wid,
            OMICS_DDB_INPUT_TEMPLATES_KEY: {}
        }

    print(f"Download all input templates for version {terra_pipeline_version}")
    with TemporaryDirectory() as tmpdir:
        s3_prefix = f"{terra_pipeline_version}/terra_pipeline/"
        objs = s3_client.list_objects_v2(
            Bucket=wdls_bucket,
            Prefix=f"{s3_prefix}wdls/input_templates/"
        )
        for obj in objs["Contents"]:
            key = obj["Key"]
            dest = Path(tmpdir) / key
            dest.parent.mkdir(parents=True, exist_ok=True)
            s3_resource.meta.client.download_file(wdls_bucket, key, dest)
            with open(dest) as f:
                template_data = json.load(f)
            workflow_name = list(template_data.keys())[0].split(".")[0]
            template_dict = {
                key.replace(s3_prefix, ""): 
                {
                    **template_data
                }
            }
            version_dict[workflow_name]["input_templates"].update(template_dict)

    print("Write to Dynamo DB")
    for w, entry in version_dict.items():
        print(f"Write {w}")
        write_workflow_in_dynamodb(entry)


def get_workflows_json_from_s3(terra_pipeline_version, wdls_bucket):
    result = s3_client.get_object(Bucket=wdls_bucket,
                               Key=f'{terra_pipeline_version}/{OMICS_WORKFLOWS_FILE_NAME}')
    workflows_file_content = result["Body"].read().decode('utf-8')
    return json.loads(workflows_file_content)


def write_workflow_in_dynamodb(worklow_entry):
    # https://stackoverflow.com/a/71446846
    # DynamoDB doesn't support floats, but for some reason cast also int to Decimal
    # need to encode back when retrieving the document
    db_item = json.loads(json.dumps(worklow_entry), parse_float=Decimal)
    db_table_resource.put_item(
        Item=db_item
    )
    return True


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("terra_pipeline_version", help="Versio to transfer")
    args = parser.parse_args()

    current_account = boto3.client("sts").get_caller_identity()["Account"]
    wdls_bucket = f"ultimagen-pipelines-{current_account}-us-east-1-wdls"

    main(args.terra_pipeline_version, wdls_bucket)
