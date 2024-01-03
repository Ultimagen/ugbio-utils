#!/usr/bin/env python
import json
import os
import shutil
import time
from pathlib import Path
from os import path
import logging
from argparse import ArgumentParser
import boto3
from botocore.exceptions import ClientError

from os.path import exists, abspath
from tempfile import TemporaryDirectory
from omics_utils import convert_gs_json_to_omics

BASE_DIR = Path(abspath(__file__)).parent.parent.parent

JSON_SECTION = "omics_release"
INPUTS_TEMPLATES = "inputs_templates"
INPUT_TEMPLATE_OVERRIDES = "aws_inputs_template_overrides"
INPUTS_TEMPLATE_S3_KEY = "inputs_templates_s3_key"
PARAMS_DEF = "params_def"
PIPELINES = "pipelines"
WDL_FILES = "wdl_files"
MAIN_WDL = "main_wdl"
WDL_TASKS = "tasks"
GLOBALS_TASK = "globals.wdl"
AWS_GLOBALS_TASK = "wdls/tasks/globals_aws.wdl"
EDIT_INPUT_JSON_SCRIPT = "google_functions/src/edit_input_json.py"
EDIT_INPUT_JSON_REMOTE_KEY = "python_libs/edit_input_json.py"
GENERATE_CONF_FROM_TEMPLATE_KEY = "python_libs/generate_conf_from_template.py"
PIPELINE_VERSION_TAG = "pipeline_version"

OMICS_WF_ACTIVE = "ACTIVE"
OMICS_WF_DONE_STATUSES = [OMICS_WF_ACTIVE, "DELETED", "FAILED"]
OMICS_PRIVATE_TYPE = "PRIVATE"

workflow_dict = {}

logging.basicConfig(level=logging.INFO)


def safe_copy(source: Path, target: Path, dryrun: bool = False):
    assert exists(source), f"Missing file {source}"
    if not dryrun:
        if source.is_file():
            logging.info(f"copy {source} -> {target}")
            shutil.copy(source.absolute(), target.absolute())


def get_path_list(list_of_local_files):
    return [get_path(x) for x in list_of_local_files]


def get_path(local_file):
    return Path(f"{BASE_DIR}/{local_file}")


def process_pipeline(pipeline_name, pipeline_args, aws_account, aws_region, version):
    wdls = get_path_list(pipeline_args[WDL_FILES])
    main_wdl = get_path(pipeline_args[MAIN_WDL])
    input_templates = get_path_list(pipeline_args[INPUTS_TEMPLATES])
    input_template_overrides = get_path(pipeline_args[INPUT_TEMPLATE_OVERRIDES])
    inputs_template_s3_key = pipeline_args[INPUTS_TEMPLATE_S3_KEY]
    params_def = get_path(pipeline_args[PARAMS_DEF])
    tasks = get_path_list(pipeline_args[WDL_TASKS]) if WDL_TASKS in pipeline_args else []

    for f in wdls + tasks + input_templates + [input_template_overrides] + [params_def] + [main_wdl]:
        assert exists(f), f"Missing file {f}"

    with TemporaryDirectory() as tmpdir:
        bucket, wdl_zip_s3_uri = zip_and_sync_workflow_files(aws_account, aws_region, pipeline_name, tasks, tmpdir,
                                                             version, wdls)

        delete_prev_workflows(pipeline_name, version)

        create_omics_workflow(main_wdl, params_def, pipeline_name, version, wdl_zip_s3_uri)

        upload_input_template(bucket, input_templates, input_template_overrides, inputs_template_s3_key, version,
                              tmpdir)


def delete_prev_workflows(pipeline_name, version):
    # delete old Omics workflows tagged with same version
    omics_client = boto3.client("omics")
    try:
        response = omics_client.list_workflows(
            type=OMICS_PRIVATE_TYPE,
            name=pipeline_name,
        )
        logging.debug(response)
        if response['items']:
            for item in response['items']:
                wf_resp = omics_client.get_workflow(id=item['id'],
                                                    type=OMICS_PRIVATE_TYPE)
                if wf_resp and wf_resp['tags'] and wf_resp['tags'][PIPELINE_VERSION_TAG] == version:
                    logging.info(
                        f"Old omics {pipeline_name} workflow id {item['id']} with same {PIPELINE_VERSION_TAG} tag: '{version}' will "
                        f"be deleted")
                    omics_client.delete_workflow(id=item['id'])

    except ClientError as e:
        logging.error(e)
        exit(1)


def create_omics_workflow(main_wdl, params_def, pipeline_name, version, wdl_zip_s3_uri):
    # create Omics workflow
    with params_def.open(encoding="UTF-8") as source:
        wdl_params = json.load(source)
    omics_client = boto3.client("omics")
    try:
        response = omics_client.create_workflow(
            name=pipeline_name,
            engine="WDL",
            definitionUri=wdl_zip_s3_uri,
            main=main_wdl.name,
            parameterTemplate=wdl_params,
            tags={
                PIPELINE_VERSION_TAG: version
            }
        )
        logging.info(response)
        workflow_id = response["id"]
        wf_status = response["status"]
        while wf_status not in OMICS_WF_DONE_STATUSES:
            time.sleep(5)
            response = omics_client.get_workflow(id=workflow_id)
            wf_status = response["status"]
        if wf_status != OMICS_WF_ACTIVE:
            logging.error(f"{pipeline_name} omics workflow creation failed with msg: {response['statusMessage']}")
            exit(1)
        workflow_dict.update({pipeline_name: workflow_id})

    except ClientError as e:
        logging.error(e)
        exit(1)


def zip_and_sync_workflow_files(aws_account, aws_region, pipeline_name, tasks, tmpdir, version, wdls):
    os.makedirs(Path(f"{tmpdir}/{pipeline_name}"), exist_ok=True)
    for wdl in wdls:
        safe_copy(wdl, Path(f"{tmpdir}/{pipeline_name}/{wdl.name}"))
    if tasks:
        os.makedirs(Path(f"{tmpdir}/{pipeline_name}/{WDL_TASKS}"), exist_ok=True)
        for task in tasks:
            task_file = get_path(AWS_GLOBALS_TASK) if task.name == GLOBALS_TASK else task
            safe_copy(task_file, Path(f"{tmpdir}/{pipeline_name}/{WDL_TASKS}/{task.name}"))
    zip_file = path.join(tmpdir, pipeline_name)
    shutil.make_archive(zip_file, "zip", Path(f"{tmpdir}/{pipeline_name}"))
    # Upload the zip file to s3
    try:
        bucket = get_bucket_name(aws_account, aws_region)
        wdl_key = f"wdls/{version}/{pipeline_name}.zip"
        wdl_zip_s3_uri = f"s3://{bucket}/{wdl_key}"
        upload_to_s3(zip_file + ".zip", bucket, wdl_key)
    except ClientError as e:
        logging.error(e)
        exit(1)
    return bucket, wdl_zip_s3_uri


def upload_input_template(bucket, input_templates, input_template_overrides, inputs_template_s3_key, version, tmpdir):
    for input_template in input_templates:
        input_template_base_name = input_template.name
        # update the input template for aws and upload it to s3

        # Read in the file
        with open(input_template, "r") as file:
            template_data = file.read()

        updated_template_data = convert_gs_json_to_omics(template_data)

        # Write the file out again
        input_template_for_aws = f"{tmpdir}/input_template.json"
        with open(input_template_for_aws, 'w') as file:
            file.write(updated_template_data)

        with open(input_template_for_aws) as base_template:
            base_template_data = json.load(base_template)
            with open(input_template_overrides) as template_overrides:
                template_overrides_data = json.load(template_overrides)
                # Merge base template with aws overrides
                json_with_aws_overrides = {**base_template_data, **template_overrides_data}
                input_template_key = f"{version}/terra_pipeline/wdls/input_templates/{inputs_template_s3_key}{input_template_base_name}"

                # Upload final template to s3
                put_dict_to_s3_json(bucket, input_template_key, json_with_aws_overrides)


def get_bucket_name(aws_account, aws_region):
    bucket = f"ultimagen-pipelines-{aws_account}-{aws_region}-wdls"
    return bucket


def upload_to_s3(local_file, bucket, key):
    s3_client = boto3.client("s3")
    logging.info(f"upload {local_file} to s3://{bucket}/{key}")
    s3_client.upload_file(local_file, bucket, key)


def upload_workflow_dict(aws_account, aws_region, version):
    logging.info(f"workflows_dict: {workflow_dict}")
    bucket = get_bucket_name(aws_account, aws_region)
    workflows_json_key = f"{version}/omics_workflows.json"
    put_dict_to_s3_json(bucket, workflows_json_key, workflow_dict)


def upload_common_resources(aws_account, aws_region, version):
    bucket = get_bucket_name(aws_account, aws_region)

    # copy google_functions/src/edit_input_json.py
    edit_input_json_script_path = get_path(EDIT_INPUT_JSON_SCRIPT)
    assert exists(edit_input_json_script_path), f"Missing file {edit_input_json_script_path}"
    edit_input_script_key = f"{version}/terra_pipeline/{EDIT_INPUT_JSON_REMOTE_KEY}"
    upload_to_s3(str(edit_input_json_script_path), bucket, edit_input_script_key)

    # copy python_libs/generate_conf_from_template.py
    generate_conf_script_path = get_path(GENERATE_CONF_FROM_TEMPLATE_KEY)
    assert exists(generate_conf_script_path), f"Missing file {generate_conf_script_path}"
    generate_conf_script_key = f"{version}/terra_pipeline/{GENERATE_CONF_FROM_TEMPLATE_KEY}"
    upload_to_s3(str(generate_conf_script_path), bucket, generate_conf_script_key)


def put_dict_to_s3_json(bucket, key, dict):
    try:
        s3 = boto3.resource("s3")
        s3object = s3.Object(bucket, key)
        s3object.put(
            Body=(bytes(json.dumps(dict, indent=4).encode("UTF-8")))
        )
        logging.info(f"json file s3://{bucket}/{key} created successfully")

    except ClientError as e:
        logging.error(e)
        exit(1)


def main():
    parser = ArgumentParser()
    parser.add_argument("release_conf", help="path to release_conf json", type=str)
    parser.add_argument("-w", "--workflow", help="workflow to deploy", type=str, default=None)
    parser.add_argument("-a", "--aws_account", type=str, help="aws account id")
    parser.add_argument("-v", "--version", type=str, help="deploy version")
    parser.add_argument("-r", "--aws_region", type=str, help="aws region", default="us-east-1")
    args = parser.parse_args()
    with open(args.release_conf) as jf:
        conf = json.load(jf)[JSON_SECTION]
    aws_account = args.aws_account
    version = args.version
    aws_region = args.aws_region

    def select_workflow(pair):
        key, pipeline_args = pair
        if args.workflow == key:
            return True  # filter pair out
        else:
            return False  # keep pair in

    pipelines = conf[PIPELINES] if not args.workflow else dict(filter(select_workflow, conf[PIPELINES].items()))

    try:
        for pipeline_name, pipeline_args in pipelines.items():
            process_pipeline(pipeline_name, pipeline_args, aws_account, aws_region, version)
    finally:
        upload_common_resources(aws_account, aws_region, version)
        upload_workflow_dict(aws_account, aws_region, version)


if __name__ == '__main__':
    main()
