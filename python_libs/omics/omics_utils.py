INPUTS_MAPPINGS = {
    "gs://<": "s3://<",
    "gs://concordanz/": "s3://genomics-pipeline-concordanz-us-east-1/",
    "gs://gcp-public-data--broad-references/hg38/v0/": "s3://genomics-pipeline-ultimagen-public-data-broad-references/references_hg38_v0_",
    "model/germline/v1.1.1/model_dyn_1500_i.onnx.serialized.p100": "model/germline/v1.1.1/model_dyn_1500_i.onnx.serialized.a10g",
    "model/germline/v1.2_rc2/model_dyn_1500_140923.onnx.serialized.p100": "model/germline/v1.2_rc2/model_dyn_1500_140923.onnx.serialized.a10g",
    "gs://ug-cromwell-tests/": "s3://gen-pipe-shared-337532070941-us-east-1/tests-inputs/",
}


def convert_gs_json_to_omics(input_json_as_string):
    for src, dst in INPUTS_MAPPINGS.items():
        input_json_as_string = input_json_as_string.replace(src, dst)
    assert input_json_as_string.find("gs://") == -1, f"input json still contain gs path/s\n{input_json_as_string}"
    return input_json_as_string
