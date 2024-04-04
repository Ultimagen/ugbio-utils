INPUTS_MAPPINGS = {
    "gs://<": "s3://<",
    "gs://concordanz/": "s3://genomics-pipeline-concordanz-us-east-1/",
    "gs://ultimagen-trimmer-dev-formats/": "s3://genomics-pipeline-trimmer-dev-formats/",
    "gs://gcp-public-data--broad-references/hg38/v0/Homo_sapiens_assembly38.haplotype_database.txt" : "s3://genomics-pipeline-ultimagen-public-data-broad-references/references_hg38_v0_Homo_sapiens_assembly38.haplotype_database.txt",
    "gs://gcp-public-data--broad-references/hg19/v0/Homo_sapiens_assembly19.haplotype_database.txt" : "s3://genomics-pipeline-ultimagen-public-data-broad-references/references_hg19_v0_Homo_sapiens_assembly19.haplotype_database.txt",
    "gs://gcp-public-data--broad-references/": "s3://broad-references/",
    "model/germline/v1.1.1/ultima_deep_variant_v1.1.1.ckpt-780000.onnx.serialized.p100": "model/germline/v1.1.1/ultima_deep_variant_v1.1.1.ckpt-780000.onnx.serialized.a10g",
    "model/germline/v1.2_rc2/model_dyn_1500_140923.onnx.serialized.p100": "model/germline/v1.2_rc2/model_dyn_1500_140923.onnx.serialized.a10g",
    "model/germline/v1.3/model.ckpt-890000.dyn_1500.onnx.serialized.p100": "model/germline/v1.3/model.ckpt-890000.dyn_1500.onnx.serialized.a10g",
    "model/somatic/wgs/deepvariant-ultima-somatic-wgs-model-v1.0_4.9.ckpt-2950000.onnx.serialized.p100": "model/somatic/wgs/deepvariant-ultima-somatic-wgs-model-v1.0_4.9.ckpt-2950000.onnx.serialized.a10g",
    "model/somatic/wgs/deepvariant-ultima-somatic-wgs-model-v1.0_4.9.ckpt-2950000.onnx.serialized.p100": "model/somatic/wgs/deepvariant-ultima-somatic-wgs-model-v1.0_4.9.ckpt-2950000.onnx.serialized.a10g",
    "model/somatic/wes/deepvariant-ultima-somatic-wes-model-v0.1.ckpt-120000.onnx.serialized.p100": "model/somatic/wes/deepvariant-ultima-somatic-wes-model-v0.1.ckpt-120000.onnx.serialized.a10g",
    "model/somatic/wgs/ffpe/deepvariant-ultima-somatic-wgs-ffpe-model-v1.3.ckpt-890000.onnx.serialized.p100": "model/somatic/wgs/ffpe/deepvariant-ultima-somatic-wgs-ffpe-model-v1.3.ckpt-890000.onnx.serialized.a10g",
    "/ground-truths-files/ground_truth_files_giab_4.2.1_hg38.json": "/ground-truths-files/aws/ground_truth_files_giab_4.2.1_hg38.json",
    "/hg19/ground_truth_files_giab_4.2.1_hg19.json": "/hg19/aws/ground_truth_files_giab_4.2.1_hg19.json",
    "gs://ug-cromwell-tests/": "s3://gen-pipe-shared-337532070941-us-east-1/tests-inputs/",
}


def convert_gs_json_to_omics(input_json_as_string):
    for src, dst in INPUTS_MAPPINGS.items():
        input_json_as_string = input_json_as_string.replace(src, dst)
    assert input_json_as_string.find("gs://") == -1, f"input json still contain gs path/s\n{input_json_as_string}"
    return input_json_as_string
