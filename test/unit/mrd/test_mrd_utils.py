import filecmp
import subprocess
import tempfile
from os.path import join as pjoin
from test import get_resource_dir, test_dir

import pandas as pd
from pandas.testing import assert_frame_equal

from ugvc.mrd.mrd_utils import (
    featuremap_to_dataframe,
    generate_synthetic_signatures,
    intersect_featuremap_with_signature,
    read_intersection_dataframes,
    read_signature,
)

general_inputs_dir = pjoin(test_dir, "resources", "general")
inputs_dir = get_resource_dir(__file__)
reference_fasta = pjoin(general_inputs_dir, "chr1_head", "Homo_sapiens_assembly38.fasta")
intersection_file_basename = "MRD_test_subsample.MRD_test_subsample_annotated_AF_vcf_gz_mrd_quality_snvs.intersection"


def test_read_signature_ug_mutect():
    signature = read_signature(pjoin(inputs_dir, "mutect_mrd_signature_test.vcf.gz"))
    signature_no_sample_name = read_signature(pjoin(inputs_dir, "mutect_mrd_signature_test.no_sample_name.vcf.gz"))
    expected_output = pd.read_hdf(pjoin(inputs_dir, "mutect_mrd_signature_test.expected_output.h5"))

    assert_frame_equal(signature, expected_output)
    assert_frame_equal(
        signature_no_sample_name.drop(columns=["af", "depth_tumor_sample"]),
        expected_output.drop(columns=["af", "depth_tumor_sample"]),
    )  # make sure we can read the dataframe even if the sample name could not be deduced from the header


def test_read_signature_external():
    signature = read_signature(pjoin(inputs_dir, "external_somatic_signature.vcf.gz"))
    expected_output = pd.read_hdf(pjoin(inputs_dir, "external_somatic_signature.expected_output.h5"))

    assert_frame_equal(signature, expected_output)


def test_intersect_featuremap_with_signature():
    signature_file = pjoin(inputs_dir, "Pa_46.FreshFrozen.chr20.70039_70995.vcf.gz")
    featuremap_file = pjoin(inputs_dir, "Pa_46.bsDNA.chr20_sample.vcf.gz")
    test_file = pjoin(inputs_dir, "intersected_featuremap.vcf.gz")

    with tempfile.TemporaryDirectory() as tmpdirname:
        output_intersection_file = pjoin(tmpdirname, "intersected.vcf.gz")
        intersect_featuremap_with_signature(
            featuremap_file, signature_file, output_intersection_file=output_intersection_file
        )
        cmd1 = f"bcftools view -H {output_intersection_file}"
        cmd2 = f"bcftools view -H {test_file}"
        assert subprocess.check_output(cmd1, shell=True) == subprocess.check_output(cmd2, shell=True)


def test_featuremap_to_dataframe():
    input_featuremap = pjoin(inputs_dir, "Pa_46.bsDNA.chr20_sample.vcf.gz")
    expected_featuremap_dataframe = pjoin(inputs_dir, "Pa_46.bsDNA.chr20_sample.parquet")
    with tempfile.TemporaryDirectory() as tmpdirname:
        tmp_out_path = pjoin(tmpdirname, "tmp_out.parquet")

        featuremap_dataframe = featuremap_to_dataframe(featuremap_vcf=input_featuremap, output_file=tmp_out_path)
        featuremap_dataframe_expected = pd.read_parquet(expected_featuremap_dataframe)
        assert_frame_equal(featuremap_dataframe, featuremap_dataframe_expected)
        assert filecmp.cmp(tmp_out_path, expected_featuremap_dataframe)


def test_read_intersection_dataframes():
    parsed_intersection_dataframe = read_intersection_dataframes(
        pjoin(inputs_dir, f"{intersection_file_basename}.expected_output.parquet")
    )
    parsed_intersection_dataframe_expected = pd.read_parquet(
        pjoin(inputs_dir, f"{intersection_file_basename}.parsed.expected_output.parquet")
    )
    parsed_intersection_dataframe2 = read_intersection_dataframes(
        [pjoin(inputs_dir, f"{intersection_file_basename}.expected_output.parquet")]
    )
    assert_frame_equal(
        parsed_intersection_dataframe.reset_index(),
        parsed_intersection_dataframe_expected,
    )
    assert_frame_equal(
        parsed_intersection_dataframe2.reset_index(),
        parsed_intersection_dataframe_expected,
    )


def test_generate_synthetic_signatures():
    signature_file = pjoin(inputs_dir, "mutect_mrd_signature_test.vcf.gz")
    db_file = pjoin(
        inputs_dir,
        "pancan_pcawg_2020.mutations_hg38_GNOMAD_dbsnp_beds.sorted.Annotated.HMER_LEN.edited.chr19.vcf.gz",
    )
    n_synthetic_signatures = 1
    with tempfile.TemporaryDirectory() as tmpdirname:
        output_dir = tmpdirname
        synthetic_signature_list = generate_synthetic_signatures(
            signature_file, db_file, n_synthetic_signatures, output_dir
        )
        output_file = synthetic_signature_list[0]
        test_signautre = pjoin(inputs_dir, "synthetic_signature_test.vcf.gz")
        # assert similar number of lines in output file as in input file
        cmd1 = f"bcftools view -H {output_file}"
        cmd2 = f"bcftools view -H {test_signautre}"
        assert subprocess.check_output(cmd1, shell=True) == subprocess.check_output(cmd2, shell=True)