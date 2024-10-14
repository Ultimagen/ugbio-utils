import subprocess
from os.path import join as pjoin
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from ugbio_mrd.mrd_utils import (
    generate_synthetic_signatures,
    intersect_featuremap_with_signature,
    read_intersection_dataframes,
    read_signature,
)
intersection_file_basename = "MRD_test_subsample.MRD_test_subsample_annotated_AF_vcf_gz_mrd_quality_snvs.intersection"


@pytest.fixture
def resources_dir():
    return Path(__file__).parent.parent / "resources"





def _assert_read_signature(signature, expected_signature, expected_columns=None, possibly_null_columns=None):
    expected_columns = expected_columns or [
        "ref",
        "alt",
        "id",
        "qual",
        "af",
    ]
    possibly_null_columns = possibly_null_columns or [
        "id",
        "qual",
    ]
    for c in expected_columns:
        assert c in signature.columns
        if c not in possibly_null_columns:
            assert not signature[c].isnull().all()
            assert (signature[c] == expected_signature[c]).all() or np.allclose(signature[c], expected_signature[c])


def test_read_signature_ug_mutect(tmpdir, resources_dir):
    signature = read_signature(pjoin(resources_dir, "mutect_mrd_signature_test.vcf.gz"), return_dataframes=True)
    signature_no_sample_name = read_signature(
        pjoin(resources_dir, "mutect_mrd_signature_test.no_sample_name.vcf.gz"),
        return_dataframes=True,
    )  # make sure we can read the dataframe even if the sample name could not be deduced from the header
    expected_signature = pd.read_hdf(pjoin(resources_dir, "mutect_mrd_signature_test.expected_output.h5"))
    _assert_read_signature(
        signature,
        expected_signature,
        expected_columns=[
            "ref",
            "alt",
            "id",
            "qual",
            "af",
            "depth_tumor_sample",
            "cycle_skip_status",
            "gc_content",
            "left_motif",
            "right_motif",
            "mutation_type",
        ],
    )
    _assert_read_signature(
        signature_no_sample_name,
        expected_signature,
        expected_columns=[
            "ref",
            "alt",
            "id",
            "qual",
            "af",
            "depth_tumor_sample",
            "cycle_skip_status",
            "gc_content",
            "left_motif",
            "right_motif",
            "mutation_type",
        ],
        possibly_null_columns=["id", "qual", "depth_tumor_sample", "af"],
    )


def test_read_signature_ug_dv(tmpdir, resources_dir):
    signature = read_signature(pjoin(resources_dir, "dv_mrd_signature_test.vcf.gz"), return_dataframes=True)
    expected_signature = pd.read_hdf(pjoin(resources_dir, "dv_mrd_signature_test.expected_output.h5"))
    _assert_read_signature(
        signature,
        expected_signature,
        expected_columns=[
            "ref",
            "alt",
            "id",
            "qual",
            "af",
            "depth_tumor_sample",
            "cycle_skip_status",
            "gc_content",
            "left_motif",
            "right_motif",
            "mutation_type",
        ],
    )


def test_read_signature_external(resources_dir):
    signature = read_signature(pjoin(resources_dir, "external_somatic_signature.vcf.gz"), return_dataframes=True)
    expected_signature = pd.read_hdf(pjoin(resources_dir, "external_somatic_signature.expected_output.h5"))

    _assert_read_signature(signature, expected_signature)


def test_intersect_featuremap_with_signature(tmpdir, resources_dir):
    signature_file = pjoin(resources_dir, "Pa_46.FreshFrozen.chr20.70039_70995.vcf.gz")
    featuremap_file = pjoin(resources_dir, "Pa_46.bsDNA.chr20_sample.vcf.gz")
    test_file = pjoin(resources_dir, "intersected_featuremap.vcf.gz")

    output_intersection_file = pjoin(tmpdir, "intersected.vcf.gz")
    intersect_featuremap_with_signature(
        featuremap_file,
        signature_file,
        output_intersection_file=output_intersection_file,
    )
    cmd1 = f"bcftools view -H {output_intersection_file}"
    cmd2 = f"bcftools view -H {test_file}"
    assert subprocess.check_output(cmd1, shell=True) == subprocess.check_output(cmd2, shell=True)


def test_read_intersection_dataframes(tmpdir, resources_dir):
    parsed_intersection_dataframe = read_intersection_dataframes(
        pjoin(resources_dir, f"{intersection_file_basename}.expected_output.parquet"),
        return_dataframes=True,
    )
    parsed_intersection_dataframe_expected = pd.read_parquet(
        pjoin(resources_dir, f"{intersection_file_basename}.parsed.expected_output.parquet")
    )
    parsed_intersection_dataframe2 = read_intersection_dataframes(
        [pjoin(resources_dir, f"{intersection_file_basename}.expected_output.parquet")],
        return_dataframes=True,
    )
    assert_frame_equal(
        parsed_intersection_dataframe.reset_index(),
        parsed_intersection_dataframe_expected,
    )
    assert_frame_equal(
        parsed_intersection_dataframe2.reset_index(),
        parsed_intersection_dataframe_expected,
    )


def test_generate_synthetic_signatures(tmpdir, resources_dir):
    signature_file = pjoin(resources_dir, "mutect_mrd_signature_test.vcf.gz")
    db_file = pjoin(
        resources_dir,
        "pancan_pcawg_2020.mutations_hg38_GNOMAD_dbsnp_beds.sorted.Annotated.HMER_LEN.edited.chr19.vcf.gz",
    )
    synthetic_signature_list = generate_synthetic_signatures(
        signature_vcf=signature_file, db_vcf=db_file, n_synthetic_signatures=1, output_dir=tmpdir
    )
    signature = read_signature(synthetic_signature_list[0], return_dataframes=True)
    expected_signature = read_signature(pjoin(resources_dir, "synthetic_signature_test.vcf.gz"), return_dataframes=True)
    # test that motif distribution is the same (0th order)
    assert (
        signature.groupby(["ref", "alt"]).value_counts() == expected_signature.groupby(["ref", "alt"]).value_counts()
    ).all()
