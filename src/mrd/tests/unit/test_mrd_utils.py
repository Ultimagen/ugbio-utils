from os.path import join as pjoin
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal
from ugbio_mrd.mrd_utils import (
    generate_synthetic_signatures,
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
            assert not signature[c].isna().all()
            assert (signature[c] == expected_signature[c]).all() or np.allclose(signature[c], expected_signature[c])


def test_read_signature_ug_mutect(tmpdir, resources_dir):
    signature = read_signature(pjoin(resources_dir, "mutect_mrd_signature_test.vcf.gz"), return_dataframes=True)
    signature_no_sample_name = read_signature(
        pjoin(resources_dir, "mutect_mrd_signature_test.no_sample_name.vcf.gz"),
        return_dataframes=True,
        tumor_sample="_10_FFPE",
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


def test_read_signature_pileup_featuremap(resources_dir):
    signature = read_signature(
        pjoin(resources_dir, "featuremap_pileup_mrd_signature_test.vcf.gz"), return_dataframes=True
    )
    expected_signature = pd.read_hdf(pjoin(resources_dir, "featuremap_pileup_mrd_signature_test.expected_output.h5"))

    _assert_read_signature(signature, expected_signature)


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
