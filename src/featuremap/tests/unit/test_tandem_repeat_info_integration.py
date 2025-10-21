import os
from collections import defaultdict
from os.path import join as pjoin
from pathlib import Path

import pysam
import pytest
from ugbio_featuremap.somatic_featuremap_utils import integrate_tandem_repeat_features


@pytest.fixture
def resources_dir():
    return Path(__file__).parent.parent / "resources"


def count_num_variants(vcf):
    # count the number of variants (excluding the header)
    cons_dict = defaultdict(dict)
    for rec in pysam.VariantFile(vcf):
        rec_id = (rec.chrom, rec.pos, rec.ref, rec.alts[0])
        if rec_id not in cons_dict:
            cons_dict[rec_id]["count"] = 0
        cons_dict[rec_id]["count"] += 1
    return len(cons_dict)


def assert_vcf_info_fields(vcf_path, expected_fields):
    """
    Assert that a VCF file contains all expected INFO fields in its header.
    """
    vcf = pysam.VariantFile(vcf_path)
    info_fields = set(vcf.header.info.keys())

    missing = [field for field in expected_fields if field not in info_fields]
    assert not missing, f"Missing INFO fields: {', '.join(missing)}"


def test_integrate_tandem_repeat_features(
    tmpdir,
    resources_dir,
):
    input_merged_vcf = pjoin(resources_dir, "TP_HG006_HG003.tumor_normal.merged.chr9.vcf.gz")
    ref_tr_file = pjoin(resources_dir, "tr.chr9.bed")
    expected_out_vcf = pjoin(resources_dir, "TP_HG006_HG003.tumor_normal.merged.chr9.tr_info.vcf.gz")

    # call the function with different arguments
    out_vcf_with_tr_data = integrate_tandem_repeat_features(input_merged_vcf, ref_tr_file, tmpdir)

    # check that the output file exists and has the expected content
    assert os.path.isfile(out_vcf_with_tr_data)

    # count the number of variants (excluding the header)
    out_num_variants = count_num_variants(out_vcf_with_tr_data)
    expected_num_variants = count_num_variants(expected_out_vcf)
    assert expected_num_variants == out_num_variants

    # check that header has the TR info fields
    # Example usage
    expected_info_fields = ["TR_START", "TR_END", "TR_SEQ", "TR_DISTANCE", "TR_LENGTH", "TR_SEQ_UNIT_LENGTH"]
    assert_vcf_info_fields(out_vcf_with_tr_data, expected_info_fields)
