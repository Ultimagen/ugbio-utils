import os
from collections import defaultdict
from os.path import join as pjoin
from pathlib import Path

import pysam
import pytest
from ugbio_featuremap.featuremap_xgb_prediction import (
    pileup_featuremap_with_agg_params_and_xgb_proba_on_an_interval_list,
)


@pytest.fixture
def resources_dir():
    return Path(__file__).parent.parent / "resources"


def test_pileup_featuremap_ppmseq_xgb_prediction(
    tmpdir,
    resources_dir,
):
    input_featuremap_vcf = pjoin(resources_dir, "five_giab.ppmSeq.dup_filtered.pileup.chr9.vcf.gz")
    interval_list_file = pjoin(resources_dir, "wgs_calling_regions.hg38.chr9_test.interval_list")
    model_file = pjoin(resources_dir, "ppmSeq_chr1_alt2_8.test_size_0.1.xgb_model.json")
    out_vcf = pjoin(tmpdir, "out_ppmSeq_featuremap_with_xgb_proba.vcf.gz")
    filter_tags = "PASS"
    expected_num_variants = 3702

    # call the function with different arguments
    pileup_featuremap_with_agg_params_and_xgb_proba_on_an_interval_list(
        input_featuremap_vcf, out_vcf, interval_list_file, filter_tags, model_file, write_agg_params=False
    )

    # check that the output file exists and has the expected content
    assert os.path.isfile(out_vcf)
    # count the number of variants (excluding the header)
    cons_dict = defaultdict(dict)
    for rec in pysam.VariantFile(out_vcf):
        rec_id = (rec.chrom, rec.pos, rec.ref, rec.alts[0])
        if rec_id not in cons_dict:
            cons_dict[rec_id]["count"] = 0
        cons_dict[rec_id]["count"] += 1
    # check the number of variants
    num_variants = len(cons_dict)
    assert num_variants == expected_num_variants
    # assert a single entry per variant
    assert all(cons["count"] == 1 for cons in cons_dict.values())


def test_pileup_featuremap_standard_xgb_prediction(
    tmpdir,
    resources_dir,
):
    input_featuremap_vcf = pjoin(resources_dir, "five_giab.Standard-WG.dup_filtered.pileup.chr9.vcf.gz")
    interval_list_file = pjoin(resources_dir, "wgs_calling_regions.hg38.chr9_test.interval_list")
    model_file = pjoin(resources_dir, "standard_chr1_alt2_8.test_size_0.1.xgb_model.json")
    out_vcf = pjoin(tmpdir, "out_standard_featuremap_with_xgb_proba.vcf.gz")
    filter_tags = "PASS"
    expected_num_variants = 45130

    # call the function with different arguments
    pileup_featuremap_with_agg_params_and_xgb_proba_on_an_interval_list(
        input_featuremap_vcf, out_vcf, interval_list_file, filter_tags, model_file, write_agg_params=False
    )

    # check that the output file exists and has the expected content
    assert os.path.isfile(out_vcf)
    # count the number of variants (excluding the header)
    cons_dict = defaultdict(dict)
    for rec in pysam.VariantFile(out_vcf):
        rec_id = (rec.chrom, rec.pos, rec.ref, rec.alts[0])
        if rec_id not in cons_dict:
            cons_dict[rec_id]["count"] = 0
        cons_dict[rec_id]["count"] += 1
    # check the number of variants
    num_variants = len(cons_dict)
    assert num_variants == expected_num_variants
    # assert a single entry per variant
    assert all(cons["count"] == 1 for cons in cons_dict.values())
