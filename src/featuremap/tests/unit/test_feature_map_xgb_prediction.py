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


def test_pileup_featuremap(
    tmpdir,
    resources_dir,
):
    input_featuremap_vcf = pjoin(resources_dir, "featuremap_vcf_for_xgb_prediction.vcf.gz")
    interval_list_file = pjoin(resources_dir, "wgs_calling_regions.hg38.chr9_test.interval_list")
    model_file = pjoin(resources_dir, "xgb_model.alt_reads_3.half_chr1.json")
    out_vcf = pjoin(tmpdir, "out_featuremap_with_xgb_proba.vcf.gz")
    expected_num_variants = 342

    # call the function with different arguments
    pileup_featuremap_with_agg_params_and_xgb_proba_on_an_interval_list(
        input_featuremap_vcf, out_vcf, interval_list_file, model_file
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
