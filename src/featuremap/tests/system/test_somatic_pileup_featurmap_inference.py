import os
from collections import defaultdict
from os.path import join as pjoin
from pathlib import Path

import pysam
import pytest
from ugbio_featuremap import somatic_pileup_featuremap_inference


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


def test_somatic_pileup_featuremap_inference(tmp_path, resources_dir):
    in_sfmp = pjoin(
        resources_dir, "expected_Pa_46.tumor_normal.merged.tumor_PASS.tr_info.chr1_3425000-4016800.mpileup.vcf.gz"
    )
    xgb_model = pjoin(resources_dir, "model_no_overfitting.HCR_AD_TRdata_5PosAD_noEDV.json")
    out_dir = tmp_path

    expected_out_sfmp_vcf = pjoin(
        resources_dir,
        "expected_Pa_46.tumor_normal.merged.tumor_PASS.tr_info.chr1_3425000-4016800.mpileup.xgb_proba.vcf.gz",
    )

    out_sfmp_vcf = pjoin(out_dir, os.path.basename(in_sfmp).replace(".vcf.gz", ".xgb_proba.vcf.gz"))
    # Run the script's main function
    somatic_pileup_featuremap_inference.run(
        [
            "somatic_pileup_featuremap_inference",
            "--in_sfmp",
            in_sfmp,
            "--xgb_model",
            xgb_model,
            "--out_directory",
            str(out_dir),
        ]
    )

    # check that the output file exists and has the expected content
    assert os.path.isfile(out_sfmp_vcf)

    # count the number of variants (excluding the header)
    out_num_variants = count_num_variants(out_sfmp_vcf)
    expected_num_variants = count_num_variants(expected_out_sfmp_vcf)
    assert expected_num_variants == out_num_variants

    # check that the output file has xgb_proba in the INFO fields
    out_vcf = pysam.VariantFile(out_sfmp_vcf)
    assert "xgb_proba" in out_vcf.header.info.keys()
    count_xgb_values = 0
    for rec in out_vcf:
        if rec.info.get("xgb_proba") is not None:
            count_xgb_values += 1
    expected_vcf = pysam.VariantFile(expected_out_sfmp_vcf)
    expected_count_xgb_values = 0
    for rec in expected_vcf:
        if rec.info.get("xgb_proba") is not None:
            expected_count_xgb_values += 1
    assert count_xgb_values == expected_count_xgb_values
