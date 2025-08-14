import os
from collections import defaultdict
from os.path import join as pjoin
from pathlib import Path

import pysam
import pytest
from ugbio_featuremap import create_somatic_pileup_featuremap


@pytest.fixture
def resources_dir():
    return Path(__file__).parent.parent / "resources"


def test_create_somatic_pileup_featuremap(tmp_path, resources_dir):
    tumor_vcf = pjoin(
        resources_dir, "Pa_46_FreshFrozen.Lb_705.runs_021146_021152_cov60.xgb.pileup.xgb.chr19_27240875_32924245.vcf.gz"
    )
    normal_vcf = pjoin(
        resources_dir,
        "Pa_46_Buffycoat.Lb_744.runs_021145_021151_cov30.xgb.pileup.xgb.chr19.chr19_27240875_32924245.vcf.gz",
    )
    sample_name = "Pa_46_FF_vs_BC"
    ref_tr_file = pjoin(resources_dir, "tr_hg38.chr19.bed")
    out_dir = tmp_path

    expected_tumor_pass_vcf = pjoin(resources_dir, "Pa_46_FF_vs_BC.tumor_normal.merged.tumor_PASS.vcf.gz")  # noqa: F841
    expected_num_variants = 11649
    out_tumor_pass_vcf = pjoin(out_dir, f"{sample_name}.tumor_normal.merged.tumor_PASS.vcf.gz")

    # Run the script's main function
    create_somatic_pileup_featuremap.run(
        [
            "create_somatic_pileup_featuremap",
            "--tumor_vcf",
            tumor_vcf,
            "--normal_vcf",
            normal_vcf,
            "--sample_name",
            sample_name,
            "--ref_tr_file",
            ref_tr_file,
            "--out_directory",
            str(out_dir),
            "--filter_for_tumor_pass_variants",
        ]
    )

    # check that the output file exists and has the expected content
    assert os.path.isfile(out_tumor_pass_vcf)
    # count the number of variants (excluding the header)
    cons_dict = defaultdict(dict)
    for rec in pysam.VariantFile(out_tumor_pass_vcf):
        rec_id = (rec.chrom, rec.pos, rec.ref, rec.alts[0])
        if rec_id not in cons_dict:
            cons_dict[rec_id]["count"] = 0
        cons_dict[rec_id]["count"] += 1
    # check the number of variants
    num_variants = len(cons_dict)
    assert num_variants == expected_num_variants
    # assert a single entry per variant
    assert all(cons["count"] == 1 for cons in cons_dict.values())
