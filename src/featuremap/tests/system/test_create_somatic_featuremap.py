import os
from collections import defaultdict
from os.path import join as pjoin
from pathlib import Path

import pysam
import pytest
from ugbio_featuremap import create_somatic_featuremap


@pytest.fixture
def resources_dir():
    return Path(__file__).parent.parent / "resources"


def test_create_somatic_featuremap(tmp_path, resources_dir):
    tumor_vcf = pjoin(resources_dir, "HG006_HG003.featuremap.chr9.vcf.gz")
    normal_vcf = pjoin(
        resources_dir,
        "HG003_sim_028059.normal_in_tumor.featuremap.chr9.vcf.gz",
    )
    sample_name = "HG006_HG003_vs_HG003_chr9"
    out_dir = tmp_path

    expected_tumor_pass_vcf = pjoin(resources_dir, "TP_HG006_HG003.tumor_normal.merged.chr9.vcf.gz")  # noqa: F841
    expected_num_variants = 1333
    out_tumor_vcf = pjoin(out_dir, f"{sample_name}.tumor_normal.merged.vcf.gz")

    # Run the script's main function
    create_somatic_featuremap.run(
        [
            "create_somatic_featuremap",
            "--tumor_vcf",
            tumor_vcf,
            "--normal_vcf",
            normal_vcf,
            "--sample_name",
            sample_name,
            "--out_directory",
            str(out_dir),
        ]
    )

    # check that the output file exists and has the expected content
    assert os.path.isfile(out_tumor_vcf)
    # count the number of variants (excluding the header)
    cons_dict = defaultdict(dict)
    for rec in pysam.VariantFile(out_tumor_vcf):
        rec_id = (rec.chrom, rec.pos, rec.ref, rec.alts[0])
        if rec_id not in cons_dict:
            cons_dict[rec_id]["count"] = 0
        cons_dict[rec_id]["count"] += 1
    # check the number of variants
    num_variants = len(cons_dict)
    assert num_variants == expected_num_variants
    # assert a single entry per variant
    assert all(cons["count"] == 1 for cons in cons_dict.values())
