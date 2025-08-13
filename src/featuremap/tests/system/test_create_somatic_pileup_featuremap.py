import os
from collections import defaultdict
from os.path import join as pjoin
from pathlib import Path

import pysam
import pytest
from ugbio_featuremap import create_somatic_pileup_featuremap, ref_nonref_per_base_window


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


def test_parse_samtools_mpileup(tmp_path, resources_dir):
    samtools_mpileup_file = pjoin(
        resources_dir,
        "Pa_46_FreshFrozen.Lb_705.runs_021146_021152_cov60.chr19_chr20.Pa_46_FreshFrozen.Lb_705.runs_021146_021152_cov60.xgb.pileup.xgb.chr19_chr20.m3_p2.bed.ver2.100.samtools.mpileup",
    )
    regions_bed_file = pjoin(
        resources_dir, "Pa_46_FreshFrozen.Lb_705.runs_021146_021152_cov60.xgb.pileup.xgb.chr19_chr20.m3_p2.bed"
    )
    base_file_name = "create_somatic_pileup_featuremap_test_output"
    out_dir = tmp_path
    output_vcf = ref_nonref_per_base_window.run(
        [
            "ref_nonref_per_base_window",
            "--input",
            samtools_mpileup_file,
            "--bed",
            regions_bed_file,
            "--distance_start_to_center",
            "2",
            "--output_dir",
            str(out_dir),
            "--base_file_name",
            base_file_name,
        ]
    )
    # output_vcf = pjoin(out_dir, f"{base_file_name}.vcf.gz")
    expected_vcf_output = pjoin(
        resources_dir,
        "Pa_46_FreshFrozen.Lb_705.runs_021146_021152_cov60.chr19_chr20.Pa_46_FreshFrozen.Lb_705.runs_021146_021152_cov60.xgb.pileup.xgb.chr19_chr20.m3_p2.bed.ver2.100.samtools.mpileup.vcf.gz",
    )
    # assert output file was generated
    assert os.path.isfile(output_vcf)
    # check that the output VCF matches the expected output
    with pysam.VariantFile(output_vcf) as out_vcf, pysam.VariantFile(expected_vcf_output) as exp_vcf:
        for out_rec, exp_rec in zip(out_vcf, exp_vcf):
            assert out_rec == exp_rec
