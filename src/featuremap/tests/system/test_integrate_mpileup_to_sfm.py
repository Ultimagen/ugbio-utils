import os
from collections import defaultdict
from os.path import join as pjoin
from pathlib import Path

import pysam
import pytest
from ugbio_featuremap import integrate_mpileup_to_sfm


@pytest.fixture
def resources_dir():
    return Path(__file__).parent.parent / "resources"


def count_num_variants(vcf):
    # count the number of variants (excluding the header)
    cons_dict = defaultdict(dict)
    pass_count = 0
    for rec in pysam.VariantFile(vcf):
        rec_id = (rec.chrom, rec.pos, rec.ref, rec.alts[0])
        if rec_id not in cons_dict:
            cons_dict[rec_id]["count"] = 0
        cons_dict[rec_id]["count"] += 1

        # Count records with PASS filter
        if rec.filter.keys() == ["PASS"]:
            pass_count += 1

    return len(cons_dict), pass_count


def validate_vcf_format_fields(vcf_path, expected_fields):
    """
    Validate that all expected FORMAT fields exist in header,
    and that every record has values for these fields for all samples.

    Parameters
    ----------
    vcf_path : str
        Path to VCF file to check
    expected_fields : list of str
        FORMAT field IDs expected in the VCF

    Returns
    -------
    bool
        True if all checks pass, False otherwise
    """
    vcf = pysam.VariantFile(vcf_path)
    header_fields = set(vcf.header.formats.keys())
    missing_header_fields = [f for f in expected_fields if f not in header_fields]

    if missing_header_fields:
        print(f"Missing FORMAT fields in header: {missing_header_fields}")
        return False

    # Check that every record has values for all expected fields
    for rec in vcf.fetch():
        for sample in rec.samples:
            for fmt_field in expected_fields:
                # Skip if field is absent in this record
                if fmt_field not in rec.format:
                    print(f"Record {rec.chrom}:{rec.pos} sample {sample} missing field {fmt_field}")
                    return False
                # Check value is not None
                val = rec.samples[sample].get(fmt_field, None)
                if val is None:
                    print(f"Record {rec.chrom}:{rec.pos} sample {sample} has None for {fmt_field}")
                    return False

    print(f"All expected FORMAT fields present and populated in {vcf_path}")
    return True


def test_mpileup_info_integration_to_merged_vcf(tmp_path, resources_dir):
    sfm_vcf = pjoin(resources_dir, "Pa_46.tumor_normal.merged.tumor_PASS.tr_info.chr1_3425000-4016800.vcf.gz")
    tumor_mpileup_vcf = pjoin(resources_dir, "Pa_46_FreshFrozen.cov60.cram_minipileup.chr1_3425000-4016800.pileup")
    normal_mpileup_vcf = pjoin(resources_dir, "Pa_46_Buffycoat.cov30.cram_minipileup.chr1_3425000-4016800.pileup")
    distance_start_to_center = "2"
    out_dir = tmp_path

    expected_out_vcf = pjoin(
        resources_dir, "expected_Pa_46.tumor_normal.merged.tumor_PASS.tr_info.chr1_3425000-4016800.mpileup.vcf.gz"
    )

    out_sfm_vcf = pjoin(out_dir, os.path.basename(sfm_vcf).replace(".vcf.gz", ".mpileup.vcf.gz"))

    # Run the script's main function
    integrate_mpileup_to_sfm.run(
        [
            "integrate_mpileup_to_sfm",
            "--sfm_vcf",
            sfm_vcf,
            "--tumor_mpileup",
            tumor_mpileup_vcf,
            "--normal_mpileup",
            normal_mpileup_vcf,
            "--distance_start_to_center",
            distance_start_to_center,
            "--out_directory",
            str(out_dir),
        ]
    )

    # check that the output file exists and has the expected content
    assert os.path.isfile(out_sfm_vcf)

    # count the number of variants (excluding the header)
    out_num_variants, out_pass_count = count_num_variants(out_sfm_vcf)
    expected_num_variants, expected_pass_count = count_num_variants(expected_out_vcf)
    assert expected_num_variants == out_num_variants
    assert expected_pass_count == out_pass_count

    mpileup_format_fields = ["ref_counts_pm_2", "nonref_counts_pm_2"]
    assert validate_vcf_format_fields(out_sfm_vcf, mpileup_format_fields)
