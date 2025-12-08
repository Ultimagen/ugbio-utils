import os
from collections import defaultdict
from os.path import join as pjoin
from pathlib import Path

import pysam
import pytest
from ugbio_featuremap.somatic_featuremap_utils import _find_closest_tandem_repeats, integrate_tandem_repeat_features


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
    genome_file = pjoin(resources_dir, "Homo_sapiens_assembly38.fasta.fai")

    # call the function with different arguments
    out_vcf_with_tr_data = integrate_tandem_repeat_features(input_merged_vcf, ref_tr_file, genome_file, tmpdir)

    # check that the output file exists and has the expected content
    assert os.path.isfile(out_vcf_with_tr_data)

    # count the number of variants (excluding the header)
    out_num_variants = count_num_variants(out_vcf_with_tr_data)
    expected_num_variants = count_num_variants(expected_out_vcf)
    assert expected_num_variants == out_num_variants

    # check that header has the TR info fields
    # Example usage
    expected_info_fields = ["TR_START", "TR_END", "TR_SEQ", "TR_LENGTH", "TR_SEQ_UNIT_LENGTH", "TR_DISTANCE"]
    assert_vcf_info_fields(out_vcf_with_tr_data, expected_info_fields)


def test_find_closest_tandem_repeats_all_records_output(tmpdir, resources_dir):
    """
    Test that _find_closest_tandem_repeats outputs all records from bed1 to the output file.
    """
    # Create test BED files
    bed1_path = pjoin(tmpdir, "variants.bed")
    bed2_path = pjoin(tmpdir, "tandem_repeats.bed")
    output_path = pjoin(tmpdir, "closest_tr.tsv")
    genome_file = pjoin(resources_dir, "Homo_sapiens_assembly38.fasta.fai")

    # Create bed1 with test variants
    bed1_data = [
        "chr1\t1000\t1001",
        "chr1\t2000\t2001",
        "chr2\t3000\t3001",
        "chr2\t4000\t4001",
        "chr3\t5000\t5001",
        "chr12\t5000\t5001",
        "chr12\t7000\t7001",
    ]

    # Create bed2 with test tandem repeats
    bed2_data = [
        "chr1\t900\t950\tTR1\t50\tA\t1",
        "chr1\t1800\t1900\tTR2\t100\tAT\t2",
        "chr2\t2800\t2850\tTR3\t50\tAAT\t3",
        "chr2\t4200\t4300\tTR4\t100\tAAAA\t4",
        "chr3\t4800\t4850\tTR5\t50\tGC\t2",
        "chr12\t4800\t4850\tTR6\t50\tGC\t2",
    ]

    # Write test files
    with open(bed1_path, "w") as f:
        f.write("\n".join(bed1_data) + "\n")

    with open(bed2_path, "w") as f:
        f.write("\n".join(bed2_data) + "\n")

    # Call the function under test
    _find_closest_tandem_repeats(bed1_path, bed2_path, genome_file, output_path)

    # Check that output file exists
    assert os.path.exists(output_path)

    # Read the output and count records
    with open(output_path) as f:
        output_lines = f.readlines()

    # Remove empty lines
    output_lines = [line.strip() for line in output_lines if line.strip()]

    # Check that all input records from bed1 are in the output
    assert len(output_lines) == len(bed1_data), f"Expected {len(bed1_data)} output records, got {len(output_lines)}"

    # Verify that each original variant position appears in the output
    input_positions = set()
    for line in bed1_data:
        parts = line.split("\t")
        chrom, start = parts[0], parts[1]
        input_positions.add((chrom, start))

    output_positions = set()
    for line in output_lines:
        parts = line.split("\t")
        if len(parts) >= 2:
            chrom, start = parts[0], parts[1]
            output_positions.add((chrom, str(int(start) - 1)))

    # Assert that all input positions are found in output
    missing_positions = input_positions - output_positions
    assert not missing_positions, f"Missing positions in output: {missing_positions}"

    # Verify output has expected structure (should have more columns due to bedtools closest)
    for line in output_lines:
        parts = line.split("\t")
        assert len(parts) >= 6, f"Output line should have at least 6 columns, got {len(parts)}: {line}"
