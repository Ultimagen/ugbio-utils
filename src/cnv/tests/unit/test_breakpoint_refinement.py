"""
Unit tests for CNV breakpoint refinement module.
"""

import pysam
import pytest
from ugbio_cnv.breakpoint_refinement import (
    estimate_refined_breakpoints,
    extract_reads_windowed,
    refine_cnv_breakpoints_from_vcf,
)


@pytest.fixture
def temp_bam_with_rg_tags(tmp_path):
    """Create synthetic BAM with multiple RG tags per read."""
    bam_path = tmp_path / "test.bam"
    header = {
        "HD": {"VN": "1.0"},
        "SQ": [{"SN": "chr1", "LN": 10000000}],
        "RG": [{"ID": "DEL"}, {"ID": "DUP"}],
    }

    with pysam.AlignmentFile(str(bam_path), "wb", header=header) as bam:
        # Create read with RG=DEL and SA tag
        read1 = pysam.AlignedSegment()
        read1.query_name = "read001"
        read1.reference_id = 0
        read1.reference_start = 999900  # Near start boundary (1000000)
        read1.query_sequence = "A" * 100
        read1.cigarstring = "100M"
        read1.set_tag("RG", "DEL")
        read1.set_tag("SA", "chr1,999950,+,50M50S,60,0;")
        bam.write(read1)

        # Same read name but RG=DUP (should be tracked separately)
        read2 = pysam.AlignedSegment()
        read2.query_name = "read001"
        read2.reference_id = 0
        read2.reference_start = 999900
        read2.query_sequence = "A" * 100
        read2.cigarstring = "100M"
        read2.set_tag("RG", "DUP")
        read2.set_tag("SA", "chr1,999960,+,50M50S,60,0;")
        bam.write(read2)

        # Additional DEL reads for median calculation
        for i in range(2, 5):
            read = pysam.AlignedSegment()
            read.query_name = f"read{i:03d}"
            read.reference_id = 0
            read.reference_start = 999900 + i * 10
            read.query_sequence = "A" * 100
            read.cigarstring = "100M"
            read.set_tag("RG", "DEL")
            read.set_tag("SA", f"chr1,{999950 + i * 5},+,50M50S,60,0;")
            bam.write(read)

        # Right window reads
        for i in range(5, 8):
            read = pysam.AlignedSegment()
            read.query_name = f"read{i:03d}"
            read.reference_id = 0
            read.reference_start = 1002900 + i * 10
            read.query_sequence = "A" * 100
            read.cigarstring = "100M"
            read.set_tag("RG", "DEL")
            read.set_tag("SA", f"chr1,{1003050 + i * 5},+,50M50S,60,0;")
            bam.write(read)

    pysam.index(str(bam_path))
    return bam_path


@pytest.fixture
def temp_vcf_with_cipos(tmp_path):
    """Create VCF with CNVs and CIPOS fields."""
    vcf_path = tmp_path / "input.vcf"
    header = pysam.VariantHeader()
    header.add_line('##INFO=<ID=CIPOS,Number=2,Type=Integer,Description="CIPOS">')
    header.add_line('##INFO=<ID=SVTYPE,Number=1,Type=String,Description="Type">')
    header.add_line('##INFO=<ID=END,Number=1,Type=Integer,Description="End">')
    header.contigs.add("chr1", length=10000000)

    with pysam.VariantFile(str(vcf_path), "w", header=header) as vcf:
        record = vcf.new_record()
        record.chrom = "chr1"
        record.pos = 1000000
        record.stop = 1003000
        record.info["CIPOS"] = (-500, 500)
        record.info["SVTYPE"] = "DEL"
        vcf.write(record)

    return vcf_path


def test_extract_reads_windowed_rg_filtering(temp_bam_with_rg_tags):
    """Test RG filtering extracts only matching reads."""
    left, right, seen = extract_reads_windowed(
        bam_files=[str(temp_bam_with_rg_tags)],
        cnv_chrom="chr1",
        cnv_start=1000000,
        cnv_end=1003000,
        cnv_type="DEL",
        cushion=2500,
    )

    # Should have 4 DEL reads in left window (read001 with RG=DEL + read002-004)
    assert len(left) == 4
    assert all(read.get_tag("RG") == "DEL" for read in left)

    # Should have 3 DEL reads in right window
    assert len(right) == 3
    assert all(read.get_tag("RG") == "DEL" for read in right)


def test_extract_reads_windowed_deduplication(temp_bam_with_rg_tags):
    """Test (read_name, RG) deduplication."""
    _, _, seen = extract_reads_windowed(
        bam_files=[str(temp_bam_with_rg_tags)],
        cnv_chrom="chr1",
        cnv_start=1000000,
        cnv_end=1003000,
        cnv_type="DEL",
        cushion=2500,
    )

    # Should track (read_name, RG) pairs
    assert ("read001", "DEL") in seen
    # Should NOT have ("read001", "DUP") because we filtered by cnv_type="DEL"


def test_extract_reads_windowed_multi_bam(temp_bam_with_rg_tags, tmp_path):
    """Test merging reads from multiple BAMs."""
    # Copy BAM to create second file
    bam2_path = tmp_path / "test2.bam"
    import shutil

    shutil.copy(temp_bam_with_rg_tags, bam2_path)
    pysam.index(str(bam2_path))

    left, right, seen = extract_reads_windowed(
        bam_files=[str(temp_bam_with_rg_tags), str(bam2_path)],
        cnv_chrom="chr1",
        cnv_start=1000000,
        cnv_end=1003000,
        cnv_type="DEL",
        cushion=2500,
    )

    # With deduplication, should still have same counts (same reads in both BAMs)
    assert len(left) == 4
    assert len(right) == 3


def test_estimate_refined_breakpoints_median_calculation():
    """Test median and max deviation calculation from soft-clipped reads."""
    # Create mock reads with soft clips at known positions
    # Left breakpoint reads: soft clip at start, positions [999950, 999955, 999960, 999965]
    # median=999957.5 → 999958
    reads_left = []
    for pos in [999950, 999955, 999960, 999965]:
        read = pysam.AlignedSegment()
        read.query_name = f"read_{pos}"
        read.reference_start = pos - 1  # Convert 1-based to 0-based
        read.cigartuples = [(4, 50), (0, 100)]  # 50S100M - left soft clip
        reads_left.append(read)

    # Right breakpoint reads: soft clip at end, positions [1003050, 1003055, 1003060, 1003065]
    # median=1003057.5 → 1003058
    reads_right = []
    for pos in [1003050, 1003055, 1003060, 1003065]:
        read = pysam.AlignedSegment()
        read.query_name = f"read_{pos}"
        read.reference_start = pos - 100  # Start 100bp before the breakpoint
        read.cigartuples = [(0, 100), (4, 50)]  # 100M50S - right soft clip
        # reference_end will be reference_start + 100 = pos
        reads_right.append(read)

    result = estimate_refined_breakpoints(
        left_reads=reads_left,
        right_reads=reads_right,
        original_cipos=(-500, 500),  # Interval size: 1000
    )

    assert result is not None
    refined_start, refined_end, refined_cipos = result

    # Check medians
    assert refined_start == 999958  # median of [999950, 999955, 999960, 999965]
    assert refined_end == 1003058  # median of [1003050, 1003055, 1003060, 1003065]

    # Max deviation from median: max(|999950-999958|, |999965-999958|) = 8
    # CIPOS should be (-8, 8) → interval size: 16
    assert refined_cipos == (-8, 8)


def test_estimate_refined_breakpoints_no_improvement():
    """Test returns None when refined CIPOS not tighter."""
    # Create reads with wide spread - refined interval won't be tighter
    reads_left = []
    for pos in [999000, 999500, 1000000, 1000500]:  # Wide spread
        read = pysam.AlignedSegment()
        read.reference_start = pos - 1  # Convert 1-based to 0-based
        read.cigartuples = [(4, 50), (0, 100)]  # 50S100M - left soft clip
        reads_left.append(read)

    reads_right = []
    for pos in [1003000, 1003500, 1004000, 1004500]:  # Wide spread
        read = pysam.AlignedSegment()
        read.reference_start = pos - 100  # Start 100bp before the breakpoint
        read.cigartuples = [(0, 100), (4, 50)]  # 100M50S - right soft clip
        reads_right.append(read)

    result = estimate_refined_breakpoints(reads_left, reads_right, original_cipos=(-100, 100))

    # Should return None because refined interval (±750) > original interval (200)
    assert result is None


def test_estimate_refined_breakpoints_insufficient_reads():
    """Test returns None with <3 reads per breakpoint."""
    # Only 2 reads for left breakpoint (less than MIN_READS_PER_BREAKPOINT=3)
    reads_left = []
    for i in range(2):
        read = pysam.AlignedSegment()
        pos = 1000000 + i * 10
        read.reference_start = pos - 1  # Convert 1-based to 0-based
        read.cigartuples = [(4, 50), (0, 100)]  # 50S100M - left soft clip
        reads_left.append(read)

    # 4 reads for right breakpoint (enough)
    reads_right = []
    for i in range(4):
        read = pysam.AlignedSegment()
        pos = 1003000 + i * 10
        read.reference_start = pos - 100
        read.cigartuples = [(0, 100), (4, 50)]  # 100M50S - right soft clip
        reads_right.append(read)

    result = estimate_refined_breakpoints(reads_left, reads_right, (-500, 500))

    # Should return None because left has <3 reads
    assert result is None


def test_refine_cnv_breakpoints_from_vcf(temp_vcf_with_cipos, temp_bam_with_rg_tags, tmp_path):
    """Test end-to-end VCF refinement."""
    output_vcf = tmp_path / "output.vcf"

    refine_cnv_breakpoints_from_vcf(
        input_vcf=str(temp_vcf_with_cipos),
        bam_files=[str(temp_bam_with_rg_tags)],
        output_vcf=str(output_vcf),
        cushion=2500,
    )

    # Verify output VCF exists and has records
    vcf_out = pysam.VariantFile(str(output_vcf))
    records = list(vcf_out)
    assert len(records) == 1

    record = records[0]
    # Should have CIPOS field (either original or refined)
    assert "CIPOS" in record.info


def test_refine_cnv_breakpoints_from_vcf_missing_cipos(tmp_path):
    """Test raises ValueError when CIPOS missing."""
    # Create VCF without CIPOS
    vcf_path = tmp_path / "no_cipos.vcf"
    header = pysam.VariantHeader()
    header.add_line('##INFO=<ID=SVTYPE,Number=1,Type=String,Description="Type">')
    header.contigs.add("chr1", length=10000000)

    with pysam.VariantFile(str(vcf_path), "w", header=header) as vcf:
        record = vcf.new_record()
        record.chrom = "chr1"
        record.pos = 1000000
        record.stop = 1003000
        record.info["SVTYPE"] = "DEL"
        vcf.write(record)

    output_vcf = tmp_path / "output.vcf"

    with pytest.raises(ValueError, match="missing CIPOS field"):
        refine_cnv_breakpoints_from_vcf(
            input_vcf=str(vcf_path),
            bam_files=[],
            output_vcf=str(output_vcf),
        )


def test_refine_cnv_breakpoints_from_vcf_sv_vcf_not_implemented(temp_vcf_with_cipos, tmp_path):
    """Test raises NotImplementedError when sv_vcf provided."""
    output_vcf = tmp_path / "output.vcf"

    with pytest.raises(NotImplementedError, match="SV VCF integration not yet implemented"):
        refine_cnv_breakpoints_from_vcf(
            input_vcf=str(temp_vcf_with_cipos),
            bam_files=[],
            output_vcf=str(output_vcf),
            sv_vcf="sv_calls.vcf",
        )
