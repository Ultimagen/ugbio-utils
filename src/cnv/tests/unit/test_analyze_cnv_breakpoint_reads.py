"""Tests for analyze_cnv_breakpoint_reads module."""

import tempfile
from pathlib import Path

import pysam
import pytest
from ugbio_cnv.analyze_cnv_breakpoint_reads import (
    analyze_cnv_breakpoints,
    analyze_interval_breakpoints,
    check_read_cnv_consistency,
    get_supplementary_alignments,
)
from ugbio_core.filter_bed import parse_bed_file


@pytest.fixture
def temp_bed_file():
    """Create a temporary BED file for testing."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".bed", delete=False) as f:
        f.write("chr1\t1000\t2000\n")
        f.write("chr1\t5000\t6000\n")
        f.write("chr2\t10000\t11000\n")
        temp_path = f.name
    yield temp_path
    Path(temp_path).unlink()


@pytest.fixture
def temp_bam_file():
    """Create a temporary BAM file for testing."""
    with tempfile.NamedTemporaryFile(suffix=".bam", delete=False) as f:
        temp_path = f.name

    # Create a simple BAM file with header
    header = {
        "HD": {"VN": "1.0"},
        "SQ": [
            {"SN": "chr1", "LN": 100000},
            {"SN": "chr2", "LN": 100000},
        ],
    }

    with pysam.AlignmentFile(temp_path, "wb", header=header) as outf:
        # Create test reads with split alignments (SA tags)
        # Note: reads must be in sorted order for indexing
        # Interval: 1000-2000, cushion: 100
        # Start region: 900-1100, End region: 1900-2100

        # Read with deletion evidence: first part (right clip) BEFORE second part (left clip)
        # First part at position 950 (start region) with right clip, second part at position 2050
        # (end region) with left clip
        read2 = pysam.AlignedSegment()
        read2.query_name = "read2"
        read2.query_sequence = (
            "ACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGT"  # 80 bases
        )
        read2.reference_id = 0  # chr1
        read2.reference_start = 950
        read2.cigartuples = [(0, 50), (4, 30)]  # 50M30S - right soft clip (first part)
        read2.is_reverse = False
        # SA tag: second part at chr1:2051 (1-based), with left clip (30S50M)
        read2.set_tag("SA", "chr1,2051,+,30S50M,60,0;")
        outf.write(read2)

        # Read with duplication evidence: first part (right clip) AFTER second part (left clip)
        # First part at position 2050 (end region) with right clip, second part at position 950
        # (start region) with left clip
        read1 = pysam.AlignedSegment()
        read1.query_name = "read1"
        read1.query_sequence = (
            "ACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGT"  # 80 bases
        )
        read1.reference_id = 0  # chr1
        read1.reference_start = 2050
        read1.cigartuples = [(0, 50), (4, 30)]  # 50M30S - right soft clip (first part)
        read1.is_reverse = False
        # SA tag: second part at chr1:951 (1-based), with left clip (30S50M)
        read1.set_tag("SA", "chr1,951,+,30S50M,60,0;")
        outf.write(read1)

    # Index the BAM file
    pysam.index(temp_path)

    yield temp_path
    Path(temp_path).unlink()
    Path(temp_path + ".bai").unlink(missing_ok=True)


def test_parse_bed_file(temp_bed_file):
    """Test BED file parsing."""
    intervals = parse_bed_file(temp_bed_file)
    assert len(intervals) == 3
    assert intervals[0] == ("chr1", 1000, 2000)
    assert intervals[1] == ("chr1", 5000, 6000)
    assert intervals[2] == ("chr2", 10000, 11000)


def test_get_supplementary_alignments():
    """Test parsing SA tag."""
    read = pysam.AlignedSegment()
    read.query_name = "test_read"
    read.set_tag("SA", "chr1,1000,+,50M30S,60,0;chr1,2000,-,30S50M,60,0;")

    # Need a dummy alignment file for the function signature
    # but it's not actually used when parsing SA tag
    supp_alns = get_supplementary_alignments(None, read)

    assert len(supp_alns) == 2
    # First: 50M means +50 on reference, 30S on right, + strand
    assert supp_alns[0] == ("chr1", 999, False, True, False)  # 0-based, no left clip, has right clip, forward
    # Second: 30S on left, 50M means +50 on reference, - strand
    assert supp_alns[1] == ("chr1", 1999, True, False, True)  # 0-based, has left clip, no right clip, reverse


def test_check_read_cnv_consistency_duplication():
    """Test duplication detection logic."""
    # Create a header for the mock read
    header = {
        "HD": {"VN": "1.0"},
        "SQ": [{"SN": "chr1", "LN": 100000}],
    }

    # Create a mock read (primary alignment) - first part with right clip at position 2050
    # Interval: 1000-2000, cushion: 100
    # Start region: 900-1100, End region: 1900-2100
    read = pysam.AlignedSegment(header=pysam.AlignmentHeader.from_dict(header))
    read.query_name = "test_read"
    read.query_sequence = "ACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGT"  # 80 bases
    read.reference_id = 0  # chr1
    read.reference_start = 2050  # First part at end region (later position)
    read.cigartuples = [(0, 50), (4, 30)]  # 50M30S - right soft clip (first part)
    read.is_reverse = False
    read.is_unmapped = False
    read.is_secondary = False
    read.is_supplementary = False

    # Supplementary alignment - second part with left clip at earlier position in start region
    # Second part (left clip) at earlier position than first part -> duplication
    # Primary is near END (2050), supplementary is near START (950)
    # Same strand (forward) as primary
    supplementary_alns = [("chr1", 950, True, False, False)]  # has left clip, no right clip, forward

    # First part (2050) > Second part (950) -> duplication
    is_dup, is_del = check_read_cnv_consistency(read, 1000, 2000, 100, supplementary_alns)
    assert is_dup is True
    assert is_del is False


def test_check_read_cnv_consistency_deletion():
    """Test deletion detection logic."""
    # Create a header for the mock read
    header = {
        "HD": {"VN": "1.0"},
        "SQ": [{"SN": "chr1", "LN": 100000}],
    }

    # Create a mock read (primary alignment) - first part with right clip at position 950
    # Interval: 1000-2000, cushion: 100
    # Start region: 900-1100, End region: 1900-2100
    read = pysam.AlignedSegment(header=pysam.AlignmentHeader.from_dict(header))
    read.query_name = "test_read"
    read.query_sequence = "ACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGT"  # 80 bases
    read.reference_id = 0  # chr1
    read.reference_start = 950  # First part at start region (earlier position)
    read.cigartuples = [(0, 50), (4, 30)]  # 50M30S - right soft clip (first part)
    read.is_reverse = False
    read.is_unmapped = False
    read.is_secondary = False
    read.is_supplementary = False

    # Supplementary alignment - second part with left clip at later position in end region
    # Second part (left clip) at later position than first part -> deletion
    # Primary is near START (950), supplementary is near END (2050)
    # Same strand (forward) as primary
    supplementary_alns = [("chr1", 2050, True, False, False)]  # has left clip, no right clip, forward

    # First part (950) < Second part (2050) -> deletion
    is_dup, is_del = check_read_cnv_consistency(read, 1000, 2000, 100, supplementary_alns)
    assert is_dup is False
    assert is_del is True


def test_analyze_interval_breakpoints(temp_bam_file):
    """Test interval analysis."""
    with pysam.AlignmentFile(temp_bam_file, "rb") as bam:
        evidence = analyze_interval_breakpoints(bam, "chr1", 1000, 2000, 100)

        assert evidence.chrom == "chr1"
        assert evidence.start == 1000
        assert evidence.end == 2000
        # We expect 2 reads total (one dup, one del)
        assert evidence.total_reads == 2
        assert evidence.duplication_reads == 1
        assert evidence.deletion_reads == 1


def test_analyze_cnv_breakpoints(temp_bam_file, temp_bed_file):
    """Test full analysis workflow."""
    results_df = analyze_cnv_breakpoints(
        bam_file=temp_bam_file,
        bed_file=temp_bed_file,
        cushion=100,
        output_file=None,
    )

    assert len(results_df) == 3  # Three intervals in the BED file
    assert "chrom" in results_df.columns
    assert "start" in results_df.columns
    assert "end" in results_df.columns
    assert "duplication_reads" in results_df.columns
    assert "deletion_reads" in results_df.columns
    assert "total_reads" in results_df.columns

    # First interval should have reads
    first_row = results_df.iloc[0]
    assert first_row["chrom"] == "chr1"
    assert first_row["start"] == 1000
    assert first_row["end"] == 2000
    assert first_row["total_reads"] == 2
    assert first_row["duplication_reads"] == 1
    assert first_row["deletion_reads"] == 1


def test_analyze_cnv_breakpoints_real_data():
    """Test with real duplication and deletion data from resources."""
    import os
    from pathlib import Path

    # Get path to resources directory
    resources_dir = Path(__file__).parent.parent / "resources"

    # Test duplication
    dup_bam_file = os.path.join(resources_dir, "duplication.bam")
    dup_bed_file = os.path.join(resources_dir, "duplication.bed")

    # Verify files exist
    assert os.path.exists(dup_bam_file), f"BAM file not found: {dup_bam_file}"
    assert os.path.exists(dup_bed_file), f"BED file not found: {dup_bed_file}"

    # Run analysis on duplication
    dup_df = analyze_cnv_breakpoints(
        bam_file=dup_bam_file,
        bed_file=dup_bed_file,
        cushion=1000,
        output_file=None,
    )

    # Should have one interval
    assert len(dup_df) == 1

    # Check the interval
    dup_row = dup_df.iloc[0]
    assert dup_row["chrom"] == "chr2"
    assert dup_row["start"] == 122526000
    assert dup_row["end"] == 122537000

    # Should have at least 10 reads consistent with duplication
    assert (
        dup_row["duplication_reads"] >= 10
    ), f"Expected at least 10 duplication reads, but found {dup_row['duplication_reads']}"

    # Test deletion
    del_bam_file = os.path.join(resources_dir, "deletion.bam")
    del_bed_file = os.path.join(resources_dir, "deletion.bed")

    # Verify files exist
    assert os.path.exists(del_bam_file), f"BAM file not found: {del_bam_file}"
    assert os.path.exists(del_bed_file), f"BED file not found: {del_bed_file}"

    # Run analysis on deletion
    del_df = analyze_cnv_breakpoints(
        bam_file=del_bam_file,
        bed_file=del_bed_file,
        cushion=1000,
        output_file=None,
    )

    # Should have one interval
    assert len(del_df) == 1

    # Check the interval
    del_row = del_df.iloc[0]
    assert del_row["chrom"] == "chr1"
    assert del_row["start"] == 113069000
    assert del_row["end"] == 113071000

    # Should have at least 10 reads consistent with deletion
    assert (
        del_row["deletion_reads"] >= 10
    ), f"Expected at least 10 deletion reads, but found {del_row['deletion_reads']}"
