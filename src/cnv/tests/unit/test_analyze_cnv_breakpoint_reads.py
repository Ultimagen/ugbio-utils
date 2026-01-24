"""Tests for analyze_cnv_breakpoint_reads module."""

import os
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


@pytest.fixture
def temp_vcf_file():
    """Create a temporary VCF file for testing."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".vcf", delete=False) as f:
        # Write VCF header matching dummy.fasta (10Kb per chromosome)
        f.write("##fileformat=VCFv4.2\n")
        f.write("##contig=<ID=chr1,length=10000>\n")
        f.write("##contig=<ID=chr2,length=10000>\n")
        f.write('##INFO=<ID=SVTYPE,Number=1,Type=String,Description="Type of structural variant">\n')
        f.write('##INFO=<ID=END,Number=1,Type=Integer,Description="End position of the variant">\n')
        f.write("#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n")
        # Add three CNV variants (positions within 10Kb chromosome limits)
        f.write("chr1\t1001\t.\tN\t<DUP>\t.\tPASS\tSVTYPE=DUP;END=2000\n")
        f.write("chr1\t5001\t.\tN\t<DEL>\t.\tPASS\tSVTYPE=DEL;END=6000\n")
        f.write("chr2\t3001\t.\tN\t<DUP>\t.\tPASS\tSVTYPE=DUP;END=4000\n")
        temp_path = f.name
    yield temp_path
    Path(temp_path).unlink()


@pytest.fixture
def dummy_fasta_file():
    """Return path to the dummy FASTA file in resources."""
    resources_dir = Path(__file__).parent.parent / "resources"
    return str(resources_dir / "dummy.fasta")


@pytest.fixture
def temp_bam_file():
    """Create a temporary BAM file for testing."""
    with tempfile.NamedTemporaryFile(suffix=".bam", delete=False) as f:
        temp_path = f.name

    # Create a simple BAM file with header matching dummy.fasta (10Kb per chromosome)
    header = {
        "HD": {"VN": "1.0"},
        "SQ": [
            {"SN": "chr1", "LN": 10000},
            {"SN": "chr2", "LN": 10000},
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


def test_get_supplementary_alignments(temp_bam_file):
    """Test parsing SA tag."""
    read = pysam.AlignedSegment()
    read.query_name = "test_read"
    read.set_tag("SA", "chr1,1000,+,50M30S,60,0;chr1,2000,-,30S50M,60,0;")

    # Open a real alignment file for the function signature
    with pysam.AlignmentFile(temp_bam_file, "rb") as bam:
        supp_alns = get_supplementary_alignments(bam, read)

    assert len(supp_alns) == 2
    # First: 50M means +50 on reference, 30S on right, + strand
    assert supp_alns[0] == ("chr1", 999, 1049, False, True, False)  # 0-based, no left clip, has right clip, forward
    # Second: 30S on left, 50M means +50 on reference, - strand
    assert supp_alns[1] == ("chr1", 1999, 2049, True, False, True)  # 0-based, has left clip, no right clip, reverse


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
    supplementary_alns = [("chr1", 950, 1000, True, False, False)]  # has left clip, no right clip, forward

    # First part (2050) > Second part (950) -> duplication
    is_dup, is_del, insert_size = check_read_cnv_consistency(read, 1000, 2000, 100, supplementary_alns)
    assert is_dup is True
    assert is_del is False
    assert insert_size is not None
    assert insert_size > 0


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
    supplementary_alns = [("chr1", 2050, 2100, True, False, False)]  # has left clip, no right clip, forward

    # First part (950) < Second part (2050) -> deletion
    is_dup, is_del, insert_size = check_read_cnv_consistency(read, 1000, 2000, 100, supplementary_alns)
    assert is_dup is False
    assert is_del is True
    assert insert_size is not None
    assert insert_size > 0


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


def test_analyze_cnv_breakpoints(temp_bam_file, temp_vcf_file, dummy_fasta_file):
    """Test full analysis workflow with VCF output."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".vcf", delete=False) as output_f:
        output_vcf_path = output_f.name

    try:
        # Run analysis
        analyze_cnv_breakpoints(
            bam_file=temp_bam_file,
            vcf_file=temp_vcf_file,
            reference_fasta=dummy_fasta_file,
            cushion=100,
            output_file=output_vcf_path,
        )

        # Read and verify output VCF
        vcf = pysam.VariantFile(output_vcf_path)

        # Check that INFO fields are added
        assert "CNV_DUP_READS" in vcf.header.info
        assert "CNV_DEL_READS" in vcf.header.info
        assert "CNV_TOTAL_READS" in vcf.header.info
        assert "CNV_DUP_FRAC" in vcf.header.info
        assert "CNV_DEL_FRAC" in vcf.header.info
        assert "DUP_READS_MEDIAN_INSERT_SIZE" in vcf.header.info
        assert "DEL_READS_MEDIAN_INSERT_SIZE" in vcf.header.info

        # Collect records
        records = list(vcf)
        assert len(records) == 3  # Three variants in the VCF file

        # Check first record (chr1:1001-2000)
        first_record = records[0]
        assert first_record.chrom == "chr1"
        assert first_record.start == 1000  # VCF is 1-based, pysam converts to 0-based
        assert first_record.stop == 2000
        assert "CNV_DUP_READS" in first_record.info
        assert "CNV_DEL_READS" in first_record.info
        assert "CNV_TOTAL_READS" in first_record.info
        assert first_record.info["CNV_TOTAL_READS"] == 2
        assert first_record.info["CNV_DUP_READS"] == 1
        assert first_record.info["CNV_DEL_READS"] == 1

        vcf.close()
    finally:
        Path(output_vcf_path).unlink(missing_ok=True)


def test_analyze_cnv_breakpoints_real_data():
    """Test with real duplication and deletion data from resources."""

    # Get path to resources directory
    resources_dir = Path(__file__).parent.parent / "resources"

    # Test duplication
    dup_bam_file = os.path.join(resources_dir, "duplication.bam")
    dup_bed_file = os.path.join(resources_dir, "duplication.bed")

    # Verify files exist
    assert os.path.exists(dup_bam_file), f"BAM file not found: {dup_bam_file}"
    assert os.path.exists(dup_bed_file), f"BED file not found: {dup_bed_file}"

    # Create a temporary VCF file from the BED file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".vcf", delete=False) as dup_vcf_f:
        dup_vcf_file = dup_vcf_f.name
        # Write VCF header
        dup_vcf_f.write("##fileformat=VCFv4.2\n")
        dup_vcf_f.write("##contig=<ID=chr2,length=1000000000>\n")
        dup_vcf_f.write('##INFO=<ID=SVTYPE,Number=1,Type=String,Description="Type of structural variant">\n')
        dup_vcf_f.write('##INFO=<ID=END,Number=1,Type=Integer,Description="End position of the variant">\n')
        dup_vcf_f.write("#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n")

        # Read BED file and convert to VCF
        with open(dup_bed_file) as bed_f:
            for line in bed_f:
                parts = line.strip().split("\t")
                chrom, start, end = parts[0], int(parts[1]), int(parts[2])
                # VCF is 1-based
                dup_vcf_f.write(f"{chrom}\t{start+1}\t.\tN\t<DUP>\t.\tPASS\tSVTYPE=DUP;END={end}\n")

    with tempfile.NamedTemporaryFile(mode="w", suffix=".vcf", delete=False) as dup_output_f:
        dup_output_vcf = dup_output_f.name

    # Use Homo_sapiens_assembly38.fasta.fai location to find reference
    # For this test, we use a minimal reference - the BAM contains reference info
    reference_fasta = os.path.join(resources_dir, "chr19.fasta")

    try:
        # Run analysis on duplication
        analyze_cnv_breakpoints(
            bam_file=dup_bam_file,
            vcf_file=dup_vcf_file,
            reference_fasta=reference_fasta,
            cushion=1000,
            output_file=dup_output_vcf,
        )

        # Read output VCF
        dup_vcf = pysam.VariantFile(dup_output_vcf)
        records = list(dup_vcf)
        dup_vcf.close()

        # Should have one interval
        assert len(records) == 1

        # Check the interval
        dup_record = records[0]
        assert dup_record.chrom == "chr2"
        assert dup_record.start == 122526000  # 0-based
        assert dup_record.stop == 122537000

        # Should have at least 10 reads consistent with duplication
        assert (
            dup_record.info["CNV_DUP_READS"] >= 10
        ), f"Expected at least 10 duplication reads, but found {dup_record.info['CNV_DUP_READS']}"

        # Check insert size statistics are present when there are supporting reads
        if dup_record.info["CNV_DUP_READS"] >= 1:
            assert "DUP_READS_MEDIAN_INSERT_SIZE" in dup_record.info

    finally:
        Path(dup_vcf_file).unlink(missing_ok=True)
        Path(dup_output_vcf).unlink(missing_ok=True)

    # Test deletion
    del_bam_file = os.path.join(resources_dir, "deletion.bam")
    del_bed_file = os.path.join(resources_dir, "deletion.bed")

    # Verify files exist
    assert os.path.exists(del_bam_file), f"BAM file not found: {del_bam_file}"
    assert os.path.exists(del_bed_file), f"BED file not found: {del_bed_file}"

    # Create a temporary VCF file from the BED file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".vcf", delete=False) as del_vcf_f:
        del_vcf_file = del_vcf_f.name
        # Write VCF header
        del_vcf_f.write("##fileformat=VCFv4.2\n")
        del_vcf_f.write("##contig=<ID=chr1,length=1000000000>\n")
        del_vcf_f.write('##INFO=<ID=SVTYPE,Number=1,Type=String,Description="Type of structural variant">\n')
        del_vcf_f.write('##INFO=<ID=END,Number=1,Type=Integer,Description="End position of the variant">\n')
        del_vcf_f.write("#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n")

        # Read BED file and convert to VCF
        with open(del_bed_file) as bed_f:
            for line in bed_f:
                parts = line.strip().split("\t")
                chrom, start, end = parts[0], int(parts[1]), int(parts[2])
                # VCF is 1-based
                del_vcf_f.write(f"{chrom}\t{start+1}\t.\tN\t<DEL>\t.\tPASS\tSVTYPE=DEL;END={end}\n")

    with tempfile.NamedTemporaryFile(mode="w", suffix=".vcf", delete=False) as del_output_f:
        del_output_vcf = del_output_f.name

    try:
        # Run analysis on deletion
        analyze_cnv_breakpoints(
            bam_file=del_bam_file,
            vcf_file=del_vcf_file,
            reference_fasta=reference_fasta,
            cushion=1000,
            output_file=del_output_vcf,
        )

        # Read output VCF
        del_vcf = pysam.VariantFile(del_output_vcf)
        records = list(del_vcf)
        del_vcf.close()

        # Should have one interval
        assert len(records) == 1

        # Check the interval
        del_record = records[0]
        assert del_record.chrom == "chr1"
        assert del_record.start == 113069000  # 0-based
        assert del_record.stop == 113071000

        # Should have at least 10 reads consistent with deletion
        assert (
            del_record.info["CNV_DEL_READS"] >= 10
        ), f"Expected at least 10 deletion reads, but found {del_record.info['CNV_DEL_READS']}"

        # Check insert size statistics are present when there are supporting reads
        if del_record.info["CNV_DEL_READS"] >= 1:
            assert "DEL_READS_MEDIAN_INSERT_SIZE" in del_record.info

    finally:
        Path(del_vcf_file).unlink(missing_ok=True)
        Path(del_output_vcf).unlink(missing_ok=True)
