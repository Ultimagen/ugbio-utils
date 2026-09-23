"""Tests for analyze_cnv_breakpoint_reads module."""

import os
import tempfile
from pathlib import Path

import pysam
import pytest
from ugbio_cnv.analyze_cnv_breakpoint_reads import (
    MIN_PAIR_SPAN,
    PAIR_READ_GROUP,
    PairedEndConfig,
    _collect_reads_from_region,
    _FragmentEvidence,
    _mate_aware_key,
    _pair_mates_flank_deleted_segment,
    _summarize_fragments,
    analyze_cnv_breakpoints,
    analyze_interval_breakpoints,
    check_pair_cnv_consistency,
    check_read_cnv_consistency,
    get_parser,
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
        # Write reads in coordinate-sorted order for proper BAM indexing
        # Position 950: read2 (primary) and read1_supp (supplementary)
        # Position 2050: read1 (primary) and read2_supp (supplementary)

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

        # Supplementary alignment for read1 (at position 950)
        read1_supp = pysam.AlignedSegment()
        read1_supp.query_name = "read1"
        read1_supp.query_sequence = (
            "ACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGT"  # 80 bases
        )
        read1_supp.reference_id = 0  # chr1
        read1_supp.reference_start = 950
        read1_supp.cigartuples = [(4, 30), (0, 50)]  # 30S50M - left soft clip (second part)
        read1_supp.is_reverse = False
        read1_supp.is_supplementary = True
        # SA tag: pointing back to primary at chr1:2051 (1-based)
        read1_supp.set_tag("SA", "chr1,2051,+,50M30S,60,0;")
        outf.write(read1_supp)

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

        # Supplementary alignment for read2 (at position 2050)
        read2_supp = pysam.AlignedSegment()
        read2_supp.query_name = "read2"
        read2_supp.query_sequence = (
            "ACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGT"  # 80 bases
        )
        read2_supp.reference_id = 0  # chr1
        read2_supp.reference_start = 2050
        read2_supp.cigartuples = [(4, 30), (0, 50)]  # 30S50M - left soft clip (second part)
        read2_supp.is_reverse = False
        read2_supp.is_supplementary = True
        # SA tag: pointing back to primary at chr1:951 (1-based)
        read2_supp.set_tag("SA", "chr1,951,+,50M30S,60,0;")
        outf.write(read2_supp)

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
    supp_alns = get_supplementary_alignments(read)

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


def test_analyze_interval_breakpoints_scans_both_breakpoint_windows():
    """Regression test: ensure both breakpoint windows are scanned.

    This catches a pysam iterator bug where chaining two ``fetch`` iterators from
    the same handle can skip the first iterator unless independent iterators are
    requested. The synthetic interval below has CNV-supporting SA evidence only
    in the START breakpoint window.
    """
    with tempfile.NamedTemporaryFile(suffix=".bam", delete=False) as bam_f:
        bam_path = bam_f.name

    header = {
        "HD": {"VN": "1.0"},
        "SQ": [
            {"SN": "chr1", "LN": 10000},
        ],
    }

    try:
        with pysam.AlignmentFile(bam_path, "wb", header=header) as bam_out:
            # START window read with SA evidence into END window.
            split_read = pysam.AlignedSegment()
            split_read.query_name = "split_read"
            split_read.query_sequence = (
                "ACGTACGTACGTACGTACGTACGTACGTACGT" "ACGTACGTACGTACGTACGTACGTACGTACGT" "ACGTACGTACGTACGT"
            )
            split_read.reference_id = 0
            split_read.reference_start = 950  # START window for interval 1000-2000, cushion=100
            split_read.cigartuples = [(0, 50), (4, 30)]  # first part: right soft clip
            split_read.is_reverse = False
            split_read.set_tag("SA", "chr1,2051,+,30S50M,60,0;")
            bam_out.write(split_read)

            # Non-SA read in END window to ensure second window has content.
            end_only_read = pysam.AlignedSegment()
            end_only_read.query_name = "end_only_read"
            end_only_read.query_sequence = "A" * 80
            end_only_read.reference_id = 0
            end_only_read.reference_start = 1950
            end_only_read.cigartuples = [(0, 80)]
            end_only_read.is_reverse = False
            bam_out.write(end_only_read)

        pysam.index(bam_path)

        with pysam.AlignmentFile(bam_path, "rb") as bam_in:
            evidence = analyze_interval_breakpoints(bam_in, "chr1", 1000, 2000, 100)

        assert evidence.total_reads == 2
        assert evidence.duplication_reads == 0
        assert evidence.deletion_reads == 1
    finally:
        Path(bam_path).unlink(missing_ok=True)
        Path(bam_path + ".bai").unlink(missing_ok=True)


def test_collect_reads_from_region_uses_independent_fetch_iterators(monkeypatch):
    """Regression test for fetch iterator invalidation between breakpoint windows."""

    class FakeAlignmentFile:
        """Fake alignment file that invalidates prior iterators unless multiple_iterators=True."""

        def __init__(self, start_reads, end_reads):
            self.start_reads = start_reads
            self.end_reads = end_reads
            self._active_state = None

        def fetch(self, chrom, start, end, *, multiple_iterators=False):  # noqa: ARG002
            reads = self.start_reads if start < 1500 else self.end_reads
            if multiple_iterators:
                return iter(reads)

            state = object()
            self._active_state = state

            def _iter_reads():
                if self._active_state is not state:
                    return
                yield from reads

            return _iter_reads()

    start_read = pysam.AlignedSegment()
    start_read.query_name = "start_read"
    start_read.reference_id = 0
    start_read.reference_start = 950
    start_read.cigartuples = [(0, 50), (4, 30)]

    end_read = pysam.AlignedSegment()
    end_read.query_name = "end_read"
    end_read.reference_id = 0
    end_read.reference_start = 1950
    end_read.cigartuples = [(0, 80)]

    fake_alignment_file = FakeAlignmentFile([start_read], [end_read])

    def _fake_process_primary_read_for_evidence(  # noqa: PLR0913
        read,
        alignment_file,
        start,
        end,
        cushion,
        duplication_reads,
        deletion_reads,
        dup_insert_sizes,
        del_insert_sizes,
        supporting_reads,
    ):
        return duplication_reads, deletion_reads

    monkeypatch.setattr(
        "ugbio_cnv.analyze_cnv_breakpoint_reads._process_primary_read_for_evidence",
        _fake_process_primary_read_for_evidence,
    )

    (
        _,
        _,
        _,
        _,
        processed_reads,
        _,
        _,
    ) = _collect_reads_from_region(
        fake_alignment_file,
        "chr1",
        1000,
        2000,
        100,
        900,
        1100,
        1900,
        2100,
    )

    # processed_reads now contains (read_name, RG) tuples for deduplication
    assert processed_reads == {("start_read", "UNKNOWN"), ("end_read", "UNKNOWN")}


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


def test_median_insert_size_none_values(temp_vcf_file, dummy_fasta_file):
    """Test that median insert size fields are set to 0.0 when no supporting reads are found.

    This is a regression test for a bug where missing insert size fields (None values)
    caused downstream filtering to crash with "Data matrix contains null in column 12".
    The fix ensures these fields are always present with 0.0 as the default value.
    """
    # Create a BAM file with NO reads (empty)
    with tempfile.NamedTemporaryFile(suffix=".bam", delete=False) as f:
        empty_bam_path = f.name

    # Create an empty BAM file with header
    header = {
        "HD": {"VN": "1.0"},
        "SQ": [
            {"SN": "chr1", "LN": 10000},
            {"SN": "chr2", "LN": 10000},
        ],
    }

    with pysam.AlignmentFile(empty_bam_path, "wb", header=header):
        pass  # Write header only, no reads

    # Index the empty BAM file
    pysam.index(empty_bam_path)

    # Create output VCF file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".vcf", delete=False) as output_f:
        output_vcf_path = output_f.name

    try:
        # Run analysis with empty BAM (no supporting reads)
        analyze_cnv_breakpoints(
            bam_file=empty_bam_path,
            vcf_file=temp_vcf_file,
            reference_fasta=dummy_fasta_file,
            cushion=100,
            output_file=output_vcf_path,
        )

        # Read and verify output VCF
        vcf = pysam.VariantFile(output_vcf_path)

        # Check that INFO fields are added to header
        assert "DUP_READS_MEDIAN_INSERT_SIZE" in vcf.header.info
        assert "DEL_READS_MEDIAN_INSERT_SIZE" in vcf.header.info

        # Collect records
        records = list(vcf)
        assert len(records) == 3  # Three variants in the VCF file

        # Check each record has the insert size fields set to 0.0 (not None).
        # Accessing missing keys in record.info will raise KeyError, ensuring
        # the test still fails if the fields are absent.
        for record in records:
            # Values must be 0.0 (not None)
            assert record.info["DUP_READS_MEDIAN_INSERT_SIZE"] == 0.0, (
                "Expected DUP_READS_MEDIAN_INSERT_SIZE=0.0, got " f"{record.info['DUP_READS_MEDIAN_INSERT_SIZE']}"
            )
            assert record.info["DEL_READS_MEDIAN_INSERT_SIZE"] == 0.0, (
                "Expected DEL_READS_MEDIAN_INSERT_SIZE=0.0, got " f"{record.info['DEL_READS_MEDIAN_INSERT_SIZE']}"
            )

        vcf.close()
    finally:
        Path(empty_bam_path).unlink(missing_ok=True)
        Path(empty_bam_path + ".bai").unlink(missing_ok=True)
        Path(output_vcf_path).unlink(missing_ok=True)


def test_analyze_cnv_breakpoints_with_bam_output(temp_bam_file, temp_vcf_file, dummy_fasta_file):
    """Test that split reads BAM output is written correctly with proper read groups."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".vcf", delete=False) as output_f:
        output_vcf_path = output_f.name

    with tempfile.NamedTemporaryFile(suffix=".bam", delete=False) as bam_f:
        output_bam_path = bam_f.name

    try:
        # Run analysis with BAM output
        analyze_cnv_breakpoints(
            bam_file=temp_bam_file,
            vcf_file=temp_vcf_file,
            reference_fasta=dummy_fasta_file,
            cushion=100,
            output_file=output_vcf_path,
            output_bam=output_bam_path,
        )

        # Read and verify output BAM
        with pysam.AlignmentFile(output_bam_path, "rb") as bam_out:
            # Check that header has read groups
            header_dict = bam_out.header.to_dict()
            assert "RG" in header_dict
            rg_ids = {rg["ID"] for rg in header_dict["RG"]}
            # Should have DUP and DEL read groups at minimum
            assert "DUP" in rg_ids
            assert "DEL" in rg_ids

            # Collect reads and their read groups
            reads = list(bam_out)
            assert len(reads) == 4  # Four reads: 2 primary + 2 supplementary

            # Group reads by query_name
            reads_by_name = {}
            for read in reads:
                if read.query_name not in reads_by_name:
                    reads_by_name[read.query_name] = []
                reads_by_name[read.query_name].append(read)

            # Verify we have both read1 and read2
            assert "read1" in reads_by_name
            assert "read2" in reads_by_name

            # Verify each read has both primary and supplementary
            assert len(reads_by_name["read1"]) == 2  # Primary + supplementary
            assert len(reads_by_name["read2"]) == 2  # Primary + supplementary

            # Verify read groups and flags for each read
            for read_name, read_list in reads_by_name.items():
                primary_reads = [r for r in read_list if not r.is_supplementary]
                supplementary_reads = [r for r in read_list if r.is_supplementary]

                assert len(primary_reads) == 1, f"Expected 1 primary read for {read_name}"
                assert len(supplementary_reads) == 1, f"Expected 1 supplementary read for {read_name}"

                # Both primary and supplementary should have the same read group
                primary_rg = primary_reads[0].get_tag("RG")
                supp_rg = supplementary_reads[0].get_tag("RG")
                assert primary_rg == supp_rg, f"Read group mismatch for {read_name}"
                assert primary_rg in ("DUP", "DEL"), f"Invalid read group: {primary_rg}"

    finally:
        Path(output_vcf_path).unlink(missing_ok=True)
        Path(output_bam_path).unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Paired-end (discordant read pair) evidence
# ---------------------------------------------------------------------------

PE_HEADER = {"HD": {"VN": "1.0"}, "SQ": [{"SN": "chr1", "LN": 10000}, {"SN": "chr2", "LN": 10000}]}
PE_SEQUENCE = "ACGT" * 20  # 80 bases

# Standard geometry for the paired-end tests: interval chr1:1000-2000 with cushion 100 gives a
# start window of 900-1100 and an end window of 1900-2100. A pair whose leftmost mate starts at
# 950 and whose span is 1050 therefore implies a right end of 2000, inside the end window.
PE_INTERVAL_START = 1000
PE_INTERVAL_END = 2000
PE_CUSHION = 100
PE_LEFT_START = 950
PE_RIGHT_START = 1950
PE_SPAN = 1050

# A longer interval, chr1:1000-6000, for the deletion-flanking tests: at cushion 100 its interior is 1100-5900.
# A leftmost mate at 950 with span 5050 ends at 6000, inside the end window, so such a pair passes bracketing
# and is rejected by the flanking rule alone.
PE_LONG_INTERVAL_END = 6000
PE_LONG_SPAN = 5050
PE_INSIDE_START = 4000  # ~2.9 kb inside the deleted segment
PE_OUTSIDE_RIGHT_START = 5950  # just past the end breakpoint


def pe_config(**overrides):
    """Build an enabled PairedEndConfig with an explicit span floor."""
    defaults = {
        "enabled": True,
        "min_pair_span": MIN_PAIR_SPAN,
        "min_pair_mapping_quality": 20,
    }
    return PairedEndConfig(**{**defaults, **overrides})


def _make_pe_mate(
    query_name,
    reference_start,
    *,
    is_read1=True,
    is_reverse=False,
    mate_reference_start=PE_RIGHT_START,
    mate_is_reverse=True,
    template_length=PE_SPAN,
    reference_id=0,
    mate_reference_id=0,
    cigartuples=None,
    mapping_quality=60,
    sa_tag=None,
    is_paired=True,
    mate_is_unmapped=False,
    is_supplementary=False,
    read_group="rg1",
):
    """Build one mate of a paired-end fragment as a pysam.AlignedSegment."""
    read = pysam.AlignedSegment(header=pysam.AlignmentHeader.from_dict(PE_HEADER))
    read.query_name = query_name
    read.query_sequence = PE_SEQUENCE
    read.reference_id = reference_id
    read.reference_start = reference_start
    read.cigartuples = cigartuples if cigartuples is not None else [(0, 80)]
    read.mapping_quality = mapping_quality
    read.is_paired = is_paired
    if is_paired:
        read.is_read1 = is_read1
        read.is_read2 = not is_read1
        read.mate_is_reverse = mate_is_reverse
        read.mate_is_unmapped = mate_is_unmapped
        read.next_reference_id = mate_reference_id
        read.next_reference_start = mate_reference_start
        read.template_length = template_length
    read.is_reverse = is_reverse
    read.is_supplementary = is_supplementary
    if sa_tag is not None:
        read.set_tag("SA", sa_tag)
    read.set_tag("RG", read_group, value_type="Z")
    return read


def _make_right_mate(query_name, **overrides):
    """Build the right-hand mate of the standard pair, i.e. the same fragment seen from 1950."""
    defaults = {
        "is_read1": False,
        "is_reverse": True,
        "mate_reference_start": PE_LEFT_START,
        "mate_is_reverse": False,
        "template_length": -PE_SPAN,
    }
    return _make_pe_mate(query_name, PE_RIGHT_START, **{**defaults, **overrides})


@pytest.fixture
def pe_bam_factory():
    """Return a factory that writes reads into an indexed temporary BAM and yields its path."""
    created = []

    def _factory(reads):
        with tempfile.NamedTemporaryFile(suffix=".bam", delete=False) as f:
            temp_path = f.name
        ordered = sorted(reads, key=lambda r: (r.reference_id, r.reference_start))
        with pysam.AlignmentFile(temp_path, "wb", header=PE_HEADER) as outf:
            for read in ordered:
                outf.write(read)
        pysam.index(temp_path)
        created.append(temp_path)
        return temp_path

    yield _factory

    for path in created:
        Path(path).unlink(missing_ok=True)
        Path(path + ".bai").unlink(missing_ok=True)


def _analyze(bam_file, config, chrom="chr1", start=PE_INTERVAL_START, end=PE_INTERVAL_END, cushion=PE_CUSHION):
    """Run analyze_interval_breakpoints over a BAM path."""
    with pysam.AlignmentFile(bam_file, "rb") as alignment_file:
        return analyze_interval_breakpoints(alignment_file, chrom, start, end, cushion, config)


def _check_pair(read, config=None, start=PE_INTERVAL_START, end=PE_INTERVAL_END, cushion=PE_CUSHION):
    """Run check_pair_cnv_consistency with the standard geometry."""
    return check_pair_cnv_consistency(read, start, end, cushion, config if config is not None else pe_config())


# --- pair classification ---------------------------------------------------


def test_check_pair_cnv_consistency_deletion():
    """An FR pair whose span is inflated across the interval is deletion evidence."""
    is_dup, is_del, insert_size = _check_pair(_make_pe_mate("frag", PE_LEFT_START))

    assert (is_dup, is_del) == (False, True)
    # The span is reported as-is; no library median is subtracted
    assert insert_size == PE_SPAN


def test_check_pair_cnv_consistency_deletion_from_right_mate():
    """Both mates of one discordant pair classify it identically."""
    is_dup, is_del, insert_size = _check_pair(_make_right_mate("frag"))

    assert (is_dup, is_del) == (False, True)
    assert insert_size == PE_SPAN


def test_check_pair_cnv_consistency_duplication_reports_the_span():
    """An everted (RF) pair is duplication evidence and its span is reported unmodified."""
    read = _make_pe_mate("frag", PE_LEFT_START, is_reverse=True, mate_is_reverse=False)

    is_dup, is_del, insert_size = _check_pair(read)

    assert (is_dup, is_del) == (True, False)
    # For a tandem duplication the everted span already approximates the duplicated length
    assert insert_size == PE_SPAN


def test_check_pair_cnv_consistency_duplication_from_right_mate():
    """The right-hand mate of an everted pair also reports duplication."""
    read = _make_right_mate("frag", is_reverse=False, mate_is_reverse=True)

    is_dup, is_del, insert_size = _check_pair(read)

    assert (is_dup, is_del) == (True, False)
    assert insert_size == PE_SPAN


def test_check_pair_cnv_consistency_ignores_same_strand_mates():
    """Same-strand mates are inversion-like and are not counted."""
    read = _make_pe_mate("frag", PE_LEFT_START, is_reverse=False, mate_is_reverse=False)

    assert _check_pair(read) == (False, False, None)


def test_check_pair_cnv_consistency_deletion_requires_minimum_span():
    """An FR pair with a span below the floor is not deletion evidence."""
    read = _make_pe_mate("frag", PE_LEFT_START)

    assert _check_pair(read, pe_config(min_pair_span=5000)) == (False, False, None)


def test_check_pair_cnv_consistency_duplication_requires_minimum_span():
    """
    An everted pair with a span below the floor is not duplication evidence.

    For everted pairs the floor is only the smallest believable duplication, not a background filter.
    """
    read = _make_pe_mate("frag", PE_LEFT_START, is_reverse=True, mate_is_reverse=False)

    assert _check_pair(read, pe_config(min_pair_span=5000)) == (False, False, None)


@pytest.mark.parametrize(
    ("is_reverse", "mate_is_reverse", "classification"),
    [(False, True, (False, True)), (True, False, (True, False))],
    ids=["deletion", "duplication"],
)
def test_pair_span_gate_is_on_the_raw_span(is_reverse, mate_is_reverse, classification):
    """Both classes compare the floor against the template length, with nothing subtracted."""
    read = _make_pe_mate("frag", PE_LEFT_START, is_reverse=is_reverse, mate_is_reverse=mate_is_reverse)

    assert _check_pair(read, pe_config(min_pair_span=PE_SPAN + 1)) == (False, False, None)
    assert _check_pair(read, pe_config(min_pair_span=PE_SPAN)) == (*classification, PE_SPAN)


@pytest.mark.parametrize(
    ("is_reverse", "mate_is_reverse", "classification"),
    [(False, True, (False, True)), (True, False, (True, False))],
    ids=["deletion", "duplication"],
)
def test_pair_span_gate_boundary_at_the_default(is_reverse, mate_is_reverse, classification):
    """At the default floor a span of 799 is rejected and a span of 800 counts, for either class."""
    short = {"start": 1000, "end": 1400, "cushion": 1500}
    orientation = {"is_reverse": is_reverse, "mate_is_reverse": mate_is_reverse}
    below = _make_pe_mate("frag", PE_LEFT_START, mate_reference_start=1669, template_length=799, **orientation)
    at = _make_pe_mate("frag", PE_LEFT_START, mate_reference_start=1670, template_length=800, **orientation)

    assert _check_pair(below, pe_config(), **short) == (False, False, None)
    assert _check_pair(at, pe_config(), **short) == (*classification, 800)


def test_pair_span_gate_bites_when_the_interval_is_short():
    """
    The floor only matters for intervals short relative to the cushion.

    A 400 bp interval against a 1500 bp cushion, so the floor and not bracketing decides. The interior is
    empty here, which also exercises the degenerate case of the flanking rule.
    """
    short = {"start": 1000, "end": 1400, "cushion": 1500}

    below = _make_pe_mate("frag", PE_LEFT_START, mate_reference_start=1470, template_length=600)
    assert _check_pair(below, pe_config(), **short) == (False, False, None)

    above = _make_pe_mate("frag", PE_LEFT_START, mate_reference_start=1870, template_length=1000)
    assert _check_pair(above, pe_config(), **short) == (False, True, 1000)


def test_paired_end_config_defaults_to_the_module_minimum_span():
    """The default floor comes from the module constant, so the CLI and the config agree."""
    assert PairedEndConfig().min_pair_span == MIN_PAIR_SPAN
    assert MIN_PAIR_SPAN == 800


def test_check_pair_cnv_consistency_ignores_pair_starting_outside_start_window():
    """The leftmost mate must start inside the start breakpoint window."""
    read = _make_pe_mate("frag", 500, template_length=1500)

    assert _check_pair(read) == (False, False, None)


def test_check_pair_cnv_consistency_ignores_pair_ending_outside_end_window():
    """The implied right end must land inside the end breakpoint window."""
    read = _make_pe_mate("frag", PE_LEFT_START, template_length=3000)

    assert _check_pair(read) == (False, False, None)


def test_check_pair_cnv_consistency_ignores_cross_contig_pair():
    """A pair whose mates are on different contigs is not DUP/DEL evidence for this interval."""
    read = _make_pe_mate("frag", PE_LEFT_START, mate_reference_id=1)

    assert _check_pair(read) == (False, False, None)


# --- deletion pairs must flank the deleted segment -------------------------

_LONG = {"end": PE_LONG_INTERVAL_END}


def test_deletion_pair_with_mate_inside_the_deleted_segment_is_rejected():
    """
    A deletion pair cannot have a mate aligned inside the deleted segment.

    Bracketing only pins the rightmost mate's end, so it admits a mate starting deep inside the interval. The
    same pair with the mate past the end breakpoint is counted, so the flanking rule alone rejects this one.
    """
    inside = _make_pe_mate("frag", PE_LEFT_START, mate_reference_start=PE_INSIDE_START, template_length=PE_LONG_SPAN)
    assert _check_pair(inside, **_LONG) == (False, False, None)

    outside = _make_pe_mate(
        "frag", PE_LEFT_START, mate_reference_start=PE_OUTSIDE_RIGHT_START, template_length=PE_LONG_SPAN
    )
    assert _check_pair(outside, **_LONG) == (False, True, PE_LONG_SPAN)


def test_mate_inside_the_deleted_segment_is_rejected_from_either_mate():
    """
    The same fragment is rejected when examined from the mate that lies inside the segment.

    A fragment counts as soon as one alignment passes, so the verdict must not depend on which mate is on hand.
    """
    read = _make_pe_mate(
        "frag",
        PE_INSIDE_START,
        is_read1=False,
        is_reverse=True,
        mate_reference_start=PE_LEFT_START,
        mate_is_reverse=False,
        template_length=-PE_LONG_SPAN,
    )

    assert _check_pair(read, **_LONG) == (False, False, None)


def test_duplication_pair_is_exempt_from_the_flanking_rule():
    """
    An everted pair supporting a tandem duplication has both mates inside the region by design.

    The rule is therefore applied on the deletion branch only; hoisting it above the
    classification would remove pair-based duplication evidence entirely.
    """
    read = _make_pe_mate(
        "frag",
        PE_LEFT_START,
        is_reverse=True,
        mate_is_reverse=False,
        mate_reference_start=PE_INSIDE_START,
        template_length=PE_LONG_SPAN,
    )

    assert _check_pair(read, **_LONG) == (True, False, PE_LONG_SPAN)


def test_flanking_rule_is_inert_for_short_intervals():
    """For an interval no longer than 2*cushion the windows cover it, so there is no interior."""
    read = _make_pe_mate("frag", PE_LEFT_START, mate_reference_start=2400, template_length=1550)

    assert _check_pair(read, pe_config(), start=1000, end=2000, cushion=1500) == (False, True, 1550)


def test_mate_starting_on_the_end_window_edge_is_kept():
    """A mate starting exactly on the window edge is inside the window, so the pair counts."""
    read = _make_pe_mate(
        "frag",
        PE_LEFT_START,
        mate_reference_start=PE_LONG_INTERVAL_END - PE_CUSHION,
        template_length=PE_LONG_SPAN,
    )

    assert _check_pair(read, **_LONG) == (False, True, PE_LONG_SPAN)


def test_leftmost_mate_ending_past_the_junction_is_still_counted():
    """
    The leftmost mate may end past the start breakpoint; only start positions are tested.

    Mates overlap in 63.5% of proper pairs on production data, so this is the normal look of real evidence.
    """
    read = _make_pe_mate("frag", 1090, mate_reference_start=PE_OUTSIDE_RIGHT_START, template_length=4910)

    assert _check_pair(read, **_LONG) == (False, True, 4910)


def test_pair_mates_flank_deleted_segment_rejects_a_leftmost_mate_inside():
    """
    The leftmost-mate clause is only reachable by calling the predicate directly.

    Bracketing already caps the leftmost mate's start at interval_start + cushion, but the
    predicate states the physical rule on its own rather than relying on its caller.
    """
    read = _make_pe_mate("frag", 2000, mate_reference_start=PE_INSIDE_START, template_length=2050)

    assert _pair_mates_flank_deleted_segment(read, PE_INTERVAL_START, PE_LONG_INTERVAL_END, PE_CUSHION) is False


def test_pair_with_both_mates_on_the_same_side_is_rejected():
    """A pair sitting entirely at the start breakpoint brackets nothing and is not evidence."""
    read = _make_pe_mate("frag", PE_LEFT_START, mate_reference_start=1000, template_length=130)

    assert _check_pair(read) == (False, False, None)


def test_check_pair_cnv_consistency_ignores_unmapped_mate():
    """A read whose mate is unmapped has no pair geometry."""
    read = _make_pe_mate("frag", PE_LEFT_START, mate_is_unmapped=True)

    assert _check_pair(read) == (False, False, None)


def test_check_pair_cnv_consistency_ignores_unpaired_read():
    """A single-end read never produces pair evidence."""
    read = _make_pe_mate("frag", PE_LEFT_START, is_paired=False)

    assert _check_pair(read) == (False, False, None)


def test_check_pair_cnv_consistency_ignores_zero_template_length():
    """A zero template length means the aligner did not report a fragment span."""
    read = _make_pe_mate("frag", PE_LEFT_START, template_length=0)

    assert _check_pair(read) == (False, False, None)


def test_check_pair_cnv_consistency_ignores_unorderable_mates():
    """Mates starting at the same position cannot be ordered, so orientation is uninterpretable."""
    read = _make_pe_mate("frag", PE_LEFT_START, mate_reference_start=PE_LEFT_START)

    assert _check_pair(read) == (False, False, None)


def test_check_pair_cnv_consistency_honours_mapping_quality_floor():
    """A read below the mapping quality floor contributes no pair evidence."""
    read = _make_pe_mate("frag", PE_LEFT_START, mapping_quality=10)

    assert _check_pair(read) == (False, False, None)


def test_check_pair_cnv_consistency_ignores_mate_mapping_quality_tag():
    """The floor applies to the alignment at hand; a mate MQ tag, if some tool wrote one, is not read."""
    read = _make_pe_mate("frag", PE_LEFT_START)
    read.set_tag("MQ", 5, value_type="i")

    assert _check_pair(read)[1] is True


# --- fragment-level counting ----------------------------------------------


def test_both_discordant_mates_count_as_one_fragment(pe_bam_factory):
    """One fragment casts one vote, however many of its alignments are discordant."""
    bam_file = pe_bam_factory([_make_pe_mate("frag", PE_LEFT_START), _make_right_mate("frag")])

    evidence = _analyze(bam_file, pe_config())

    assert evidence.total_reads == 1
    assert evidence.deletion_reads == 1
    assert evidence.duplication_reads == 0
    assert evidence.del_median_insert_size == PE_SPAN
    # Both mates enter the evidence BAM, under the pair read group
    assert len(evidence.supporting_reads) == 2
    assert {read_group for _, read_group in evidence.supporting_reads} == {PAIR_READ_GROUP}


def test_fragment_with_a_mate_inside_the_deleted_segment_casts_no_vote(pe_bam_factory):
    """
    Per-fragment voting cannot rescue a pair whose mate sits inside the deleted segment.

    Both alignments of the fragment are present, and both must reject it, so the fragment reaches
    the denominator without supporting either class.
    """
    left = _make_pe_mate("frag", PE_LEFT_START, mate_reference_start=PE_INSIDE_START, template_length=PE_LONG_SPAN)
    inside = _make_pe_mate(
        "frag",
        PE_INSIDE_START,
        is_read1=False,
        is_reverse=True,
        mate_reference_start=PE_LEFT_START,
        mate_is_reverse=False,
        template_length=-PE_LONG_SPAN,
    )
    bam_file = pe_bam_factory([left, inside])

    evidence = _analyze(bam_file, pe_config(), end=PE_LONG_INTERVAL_END)

    assert evidence.total_reads == 1
    assert evidence.deletion_reads == 0
    assert evidence.duplication_reads == 0


def test_conflicting_mates_count_towards_denominator_only(pe_bam_factory):
    """A fragment whose mates disagree is counted in the total but in neither numerator."""
    left = _make_pe_mate("frag", PE_LEFT_START)  # FR from the left mate's view -> DEL
    right = _make_right_mate("frag", is_reverse=False, mate_is_reverse=True)  # everted -> DUP
    bam_file = pe_bam_factory([left, right])

    evidence = _analyze(bam_file, pe_config())

    assert evidence.total_reads == 1
    assert evidence.deletion_reads == 0
    assert evidence.duplication_reads == 0
    # A fragment that casts no vote writes nothing to the evidence BAM
    assert evidence.supporting_reads == []


def test_split_evidence_outranks_pair_evidence_in_one_fragment(pe_bam_factory):
    """When a fragment has both kinds of evidence it votes once, using the split insert size."""
    primary = _make_pe_mate(
        "frag",
        PE_LEFT_START,
        cigartuples=[(0, 50), (4, 30)],  # 50M30S - right clip, first part
        sa_tag="chr1,2051,+,30S50M,60,0;",
    )
    supplementary = _make_pe_mate(
        "frag",
        2050,
        cigartuples=[(4, 30), (0, 50)],
        sa_tag="chr1,951,+,50M30S,60,0;",
        is_supplementary=True,
    )
    bam_file = pe_bam_factory([primary, supplementary, _make_right_mate("frag")])

    evidence = _analyze(bam_file, pe_config())

    assert evidence.total_reads == 1
    assert evidence.deletion_reads == 1
    assert len(evidence.del_insert_sizes) == 1
    # The pair estimate is discarded because a split estimate exists for this fragment
    assert evidence.del_median_insert_size == evidence.del_insert_sizes[0]
    # Only the split alignment is written, under its label; the pair-supporting mate is not
    assert len(evidence.supporting_reads) == 1
    read, read_group = evidence.supporting_reads[0]
    assert read_group == "DEL"
    assert read.reference_start == PE_LEFT_START


def test_overlapping_breakpoint_windows_count_an_alignment_once(pe_bam_factory):
    """For a CNV shorter than 2*cushion the windows overlap; a read fetched twice votes once."""
    # Interval 1000-1100 with cushion 100: start window 900-1200, end window 1000-1200.
    # A read at 1150 sits in both, so it is returned by both fetches. template_length is 0
    # so that only split-read evidence is exercised.
    primary = _make_pe_mate(
        "frag",
        1150,
        cigartuples=[(0, 50), (4, 30)],  # 50M30S - right clip, first part
        sa_tag="chr1,1051,+,30S50M,60,0;",  # second part before the first part -> DUP
        template_length=0,
    )
    supplementary = _make_pe_mate(
        "frag",
        1050,
        cigartuples=[(4, 30), (0, 50)],
        sa_tag="chr1,1151,+,50M30S,60,0;",
        is_supplementary=True,
        template_length=0,
    )
    bam_file = pe_bam_factory([primary, supplementary])

    evidence = _analyze(bam_file, pe_config(), start=1000, end=1100)

    assert evidence.total_reads == 1
    assert evidence.duplication_reads == 1
    assert evidence.dup_insert_sizes == [150]
    # Without the per-alignment dedup the doubly fetched read would be written twice
    assert len(evidence.supporting_reads) == 1


# --- regressions for the two paired-end bugs ------------------------------


def test_mate2_split_evidence_is_detected(pe_bam_factory):
    """A fragment whose second mate carries the SA tag is no longer discarded.

    The single-end path keys processed reads on (query_name, read group), which both mates
    share, so the first mate fetched claimed the key and the SA-carrying mate was dropped.
    """
    # template_length is 0 on both mates so that only split-read evidence is exercised
    mate1 = _make_pe_mate("frag", PE_LEFT_START, template_length=0)
    mate2 = _make_pe_mate(
        "frag",
        2050,
        is_read1=False,
        cigartuples=[(0, 50), (4, 30)],
        sa_tag="chr1,951,+,30S50M,60,0;",  # first part after second part -> DUP
        template_length=0,
    )
    bam_file = pe_bam_factory([mate1, mate2])

    pe_evidence = _analyze(bam_file, pe_config())
    assert pe_evidence.duplication_reads == 1
    assert pe_evidence.total_reads == 1

    # The single-end path still drops it, which is the bug this test pins
    se_evidence = _analyze(bam_file, None)
    assert se_evidence.duplication_reads == 0


def test_supplementary_lookup_is_mate_aware(pe_bam_factory):
    """Each mate's supplementary alignments stay in their own bucket, with no cross-mate leakage."""
    mate1 = _make_pe_mate(
        "frag",
        PE_LEFT_START,
        cigartuples=[(0, 50), (4, 30)],
        sa_tag="chr1,2051,+,30S50M,60,0;",
        template_length=0,
    )
    mate1_supp = _make_pe_mate(
        "frag",
        2050,
        cigartuples=[(4, 30), (0, 50)],
        sa_tag="chr1,951,+,50M30S,60,0;",
        is_supplementary=True,
        template_length=0,
    )
    mate2 = _make_pe_mate(
        "frag",
        960,
        is_read1=False,
        cigartuples=[(0, 50), (4, 30)],
        sa_tag="chr1,2061,+,30S50M,60,0;",
        template_length=0,
    )
    mate2_supp = _make_pe_mate(
        "frag",
        2060,
        is_read1=False,
        cigartuples=[(4, 30), (0, 50)],
        sa_tag="chr1,961,+,50M30S,60,0;",
        is_supplementary=True,
        template_length=0,
    )
    bam_file = pe_bam_factory([mate1, mate1_supp, mate2, mate2_supp])

    evidence = _analyze(bam_file, pe_config())

    assert evidence.total_reads == 1
    assert evidence.deletion_reads == 1
    assert len(evidence.supporting_reads) == 2

    keys = {_mate_aware_key(read) for read, _ in evidence.supporting_reads}
    assert len(keys) == 2, "the two mates must not collapse onto one key"
    assert set(evidence.supplementary_reads) == keys
    for key, supplementaries in evidence.supplementary_reads.items():
        assert len(supplementaries) == 1, f"cross-mate leakage for {key}"

    mate1_key = next(key for key in keys if key[1] == 0x40)
    assert evidence.supplementary_reads[mate1_key][0].reference_start == 2050


def test_single_end_path_ignores_discordant_pairs(pe_bam_factory):
    """With paired-end support off, a discordant pair contributes nothing but the denominator."""
    bam_file = pe_bam_factory([_make_pe_mate("frag", PE_LEFT_START), _make_right_mate("frag")])

    evidence = _analyze(bam_file, None)

    assert evidence.duplication_reads == 0
    assert evidence.deletion_reads == 0
    assert evidence.total_reads == 1
    assert evidence.del_median_insert_size is None


def test_disabled_paired_end_config_ignores_discordant_pairs(pe_bam_factory):
    """An explicitly disabled config behaves exactly like passing None."""
    bam_file = pe_bam_factory([_make_pe_mate("frag", PE_LEFT_START), _make_right_mate("frag")])

    evidence = _analyze(bam_file, PairedEndConfig(enabled=False))

    assert evidence.deletion_reads == 0
    assert evidence.total_reads == 1


# --- median insert size ---------------------------------------------------


def test_summarize_fragments_prefers_split_values_over_pair_values():
    """Split values are base-pair exact, so any of them suppress the pair values for that class."""
    split_dup, pair_dup, pair_del = _FragmentEvidence(), _FragmentEvidence(), _FragmentEvidence()
    split_dup.split.add(_make_pe_mate("a", PE_LEFT_START), "DUP", 100)
    pair_dup.pair.add(_make_pe_mate("b", PE_LEFT_START), "DUP", 400)
    pair_del.pair.add(_make_pe_mate("c", PE_LEFT_START), "DEL", 800)
    fragments = {("a", "RG"): split_dup, ("b", "RG"): pair_dup, ("c", "RG"): pair_del}

    evidence = _summarize_fragments(fragments, ("chr1", 1000, 2000))

    assert evidence.dup_insert_sizes == [100]  # a single split value outranks the pair value
    assert evidence.del_insert_sizes == [800]  # no split values, so the pair value is used


def _write_evidence_bam(bam_file, fasta_file, config):
    """Run analyze_cnv_breakpoints on the standard DEL interval; return (read group IDs, BAM reads)."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".vcf", delete=False) as vcf_f:
        vcf_path = vcf_f.name
        vcf_f.write("##fileformat=VCFv4.2\n")
        vcf_f.write("##contig=<ID=chr1,length=10000>\n")
        vcf_f.write('##INFO=<ID=SVTYPE,Number=1,Type=String,Description="Type of structural variant">\n')
        vcf_f.write('##INFO=<ID=END,Number=1,Type=Integer,Description="End position of the variant">\n')
        vcf_f.write("#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n")
        vcf_f.write("chr1\t1001\t.\tN\t<DEL>\t.\tPASS\tSVTYPE=DEL;END=2000\n")

    with tempfile.NamedTemporaryFile(suffix=".vcf", delete=False) as out_f:
        output_vcf_path = out_f.name
    with tempfile.NamedTemporaryFile(suffix=".bam", delete=False) as out_bam_f:
        output_bam_path = out_bam_f.name

    try:
        analyze_cnv_breakpoints(
            bam_file=bam_file,
            vcf_file=vcf_path,
            reference_fasta=fasta_file,
            cushion=PE_CUSHION,
            output_file=output_vcf_path,
            output_bam=output_bam_path,
            paired_end_config=config,
        )
        with pysam.AlignmentFile(output_bam_path, "rb") as bam_out:
            rg_ids = {rg["ID"] for rg in bam_out.header.to_dict()["RG"]}
            reads = list(bam_out)
    finally:
        Path(vcf_path).unlink(missing_ok=True)
        Path(output_vcf_path).unlink(missing_ok=True)
        Path(output_bam_path).unlink(missing_ok=True)
    return rg_ids, reads


def test_pair_only_support_is_written_under_the_pair_read_group(pe_bam_factory, dummy_fasta_file):
    """Pair-only fragments enter the evidence BAM, but under their own read group, never DUP/DEL.

    breakpoint_refinement selects reads by RG == DUP/DEL and would read breakpoints off the mates'
    incidental soft clips instead of a real junction.
    """
    bam_file = pe_bam_factory([_make_pe_mate("frag", PE_LEFT_START), _make_right_mate("frag")])

    rg_ids, reads = _write_evidence_bam(bam_file, dummy_fasta_file, pe_config())

    assert {"DUP", "DEL", PAIR_READ_GROUP} <= rg_ids
    assert sorted(read.reference_start for read in reads) == [PE_LEFT_START, PE_RIGHT_START]
    assert all(read.query_name == "frag" for read in reads)
    assert {read.get_tag("RG") for read in reads} == {PAIR_READ_GROUP}


def test_single_end_evidence_bam_header_has_no_pair_read_group(pe_bam_factory, dummy_fasta_file):
    """Without paired-end evidence the evidence BAM header is unchanged: no PAIR read group."""
    bam_file = pe_bam_factory([_make_pe_mate("frag", PE_LEFT_START), _make_right_mate("frag")])

    rg_ids, reads = _write_evidence_bam(bam_file, dummy_fasta_file, pe_config(enabled=False))

    assert rg_ids == {"REF1", "REF2", "DUP", "DEL"}
    assert reads == []


# --- single-end input -----------------------------------------------------


def test_paired_end_flag_is_harmless_on_single_end_input(dummy_fasta_file, temp_bam_file, temp_vcf_file):
    """Passing --paired-end on unpaired input leaves the output unchanged: unpaired reads never qualify."""
    with tempfile.NamedTemporaryFile(suffix=".vcf", delete=False) as f:
        pe_output = f.name
    with tempfile.NamedTemporaryFile(suffix=".vcf", delete=False) as f:
        se_output = f.name

    try:
        for output_path, config in ((se_output, None), (pe_output, pe_config())):
            analyze_cnv_breakpoints(
                bam_file=temp_bam_file,
                vcf_file=temp_vcf_file,
                reference_fasta=dummy_fasta_file,
                cushion=PE_CUSHION,
                output_file=output_path,
                paired_end_config=config,
            )

        assert Path(se_output).read_text() == Path(pe_output).read_text()
    finally:
        Path(se_output).unlink(missing_ok=True)
        Path(pe_output).unlink(missing_ok=True)


# --- CLI ------------------------------------------------------------------


def test_parser_paired_end_defaults():
    """Paired-end support is off by default, so the CLI keeps single-end behavior."""
    args = get_parser().parse_args(["--bam-file", "in.bam", "--vcf-file", "in.vcf", "--reference-fasta", "ref.fasta"])

    assert args.paired_end is False
    assert args.min_pair_span == MIN_PAIR_SPAN
    assert args.min_pair_mapping_quality == 20
    assert PairedEndConfig.from_args(args) == PairedEndConfig()


def test_parser_paired_end_options():
    """The paired-end flags map onto PairedEndConfig fields."""
    args = get_parser().parse_args(
        [
            "--bam-file",
            "in.bam",
            "--vcf-file",
            "in.vcf",
            "--reference-fasta",
            "ref.fasta",
            "--paired-end",
            "--min-pair-span",
            "900",
            "--min-pair-mapping-quality",
            "30",
        ]
    )

    config = PairedEndConfig.from_args(args)

    assert config.enabled is True
    assert config.min_pair_span == 900
    assert config.min_pair_mapping_quality == 30
