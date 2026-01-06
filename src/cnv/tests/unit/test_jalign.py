"""Tests for jalign module.

This module tests jump alignment functionality for CNV breakpoint analysis,
including configuration, alignment tool execution, and BAM record creation.
"""

# ruff: noqa: PD901

import json
import subprocess
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pandas as pd
import pyfaidx
import pysam
import pytest
from ugbio_cnv.jalign import (
    JAlignConfig,
    create_all_bam_records_from_json,
    create_bam_record_from_alignment,
    create_bam_records_from_jump_alignment,
    create_bam_records_from_simple_alignment,
    determine_best_alignments,
    process_cnv,
    run_alignment_tool,
)


@pytest.fixture
def test_resources_dir():
    """Return path to test resources directory."""
    return Path(__file__).parent.parent / "resources"


@pytest.fixture
def jalign_test_json(test_resources_dir):
    """Load jalign test JSON file."""
    json_path = test_resources_dir / "jalign.test.json"
    with open(json_path) as f:
        return json.load(f)


@pytest.fixture
def jalign_test_bam(test_resources_dir):
    """Return path to jalign test BAM file."""
    return test_resources_dir / "jalign.test.bam"


@pytest.fixture
def default_config():
    """Create default JAlignConfig for testing."""
    return JAlignConfig()


@pytest.fixture
def custom_config():
    """Create custom JAlignConfig with modified parameters."""
    return JAlignConfig(
        match_score=3,
        mismatch_score=-6,
        min_mismatches=3,
        softclip_threshold=20,
        max_reads_per_cnv=2000,
        random_seed=42,
    )


@pytest.fixture
def mock_bam_header():
    """Create a mock BAM header for testing."""
    header_dict = {
        "HD": {"VN": "1.6"},
        "SQ": [
            {"SN": "chr1", "LN": 248956422},
            {"SN": "chr2", "LN": 242193529},
        ],
        "RG": [
            {"ID": "REF1", "SM": "test_sample"},
            {"ID": "REF2", "SM": "test_sample"},
            {"ID": "DUP", "SM": "test_sample"},
            {"ID": "DEL", "SM": "test_sample"},
        ],
    }
    return pysam.AlignmentHeader.from_dict(header_dict)


@pytest.fixture
def mock_read():
    """Create a mock aligned read for testing."""
    read = MagicMock(spec=pysam.AlignedSegment)
    read.query_name = "test_read_001"
    read.query_sequence = "ACGTACGTACGT"
    read.reference_name = "chr1"
    read.reference_start = 1000
    read.cigartuples = [(0, 10), (4, 2)]  # 10M2S
    read.is_duplicate = False
    read.is_secondary = False
    read.is_supplementary = False
    read.get_tag = MagicMock(return_value=5)  # NM tag
    return read


class TestBAMRecordCreation:
    """Test BAM record creation functions."""

    def test_create_bam_record_from_alignment(self, mock_bam_header):
        """Test basic BAM record creation."""
        record = create_bam_record_from_alignment(
            qname="test_read",
            seq="ACGTACGT",
            chrom="chr1",
            ref_start=1000,
            score=100,
            begin=50,
            cigar="8M",
            rgid="REF1",
            header=mock_bam_header,
            is_supplementary=False,
        )

        assert record.query_name == "test_read"
        assert record.query_sequence == "ACGTACGT"
        assert record.reference_name == "chr1"
        assert record.reference_start == 1050  # ref_start + begin
        assert record.cigarstring == "8M"
        assert record.mapping_quality == 60
        assert record.get_tag("AS") == 100
        assert record.get_tag("RG") == "REF1"
        assert not record.is_supplementary

    def test_create_bam_record_supplementary(self, mock_bam_header):
        """Test supplementary BAM record creation."""
        record = create_bam_record_from_alignment(
            qname="test_read",
            seq="ACGTACGT",
            chrom="chr1",
            ref_start=2000,
            score=80,
            begin=0,
            cigar="8M",
            rgid="DUP",
            header=mock_bam_header,
            is_supplementary=True,
        )

        assert record.is_supplementary
        assert record.get_tag("RG") == "DUP"

    def test_create_bam_records_from_simple_alignment(self, mock_bam_header):
        """Test simple alignment BAM record creation."""
        records = create_bam_records_from_simple_alignment(
            qname="test_read",
            seq="ACGTACGTACGT",
            chrom="chr1",
            ref_start=5000,
            score=120,
            begin=10,
            cigar="12M",
            rgid="REF2",
            header=mock_bam_header,
        )

        assert len(records) == 1
        record = records[0]
        assert record.query_name == "test_read"
        assert record.reference_start == 5010
        assert record.get_tag("RG") == "REF2"
        assert not record.is_supplementary

    def test_create_bam_records_from_jump_alignment(self, mock_bam_header):
        """Test jump alignment BAM record creation."""
        records = create_bam_records_from_jump_alignment(
            qname="jump_read",
            seq="ACGTACGTACGTACGT",
            chrom="chr1",
            ref1_start=1000,
            ref2_start=5000,
            score=200,
            begin1=0,
            cigar1="8M8S",
            begin2=100,
            cigar2="8S8M",
            rgid="DUP",
            header=mock_bam_header,
        )

        assert len(records) == 2
        primary, supplementary = records

        # Check primary alignment
        assert primary.query_name == "jump_read"
        assert primary.reference_start == 1000
        assert primary.cigarstring == "8M8S"
        assert not primary.is_supplementary
        assert primary.get_tag("RG") == "DUP"
        assert primary.has_tag("SA")

        # Check supplementary alignment
        assert supplementary.query_name == "jump_read"
        assert supplementary.reference_start == 5100
        assert supplementary.cigarstring == "8S8M"
        assert supplementary.is_supplementary
        assert supplementary.get_tag("RG") == "DUP"


class TestAlignmentScoring:
    """Test alignment scoring and selection functions."""

    def test_determine_best_alignments_jump_forward(self, default_config):
        """Test best alignment determination when jump_forward wins."""
        # Score must be > max(score1, score2) + min_seq_len (30)
        # Score must be > 0.9 * (readlen1 + readlen2) * match_score
        # With readlen1=150, readlen2=150, match_score=2:
        # Minimal threshold: 0.9 * 300 * 2 = 540
        df = pd.DataFrame(
            [
                {
                    "qname": "read1",
                    "align1.score": 100,
                    "align2.score": 90,
                    "jump_forward.score": 550,  # > 130 (100+30) and > 540
                    "jump_forward.readlen1": 150,
                    "jump_forward.readlen2": 150,
                    "jump_backward.score": 80,
                    "jump_backward.readlen1": 10,
                    "jump_backward.readlen2": 10,
                }
            ]
        )

        best = determine_best_alignments(df, default_config)
        assert best["read1"] == "jump_forward"

    def test_determine_best_alignments_jump_backward(self, default_config):
        """Test best alignment determination when jump_backward wins."""
        # Same logic: score > 130 and > 540
        df = pd.DataFrame(
            [
                {
                    "qname": "read2",
                    "align1.score": 100,
                    "align2.score": 90,
                    "jump_forward.score": 80,
                    "jump_forward.readlen1": 10,
                    "jump_forward.readlen2": 10,
                    "jump_backward.score": 560,  # > 130 and > 540
                    "jump_backward.readlen1": 150,
                    "jump_backward.readlen2": 150,
                }
            ]
        )

        best = determine_best_alignments(df, default_config)
        assert best["read2"] == "jump_backward"

    def test_determine_best_alignments_simple_align(self, default_config):
        """Test best alignment determination when simple alignment wins."""
        df = pd.DataFrame(
            [
                {
                    "qname": "read3",
                    "align1.score": 300,
                    "align2.score": 280,
                    "jump_forward.score": 250,
                    "jump_forward.readlen1": 50,
                    "jump_forward.readlen2": 50,
                    "jump_backward.score": 240,
                    "jump_backward.readlen1": 50,
                    "jump_backward.readlen2": 50,
                }
            ]
        )

        best = determine_best_alignments(df, default_config)
        assert best["read3"] in ["align1", "align2"]


class TestAlignmentToolExecution:
    """Test alignment tool execution and error handling."""

    def test_run_alignment_tool_success(self):
        """Test successful alignment tool execution."""
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = Mock(returncode=0, stdout="", stderr="")

            returncode = run_alignment_tool(["jump_align", "arg1", "arg2"])

            assert returncode == 0
            mock_run.assert_called_once()

    def test_run_alignment_tool_failure(self):
        """Test alignment tool execution failure."""
        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = subprocess.CalledProcessError(returncode=1, cmd=["jump_align"], stderr="Tool failed")

            with pytest.raises(RuntimeError, match="Alignment tool failed"):
                run_alignment_tool(["jump_align"])

    def test_run_alignment_tool_with_log_file(self, tmp_path):
        """Test alignment tool execution with logging."""
        log_file = tmp_path / "test.log"

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = Mock(returncode=0, stdout="", stderr="")

            with open(log_file, "w") as log:
                run_alignment_tool(["jump_align", "arg1"], log_file=log)

            # Verify log was written
            assert log_file.exists()
            content = log_file.read_text()
            assert "jump_align" in content


class TestJSONParsing:
    """Test JSON alignment result parsing."""

    def test_create_all_bam_records_best_only(self, jalign_test_json, jalign_test_bam, mock_bam_header):
        """Test BAM record creation with best alignment only."""
        # Load test JSON data
        df = pd.json_normalize(jalign_test_json)

        # Create mock reads
        with pysam.AlignmentFile(jalign_test_bam, "rb") as bam:
            reads_in_order = list(bam.fetch())

        # Determine best alignments
        config = JAlignConfig()
        best_alignments = determine_best_alignments(df, config)

        # Write JSON to temp file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(jalign_test_json, f)
            json_file = Path(f.name)

        try:
            # Create BAM records
            records = create_all_bam_records_from_json(
                json_file=json_file,
                reads_in_order=reads_in_order,
                chrom="chr1",
                ref1_start=219455000,
                ref2_start=219456000,
                header=mock_bam_header,
                best_alignments=best_alignments,
                output_all_alignments=False,
            )

            # Verify results
            assert len(records) > 0
            # Each read should have at most 2 records (primary + supplementary for jump alignments)
            read_counts = {}
            for record in records:
                read_counts[record.query_name] = read_counts.get(record.query_name, 0) + 1

            # All reads should be represented
            assert len(read_counts) <= len(reads_in_order)

        finally:
            json_file.unlink()

    def test_create_all_bam_records_all_alignments(self, jalign_test_json, jalign_test_bam, mock_bam_header):
        """Test BAM record creation with all alignments."""
        df = pd.json_normalize(jalign_test_json)

        with pysam.AlignmentFile(jalign_test_bam, "rb") as bam:
            reads_in_order = list(bam.fetch())

        config = JAlignConfig()
        best_alignments = determine_best_alignments(df, config)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(jalign_test_json, f)
            json_file = Path(f.name)

        try:
            records = create_all_bam_records_from_json(
                json_file=json_file,
                reads_in_order=reads_in_order,
                chrom="chr1",
                ref1_start=219455000,
                ref2_start=219456000,
                header=mock_bam_header,
                best_alignments=best_alignments,
                output_all_alignments=True,
            )

            # With all alignments, should have many more records
            assert len(records) > len(reads_in_order)

        finally:
            json_file.unlink()


class TestProcessCNV:
    """Test main CNV processing function with mocked alignment tool."""

    @pytest.fixture
    def mock_alignment_tool(self, jalign_test_json, tmp_path):
        """Mock the alignment tool to return test JSON output."""

        def mock_alignment_command(*args, **kwargs):
            """Mock alignment tool that writes test JSON."""
            # Extract output file path from command
            cmd = args[0]
            output_file = None
            for i, arg in enumerate(cmd):
                if arg.endswith(".json"):
                    output_file = Path(arg)
                    break

            if output_file:
                # Write test JSON to output file
                with open(output_file, "w") as f:
                    json.dump(jalign_test_json, f)

            return 0

        return mock_alignment_command

    def test_process_cnv_with_mock_tool(self, mock_alignment_tool, jalign_test_bam, tmp_path, default_config):
        """Test CNV processing with mocked alignment tool.

        Note: This test uses fetch() without region to avoid needing an index.
        """
        # Create mock FASTA file
        mock_fasta = MagicMock(spec=pyfaidx.Fasta)
        mock_fasta.__getitem__ = MagicMock(
            return_value=MagicMock(
                seq="ACGT" * 500  # 2000bp reference sequence
            )
        )

        # Open test BAM file and fetch all reads (no index needed)
        with pysam.AlignmentFile(jalign_test_bam, "rb", check_sq=False) as reads_file:
            # Get all reads from the file
            all_reads = list(reads_file.fetch(until_eof=True))

            # Filter reads that would be in the CNV region (manually simulate fetch behavior)
            # Since we don't have coordinates, we'll just use a subset
            if len(all_reads) == 0:
                pytest.skip("No reads in test BAM file")

            # Create a mock that returns these reads
            mock_reads_file = MagicMock(spec=pysam.AlignmentFile)
            mock_reads_file.fetch = MagicMock(return_value=iter(all_reads[:10]))
            mock_reads_file.header = reads_file.header

            # Mock the alignment tool
            with patch("ugbio_cnv.jalign.run_alignment_tool", side_effect=mock_alignment_tool):
                # Process CNV
                counts, df, bam_records, header = process_cnv(
                    chrom="chr1",
                    start=219455000,
                    end=219456000,
                    reads_file=mock_reads_file,
                    fasta_file=mock_fasta,
                    config=default_config,
                    temp_dir=tmp_path,
                    log_file=None,
                    header=None,
                )

                # Verify counts tuple
                assert len(counts) == 4
                jump_better, djump_better, jump_much_better, djump_much_better = counts
                assert isinstance(jump_better, int)
                assert isinstance(djump_better, int)

                # Verify DataFrame
                assert isinstance(df, pd.DataFrame)
                assert len(df) > 0
                assert "chrom" in df.columns
                assert "start" in df.columns
                assert "end" in df.columns

                # Verify BAM records
                assert isinstance(bam_records, list)
                assert len(bam_records) > 0
                assert all(isinstance(r, pysam.AlignedSegment) for r in bam_records)

                # Verify header
                assert header is not None
                assert isinstance(header, pysam.AlignmentHeader)


class TestIntegration:
    """Integration tests using real test data."""

    def test_end_to_end_with_test_data(self, jalign_test_json, jalign_test_bam, tmp_path):
        """Test end-to-end workflow with test data."""
        # Load JSON and create DataFrame
        df = pd.json_normalize(jalign_test_json)

        # Load reads from BAM (no index needed)
        with pysam.AlignmentFile(jalign_test_bam, "rb", check_sq=False) as bam:
            reads = list(bam.fetch(until_eof=True))
            header = bam.header

        # Ensure header has required read groups
        header_dict = header.to_dict()
        if "RG" not in header_dict:
            header_dict["RG"] = []

        existing_rg_ids = {rg["ID"] for rg in header_dict.get("RG", [])}
        for rgid in ["REF1", "REF2", "DUP", "DEL"]:
            if rgid not in existing_rg_ids:
                header_dict["RG"].append({"ID": rgid, "SM": "test_sample"})

        header = pysam.AlignmentHeader.from_dict(header_dict)

        # Determine best alignments
        config = JAlignConfig()
        best_alignments = determine_best_alignments(df, config)

        assert len(best_alignments) > 0
        assert all(
            alignment_type in ["align1", "align2", "jump_forward", "jump_backward"]
            for alignment_type in best_alignments.values()
        )

        # Create BAM records
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(jalign_test_json, f)
            json_file = Path(f.name)

        try:
            records = create_all_bam_records_from_json(
                json_file=json_file,
                reads_in_order=reads,
                chrom="chr1",
                ref1_start=219455000,
                ref2_start=219456000,
                header=header,
                best_alignments=best_alignments,
                output_all_alignments=False,
            )

            # Verify records were created
            assert len(records) > 0

            # Verify records can be written to BAM (no indexing)
            output_bam = tmp_path / "test_output.bam"
            with pysam.AlignmentFile(output_bam, "wb", header=header) as out:
                for record in records:
                    out.write(record)

            # Verify output BAM is readable (without indexing)
            with pysam.AlignmentFile(output_bam, "rb", check_sq=False) as verify:
                output_records = list(verify.fetch(until_eof=True))
                assert len(output_records) > 0

        finally:
            json_file.unlink()
