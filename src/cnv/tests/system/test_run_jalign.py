"""Integration tests for run_jalign CLI."""

import json
import os
import shutil
from os.path import join as pjoin
from pathlib import Path
from unittest.mock import Mock, patch

import pandas as pd
import pysam
import pytest
from ugbio_cnv import run_jalign
from ugbio_core.test_utils import compare_vcfs


def is_para_jalign_available():
    """Check if para_jalign tool is available on the system."""
    # Check if para_jalign is in PATH
    if shutil.which("para_jalign"):
        return True
    # Check if para_jalign is in /opt
    if os.path.exists("/opt/para_jalign"):
        return True
    return False


@pytest.fixture
def resources_dir():
    """Return path to test resources directory."""
    return Path(__file__).parent.parent / "resources"


@pytest.fixture
def mock_para_jalign(resources_dir):
    """Mock para_jalign tool to write JSON output files.

    This fixture patches subprocess.run to simulate para_jalign execution.
    Uses pre-generated expected JSON files (created with real para_jalign tool)
    so that mocked tests produce the same results as real tool execution.
    """

    def mock_run(*args, **kwargs):
        """Mock subprocess.run for para_jalign."""
        cmd = args[0] if args else kwargs.get("args", [])

        # Find the output JSON file and input file from command
        json_output = None
        input_file = None
        for arg in cmd:
            if arg.endswith(".json"):
                json_output = Path(arg)
            elif arg.endswith(".txt") and "jalign_chr" in arg:
                input_file = Path(arg)

        if json_output and input_file:
            # Extract CNV coordinates from input filename
            # Format: jalign_chr1_START_END_PID.txt
            parts = input_file.stem.split("_")
            if len(parts) >= 4:
                chrom = parts[1]
                start = parts[2]
                end = parts[3]

                # Map to expected JSON file (using 0-based coordinates from code)
                json_file_map = {
                    ("chr1", "2651000", "2658000"): "expected_chr1_2651000_2658000.jalign.json",
                    ("chr1", "2652000", "2677500"): "expected_chr1_2652000_2677500.jalign.json",
                    ("chr1", "2678000", "2696000"): "expected_chr1_2678000_2696000.jalign.json",
                    ("chr1", "2690000", "2701000"): "expected_chr1_2690000_2701000.jalign.json",
                }

                expected_json = json_file_map.get((chrom, start, end))
                if expected_json:
                    expected_json_path = resources_dir / expected_json
                    if expected_json_path.exists():
                        # Copy expected JSON (real para_jalign output) to mock location
                        with open(expected_json_path) as f:
                            data = json.load(f)
                        with open(json_output, "w") as f:
                            json.dump(data, f)
                    else:
                        # Fallback: use jalign.test.json subset
                        fallback_json = resources_dir / "jalign.test.json"
                        if fallback_json.exists():
                            with open(fallback_json) as f:
                                all_data = json.load(f)
                            # Use first 20 reads as mock data
                            subset_data = all_data[:20]
                            with open(json_output, "w") as f:
                                json.dump(subset_data, f)
                        else:
                            # Last resort: write empty array
                            with open(json_output, "w") as f:
                                json.dump([], f)

        # Return successful mock result
        return Mock(returncode=0, stdout="", stderr="")

    return mock_run


class TestRunJalign:
    """Integration tests for run_jalign CLI with mocked para_jalign tool."""

    def test_run_jalign_multi_thread_mocked(self, tmp_path, resources_dir, mock_para_jalign):
        """Test run_jalign with multi-threaded processing and mocked tool.

        Validates that parallel processing produces valid outputs.
        """
        # Input files
        input_cram = pjoin(resources_dir, "test.jalign.cram")
        cnv_vcf = pjoin(resources_dir, "test.jalign.vcf.gz")
        ref_fasta = pjoin(resources_dir, "chr1.3M.fasta.gz")
        output_prefix = str(tmp_path / "test_output_mt")

        # Expected output files
        output_vcf = f"{output_prefix}.jalign.vcf.gz"
        output_bam = f"{output_prefix}.jalign.bam"
        output_csv = f"{output_prefix}.jalign.csv"

        # Mock para_jalign execution
        with patch("subprocess.run", side_effect=mock_para_jalign):
            # Run CLI with 2 threads
            exit_code = run_jalign.main(
                [
                    input_cram,
                    cnv_vcf,
                    ref_fasta,
                    output_prefix,
                    "--min-mismatches",
                    "1",
                    "--gap-open-score",
                    "-12",
                    "--softclip-threshold",
                    "20",
                    "--threads",
                    "2",
                    "--tool-path",
                    "para_jalign",
                ]
            )

        # Validate successful execution
        assert exit_code == 0, "run_jalign should exit successfully with multiple threads"

        # Validate output files exist
        assert os.path.exists(output_vcf), f"Output VCF not found: {output_vcf}"
        assert os.path.exists(output_bam), f"Output BAM not found: {output_bam}"
        assert os.path.exists(output_csv), f"Output CSV not found: {output_csv}"

        # Validate VCF structure (same checks as single-threaded)
        with pysam.VariantFile(output_vcf) as vcf:
            records = list(vcf)
            assert len(records) > 0, "VCF should contain CNV records"

            for rec in records:
                assert "JALIGN_DUP_SUPPORT" in rec.info
                assert "JALIGN_DEL_SUPPORT" in rec.info
                assert "JALIGN_DUP_SUPPORT_STRONG" in rec.info
                assert "JALIGN_DEL_SUPPORT_STRONG" in rec.info

        # Validate BAM structure
        with pysam.AlignmentFile(output_bam, "rb") as bam:
            rg_ids = {rg["ID"] for rg in bam.header["RG"]}
            expected_rgs = {"REF1", "REF2", "DUP", "DEL"}
            assert expected_rgs.issubset(rg_ids)

    def test_run_jalign_compare_golden_outputs(self, tmp_path, resources_dir, mock_para_jalign):
        """Test run_jalign output against golden reference files.

        This test compares generated outputs against pre-validated golden files.
        Since the mock uses real para_jalign JSON outputs, the results should
        match the golden files that were generated with the real tool.
        """
        # Input files
        input_cram = pjoin(resources_dir, "test.jalign.cram")
        cnv_vcf = pjoin(resources_dir, "test.jalign.vcf.gz")
        ref_fasta = pjoin(resources_dir, "chr1.3M.fasta.gz")
        output_prefix = str(tmp_path / "test_output_golden")

        # Golden output files
        golden_vcf = pjoin(resources_dir, "expected_test.jalign.vcf.gz")
        golden_csv = pjoin(resources_dir, "expected_test.jalign.csv")

        # Skip test if golden files or expected JSON files don't exist yet
        expected_json_files = [
            "expected_chr1_2651000_2658000.jalign.json",
            "expected_chr1_2652000_2677500.jalign.json",
            "expected_chr1_2678000_2696000.jalign.json",
            "expected_chr1_2690000_2701000.jalign.json",
        ]

        if not os.path.exists(golden_vcf):
            pytest.skip("Golden output files not yet generated")

        for json_file in expected_json_files:
            if not os.path.exists(pjoin(resources_dir, json_file)):
                pytest.skip(f"Expected JSON file {json_file} not yet generated")

        # Mock para_jalign execution (will use real JSON outputs)
        with patch("subprocess.run", side_effect=mock_para_jalign):
            # Run CLI
            exit_code = run_jalign.main(
                [
                    input_cram,
                    cnv_vcf,
                    ref_fasta,
                    output_prefix,
                    "--min-mismatches",
                    "1",
                    "--gap-open-score",
                    "-12",
                    "--softclip-threshold",
                    "20",
                    "--threads",
                    "1",
                    "--tool-path",
                    "para_jalign",
                    "--random-seed",
                    "0",  # Ensure deterministic output
                ]
            )

        assert exit_code == 0

        # Compare VCF outputs
        output_vcf = f"{output_prefix}.jalign.vcf.gz"
        compare_vcfs(output_vcf, golden_vcf)

        # Compare CSV outputs (if golden exists)
        if os.path.exists(golden_csv):
            output_csv = f"{output_prefix}.jalign.csv"
            expected_df = pd.read_csv(golden_csv)
            actual_df = pd.read_csv(output_csv)

            # Compare structure and row count
            assert list(expected_df.columns) == list(actual_df.columns), "CSV columns should match"
            assert len(expected_df) == len(actual_df), "CSV row count should match"

    @pytest.mark.skipif(
        not is_para_jalign_available(), reason="para_jalign not available (not in PATH or /opt/para_jalign)"
    )
    def test_run_jalign_with_real_para_jalign_single_record(self, tmp_path, resources_dir):
        """Test run_jalign with real para_jalign tool on first CNV record.

        This test validates:
        - Successful execution with real alignment tool
        - Output file generation with actual alignment results
        - VCF annotation with real alignment counts
        - BAM generation with real realigned reads

        Only processes the first CNV record to keep test execution time reasonable.
        Uses pre-generated single-record VCF from resources directory.
        """
        # Input files
        input_cram = pjoin(resources_dir, "test.jalign.cram")
        ref_fasta = pjoin(resources_dir, "chr1.3M.fasta.gz")
        single_record_vcf_gz = pjoin(resources_dir, "test.jalign.single_record.vcf.gz")
        output_prefix = str(tmp_path / "test_output_real")

        # Expected output files
        output_vcf = f"{output_prefix}.jalign.vcf.gz"
        output_bam = f"{output_prefix}.jalign.bam"
        output_csv = f"{output_prefix}.jalign.csv"

        # Determine para_jalign path
        tool_path = "para_jalign"
        if not shutil.which("para_jalign") and os.path.exists("/opt/para_jalign"):
            tool_path = "/opt/para_jalign"

        # Run CLI with real tool
        exit_code = run_jalign.main(
            [
                input_cram,
                single_record_vcf_gz,
                ref_fasta,
                output_prefix,
                "--min-mismatches",
                "1",
                "--gap-open-score",
                "-12",
                "--softclip-threshold",
                "20",
                "--threads",
                "1",
                "--tool-path",
                tool_path,
            ]
        )

        # Validate successful execution
        assert exit_code == 0, "run_jalign should exit successfully with real tool"

        # Validate output files exist
        assert os.path.exists(output_vcf), f"Output VCF not found: {output_vcf}"
        assert os.path.exists(output_bam), f"Output BAM not found: {output_bam}"
        assert os.path.exists(output_csv), f"Output CSV not found: {output_csv}"

        # Validate VCF contains exactly one record with jalign annotations
        with pysam.VariantFile(output_vcf) as vcf:
            records = list(vcf)
            assert len(records) == 1, "VCF should contain exactly one CNV record"

            rec = records[0]
            # Check all jalign INFO fields are present
            assert "JALIGN_DUP_SUPPORT" in rec.info
            assert "JALIGN_DEL_SUPPORT" in rec.info
            assert "JALIGN_DUP_SUPPORT_STRONG" in rec.info
            assert "JALIGN_DEL_SUPPORT_STRONG" in rec.info

            # Values should be non-negative integers
            assert rec.info["JALIGN_DUP_SUPPORT"] >= 0
            assert rec.info["JALIGN_DEL_SUPPORT"] >= 0
            assert rec.info["JALIGN_DUP_SUPPORT_STRONG"] >= 0
            assert rec.info["JALIGN_DEL_SUPPORT_STRONG"] >= 0

            print(f"CNV {rec.chrom}:{rec.start}-{rec.stop}")
            print(
                f"  DUP support: {rec.info['JALIGN_DUP_SUPPORT']} " f"(strong: {rec.info['JALIGN_DUP_SUPPORT_STRONG']})"
            )
            print(
                f"  DEL support: {rec.info['JALIGN_DEL_SUPPORT']} " f"(strong: {rec.info['JALIGN_DEL_SUPPORT_STRONG']})"
            )

        # Validate BAM structure
        with pysam.AlignmentFile(output_bam, "rb") as bam:
            # Check header contains required read groups
            assert "RG" in bam.header, "BAM header should contain read groups"
            rg_ids = {rg["ID"] for rg in bam.header["RG"]}
            expected_rgs = {"REF1", "REF2", "DUP", "DEL"}
            assert expected_rgs.issubset(rg_ids), f"Expected read groups {expected_rgs}"

            # Count reads by read group
            reads_by_rg = {"REF1": 0, "REF2": 0, "DUP": 0, "DEL": 0}
            for read in bam.fetch(until_eof=True):
                if read.has_tag("RG"):
                    rg = read.get_tag("RG")
                    if rg in reads_by_rg:
                        reads_by_rg[rg] += 1

            print(f"Realigned reads by read group: {reads_by_rg}")

            # At least some reads should be present
            total_reads = sum(reads_by_rg.values())
            assert total_reads > 0, "BAM should contain realigned reads"

        # Validate CSV exists and has content
        assert os.path.getsize(output_csv) > 0, "CSV file should not be empty"

        print("Test completed successfully with real para_jalign tool")
        print(f"Output files: {output_vcf}, {output_bam}, {output_csv}")
