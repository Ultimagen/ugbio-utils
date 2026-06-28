from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest
import ugbio_core.vcfbed.vcftools as vcftools
from ugbio_filtering import vcf_hmer_update


@pytest.fixture
def resources_dir():
    return Path(__file__).parent.parent / "resources" / "hmer"


class OffsetFasta:
    """Wrapper around pyfaidx.Fasta that offsets coordinates for subset reference.

    This allows us to use a small subset of the reference genome (e.g., chr1:113750000-113950000)
    while the code expects full coordinates (e.g., chr1:113800000).
    """

    def __init__(self, fasta_path, offset_start, **kwargs):
        from pyfaidx import Fasta

        self.fasta = Fasta(str(fasta_path), **kwargs)
        self.offset = offset_start  # e.g., 113750000
        # Get the actual chromosome name from the file (should be 'chr1')
        self.chrom_name = list(self.fasta.keys())[0]

    def __getitem__(self, chrom):
        """Return a chromosome with coordinate offsetting."""
        # Map any requested chromosome to our single chromosome with offset
        return OffsetChromosome(self.fasta[self.chrom_name], self.offset)

    def close(self):
        self.fasta.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


class OffsetChromosome:
    """Wrapper that translates coordinates by subtracting an offset."""

    def __init__(self, sequence, offset):
        self.sequence = sequence
        self.offset = offset

    def __getitem__(self, key):
        """Translate coordinates by subtracting offset."""
        if isinstance(key, slice):
            start = (key.start - self.offset) if key.start is not None else None
            stop = (key.stop - self.offset) if key.stop is not None else None
            return self.sequence[start:stop]
        else:
            return self.sequence[key - self.offset]

    def __str__(self):
        return str(self.sequence)

    def __len__(self):
        return len(self.sequence)


class TestVcfHmerUpdateSystem:
    """System tests for vcf_hmer_update module.

    These tests use real BAM/VCF files and a subset reference genome with coordinate offsetting.
    The test data covers chr1:113800000-113900000 region.
    """

    def test_hmer_update_basic(self, tmpdir, resources_dir):
        """Test basic hmer update functionality with real BAM files.

        This test runs the complete vcf_hmer_update pipeline with:
        - Input VCF with h-indel variants
        - Normal and tumor BAM files
        - Germline VCFs for both samples
        - BED interval file
        - Subset reference genome (chr1:113750000-113950000)

        The test validates that:
        - Output VCF is created and readable
        - Required INFO fields (mixture, tot_score) are present
        - Output matches the golden file
        """
        # Setup: Create offset wrapper for subset reference
        subset_fasta_path = str(resources_dir / "chr1_subset.fasta")
        offset_fasta = OffsetFasta(subset_fasta_path, offset_start=113750000, rebuild=False)

        # Patch pyfaidx.Fasta to use our wrapper
        with patch("ugbio_filtering.vcf_hmer_update.Fasta", return_value=offset_fasta):
            output_vcf = str(Path(tmpdir) / "output.vcf.gz")

            # Build config matching the "hir" launch.json configuration
            config = {
                "vcf_file": str(resources_dir / "hmer_test_input.vcf.gz"),
                "normal_reads_files": str(resources_dir / "hmer_test_normal.bam"),
                "normal_reads_files_list": [str(resources_dir / "hmer_test_normal.bam")],
                "tumor_reads_file": str(resources_dir / "hmer_test_tumor.bam"),
                "vcf_out_file": output_vcf,
                "min_hmer": 4,
                "max_hmer": 12,
                "min_tumor_cvg": 0,  # Relaxed for test
                "min_normal_cvg": 0,  # Relaxed for test
                "pseudocounts": 0.5,
                "target_intervals_bed_file": str(resources_dir / "hmer_test_target.bed"),
                "tumor_germline_file": str(resources_dir / "hmer_test_tumor_germline.vcf.gz"),
                "normal_germline_file": str(resources_dir / "hmer_test_normal_germline.vcf.gz"),
                "normal_germline_files": [str(resources_dir / "hmer_test_normal_germline.vcf.gz")],
                "verbose": True,
                "score_bound": 2.0,
                "mixture_bound": 0.03,
                "ref_fasta_path": subset_fasta_path,
            }

            # Run the pipeline
            vcf_hmer_update.variant_calling(config)

            # Validate output exists and is readable
            assert Path(output_vcf).exists(), "Output VCF was not created"
            out_df = vcftools.get_vcf_df(output_vcf, custom_info_fields=["mixture", "tot_score"])

            # Validate required fields are present
            assert "mixture" in out_df.columns, "mixture field not found in output"
            assert "tot_score" in out_df.columns, "tot_score field not found in output"

            # Validate at least one variant was annotated
            annotated = out_df[out_df["mixture"].notna()]
            assert len(annotated) > 0, "No variants were annotated with mixture/tot_score"

            # Compare with golden file
            golden_file = resources_dir / "hmer_test_expected_output.vcf.gz"
            if golden_file.exists():
                golden_df = vcftools.get_vcf_df(str(golden_file), custom_info_fields=["mixture", "tot_score"])

                # Compare key fields for annotated variants
                # We only compare variants that have mixture annotations
                out_annotated = out_df[out_df["mixture"].notna()][["chrom", "pos", "mixture", "tot_score"]].reset_index(
                    drop=True
                )
                golden_annotated = golden_df[golden_df["mixture"].notna()][
                    ["chrom", "pos", "mixture", "tot_score"]
                ].reset_index(drop=True)

                # Allow small floating point differences
                pd.testing.assert_frame_equal(
                    out_annotated,
                    golden_annotated,
                    rtol=1e-4,
                    check_dtype=False,  # Allow int vs float differences
                )

    def test_hmer_update_verbose_fields(self, tmpdir, resources_dir):
        """Test that verbose mode adds all expected direction-specific fields.

        When verbose=True, the output should contain:
        - fw_* fields (forward strand metrics)
        - bw_* fields (backward strand metrics)
        - ref_hmer_size, other_variant flags
        """
        # Setup: Create offset wrapper for subset reference
        subset_fasta_path = str(resources_dir / "chr1_subset.fasta")
        offset_fasta = OffsetFasta(subset_fasta_path, offset_start=113750000, rebuild=False)

        # Patch pyfaidx.Fasta to use our wrapper
        with patch("ugbio_filtering.vcf_hmer_update.Fasta", return_value=offset_fasta):
            output_vcf = str(Path(tmpdir) / "output_verbose.vcf.gz")

            # Build config with verbose=True
            config = {
                "vcf_file": str(resources_dir / "hmer_test_input.vcf.gz"),
                "normal_reads_files": str(resources_dir / "hmer_test_normal.bam"),
                "normal_reads_files_list": [str(resources_dir / "hmer_test_normal.bam")],
                "tumor_reads_file": str(resources_dir / "hmer_test_tumor.bam"),
                "vcf_out_file": output_vcf,
                "min_hmer": 4,
                "max_hmer": 12,
                "min_tumor_cvg": 0,
                "min_normal_cvg": 0,
                "pseudocounts": 0.5,
                "target_intervals_bed_file": str(resources_dir / "hmer_test_target.bed"),
                "tumor_germline_file": str(resources_dir / "hmer_test_tumor_germline.vcf.gz"),
                "normal_germline_file": str(resources_dir / "hmer_test_normal_germline.vcf.gz"),
                "normal_germline_files": [str(resources_dir / "hmer_test_normal_germline.vcf.gz")],
                "verbose": True,
                "score_bound": 2.0,
                "mixture_bound": 0.03,
                "ref_fasta_path": subset_fasta_path,
            }

            # Run the pipeline
            vcf_hmer_update.variant_calling(config)

            # Load output with verbose fields
            out_df = vcftools.get_vcf_df(
                output_vcf,
                custom_info_fields=[
                    "mixture",
                    "tot_score",
                    "fw_mixture",
                    "bw_mixture",
                    "fw_tot_score",
                    "bw_tot_score",
                    "ref_hmer_size",
                    "other_variant",
                ],
            )

            # Check verbose fields are present in annotated variants
            annotated = out_df[out_df["mixture"].notna()]
            if len(annotated) > 0:
                # Check direction-specific fields
                assert "fw_mixture" in out_df.columns, "fw_mixture field not found"
                assert "bw_mixture" in out_df.columns, "bw_mixture field not found"
                assert "fw_tot_score" in out_df.columns, "fw_tot_score field not found"
                assert "bw_tot_score" in out_df.columns, "bw_tot_score field not found"

                # Check debugging fields
                assert "ref_hmer_size" in out_df.columns, "ref_hmer_size field not found"
                assert "other_variant" in out_df.columns, "other_variant field not found"

                # Validate field values are within expected ranges
                # Handle both scalar and tuple values (for multi-allelic variants)
                mixture_values = annotated["mixture"].apply(lambda x: x[0] if isinstance(x, tuple) else x)
                assert (mixture_values >= 0).all() and (mixture_values <= 1).all(), "mixture values outside [0,1] range"
                assert annotated["tot_score"].notna().any(), "tot_score has no valid values"

    def test_hmer_update_with_coverage_thresholds(self, tmpdir, resources_dir):
        """Test that high coverage thresholds prevent variant annotation.

        When min_tumor_cvg=200 and min_normal_cvg=200 are set, variants with lower
        coverage should not be annotated (no mixture/tot_score values). Instead,
        they should have insufficient_cvg flag set.

        This validates that coverage filtering works correctly.
        """
        # Setup: Create offset wrapper for subset reference
        subset_fasta_path = str(resources_dir / "chr1_subset.fasta")
        offset_fasta = OffsetFasta(subset_fasta_path, offset_start=113750000, rebuild=False)

        # Patch pyfaidx.Fasta to use our wrapper
        with patch("ugbio_filtering.vcf_hmer_update.Fasta", return_value=offset_fasta):
            output_vcf = str(Path(tmpdir) / "output_high_cvg.vcf.gz")

            # Build config with HIGH coverage thresholds
            config = {
                "vcf_file": str(resources_dir / "hmer_test_input.vcf.gz"),
                "normal_reads_files": str(resources_dir / "hmer_test_normal.bam"),
                "normal_reads_files_list": [str(resources_dir / "hmer_test_normal.bam")],
                "tumor_reads_file": str(resources_dir / "hmer_test_tumor.bam"),
                "vcf_out_file": output_vcf,
                "min_hmer": 4,
                "max_hmer": 12,
                "min_tumor_cvg": 200,  # HIGH threshold - should exclude most variants
                "min_normal_cvg": 200,  # HIGH threshold - should exclude most variants
                "pseudocounts": 0.5,
                "target_intervals_bed_file": str(resources_dir / "hmer_test_target.bed"),
                "tumor_germline_file": str(resources_dir / "hmer_test_tumor_germline.vcf.gz"),
                "normal_germline_file": str(resources_dir / "hmer_test_normal_germline.vcf.gz"),
                "normal_germline_files": [str(resources_dir / "hmer_test_normal_germline.vcf.gz")],
                "verbose": True,
                "score_bound": 2.0,
                "mixture_bound": 0.03,
                "ref_fasta_path": subset_fasta_path,
            }

            # Run the pipeline
            vcf_hmer_update.variant_calling(config)

            # Load output with mixture and insufficient_cvg fields
            out_df = vcftools.get_vcf_df(
                output_vcf,
                custom_info_fields=["mixture", "tot_score", "insufficient_cvg"],
            )

            # With high coverage thresholds, very few (or no) variants should be annotated
            annotated = out_df[out_df["mixture"].notna()]

            # Most variants should NOT be annotated due to insufficient coverage
            total_variants = len(out_df)
            annotated_count = len(annotated)

            # Assert that most variants are NOT annotated (at least 95% should be unannotated)
            assert annotated_count < total_variants * 0.05, (
                f"Too many variants annotated with high coverage thresholds: "
                f"{annotated_count}/{total_variants} (expected < 5%)"
            )

            # Check that insufficient_cvg flag is set for skipped variants
            if "insufficient_cvg" in out_df.columns:
                # At least some variants should have insufficient_cvg flag
                insufficient_cvg_count = (out_df["insufficient_cvg"] == 1).sum()
                assert (
                    insufficient_cvg_count > 0
                ), "No variants marked with insufficient_cvg flag despite high thresholds"
