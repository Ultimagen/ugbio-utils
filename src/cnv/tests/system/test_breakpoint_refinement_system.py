"""System tests for CNV breakpoint refinement functionality."""

import os
from os.path import join as pjoin
from pathlib import Path

import pysam
import pytest
from ugbio_cnv.breakpoint_refinement import main


@pytest.fixture
def resources_dir():
    """Return path to test resources directory."""
    return Path(__file__).parent.parent / "resources"


class TestBreakpointRefinementIntegration:
    """Integration tests for the complete breakpoint refinement pipeline."""

    def test_refine_cnv_cli_with_multiple_variants(self, tmpdir, resources_dir):
        """
        Test CNV breakpoint refinement with real data from chr17.

        Uses extracted subset from launch.json configuration containing:
        - chr17:19287001-29287001 (273 variants, 10 Mb region)
        - Multiple variants with refined breakpoints (CIPOS reductions up to 989 bp)
        - Tests multi-variant workflow with both refined and unrefined variants

        Expected refined variants (examples):
        - cnmops_dup_3785 at 22978001: CIPOS (-500,501) -> (-6,6), reduction=989bp
        - cnmops_dup_3859 at 24029001: CIPOS (-500,501) -> (-13,13), reduction=975bp
        - cnmops_del_3044 at 26861001: CIPOS (-500,501) -> (-76,76), reduction=849bp
        """
        # Input files from resources
        input_vcf = pjoin(resources_dir, "HG002.chr17.vcf.gz")
        jalign_bam = pjoin(resources_dir, "HG002.chr17.jalign.bam")
        split_bam = pjoin(resources_dir, "HG002.chr17.split.bam")
        output_vcf = pjoin(tmpdir, "refined.vcf.gz")

        # Verify input files exist
        assert os.path.exists(input_vcf), f"Input VCF not found: {input_vcf}"
        assert os.path.exists(jalign_bam), f"JALIGN BAM not found: {jalign_bam}"
        assert os.path.exists(split_bam), f"Split BAM not found: {split_bam}"

        # Read original VCF to capture baseline
        with pysam.VariantFile(input_vcf) as vcf_in:
            original_records = list(vcf_in)

        original_count = len(original_records)
        assert original_count == 273, f"Expected 273 variants, got {original_count}"

        # Build lookup for original records by ID (positions may change during refinement)
        original_dict = {r.id: r for r in original_records}

        # Run the CLI (mirrors launch.json configuration)
        exit_code = main(
            [
                "--input-vcf",
                input_vcf,
                "--bam-files",
                jalign_bam,
                split_bam,
                "--output-vcf",
                output_vcf,
            ]
        )

        # Verify successful execution
        assert exit_code == 0, "Refinement CLI should exit with code 0"
        assert os.path.exists(output_vcf), f"Output VCF not created: {output_vcf}"

        # Parse refined VCF
        with pysam.VariantFile(output_vcf) as vcf_out:
            refined_records = list(vcf_out)

        assert len(refined_records) == original_count, "Output should have same number of records as input"

        # Build lookup for refined records by ID
        refined_dict = {r.id: r for r in refined_records}

        # Track refinement statistics
        refined_count = 0
        unrefined_count = 0
        refinement_details = []

        for variant_id, _ in original_dict.items():
            if variant_id in refined_dict:
                original_rec = original_dict[variant_id]
                refined_rec = refined_dict[variant_id]

                original_cipos = original_rec.info.get("CIPOS", (0, 0))
                refined_cipos = refined_rec.info.get("CIPOS", (0, 0))

                original_ci_size = original_cipos[1] - original_cipos[0]
                refined_ci_size = refined_cipos[1] - refined_cipos[0]

                if refined_ci_size < original_ci_size:
                    refined_count += 1
                    reduction = original_ci_size - refined_ci_size
                    refinement_details.append(
                        {
                            "id": original_rec.id,
                            "pos": original_rec.pos,
                            "original_cipos": original_cipos,
                            "refined_cipos": refined_cipos,
                            "reduction": reduction,
                        }
                    )
                else:
                    unrefined_count += 1

                # Verify SVTYPE preserved for all records
                assert (
                    refined_rec.info["SVTYPE"] == original_rec.info["SVTYPE"]
                ), f"SVTYPE should be preserved for {variant_id}"

        # Log refinement summary
        print("\n=== Refinement Summary ===")
        print(f"Total variants:     {original_count}")
        print(f"Refined variants:   {refined_count}")
        print(f"Unrefined variants: {unrefined_count}")
        print("\nTop 10 refined variants by CIPOS reduction:")
        refinement_details.sort(key=lambda x: x["reduction"], reverse=True)
        for i, detail in enumerate(refinement_details[:10], 1):
            print(
                f"  {i}. {detail['id']} @ {detail['pos']}: "
                f"{detail['original_cipos']} -> {detail['refined_cipos']} "
                f"(reduction: {detail['reduction']} bp)"
            )

        # Assert that we found refined variants
        assert refined_count > 0, "Expected at least one variant to be refined"

        # Verify specific known refined variants from user's table (match by ID)
        expected_refined = {
            "cnmops_dup_3785": {"min_reduction": 900, "orig_pos": 22978001},  # Expected ~989 bp
            "cnmops_dup_3859": {"min_reduction": 900, "orig_pos": 24029001},  # Expected ~975 bp
            "cnmops_del_3044": {"min_reduction": 800, "orig_pos": 26861001},  # Expected ~849 bp
        }

        found_expected = 0
        for variant_id, expected in expected_refined.items():
            if variant_id in original_dict and variant_id in refined_dict:
                original_rec = original_dict[variant_id]
                refined_rec = refined_dict[variant_id]

                # Verify position is approximately correct
                assert abs(original_rec.pos - expected["orig_pos"]) < 10, (
                    f"Expected variant {variant_id} near position {expected['orig_pos']}, " f"got {original_rec.pos}"
                )

                original_cipos = original_rec.info.get("CIPOS", (0, 0))
                refined_cipos = refined_rec.info.get("CIPOS", (0, 0))

                original_ci_size = original_cipos[1] - original_cipos[0]
                refined_ci_size = refined_cipos[1] - refined_cipos[0]

                reduction = original_ci_size - refined_ci_size

                # Verify refinement occurred with expected magnitude
                assert reduction >= expected["min_reduction"], (
                    f"Variant {variant_id}: "
                    f"Expected reduction >= {expected['min_reduction']} bp, got {reduction} bp"
                )

                print(f"\n✓ Verified expected refinement for {variant_id}:")
                print(f"  Original position: chr17:{original_rec.pos}")
                print(f"  Refined position:  chr17:{refined_rec.pos}")
                print(f"  CIPOS: {original_cipos} -> {refined_cipos}")
                print(f"  Reduction: {reduction} bp")

                found_expected += 1

        assert found_expected >= 2, f"Expected to find at least 2 known refined variants, found {found_expected}"

    def test_refine_cnv_single_bam(self, tmpdir, resources_dir):
        """Test breakpoint refinement with a single BAM file."""
        # Input files
        input_vcf = pjoin(resources_dir, "HG002.chr17.vcf.gz")
        jalign_bam = pjoin(resources_dir, "HG002.chr17.jalign.bam")
        output_vcf = pjoin(tmpdir, "refined_single_bam.vcf.gz")

        # Read original for baseline
        with pysam.VariantFile(input_vcf) as vcf_in:
            original_records = list(vcf_in)

        original_count = len(original_records)

        # Run with single BAM
        exit_code = main(
            [
                "--input-vcf",
                input_vcf,
                "--bam-files",
                jalign_bam,
                "--output-vcf",
                output_vcf,
            ]
        )

        assert exit_code == 0, "Should succeed with single BAM"
        assert os.path.exists(output_vcf), "Output VCF should be created"

        # Verify output is valid
        with pysam.VariantFile(output_vcf) as vcf_out:
            refined_records = list(vcf_out)

        assert len(refined_records) == original_count, "Output should contain all variants"

        # Check for any refined variants
        refined_count = 0
        for i, (orig, refined) in enumerate(zip(original_records, refined_records)):
            orig_cipos = orig.info.get("CIPOS", (0, 0))
            refined_cipos = refined.info.get("CIPOS", (0, 0))

            orig_size = orig_cipos[1] - orig_cipos[0]
            refined_size = refined_cipos[1] - refined_cipos[0]

            if refined_size < orig_size:
                refined_count += 1

        print(f"\nSingle BAM refinement: {refined_count}/{original_count} variants refined")
        assert refined_count > 0, "Expected at least some variants to be refined with single BAM"
