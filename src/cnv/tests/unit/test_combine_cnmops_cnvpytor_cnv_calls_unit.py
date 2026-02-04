import os
import sys
from pathlib import Path
from unittest.mock import patch

import pysam
import pytest
from ugbio_cnv import combine_cnmops_cnvpytor_cnv_calls


@pytest.fixture
def resources_dir():
    """Fixture providing path to test resources directory."""
    return Path(__file__).parent.parent / "resources"


@pytest.fixture
def sample_name():
    """Fixture providing a sample name for testing."""
    return "TEST_SAMPLE"


def create_test_fasta(fasta_path: str, sequences: dict[str, str]) -> None:
    """Create a test FASTA file with specified sequences.

    Parameters
    ----------
    fasta_path : str
        Path to create the FASTA file
    sequences : dict[str, str]
        Dictionary mapping chromosome names to sequences
    """
    with open(fasta_path, "w") as f:
        for chrom, seq in sequences.items():
            f.write(f">{chrom}\n")
            # Write sequence in 50-char lines
            for i in range(0, len(seq), 50):
                f.write(seq[i : i + 50] + "\n")

    # Create index
    from pyfaidx import Fasta

    Fasta(fasta_path)  # This creates the .fai file


def create_test_vcf_for_gap_perc(vcf_path: str, records: list[dict], contigs: dict[str, int]) -> None:
    """Create a test VCF file with CNV records.

    Parameters
    ----------
    vcf_path : str
        Path to create the VCF file
    records : list[dict]
        List of record dicts with keys: chrom, start, stop, alleles, svtype
    contigs : dict[str, int]
        Dictionary mapping contig names to lengths
    """
    header = pysam.VariantHeader()
    header.add_meta("fileformat", value="VCFv4.2")

    for contig, length in contigs.items():
        header.contigs.add(contig, length=length)

    header.info.add("SVTYPE", number=1, type="String", description="Type of structural variant")
    header.info.add("SVLEN", number=".", type="Integer", description="Length of structural variant")
    header.add_sample("test_sample")

    with pysam.VariantFile(vcf_path, "w", header=header) as vcf:
        for rec_data in records:
            rec = vcf.new_record(
                contig=rec_data["chrom"],
                start=rec_data["start"],
                stop=rec_data["stop"],
                alleles=rec_data.get("alleles", ("N", "<DEL>")),
            )
            if "svtype" in rec_data:
                rec.info["SVTYPE"] = rec_data["svtype"]
            vcf.write(rec)

    pysam.tabix_index(vcf_path, preset="vcf", force=True)


class TestAnnotateVcfWithGapPerc:
    """Tests for annotate_vcf_with_gap_perc function."""

    def test_gap_perc_all_ns(self, tmp_path):
        """Test that a region with 100% N bases returns GAP_PERCENTAGE = 1.0."""
        fasta_path = os.path.join(tmp_path, "ref.fa")
        input_vcf = os.path.join(tmp_path, "input.vcf.gz")
        output_vcf = os.path.join(tmp_path, "output.vcf.gz")

        # Create reference with all N's in the region 0-100
        create_test_fasta(fasta_path, {"chr1": "N" * 200})
        create_test_vcf_for_gap_perc(
            input_vcf,
            [{"chrom": "chr1", "start": 0, "stop": 100, "svtype": "DEL"}],
            {"chr1": 200},
        )

        combine_cnmops_cnvpytor_cnv_calls.annotate_vcf_with_gap_perc(input_vcf, fasta_path, output_vcf)

        with pysam.VariantFile(output_vcf) as vcf:
            records = list(vcf)
            assert len(records) == 1
            assert "GAP_PERCENTAGE" in records[0].info
            # region_len = stop - start + 1 = 100 - 0 + 1 = 101
            # seq = genome[0:101] = 101 N's
            # gap_perc = 101/101 = 1.0
            assert records[0].info["GAP_PERCENTAGE"] == 1.0

    def test_gap_perc_no_ns(self, tmp_path):
        """Test that a region with 0% N bases returns GAP_PERCENTAGE = 0.0."""
        fasta_path = os.path.join(tmp_path, "ref.fa")
        input_vcf = os.path.join(tmp_path, "input.vcf.gz")
        output_vcf = os.path.join(tmp_path, "output.vcf.gz")

        # Create reference with all A's (no N's)
        create_test_fasta(fasta_path, {"chr1": "A" * 200})
        create_test_vcf_for_gap_perc(
            input_vcf,
            [{"chrom": "chr1", "start": 0, "stop": 100, "svtype": "DEL"}],
            {"chr1": 200},
        )

        combine_cnmops_cnvpytor_cnv_calls.annotate_vcf_with_gap_perc(input_vcf, fasta_path, output_vcf)

        with pysam.VariantFile(output_vcf) as vcf:
            records = list(vcf)
            assert len(records) == 1
            assert "GAP_PERCENTAGE" in records[0].info
            assert records[0].info["GAP_PERCENTAGE"] == 0.0

    def test_gap_perc_mixed_bases(self, tmp_path):
        """Test that a region with mixed N's and bases returns correct fraction."""
        fasta_path = os.path.join(tmp_path, "ref.fa")
        input_vcf = os.path.join(tmp_path, "input.vcf.gz")
        output_vcf = os.path.join(tmp_path, "output.vcf.gz")

        # Create reference: 50 N's followed by 51 A's = region has 50 N's in first 50 positions
        # Region 0:100 extracts 101 bases (positions 0-100 inclusive)
        # seq = 50 N's + 51 A's = 50 N's out of 101 bases
        create_test_fasta(fasta_path, {"chr1": "N" * 50 + "A" * 51 + "G" * 100})
        create_test_vcf_for_gap_perc(
            input_vcf,
            [{"chrom": "chr1", "start": 0, "stop": 100, "svtype": "DEL"}],
            {"chr1": 201},
        )

        combine_cnmops_cnvpytor_cnv_calls.annotate_vcf_with_gap_perc(input_vcf, fasta_path, output_vcf)

        with pysam.VariantFile(output_vcf) as vcf:
            records = list(vcf)
            assert len(records) == 1
            # region_len = 101, n_count = 50
            expected_gap_perc = 50 / 101
            assert records[0].info["GAP_PERCENTAGE"] == pytest.approx(expected_gap_perc, rel=1e-4)

    def test_gap_perc_lowercase_ns_counted(self, tmp_path):
        """Test that lowercase 'n' bases are also counted as gaps."""
        fasta_path = os.path.join(tmp_path, "ref.fa")
        input_vcf = os.path.join(tmp_path, "input.vcf.gz")
        output_vcf = os.path.join(tmp_path, "output.vcf.gz")

        # Create reference with lowercase n's
        create_test_fasta(fasta_path, {"chr1": "n" * 200})
        create_test_vcf_for_gap_perc(
            input_vcf,
            [{"chrom": "chr1", "start": 0, "stop": 100, "svtype": "DEL"}],
            {"chr1": 200},
        )

        combine_cnmops_cnvpytor_cnv_calls.annotate_vcf_with_gap_perc(input_vcf, fasta_path, output_vcf)

        with pysam.VariantFile(output_vcf) as vcf:
            records = list(vcf)
            assert len(records) == 1
            # Lowercase n's should be counted as N's after .upper() conversion
            # region_len = 101, n_count = 101
            assert records[0].info["GAP_PERCENTAGE"] == 1.0

    def test_gap_perc_multiple_records(self, tmp_path):
        """Test that multiple VCF records are all annotated correctly."""
        fasta_path = os.path.join(tmp_path, "ref.fa")
        input_vcf = os.path.join(tmp_path, "input.vcf.gz")
        output_vcf = os.path.join(tmp_path, "output.vcf.gz")

        # Create reference with different gap patterns in different regions
        # Position 0-50: all N's (51 bases)
        # Position 51-100: all A's (50 bases)
        # Position 101-150: 25 N's + 25 A's
        seq = "N" * 51 + "A" * 50 + "N" * 25 + "A" * 25 + "G" * 50
        create_test_fasta(fasta_path, {"chr1": seq})
        create_test_vcf_for_gap_perc(
            input_vcf,
            [
                {"chrom": "chr1", "start": 0, "stop": 50, "svtype": "DEL"},  # All N's
                {"chrom": "chr1", "start": 51, "stop": 100, "svtype": "DUP"},  # All A's
                {"chrom": "chr1", "start": 101, "stop": 150, "svtype": "DEL"},  # 50% N's
            ],
            {"chr1": 201},
        )

        combine_cnmops_cnvpytor_cnv_calls.annotate_vcf_with_gap_perc(input_vcf, fasta_path, output_vcf)

        with pysam.VariantFile(output_vcf) as vcf:
            records = list(vcf)
            assert len(records) == 3

            # Region 0-50: 51 N's, region_len = 51, gap_perc = 1.0
            assert records[0].info["GAP_PERCENTAGE"] == 1.0

            # Region 51-100: 50 A's, region_len = 50, gap_perc = 0.0
            assert records[1].info["GAP_PERCENTAGE"] == 0.0

            # Region 101-150: 25 N's + 25 A's, region_len = 50, gap_perc = 0.5
            assert records[2].info["GAP_PERCENTAGE"] == 0.5

    def test_gap_perc_header_added(self, tmp_path):
        """Test that GAP_PERCENTAGE INFO field is added to the VCF header."""
        fasta_path = os.path.join(tmp_path, "ref.fa")
        input_vcf = os.path.join(tmp_path, "input.vcf.gz")
        output_vcf = os.path.join(tmp_path, "output.vcf.gz")

        create_test_fasta(fasta_path, {"chr1": "A" * 100})
        create_test_vcf_for_gap_perc(
            input_vcf,
            [{"chrom": "chr1", "start": 0, "stop": 50, "svtype": "DEL"}],
            {"chr1": 100},
        )

        combine_cnmops_cnvpytor_cnv_calls.annotate_vcf_with_gap_perc(input_vcf, fasta_path, output_vcf)

        with pysam.VariantFile(output_vcf) as vcf:
            assert "GAP_PERCENTAGE" in vcf.header.info
            gap_perc_info = vcf.header.info["GAP_PERCENTAGE"]
            assert gap_perc_info.type == "Float"
            assert gap_perc_info.number == 1

    def test_gap_perc_output_vcf_indexed(self, tmp_path):
        """Test that output VCF is properly indexed after annotation."""
        fasta_path = os.path.join(tmp_path, "ref.fa")
        input_vcf = os.path.join(tmp_path, "input.vcf.gz")
        output_vcf = os.path.join(tmp_path, "output.vcf.gz")

        create_test_fasta(fasta_path, {"chr1": "A" * 100})
        create_test_vcf_for_gap_perc(
            input_vcf,
            [{"chrom": "chr1", "start": 0, "stop": 50, "svtype": "DEL"}],
            {"chr1": 100},
        )

        combine_cnmops_cnvpytor_cnv_calls.annotate_vcf_with_gap_perc(input_vcf, fasta_path, output_vcf)

        # Check that index file exists (.tbi or .csi)
        assert os.path.exists(output_vcf + ".tbi") or os.path.exists(output_vcf + ".csi")


class TestAnnotateVcfWithRegions:
    """Tests for annotate_vcf_with_regions function."""

    def create_test_bed(self, bed_path: str, regions: list[dict]) -> None:
        """Create a test BED file with annotation regions.

        Parameters
        ----------
        bed_path : str
            Path to create the BED file
        regions : list[dict]
            List of region dicts with keys: chrom, start, end, annotation
        """
        with open(bed_path, "w") as f:
            for region in regions:
                f.write(f"{region['chrom']}\t{region['start']}\t{region['end']}\t{region['annotation']}\n")

    def test_annotate_regions_basic(self, tmp_path, resources_dir):
        """Test basic annotation with single overlapping region."""
        input_vcf = os.path.join(tmp_path, "input.vcf.gz")
        annotation_bed = os.path.join(tmp_path, "annotations.bed")
        output_vcf = os.path.join(tmp_path, "output.vcf.gz")

        # Create input VCF with one CNV
        create_test_vcf_for_gap_perc(
            input_vcf,
            [{"chrom": "chr1", "start": 1000, "stop": 2000, "svtype": "DEL"}],
            {"chr1": 10000},
        )

        # Create annotation BED with overlapping region
        self.create_test_bed(
            annotation_bed,
            [{"chrom": "chr1", "start": 500, "end": 1500, "annotation": "Telomere_Centromere"}],
        )

        combine_cnmops_cnvpytor_cnv_calls.annotate_vcf_with_regions(
            input_vcf,
            annotation_bed,
            output_vcf,
            overlap_fraction=0.3,
            genome=str(resources_dir / "Homo_sapiens_assembly38.fasta.fai"),
        )

        with pysam.VariantFile(output_vcf) as vcf:
            records = list(vcf)
            assert len(records) == 1
            # Overlap: 1000-1500 = 500bp, CNV length = 1001bp, fraction = 500/1001 = 0.499 > 0.3
            assert "REGION_ANNOTATIONS" in records[0].info
            # pysam returns INFO fields with Number=. as tuples
            annotations = records[0].info["REGION_ANNOTATIONS"]
            assert isinstance(annotations, tuple)
            assert set(annotations) == {"Telomere_Centromere"}

    def test_annotate_regions_multiple_annotations(self, tmp_path, resources_dir):
        """Test that multiple overlapping regions combine their annotations."""
        input_vcf = os.path.join(tmp_path, "input.vcf.gz")
        annotation_bed = os.path.join(tmp_path, "annotations.bed")
        output_vcf = os.path.join(tmp_path, "output.vcf.gz")

        create_test_vcf_for_gap_perc(
            input_vcf,
            [{"chrom": "chr1", "start": 1000, "stop": 2000, "svtype": "DEL"}],
            {"chr1": 10000},
        )

        # Create multiple overlapping annotation regions
        self.create_test_bed(
            annotation_bed,
            [
                {"chrom": "chr1", "start": 500, "end": 1500, "annotation": "Telomere_Centromere"},
                {"chrom": "chr1", "start": 1200, "end": 1800, "annotation": "Coverage-Mappability"},
                {"chrom": "chr1", "start": 1700, "end": 2200, "annotation": "Clusters"},
            ],
        )

        combine_cnmops_cnvpytor_cnv_calls.annotate_vcf_with_regions(
            input_vcf,
            annotation_bed,
            output_vcf,
            overlap_fraction=0.3,
            genome=str(resources_dir / "Homo_sapiens_assembly38.fasta.fai"),
        )

        with pysam.VariantFile(output_vcf) as vcf:
            records = list(vcf)
            assert len(records) == 1
            assert "REGION_ANNOTATIONS" in records[0].info
            # pysam returns INFO fields with Number=. as tuples
            annotations = records[0].info["REGION_ANNOTATIONS"]
            assert isinstance(annotations, tuple)
            assert set(annotations) == {"Telomere_Centromere", "Coverage-Mappability", "Clusters"}

    def test_annotate_regions_pipe_separated_annotations(self, tmp_path, resources_dir):
        """Test handling of annotation values that contain pipe separators."""
        input_vcf = os.path.join(tmp_path, "input.vcf.gz")
        annotation_bed = os.path.join(tmp_path, "annotations.bed")
        output_vcf = os.path.join(tmp_path, "output.vcf.gz")

        create_test_vcf_for_gap_perc(
            input_vcf,
            [{"chrom": "chr1", "start": 1000, "stop": 2000, "svtype": "DEL"}],
            {"chr1": 10000},
        )

        # Create annotation BED with pipe-separated values
        self.create_test_bed(
            annotation_bed,
            [
                {"chrom": "chr1", "start": 500, "end": 1500, "annotation": "Telomere_Centromere|Coverage-Mappability"},
                {"chrom": "chr1", "start": 1700, "end": 2200, "annotation": "Clusters"},
            ],
        )

        combine_cnmops_cnvpytor_cnv_calls.annotate_vcf_with_regions(
            input_vcf,
            annotation_bed,
            output_vcf,
            overlap_fraction=0.3,
            genome=str(resources_dir / "Homo_sapiens_assembly38.fasta.fai"),
        )

        with pysam.VariantFile(output_vcf) as vcf:
            records = list(vcf)
            assert len(records) == 1
            assert "REGION_ANNOTATIONS" in records[0].info
            # pysam returns INFO fields with Number=. as tuples
            annotations = records[0].info["REGION_ANNOTATIONS"]
            assert isinstance(annotations, tuple)
            # All three annotations should be present and unique
            assert set(annotations) == {"Telomere_Centromere", "Coverage-Mappability", "Clusters"}

    def test_annotate_regions_below_threshold(self, tmp_path, resources_dir):
        """Test that CNVs below overlap threshold are not annotated."""
        input_vcf = os.path.join(tmp_path, "input.vcf.gz")
        annotation_bed = os.path.join(tmp_path, "annotations.bed")
        output_vcf = os.path.join(tmp_path, "output.vcf.gz")

        create_test_vcf_for_gap_perc(
            input_vcf,
            [{"chrom": "chr1", "start": 1000, "stop": 2000, "svtype": "DEL"}],
            {"chr1": 10000},
        )

        # Create annotation that overlaps by less than threshold
        # Overlap: 1000-1200 = 200bp, CNV length = 1001bp, fraction = 200/1001 = 0.199 < 0.5
        self.create_test_bed(
            annotation_bed,
            [{"chrom": "chr1", "start": 500, "end": 1200, "annotation": "Telomere_Centromere"}],
        )

        combine_cnmops_cnvpytor_cnv_calls.annotate_vcf_with_regions(
            input_vcf,
            annotation_bed,
            output_vcf,
            overlap_fraction=0.5,
            genome=str(resources_dir / "Homo_sapiens_assembly38.fasta.fai"),
        )

        with pysam.VariantFile(output_vcf) as vcf:
            records = list(vcf)
            assert len(records) == 1
            # Should not have annotation because overlap < 50%
            # With list-based format, field may exist but should be empty
            if "REGION_ANNOTATIONS" in records[0].info:
                annotations = records[0].info["REGION_ANNOTATIONS"]
                # Should be empty tuple or have no elements
                assert len(annotations) == 0, f"Expected no annotations but got: {annotations}"

    def test_annotate_regions_total_overlap_threshold(self, tmp_path, resources_dir):
        """Test that total overlap from multiple regions is calculated correctly."""
        input_vcf = os.path.join(tmp_path, "input.vcf.gz")
        annotation_bed = os.path.join(tmp_path, "annotations.bed")
        output_vcf = os.path.join(tmp_path, "output.vcf.gz")

        # CNV: chr1:1000-2000 (length = 1001bp)
        create_test_vcf_for_gap_perc(
            input_vcf,
            [{"chrom": "chr1", "start": 1000, "stop": 2000, "svtype": "DEL"}],
            {"chr1": 10000},
        )

        # Two regions that individually don't meet threshold but together do
        # Region 1: 1000-1250 = 250bp overlap (24.9%)
        # Region 2: 1250-1500 = 250bp overlap (24.9%)
        # Total: 500bp overlap (49.9%) >= 0.4 threshold
        self.create_test_bed(
            annotation_bed,
            [
                {"chrom": "chr1", "start": 800, "end": 1250, "annotation": "Telomere_Centromere"},
                {"chrom": "chr1", "start": 1250, "end": 1500, "annotation": "Coverage-Mappability"},
            ],
        )

        combine_cnmops_cnvpytor_cnv_calls.annotate_vcf_with_regions(
            input_vcf,
            annotation_bed,
            output_vcf,
            overlap_fraction=0.4,
            genome=str(resources_dir / "Homo_sapiens_assembly38.fasta.fai"),
        )

        with pysam.VariantFile(output_vcf) as vcf:
            records = list(vcf)
            assert len(records) == 1
            # Should have both annotations because total overlap >= 40%
            assert "REGION_ANNOTATIONS" in records[0].info
            # pysam returns INFO fields with Number=. as tuples
            annotations = records[0].info["REGION_ANNOTATIONS"]
            assert isinstance(annotations, tuple)
            assert set(annotations) == {"Telomere_Centromere", "Coverage-Mappability"}

    def test_annotate_regions_preserves_existing(self, tmp_path, resources_dir):
        """Test that existing REGION_ANNOTATIONS are preserved and merged."""
        input_vcf = os.path.join(tmp_path, "input.vcf.gz")
        annotation_bed = os.path.join(tmp_path, "annotations.bed")
        output_vcf = os.path.join(tmp_path, "output.vcf.gz")

        # Create input VCF with existing REGION_ANNOTATIONS
        header = pysam.VariantHeader()
        header.add_meta("fileformat", value="VCFv4.2")
        header.contigs.add("chr1", length=10000)
        header.info.add("SVTYPE", number=1, type="String", description="Type of structural variant")
        header.info.add(
            "REGION_ANNOTATIONS",
            number=".",
            type="String",
            description="Aggregated region-based annotations for the CNV",
        )
        header.add_sample("test_sample")

        with pysam.VariantFile(input_vcf, "w", header=header) as vcf:
            rec = vcf.new_record(contig="chr1", start=1000, stop=2000, alleles=("N", "<DEL>"))
            rec.info["SVTYPE"] = "DEL"
            rec.info["REGION_ANNOTATIONS"] = "ExistingAnnotation"
            vcf.write(rec)

        pysam.tabix_index(input_vcf, preset="vcf", force=True)

        # Create annotation BED
        self.create_test_bed(
            annotation_bed,
            [{"chrom": "chr1", "start": 500, "end": 1500, "annotation": "NewAnnotation"}],
        )

        combine_cnmops_cnvpytor_cnv_calls.annotate_vcf_with_regions(
            input_vcf,
            annotation_bed,
            output_vcf,
            overlap_fraction=0.3,
            genome=str(resources_dir / "Homo_sapiens_assembly38.fasta.fai"),
        )

        with pysam.VariantFile(output_vcf) as vcf:
            records = list(vcf)
            assert len(records) == 1
            assert "REGION_ANNOTATIONS" in records[0].info
            # pysam returns INFO fields with Number=. as tuples
            annotations = records[0].info["REGION_ANNOTATIONS"]
            assert isinstance(annotations, tuple)
            # Should have both existing and new annotations
            assert set(annotations) == {"ExistingAnnotation", "NewAnnotation"}

    def test_annotate_regions_multiple_cnvs(self, tmp_path, resources_dir):
        """Test annotation of multiple CNVs with different overlap patterns."""
        input_vcf = os.path.join(tmp_path, "input.vcf.gz")
        annotation_bed = os.path.join(tmp_path, "annotations.bed")
        output_vcf = os.path.join(tmp_path, "output.vcf.gz")

        # Create three CNVs
        create_test_vcf_for_gap_perc(
            input_vcf,
            [
                {"chrom": "chr1", "start": 1000, "stop": 2000, "svtype": "DEL"},  # Will be annotated
                {"chrom": "chr1", "start": 5000, "stop": 6000, "svtype": "DUP"},  # Will be annotated
                {"chrom": "chr1", "start": 8000, "stop": 9000, "svtype": "DEL"},  # Won't be annotated (no overlap)
            ],
            {"chr1": 10000},
        )

        # Create annotations
        self.create_test_bed(
            annotation_bed,
            [
                {"chrom": "chr1", "start": 500, "end": 1600, "annotation": "Telomere_Centromere"},
                {"chrom": "chr1", "start": 5200, "end": 6500, "annotation": "Coverage-Mappability"},
            ],
        )

        combine_cnmops_cnvpytor_cnv_calls.annotate_vcf_with_regions(
            input_vcf,
            annotation_bed,
            output_vcf,
            overlap_fraction=0.5,
            genome=str(resources_dir / "Homo_sapiens_assembly38.fasta.fai"),
        )

        with pysam.VariantFile(output_vcf) as vcf:
            records = list(vcf)
            assert len(records) == 3

            # First CNV: should be annotated
            assert "REGION_ANNOTATIONS" in records[0].info
            annotations = records[0].info["REGION_ANNOTATIONS"]
            assert isinstance(annotations, tuple)
            assert set(annotations) == {"Telomere_Centromere"}

            # Second CNV: should be annotated
            assert "REGION_ANNOTATIONS" in records[1].info
            annotations = records[1].info["REGION_ANNOTATIONS"]
            assert isinstance(annotations, tuple)
            assert set(annotations) == {"Coverage-Mappability"}

            # Third CNV: should not be annotated (no overlap)
            assert "REGION_ANNOTATIONS" not in records[2].info


def test_main_merge_records_calls_with_pick_best_true():
    """Test that main_merge_records calls merge_cnvs_in_vcf with pick_best=True."""
    test_input_vcf = "/path/to/input.vcf.gz"
    test_output_vcf = "/path/to/output.vcf.gz"
    test_distance = 1500

    # Mock sys.argv to simulate command-line arguments
    test_argv = [
        "merge_records",
        "--input_vcf",
        test_input_vcf,
        "--output_vcf",
        test_output_vcf,
        "--distance",
        str(test_distance),
    ]

    with patch.object(sys, "argv", test_argv):
        with patch("ugbio_cnv.combine_cnmops_cnvpytor_cnv_calls.merge_cnvs_in_vcf") as mock_merge:
            # Call the main_merge_records function
            combine_cnmops_cnvpytor_cnv_calls.main_merge_records()

            # Verify that merge_cnvs_in_vcf was called with the correct parameters
            mock_merge.assert_called_once_with(
                input_vcf=test_input_vcf,
                output_vcf=test_output_vcf,
                distance=test_distance,
                do_not_merge_collapsed_filtered=True,
                pick_best=True,
                ignore_sv_type=True,
            )


def test_main_analyze_breakpoints_calls_analyze_cnv_breakpoints():
    """Test that main_analyze_breakpoints calls analyze_cnv_breakpoints with correct params."""
    test_bam_file = "/path/to/reads.bam"
    test_vcf_file = "/path/to/cnvs.vcf.gz"
    test_output_file = "/path/to/output.vcf.gz"
    test_cushion = 200
    test_reference_fasta = "/path/to/ref.fa"

    # Mock sys.argv to simulate command-line arguments
    # Note: argument names use hyphens (--bam-file) as defined in analyze_cnv_breakpoint_reads.get_parser
    test_argv = [
        "analyze_breakpoint_reads",
        "--bam-file",
        test_bam_file,
        "--vcf-file",
        test_vcf_file,
        "--output-file",
        test_output_file,
        "--cushion",
        str(test_cushion),
        "--reference-fasta",
        test_reference_fasta,
    ]

    with patch.object(sys, "argv", test_argv):
        with patch("ugbio_cnv.combine_cnmops_cnvpytor_cnv_calls.analyze_cnv_breakpoints") as mock_analyze:
            # Call the main_analyze_breakpoints function
            combine_cnmops_cnvpytor_cnv_calls.main_analyze_breakpoints()

            # Verify that analyze_cnv_breakpoints was called with the correct parameters
            mock_analyze.assert_called_once_with(
                bam_file=test_bam_file,
                vcf_file=test_vcf_file,
                cushion=test_cushion,
                output_file=test_output_file,
                reference_fasta=test_reference_fasta,
            )
