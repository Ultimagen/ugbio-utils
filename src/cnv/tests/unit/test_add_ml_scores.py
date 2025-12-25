"""Unit tests for add_ml_scores tool in combine_cnmops_cnvpytor_cnv_calls module."""

import os
from pathlib import Path

import pysam
import pytest
from ugbio_cnv.combine_cnv_vcf_utils import annotate_vcf_with_ml_scores


@pytest.fixture
def resources_dir():
    """Fixture providing path to test resources directory."""
    return Path(__file__).parent.parent / "resources"


def create_test_vcf(vcf_path: str, records: list[dict], contigs: dict[str, int]) -> None:
    """Create a test VCF file with CNV records.

    Parameters
    ----------
    vcf_path : str
        Path to create the VCF file
    records : list[dict]
        List of record dicts with keys: chrom, start, stop, svtype
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
            if "svlen" in rec_data:
                rec.info["SVLEN"] = rec_data["svlen"]
            vcf.write(rec)

    pysam.tabix_index(vcf_path, preset="vcf", force=True)


def create_test_ml_bed(bed_path: str, intervals: list[dict]) -> None:
    """Create a test ML predictions BED file.

    Parameters
    ----------
    bed_path : str
        Path to create the BED file
    intervals : list[dict]
        List of interval dicts with keys: chrom, start, end, prob, pred, sample
    """
    with open(bed_path, "w") as bed_fh:
        bed_fh.write("#chrom\tstart\tstop\tprob\tpred\tinput_file_name\n")
        for interval in intervals:
            bed_fh.write(
                f"{interval['chrom']}\t{interval['start']}\t{interval['end']}\t"
                f"{interval['prob']}\t{interval.get('pred', 1)}\t{interval.get('sample', 'TEST_SAMPLE')}\n"
            )


class TestAnnotateVcfWithMlScores:
    """Tests for annotate_vcf_with_ml_scores function."""

    def test_both_endpoints_overlap(self, tmp_path):
        """Test that when both CNV endpoints overlap with ML intervals, max score is returned."""
        input_vcf = os.path.join(tmp_path, "input.vcf.gz")
        ml_bed = os.path.join(tmp_path, "ml_predictions.bed")
        output_vcf = os.path.join(tmp_path, "output.vcf.gz")

        # Create VCF with one CNV at chr1:1000-5000
        create_test_vcf(
            input_vcf,
            [{"chrom": "chr1", "start": 1000, "stop": 5000, "svtype": "DEL"}],
            {"chr1": 10000},
        )

        # Create ML BED with intervals covering both endpoints
        # Start position (1000) falls in [500, 1500) with prob 0.3
        # End position (5000) falls in [4500, 5500) with prob 0.7
        create_test_ml_bed(
            ml_bed,
            [
                {"chrom": "chr1", "start": 500, "end": 1500, "prob": 0.3},
                {"chrom": "chr1", "start": 4500, "end": 5500, "prob": 0.7},
            ],
        )

        annotate_vcf_with_ml_scores(input_vcf, ml_bed, output_vcf)

        # Verify CNV_PRED is max(0.3, 0.7) = 0.7
        with pysam.VariantFile(output_vcf) as vcf:
            records = list(vcf)
            assert len(records) == 1
            assert "CNV_PRED" in records[0].info
            assert records[0].info["CNV_PRED"] == pytest.approx(0.7, abs=1e-5)

    def test_one_endpoint_overlap(self, tmp_path):
        """Test that when only one endpoint overlaps, that score is returned."""
        input_vcf = os.path.join(tmp_path, "input.vcf.gz")
        ml_bed = os.path.join(tmp_path, "ml_predictions.bed")
        output_vcf = os.path.join(tmp_path, "output.vcf.gz")

        # Create VCF with one CNV at chr1:1000-5000
        create_test_vcf(
            input_vcf,
            [{"chrom": "chr1", "start": 1000, "stop": 5000, "svtype": "DEL"}],
            {"chr1": 10000},
        )

        # Create ML BED with interval covering only start position
        # Start position (1000) falls in [500, 1500) with prob 0.45
        # End position (5000) does not overlap any interval
        create_test_ml_bed(
            ml_bed,
            [
                {"chrom": "chr1", "start": 500, "end": 1500, "prob": 0.45},
            ],
        )

        annotate_vcf_with_ml_scores(input_vcf, ml_bed, output_vcf)

        # Verify CNV_PRED is max(0.45, 0.0) = 0.45
        with pysam.VariantFile(output_vcf) as vcf:
            records = list(vcf)
            assert len(records) == 1
            assert "CNV_PRED" in records[0].info
            assert records[0].info["CNV_PRED"] == pytest.approx(0.45, abs=1e-5)

    def test_no_overlap(self, tmp_path):
        """Test that when no endpoints overlap, CNV_PRED is 0.0."""
        input_vcf = os.path.join(tmp_path, "input.vcf.gz")
        ml_bed = os.path.join(tmp_path, "ml_predictions.bed")
        output_vcf = os.path.join(tmp_path, "output.vcf.gz")

        # Create VCF with one CNV at chr1:1000-5000
        create_test_vcf(
            input_vcf,
            [{"chrom": "chr1", "start": 1000, "stop": 5000, "svtype": "DEL"}],
            {"chr1": 10000},
        )

        # Create ML BED with intervals that don't overlap with CNV endpoints
        create_test_ml_bed(
            ml_bed,
            [
                {"chrom": "chr1", "start": 100, "end": 500, "prob": 0.3},
                {"chrom": "chr1", "start": 6000, "end": 7000, "prob": 0.7},
            ],
        )

        annotate_vcf_with_ml_scores(input_vcf, ml_bed, output_vcf)

        # Verify CNV_PRED is max(0.0, 0.0) = 0.0
        with pysam.VariantFile(output_vcf) as vcf:
            records = list(vcf)
            assert len(records) == 1
            assert "CNV_PRED" in records[0].info
            assert records[0].info["CNV_PRED"] == pytest.approx(0.0, abs=1e-5)

    def test_multiple_overlapping_intervals_raises_error(self, tmp_path):
        """Test that multiple overlapping intervals in ML BED raises ValueError."""
        input_vcf = os.path.join(tmp_path, "input.vcf.gz")
        ml_bed = os.path.join(tmp_path, "ml_predictions.bed")
        output_vcf = os.path.join(tmp_path, "output.vcf.gz")

        # Create VCF with one CNV
        create_test_vcf(
            input_vcf,
            [{"chrom": "chr1", "start": 1000, "stop": 5000, "svtype": "DEL"}],
            {"chr1": 10000},
        )

        # Create ML BED with overlapping intervals
        create_test_ml_bed(
            ml_bed,
            [
                {"chrom": "chr1", "start": 500, "end": 1500, "prob": 0.3},
                {"chrom": "chr1", "start": 1000, "end": 2000, "prob": 0.5},  # Overlaps with previous
            ],
        )

        # Should raise ValueError due to overlapping intervals
        with pytest.raises(ValueError, match="Multiple overlapping intervals found in ML BED file"):
            annotate_vcf_with_ml_scores(input_vcf, ml_bed, output_vcf)

    def test_missing_chromosome_warning(self, tmp_path, caplog):
        """Test that missing chromosome in ML BED logs warning and sets CNV_PRED to 0.0."""
        input_vcf = os.path.join(tmp_path, "input.vcf.gz")
        ml_bed = os.path.join(tmp_path, "ml_predictions.bed")
        output_vcf = os.path.join(tmp_path, "output.vcf.gz")

        # Create VCF with CNV on chr2
        create_test_vcf(
            input_vcf,
            [{"chrom": "chr2", "start": 1000, "stop": 5000, "svtype": "DEL"}],
            {"chr2": 10000},
        )

        # Create ML BED with only chr1 data
        create_test_ml_bed(
            ml_bed,
            [
                {"chrom": "chr1", "start": 500, "end": 1500, "prob": 0.3},
            ],
        )

        annotate_vcf_with_ml_scores(input_vcf, ml_bed, output_vcf)

        # Verify warning was logged
        assert "Chromosome chr2 not found in ML predictions" in caplog.text

        # Verify CNV_PRED is 0.0
        with pysam.VariantFile(output_vcf) as vcf:
            records = list(vcf)
            assert len(records) == 1
            assert "CNV_PRED" in records[0].info
            assert records[0].info["CNV_PRED"] == pytest.approx(0.0, abs=1e-5)

    def test_exact_boundary_matching(self, tmp_path):
        """Test that BED intervals follow half-open [start, end) convention."""
        input_vcf = os.path.join(tmp_path, "input.vcf.gz")
        ml_bed = os.path.join(tmp_path, "ml_predictions.bed")
        output_vcf = os.path.join(tmp_path, "output.vcf.gz")

        # Create VCF with CNV where start=1000 and end=2000
        create_test_vcf(
            input_vcf,
            [{"chrom": "chr1", "start": 1000, "stop": 2000, "svtype": "DEL"}],
            {"chr1": 10000},
        )

        # Create ML BED with interval [1000, 2000)
        # Position 1000 should be IN the interval (start is inclusive)
        # Position 2000 should be OUT of the interval (end is exclusive)
        create_test_ml_bed(
            ml_bed,
            [
                {"chrom": "chr1", "start": 1000, "end": 2000, "prob": 0.5},
            ],
        )

        annotate_vcf_with_ml_scores(input_vcf, ml_bed, output_vcf)

        # Position 1000 is in [1000, 2000) -> prob = 0.5
        # Position 2000 is NOT in [1000, 2000) -> prob = 0.0
        # Max should be 0.5
        with pysam.VariantFile(output_vcf) as vcf:
            records = list(vcf)
            assert len(records) == 1
            assert "CNV_PRED" in records[0].info
            assert records[0].info["CNV_PRED"] == pytest.approx(0.5, abs=1e-5)

    def test_multi_chromosome_vcf(self, tmp_path):
        """Test that multi-chromosome VCF is handled correctly."""
        input_vcf = os.path.join(tmp_path, "input.vcf.gz")
        ml_bed = os.path.join(tmp_path, "ml_predictions.bed")
        output_vcf = os.path.join(tmp_path, "output.vcf.gz")

        # Create VCF with CNVs on multiple chromosomes
        create_test_vcf(
            input_vcf,
            [
                {"chrom": "chr1", "start": 1000, "stop": 2000, "svtype": "DEL"},
                {"chrom": "chr2", "start": 5000, "stop": 6000, "svtype": "DUP"},
                {"chrom": "chr3", "start": 3000, "stop": 4000, "svtype": "DEL"},
            ],
            {"chr1": 10000, "chr2": 10000, "chr3": 10000},
        )

        # Create ML BED with predictions for chr1 and chr2 only
        create_test_ml_bed(
            ml_bed,
            [
                {"chrom": "chr1", "start": 500, "end": 1500, "prob": 0.25},
                {"chrom": "chr1", "start": 1500, "end": 2500, "prob": 0.35},
                {"chrom": "chr2", "start": 4500, "end": 5500, "prob": 0.60},
                {"chrom": "chr2", "start": 5500, "end": 6500, "prob": 0.80},
            ],
        )

        annotate_vcf_with_ml_scores(input_vcf, ml_bed, output_vcf)

        # Verify results
        with pysam.VariantFile(output_vcf) as vcf:
            records = list(vcf)
            assert len(records) == 3

            # chr1:1000-2000 -> start in [500,1500) prob=0.25, end in [1500,2500) prob=0.35 -> max=0.35
            assert records[0].info["CNV_PRED"] == pytest.approx(0.35, abs=1e-5)

            # chr2:5000-6000 -> start in [4500,5500) prob=0.60, end in [5500,6500) prob=0.80 -> max=0.80
            assert records[1].info["CNV_PRED"] == pytest.approx(0.80, abs=1e-5)

            # chr3:3000-4000 -> no ML data for chr3 -> max=0.0
            assert records[2].info["CNV_PRED"] == pytest.approx(0.0, abs=1e-5)

    def test_header_contains_cnv_pred(self, tmp_path):
        """Test that output VCF header contains CNV_PRED INFO field."""
        input_vcf = os.path.join(tmp_path, "input.vcf.gz")
        ml_bed = os.path.join(tmp_path, "ml_predictions.bed")
        output_vcf = os.path.join(tmp_path, "output.vcf.gz")

        create_test_vcf(
            input_vcf,
            [{"chrom": "chr1", "start": 1000, "stop": 2000, "svtype": "DEL"}],
            {"chr1": 10000},
        )

        create_test_ml_bed(
            ml_bed,
            [{"chrom": "chr1", "start": 500, "end": 1500, "prob": 0.5}],
        )

        annotate_vcf_with_ml_scores(input_vcf, ml_bed, output_vcf)

        # Verify header
        with pysam.VariantFile(output_vcf) as vcf:
            assert "CNV_PRED" in vcf.header.info
            info_record = vcf.header.info["CNV_PRED"]
            assert info_record.number == 1
            assert info_record.type == "Float"
            assert info_record.description is not None
            assert "ML prediction" in info_record.description

    def test_empty_ml_bed(self, tmp_path):
        """Test that empty ML BED file results in all CNV_PRED values being 0.0."""
        input_vcf = os.path.join(tmp_path, "input.vcf.gz")
        ml_bed = os.path.join(tmp_path, "ml_predictions.bed")
        output_vcf = os.path.join(tmp_path, "output.vcf.gz")

        create_test_vcf(
            input_vcf,
            [{"chrom": "chr1", "start": 1000, "stop": 2000, "svtype": "DEL"}],
            {"chr1": 10000},
        )

        # Create empty BED file (only header)
        with open(ml_bed, "w") as bed_fh:
            bed_fh.write("#chrom\tstart\tstop\tprob\tpred\tinput_file_name\n")

        annotate_vcf_with_ml_scores(input_vcf, ml_bed, output_vcf)

        with pysam.VariantFile(output_vcf) as vcf:
            records = list(vcf)
            assert len(records) == 1
            assert records[0].info["CNV_PRED"] == pytest.approx(0.0, abs=1e-5)
