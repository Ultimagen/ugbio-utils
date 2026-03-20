from pathlib import Path

import pysam
import pytest
from ugbio_cnv.merge_cnv_sv import merge_cnv_sv_vcfs


def make_cnv_record(
    vcf,
    contig,
    pos,
    stop,
    record_id,
    svtype="DEL",
    svlen=None,
    qual=50.0,
    filter_val="PASS",
    **info,
):
    """Create a simple CNV-style record for merge tests."""
    record = vcf.new_record()
    record.contig = contig
    record.pos = pos
    record.stop = stop
    record.id = record_id
    record.alleles = ("N", f"<{svtype}>")
    record.info["SVTYPE"] = svtype
    record.info["SVLEN"] = (svlen if svlen is not None else stop - pos,)
    record.info["CIPOS"] = (-250, 251)
    record.qual = qual
    if filter_val:
        record.filter.add(filter_val)
    record.samples["test_sample"]["GT"] = (0, 1)

    for key, value in info.items():
        record.info[key] = value

    return record


class TestMergeCnvSvVcfs:
    """Tests for merge_cnv_sv_vcfs."""

    @pytest.fixture
    def cnv_vcf_header(self):
        """Create VCF header with required fields."""
        header = pysam.VariantHeader()
        header.add_line("##fileformat=VCFv4.2")
        header.add_line("##contig=<ID=chr1,length=248956422>")
        header.add_line('##INFO=<ID=END,Number=1,Type=Integer,Description="End position">')
        header.add_line('##INFO=<ID=SVTYPE,Number=1,Type=String,Description="Type of SV">')
        header.add_line('##INFO=<ID=SVLEN,Number=.,Type=Integer,Description="SV length">')
        header.add_line('##INFO=<ID=CIPOS,Number=2,Type=Integer,Description="Confidence interval around POS">')
        header.add_line('##INFO=<ID=CNV_SOURCE,Number=.,Type=String,Description="CNV caller source">')
        header.add_line('##FILTER=<ID=PASS,Description="Passed all filters">')
        header.add_line('##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">')
        header.add_sample("test_sample")
        return header

    @pytest.fixture
    def simple_cnv_vcf(self, tmp_path, cnv_vcf_header):
        """Create simple CNV VCF with non-overlapping variants."""
        vcf_path = tmp_path / "test_cnv.vcf.gz"
        with pysam.VariantFile(str(vcf_path), "w", header=cnv_vcf_header) as vcf:
            vcf.write(make_cnv_record(vcf, "chr1", 1000, 2000, "CNV1", "DEL", qual=35.0))
            vcf.write(make_cnv_record(vcf, "chr1", 8000, 9000, "CNV2", "DUP", qual=38.0))
            vcf.write(make_cnv_record(vcf, "chr1", 10000, 15000, "CNV3", "DEL", qual=32.0))
        pysam.tabix_index(str(vcf_path), preset="vcf", force=True)
        return vcf_path

    @pytest.fixture
    def simple_sv_vcf(self, tmp_path, cnv_vcf_header):
        """Create SV VCF with variants at different length and quality thresholds."""
        vcf_path = tmp_path / "test_sv.vcf.gz"
        with pysam.VariantFile(str(vcf_path), "w", header=cnv_vcf_header) as vcf:
            vcf.write(make_cnv_record(vcf, "chr1", 1000, 1800, "SV1", "DEL", qual=700.0))
            vcf.write(make_cnv_record(vcf, "chr1", 3000, 4500, "SV2", "DEL", qual=500.0))
            vcf.write(make_cnv_record(vcf, "chr1", 5000, 7000, "SV3", "DUP", qual=800.0))
            vcf.write(make_cnv_record(vcf, "chr1", 20000000, 26000000, "SV4", "DEL", qual=900.0))
        pysam.tabix_index(str(vcf_path), preset="vcf", force=True)
        return vcf_path

    def test_parameter_filtering(self, tmp_path, simple_cnv_vcf, simple_sv_vcf):
        """Verify min_sv_length, max_sv_length, and min_sv_qual filtering."""
        output_vcf = tmp_path / "merged_output.vcf.gz"

        result_vcf = merge_cnv_sv_vcfs(
            cnv_vcf=str(simple_cnv_vcf),
            sv_vcf=str(simple_sv_vcf),
            output_vcf=str(output_vcf),
            min_sv_length=1000,
            max_sv_length=5000000,
            min_sv_qual=600,
            distance=0,
            pctsize=0.5,
            output_directory=str(tmp_path),
        )

        assert Path(result_vcf).exists()

        with pysam.VariantFile(result_vcf) as vcf:
            records = list(vcf)
            record_ids = [record.id for record in records]

            # CNV records should all be present
            assert "CNV1" in record_ids
            assert "CNV2" in record_ids
            assert "CNV3" in record_ids

            # SV3 passes all filters (qual >= 600, size OK)
            assert "SV3" in record_ids
            # SV1 filtered by overlapping higher-quality CNV or quality
            assert "SV1" not in record_ids
            # SV2 filtered by min_sv_qual (qual=500 < 600)
            assert "SV2" not in record_ids
            # SV4 filtered by max_sv_length (6Mb > 5Mb) and no CNV overlap
            assert "SV4" not in record_ids

            assert len(records) == 4

    def test_large_sv_with_cnv_overlap(self, tmp_path, cnv_vcf_header):
        """Test that large SVs are kept if they overlap with CNV calls."""
        # Create CNV VCF with a variant that will overlap with large SV
        cnv_vcf_path = tmp_path / "cnv_with_large_region.vcf.gz"
        with pysam.VariantFile(str(cnv_vcf_path), "w", header=cnv_vcf_header) as vcf:
            # Write in sorted order (by position)
            # Another non-overlapping CNV
            vcf.write(make_cnv_record(vcf, "chr1", 10000, 15000, "CNV_SMALL", "DUP", qual=38.0))
            # Large CNV (5Mb) that will overlap substantially with the large SV
            # This ensures >50% reciprocal overlap for collapse to occur
            vcf.write(make_cnv_record(vcf, "chr1", 1000000, 6000000, "CNV_LARGE", "DEL", qual=35.0))
        pysam.tabix_index(str(cnv_vcf_path), preset="vcf", force=True)

        # Create SV VCF with a large SV (7Mb) that overlaps with CNV_LARGE
        sv_vcf_path = tmp_path / "sv_with_large.vcf.gz"
        with pysam.VariantFile(str(sv_vcf_path), "w", header=cnv_vcf_header) as vcf:
            # Write in sorted order (by position)
            # Regular SV that should pass filters
            vcf.write(make_cnv_record(vcf, "chr1", 10500, 12500, "SV_SMALL", "DUP", qual=850.0))
            # Large SV (7Mb) that overlaps substantially with CNV_LARGE (1M-6M)
            # Overlap: 1M-6M = 5Mb out of 7Mb = 71% of SV, 100% of CNV
            vcf.write(make_cnv_record(vcf, "chr1", 1000000, 8000000, "SV_LARGE", "DEL", qual=800.0))
        pysam.tabix_index(str(sv_vcf_path), preset="vcf", force=True)

        output_vcf = tmp_path / "merged_large_sv.vcf.gz"

        result_vcf = merge_cnv_sv_vcfs(
            cnv_vcf=str(cnv_vcf_path),
            sv_vcf=str(sv_vcf_path),
            output_vcf=str(output_vcf),
            min_sv_length=1000,
            max_sv_length=5000000,  # 5Mb limit
            min_sv_qual=0,
            pctsize=0.5,
            distance=0,
            output_directory=str(tmp_path),
        )

        assert Path(result_vcf).exists()

        with pysam.VariantFile(result_vcf) as vcf:
            records = list(vcf)
            record_ids = [record.id for record in records]

            # Large SV should be kept because it overlaps with CNV_LARGE
            assert "SV_LARGE" in record_ids, "Large SV with CNV overlap should be kept"

            # Regular SV should be kept (within size limits)
            assert "SV_SMALL" in record_ids

            # Check that the large SV has CNV_SOURCE indicating merge
            large_sv_record = next(r for r in records if r.id == "SV_LARGE")
            cnv_sources = large_sv_record.info.get("CNV_SOURCE", [])
            # Should have both gridss (SV source) and CNV source
            assert len(cnv_sources) > 1, "Large SV should have multiple CNV_SOURCE entries"
            assert any(src != "gridss" for src in cnv_sources), "Large SV should have non-gridss CNV_SOURCE"

    def test_large_sv_without_cnv_overlap(self, tmp_path, cnv_vcf_header):
        """Test that large SVs without CNV overlap are filtered out."""
        # Create CNV VCF with variants that DON'T overlap with large SV
        cnv_vcf_path = tmp_path / "cnv_no_overlap.vcf.gz"
        with pysam.VariantFile(str(cnv_vcf_path), "w", header=cnv_vcf_header) as vcf:
            vcf.write(make_cnv_record(vcf, "chr1", 1000, 2000, "CNV1", "DEL", qual=35.0))
            vcf.write(make_cnv_record(vcf, "chr1", 10000, 15000, "CNV2", "DUP", qual=38.0))
        pysam.tabix_index(str(cnv_vcf_path), preset="vcf", force=True)

        # Create SV VCF with a large SV (7Mb) that does NOT overlap with any CNV
        sv_vcf_path = tmp_path / "sv_no_overlap.vcf.gz"
        with pysam.VariantFile(str(sv_vcf_path), "w", header=cnv_vcf_header) as vcf:
            # Write in sorted order (by position)
            # Regular SV that should pass filters
            vcf.write(make_cnv_record(vcf, "chr1", 5000, 7000, "SV_REGULAR", "DUP", qual=800.0))
            # Large SV (7Mb) at a location with no CNV overlap
            vcf.write(make_cnv_record(vcf, "chr1", 20000000, 27000000, "SV_LARGE_NO_OVERLAP", "DEL", qual=900.0))
        pysam.tabix_index(str(sv_vcf_path), preset="vcf", force=True)

        output_vcf = tmp_path / "merged_no_overlap.vcf.gz"

        result_vcf = merge_cnv_sv_vcfs(
            cnv_vcf=str(cnv_vcf_path),
            sv_vcf=str(sv_vcf_path),
            output_vcf=str(output_vcf),
            min_sv_length=1000,
            max_sv_length=5000000,  # 5Mb limit
            min_sv_qual=0,
            pctsize=0.5,
            distance=0,
            output_directory=str(tmp_path),
        )

        assert Path(result_vcf).exists()

        with pysam.VariantFile(result_vcf) as vcf:
            records = list(vcf)
            record_ids = [record.id for record in records]

            # Large SV without CNV overlap should be filtered out
            assert "SV_LARGE_NO_OVERLAP" not in record_ids, "Large SV without CNV overlap should be filtered"

            # Regular SV should be kept (within size limits)
            assert "SV_REGULAR" in record_ids

            # CNVs should still be present
            assert "CNV1" in record_ids
            assert "CNV2" in record_ids
