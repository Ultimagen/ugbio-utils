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

            assert "CNV1" in record_ids
            assert "CNV2" in record_ids
            assert "CNV3" in record_ids

            assert "SV3" in record_ids
            assert "SV1" not in record_ids
            assert "SV2" not in record_ids
            assert "SV4" not in record_ids

            assert len(records) == 4
