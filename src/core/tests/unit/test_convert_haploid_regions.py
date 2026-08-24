import pysam
import pytest
from ugbio_core.convert_haploid_regions import (
    _B37_NON_PAR,
    _HG38_NON_PAR,
    _convert_to_haploid,
    _in_regions,
    convert_haploid_regions,
)


class TestInRegions:
    def test_inside_nonpar(self):
        assert _in_regions("chrX", 5000000, _HG38_NON_PAR) is True

    def test_par1_not_in_nonpar(self):
        # PAR1 is chrX:10001-2781479 — positions inside PAR1 should NOT be in non-PAR
        assert _in_regions("chrX", 50000, _HG38_NON_PAR) is False
        assert _in_regions("chrX", 2000000, _HG38_NON_PAR) is False

    def test_par1_region_not_in_nonpar(self):
        # Positions in PAR1 (10001-2781479) are NOT in non-PAR list
        # The first non-PAR entry is (chrX, 1, 10001), second is (chrX, 2781479, ...)
        # pos=10001 is in first entry (1 < 10001 <= 10001), pos=10002 is NOT in any entry
        assert _in_regions("chrX", 10001, _HG38_NON_PAR) is True
        assert _in_regions("chrX", 10002, _HG38_NON_PAR) is False  # in PAR1, not non-PAR

    def test_autosome_not_in_nonpar(self):
        assert _in_regions("chr1", 1000000, _HG38_NON_PAR) is False

    def test_b37_regions(self):
        assert _in_regions("X", 5000000, _B37_NON_PAR) is True
        assert _in_regions("chrX", 5000000, _B37_NON_PAR) is False  # wrong naming


class TestConvertToHaploid:
    def _make_vcf_with_variant(self, tmp_path, gt, pls, chrom="chrX", pos=5000000):
        vcf_path = str(tmp_path / "test.vcf")
        header = pysam.VariantHeader()
        header.add_sample("SAMPLE")
        header.add_line(f"##contig=<ID={chrom}>")
        header.add_line('##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">')
        header.add_line('##FORMAT=<ID=GQ,Number=1,Type=Integer,Description="Genotype Quality">')
        header.add_line('##FORMAT=<ID=PL,Number=G,Type=Integer,Description="Phred-scaled likelihoods">')
        vcf = pysam.VariantFile(vcf_path, "w", header=header)
        rec = vcf.new_record(contig=chrom, start=pos - 1, stop=pos, alleles=("A", "G"))
        rec.samples["SAMPLE"]["GT"] = gt
        rec.samples["SAMPLE"]["GQ"] = min(pls) if min(pls) > 0 else sorted(pls)[1] if len(pls) > 1 else 0
        rec.samples["SAMPLE"]["PL"] = pls
        vcf.write(rec)
        vcf.close()
        return vcf_path

    def test_diploid_homalt_to_haploid(self, tmp_path):
        vcf_path = self._make_vcf_with_variant(tmp_path, (1, 1), [63, 42, 0])
        reader = pysam.VariantFile(vcf_path)
        variant = next(reader)
        result = _convert_to_haploid(variant)
        # Should convert to haploid GT=1, PL should have 2 values
        assert len(result.samples[0]["PL"]) == 2
        assert result.samples[0]["GT"] == (1,)

    def test_already_haploid_unchanged(self, tmp_path):
        vcf_path = self._make_vcf_with_variant(tmp_path, (1,), [30, 0])
        reader = pysam.VariantFile(vcf_path)
        variant = next(reader)
        result = _convert_to_haploid(variant)
        assert result.samples[0]["PL"] == (30, 0)
        assert result.samples[0]["GT"] == (1,)


class TestConvertHaploidRegions:
    def _make_test_vcf(self, tmp_path, records):
        vcf_path = str(tmp_path / "input.vcf.gz")
        header = pysam.VariantHeader()
        header.add_sample("SAMPLE")
        for chrom in ("chrX", "chrY", "chr1"):
            header.add_line(f"##contig=<ID={chrom}>")
        header.add_line('##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">')
        header.add_line('##FORMAT=<ID=GQ,Number=1,Type=Integer,Description="Genotype Quality">')
        header.add_line('##FORMAT=<ID=PL,Number=G,Type=Integer,Description="Phred-scaled likelihoods">')
        vcf = pysam.VariantFile(vcf_path, "wz", header=header)
        for chrom, pos, gt, pls in records:
            rec = vcf.new_record(contig=chrom, start=pos - 1, stop=pos, alleles=("A", "G"))
            rec.samples["SAMPLE"]["GT"] = gt
            rec.samples["SAMPLE"]["GQ"] = 30
            rec.samples["SAMPLE"]["PL"] = pls
            vcf.write(rec)
        vcf.close()
        pysam.tabix_index(vcf_path, preset="vcf", force=True)
        return vcf_path

    def test_nonpar_converted_par_preserved(self, tmp_path):
        records = [
            ("chr1", 1000000, (0, 1), [30, 0, 40]),  # autosome — should stay diploid
            ("chrX", 15000, (1, 1), [60, 30, 0]),  # PAR1 — should stay diploid
            ("chrX", 5000000, (1, 1), [60, 30, 0]),  # non-PAR — should become haploid
        ]
        input_vcf = self._make_test_vcf(tmp_path, records)
        output_vcf = str(tmp_path / "output.vcf.gz")

        convert_haploid_regions(input_vcf, output_vcf, "hg38_non_par")

        reader = pysam.VariantFile(output_vcf)
        results = list(reader)

        # chr1: diploid preserved
        assert len(results[0].samples[0]["PL"]) == 3
        # chrX PAR: diploid preserved
        assert len(results[1].samples[0]["PL"]) == 3
        # chrX non-PAR: haploid
        assert len(results[2].samples[0]["PL"]) == 2

    def test_auto_detect_hg38(self, tmp_path):
        records = [("chrX", 5000000, (1, 1), [60, 30, 0])]
        input_vcf = self._make_test_vcf(tmp_path, records)
        output_vcf = str(tmp_path / "output.vcf.gz")

        convert_haploid_regions(input_vcf, output_vcf, "auto")

        reader = pysam.VariantFile(output_vcf)
        result = next(reader)
        assert len(result.samples[0]["PL"]) == 2

    def test_multi_sample_raises(self, tmp_path):
        vcf_path = str(tmp_path / "multi.vcf.gz")
        header = pysam.VariantHeader()
        header.add_sample("SAMPLE1")
        header.add_sample("SAMPLE2")
        header.add_line("##contig=<ID=chrX>")
        header.add_line('##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">')
        header.add_line('##FORMAT=<ID=PL,Number=G,Type=Integer,Description="Phred-scaled likelihoods">')
        vcf = pysam.VariantFile(vcf_path, "wz", header=header)
        vcf.close()

        with pytest.raises(ValueError, match="Expected single-sample VCF"):
            convert_haploid_regions(vcf_path, str(tmp_path / "out.vcf.gz"))
