import os
import tempfile
from pathlib import Path

import pysam
import pytest
import ugbio_core.vcfbed.pysam_utils as pysam_utils


def _generate_indel_tests():
    vcfh = pysam.VariantHeader()
    vcfh.add_sample("ahstram")
    vcfh.add_meta("FILTER", items=[("ID", "RF"), ("Description", "Variant failed filter due to low RF")])
    vcfh.add_meta("contig", items=[("ID", 1)])
    vcfh.add_meta("FORMAT", items=[("ID", "GT"), ("Number", 1), ("Type", "String"), ("Description", "Genotype")])
    tmpfilename = tempfile.mktemp(suffix="vcf")
    vcf = pysam.VariantFile(tmpfilename, "w", header=vcfh)

    records = []
    r = vcf.new_record(contig=str(1), start=999, stop=1000, alleles=("A", "T"), filter="RF")
    records.append(r)
    r = vcf.new_record(contig=str(1), start=999, stop=1000, alleles=("A", "AT"), filter="RF")
    records.append(r)
    r = vcf.new_record(contig=str(1), start=999, stop=1000, alleles=("AT", "A"), filter="RF")
    records.append(r)
    r = vcf.new_record(contig=str(1), start=999, stop=1000, alleles=("AT", "A", "AG", "ATC"), filter="RF")
    records.append(r)
    r = vcf.new_record(contig=str(1), start=999, stop=1000, alleles=("AT", "A", "<NON_REF>"), filter="RF")
    records.append(r)

    os.unlink(tmpfilename)
    return records


class TestPysamUtils:
    def __init__(self):
        self.vcf = pysam.VariantFile(Path(__file__).parent.parent.parent / "resources" / " single_sample_example.vcf")
        self.variant = next(self.vcf)

    test_inputs = _generate_indel_tests()

    test_alleles = ("AT", "A", "<NON_REF>")

    @pytest.mark.parametrize("inp,expected", zip(test_alleles, [False, False, True]))
    def test_is_symbolic(self, inp, expected):
        assert pysam_utils.is_symbolic(inp) == expected

    @pytest.mark.parametrize(
        "inp,expected",
        zip(
            test_inputs,
            [[False, False], [False, True], [False, True], [False, True, False, True], [False, True, False]],
        ),
    )
    def test_is_indel(self, inp, expected):
        assert pysam_utils.is_indel(inp) == expected

    @pytest.mark.parametrize(
        "inp,expected",
        zip(
            test_inputs,
            [[False, False], [False, False], [False, True], [False, True, False, False], [False, True, False]],
        ),
    )
    def test_is_deletion(self, inp, expected):
        assert pysam_utils.is_deletion(inp) == expected

    @pytest.mark.parametrize(
        "inp,expected",
        zip(
            test_inputs,
            [[False, False], [False, True], [False, False], [False, False, False, True], [False, False, False]],
        ),
    )
    def test_is_insertion(self, inp, expected):
        assert pysam_utils.is_insertion(inp) == expected

    @pytest.mark.parametrize("inp,expected", zip(test_inputs, [[0, 0], [0, 1], [0, 1], [0, 1, 0, 1], [0, 1, 0]]))
    def test_indel_length(self, inp, expected):
        assert pysam_utils.indel_length(inp) == expected

    def test_get_alleles_str(self):
        assert "T,TG,<NON_REF>" == pysam_utils.get_alleles_str(self.variant)

    def test_get_filtered_alleles_list(self):
        assert ["T", "TG"] == pysam_utils.get_filtered_alleles_list(self.variant)
        assert ["T"] == pysam_utils.get_filtered_alleles_list(self.variant, filter_list=["TG", "<NON_REF>"])

        # '*' as minor allele is filtered out automatically
        self.variant.alts = ("TG", "*")
        assert ["T", "TG"] == pysam_utils.get_filtered_alleles_list(self.variant)

        # '*' as major allele is not filtered out
        self.variant.alts = ("*", "TG")
        assert ["T", "*", "TG"] == pysam_utils.get_filtered_alleles_list(self.variant)

        # '*' as major allele can be filtered out explicitly
        self.variant.alts = ("*", "TG")
        assert ["T", "TG"] == pysam_utils.get_filtered_alleles_list(self.variant, filter_list=["*"])

    def test_get_filtered_alleles_str(self):
        assert "T,TG" == pysam_utils.get_filtered_alleles_str(self.variant)
        assert "T" == pysam_utils.get_filtered_alleles_str(self.variant, filter_list=["TG", "<NON_REF>"])

        # '*' as minor allele is filtered out automatically
        self.variant.alts = ("TG", "*")
        assert "T,TG" == pysam_utils.get_filtered_alleles_str(self.variant)

        # '*' as major allele is not filtered out
        self.variant.alts = ("*", "TG")
        assert "T,*,TG" == pysam_utils.get_filtered_alleles_str(self.variant)

        # '*' as major allele can be filtered out explicitly
        self.variant.alts = ("*", "TG")
        assert "T,TG" == pysam_utils.get_filtered_alleles_str(self.variant, filter_list=["*"])

    def test_get_genotype(self):
        assert "T/T" == pysam_utils.get_genotype(self.variant.samples[0])

    def test_get_genotype_indices(self):
        assert "0/0" == pysam_utils.get_genotype_indices(self.variant.samples[0])

    def test_has_candidate_alternatives(self):
        assert ("T", "TG", "<NON_REF>") == self.variant.alleles
        assert pysam_utils.has_candidate_alternatives(self.variant)

        # scroll to first variant without alternative
        variant = None
        for i in range(7):
            variant = next(self.vcf)
        assert ("C", "<NON_REF>") == variant.alleles
        assert not pysam_utils.has_candidate_alternatives(variant)

    def test_is_snp(self):
        assert pysam_utils.is_snp(["A", "T"])
        assert not pysam_utils.is_snp(["A", "AG"])
        assert pysam_utils.is_snp(["A", "T", "C"])
        assert not pysam_utils.is_snp(["A", "T", "AG"])
        assert not pysam_utils.is_snp(["AT", "GC"])
