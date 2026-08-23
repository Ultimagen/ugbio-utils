from pathlib import Path

import pysam
import pytest
from ugbio_filtering.regenotype_mitochondria import regenotype_mitochondria

# A callset shaped like the caller's own output: INFO/AF derived from GT (hence a constant 0.5),
# with the real evidence in FORMAT/VAF and FORMAT/AD.  INFO/END must stay declared even though no
# record here uses it: htslib registers END in the header dictionary when it parses a record with a
# symbolic <*> allele, and without the declaration that happens after the output header has been
# built, which makes the write fail.  Every real gVCF declares it.
GENOTYPE_VCF_HEADER = """##fileformat=VCFv4.2
##contig=<ID=chrM,length=800>
##contig=<ID=chr1,length=800>
##FILTER=<ID=RefCall,Description="Genotyping model thinks this site is reference.">
##INFO=<ID=END,Number=1,Type=Integer,Description="End position (for use with symbolic alleles)">
##INFO=<ID=AF,Number=A,Type=Float,Description="Allele Frequency">
##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">
##FORMAT=<ID=VAF,Number=A,Type=Float,Description="Variant allele fractions.">
##FORMAT=<ID=AD,Number=R,Type=Integer,Description="Read depth for each allele">
##FORMAT=<ID=PL,Number=G,Type=Integer,Description="Phred-scaled genotype likelihoods">
#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tsample1
"""

# chrM 51   homoplasmic, the case the whole correction exists for
# chrM 101  real heteroplasmy: the genotype is right, only INFO/AF is wrong
# chrM 151  gVCF-shaped, with the symbolic allele occupying a slot in the Number=A vector
# chrM 201  the model declined to call
# chrM 251  already homozygous
# chrM 351  no VAF at all
# chr1 301  homoplasmic but on another contig; last, so the contig blocks stay indexable
GENOTYPE_VCF_RECORDS = """chrM\t51\t.\tA\tG\t60\tPASS\tAF=0.5\tGT:VAF:AD:PL\t0/1:0.992:9,1183:0,0,33
chrM\t101\t.\tA\tG\t55\tPASS\tAF=0.5\tGT:VAF:AD:PL\t0/1:0.3:70,30:30,0,90
chrM\t151\t.\tA\tG,<*>\t50\tPASS\tAF=0.5,0\tGT:VAF:AD:PL\t0/1:0.99,0:5,900,0:0,0,20,99,99,99
chrM\t201\t.\tA\tG\t2\tRefCall\tAF=0.5\tGT:VAF:AD:PL\t./.:0.95:4,80:0,0,10
chrM\t251\t.\tA\tG\t70\tPASS\tAF=0.5\tGT:VAF:AD:PL\t1/1:0.99:2,300:99,50,0
chrM\t351\t.\tA\tG\t60\tPASS\tAF=0.5\tGT:VAF:AD:PL\t0/1:.:5,500:0,0,40
chr1\t301\t.\tA\tG\t60\tPASS\tAF=0.5\tGT:VAF:AD:PL\t0/1:0.99:3,400:0,0,40
"""


def _write(path: Path, content: str) -> str:
    path.write_text(content)
    return str(path)


@pytest.fixture
def genotype_vcf(tmp_path: Path):
    return _write(tmp_path / "genotypes.vcf", GENOTYPE_VCF_HEADER + GENOTYPE_VCF_RECORDS)


def _regenotype(input_vcf: str, tmp_path: Path, *, homoplasmy_vaf: float = 0.85, name: str = "out.vcf.gz"):
    """Run regenotyping and return the output records keyed by position."""
    output_vcf = str(tmp_path / name)
    regenotype_mitochondria(input_vcf, output_vcf, contig="chrM", homoplasmy_vaf=homoplasmy_vaf)
    with pysam.VariantFile(output_vcf) as vcf:
        return {record.pos: record for record in vcf.fetch()}, output_vcf


def test_homoplasmic_call_becomes_homozygous(genotype_vcf, tmp_path: Path):
    """The defect this exists for: VAF 0.992 on 1183 alt reads, genotyped 0/1 by a diploid model."""
    records, output_vcf = _regenotype(genotype_vcf, tmp_path)
    with pysam.VariantFile(output_vcf) as vcf:
        assert "OGT" in vcf.header.formats
        assert "MT_REGT" in vcf.header.info

    record = records[51]
    assert record.samples[0]["GT"] == (1, 1)
    assert record.samples[0]["OGT"] == "0/1", "the model's own genotype stays recoverable"
    assert record.info.get("MT_REGT")
    assert record.info["AF"] == pytest.approx((0.992,), abs=1e-4)
    assert record.samples[0]["PL"] == (0, 0, 33), "PL is left as emitted, even though it disagrees with GT"
    assert record.qual == pytest.approx(60)


def test_heteroplasmy_keeps_its_genotype_but_gets_a_real_af(genotype_vcf, tmp_path: Path):
    """Below the threshold the genotype is already right; INFO/AF is wrong regardless."""
    records, _ = _regenotype(genotype_vcf, tmp_path)
    record = records[101]
    assert record.samples[0]["GT"] == (0, 1)
    assert record.samples[0].get("OGT") is None, "no OGT is written where nothing changed"
    assert not record.info.get("MT_REGT")
    assert record.info["AF"] == pytest.approx((0.3,), abs=1e-4), "no longer the GT-derived constant"


def test_symbolic_allele_is_never_genotyped(genotype_vcf, tmp_path: Path):
    """A gVCF <*> allele occupies a Number=A slot but must not win the argmax."""
    records, _ = _regenotype(genotype_vcf, tmp_path)
    record = records[151]
    assert record.samples[0]["GT"] == (1, 1)
    assert record.info["AF"] == pytest.approx((0.99, 0.0), abs=1e-4)


def test_uncalled_genotype_is_left_uncalled(genotype_vcf, tmp_path: Path):
    """A ./. record means the model declined; do not manufacture a call from the reads."""
    records, _ = _regenotype(genotype_vcf, tmp_path)
    record = records[201]
    assert record.samples[0]["GT"] == (None, None)
    assert not record.info.get("MT_REGT")
    assert "RefCall" in record.filter.keys(), "the FILTER is untouched by this change"


def test_already_homozygous_record_is_not_reflagged(genotype_vcf, tmp_path: Path):
    records, _ = _regenotype(genotype_vcf, tmp_path)
    record = records[251]
    assert record.samples[0]["GT"] == (1, 1)
    assert record.samples[0].get("OGT") is None, "no OGT is written where nothing changed"
    assert not record.info.get("MT_REGT")
    assert record.info["AF"] == pytest.approx((0.99,), abs=1e-4)


def test_only_the_named_contig_is_touched(genotype_vcf, tmp_path: Path):
    """The contig guard is the only thing standing between this and a genome-wide GT rewrite."""
    records, _ = _regenotype(genotype_vcf, tmp_path)
    record = records[301]
    assert record.contig == "chr1"
    assert record.samples[0]["GT"] == (0, 1)
    assert not record.info.get("MT_REGT")
    assert record.info["AF"] == pytest.approx((0.5,), abs=1e-4), "INFO/AF is not rewritten off-contig"


def test_record_without_vaf_is_untouched(genotype_vcf, tmp_path: Path):
    """gVCF reference blocks carry no VAF; there is nothing to regenotype from."""
    records, _ = _regenotype(genotype_vcf, tmp_path)
    record = records[351]
    assert record.samples[0]["GT"] == (0, 1)
    assert not record.info.get("MT_REGT")
    assert record.info["AF"] == pytest.approx((0.5,), abs=1e-4)


@pytest.mark.parametrize(
    ("homoplasmy_vaf", "expected_gt"),
    [(0.75, (1, 1)), (0.85, (1, 1)), (0.95, (1, 1)), (0.99, (1, 1)), (0.995, (0, 1)), (1.0, (0, 1))],
)
def test_threshold_is_inclusive_and_takes_effect(genotype_vcf, tmp_path: Path, homoplasmy_vaf, expected_gt):
    """chrM:51 sits at VAF 0.992, so it flips for every threshold at or below that and no higher one."""
    records, _ = _regenotype(genotype_vcf, tmp_path, homoplasmy_vaf=homoplasmy_vaf, name=f"t{homoplasmy_vaf}.vcf.gz")
    assert records[51].samples[0]["GT"] == expected_gt


def test_no_record_is_dropped_by_regenotyping(genotype_vcf, tmp_path: Path):
    records, _ = _regenotype(genotype_vcf, tmp_path)
    assert len(records) == len(GENOTYPE_VCF_RECORDS.strip().splitlines())


def test_regenotyping_rejects_a_multi_sample_vcf(tmp_path: Path):
    """INFO/AF is per-record, so a per-sample VAF cannot be written back for more than one sample."""
    multi = GENOTYPE_VCF_HEADER.replace("\tsample1\n", "\tsample1\tsample2\n") + (
        "chrM\t51\t.\tA\tG\t60\tPASS\tAF=0.5\tGT:VAF:AD:PL\t0/1:0.99:9,1183:0,0,33\t0/1:0.1:900,100:0,0,33\n"
    )
    input_vcf = _write(tmp_path / "multi.vcf", multi)
    with pytest.raises(ValueError, match="single-sample"):
        regenotype_mitochondria(input_vcf, str(tmp_path / "out.vcf.gz"), contig="chrM")
