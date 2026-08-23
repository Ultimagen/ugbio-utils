from pathlib import Path

import pysam
import pytest
from ugbio_filtering.postprocess_mitochondria import (
    info_id_from_bed,
    match_allele,
    postprocess_mitochondria,
)

# chrM NUMT-homology intervals: the control region and one interior block.
CHRM_BED = """##INFO=<ID=NUMT,Number=0,Type=Flag,Description="Overlaps a chrM region with a NUMT paralog">
chrM\t0\t200
chrM\t500\t600
"""

# The nuclear side of the pair; SA targets are tested against these.
NUCLEAR_BED = "chr1\t629000\t635000\n"

REFERENCE = ">chrM\n" + ("ACGTACGTAC" * 80) + "\n"

VCF_HEADER = """##fileformat=VCFv4.2
##contig=<ID=chrM,length=800>
##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">
#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tsample1
"""

# POS 51 and 501 are inside a NUMT interval, POS 301 is not.
VCF_RECORDS = """chrM\t51\t.\tA\tG\t.\tPASS\t.\tGT\t0/1
chrM\t301\t.\tA\tG\t.\tPASS\t.\tGT\t0/1
chrM\t501\t.\tA\tG\t.\tPASS\t.\tGT\t0/1
"""

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
def numt_beds(tmp_path: Path):
    return _write(tmp_path / "numt.chrM.bed", CHRM_BED), _write(tmp_path / "numt.nuclear.bed", NUCLEAR_BED)


@pytest.fixture
def reference(tmp_path: Path):
    fasta = _write(tmp_path / "chrM.fa", REFERENCE)
    pysam.faidx(fasta)
    return fasta


@pytest.fixture
def example_vcf(tmp_path: Path):
    return _write(tmp_path / "input.vcf", VCF_HEADER + VCF_RECORDS)


@pytest.fixture
def genotype_vcf(tmp_path: Path):
    return _write(tmp_path / "genotypes.vcf", GENOTYPE_VCF_HEADER + GENOTYPE_VCF_RECORDS)


def _regenotype(input_vcf: str, tmp_path: Path, *, homoplasmy_vaf: float = 0.85, name: str = "out.vcf.gz"):
    """Run regenotyping alone and return the output records keyed by position."""
    output_vcf = str(tmp_path / name)
    postprocess_mitochondria(input_vcf, output_vcf, regenotype_contig="chrM", homoplasmy_vaf=homoplasmy_vaf)
    with pysam.VariantFile(output_vcf) as vcf:
        return {record.pos: record for record in vcf.fetch()}, output_vcf


def _make_bam(  # noqa: PLR0913
    path: Path,
    reference: str,
    *,
    pos: int,
    alt_reads: int,
    ref_reads: int,
    alt_sa: int,
    ref_sa: int,
    sa_contig: str = "chr1",
    sa_pos: int = 630000,
):
    """Build a BAM covering one site, with SA tags on a controlled subset of alt and ref reads."""
    with pysam.FastaFile(reference) as fasta:
        ref_seq = fasta.fetch("chrM")
    read_length = 100
    start = pos - 1 - read_length // 2  # centre the read on the site
    offset = pos - 1 - start

    header = {"HD": {"VN": "1.6"}, "SQ": [{"SN": "chrM", "LN": len(ref_seq)}, {"SN": sa_contig, "LN": 250000000}]}
    with pysam.AlignmentFile(str(path), "wb", header=header) as bam:
        for index in range(alt_reads + ref_reads):
            is_alt = index < alt_reads
            sequence = list(ref_seq[start : start + read_length])
            sequence[offset] = "G" if is_alt else ref_seq[pos - 1]
            read = pysam.AlignedSegment()
            read.query_name = f"{'alt' if is_alt else 'ref'}_{index}"
            read.query_sequence = "".join(sequence)
            read.query_qualities = pysam.qualitystring_to_array("I" * read_length)
            read.flag = 0
            read.reference_id = 0
            read.reference_start = start
            read.mapping_quality = 60
            read.cigartuples = [(0, read_length)]
            n_tagged = alt_sa if is_alt else ref_sa
            position_in_class = index if is_alt else index - alt_reads
            if position_in_class < n_tagged:
                read.set_tag("SA", f"{sa_contig},{sa_pos},+,50S50M,60,0;")
            bam.write(read)
    pysam.index(str(path))
    return str(path)


def test_positional_flag_only(example_vcf, numt_beds, reference, tmp_path: Path):
    """With no SA evidence, in-interval records get INFO/NUMT but no FILTER."""
    numt_bed, nuclear_bed = numt_beds
    bam = _make_bam(tmp_path / "clean.bam", reference, pos=51, alt_reads=20, ref_reads=20, alt_sa=0, ref_sa=0)
    output_vcf = str(tmp_path / "out.vcf.gz")
    postprocess_mitochondria(example_vcf, output_vcf, numt_bed, nuclear_bed, [bam], reference)

    with pysam.VariantFile(output_vcf) as vcf:
        assert "NUMT" in vcf.header.info
        assert "NUMT" in vcf.header.filters
        records = list(vcf.fetch())
    assert len(records) == 3, "no record may be dropped"
    assert records[0].info.get("NUMT") and records[2].info.get("NUMT")
    assert not records[1].info.get("NUMT"), "chrM:301 is outside every interval"
    for record in records:
        assert "NUMT" not in record.filter.keys()


def test_alt_enriched_record_is_filtered(example_vcf, numt_beds, reference, tmp_path: Path):
    """Alt reads carrying NUMT SA tags well above the local ref rate, at a moderate VAF, are flagged."""
    numt_bed, nuclear_bed = numt_beds
    bam = _make_bam(tmp_path / "enriched.bam", reference, pos=51, alt_reads=20, ref_reads=20, alt_sa=10, ref_sa=0)
    output_vcf = str(tmp_path / "out.vcf.gz")
    postprocess_mitochondria(example_vcf, output_vcf, numt_bed, nuclear_bed, [bam], reference)

    with pysam.VariantFile(output_vcf) as vcf:
        records = {record.pos: record for record in vcf.fetch()}
    assert "NUMT" in records[51].filter.keys()
    assert "NUMT" not in records[301].filter.keys(), "out-of-interval records are never touched"
    assert "NUMT" not in records[501].filter.keys(), "no reads cover this site"


def test_vaf_ceiling_protects_homoplasmic_calls(example_vcf, numt_beds, reference, tmp_path: Path):
    """Term 3: a near-homoplasmic call is never flagged, however NUMT-enriched its alt reads are."""
    numt_bed, nuclear_bed = numt_beds
    bam = _make_bam(tmp_path / "homoplasmic.bam", reference, pos=51, alt_reads=40, ref_reads=1, alt_sa=40, ref_sa=0)
    output_vcf = str(tmp_path / "out.vcf.gz")
    postprocess_mitochondria(example_vcf, output_vcf, numt_bed, nuclear_bed, [bam], reference)

    with pysam.VariantFile(output_vcf) as vcf:
        records = {record.pos: record for record in vcf.fetch()}
    assert records[51].info.get("NUMT"), "the positional flag still applies"
    assert "NUMT" not in records[51].filter.keys()


def test_local_ref_rate_is_the_background(example_vcf, numt_beds, reference, tmp_path: Path):
    """Term 2: uniformly high NUMT SA traffic (the D-loop case) must not flag anything."""
    numt_bed, nuclear_bed = numt_beds
    bam = _make_bam(tmp_path / "uniform.bam", reference, pos=51, alt_reads=20, ref_reads=20, alt_sa=5, ref_sa=5)
    output_vcf = str(tmp_path / "out.vcf.gz")
    postprocess_mitochondria(example_vcf, output_vcf, numt_bed, nuclear_bed, [bam], reference)

    with pysam.VariantFile(output_vcf) as vcf:
        records = {record.pos: record for record in vcf.fetch()}
    assert "NUMT" not in records[51].filter.keys()


def test_sa_to_a_non_numt_locus_is_ignored(example_vcf, numt_beds, reference, tmp_path: Path):
    """A supplementary alignment outside the nuclear NUMT bed carries no weight."""
    numt_bed, nuclear_bed = numt_beds
    bam = _make_bam(
        tmp_path / "elsewhere.bam",
        reference,
        pos=51,
        alt_reads=20,
        ref_reads=20,
        alt_sa=20,
        ref_sa=0,
        sa_pos=90000000,
    )
    output_vcf = str(tmp_path / "out.vcf.gz")
    postprocess_mitochondria(example_vcf, output_vcf, numt_bed, nuclear_bed, [bam], reference)

    with pysam.VariantFile(output_vcf) as vcf:
        records = {record.pos: record for record in vcf.fetch()}
    assert "NUMT" not in records[51].filter.keys()


def test_tag_name_comes_from_the_bed_header(tmp_path: Path):
    bed = _write(tmp_path / "custom.bed", CHRM_BED.replace("ID=NUMT", "ID=MYTAG"))
    tag, info_line = info_id_from_bed(bed)
    assert tag == "MYTAG"
    assert info_line.startswith("##INFO=<ID=MYTAG")


def test_bed_without_info_header_is_rejected(tmp_path: Path):
    bed = _write(tmp_path / "plain.bed", "chrM\t0\t200\n")
    with pytest.raises(ValueError, match="must be a VCF ##INFO"):
        info_id_from_bed(bed)


@pytest.mark.parametrize(
    ("ref", "alt", "expected"),
    [
        ("A", "G", ("G", "snp")),
        ("C", "CTA", ("+2TA", "indel")),
        ("CTA", "C", ("-2TA", "indel")),
        ("AT", "GC", (None, "complex")),
    ],
)
def test_match_allele(ref, alt, expected):
    assert match_allele(ref, alt) == expected


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


def test_both_corrections_apply_in_one_pass(genotype_vcf, numt_beds, reference, tmp_path: Path):
    """NUMT reads the CRAM and regenotyping reads FORMAT/VAF, so neither can see the other's edits."""
    numt_bed, nuclear_bed = numt_beds
    bam = _make_bam(tmp_path / "both.bam", reference, pos=51, alt_reads=20, ref_reads=20, alt_sa=10, ref_sa=0)
    output_vcf = str(tmp_path / "both.vcf.gz")
    postprocess_mitochondria(
        genotype_vcf,
        output_vcf,
        numt_bed,
        nuclear_bed,
        [bam],
        reference,
        regenotype_contig="chrM",
    )
    with pysam.VariantFile(output_vcf) as vcf:
        records = {record.pos: record for record in vcf.fetch()}

    record = records[51]
    assert record.info.get("NUMT"), "term 1 still fires"
    assert "NUMT" in record.filter.keys(), "the read-derived VAF of 0.5 is below the ceiling, so it is flagged"
    assert record.samples[0]["GT"] == (1, 1), "and the genotype is still corrected from FORMAT/VAF 0.992"
    assert record.info.get("MT_REGT")


def test_nothing_to_do_is_an_error(genotype_vcf, tmp_path: Path):
    with pytest.raises(ValueError, match="nothing to do"):
        postprocess_mitochondria(genotype_vcf, str(tmp_path / "out.vcf.gz"))


def test_regenotyping_rejects_a_multi_sample_vcf(tmp_path: Path):
    """INFO/AF is per-record, so a per-sample VAF cannot be written back for more than one sample."""
    multi = GENOTYPE_VCF_HEADER.replace("\tsample1\n", "\tsample1\tsample2\n") + (
        "chrM\t51\t.\tA\tG\t60\tPASS\tAF=0.5\tGT:VAF:AD:PL\t0/1:0.99:9,1183:0,0,33\t0/1:0.1:900,100:0,0,33\n"
    )
    input_vcf = _write(tmp_path / "multi.vcf", multi)
    with pytest.raises(ValueError, match="single-sample"):
        postprocess_mitochondria(input_vcf, str(tmp_path / "out.vcf.gz"), regenotype_contig="chrM")
