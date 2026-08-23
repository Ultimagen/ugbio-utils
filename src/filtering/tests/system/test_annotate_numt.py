from pathlib import Path

import pysam
import pytest
from ugbio_filtering.annotate_numt import annotate_numt, info_id_from_bed, match_allele

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
    annotate_numt(example_vcf, output_vcf, numt_bed, nuclear_bed, [bam], reference)

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
    annotate_numt(example_vcf, output_vcf, numt_bed, nuclear_bed, [bam], reference)

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
    annotate_numt(example_vcf, output_vcf, numt_bed, nuclear_bed, [bam], reference)

    with pysam.VariantFile(output_vcf) as vcf:
        records = {record.pos: record for record in vcf.fetch()}
    assert records[51].info.get("NUMT"), "the positional flag still applies"
    assert "NUMT" not in records[51].filter.keys()


def test_local_ref_rate_is_the_background(example_vcf, numt_beds, reference, tmp_path: Path):
    """Term 2: uniformly high NUMT SA traffic (the D-loop case) must not flag anything."""
    numt_bed, nuclear_bed = numt_beds
    bam = _make_bam(tmp_path / "uniform.bam", reference, pos=51, alt_reads=20, ref_reads=20, alt_sa=5, ref_sa=5)
    output_vcf = str(tmp_path / "out.vcf.gz")
    annotate_numt(example_vcf, output_vcf, numt_bed, nuclear_bed, [bam], reference)

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
    annotate_numt(example_vcf, output_vcf, numt_bed, nuclear_bed, [bam], reference)

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
