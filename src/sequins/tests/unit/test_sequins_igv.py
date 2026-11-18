from pathlib import Path

import pytest
from ugbio_sequins import sequins_igv

HEADER = "##fileformat=VCFv4.2\n##contig=<ID=chrQ_mirror,length=1000>\n#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n"
TABLE = (
    "class\ttype\tchrom\tpos\tref\talt\tgt\tqual\tfilter\n"
    "FN\tindels\tchrQ_mirror\t300\tGT\tG\t1|0\t.\t.\n"
    "FP\tsnps\tchrQ_mirror\t100\tC\tT\t1/1\t8.7\tPASS\n"
)


@pytest.fixture
def files(tmp_path):
    (tmp_path / "truth.vcf").write_text(HEADER + "chrQ_mirror\t100\t.\tC\tT\t.\t.\t.\n")
    (tmp_path / "disc.tsv").write_text(TABLE)
    return tmp_path


def argv(d, table="disc.tsv"):
    return [
        "sequins_igv", "--genome-name", "sequins_core",
        "--ref-fasta", "/some/dir/ref.fasta", "--cram", "/x/y/calibrated.cram",
        "--truth-vcf", str(d / "truth.vcf"), "--discordant-table", str(d / table),
        "--output-prefix", str(d / "out"),
    ]  # fmt: skip


def test_batch_and_bed(files):
    batch, bed = sequins_igv.run(argv(files))
    bed_lines = Path(bed).read_text().splitlines()
    batch_lines = Path(batch).read_text().splitlines()
    assert bed_lines == [
        "chrQ_mirror\t99\t100\tFP_snps_C>T",
        "chrQ_mirror\t299\t301\tFN_indels_GT>G",
    ]
    assert batch_lines == [
        "new",
        "genome ref.fasta",
        "load calibrated.cram",
        "load truth.vcf",
        "load out.discordant.bed",
        "goto chrQ_mirror",
    ]


def test_no_discordant_variants(files):
    (files / "empty.tsv").write_text(TABLE.splitlines()[0] + "\n")
    _, bed = sequins_igv.run(argv(files, "empty.tsv"))
    bed_content = Path(bed).read_text()
    assert bed_content == ""


def test_empty_truth_vcf_fails(files):
    (files / "truth.vcf").write_text(HEADER)
    with pytest.raises(ValueError, match="No records"):
        sequins_igv.run(argv(files))
