from pathlib import Path

import pytest
from ugbio_sequins import sequins_igv

HEADER = "##fileformat=VCFv4.2\n##contig=<ID=chrQ_mirror,length=1000>\n#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n"
FP_VCF = HEADER + "chrQ_mirror\t100\t.\tC\tT\t8.7\tPASS\t.\n"
FN_VCF = HEADER + "chrQ_mirror\t300\t.\tGT\tG\t.\t.\t.\n"


@pytest.fixture
def files(tmp_path):
    (tmp_path / "truth.vcf").write_text(HEADER + "chrQ_mirror\t100\t.\tC\tT\t.\t.\t.\n")
    (tmp_path / "fp.vcf").write_text(FP_VCF)
    (tmp_path / "fn.vcf").write_text(FN_VCF)
    return tmp_path


def argv(d, fp="fp.vcf", fn="fn.vcf"):
    return [
        "sequins_igv", "--genome-name", "sequins_core",
        "--ref-fasta", "/some/dir/ref.fasta", "--cram", "/x/y/calibrated.cram",
        "--truth-vcf", str(d / "truth.vcf"), "--fp-vcf", str(d / fp), "--fn-vcf", str(d / fn),
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
    (files / "empty.vcf").write_text(HEADER)
    _, bed = sequins_igv.run(argv(files, "empty.vcf", "empty.vcf"))
    bed_content = Path(bed).read_text()
    assert bed_content == ""


def test_empty_truth_vcf_fails(files):
    (files / "truth.vcf").write_text(HEADER)
    with pytest.raises(ValueError, match="No records"):
        sequins_igv.run(argv(files))
