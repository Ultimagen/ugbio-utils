"""Tests of the SigProfiler project; they run in its own environment:

uv run --project src/comparison/sigprofiler pytest src/comparison/sigprofiler/tests
"""

import gzip
from pathlib import Path

import pytest

somatic_signatures = pytest.importorskip("ugbio_sigprofiler.somatic_signatures")

VCF = """##fileformat=VCFv4.2
#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tsample
chr1\t100\t.\tA\tC\t50\tPASS\t.\tGT\t0/1
chr1\t200\t.\tG\t.\t50\tRefCall\t.\tGT\t0/0
chr1\t300\t.\tAT\tA\t50\tPASS\t.\tGT\t0/1
"""
MATRICES_DIR = Path(__file__).parents[2] / "tests" / "resources" / "sigprofiler_matrices"


@pytest.mark.parametrize("compressed", [False, True])
def test_prepare_input_vcf(tmp_path, compressed):
    input_vcf = tmp_path / ("tumor.trimmed.vcf.gz" if compressed else "tumor.trimmed.vcf")
    opener = gzip.open if compressed else open
    with opener(input_vcf, "wt") as vcf:
        vcf.write(VCF)

    output_vcf = somatic_signatures.prepare_input_vcf(input_vcf, tmp_path / "input")

    assert output_vcf == tmp_path / "input" / "tumor.vcf"
    records = [line.split("\t")[1] for line in output_vcf.read_text().splitlines() if not line.startswith("#")]
    assert records == ["100", "300"]


def test_prepare_input_vcf_requires_new_dir(tmp_path):
    input_vcf = tmp_path / "tumor.vcf"
    input_vcf.write_text(VCF)
    (tmp_path / "input").mkdir()
    with pytest.raises(FileExistsError):
        somatic_signatures.prepare_input_vcf(input_vcf, tmp_path / "input")


def test_png_name():
    assert somatic_signatures.CONTEXTS[0].png_name("SBS1") == "SBS_96_plots_SBS1.png"
    assert somatic_signatures.CONTEXTS[1].png_name("sample") == "ID_83_plots_sample.png"
    assert somatic_signatures.CONTEXTS[2].png_name("DBS2") == "DBS_78_plots_DBS2.png"


def test_parse_args_requires_genome_archive_with_vcf():
    with pytest.raises(SystemExit):
        somatic_signatures.parse_args(["--input_vcf", "in.vcf", "--output_dir", "out"])


def test_run_from_matrices(tmp_path):
    summary = somatic_signatures.run_somatic_signatures(
        output_dir=tmp_path, genome="GRCh38", cosmic_version=3.3, matrices_dir=MATRICES_DIR
    )
    assert summary == {"sample_name": "somatic_test", "contexts": ["SBS96", "ID83", "DBS78"]}
    assert (tmp_path / "final_results" / "SBS_96_plots_somatic_test.png").is_file()
