from pathlib import Path

import pytest
from ugbio_sequins import sequins_report

HEADER = (
    "##fileformat=VCFv4.2\n##contig=<ID=chrQ,length=1000>\n"
    '##FORMAT=<ID=GT,Number=1,Type=String,Description="g">\n'
    '##FORMAT=<ID=DP,Number=1,Type=Integer,Description="d">\n'
    '##INFO=<ID=DP,Number=1,Type=Integer,Description="d">\n'
    "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n"
)
SNP = ("A", "C")
INDEL = ("A", "AT")


def write_vcf(path, variants):
    rows = "".join(f"chrQ\t{100 + i}\t.\t{ref}\t{alt}\t.\t.\t.\n" for i, (ref, alt) in enumerate(variants))
    path.write_text(HEADER + rows)
    return str(path)


def make_genome(tmp_path, name, tpb, tp, fp, fn):
    d = tmp_path / name
    d.mkdir()
    return {
        "tpb": write_vcf(d / "tpb.vcf", tpb),
        "tp": write_vcf(d / "tp.vcf", tp),
        "fp": write_vcf(d / "fp.vcf", fp),
        "fn": write_vcf(d / "fn.vcf", fn),
    }


def cli(tmp_path, genomes, medians, detected, results, min_depth=5):
    argv = [
        "sequins_report",
        "--base-file-name", "sample1",
        "--output", str(tmp_path / "report.html"),
        "--input-crams", "s3://bucket/a.cram", "s3://bucket/b<x>.cram",
        "--target-coverage", "40",
        "--min-median-depth", str(min_depth),
        "--genomes", *genomes,
        "--median-depths", *map(str, medians),
        "--detected", *detected,
        "--tp-baseline-vcfs", *[r["tpb"] for r in results],
        "--tp-vcfs", *[r["tp"] for r in results],
        "--fp-vcfs", *[r["fp"] for r in results],
        "--fn-vcfs", *[r["fn"] for r in results],
    ]  # fmt: skip
    return Path(sequins_report.run(argv)).read_text()


def test_input_crams_and_coverage(tmp_path):
    g = make_genome(tmp_path, "g", [SNP], [SNP], [], [])
    text = cli(tmp_path, ["sequins_clinical", "sequins_core"], [30, 1], ["true", "false"], [g])
    assert "<li>s3://bucket/a.cram</li>" in text
    assert "<li>s3://bucket/b&lt;x&gt;.cram</li>" in text
    assert "Downsampled to 40x" in text


def test_default_output_name(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    g = make_genome(tmp_path, "g", [], [], [], [])
    argv = [
        "sequins_report", "--base-file-name", "s", "--min-median-depth", "5",
        "--input-crams", "a.cram", "--target-coverage", "40",
        "--genomes", "x", "--median-depths", "9", "--detected", "true",
        "--tp-baseline-vcfs", g["tpb"], "--tp-vcfs", g["tp"], "--fp-vcfs", g["fp"], "--fn-vcfs", g["fn"],
    ]  # fmt: skip
    assert sequins_report.run(argv) == "report.html"
    assert (tmp_path / "report.html").exists()


def test_precision_recall_f1():
    p, r, f1 = sequins_report.precision_recall_f1(tp_baseline=8, tp_call=9, fp=1, fn=2)
    assert p == pytest.approx(0.9)
    assert r == pytest.approx(0.8)
    assert f1 == pytest.approx(2 * 0.9 * 0.8 / 1.7)
    assert sequins_report.precision_recall_f1(0, 0, 0, 0) == (0.0, 0.0, 0.0)


def test_count_variants(tmp_path):
    vcf = write_vcf(tmp_path / "x.vcf", [SNP, SNP, INDEL])
    assert sequins_report.count_variants(vcf, "snps") == 2
    assert sequins_report.count_variants(vcf, "indels") == 1


def test_both_detected(tmp_path):
    g1 = make_genome(tmp_path, "g1", [SNP] * 4, [SNP] * 4, [SNP], [INDEL])
    g2 = make_genome(tmp_path, "g2", [SNP], [SNP], [], [])
    text = cli(tmp_path, ["sequins_clinical", "sequins_core"], [30, 20], ["true", "true"], [g1, g2])
    assert "Detected: sequins_clinical sequins_core" in text
    assert "<td>snps</td><td>4</td><td>4</td><td>1</td><td>0</td><td>0.8000</td><td>1.0000</td>" in text
    assert "<td>FP</td><td>snps</td><td>chrQ</td><td>100</td><td>A</td><td>C</td>" in text
    assert "<td>FN</td><td>indels</td>" in text
    assert "<p>None.</p>" in text  # g2 has no discordant variants


def test_second_not_detected_uses_k_index(tmp_path):
    g = make_genome(tmp_path, "g", [SNP], [SNP], [], [])
    text = cli(tmp_path, ["sequins_clinical", "sequins_core"], [30, 1], ["true", "false"], [g])
    assert "Detected: sequins_clinical" in text
    assert "<p>Not detected</p>" in text
    assert '<span class="no">no</span>' in text


def test_none_detected(tmp_path):
    text = cli(tmp_path, ["sequins_clinical", "sequins_core"], [1, 2], ["false", "false"], [])
    assert "No sequins spike was detected." in text
    assert "<h3>Discordant variants</h3>" not in text


def test_html_escaping(tmp_path):
    g = make_genome(tmp_path, "g", [], [], [("A", "<DEL>&")], [])
    text = cli(tmp_path, ["sequins_clinical"], [30], ["true"], [g])
    assert "&lt;DEL&gt;&amp;" in text
    assert "<DEL>" not in text


def test_mismatched_lengths_fail(tmp_path):
    with pytest.raises(SystemExit):
        sequins_report.parse_args(
            ["--base-file-name", "x", "--min-median-depth", "5", "--input-crams", "a.cram", "--target-coverage", "40",
             "--genomes", "a", "b",
             "--median-depths", "1", "--detected", "false"]
        )  # fmt: skip


def test_discordant_table_from_vcfs(tmp_path):
    g = make_genome(tmp_path, "g", [], [], [SNP], [INDEL])
    Path(g["fp"]).write_text(HEADER.rstrip("\n") + "\tFORMAT\tS\nchrQ\t100\t.\tA\tC\t8.7\tPASS\t.\tGT:DP\t0/1:17\n")
    Path(g["fn"]).write_text(HEADER + "chrQ\t100\t.\tA\tAT\t.\t.\tDP=23\n")
    text = cli(tmp_path, ["sequins_clinical"], [30], ["true"], [g])
    assert "<th>DP</th>" in text
    assert (
        "<td>FP</td><td>snps</td><td>chrQ</td><td>100</td><td>A</td><td>C</td><td>0/1</td><td>8.7</td><td>PASS</td><td>17</td>"
        in text
    )
    assert (
        "<td>FN</td><td>indels</td><td>chrQ</td><td>100</td><td>A</td><td>AT</td><td>.</td><td>.</td><td>.</td><td>23</td>"
        in text
    )
