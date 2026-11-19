import pandas as pd
import pytest
from ugbio_comparison import sequins_report
from ugbio_comparison.sequins_vcfeval import H5_KEY


def row(call, base, alleles=("A", "C"), pos=100, gt=(0, 1), gt_truth=(0, 1), **kw):
    return {
        "chrom": "chrQ", "pos": pos, "alleles": alleles, "indel": len({len(a) for a in alleles}) > 1,
        "call": call, "base": base, "gt_ultima": gt, "gt_ground_truth": gt_truth,
        "qual": 8.7, "filter": "PASS", "dp": 17, **kw,
    }  # fmt: skip


def make_h5(path, rows):
    pd.DataFrame(rows).to_hdf(path, key=H5_KEY, mode="w")
    return str(path)


def run(tmp_path, genomes, medians, detected, h5s):
    argv = [
        "sequins_report", "--base-file-name", "sample1", "--output", str(tmp_path / "report.html"),
        "--input-crams", "s3://bucket/a.cram", "s3://bucket/b<x>.cram", "--target-coverage", "40",
        "--min-median-depth", "5", "--genomes", *genomes, "--median-depths", *map(str, medians),
        "--detected", *detected, "--concordance-h5s", *h5s,
    ]  # fmt: skip
    return (tmp_path / sequins_report.run(argv)).read_text()


def test_metrics_table():
    conc = pd.DataFrame(
        [row("TP", "TP", pos=i) for i in range(4)]
        + [row("FP", None, pos=10)]
        + [row(None, "FN", alleles=("A", "AT"), pos=20, gt=(None,))]
    )
    snps, indels = sequins_report.metrics_table(conc).to_dict("records")
    assert (snps["TP (truth)"], snps["TP (call)"], snps["FP"], snps["FN"]) == (4, 4, 1, 0)
    assert (snps["precision"], snps["recall"], snps["F1"]) == ("0.8000", "1.0000", "0.8889")
    assert (indels["FN"], indels["recall"], indels["F1"]) == (1, "0.0000", "0.0000")


def test_discordant_table():
    conc = pd.DataFrame([row("TP", "TP"), row("FP", None, pos=5), row(None, "FN", ("A", "AT"), 7, (None,), (0, 1))])
    fp, fn = sequins_report.discordant_table(conc).to_dict("records")
    assert (fp["class"], fp["type"], fp["pos"], fp["alt"], fp["gt"], fp["DP"]) == ("FP", "snps", 5, "C", "0/1", 17)
    assert (fn["class"], fn["type"], fn["alt"]) == ("FN", "indels", "AT")


def test_report_detected_and_not(tmp_path):
    h5 = make_h5(tmp_path / "a.h5", [row("TP", "TP"), row("FP", None, pos=5)])
    text = run(tmp_path, ["sequins_clinical", "sequins_core"], [30, 1], ["true", "false"], [h5])
    assert "<li>s3://bucket/b&lt;x&gt;.cram</li>" in text
    assert "Downsampled to 40x" in text
    assert "Detected: sequins_clinical" in text
    assert "Sequins WGS Clinical Control Set" in text
    assert "<p>Not detected</p>" in text
    assert "<td>FP</td>" in text


def test_report_no_discordant(tmp_path):
    h5 = make_h5(tmp_path / "a.h5", [row("TP", "TP")])
    assert "<p>None.</p>" in run(tmp_path, ["sequins_clinical"], [30], ["true"], [h5])


def test_none_detected(tmp_path):
    text = run(tmp_path, ["sequins_clinical", "sequins_core"], [1, 2], ["false", "false"], [])
    assert "No sequins spike was detected." in text


def test_html_escaping(tmp_path):
    h5 = make_h5(tmp_path / "a.h5", [row("FP", None, alleles=("A", "<DEL>&"))])
    text = run(tmp_path, ["sequins_clinical"], [30], ["true"], [h5])
    assert "&lt;DEL&gt;&amp;" in text
    assert "<DEL>" not in text


def test_mismatched_lengths_fail():
    with pytest.raises(SystemExit):
        sequins_report.parse_args(
            ["--base-file-name", "x", "--min-median-depth", "5", "--input-crams", "a", "--target-coverage", "40",
             "--genomes", "a", "b", "--median-depths", "1", "--detected", "false"]
        )  # fmt: skip
