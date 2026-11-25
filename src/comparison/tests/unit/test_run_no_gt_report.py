import argparse
from pathlib import Path
from unittest import mock

import pandas as pd
import pytest
from ugbio_comparison import run_no_gt_report

RESOURCES_DIR = Path(__file__).parent.parent / "resources"
INPUTS_DIR = RESOURCES_DIR / "run_no_gt_report"
GATK_TABLES = [
    "CompOverlap",
    "CountVariants",
    "IndelLengthHistogram",
    "IndelSummary",
    "MetricsCollection",
    "TiTvVariantEvaluator",
    "ValidationReport",
    "VariantSummary",
]


def test_allele_freq_hist():
    variants = pd.DataFrame(
        {
            "af": [0.1, 0.2, 0.3, 0.4, 0.5],
            "variant_type": ["snp", "snp", "h-indel", "non-h-indel", "non-h-indel"],
        }
    )
    result = run_no_gt_report.allele_freq_hist(variants)
    assert set(result.columns) == {"snp", "h-indel", "non-h-indel"}
    assert len(result) == 100
    assert result["snp"].iloc[10] == 1
    assert result["snp"].iloc[20] == 1
    assert sum(result["snp"] == 0) == 98


@mock.patch("subprocess.check_call")
def test_variant_eval_statistics(mocked_check_call):
    output_prefix = str(INPUTS_DIR / "collect_stats_unfiltered")
    data = run_no_gt_report.variant_eval_statistics(
        "vcf", "ref", "db_snp", output_prefix, 0, ["exome", "extended_exome", "high_conf"], append_confident_calls=True
    )
    mocked_check_call.assert_called_once_with(
        [
            "gatk",
            "VariantEval",
            "--eval",
            "vcf",
            "--reference",
            "ref",
            "--dbsnp",
            "db_snp",
            "--output",
            f"{output_prefix}.txt",
            "--selectNames",
            "exome",
            "--selectNames",
            "extended_exome",
            "--selectNames",
            "high_conf",
            "--selectNames",
            "HIGHCONF",
            "--select",
            'vc.hasAttribute("exome")',
            "--select",
            'vc.hasAttribute("extended_exome")',
            "--select",
            'vc.hasAttribute("high_conf")',
            "--select",
            "vc.getGenotype(0).getGQ()>=20",
        ]
    )
    for name in GATK_TABLES:
        assert isinstance(data[name], pd.DataFrame)
        assert not data[name].empty


@mock.patch("subprocess.check_call")
def test_variant_eval_statistics_sample_name(mocked_check_call):
    output_prefix = str(INPUTS_DIR / "collect_stats_unfiltered")
    run_no_gt_report.variant_eval_statistics(
        "vcf", "ref", "db_snp", output_prefix, "sample1", [], append_confident_calls=True
    )
    cmd = mocked_check_call.call_args.args[0]
    assert cmd[-2:] == ["--select", 'vc.getGenotype("sample1").getGQ()>=20']


@mock.patch("subprocess.check_call")
def test_variant_eval_statistics_no_confident_calls(mocked_check_call):
    output_prefix = str(INPUTS_DIR / "collect_stats_unfiltered")
    run_no_gt_report.variant_eval_statistics("vcf", "ref", "db_snp", output_prefix)
    cmd = mocked_check_call.call_args.args[0]
    assert cmd[-2:] == ["--output", f"{output_prefix}.txt"]


def test_insertion_deletion_statistics():
    variants = pd.read_hdf(INPUTS_DIR / "df.hdf", "concordance")
    result = run_no_gt_report.insertion_deletion_statistics(variants)

    assert result["homo"].shape == (4, 12)
    assert result["hete"].shape == (4, 12)
    assert result["homo"].loc["ins G", 2] == 1
    assert result["homo"].to_numpy().sum() == 1
    assert (result["hete"] == 0).all().all()


def test_snp_statistics():
    variants = pd.read_hdf(INPUTS_DIR / "df.hdf", "chr20")
    result = run_no_gt_report.snp_statistics(variants, str(RESOURCES_DIR / "sample.fasta"))

    assert len(result) == 96
    assert result.loc[("CCA", "A")] == 1
    assert result.loc[("CCA", "T")] == 1
    assert result.loc[("AAT", "G")] == 1
    assert result.loc[("ACT", "T")] == 2


def test_parse_args_somatic_analysis():
    args = run_no_gt_report.parse_args(
        [
            "somatic_analysis",
            "--input_file",
            "in.vcf",
            "--genome_archive",
            "GRCh38.tar.gz",
            "--output_prefix",
            "out",
        ]
    )
    assert args.func is run_no_gt_report.run_somatic_analysis
    assert args.reference_name == "GRCh38"
    assert args.cosmic_version == "3.3"
    assert args.reference_dir is None


def test_parse_args_somatic_analysis_requires_genome_archive():
    with pytest.raises(SystemExit):
        run_no_gt_report.parse_args(["somatic_analysis", "--input_file", "in.vcf", "--output_prefix", "out"])


def test_run_somatic_analysis(tmp_path):
    output_prefix = str(tmp_path / "sample_wgs")
    arg_values = argparse.Namespace(
        input_file="in.vcf",
        genome_archive="GRCh38.tar.gz",
        reference_name="GRCh38",
        cosmic_version="3.4",
        reference_dir="/ref",
        output_prefix=output_prefix,
    )
    summary = {"sample_name": "in", "contexts": ["SBS96"]}
    with (
        mock.patch.object(run_no_gt_report, "run_somatic_signatures", return_value=summary) as run_signatures,
        mock.patch.object(run_no_gt_report, "save_signatures_to_h5") as save_signatures,
    ):
        run_no_gt_report.run_somatic_analysis(arg_values)

    run_signatures.assert_called_once_with(
        Path(output_prefix),
        [
            "--input_vcf",
            "in.vcf",
            "--genome_archive",
            "GRCh38.tar.gz",
            "--genome",
            "GRCh38",
            "--cosmic_version",
            "3.4",
            "--reference_dir",
            "/ref",
        ],
    )
    save_signatures.assert_called_once_with(Path(output_prefix), summary, f"{output_prefix}.h5")
