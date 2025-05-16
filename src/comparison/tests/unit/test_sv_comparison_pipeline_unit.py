import pandas as pd
from ugbio_comparison.sv_comparison_pipeline import SVComparison  # Adjust the import path as needed


def test_collapse_vcf(mocker):
    mock_logger = mocker.Mock()
    mock_sp = mocker.Mock()
    sv_comparison = SVComparison(simple_pipeline=mock_sp, logger=mock_logger)

    mock_subprocess_popen = mocker.patch("subprocess.Popen")
    mock_p1 = mocker.Mock()
    mock_p2 = mocker.Mock()
    mock_subprocess_popen.side_effect = [mock_p1, mock_p2]
    mock_p1.stdout = mocker.Mock()
    mock_p1.returncode = 0
    mock_p2.returncode = 0

    sv_comparison.collapse_vcf("input.vcf", "output.vcf.gz", bed="regions.bed", pctseq=0.9, pctsize=0.8)

    mock_logger.info.assert_called_with(
        "truvari command: truvari collapse -i input.vcf --passonly -t --includebed regions.bed"
        " --pctseq 0.9 --pctsize 0.8"
    )
    mock_subprocess_popen.assert_any_call(
        [
            "truvari",
            "collapse",
            "-i",
            "input.vcf",
            "--passonly",
            "-t",
            "--bed",
            "regions.bed",
            "--pctseq",
            "0.9",
            "--pctsize",
            "0.8",
        ],
        stdout=mocker.ANY,
    )
    mock_subprocess_popen.assert_any_call(["bcftools", "view", "-Oz", "-o", "output.vcf.gz"], stdin=mock_p1.stdout)


def test_run_truvari(mocker):
    mock_logger = mocker.Mock()
    mock_sp = mocker.Mock()
    sv_comparison = SVComparison(simple_pipeline=mock_sp, logger=mock_logger)

    mock_execute = mocker.patch.object(sv_comparison, "_SVComparison__execute")

    sv_comparison.run_truvari(
        calls="calls.vcf",
        gt="ground_truth.vcf",
        outdir="output_dir",
        bed="regions.bed",
        pctseq=0.9,
        pctsize=0.8,
        erase_outdir=True,
    )

    mock_logger.info.assert_called_with(
        "truvari command: truvari bench -b ground_truth.vcf -c calls.vcf -o output_dir"
        " -t --passonly --includebed regions.bed --pctseq 0.9 --pctsize 0.8"
    )
    mock_execute.assert_called_once_with(
        "truvari bench -b ground_truth.vcf -c calls.vcf -o output_dir -t"
        " --passonly --includebed regions.bed --pctseq 0.9 --pctsize 0.8"
    )


def test_truvari_to_dataframes(mocker):
    mock_vcftools = mocker.patch("ugbio_core.vcfbed.vcftools.get_vcf_df")  # Adjusted import path

    mock_vcftools.side_effect = [
        pd.DataFrame([{"SVTYPE": "DEL", "SVLEN": 100}]),  # tp-base.vcf.gz
        pd.DataFrame([{"SVTYPE": "INS", "SVLEN": 200}]),  # fn.vcf.gz
        pd.DataFrame([{"SVTYPE": "DEL", "SVLEN": 150}]),  # tp-comp.vcf.gz
        pd.DataFrame([{"SVTYPE": "INS", "SVLEN": 250}]),  # fp.vcf.gz
    ]

    sv_comparison = SVComparison()

    df_base, df_calls = sv_comparison.truvari_to_dataframes("truvari_dir")

    assert not df_base.empty
    assert not df_calls.empty
    assert "label" in df_base.columns
    assert "label" in df_calls.columns


def test_run_pipeline(mocker):
    mock_logger = mocker.Mock()
    mock_sp = mocker.Mock()
    sv_comparison = SVComparison(simple_pipeline=mock_sp, logger=mock_logger)

    mock_collapse_vcf = mocker.patch.object(sv_comparison, "collapse_vcf")
    mock_sort_vcf = mocker.patch.object(sv_comparison.vpu, "sort_vcf")
    mock_index_vcf = mocker.patch.object(sv_comparison.vpu, "index_vcf")
    mock_run_truvari = mocker.patch.object(sv_comparison, "run_truvari")
    mock_truvari_to_dataframes = mocker.patch.object(sv_comparison, "truvari_to_dataframes")
    mock_truvari_to_dataframes.return_value = (pd.DataFrame(), pd.DataFrame())
    mock_to_hdf = mocker.patch("pandas.DataFrame.to_hdf")

    sv_comparison.run_pipeline(
        calls="calls.vcf",
        gt="ground_truth.vcf",
        output_file_name="output.h5",
        outdir="output_dir",
        hcr_bed="regions.bed",
        pctseq=0.9,
        pctsize=0.8,
        erase_outdir=True,
    )

    mock_collapse_vcf.assert_any_call("calls.vcf", "calls_collapsed.vcf.gz", bed="regions.bed", pctseq=0.9, pctsize=0.8)
    mock_sort_vcf.assert_called()
    mock_index_vcf.assert_called()
    mock_run_truvari.assert_called_once_with(
        calls="calls_sort.vcf.gz",
        gt="ground_truth_sort.vcf.gz",
        outdir="output_dir",
        bed="regions.bed",
        pctseq=0.9,
        pctsize=0.8,
        erase_outdir=True,
    )
    mock_to_hdf.assert_called()
