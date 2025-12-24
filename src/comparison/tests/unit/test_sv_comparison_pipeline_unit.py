import pandas as pd
from ugbio_comparison.sv_comparison_pipeline import SVComparison, get_parser  # Adjust the import path as needed


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


def test_run_truvari_ignore_filter(mocker):
    """Test run_truvari with ignore_filter=True removes --passonly flag"""
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
        ignore_filter=True,
    )

    mock_logger.info.assert_called_with(
        "truvari command: truvari bench -b ground_truth.vcf -c calls.vcf -o output_dir"
        " -t --includebed regions.bed --pctseq 0.9 --pctsize 0.8"
    )
    mock_execute.assert_called_once_with(
        "truvari bench -b ground_truth.vcf -c calls.vcf -o output_dir -t"
        " --includebed regions.bed --pctseq 0.9 --pctsize 0.8"
    )


def test_truvari_to_dataframes(mocker):
    mock_vcftools = mocker.patch("ugbio_core.vcfbed.vcftools.get_vcf_df")  # Adjusted import path

    mock_vcftools.side_effect = [
        pd.DataFrame([{"svtype": "DEL", "svlen": 100, "matchid": "match1"}]),  # tp-base.vcf.gz
        pd.DataFrame([{"svtype": "INS", "svlen": 200, "matchid": None}]),  # fn.vcf.gz
        pd.DataFrame([{"svtype": "INV", "svlen": 150, "matchid": "match1"}]),  # tp-comp.vcf.gz
        pd.DataFrame([{"svtype": "INS", "svlen": 250, "matchid": None}]),  # fp.vcf.gz
    ]

    sv_comparison = SVComparison()

    df_base, df_calls = sv_comparison.truvari_to_dataframes("truvari_dir")

    assert not df_base.empty
    assert not df_calls.empty
    assert "label" in df_base.columns
    assert "label" in df_calls.columns
    assert "label_type" in df_base.columns
    assert "label_type" in df_calls.columns
    # Check that TP base variant has label_type from tp-comp (INV)
    assert df_base.iloc[0]["label_type"] == "INV", "TP base should have label_type from matching tp-comp"
    # Check that FN has label_type equal to label
    assert df_base.iloc[1]["label_type"] == "FN", "FN should have label_type equal to label"
    # Check that TP calls variant has label_type from tp-base (DEL)
    assert df_calls.iloc[0]["label_type"] == "DEL", "TP calls should have label_type from matching tp-base"
    # Check that FP has label_type equal to label
    assert df_calls.iloc[1]["label_type"] == "FP", "FP should have label_type equal to label"


def test_truvari_to_dataframes_svlen_processing(mocker):
    """Test that svlen column is preserved and svlen_int column is created correctly"""
    mock_vcftools = mocker.patch("ugbio_core.vcfbed.vcftools.get_vcf_df")

    # Mock dataframes with various svlen formats including tuples and single values
    mock_vcftools.side_effect = [
        pd.DataFrame([{"svtype": "DEL", "svlen": (100,), "matchid": "m1"}]),  # tp-base.vcf.gz with tuple
        pd.DataFrame([{"svtype": "INS", "svlen": 200, "matchid": None}]),  # fn.vcf.gz with single value
        pd.DataFrame([{"svtype": "DUP", "svlen": (150, 75), "matchid": "m1"}]),  # tp-comp.vcf.gz with tuple
        pd.DataFrame([{"svtype": "INS", "svlen": None, "matchid": None}]),  # fp.vcf.gz with None value
    ]

    sv_comparison = SVComparison()
    df_base, df_calls = sv_comparison.truvari_to_dataframes("truvari_dir")

    # Check that both dataframes have the required columns
    assert "svlen" in df_base.columns, "Original svlen column should be preserved"
    assert "svlen_int" in df_base.columns, "svlen_int column should be created"
    assert "svlen" in df_calls.columns, "Original svlen column should be preserved"
    assert "svlen_int" in df_calls.columns, "svlen_int column should be created"
    assert "label_type" in df_base.columns, "label_type column should be created"
    assert "label_type" in df_calls.columns, "label_type column should be created"

    # Check that original svlen values are preserved
    assert df_base.iloc[0]["svlen"] == (100,), "Original tuple svlen should be preserved"
    assert df_base.iloc[1]["svlen"] == 200, "Original single svlen should be preserved"

    # Check that svlen_int contains correct integer values
    assert df_base.iloc[0]["svlen_int"] == 100, "Tuple svlen should be converted to first element"
    assert df_base.iloc[1]["svlen_int"] == 200, "Single svlen should be preserved as integer"
    assert df_calls.iloc[0]["svlen_int"] == 150, "Tuple svlen should be converted to first element"
    assert df_calls.iloc[1]["svlen_int"] == 0, "None svlen should be filled with 0"

    # Check label_type values
    assert df_base.iloc[0]["label_type"] == "DUP", "TP base should have label_type from matching tp-comp"
    assert df_base.iloc[1]["label_type"] == "FN", "FN should have label_type equal to label"
    assert df_calls.iloc[0]["label_type"] == "DEL", "TP calls should have label_type from matching tp-base"
    assert df_calls.iloc[1]["label_type"] == "FP", "FP should have label_type equal to label"


def test_run_pipeline(mocker):
    mock_logger = mocker.Mock()
    mock_sp = mocker.Mock()
    sv_comparison = SVComparison(simple_pipeline=mock_sp, logger=mock_logger)
    mock_collapse_vcf = mocker.patch.object(sv_comparison.vu, "collapse_vcf")
    mock_sort_vcf = mocker.patch.object(sv_comparison.vu, "sort_vcf")
    mock_index_vcf = mocker.patch.object(sv_comparison.vu, "index_vcf")
    mock_run_truvari = mocker.patch.object(sv_comparison, "run_truvari")
    mock_truvari_to_dataframes = mocker.patch.object(sv_comparison, "truvari_to_dataframes")
    mock_truvari_to_dataframes.return_value = (pd.DataFrame(), pd.DataFrame())
    mock_to_hdf = mocker.patch("pandas.DataFrame.to_hdf")

    # Mock TemporaryDirectory as a context manager
    mock_temp_dir = mocker.patch("tempfile.TemporaryDirectory")
    mock_temp_dir.return_value.__enter__.return_value = "/tmp/test_dir"
    mock_temp_dir.return_value.__exit__.return_value = None

    mock_move = mocker.patch("shutil.move")
    mock_exists = mocker.patch("os.path.exists")
    mock_exists.return_value = True

    sv_comparison.run_pipeline(
        calls="calls.vcf.gz",
        gt="ground_truth.vcf.gz",
        output_file_name="output.h5",
        outdir="output_dir",
        hcr_bed="regions.bed",
        pctseq=0.9,
        pctsize=0.8,
        erase_outdir=True,
    )

    # Verify collapse_vcf calls with temporary directory paths
    mock_collapse_vcf.assert_any_call(
        "calls.vcf.gz",
        "/tmp/test_dir/calls_collapsed.vcf.gz",
        bed="regions.bed",
        pctseq=0.9,
        pctsize=0.8,
        ignore_filter=False,
        ignore_sv_type=True,
    )
    mock_collapse_vcf.assert_any_call(
        "ground_truth.vcf.gz",
        "/tmp/test_dir/ground_truth_collapsed.vcf.gz",
        bed="regions.bed",
        pctseq=0.9,
        pctsize=0.8,
        ignore_filter=False,
    )

    # Verify sort and index operations are called
    mock_sort_vcf.assert_called()
    mock_index_vcf.assert_called()

    # Verify truvari is called with the correct temporary file paths
    mock_run_truvari.assert_called_once_with(
        calls="/tmp/test_dir/calls_collapsed.sort.vcf.gz",
        gt="/tmp/test_dir/ground_truth_collapsed.sort.vcf.gz",
        outdir="output_dir",
        bed="regions.bed",
        pctseq=0.9,
        pctsize=0.8,
        erase_outdir=True,
        ignore_filter=False,
        ignore_type=True,
        skip_collapse=False,
    )
    mock_to_hdf.assert_called()

    # Verify temporary files are moved to output directory
    mock_move.assert_called()

    # Verify TemporaryDirectory context manager was used
    mock_temp_dir.assert_called_once()
    mock_temp_dir.return_value.__enter__.assert_called_once()
    mock_temp_dir.return_value.__exit__.assert_called_once()


def test_run_pipeline_ignore_filter(mocker):
    """Test run_pipeline with ignore_filter=True passes the flag correctly to sub-methods"""
    mock_logger = mocker.Mock()
    mock_sp = mocker.Mock()
    sv_comparison = SVComparison(simple_pipeline=mock_sp, logger=mock_logger)
    mock_collapse_vcf = mocker.patch.object(sv_comparison.vu, "collapse_vcf")
    mock_remove_filters = mocker.patch.object(sv_comparison.vu, "remove_filters")
    _ = mocker.patch.object(sv_comparison.vu, "sort_vcf")
    _ = mocker.patch.object(sv_comparison.vu, "index_vcf")
    mock_run_truvari = mocker.patch.object(sv_comparison, "run_truvari")
    mock_truvari_to_dataframes = mocker.patch.object(sv_comparison, "truvari_to_dataframes")
    mock_truvari_to_dataframes.return_value = (pd.DataFrame(), pd.DataFrame())
    _ = mocker.patch("pandas.DataFrame.to_hdf")

    # Mock TemporaryDirectory as a context manager
    mock_temp_dir = mocker.patch("tempfile.TemporaryDirectory")
    mock_temp_dir.return_value.__enter__.return_value = "/tmp/test_dir"
    mock_temp_dir.return_value.__exit__.return_value = None

    _ = mocker.patch("shutil.move")
    mock_exists = mocker.patch("os.path.exists")
    mock_exists.return_value = True

    sv_comparison.run_pipeline(
        calls="calls.vcf.gz",
        gt="ground_truth.vcf.gz",
        output_file_name="output.h5",
        outdir="output_dir",
        hcr_bed="regions.bed",
        pctseq=0.9,
        pctsize=0.8,
        erase_outdir=True,
        ignore_filter=True,
    )

    # Verify remove_filters is called to create the nofilter version
    mock_remove_filters.assert_called_once_with(
        input_vcf="calls.vcf.gz", output_vcf="/tmp/test_dir/calls.nofilter.vcf.gz"
    )

    # Verify collapse_vcf calls with the nofilter file for calls
    mock_collapse_vcf.assert_any_call(
        "/tmp/test_dir/calls.nofilter.vcf.gz",
        "/tmp/test_dir/calls.nofilter_collapsed.vcf.gz",
        bed="regions.bed",
        pctseq=0.9,
        pctsize=0.8,
        ignore_filter=True,
        ignore_sv_type=True,
    )
    mock_collapse_vcf.assert_any_call(
        "ground_truth.vcf.gz",
        "/tmp/test_dir/ground_truth_collapsed.vcf.gz",
        bed="regions.bed",
        pctseq=0.9,
        pctsize=0.8,
        ignore_filter=True,
    )

    # Verify truvari is called with ignore_filter=True and the processed files
    mock_run_truvari.assert_called_once_with(
        calls="/tmp/test_dir/calls.nofilter_collapsed.sort.vcf.gz",
        gt="/tmp/test_dir/ground_truth_collapsed.sort.vcf.gz",
        outdir="output_dir",
        bed="regions.bed",
        pctseq=0.9,
        pctsize=0.8,
        erase_outdir=True,
        ignore_filter=True,
        ignore_type=True,
        skip_collapse=False,
    )


def test_argument_parser_ignore_filter():
    """Test that the argument parser correctly handles the --ignore_filter flag"""
    parser = get_parser()

    # Test with --ignore_filter flag
    args = parser.parse_args(
        [
            "--calls",
            "calls.vcf.gz",
            "--gt",
            "ground_truth.vcf.gz",
            "--output_filename",
            "output.h5",
            "--outdir",
            "output_dir",
            "--ignore_filter",
        ]
    )
    assert args.ignore_filter is True

    # Test without --ignore_filter flag (should default to False)
    args = parser.parse_args(
        [
            "--calls",
            "calls.vcf.gz",
            "--gt",
            "ground_truth.vcf.gz",
            "--output_filename",
            "output.h5",
            "--outdir",
            "output_dir",
        ]
    )
    assert args.ignore_filter is False
