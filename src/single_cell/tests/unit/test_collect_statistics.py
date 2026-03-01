import shutil
from gzip import BadGzipFile
from pathlib import Path

import pandas as pd
import pytest
from ugbio_single_cell.collect_statistics import (
    collect_statistics,
    extract_cell_barcode_filter_data,
    extract_statistics_table,
    get_insert_properties,
    read_star_stats,
)
from ugbio_single_cell.sc_qc_dataclasses import H5Keys, Inputs, Thresholds


@pytest.fixture
def inputs_dir():
    inputs_dir = Path(__file__).parent.parent / "resources"
    return inputs_dir


@pytest.fixture
def inputs(inputs_dir):
    return Inputs(
        trimmer_stats_csv=str(inputs_dir / "trimmer_stats.csv"),
        trimmer_histogram_csv=[str(inputs_dir / "trimmer_histogram.csv")],
        trimmer_failure_codes_csv=str(inputs_dir / "trimmer_failure_codes.csv"),
        sorter_stats_csv=str(inputs_dir / "sorter_stats.csv"),
        star_stats=str(inputs_dir / "star_insert_Log.final.out"),
        star_reads_per_gene=str(inputs_dir / "star_insert_ReadsPerGene.out.tab"),
        insert=str(inputs_dir / "insert_subsample.fastq.gz"),
    )


@pytest.fixture
def thresholds():
    return Thresholds(
        pass_trim_rate=0.9,
        read_length=100,
        fraction_below_read_length=0.1,
        percent_aligned=0.9,
    )


@pytest.fixture
def sample_name():
    return "test_sample"


@pytest.fixture
def output_path(tmpdir):
    return Path(tmpdir)


def test_collect_statistics(output_path, inputs, sample_name):
    h5_file = collect_statistics(
        input_files=inputs,
        output_path=output_path,
        sample_name=sample_name,
    )

    # assert output file exists
    assert h5_file.exists()
    # assert output file is not empty
    assert h5_file.stat().st_size > 0
    # assert output file contains expected keys
    with pd.HDFStore(h5_file, mode="r") as store:
        expected_keys = [
            "/" + key.value
            for key in H5Keys
            if key not in [H5Keys.STATISTICS_SHORTLIST, H5Keys.TRIMMER_HISTOGRAM, H5Keys.SORTER_STATS_JSON]
        ]
        assert set(store.keys()) == set(expected_keys)


def test_collect_statistics_with_sorter_json(inputs_dir, output_path, inputs, sample_name):
    inputs.sorter_stats_json = str(inputs_dir / "sorter_stats.json")
    h5_file = collect_statistics(
        input_files=inputs,
        output_path=output_path,
        sample_name=sample_name,
    )

    # assert output file exists
    assert h5_file.exists()
    # assert output file contains expected keys
    with pd.HDFStore(h5_file, mode="r") as store:
        assert "/" + H5Keys.SORTER_STATS_JSON.value in store.keys()


def test_read_star_stats(inputs):
    star_stats_file = inputs.star_stats
    s = read_star_stats(star_stats_file)
    assert isinstance(s, pd.Series)
    assert len(s) == 33
    assert s.isna().sum().sum() == 0


def test_read_star_stats_missing_file():
    star_stats_file = "missing_file.txt"
    with pytest.raises(FileNotFoundError):
        read_star_stats(star_stats_file)


def test_get_insert_properties(inputs):
    insert = inputs.insert
    df_insert_quality, insert_lengths = get_insert_properties(insert)
    assert isinstance(df_insert_quality, pd.DataFrame)
    assert isinstance(insert_lengths, list)
    assert len(insert_lengths) == 9652
    assert df_insert_quality.shape == (11, 219)


def test_get_insert_properties_no_gzip(inputs):
    insert = inputs.star_stats
    with pytest.raises(BadGzipFile):
        get_insert_properties(insert)


def test_get_insert_properties_no_fastq(inputs):
    insert = inputs.trimmer_stats_csv
    with pytest.raises(BadGzipFile):
        get_insert_properties(insert)


def test_extract_statistics_table(inputs_dir, output_path):
    original_h5_file = inputs_dir / "single_cell_qc_stats_no_shortlist.scRNA.applicationQC.h5"
    # copy the file to tmpdir to avoid modifying the original file
    h5_file = output_path / original_h5_file.name
    shutil.copyfile(original_h5_file, str(h5_file))

    # assert h5 doesnt contain the STATISTICS_SHORTLIST key
    with pd.HDFStore(h5_file, mode="r") as store:
        assert "/" + H5Keys.STATISTICS_SHORTLIST.value not in store.keys()

    extract_statistics_table(h5_file)

    # assert h5 contains the STATISTICS_SHORTLIST key
    with pd.HDFStore(h5_file, mode="r") as store:
        s = store[H5Keys.STATISTICS_SHORTLIST.value]
        assert "/" + H5Keys.STATISTICS_SHORTLIST.value in store.keys()
        assert len(s) == 17
        # assert that entries that start with "pct_" are between 0 to 100
        assert s[s.index.str.startswith("pct_")].astype(float).between(0, 100).all()


def test_extract_statistics_table__num_input_reads_all_zero(inputs, output_path):
    df_trimmmer_stats = pd.read_csv(inputs.trimmer_stats_csv)
    df_trimmmer_stats["num input reads"] = 0
    h5_file = output_path / "test.h5"
    with pd.HDFStore(h5_file, mode="w") as store:
        store.put(H5Keys.TRIMMER_STATS.value, df_trimmmer_stats, format="table")

    with pytest.raises(ValueError):
        extract_statistics_table(h5_file)


def test_extract_statistics_table_with_sorter_json(inputs_dir, output_path):
    original_h5_file = inputs_dir / "single_cell_qc_stats_with_sorter_json.scRNA.applicationQC.h5"
    # copy the file to tmpdir to avoid modifying the original file
    h5_file = output_path / original_h5_file.name
    shutil.copyfile(original_h5_file, str(h5_file))

    extract_statistics_table(h5_file)

    # assert h5 contains the STATISTICS_SHORTLIST key
    with pd.HDFStore(h5_file, mode="r") as store:
        s = store[H5Keys.STATISTICS_SHORTLIST.value]
        assert "/" + H5Keys.STATISTICS_SHORTLIST.value in store.keys()
        assert len(s) == 21
        expected_pct_suspicious_cbcs = 100 * 0.02311600563934775
        expected_pct_suspicious_reads = 100 * 0.0018177121169518373
        assert s["pct_suspicious_cbcs_above_threshold"] == expected_pct_suspicious_cbcs
        assert s["pct_cbc_filter_suspicious_reads"] == expected_pct_suspicious_reads
        # backward-compatible keys
        assert s["pct_failed_cbcs_above_threshold"] == expected_pct_suspicious_cbcs
        assert s["pct_cbc_filter_failed_reads"] == expected_pct_suspicious_reads


# ---------------------------------------------------------------------------
# Integration tests with the 4420777 dataset
# ---------------------------------------------------------------------------

_DATA_ROOT = Path("/data/Runs/data_tmp/small_checks/tmp_ugbio_test/4420777")
_TRIMMER_TASK = _DATA_ROOT / "3853099/8d8fcc43-005c-3abd-a9c0-f6eb3a7487ab/out"
_INSERT_SORTER_TASK = _DATA_ROOT / "7236081/9e42d0e8-5c4d-3715-bcdc-790eef244ea6/out"
_STAR_TASK = _DATA_ROOT / "6486319/6b018dfb-d769-37b9-be46-46cd0210d12a/out"
_SAMPLE_PREFIX = "600428-IMU_Sample4_44-Z0140-CTCAGATCCTGCATGAT"

_skip_no_data = pytest.mark.skipif(not _DATA_ROOT.exists(), reason="4420777 dataset not available")


@pytest.fixture
def inputs_4420777():
    return Inputs(
        trimmer_stats_csv=str(_TRIMMER_TASK / "trimmer_stats" / f"{_SAMPLE_PREFIX}.trimmer_stats.csv"),
        trimmer_histogram_csv=[
            str(_TRIMMER_TASK / "histogram" / str(i) / f"CBC_pattern_fw_{bp}bp.histogram.csv")
            for i, bp in enumerate([12, 13, 14, 15])
        ],
        trimmer_failure_codes_csv=str(
            _TRIMMER_TASK / "trimmer_failure_codes_csv" / f"{_SAMPLE_PREFIX}.failure_codes.csv"
        ),
        sorter_stats_csv=str(_INSERT_SORTER_TASK / "insert_sorter_stats_csv" / f"{_SAMPLE_PREFIX}_S1_L001_R2_001.csv"),
        sorter_stats_json=str(
            _INSERT_SORTER_TASK / "insert_sorter_stats_json" / f"{_SAMPLE_PREFIX}_S1_L001_R2_001.json"
        ),
        star_stats=str(_STAR_TASK / "star_log_file" / f"{_SAMPLE_PREFIX}.Log.final.out"),
        star_reads_per_gene=str(_STAR_TASK / "reads_per_gene_file" / f"{_SAMPLE_PREFIX}.ReadsPerGene.out.tab"),
        insert=str(
            _INSERT_SORTER_TASK / "insert_sub_sample_fastq" / f"{_SAMPLE_PREFIX}_S1_L001_R2_001_sample.fastq.gz"
        ),
    )


@_skip_no_data
def test_collect_statistics_4420777(inputs_4420777, output_path):
    """End-to-end: collect_statistics produces a valid H5 with all expected keys."""
    h5_file = collect_statistics(
        input_files=inputs_4420777,
        output_path=str(output_path),
        sample_name="test_4420777_",
    )

    assert h5_file.exists()
    assert h5_file.stat().st_size > 0

    with pd.HDFStore(h5_file, mode="r") as store:
        expected_keys = {
            "/" + key.value for key in H5Keys if key not in [H5Keys.STATISTICS_SHORTLIST, H5Keys.TRIMMER_HISTOGRAM]
        }
        assert expected_keys.issubset(set(store.keys()))
        assert "/" + H5Keys.SORTER_STATS_JSON.value in store.keys()


@_skip_no_data
def test_extract_statistics_table_4420777(inputs_4420777, output_path):
    """
    Verify extract_statistics_table populates both new (suspicious) and
    backward-compatible (failed) stat keys when the sorter JSON uses the
    new naming convention.
    """
    h5_file = collect_statistics(
        input_files=inputs_4420777,
        output_path=str(output_path),
        sample_name="test_4420777_",
    )

    extract_statistics_table(h5_file)

    with pd.HDFStore(h5_file, mode="r") as store:
        s = store[H5Keys.STATISTICS_SHORTLIST.value]

    # Values from the sorter JSON
    nr_suspicious_cbcs = 322
    nr_good_cbcs_above_threshold = 93123
    nr_suspicious_reads = 143673
    total_reads = 47679494

    expected_pct_suspicious_cbcs = 100 * nr_suspicious_cbcs / (nr_suspicious_cbcs + nr_good_cbcs_above_threshold)
    expected_pct_suspicious_reads = 100 * nr_suspicious_reads / total_reads

    assert s["pct_suspicious_cbcs_above_threshold"] == pytest.approx(expected_pct_suspicious_cbcs)
    assert s["pct_cbc_filter_suspicious_reads"] == pytest.approx(expected_pct_suspicious_reads)
    # backward-compatible keys must carry the same values
    assert s["pct_failed_cbcs_above_threshold"] == pytest.approx(expected_pct_suspicious_cbcs)
    assert s["pct_cbc_filter_failed_reads"] == pytest.approx(expected_pct_suspicious_reads)

    # all pct_ entries should be in [0, 100]
    pct_entries = s[s.index.str.startswith("pct_")]
    assert pct_entries.astype(float).between(0, 100).all()


@_skip_no_data
def test_extract_cell_barcode_filter_new_keys(inputs_4420777, output_path):
    """
    Verify extract_cell_barcode_filter_data correctly reads the new
    'suspicious' key names (including the 'suspicous' typo variant)
    and populates both new and backward-compat output keys.
    """
    h5_file = collect_statistics(
        input_files=inputs_4420777,
        output_path=str(output_path),
        sample_name="test_4420777_",
    )

    stats = {}
    with pd.HDFStore(h5_file, mode="r") as store:
        extract_cell_barcode_filter_data(stats, store)

    assert "pct_suspicious_cbcs_above_threshold" in stats
    assert "pct_cbc_filter_suspicious_reads" in stats
    assert "pct_failed_cbcs_above_threshold" in stats
    assert "pct_cbc_filter_failed_reads" in stats

    assert stats["pct_suspicious_cbcs_above_threshold"] == stats["pct_failed_cbcs_above_threshold"]
    assert stats["pct_cbc_filter_suspicious_reads"] == stats["pct_cbc_filter_failed_reads"]


def test_extract_cell_barcode_filter_old_keys(output_path):
    """
    Verify backward compatibility: extract_cell_barcode_filter_data works
    when the sorter JSON uses the old 'nr_failed_cbcs' / 'nr_failed_reads' keys.
    """
    sorter_json_data = {
        "cell_barcode_filter": {
            "failed_cbcs": ["AAAA"],
            "nr_failed_cbcs": 10,
            "nr_good_cbcs_above_threshold": 990,
            "nr_failed_reads": 500,
        },
        "total_reads": 50000,
    }

    h5_file = output_path / "test_old_keys.h5"
    with pd.HDFStore(h5_file, mode="w") as store:
        store.put(H5Keys.SORTER_STATS_JSON.value, pd.DataFrame([sorter_json_data]))

    stats = {}
    with pd.HDFStore(h5_file, mode="r") as store:
        extract_cell_barcode_filter_data(stats, store)

    expected_pct_cbcs = 100 * 10 / (10 + 990)
    expected_pct_reads = 100 * 500 / 50000

    assert stats["pct_suspicious_cbcs_above_threshold"] == pytest.approx(expected_pct_cbcs)
    assert stats["pct_cbc_filter_suspicious_reads"] == pytest.approx(expected_pct_reads)
    assert stats["pct_failed_cbcs_above_threshold"] == pytest.approx(expected_pct_cbcs)
    assert stats["pct_cbc_filter_failed_reads"] == pytest.approx(expected_pct_reads)
