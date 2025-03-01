import shutil
from gzip import BadGzipFile
from pathlib import Path

import pandas as pd
import pytest
from ugbio_single_cell.collect_statistics import (
    collect_statistics,
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
        assert len(s) == 19
        assert s["pct_failed_cbcs_above_threshold"] == 100 * 0.02311600563934775
        assert s["pct_cbc_filter_failed_reads"] == 100 * 0.0018177121169518373
