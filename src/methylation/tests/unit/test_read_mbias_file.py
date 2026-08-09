from pathlib import Path

import pandas as pd
import pytest
from ugbio_methylation.methyldackel_utils import read_mbias_file

MBIAS_TABLE = (
    "Strand\tRead\tPosition\tnMethylated\tnUnmethylated\n"
    "OT\t1\t11\t46447\t48535\n"
    "OT\t1\t12\t46181\t48090\n"
    "OB\t1\t11\t45012\t47733\n"
)

# MethylDackel writes this to stderr for every bounds field that is 0 (a stale-errno quirk in its parseBounds).
# The APL pipeline captures the container's combined stdout+stderr into the mbias table file, so these lines
# arrive ahead of the header.
STDERR_NOISE = "Invalid bounds string, 10,10,0,0\nInvalid bounds string, 10,10,0,0\n"


@pytest.fixture
def resources_dir():
    return Path(__file__).parent.parent / "resources"


def test_read_mbias_file_clean_table(tmp_path):
    in_file = tmp_path / "clean_mbias.tsv"
    in_file.write_text(MBIAS_TABLE, encoding="utf-8")

    df_mbias = read_mbias_file(str(in_file))

    assert list(df_mbias.columns) == ["Strand", "Read", "Position", "nMethylated", "nUnmethylated"]
    assert len(df_mbias) == 3


def test_read_mbias_file_skips_leading_stderr(tmp_path, caplog):
    in_file = tmp_path / "polluted_mbias.tsv"
    in_file.write_text(STDERR_NOISE + MBIAS_TABLE, encoding="utf-8")

    df_mbias = read_mbias_file(str(in_file))

    clean_file = tmp_path / "clean_mbias.tsv"
    clean_file.write_text(MBIAS_TABLE, encoding="utf-8")
    pd.testing.assert_frame_equal(df_mbias, read_mbias_file(str(clean_file)))
    assert "Skipped 2 non-table line(s)" in caplog.text


def test_read_mbias_file_no_header_raises(tmp_path):
    in_file = tmp_path / "no_header_mbias.tsv"
    in_file.write_text(STDERR_NOISE, encoding="utf-8")

    with pytest.raises(ValueError, match="No 'Strand' header line found"):
        read_mbias_file(str(in_file))


def test_read_mbias_file_matches_plain_read_csv_on_reference_input(resources_dir):
    """The clean-input path must stay byte-for-byte equivalent to the pd.read_csv call it replaced."""
    in_file = f"{resources_dir}/input_Mbias.bedGraph"

    pd.testing.assert_frame_equal(read_mbias_file(in_file), pd.read_csv(in_file, sep="\t"))
