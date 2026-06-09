from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd
import pytest
from ugbio_methylation.generate_methylation_report import generate_methylation_report
from ugbio_methylation.globals import MethylDackelConcatenationCsvs
from ugbio_methylation.qc_report_generator import ControlGenomeSection, DataProcessor, ReportConfig


@pytest.fixture
def resources_dir():
    return Path(__file__).parent.parent / "resources"


@pytest.fixture
def output_path(tmpdir):
    return Path(tmpdir)


def test_generate_methylation_report(output_path, resources_dir):
    methyl_dackel_concatenation_csvs = MethylDackelConcatenationCsvs(
        mbias=resources_dir / "ProcessMethylDackelMbias.csv",
        mbias_non_cpg=resources_dir / "ProcessMethylDackelMbiasNoCpG.csv",
        merge_context=resources_dir / "ProcessConcatMethylDackelMergeContext.csv",
        merge_context_non_cpg=resources_dir / "ProcessMethylDackelMergeContextNoCpG.csv",
        per_read=resources_dir / "ProcessMethylDackelPerRead.csv",
    )
    base_file_name = "test"

    output_report_html = generate_methylation_report(
        methyl_dackel_concatenation_csvs, base_file_name, output_prefix=output_path
    )

    # assert report_html exists
    assert output_report_html.exists()

    # assert report_html is not empty
    assert output_report_html.stat().st_size > 0


@pytest.fixture
def make_control_genome_section():
    def _make(h5_path):
        data_processor = DataProcessor(str(h5_path))
        return ControlGenomeSection(data_processor, MagicMock(), MagicMock(), MagicMock())

    return _make


def test_check_control_genomes_exist_returns_false_when_table_missing(tmp_path, make_control_genome_section):
    """When the reference has no control genomes (Lambda/pUC19), the
    merge_context_per_position table is not written to the HDF5 file.
    check_control_genomes_exist should return False instead of raising KeyError."""
    h5_path = tmp_path / "test.h5"
    with pd.HDFStore(str(h5_path), mode="w") as store:
        store.put("merge_context_desc", pd.Series(["dummy"], name="value"))

    section = make_control_genome_section(h5_path)
    config = ReportConfig(input_h5_file=str(h5_path), input_base_file_name="test")

    result = section.check_control_genomes_exist(config.control_genomes)
    assert result is False


def test_check_control_genomes_exist_returns_false_when_genomes_not_present(tmp_path, make_control_genome_section):
    """When the table exists but doesn't contain the expected control genomes,
    check_control_genomes_exist should return False."""
    h5_path = tmp_path / "test.h5"
    per_position = pd.DataFrame({"detail": ["hg", "hg"], "metric": ["m1", "m2"], "value": [1.0, 2.0]})
    per_position = per_position.set_index(["detail", "metric"]).squeeze(axis=1)
    with pd.HDFStore(str(h5_path), mode="w") as store:
        store.put("merge_context_per_position", per_position, format="table", data_columns=True)

    section = make_control_genome_section(h5_path)
    config = ReportConfig(input_h5_file=str(h5_path), input_base_file_name="test")

    result = section.check_control_genomes_exist(config.control_genomes)
    assert result is False


def test_check_control_genomes_exist_returns_true_when_present(tmp_path, make_control_genome_section):
    """When control genomes are present in the table, should return True."""
    h5_path = tmp_path / "test.h5"
    per_position = pd.DataFrame(
        {
            "detail": ["Lambda", "pUC19", "hg"],
            "metric": ["m1", "m2", "m3"],
            "value": [1.0, 2.0, 3.0],
        }
    )
    per_position = per_position.set_index(["detail", "metric"]).squeeze(axis=1)
    with pd.HDFStore(str(h5_path), mode="w") as store:
        store.put("merge_context_per_position", per_position, format="table", data_columns=True)

    section = make_control_genome_section(h5_path)
    config = ReportConfig(input_h5_file=str(h5_path), input_base_file_name="test")

    result = section.check_control_genomes_exist(config.control_genomes)
    assert result is True
