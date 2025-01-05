from pathlib import Path

import pytest
from ugbio_methylation.generate_methylation_report import generate_methylation_report
from ugbio_methylation.globals import MethylDackelConcatenationCsvs


@pytest.fixture
def resources_dir():
    return Path(__file__).parent.parent / "resources"


@pytest.fixture
def output_path(tmpdir):
    return Path(tmpdir)


def test_generate_methylation_report(output_path, resources_dir):
    # input_h5_file = resources_dir / "input_for_html_report.h5"
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
