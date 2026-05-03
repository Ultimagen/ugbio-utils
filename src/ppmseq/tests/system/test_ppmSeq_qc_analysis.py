# noqa: N999
from pathlib import Path

import pytest
from ugbio_ppmseq import ppmSeq_qc_analysis


@pytest.fixture
def resources_dir():
    return Path(__file__).parent.parent / "resources"


@pytest.fixture
def subsampled_sam(resources_dir):
    return resources_dir / "ppmseq_sr_tag" / "Z0263_sample.sam.gz"


@pytest.mark.parametrize("adapter_version", ["v1", "legacy_v5"])
def test_ppmseq_qc_analysis_runs(tmp_path, subsampled_sam, adapter_version):
    ppmSeq_qc_analysis.run(
        [
            "ppmSeq_qc_analysis",
            "--adapter-version",
            adapter_version,
            "--subsampled-sam",
            str(subsampled_sam),
            "--output-path",
            str(tmp_path),
            "--output-basename",
            f"ppmseq_sr_tag_{adapter_version}",
            "--generate-report",
            "",
        ]
    )
    h5 = tmp_path / f"ppmseq_sr_tag_{adapter_version}.ppmSeq.applicationQC.h5"
    assert h5.exists()
    json_out = tmp_path / f"ppmseq_sr_tag_{adapter_version}.ppmSeq.applicationQC.json"
    assert json_out.exists()
