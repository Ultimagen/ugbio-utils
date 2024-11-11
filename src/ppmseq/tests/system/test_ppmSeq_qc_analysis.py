# noqa: N999
from os.path import join as pjoin
from pathlib import Path

import pytest
from ugbio_ppmseq import ppmSeq_qc_analysis


@pytest.fixture
def resources_dir():
    return Path(__file__).parent.parent / "resources"


def test_ppmseq_analysis_ppmseq_legacy_v5(tmpdir, resources_dir):
    trimmer_histogram = pjoin(
        resources_dir,
        "A_hmer_start.T_hmer_start.A_hmer_end.T_hmer_end.native_adapter_with_leading_C.histogram.csv",
    )
    trimmer_failure_codes = pjoin(
        resources_dir,
        "ppmSeq.healthy.022782-Lb_2146-UGAv3-168.failure_codes.csv",
    )
    sorter_csv = pjoin(resources_dir, "ppmSeq.healthy.022782-Lb_2146-UGAv3-168.csv")
    sorter_json = pjoin(resources_dir, "ppmSeq.healthy.022782-Lb_2146-UGAv3-168.json")

    ppmSeq_qc_analysis.run(
        [
            "ppmSeq_qc_analysis",
            "--adapter-version",
            "legacy_v5",
            "--trimmer-histogram-csv",
            trimmer_histogram,
            "--trimmer-failure-codes-csv",
            trimmer_failure_codes,
            "--sorter-stats-csv",
            sorter_csv,
            "--sorter-stats-json",
            sorter_json,
            "--output-path",
            tmpdir.dirname,
            "--output-basename",
            "ppmSeq.healthy.022782-Lb_2146-UGAv3-168",
            "--legacy-histogram-column-names",
        ]
    )


def test_ppmseq_analysis_ppmseq_v1(tmpdir, resources_dir):
    trimmer_histogram = pjoin(
        resources_dir,
        "037239-CgD1502_Cord_Blood-Z0032-CTCTGTATTGCAGAT."
        "Start_loop.Start_loop.End_loop.End_loop.native_adapter.histogram.csv",
    )
    trimmer_failure_codes = pjoin(
        resources_dir,
        "037239-CgD1502_Cord_Blood-Z0032-CTCTGTATTGCAGAT.failure_codes.csv",
    )
    sorter_csv = pjoin(resources_dir, "037239-CgD1502_Cord_Blood-Z0032-CTCTGTATTGCAGAT.csv")
    sorter_json = pjoin(resources_dir, "037239-CgD1502_Cord_Blood-Z0032-CTCTGTATTGCAGAT.json")

    ppmSeq_qc_analysis.run(
        [
            "ppmSeq_qc_analysis",
            "--adapter-version",
            "v1",
            "--trimmer-histogram-csv",
            trimmer_histogram,
            "--trimmer-failure-codes-csv",
            trimmer_failure_codes,
            "--sorter-stats-csv",
            sorter_csv,
            "--sorter-stats-json",
            sorter_json,
            "--output-path",
            tmpdir.dirname,
            "--output-basename",
            "037239-CgD1502_Cord_Blood-Z0032",
            "--legacy-histogram-column-names",
        ]
    )


def test_ppmseq_analysis_ppmseq_post_native_adapter_trimming(tmpdir, resources_dir):
    this_resource_dir = resources_dir / "409271-UGAv3-377_post_native_adapter_trimming"
    trimmer_histogram = (
        this_resource_dir
        / "Start_loop_name.Start_loop_pattern_fw.End_loop_name.End_loop_pattern_fw.Stem_end_length.histogram.csv"
    )

    trimmer_failure_codes = this_resource_dir / "409271-UGAv3-377-CAGAATACATGCGAT_CR0-244.failure_codes.csv"
    sorter_csv = this_resource_dir / "409271-UGAv3-377-CAGAATACATGCGAT_CR0-244.csv"
    sorter_json = this_resource_dir / "409271-UGAv3-377-CAGAATACATGCGAT_CR0-244.json"

    ppmSeq_qc_analysis.run(
        [
            "ppmSeq_qc_analysis",
            "--adapter-version",
            "v1",
            "--trimmer-histogram-csv",
            trimmer_histogram,
            "--trimmer-failure-codes-csv",
            trimmer_failure_codes,
            "--sorter-stats-csv",
            sorter_csv,
            "--sorter-stats-json",
            sorter_json,
            "--output-path",
            tmpdir.dirname,
            "--output-basename",
            "409271-UGAv3-377-CAGAATACATGCGAT_CR0-244",
        ]
    )
