import filecmp
from os.path import join as pjoin
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from ugbio_core.coverage_analysis import run_coverage_collection, run_full_coverage_analysis


@pytest.fixture
def resources_dir():
    return Path(__file__).parent.parent / "resources"


def test_coverage_analysis(tmpdir, resources_dir):
    f_in = resources_dir / "170201-BC23.chr9_1000000_2001000.aligned.unsorted.duplicates_marked.bam"
    f_ref = resources_dir / "170201-BC23.chr9_1000000_2001000.coverage_percentiles.parquet"
    ref_fasta = resources_dir / "sample.fasta"
    run_full_coverage_analysis(
        bam_file=f_in,
        out_path=tmpdir,
        regions=["chr9:1000000-2001000"],
        windows=[100_000],
        ref_fasta=ref_fasta,
        coverage_intervals_dict=resources_dir / "coverage_chr9_extended_intervals.tsv",
    )
    df = pd.read_hdf(pjoin(tmpdir, "170201-BC23.coverage_stats.q0.Q0.l0.h5"), "percentiles")  # noqa PD901
    df_ref = pd.read_parquet(f_ref)
    assert np.allclose(df.fillna(-1), df_ref.fillna(-1))


def test_coverage_collection(tmpdir, resources_dir):
    f_in = resources_dir / "170201-BC23.chr9_1000000_2001000.aligned.unsorted.duplicates_marked.bam"
    bg_ref = resources_dir / "170201-BC23.chr9_1000000_2001000.bedgraph"
    bw_ref = resources_dir / "170201-BC23.chr9_1000000_2001000.bw"
    ref_fasta = resources_dir / "sample.fasta"

    run_coverage_collection(
        bam_file=f_in,
        out_path=tmpdir,
        ref_fasta=ref_fasta,
        regions=["chr9:1000000-2001000"],
        zip_bg=False,
    )
    assert filecmp.cmp(
        pjoin(tmpdir, "170201-BC23.chr9_1000000-2001000.q0.Q0.l0.w1.depth.bedgraph"), bg_ref, shallow=False
    )
    assert filecmp.cmp(pjoin(tmpdir, "170201-BC23.chr9_1000000-2001000.q0.Q0.l0.w1.depth.bw"), bw_ref, shallow=False)
