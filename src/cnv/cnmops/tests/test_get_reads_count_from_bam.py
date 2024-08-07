import subprocess
from os.path import join as pjoin
from pathlib import Path

import numpy as np
import pandas as pd

from . import get_resource_dir

resources_dir = get_resource_dir(__file__)
base_path = Path(__file__).resolve().parent.parent
script_path = base_path / "get_reads_count_from_bam.R"


def test_get_reads_count_from_bam(tmpdir):
    in_bam_file = pjoin(resources_dir, "test.bam")
    expected_out_file = pjoin(resources_dir, "test.ReadCounts.csv")
    out_prefix = pjoin(tmpdir, "out_test")
    out_file = pjoin(tmpdir, "out_test.ReadCounts.csv")

    cmd = [
        "Rscript",
        "--vanilla",
        script_path,
        "-i",
        in_bam_file,
        "-refseq",
        "chr1",
        "-wl",
        "1000",
        "-p",
        "1",
        "-o",
        out_prefix,
        "--save_csv"
    ]
    assert subprocess.check_call(cmd, cwd=tmpdir) == 0
    df = pd.read_csv(out_file)
    df_ref = pd.read_csv(expected_out_file)
    assert np.allclose(df.iloc[:, -1], df_ref.iloc[:, -1])
