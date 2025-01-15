#!/env/python
# Copyright 2022 Ultima Genomics Inc.
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.
# DESCRIPTION
#    Combine Csv files produced from MethylDackel output files
# ==========================================

# Copyright (c) 2019 Devon Ryan and contributors
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# ==========================================
import argparse
import sys
from pathlib import Path

import pandas as pd
from ugbio_core.logger import logger

from ugbio_methylation.globals import H5_FILE, MethylDackelConcatenationCsvs


def parse_args(argv: list[str]) -> tuple[MethylDackelConcatenationCsvs, str]:
    ap_var = argparse.ArgumentParser(
        prog="concat_methyldackel_csvs.py",
        description=run.__doc__,
    )
    ap_var.add_argument("--mbias", help="csv summary of MethylDackelMbias", type=str, required=True)
    ap_var.add_argument(
        "--mbias_non_cpg", help="csv summary of MethylDackelMbias in the non-CpG mode", type=str, required=True
    )
    ap_var.add_argument("--merge_context", help="csv summary of MethylDackelMergeContext", type=str, required=True)
    ap_var.add_argument(
        "--merge_context_non_cpg",
        help="csv summary of MethylDackelMergeContext in the non-CpG mode",
        type=str,
        required=True,
    )
    ap_var.add_argument("--per_read", help="csv summary of MethylDackelPerRead", type=str, required=False, default=None)
    ap_var.add_argument("--output", help="Output file basename", type=str, required=True)

    args = ap_var.parse_args(argv[1:])
    methyl_dackel_concatenation_csvs = MethylDackelConcatenationCsvs(
        mbias=args.mbias,
        mbias_non_cpg=args.mbias_non_cpg,
        merge_context=args.merge_context,
        merge_context_non_cpg=args.merge_context_non_cpg,
        per_read=args.per_read,
    )
    return methyl_dackel_concatenation_csvs, args.output


def split_position_hist_desc(df):
    df_per_position = df.loc[df["metric"].str.startswith("PercentMethylationPosition")].copy()
    df_hist = df.loc[df["metric"].str.contains("PercentMethylation_[0-9]+|Coverage_[0-9]+")].copy()
    df_desc = df.loc[~df["metric"].str.contains("_[0-9]+")].copy()
    return df_per_position, df_hist, df_desc


def concat_methyldackel_csvs(
    methyl_dackel_concatenation_csvs: MethylDackelConcatenationCsvs, output: str, output_prefix: Path = None
) -> Path:
    "Combine csvs from POST-MethylDackel processing"
    h5_output = output + H5_FILE
    if output_prefix:
        h5_output = Path(output_prefix) / h5_output

    logger.info(f"Concatenating MethylDackel CSVs {methyl_dackel_concatenation_csvs=} into {h5_output=}")

    with pd.HDFStore(h5_output, mode="w") as store:
        for table, input_file in methyl_dackel_concatenation_csvs.iterate_fields():
            if table == "per_read" and input_file is None:
                continue
            input_df = pd.read_csv(input_file)
            df_per_position, df_hist, df_desc = split_position_hist_desc(input_df)
            tables_to_take = {"per_position": df_per_position, "hist": df_hist, "desc": df_desc}
            for table_ext, tbl_df in tables_to_take.items():
                tbl_df = tbl_df.set_index(["detail", "metric"])  # noqa: PLW2901
                tbl_df = tbl_df.squeeze(axis=1)  # noqa: PLW2901
                table_name = f"{table}_{table_ext}"
                store.put(table_name, tbl_df, format="table", data_columns=True)

        keys_to_convert = methyl_dackel_concatenation_csvs.get_keys_to_convert()
        store.put("keys_to_convert", pd.Series(keys_to_convert))

        # stats for nexus
        df_merge_context = pd.read_csv(methyl_dackel_concatenation_csvs.merge_context)
        df_merge_context_non_cpg = pd.read_csv(methyl_dackel_concatenation_csvs.merge_context_non_cpg)
        tbl_df = pd.concat(
            [
                df_merge_context.query('metric == "PercentMethylation_mean"'),
                df_merge_context_non_cpg.query('metric == "PercentMethylation_mean"'),
            ]
        )
        tbl_df["key"] = (
            tbl_df["metric"].str.replace("PercentMethylation_mean", "PCT_methylation_mean") + "_" + tbl_df["detail"]
        )
        tbl_df = tbl_df.drop(columns=["detail", "metric"])
        tbl_df = tbl_df.set_index("key")
        tbl_df = tbl_df.squeeze(axis=1)
        store.put("stats_for_nexus", tbl_df)

    logger.info(f"Finished concatenating MethylDackel CSVs to {h5_output=}")
    return Path(h5_output)


def run(argv: list[str] | None = None):
    """Concatenate CSV output files of MethylDackel processing into an HDF5 file"""
    if argv is None:
        argv: list[str] = sys.argv

    methyl_dackel_concatenation_csvs, output = parse_args(argv)
    concat_methyldackel_csvs(methyl_dackel_concatenation_csvs, output)


if __name__ == "__main__":
    run()
