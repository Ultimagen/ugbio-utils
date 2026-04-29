import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from ugbio_methylation import (
    process_mbias,
    process_merge_context,
    process_merge_context_no_cp_g,
    process_per_read,
)
from ugbio_methylation.concat_methyldackel_csvs import run as concat_methyldackestol_csvs_run
from ugbio_methylation.globals import H5_FILE
from ugbio_methylation.methyldackel_utils import (
    calc_coverage_methylation,
    calc_percent_methylation,
    calc_total_cp_gs,
    get_dict_from_dataframe,
)


@pytest.fixture
def resources_dir():
    return Path(__file__).parent.parent / "resources"


class TestParsers:
    def test_process_mbias(self, tmpdir, resources_dir):
        output_prefix = f"{tmpdir}/output_Mbias"
        output_file = output_prefix + ".csv"
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        process_mbias.run(
            [
                "process_Mbias",
                "--input",
                f"{resources_dir}/input_Mbias.bedGraph",
                "--output",
                f"{output_prefix}",
            ]
        )

        result_csv = pd.read_csv(output_file)
        ref_csv = pd.read_csv(open(f"{resources_dir}/ProcessMethylDackelMbias.csv"))

        pd.testing.assert_frame_equal(result_csv, ref_csv)

    # ------------------------------------------------------

    def test_process_per_read(self, tmpdir, resources_dir):
        output_prefix = f"{tmpdir}/output_perRead"
        output_file = output_prefix + ".csv"
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        process_per_read.run(
            [
                "process_perRead",
                "--input",
                f"{resources_dir}/input_perRead.bedGraph",
                "--output",
                f"{output_prefix}",
            ]
        )

        result_csv = pd.read_csv(output_file)
        ref_csv = pd.read_csv(open(f"{resources_dir}/ProcessMethylDackelPerRead.csv"))

        pd.testing.assert_frame_equal(result_csv, ref_csv)

    # ------------------------------------------------------

    def test_process_merge_context_no_cp_g(self, tmpdir, resources_dir):
        output_prefix = f"{tmpdir}/output_mergeContextNoCpG"
        output_file = output_prefix + ".csv"
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        process_merge_context_no_cp_g.run(
            [
                "process_mergeContextNoCpG",
                "--input_chg",
                f"{resources_dir}/input_mergeContextNoCpG_CHG.bedGraph",
                "--input_chh",
                f"{resources_dir}/input_mergeContextNoCpG_CHH.bedGraph",
                "--output",
                f"{output_prefix}",
            ]
        )

        result_csv = pd.read_csv(output_file)
        ref_csv = pd.read_csv(open(f"{resources_dir}/ProcessMethylDackelMergeContextNoCpG.csv"))

        pd.testing.assert_frame_equal(result_csv, ref_csv)

    # ------------------------------------------------------

    def test_process_merge_context(self, tmpdir, resources_dir):
        output_prefix = f"{tmpdir}/output_mergeContext"
        output_file = output_prefix + ".csv"
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        process_merge_context.run(
            [
                "process_mergeContext",
                "--input",
                f"{resources_dir}/input_mergeContext.bedGraph",
                "--output",
                f"{output_prefix}",
            ]
        )

        result_csv = pd.read_csv(output_file)
        pat = r"^hg"
        idx = result_csv.detail.str.contains(pat)
        if idx.any(axis=None):
            result_csv_sub = result_csv.loc[idx, :].copy()

        ref_csv = pd.read_csv(open(f"{resources_dir}/ProcessConcatMethylDackelMergeContext.csv"))
        idx = ref_csv.detail.str.contains(pat)
        if idx.any(axis=None):
            ref_csv_sub = ref_csv.loc[idx, :].copy()

        pat = r"^PercentMethylation_"
        idx = result_csv_sub.metric.str.contains(pat)
        if idx.any(axis=None):
            result_output = result_csv_sub.loc[idx, :].copy()

        idx = ref_csv_sub.metric.str.contains(pat)
        if idx.any(axis=None):
            ref_output = ref_csv_sub.loc[idx, :].copy()

        assert np.all(np.sum(result_output.value) == np.sum(ref_output.value))

    # ------------------------------------------------------

    def test_concat_methyldackel_csvs(self, tmpdir, resources_dir):
        output_prefix = f"{tmpdir}/concat_methyldackel_csvs"
        output_h5_file = output_prefix + H5_FILE
        os.makedirs(tmpdir, exist_ok=True)

        concat_methyldackestol_csvs_run(
            [
                "concat_methyldackel_csvs",
                "--mbias",
                f"{resources_dir}/ProcessMethylDackelMbias.csv",
                "--mbias_non_cpg",
                f"{resources_dir}/ProcessMethylDackelMbiasNoCpG.csv",
                "--merge_context",
                f"{resources_dir}/ProcessConcatMethylDackelMergeContext.csv",
                "--merge_context_non_cpg",
                f"{resources_dir}/ProcessMethylDackelMergeContextNoCpG.csv",
                "--per_read",
                f"{resources_dir}/ProcessMethylDackelPerRead.csv",
                "--output",
                f"{output_prefix}",
            ]
        )
        input_files = [
            "ProcessMethylDackelMbias.csv",
            "ProcessMethylDackelMbiasNoCpG.csv",
            "ProcessConcatMethylDackelMergeContext.csv",
            "ProcessMethylDackelMergeContextNoCpG.csv",
            "ProcessMethylDackelPerRead.csv",
        ]

        df_result = pd.DataFrame()
        with pd.HDFStore(output_h5_file, "r") as store:
            for key in store.keys():
                if (key == "/keys_to_convert") or (key == "/stats_for_nexus"):
                    continue
                df = pd.DataFrame(store[key])  # noqa: PD901
                df = df.reset_index()  # noqa: PD901
                df_result = pd.concat((df_result, df))

        df_ref = pd.concat(
            pd.read_csv(f"{resources_dir}/{value}", dtype={"metric": str, "value": np.float64, "detail": str})
            for value in input_files
        )
        assert np.allclose(np.ceil(np.sum(df_ref["value"])), np.ceil(np.sum(df_result["value"])))

    # ------------------------------------------------------

    def test_methyldackel_utils_calc_percent_methylation(self, tmpdir, resources_dir):
        output_prefix = f"{tmpdir}/methyldackel_utils_pcnt_meth"
        output_file = output_prefix + ".csv"
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        in_file_name = f"{resources_dir}/input_mergeContext.bedGraph"
        col_names = ["chr", "start", "end", "PercentMethylation", "coverage_methylated", "coverage_unmethylated"]
        df_in_report = pd.read_csv(in_file_name, sep="\t", header=0, names=col_names)
        df_in_report["Coverage"] = df_in_report["coverage_methylated"] + df_in_report["coverage_unmethylated"]

        key_word = "hg"
        data_frame = pd.DataFrame()
        pat = r"^chr[0-9]+\b"
        idx = df_in_report.chr.str.contains(pat)
        if idx.any(axis=None):
            data_frame = df_in_report.loc[idx, :].copy()

        result_calc = calc_percent_methylation(key_word, data_frame, rel=False)

        input_file_name = "ProcessConcatMethylDackelMergeContext.csv"
        input_file_name = f"{resources_dir}/" + input_file_name
        input_csv = pd.read_csv(open(input_file_name))

        pat = r"^hg"
        idx = input_csv.detail.str.contains(pat)
        if idx.any(axis=None):
            ref_csv = input_csv.loc[idx, :].copy()

        pat = r"^PercentMethylation|TotalCpGs"
        idx = ref_csv.metric.str.contains(pat)
        if idx.any(axis=None):
            ref_csv_output = ref_csv.loc[idx, :].copy()

        assert np.all(np.sum(ref_csv_output.value) == np.sum(result_calc.value))

    # ------------------------------------------------------

    def test_methyldackel_utils_calc_coverage_methylation(self, tmpdir, resources_dir):
        in_file_name = f"{resources_dir}/input_mergeContextNoCpG_CHG.bedGraph"
        col_names = ["chr", "start", "end", "PercentMethylation", "coverage_methylated", "coverage_unmethylated"]
        df_chg_input = pd.read_csv(in_file_name, sep="\t", header=0, names=col_names)

        # calculate total coverage
        df_chg_input["Coverage"] = df_chg_input.apply(
            lambda x: x["coverage_methylated"] + x["coverage_unmethylated"], axis=1
        )
        # remove non chr1-22 chromosomes
        pat = r"^chr[0-9]+\b"  # remove non
        idx = df_chg_input.chr.str.contains(pat)

        if idx.any(axis=None):
            df_chg = df_chg_input.loc[idx, :].copy()

        result_calc = calc_coverage_methylation("CHG", df_chg, rel=True)

        input_file_name = "ProcessMethylDackelMergeContextNoCpG.csv"
        input_file_name = f"{resources_dir}/" + input_file_name
        input_csv = pd.read_csv(open(input_file_name))
        ref_csv = pd.DataFrame()

        pat = r"CHG"
        idx = input_csv.detail.str.contains(pat)
        if idx.any(axis=None):
            ref_csv = input_csv.loc[idx, :].copy()

        pat = r"Coverage"
        idx = ref_csv.metric.str.contains(pat)
        if idx.any(axis=None):
            ref_csv = ref_csv.loc[idx, :]

        assert np.all(np.sum(result_calc.value) == np.sum(ref_csv.value))

    # ------------------------------------------------------

    def test_methyldackel_utils_total_cpgs(self, tmpdir, resources_dir):
        in_file_name = f"{resources_dir}/input_perRead.bedGraph"

        col_names = ["read_name", "chr", "start", "PercentMethylation", "TotalCpGs"]
        df_per_read = pd.read_csv(in_file_name, sep="\t", header=0, names=col_names)
        df_per_read = df_per_read.drop(columns="read_name")
        df_per_read = df_per_read.dropna()

        df_pcnt_meth = calc_percent_methylation("PercentMethylation", df_per_read, rel=True)
        df_total_cpgs = calc_total_cp_gs("TotalCpGs", df_per_read)
        result_calc = pd.concat([df_pcnt_meth, df_total_cpgs], axis=0, ignore_index=True)

        input_file_name = "ProcessMethylDackelPerRead.csv"
        input_file_name = f"{resources_dir}/" + input_file_name
        ref_csv = pd.read_csv(open(input_file_name))

        assert np.allclose(np.sum(result_calc.value), np.sum(ref_csv.value))

    # ------------------------------------------------------

    def test_methyldackel_utils_get_dict(self, tmpdir, resources_dir):
        input_file_name = "ProcessConcatMethylDackelMergeContext.csv"
        input_file_name = f"{resources_dir}/" + input_file_name
        ref_csv = pd.read_csv(open(input_file_name))

        dict_json_output = {}
        for detail in ref_csv["detail"].unique():
            temp_dict = get_dict_from_dataframe(ref_csv, detail)
            dict_json_output.update(temp_dict)

        calc_json = {"metrics": {}}
        calc_json["metrics"] = {"MergeContext": dict_json_output}

        input_file_name = "ProcessConcatMethylDackelMergeContext.json"
        input_file_name = f"{resources_dir}/" + input_file_name

        ref_json = json.load(open(input_file_name))

        assert len(calc_json["metrics"]["MergeContext"]["hg"]) == len(ref_json["metrics"]["MergeContext"]["hg"])
