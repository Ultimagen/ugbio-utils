import json
import os
import subprocess
from os.path import join as pjoin
from pathlib import Path

import pandas as pd
import pytest
from ugbio_core.vcfbed import vcftools


@pytest.fixture
def resources_dir():
    return Path(__file__).parent.parent.parent / "resources"


def test_snp_bed_files_output(resources_dir):
    # snp_fp testing
    data = pd.read_hdf(pjoin(resources_dir, "BC10.chr1.h5"), key="concordance")
    snp_fp = vcftools.FilterWrapper(data).get_snp().get_fp().get_df()
    assert not snp_fp["indel"].any()
    assert all(x == "fp" for x in snp_fp["classify"])

    # snp_fn testing
    snp_fn = vcftools.FilterWrapper(data).get_snp().get_fn().get_df()
    assert not snp_fn["indel"].any()
    assert all(
        row["classify"] == "fn"
        or (row["classify"] == "tp" and (row["filter"] == "LOW_SCORE") and (row["filter"] != "PASS"))
        for index, row in snp_fn.iterrows()
    )


def test_hmer_bed_files_output(resources_dir):
    data = pd.read_hdf(pjoin(resources_dir, "BC10.chr1.h5"), key="concordance")
    # hmer
    hmer_fn = vcftools.FilterWrapper(data).get_h_mer().get_df()
    assert all(hmer_fn["indel"])
    assert all(x > 0 for x in hmer_fn["hmer_indel_length"])
    hmer_fn = vcftools.FilterWrapper(data).get_h_mer(val_start=1, val_end=1).get_df()
    assert all(hmer_fn["indel"])
    assert all(x == 1 for x in hmer_fn["hmer_indel_length"])
    hmer_fn = vcftools.FilterWrapper(data).get_h_mer(val_start=3, val_end=10).get_df()
    assert all(hmer_fn["indel"])
    assert all(x >= 3 and x <= 10 for x in hmer_fn["hmer_indel_length"])

    # hmer_fp testing
    hmer_fp = vcftools.FilterWrapper(data).get_h_mer().get_fp().get_df()
    assert all(hmer_fp["indel"])
    assert all(x > 0 for x in hmer_fp["hmer_indel_length"])
    assert all(x == "fp" for x in hmer_fp["classify"])

    # hmer_fn testing
    hmer_fn = vcftools.FilterWrapper(data).get_h_mer().get_fn().get_df()
    assert all(hmer_fn["indel"])
    assert all(x > 0 for x in hmer_fn["hmer_indel_length"])
    assert all(
        row["classify"] == "fn"
        or (row["classify"] == "tp" and (row["filter"] == "LOW_SCORE") and (row["filter"] != "PASS"))
        for index, row in hmer_fn.iterrows()
    )


def test_non_hmer_bed_files_output(resources_dir):
    data = pd.read_hdf(pjoin(resources_dir, "BC10.chr1.h5"), key="concordance")
    # non_hmer_fp testing
    non_hmer_fp = vcftools.FilterWrapper(data).get_non_h_mer().get_fp().get_df()
    assert all(non_hmer_fp["indel"])
    assert (non_hmer_fp["hmer_indel_length"] == 0).all()
    assert all(x == "fp" for x in non_hmer_fp["classify"])

    # non_hmer_fn testing
    non_hmer_fn = vcftools.FilterWrapper(data).get_non_h_mer().get_fn().get_df()
    assert all(non_hmer_fn["indel"])
    assert (non_hmer_fn["hmer_indel_length"] == 0).all()
    assert all(
        row["classify"] == "fn"
        or (row["classify"] == "tp" and (row["filter"] == "LOW_SCORE") and (row["filter"] != "PASS"))
        for index, row in non_hmer_fn.iterrows()
    )


def test_bed_output_when_no_tree_score(
    resources_dir,
):  # testing the case when there is no tree_score and there is blacklist
    data = pd.read_hdf(pjoin(resources_dir, "exome.h5"), key="concordance")
    df_data = vcftools.FilterWrapper(data)
    result = dict(df_data.get_fn().bed_format(kind="fn").get_df()["itemRgb"].value_counts())
    expected_result = {
        vcftools.FilteringColors.BLACKLIST.value: 169,
        vcftools.FilteringColors.CLEAR.value: 89,
        vcftools.FilteringColors.BORDERLINE.value: 39,
    }
    for k in result:
        assert result[k] == expected_result[k]

    df_data = vcftools.FilterWrapper(data)
    # since there is no tree_score all false positives should be the same color
    result = dict(df_data.get_fp().bed_format(kind="fp").get_df()["itemRgb"].value_counts())

    assert len(result.keys()) == 1


def test_get_region_around_variant():
    vpos = 100
    vlocs = []
    assert vcftools.get_region_around_variant(vpos, vlocs, 10) == (95, 105)


class TestGetVcfDf:
    def test_get_vcf_df(self, resources_dir):
        input_vcf = pjoin(resources_dir, "test_get_vcf_df.vcf.gz")
        df_vcf = vcftools.get_vcf_df(input_vcf)
        non_nan_columns = list(df_vcf.dropna(axis=1, how="all").columns)
        non_nan_columns.sort()
        assert non_nan_columns == [
            "ac",
            "ad",
            "af",
            "alleles",
            "an",
            "baseqranksum",
            "chrom",
            "db",
            "dp",
            "excesshet",
            "filter",
            "filtered_haps",
            "fs",
            "gnomad_af",
            "gq",
            "gt",
            "hapcomp",
            "hapdom",
            "hec",
            "id",
            "indel",
            "mleac",
            "mleaf",
            "mq",
            "mqranksum",
            "pl",
            "pos",
            "qd",
            "qual",
            "readposranksum",
            "ref",
            "sb",
            "sor",
            "tree_score",
            "variant_type",
            "x_css",
            "x_gcc",
            "x_ic",
            "x_il",
            "x_lm",
            "x_rm",
        ]

    def test_get_vcf_df_use_qual(self, resources_dir):
        input_vcf = pjoin(resources_dir, "test_get_vcf_df.vcf.gz")
        df_vcf = vcftools.get_vcf_df(input_vcf, scoring_field="QUAL")
        assert all(df_vcf["qual"] == df_vcf["tree_score"])

    def test_get_vcf_df_ignore_fields(self, resources_dir):
        input_vcf = pjoin(resources_dir, "test_get_vcf_df.vcf.gz")
        ignore_fields = ["x_css", "x_gcc", "x_ic", "x_il", "x_lm", "x_rm"]
        df_vcf = vcftools.get_vcf_df(input_vcf, ignore_fields=ignore_fields)
        non_nan_columns = list(df_vcf.dropna(axis=1, how="all").columns)
        non_nan_columns.sort()
        assert non_nan_columns == [
            "ac",
            "ad",
            "af",
            "alleles",
            "an",
            "baseqranksum",
            "chrom",
            "db",
            "dp",
            "excesshet",
            "filter",
            "filtered_haps",
            "fs",
            "gnomad_af",
            "gq",
            "gt",
            "hapcomp",
            "hapdom",
            "hec",
            "id",
            "indel",
            "mleac",
            "mleaf",
            "mq",
            "mqranksum",
            "pl",
            "pos",
            "qd",
            "qual",
            "readposranksum",
            "ref",
            "sb",
            "sor",
            "tree_score",
            "variant_type",
        ]
        for x in ignore_fields:
            assert x not in df_vcf.columns


class TestReplaceDataInSpecificChromosomes:
    def test_chr1_chr10(self, tmpdir, resources_dir):
        os.makedirs(tmpdir, exist_ok=True)
        input_json = os.path.join(tmpdir, "input.json")
        with open(input_json, "w") as f:
            json.dump(
                {
                    "chr1": os.path.join(resources_dir, "chr1.vcf.gz"),
                    "chr10": os.path.join(resources_dir, "chr10.vcf.gz"),
                },
                f,
            )
        input_vcf = os.path.join(resources_dir, "few_contigs.vcf.gz")
        header = os.path.join(resources_dir, "header.txt")
        output_vcf = os.path.join(tmpdir, "replaced.vcf.gz")
        vcftools.replace_data_in_specific_chromosomes(input_vcf, input_json, header, output_vcf, tmpdir)
        vcounts = []
        bcftools_out = subprocess.check_output(f"bcftools index -s {output_vcf}", shell=True)
        bcftools_lines = bcftools_out.decode().strip().split("\n")
        for line in bcftools_lines:
            columns = line.split("\t")
            vcounts.append(int(columns[2]))
        assert vcounts == [116, 227, 523]

    def test_chr10_chr11(self, tmpdir, resources_dir):
        os.makedirs(tmpdir, exist_ok=True)
        input_json = os.path.join(tmpdir, "input.json")
        with open(input_json, "w") as f:
            json.dump(
                {
                    "chr11": os.path.join(resources_dir, "chr11.vcf.gz"),
                    "chr10": os.path.join(resources_dir, "chr10.vcf.gz"),
                },
                f,
            )
        input_vcf = os.path.join(resources_dir, "few_contigs.vcf.gz")
        header = os.path.join(resources_dir, "header.txt")
        output_vcf = os.path.join(tmpdir, "replaced.vcf.gz")
        vcftools.replace_data_in_specific_chromosomes(input_vcf, input_json, header, output_vcf, tmpdir)
        vcounts = []
        bcftools_out = subprocess.check_output(f"bcftools index -s {output_vcf}", shell=True)
        bcftools_lines = bcftools_out.decode().strip().split("\n")
        for line in bcftools_lines:
            columns = line.split("\t")
            vcounts.append(int(columns[2]))
        assert vcounts == [653, 227, 523, 1147]


# TODO: tests for subsample_to_alleles and header_record_number
