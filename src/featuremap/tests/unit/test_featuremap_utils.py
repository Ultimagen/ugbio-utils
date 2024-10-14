import os
from os.path import join as pjoin
from pathlib import Path

import numpy as np
import pandas as pd
import pysam
import pytest
from ugbio_core import variant_annotation
from ugbio_core.consts import DEFAULT_FLOW_ORDER
from ugbio_core.variant_annotation import VcfAnnotator
from ugbio_featuremap.featuremap_utils import (
    FeaturemapAnnotator,
    FeatureMapFields,
    RefContextVcfAnnotator,
    featuremap_to_dataframe,
    filter_featuremap_with_bcftools_view,
)


@pytest.fixture
def resources_dir():
    return Path(__file__).parent.parent / "resources"


@pytest.mark.parametrize(
    "min_coverage, max_coverage, bcftools_include_filter, regions_string, expected_num_variants",
    [
        (None, None, None, "chr20\t85258\t85260\n", 79),
        (10, None, None, "chr20\t85258\t85260\n", 79),
        (None, 100, None, "chr20\t85258\t85260\n", 79),
        (None, None, "(X_SCORE >= 10)", "chr20\t85258\t85260\n", 79),
        (None, 100, "(X_SCORE >= 4) && (X_EDIST <= 5)", "chr20\t85258\t85260\n", 78),
        (None, 100, "(X_SCORE >= 4)", "chr20\t85258\t96322\n", 214),
        (84, 100, "(X_SCORE >= 4)", "chr20\t85258\t96322\n", 79),
    ],
)
def test_filter_featuremap_with_bcftools_view_with_params(
    tmpdir, min_coverage, max_coverage, bcftools_include_filter, regions_string, expected_num_variants, resources_dir
):
    # create input featuremap vcf file
    input_featuremap_vcf = pjoin(resources_dir, "333_LuNgs_08.annotated_featuremap.vcf.gz")

    # create regions file
    regions_file = pjoin(tmpdir, "regions.bed")
    with open(regions_file, "w") as f:
        f.write(regions_string)

    # call the function with different arguments
    intersect_featuremap_vcf = pjoin(tmpdir, "intersect_featuremap.vcf.gz")
    filter_featuremap_with_bcftools_view(
        input_featuremap_vcf=input_featuremap_vcf,
        intersect_featuremap_vcf=intersect_featuremap_vcf,
        min_coverage=min_coverage,
        max_coverage=max_coverage,
        regions_file=regions_file,
        bcftools_include_filter=bcftools_include_filter,
    )

    # check that the output file exists and has the expected content
    assert os.path.isfile(intersect_featuremap_vcf)
    # count the number of variants (excluding the header)
    num_variants = 0
    for _ in pysam.VariantFile(intersect_featuremap_vcf):
        num_variants += 1
    assert num_variants == expected_num_variants


def test_featuremap_to_dataframe(tmpdir, resources_dir):
    input_featuremap = pjoin(resources_dir, "Pa_46.bsDNA.chr20_sample.vcf.gz")
    expected_featuremap_dataframe = pjoin(resources_dir, "Pa_46.bsDNA.chr20_sample.parquet")
    tmp_out_path = pjoin(tmpdir, "tmp_out.parquet")
    featuremap_dataframe = featuremap_to_dataframe(featuremap_vcf=input_featuremap, output_file=tmp_out_path)
    featuremap_dataframe_expected = pd.read_parquet(expected_featuremap_dataframe)
    _assert_read_signature(
        featuremap_dataframe,
        featuremap_dataframe_expected,
        expected_columns=[
            "chrom",
            "pos",
            "ref",
            "alt",
            "qual",
            "filter",
            "MLEAC",
            "MLEAF",
            "X_CIGAR",
            "X_EDIST",
            "X_FC1",
            "X_FC2",
            "X_FILTERED_COUNT",
            "X_FLAGS",
            "X_INDEX",
            "X_LENGTH",
            "X_MAPQ",
            "X_READ_COUNT",
            "X_RN",
            "X_SCORE",
            "X_SMQ_LEFT",
            "X_SMQ_LEFT_MEAN",
            "X_SMQ_RIGHT",
            "X_SMQ_RIGHT_MEAN",
            "a3",
            "ae",
            "as",
            "rq",
            "s2",
            "s3",
            "te",
            "tm",
            "ts",
        ],
        possibly_null_columns=["tm", "filter", "qual", "MLEAC", "MLEAF"],
    )


def _assert_read_signature(signature, expected_signature, expected_columns=None, possibly_null_columns=None):
    expected_columns = expected_columns or [
        "ref",
        "alt",
        "id",
        "qual",
        "af",
    ]
    possibly_null_columns = possibly_null_columns or [
        "id",
        "qual",
    ]
    for c in expected_columns:
        assert c in signature.columns
        if c not in possibly_null_columns:
            assert not signature[c].isnull().all()
            assert (signature[c] == expected_signature[c]).all() or np.allclose(signature[c], expected_signature[c])


def test_featuremap_annotator(tmpdir, resources_dir):
    input_featuremap = pjoin(resources_dir, "Pa_46.bsDNA.chr20_sample.vcf.gz")
    tmpfile = f"{tmpdir}/output_featuremap.vcf.gz"
    VcfAnnotator.process_vcf(
        input_path=input_featuremap,
        output_path=tmpfile,
        annotators=[FeaturemapAnnotator()],
    )
    output_variants = pysam.VariantFile(tmpfile)
    forward_events = 0
    reverse_events = 0
    total_max_softclip_bases = 0
    for v in output_variants:
        assert FeatureMapFields.MAX_SOFTCLIP_LENGTH.value in v.info
        total_max_softclip_bases += int(v.info[FeatureMapFields.MAX_SOFTCLIP_LENGTH.value])
        if FeatureMapFields.MAX_SOFTCLIP_LENGTH.IS_FORWARD.value in v.info:
            forward_events += 1
        else:
            reverse_events += 1
    assert (
        FeatureMapFields.IS_DUPLICATE.value in output_variants.header.info
    ), f"{FeatureMapFields.IS_DUPLICATE.value} is not in info header {output_variants.header.info}"
    assert forward_events == 9
    assert reverse_events == 22
    assert total_max_softclip_bases == 81


def test_ref_context_vcf_annotator_somatic_dv(tmpdir, resources_dir):
    # data files
    sample_vcf = pjoin(resources_dir, "Pa_46.FFPE.chr20_sample.vcf.gz")
    motif_length = 3
    ref_contetxt_variant_annotator = RefContextVcfAnnotator(
        ref_fasta=pjoin(resources_dir, "sample.fasta"),
        flow_order=DEFAULT_FLOW_ORDER,
        motif_length_to_annotate=motif_length,
    )

    # test on vcf file (from somatic DV)
    annotated_somatic_dv_vcf = pjoin(tmpdir, "annotated_somatic_dv.vcf.gz")
    variant_annotation.VcfAnnotator.process_vcf(
        input_path=sample_vcf,
        output_path=annotated_somatic_dv_vcf,
        annotators=[ref_contetxt_variant_annotator],
    )
    __ref_context_annotator_assertion(
        annotated_vcf=annotated_somatic_dv_vcf,
        motif_length=motif_length,
        num_cycle_skips=3,
        num_non_cycle_skips=9,
        hmer_context_ref_sum=27,
        hmer_context_alt_sum=23,
    )


def test_ref_context_vcf_annotator_featuremap_with_bs(tmpdir, resources_dir):
    # test on FeatureMap file (from bsDNA)
    sample_featuremap = pjoin(resources_dir, "Pa_46.bsDNA.chr20_sample.vcf.gz")
    motif_length = 3
    ref_contetxt_variant_annotator = RefContextVcfAnnotator(
        ref_fasta=pjoin(resources_dir, "sample.fasta"),
        flow_order=DEFAULT_FLOW_ORDER,
        motif_length_to_annotate=motif_length,
    )
    annotated_balanced_featuremap = pjoin(tmpdir, "annotated_balanced_featuremap.vcf.gz")
    variant_annotation.VcfAnnotator.process_vcf(
        input_path=sample_featuremap,
        output_path=annotated_balanced_featuremap,
        annotators=[ref_contetxt_variant_annotator],
    )
    __ref_context_annotator_assertion(
        annotated_vcf=annotated_balanced_featuremap,
        motif_length=motif_length,
        num_cycle_skips=0,
        num_non_cycle_skips=31,
        hmer_context_ref_sum=64,
        hmer_context_alt_sum=41,
    )


def test_ref_context_vcf_annotator_multi_processing(tmpdir, resources_dir):
    # test on FeatureMap file, with multi-processing)
    sample_featuremap = pjoin(resources_dir, "Pa_46.bsDNA.chr20_sample.vcf.gz")
    motif_length = 3
    ref_contetxt_variant_annotator = RefContextVcfAnnotator(
        ref_fasta=pjoin(resources_dir, "sample.fasta"),
        flow_order=DEFAULT_FLOW_ORDER,
        motif_length_to_annotate=motif_length,
    )
    annotated_balanced_featuremap = pjoin(tmpdir, "annotated_balanced_featuremap.vcf.gz")
    variant_annotation.VcfAnnotator.process_vcf(
        input_path=sample_featuremap,
        output_path=annotated_balanced_featuremap,
        annotators=[ref_contetxt_variant_annotator],
        process_number=1,
    )
    __ref_context_annotator_assertion(
        annotated_vcf=annotated_balanced_featuremap,
        motif_length=motif_length,
        num_cycle_skips=0,
        num_non_cycle_skips=31,
        hmer_context_ref_sum=64,
        hmer_context_alt_sum=41,
    )


def __ref_context_annotator_assertion(
    annotated_vcf: str,
    motif_length: int,
    num_cycle_skips: int,
    num_non_cycle_skips: int,
    hmer_context_ref_sum: int,
    hmer_context_alt_sum: int,
):
    annotated_variants = pysam.VariantFile(annotated_vcf, "r")
    assert FeatureMapFields.TRINUC_CONTEXT_WITH_ALT.value in annotated_variants.header.info
    assert FeatureMapFields.HMER_CONTEXT_REF.value in annotated_variants.header.info
    assert FeatureMapFields.HMER_CONTEXT_ALT.value in annotated_variants.header.info
    assert FeatureMapFields.IS_CYCLE_SKIP.value in annotated_variants.header.info
    for i in range(motif_length):
        assert f"prev_{i + 1}" in annotated_variants.header.info
        assert f"next_{i + 1}" in annotated_variants.header.info

    cycle_skips = 0
    non_cycle_skips = 0
    hmer_context_ref_sum = 0
    hmer_context_alt_sum = 0
    for variant in annotated_variants:
        assert len(variant.info[FeatureMapFields.TRINUC_CONTEXT_WITH_ALT.value]) == motif_length + 1
        assert len(variant.info[FeatureMapFields.TRINUC_CONTEXT_WITH_ALT.value]) == motif_length + 1
        if variant.info[FeatureMapFields.IS_CYCLE_SKIP.value]:
            cycle_skips += 1
        else:
            non_cycle_skips += 1
        hmer_context_ref_sum += variant.info[FeatureMapFields.HMER_CONTEXT_REF.value]
        hmer_context_alt_sum += variant.info[FeatureMapFields.HMER_CONTEXT_ALT.value]

    assert cycle_skips == num_cycle_skips
    assert non_cycle_skips == num_non_cycle_skips
    assert hmer_context_ref_sum == hmer_context_ref_sum
    assert hmer_context_alt_sum == hmer_context_alt_sum
