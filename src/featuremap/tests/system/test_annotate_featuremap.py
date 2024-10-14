from os.path import join as pjoin
from pathlib import Path

import pysam
import pytest

from ugbio_featuremap.featuremap_utils import FeatureMapFields, annotate_featuremap
from ugbio_ppmseq.ppmSeq_utils import HistogramColumnNames

@pytest.fixture
def resources_dir():
    return Path(__file__).parent / "resources"


def test_annotate_featuremap(tmpdir, resources_dir):
    input_featuremap = pjoin(resources_dir, "tp_featuremap_chr20.vcf.gz")
    output_featuremap = pjoin(tmpdir, "tp_featuremap_chr20.annotated.vcf.gz")
    ref_fasta = pjoin(resources_dir, "sample.fasta")
    annotate_featuremap(
        input_featuremap,
        output_featuremap,
        ref_fasta=ref_fasta,
        ppmSeq_adapter_version="legacy_v5",
        flow_order="TGCA",
        motif_length_to_annotate=3,
        max_hmer_length=20,
    )
    out = pysam.VariantFile(output_featuremap)

    for info_field in [
        "X_CIGAR",
        "X_EDIST",
        "X_FC1",
        "X_FC2",
        "X_READ_COUNT",
        "X_FILTERED_COUNT",
        "X_FLAGS",
        "X_LENGTH",
        "X_MAPQ",
        "X_INDEX",
        "X_RN",
        "X_SCORE",
        "rq",
        FeatureMapFields.IS_FORWARD.value,
        FeatureMapFields.IS_DUPLICATE.value,
        FeatureMapFields.MAX_SOFTCLIP_LENGTH.value,
        HistogramColumnNames.STRAND_RATIO_CATEGORY_START.value,
        HistogramColumnNames.STRAND_RATIO_CATEGORY_END.value,
        HistogramColumnNames.STRAND_RATIO_START.value,
        HistogramColumnNames.STRAND_RATIO_END.value,
        "trinuc_context_with_alt",
        "hmer_context_ref",
        "hmer_context_alt",
        "is_cycle_skip",
        "prev_1",
        "prev_2",
        "prev_3",
        "next_1",
        "next_2",
        "next_3",
    ]:
        assert str(info_field) in out.header.info
