import polars as pl
from ugbio_featuremap.featuremap_to_dataframe import VCFJobConfig, _apply_annotation_filters


def _make_job_cfg(**kwargs) -> VCFJobConfig:
    defaults = {
        "bcftools_path": "bcftools",
        "awk_script": "",
        "columns": [],
        "schema": {},
        "column_config": None,
        "sample_list": ["S1"],
        "log_level": 20,
    }
    defaults.update(kwargs)
    return VCFJobConfig(**defaults)


def _sample_frame() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "CHROM": ["chr1", "chr1", "chr1", "chr2", "chr2"],
            "POS": [100, 200, 300, 400, 500],
            "PCAWG": ["rs1", None, "rs3", None, "rs5"],
            "EXCLUDE_TRAINING": [None, "ex1", None, "ex2", None],
            "INCLUDE_INFERENCE": ["inc1", None, None, None, "inc5"],
        }
    )


def test_exclude_single_field():
    cfg = _make_job_cfg(exclude_if_annotated=["PCAWG"])
    result = _apply_annotation_filters(_sample_frame(), cfg, "test")
    assert result.height == 2
    assert result["POS"].to_list() == [200, 400]


def test_exclude_multiple_fields():
    cfg = _make_job_cfg(exclude_if_annotated=["PCAWG", "EXCLUDE_TRAINING"])
    result = _apply_annotation_filters(_sample_frame(), cfg, "test")
    assert result.height == 0


def test_include_single_field():
    cfg = _make_job_cfg(include_only_annotated=["INCLUDE_INFERENCE"])
    result = _apply_annotation_filters(_sample_frame(), cfg, "test")
    assert result.height == 2
    assert result["POS"].to_list() == [100, 500]


def test_include_multiple_fields_or_logic():
    cfg = _make_job_cfg(include_only_annotated=["INCLUDE_INFERENCE", "PCAWG"])
    result = _apply_annotation_filters(_sample_frame(), cfg, "test")
    assert result.height == 3
    assert result["POS"].to_list() == [100, 300, 500]


def test_missing_column_no_op():
    cfg = _make_job_cfg(exclude_if_annotated=["NONEXISTENT_FIELD"])
    frame = _sample_frame()
    result = _apply_annotation_filters(frame, cfg, "test")
    assert result.height == frame.height


def test_include_missing_column_no_op():
    cfg = _make_job_cfg(include_only_annotated=["NONEXISTENT_FIELD"])
    frame = _sample_frame()
    result = _apply_annotation_filters(frame, cfg, "test")
    assert result.height == frame.height


def test_no_filters_no_op():
    cfg = _make_job_cfg()
    frame = _sample_frame()
    result = _apply_annotation_filters(frame, cfg, "test")
    assert result.height == frame.height


def test_exclude_and_include_combined():
    cfg = _make_job_cfg(
        exclude_if_annotated=["EXCLUDE_TRAINING"],
        include_only_annotated=["PCAWG"],
    )
    result = _apply_annotation_filters(_sample_frame(), cfg, "test")
    assert result.height == 3
    assert result["POS"].to_list() == [100, 300, 500]


def test_exclude_all_rows():
    frame = pl.DataFrame({"CHROM": ["chr1", "chr1"], "POS": [1, 2], "PCAWG": ["a", "b"]})
    cfg = _make_job_cfg(exclude_if_annotated=["PCAWG"])
    result = _apply_annotation_filters(frame, cfg, "test")
    assert result.height == 0


def test_include_filters_empty_when_no_match():
    frame = pl.DataFrame({"CHROM": ["chr1", "chr1"], "POS": [1, 2], "PCAWG": [None, None]})
    cfg = _make_job_cfg(include_only_annotated=["PCAWG"])
    result = _apply_annotation_filters(frame, cfg, "test")
    assert result.height == 0
