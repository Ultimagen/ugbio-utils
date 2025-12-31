import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from ugbio_srsnv.srsnv_training import NaNToNullEncoder, _parse_model_params


@pytest.fixture
def resources_dir():
    return Path(__file__).parent.parent / "resources"


# def __count_variants(vcf_file):
#     counter = 0
#     for _ in pysam.VariantFile(vcf_file):
#         counter += 1
#     return counter


# def test_prepare_featuremap_for_model(tmpdir, resources_dir):
#     """Test that downsampling training-set works as expected"""

#     input_featuremap_vcf = pjoin(
#         resources_dir,
#         "333_CRCs_39_legacy_v5.featuremap.single_substitutions.subsample.vcf.gz",
#     )
#     rng = np.random.default_rng(0)
#     downsampled_training_featuremap_vcf, _, _, _, _ = prepare_featuremap_for_model(
#         workdir=tmpdir,
#         input_featuremap_vcf=input_featuremap_vcf,
#         train_set_size=12,
#         test_set_size=3,
#         balanced_sampling_info_fields=None,
#         rng=rng,
#     )

#     # Since we use random downsampling the train_set_size might differ slightly from expected
#     n_variants = __count_variants(downsampled_training_featuremap_vcf)
#     assert n_variants >= 8 and n_variants <= 16


# def test_prepare_featuremap_for_model_with_prefilter(tmpdir, resources_dir):
#     """Test that downsampling training-set works as expected"""

#     input_featuremap_vcf = pjoin(
#         resources_dir,
#         "333_CRCs_39_legacy_v5.featuremap.single_substitutions.subsample.vcf.gz",
#     )
#     rng = np.random.default_rng(0)
#     pre_filter_bcftools_include = "(X_SCORE>4) && (X_EDIST<10)"
#     (downsampled_training_featuremap_vcf, downsampled_test_featuremap_vcf, _, _, _) = prepare_featuremap_for_model(
#         workdir=tmpdir,
#         input_featuremap_vcf=input_featuremap_vcf,
#         train_set_size=100,
#         test_set_size=100,
#         balanced_sampling_info_fields=None,
#         pre_filter_bcftools_include=pre_filter_bcftools_include,
#         rng=rng,
#     )
#     # In this scenario we are pre-filtering the test data so that only 4 FeatureMap entries pass:
#     total_variants = int(
#         subprocess.check_output(
#             f"bcftools view -H {input_featuremap_vcf} -i '{pre_filter_bcftools_include}' | wc -l",
#             shell=True,
#         )
#         .decode()
#         .strip()
#     )
#     assert total_variants == 4
#     # and since we are asking for more entries than are available, we should get all of them in equal ratios
#     n_variants = __count_variants(downsampled_training_featuremap_vcf)
#     assert n_variants == 2
#     n_variants = __count_variants(downsampled_test_featuremap_vcf)
#     assert n_variants == 2


# def test_prepare_featuremap_for_model_with_motif_balancing(tmpdir, resources_dir):
#     """Test that downsampling training-set works as expected"""

#     input_featuremap_vcf = pjoin(
#         resources_dir,
#         "333_LuNgs_08.annotated_featuremap.vcf.gz",
#     )
#     balanced_sampling_info_fields = ["trinuc_context", "is_forward"]
#     train_set_size = (4**3) * 10  # 10 variants per context
#     for random_seed in range(2):
#         rng = np.random.default_rng(random_seed)
#         downsampled_training_featuremap_vcf, _, _, _, _ = prepare_featuremap_for_model(
#             workdir=tmpdir,
#             input_featuremap_vcf=input_featuremap_vcf,
#             train_set_size=train_set_size,
#             test_set_size=train_set_size,
#             balanced_sampling_info_fields=balanced_sampling_info_fields,
#             rng=rng,
#         )
#         assert __count_variants(downsampled_training_featuremap_vcf) == train_set_size

#         balanced_sampling_info_fields_counter = defaultdict(int)
#         with pysam.VariantFile(downsampled_training_featuremap_vcf) as fmap:
#             for record in fmap.fetch():
#                 balanced_sampling_info_fields_counter[
#                     tuple(record.info.get(info_field) for info_field in balanced_sampling_info_fields)
#                 ] += 1
#         assert sum(balanced_sampling_info_fields_counter.values()) == train_set_size
#         # T-test that the number of variants per context is in line with a uniform with to 99% confidence
#         _, pvalue = ttest_1samp(
#             list(balanced_sampling_info_fields_counter.values()),
#             np.mean(list(balanced_sampling_info_fields_counter.values())),
#         )
#         assert pvalue > 0.01
#         os.remove(downsampled_training_featuremap_vcf)
#         os.remove(downsampled_training_featuremap_vcf + ".tbi")


# def test_prepare_featuremap_for_model_training_and_test_sets(tmpdir, resources_dir):
#     """Test that downsampling of training and test sets works as expected"""
#     input_featuremap_vcf = pjoin(
#         resources_dir,
#         "333_CRCs_39_legacy_v5.featuremap.single_substitutions.subsample.vcf.gz",
#     )
#     rng = np.random.default_rng(0)
#     (downsampled_training_featuremap_vcf, downsampled_test_featuremap_vcf, _, _, _) = prepare_featuremap_for_model(
#         workdir=tmpdir,
#         input_featuremap_vcf=input_featuremap_vcf,
#         train_set_size=12,
#         test_set_size=3,
#         balanced_sampling_info_fields=None,
#         rng=rng,
#     )
#     assert __count_variants(downsampled_training_featuremap_vcf) == 12
#     assert __count_variants(downsampled_test_featuremap_vcf) == 2


@pytest.mark.parametrize(
    "raw, expected",
    [
        (None, {}),
        (
            "eta=0.1:max_depth=8:n_estimators=200",
            {"eta": 0.1, "max_depth": 8, "n_estimators": 200},
        ),
        ("verbosity=debug:subsample=0.9", {"verbosity": "debug", "subsample": 0.9}),
    ],
)
def test_parse_model_params_inline(raw: str | None, expected: dict[str, Any]) -> None:
    assert _parse_model_params(raw) == expected


def test_parse_model_params_json(tmp_path: Path) -> None:  # noqa: D103
    json_path = tmp_path / "xgb_params.json"
    payload = {"eta": 0.05, "max_depth": 6, "enable_categorical": True}
    json_path.write_text(json.dumps(payload))

    parsed = _parse_model_params(str(json_path))
    assert parsed == payload


def test_parse_model_params_invalid() -> None:  # noqa: D103
    with pytest.raises(ValueError):
        _parse_model_params("eta=0.1:max_depth")  # uneven tokens
    with pytest.raises(ValueError):
        _parse_model_params("eta")  # missing '='


def test_nan_to_null_encoder() -> None:
    """Test that NaNToNullEncoder properly converts NaN to null in JSON output."""
    # Test with various data structures containing NaN values
    test_data = {
        "simple_nan": float("nan"),
        "list_with_nan": [1.0, float("nan"), 3.0],
        "nested_dict": {
            "value": float("nan"),
            "array": [float("nan"), 2.0, float("nan")],
        },
        "normal_values": [1, 2, 3],
        "string": "test",
    }

    # Encode using custom encoder
    json_string = json.dumps(test_data, cls=NaNToNullEncoder)

    # Verify that 'NaN' does not appear in the output
    assert "NaN" not in json_string, "JSON output should not contain 'NaN' strings"

    # Verify that 'null' appears where NaN values were
    assert "null" in json_string, "JSON output should contain 'null' values"

    # Verify that the JSON can be parsed
    parsed_data = json.loads(json_string)

    # Check that NaN values were converted to None (which becomes null in JSON)
    assert parsed_data["simple_nan"] is None
    assert parsed_data["list_with_nan"][1] is None
    assert parsed_data["nested_dict"]["value"] is None
    assert parsed_data["nested_dict"]["array"][0] is None
    assert parsed_data["nested_dict"]["array"][2] is None

    # Check that normal values are preserved
    assert parsed_data["normal_values"] == [1, 2, 3]
    assert parsed_data["string"] == "test"


def test_nan_to_null_encoder_with_numpy() -> None:
    """Test NaNToNullEncoder with NumPy NaN values."""
    test_data = {
        "numpy_nan": np.nan,
        "numpy_array_list": [np.nan, 1.0, np.nan],
        "mixed": {"np_nan": np.nan, "py_nan": float("nan"), "value": 42},
    }

    json_string = json.dumps(test_data, cls=NaNToNullEncoder)

    # Verify no NaN in output
    assert "NaN" not in json_string

    # Verify it's valid JSON
    parsed_data = json.loads(json_string)
    assert parsed_data["numpy_nan"] is None
    assert parsed_data["numpy_array_list"] == [None, 1.0, None]
    assert parsed_data["mixed"]["np_nan"] is None
    assert parsed_data["mixed"]["py_nan"] is None
    assert parsed_data["mixed"]["value"] == 42
