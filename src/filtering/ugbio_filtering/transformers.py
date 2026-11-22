import itertools
from collections import defaultdict
from collections.abc import Iterable
from functools import lru_cache

import numpy as np
import pandas as pd
from sklearn import compose, impute, preprocessing
from sklearn.pipeline import make_pipeline

from ugbio_filtering.tprep_constants import VcfType


# transformers for VCF columns
def tuple_break(x):
    """Returns the first element in the tuple"""
    if isinstance(x, tuple):
        return x[0] if x[0] is not None else np.nan
    return 0 if (x is None or np.isnan(x)) else x


def tuple_break_second(x):
    """Returns the second element in the tuple"""
    if isinstance(x, tuple) and len(x) > 1:
        return x[1]
    return 0 if (x is None or (isinstance(x, tuple) and len(x) < 2) or np.isnan(x)) else x  # noqa PLR2004


def tuple_break_third(x):
    """Returns the third element in the tuple"""
    if isinstance(x, tuple) and len(x) > 1:
        return x[2]
    return 0 if (x is None or (isinstance(x, tuple) and len(x) < 2) or np.isnan(x)) else x  # noqa PLR2004


def motif_encode_left(x):
    """Gets motif as input and translates it into integer
    by bases mapping and order of the bases
    The closes to the variant is the most significant bit
    """
    bases = {"A": 1, "T": 2, "G": 3, "C": 4, "N": 5}
    x_list = list(x)
    x_list.reverse()
    num = 0
    for c_val in x_list:
        num = 10 * num + bases.get(c_val, 0)
    return num


def motif_encode_right(x):
    """Gets motif as input and translates it into integer
    by bases mapping and order of the bases
    The closes to the variant is the most significant bit
    """
    bases = {"A": 1, "T": 2, "G": 3, "C": 4, "N": 5}
    x_list = list(x)
    num = 0
    for c_val in x_list:
        num = 10 * num + bases.get(c_val, 0)
    return num


# def trinucleotide_encode(q):
#     bases = {"A": 1, "T": 2, "G": 3, "C": 4, "N": 5}

#     result = 0
#     for i in range(4):
#         result += result * 10 + bases[q[i]]
#     return result


def allele_encode(x):
    """Translate base into integer.
    In case we don't get a single base, we return zero
    """
    bases = {"A": 1, "T": 2, "G": 3, "C": 4}
    return bases.get(x, 0)


def gt_encode(x):
    """Checks whether the variant is heterozygous(0) or homozygous(1)"""
    if x == (1, 1):
        return 1
    return 0


INS_DEL_ENCODE = {"ins": -1, "del": 1, "NA": 0}


def ins_del_encode(x, encode_dct=INS_DEL_ENCODE):  # pylint: disable=dangerous-default-value
    return encode_dct[x]


@lru_cache(maxsize=1)
def _get_region_encoding():
    regions = ["Telomere_Centromere", "Clusters", "Coverage-Mappability"]
    subsets = []
    for r in range(len(regions) + 1):
        subsets.extend(itertools.combinations(sorted(regions), r))
    for i in range(len(subsets)):
        subsets[i] = tuple(sorted(subsets[i]))
    return dict(zip(subsets, range(1, len(subsets) + 1), strict=False))


def region_annotation_encode(x):
    encoding = _get_region_encoding()
    if x is None:
        return 0
    return encoding[tuple(sorted(x))]


# Depth CNV caller (ugbio_cnv.depth_cnv_caller): the INFO fields the filter reads
CNV_INFO_FIELDS = [
    "SVTYPE",
    "SVLEN",
    "CN_MEAN",
    "BASE_CN",
    "Z",
    "N_MERGED",
    "UNIQ_FRAC",
    "CNV_SOURCE",
    "CLUSTER_SUPPORT",
    "BREAKPOINT_SOURCE",
    "BREAKPOINT_SUPPORT",
    "DEPTH_START",
    "DEPTH_END",
    "CNV_DEL_READS",
    "CNV_DUP_READS",
    "CNV_TOTAL_READS",
    "CNV_DEL_FRAC",
    "CNV_DUP_FRAC",
    "CNV_DEL_PAIRS",
    "CNV_DUP_PAIRS",
    "CNV_DEL_PAIRS_FRAC",
    "CNV_DUP_PAIRS_FRAC",
    "DEL_READS_MEDIAN_INSERT_SIZE",
    "DUP_READS_MEDIAN_INSERT_SIZE",
    "DEL_PAIRS_MEDIAN_SPAN",
    "DUP_PAIRS_MEDIAN_SPAN",
]
# INFO fields that are features as they are (missing = 0)
CNV_NUMERIC_FIELDS = [
    "cn_mean",
    "base_cn",
    "z",
    "n_merged",
    "uniq_frac",
    "cluster_support",
    "breakpoint_support",
    "cnv_del_reads",
    "cnv_dup_reads",
    "cnv_total_reads",
    "cnv_del_frac",
    "cnv_dup_frac",
    "cnv_del_pairs",
    "cnv_dup_pairs",
    "cnv_del_pairs_frac",
    "cnv_dup_pairs_frac",
    "del_reads_median_insert_size",
    "dup_reads_median_insert_size",
    "del_pairs_median_span",
    "dup_pairs_median_span",
]
CNV_SOURCE_ENCODE = {"depth": 0, "split_read": 1, "pair": 2, "cn.mops": 3}
BREAKPOINT_SOURCE_ENCODE = {"split_read": 1, "pair": 2}


def svtype_encode(x):
    """Translate SVTYPE into integer."""
    encoding = {"DEL": 1, "DUP": 2}
    return encoding.get(x, 0)


def cnv_source_encode(x):
    """Translate CNV_SOURCE (the first source of a call) into integer."""
    if isinstance(x, tuple | list):
        x = x[0] if x else None
    return CNV_SOURCE_ENCODE.get(x, 0)


def _cnv_numbers(df: pd.DataFrame) -> pd.DataFrame:
    """INFO columns as floats, the first value of a tuple, missing = 0."""
    return df.apply(lambda s: pd.to_numeric(s.apply(tuple_break), errors="coerce")).fillna(0.0).astype(float)


def cnv_number(df: pd.DataFrame) -> pd.DataFrame:
    """Numeric INFO field as it is."""
    return _cnv_numbers(df)


def cnv_length(df: pd.DataFrame) -> pd.DataFrame:
    """SVLEN without its sign."""
    return _cnv_numbers(df).abs()


def cnv_svtype(df: pd.DataFrame) -> pd.DataFrame:
    """SVTYPE encoded by svtype_encode."""
    return df.map(svtype_encode).astype(float)


def cnv_source(df: pd.DataFrame) -> pd.DataFrame:
    """CNV_SOURCE encoded by cnv_source_encode."""
    return df.map(cnv_source_encode).astype(float)


def breakpoint_source(df: pd.DataFrame) -> pd.DataFrame:
    """BREAKPOINT_SOURCE: 1 split read, 2 pair, 0 depth or missing."""
    return df.map(lambda x: BREAKPOINT_SOURCE_ENCODE.get(x, 0)).astype(float)


def dcn(df: pd.DataFrame) -> pd.DataFrame:
    """Copy-number change of the call, |CN_MEAN - BASE_CN| (BASE_CN 1 on haploid contigs, 2 when missing)."""
    x = _cnv_numbers(df)
    return pd.DataFrame({"dcn": (x["cn_mean"] - x["base_cn"].replace(0, 2.0)).abs()}, index=df.index)


def abs_z(df: pd.DataFrame) -> pd.DataFrame:
    """|Z|: the depth evidence of a deletion (Z < 0) or a duplication (Z > 0) in one feature."""
    return pd.DataFrame({"abs_z": _cnv_numbers(df)["z"].abs()}, index=df.index)


def refine_shrink(df: pd.DataFrame) -> pd.DataFrame:
    """Fraction of the depth segment (DEPTH_START - DEPTH_END) that breakpoint refinement removed."""
    x = _cnv_numbers(df)
    depth_len = x["depth_end"] - x["depth_start"]
    shrink = np.where(depth_len > 0, 1 - x["svlen"].abs() / depth_len.clip(lower=1), 0.0)
    return pd.DataFrame({"refine_shrink": shrink}, index=df.index)


def function_name_out(transformer, _input_features) -> np.ndarray:
    """Output name of a FunctionTransformer that makes one feature: the name of its function."""
    return np.array([transformer.func.__name__])


def _cnv_transform_list() -> list:
    """Per-field transformers of the depth CNV model, plus three that combine fields (dcn, abs_z, refine_shrink).

    The functions are module level so that plain pickle (FilterVCF) can load the model.
    """

    def one_to_one(func):
        return preprocessing.FunctionTransformer(func, feature_names_out="one-to-one")

    def combined(func):
        return preprocessing.FunctionTransformer(func, feature_names_out=function_name_out)

    return [
        ("svtype", one_to_one(cnv_svtype), ["svtype"]),
        ("dcn", combined(dcn), ["cn_mean", "base_cn"]),
        ("svlen", one_to_one(cnv_length), ["svlen"]),
        *[(f, one_to_one(cnv_number), [f]) for f in CNV_NUMERIC_FIELDS],
        ("cnv_source", one_to_one(cnv_source), ["cnv_source"]),
        ("breakpoint_source", one_to_one(breakpoint_source), ["breakpoint_source"]),
        ("abs_z", combined(abs_z), ["z"]),
        ("refine_shrink", combined(refine_shrink), ["svlen", "depth_start", "depth_end"]),
    ]


def get_needed_features(vtype: VcfType = VcfType.SINGLE_SAMPLE, custom_annotations: list | None = None) -> list:
    """Get the list of features that are needed for the model

    Parameters
    ----------
    vtype: string
        The type of the input vcf. Either "single_sample", "joint" or "dv"

    custom_annotations: list, optional
        Additional custom (non-required) annotations. Some will have pre-registered custom transforms
    Returns
    -------
    list of features
    """
    features, _, _ = modify_features_based_on_vcf_type(vtype, custom_annotations)
    return features


def modify_features_based_on_vcf_type(  # noqa C901
    vtype: VcfType = VcfType.SINGLE_SAMPLE, custom_annotations: list | None = None
) -> tuple[list, list, str]:
    """Modify training features based on the type of the vcf

    Parameters
    ----------
    vtype: string
        The type of the input vcf. Either "single_sample", "joint", "dv" or cnv
    custom_annotations: list, optional
        Additional custom (non-required) annotations. Some will have pre-registered custom transforms
    Returns
    -------
    list of features, list of transform, column used for qual (qual or qd)

    Raises
    ------
    ValueError
        If vcf is of unrecognized type.
    """

    def tuple_encode_df(s):
        return pd.DataFrame(np.array([tuple_break(y) for y in s]).reshape((-1, 1)), index=s.index)

    def tuple_encode_doublet_df(s):
        return pd.DataFrame(np.array([y[:2] for y in s]).reshape((-1, 2)), index=s.index)

    def tuple_uniform_encode(s):
        return pd.DataFrame(list(s), index=s.index).fillna(1000)

    def motif_encode_left_df(s):
        return pd.DataFrame(np.array([motif_encode_left(y) for y in s]).reshape((-1, 1)), index=s.index)

    def motif_encode_right_df(s):
        return pd.DataFrame(np.array([motif_encode_right(y) for y in s]).reshape((-1, 1)), index=s.index)

    def allele_encode_df(s):
        return pd.DataFrame(np.array([x[:2] for x in s]), index=s.index).applymap(allele_encode)

    def allele_encode_single(df):
        return df.applymap(allele_encode)

    def gt_encode_df(s):
        return pd.DataFrame(np.array([gt_encode(y) for y in s]).reshape((-1, 1)), index=s.index)

    def ins_del_encode_df(df):
        return pd.DataFrame(np.array(df[0].apply(ins_del_encode)).reshape(-1, 1), index=df.index)

    def region_annotation_encode_df(df):
        return pd.DataFrame(
            np.array(df["region_annotations"].apply(region_annotation_encode)).reshape(-1, 1), index=df.index
        )

    default_filler = impute.SimpleImputer(strategy="constant", fill_value=0)
    tuple_filter = preprocessing.FunctionTransformer(tuple_encode_df)
    ins_del_encode_filter = preprocessing.FunctionTransformer(ins_del_encode_df)
    tuple_encode_df_transformer = preprocessing.FunctionTransformer(tuple_encode_df)
    tuple_encode_doublet_df_transformer = preprocessing.FunctionTransformer(tuple_encode_doublet_df)
    tuple_uniform_encode_df_transformer = preprocessing.FunctionTransformer(tuple_uniform_encode)
    left_motif_filter = preprocessing.FunctionTransformer(motif_encode_left_df)
    right_motif_filter = preprocessing.FunctionTransformer(motif_encode_right_df)
    allele_filter_single = preprocessing.FunctionTransformer(allele_encode_single)
    allele_filter = preprocessing.FunctionTransformer(allele_encode_df)
    gt_filter = preprocessing.FunctionTransformer(gt_encode_df)
    region_annotation_encode_filter = preprocessing.FunctionTransformer(region_annotation_encode_df)
    transform_list = [
        ("ad", tuple_encode_doublet_df_transformer, "ad"),
        ("gt", gt_filter, "gt"),
        ("gq", default_filler, ["gq"]),
        ("pl", tuple_uniform_encode_df_transformer, "pl"),
        ("af", tuple_encode_df_transformer, "af"),
        ("sor", default_filler, ["sor"]),
        ("dp", default_filler, ["dp"]),
        ("alleles", allele_filter, "alleles"),
        ("x_hin", make_pipeline(tuple_filter, allele_filter_single), "x_hin"),
        ("x_hil", make_pipeline(tuple_filter, default_filler), "x_hil"),
        ("x_il", make_pipeline(tuple_filter, default_filler), "x_il"),
        ("indel", "passthrough", ["indel"]),
        ("x_ic", make_pipeline(tuple_filter, ins_del_encode_filter), "x_ic"),
        ("x_lm", left_motif_filter, "x_lm"),
        ("x_rm", right_motif_filter, "x_rm"),
        ("x_css", make_pipeline(tuple_filter, preprocessing.OrdinalEncoder()), "x_css"),
        ("x_gcc", default_filler, ["x_gcc"]),
    ]
    features = [x[0] for x in transform_list]
    if vtype == VcfType.DEEP_VARIANT:
        transform_list.extend([("vaf", tuple_filter, "vaf")])
        features = [x[0] for x in transform_list]

    elif vtype == VcfType.DEEP_VARIANT_WITH_SOFTCLIP_COUNTS:
        transform_list.extend(
            [
                ("mq0_ref", default_filler, ["mq0_ref"]),
                ("mq0_alt", default_filler, ["mq0_alt"]),
                ("ls_ref", default_filler, ["ls_ref"]),
                ("ls_alt", default_filler, ["ls_alt"]),
                ("rs_ref", default_filler, ["rs_ref"]),
                ("rs_alt", default_filler, ["rs_alt"]),
                ("mean_nm_ref", default_filler, ["mean_nm_ref"]),
                ("median_nm_ref", default_filler, ["median_nm_ref"]),
                ("mean_nm_alt", default_filler, ["mean_nm_alt"]),
                ("median_nm_alt", default_filler, ["median_nm_alt"]),
                ("mean_mis_ref", default_filler, ["mean_mis_ref"]),
                ("median_mis_ref", default_filler, ["median_mis_ref"]),
                ("mean_mis_alt", default_filler, ["mean_mis_alt"]),
                ("median_mis_alt", default_filler, ["median_mis_alt"]),
                ("qual", "passthrough", ["qual"]),
            ]
        )
        features = [x[0] for x in transform_list]
    elif vtype == VcfType.SINGLE_SAMPLE:
        transform_list.extend(
            [
                ("qual", "passthrough", ["qual"]),
                ("fs", default_filler, ["fs"]),
                ("qd", default_filler, ["qd"]),
                ("mq", default_filler, ["mq"]),
                ("an", default_filler, ["an"]),
                ("baseqranksum", default_filler, ["baseqranksum"]),
                ("excesshet", default_filler, ["excesshet"]),
                ("mqranksum", default_filler, ["mqranksum"]),
                ("readposranksum", default_filler, ["readposranksum"]),
                ("ac", tuple_encode_df_transformer, "ac"),
                ("xc", default_filler, ["xc"]),
                ("mleac", tuple_encode_df_transformer, "mleac"),
                ("mleaf", tuple_encode_df_transformer, "mleaf"),
                ("hapcomp", tuple_encode_df_transformer, "hapcomp"),
                ("mq0c", tuple_encode_doublet_df_transformer, "mq0c"),
                ("scl", tuple_encode_doublet_df_transformer, "scl"),
                ("scr", tuple_encode_doublet_df_transformer, "scr"),
            ]
        )
        features = [x[0] for x in transform_list]
    elif vtype == VcfType.JOINT:
        pass
    elif vtype == VcfType.CNV:
        transform_list = _cnv_transform_list()
        features = [f.lower() for f in CNV_INFO_FIELDS]

    else:
        raise ValueError("Unrecognized VCF type")

    # Taking care of custom annotations
    # In general we assume that custom annotations are TRUE/NONE
    # Some custom annotation require special treatment and this is defined in custom_fields_dict
    default_transformer = make_pipeline(
        impute.SimpleImputer(strategy="constant", missing_values=None, fill_value="FALSE"),
        preprocessing.OrdinalEncoder(),
    )

    def convert_to_numeric(df):
        return pd.DataFrame(pd.to_numeric(pd.Series(df.iloc[:, 0])))

    convert_to_numeric_transformer = preprocessing.FunctionTransformer(convert_to_numeric)

    long_hmer_transformer = make_pipeline(
        impute.SimpleImputer(strategy="constant", fill_value="0", missing_values=None), convert_to_numeric_transformer
    )
    region_annotation_transformer = make_pipeline(region_annotation_encode_filter)

    if custom_annotations is None:
        custom_annotations = []
    custom_fields_dict = defaultdict(lambda: default_transformer)
    custom_fields_dict["long_hmer"] = long_hmer_transformer
    custom_fields_dict["region_annotations"] = region_annotation_transformer

    for an in custom_annotations:
        features.append(an)
        transform_list.append((an, custom_fields_dict[an], [an]))

    qual_column = "qual"
    return features, transform_list, f"{qual_column}__{qual_column}"


def get_transformer(vtype: VcfType, annots: list | None = None) -> compose.ColumnTransformer:
    """Prepare dataframe for analysis (encode features, normalize etc.)

    Parameters
    ----------
    vtype: VcfType
        The type of the input vcf. Either "single_sample" or "joint"
    annots: list, optional
        List of annotation features (will be transformed with "None")

    Returns
    -------
    compose.ColumnTransformer
        Transformer of the dataframe to dataframe
    """
    _, transform_list, _ = modify_features_based_on_vcf_type(vtype, annots)
    transformer = compose.ColumnTransformer(transform_list)
    transformer.set_output(transform="pandas")

    return transformer


def encode_labels(ll: Iterable[tuple[int, int]]) -> list[int]:
    """Convert genotype label (tuple) to number

    Parameters
    ----------
    ll : Iterable[tuple[int, int]]
        List of labels

    Returns
    -------
    list[int]
        List of encoded labels
    """
    return [encode_label(x) for x in ll]


def encode_label(label: tuple[int, ...]) -> int:
    "Convert genotype label (tuple) to number"
    label = tuple(sorted(label))
    if label == (0, 0):
        return 0
    if label in [(0, 1), (1, 0)]:
        return 1
    if label == (1, 1):
        return 2
    raise ValueError(f"Encoding of gt={label} not supported")


def decode_label(label: int) -> tuple[int, int]:
    "Numerical encoding of genotype to tuple of genotypes"
    decode_dct = {0: (0, 1), 1: (1, 1), 2: (0, 0)}
    return decode_dct[label]


label_encode = preprocessing.FunctionTransformer(encode_labels)
