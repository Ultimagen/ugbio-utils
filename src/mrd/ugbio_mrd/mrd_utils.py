import itertools
import json
import logging
import os
import re
from collections.abc import Iterable
from os.path import basename, splitext
from os.path import join as pjoin

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyBigWig as bw  # noqa: N813
import pysam
import seaborn as sns
from tqdm import tqdm
from ugbio_core.dna_sequence_utils import revcomp
from ugbio_core.logger import logger
from ugbio_core.vcfbed.variant_annotation import (
    get_trinuc_substitution_dist,
    parse_trinuc_sub,
)

default_featuremap_info_fields = {
    "X_CIGAR": str,
    "X_EDIST": int,
    "X_FC1": str,
    "X_FC2": str,
    "X_READ_COUNT": int,
    "X_FILTERED_COUNT": int,
    "X_FLAGS": int,
    "X_LENGTH": int,
    "X_MAPQ": float,
    "X_INDEX": int,
    "X_RN": str,
    "X_SCORE": float,
    "rq": float,
}


def _get_sample_name_from_file_name(file_name, split_position=0):
    """
    Internal formatting of filename for mrd pipeline

    Parameters
    ----------
    file_name: str
        file name
    split_position: int
        which position relative to splitting by "." the sample name is

    Returns
    -------
    out_file_name
        reformatted file name

    """
    return (
        basename(file_name)
        .split(".")[split_position]
        .replace("-", "_")
        .replace("_filtered_signature", "")  # remove generic suffixes
        .replace("_signature", "")  # remove generic suffixes
        .replace("signature_", "")  # remove generic suffixes
        .replace("_filtered", "")  # remove generic suffixes
        .replace("filtered", "")  # remove generic suffixes
        .replace("featuremap_", "")  # remove generic suffixes
        .replace("_featuremap", "")  # remove generic suffixes
        .replace("featuremap", "")  # remove generic suffixes
    )


def _get_hmer_length(ref: str, left_motif: str, right_motif: str):
    """
    Calculate the length of the hmer the ref is contained in (limited by input motif length)
    Parameters
    ----------
    ref: str
        reference base (single base)
    left_motif: str
        left motif in forward orientation
    right_motif: str
        right motif in forward orientation

    Returns
    -------
    hlen: int
        The homopolymer length the reference base is contained in (limited by input motif length)

    """

    left_motif_len = len(left_motif)
    x = np.array([(k == ref, sum(1 for _ in v)) for k, v in itertools.groupby(left_motif + ref + right_motif)])
    return x[np.argmax(x[:, 1].cumsum() >= left_motif_len + 1), 1]


def _safe_tabix_index(vcf_file: str):
    try:
        pysam.tabix_index(vcf_file, preset="vcf", force=True)  # make sure vcf is indexed
        if (not os.path.isfile(vcf_file)) and os.path.isfile(vcf_file + ".gz"):  # was compressed by tabix
            vcf_file = vcf_file + ".gz"
    except Exception as e:
        # Catching a broad exception because this is optional and we never want to have a run fail because of the index
        logger.warning(f"Could not index signature file {vcf_file}\n{str(e)}")
    return vcf_file


def _validate_info(x):
    """
    Validate INFO field in a VCF record.
    In some cases it is a string of boolean ("TRUE"/"FALSE")
    In the case of long hmers it's numeric: if the hmer >= 7
    it appears in the INFO field as a tuple of (hmer, hmer_length),
    then the function should return True
    """
    max_hmer_length = 7
    if isinstance(x, bool):
        return x
    if isinstance(x, str):
        if x.lower() == "true":
            return True
        if x.lower() == "false":
            return False
        if x.isnumeric():
            return bool(float(x) >= max_hmer_length)
        raise ValueError(f"Cannot convert {x} to bool")
    return None


def collect_coverage_per_locus(coverage_bw_files, df_sig):
    try:
        logger.debug("Reading input from bigwig coverage data")
        f_bw = [bw.open(x) for x in coverage_bw_files]
        df_list = []

        for chrom, df_tmp in tqdm(df_sig.groupby(level="chrom")):
            if df_tmp.shape[0] == 0:
                continue
            found_correct_file = False

            f_bw_chrom = {}
            for f_bw_chrom in f_bw:
                if chrom in f_bw_chrom.chroms():
                    found_correct_file = True
                    break

            if not found_correct_file:
                raise ValueError(f"Could not find a bigwig file with {chrom} in:\n{', '.join(coverage_bw_files)}")

            chrom_start = df_tmp.index.get_level_values("pos") - 1
            chrom_end = df_tmp.index.get_level_values("pos")
            df_list.append(
                df_tmp.assign(
                    coverage=np.concatenate(
                        [
                            f_bw_chrom.values(chrom, x, y, numpy=True)
                            for x, y in zip(chrom_start, chrom_end, strict=False)
                        ]
                    )
                )
            )
    finally:
        if "f_bw" in locals():
            for variant_file in f_bw:
                variant_file.close()
    return df_list


def collect_coverage_per_locus_mosdepth(coverage_bed, df_sig):
    """
    Merge coverage data from mosdepth output to signature dataframe
    """
    coverage_mosdepth = pd.read_csv(
        coverage_bed,
        compression="gzip",
        sep="\t",
        names=["chrom", "start", "pos", "coverage"],
    )
    coverage_mosdepth["coverage"] = coverage_mosdepth["coverage"].astype(int)
    coverage_mosdepth = coverage_mosdepth.set_index(["chrom", "pos"])
    df_sig = df_sig.join(coverage_mosdepth, how="left")
    return df_sig


def match_vaf_field(header):
    """ "
    Match the variant allele frequency (VAF) field, as it is variable between signature-generating programs
    """
    format_fields = [x.name for x in header.formats.values()]
    info_keys = list(header.info.keys())
    if "AF" in format_fields:
        if header.formats["AF"].number == "A":
            # mutect
            af_field_type = "AF,A"
    elif "VAF" in format_fields:
        if header.formats["VAF"].number == "A":
            # DV
            af_field_type = "VAF,A"
        elif header.formats["VAF"].number == 1:
            # pileup_featuremap
            af_field_type = "VAF,1"
    elif "DNA_VAF" in info_keys:
        # synthetic signature
        af_field_type = "DNA_VAF,1"
    else:
        af_field_type = "no_AF_field_found"
    return af_field_type


def extract_vaf_val(rec, tumor_sample, af_field_type):
    match af_field_type:
        case "AF,A":
            return rec.samples[tumor_sample]["AF"][0]
        case "VAF,A":
            return rec.samples[tumor_sample]["VAF"][0]
        case "VAF,1":
            return rec.samples[tumor_sample]["VAF"]
        case "DNA_VAF,1":
            return rec.info["DNA_VAF"]
        case "no_AF_field_found":
            return np.nan


def read_signature(  # noqa: C901, PLR0912, PLR0913, PLR0915 #TODO: refactor
    signature_vcf_files: list[str],
    output_parquet: str = None,
    coverage_bed: str = None,
    tumor_sample: str = None,
    x_columns_name_dict: dict = None,
    columns_to_drop: list = None,
    signature_type: str = None,
    *,
    verbose: bool = True,
    raise_exception_on_sample_not_found: bool = False,
    return_dataframes: bool = False,
    concat_to_existing_output_parquet: bool = False,
):
    """
    Read signature (variant calling output, generally mutect) results to dataframe.

    signature_vcf_files: str or list[str]
        File name or a list of file names
    output_parquet: str, optional
        File name to save result to, unless None (default).
        If this file exists and concat_to_existing_output_parquet is True data is appended
    tumor_sample: str, optional
        tumor sample name in the vcf to take allele fraction (AF) from. If not given then a line starting with
        '##tumor_sample=' is looked for in the header, and if it's not found sample data is not read.
    x_columns_name_dict: dict, optional
        Dictionary of INFO fields starting with "X-" (custom UG fields) to read as keys and respective columns names
        as values, if None (default) this value if used:
        {"X-CSS": "cycle_skip_status","X-GCC": "gc_content","X-LM": "left_motif","X-RM": "right_motif"}
    columns_to_drop: list, optional
        List of columns to drop, as values, if None (default) this value if used: ["X-IC"]
    verbose: bool, optional
        show verbose debug messages
    raise_exception_on_sample_not_found: bool, optional
        if True (default False) and the sample name could not be found to determine AF, raise a ValueError
    signature_type: str, optional
        Set signature_type column in the dataframe to be equal to the input value, if it is not None
    concat_to_existing_output_parquet: bool, optional
        If True (default False) and output_parquet is not None and it exists, the new data is concatenated to the


    Raises
    ------
    ValueError
        may be raised

    """
    if output_parquet and os.path.isfile(output_parquet):  # concat new data with existing
        if not concat_to_existing_output_parquet:
            raise ValueError(
                f"output_parquet {output_parquet} exists and concat_to_existing_output_parquet is False"
                " - cannot continue"
            )
        logger.debug(f"Reading existing data from {output_parquet}")
        df_existing = pd.read_parquet(output_parquet, engine="fastparquet")
    else:
        df_existing = None

    if output_parquet is None and not return_dataframes:
        raise ValueError("output_parquet is not None and return_dataframes is False - nothing to do")

    if signature_vcf_files is None or len(signature_vcf_files) == 0:
        logger.debug("Empty input to read_signature - exiting without doing anything")
        return df_existing
    if verbose:
        logger.setLevel(logging.DEBUG)
        for handler in logger.handlers:
            handler.setLevel(logging.DEBUG)
    if not isinstance(signature_vcf_files, str) and isinstance(signature_vcf_files, Iterable):
        logger.debug(f"Reading and merging signature files:\n {signature_vcf_files}")
        df_sig = pd.concat(
            (
                read_signature(
                    file_name,
                    output_parquet=None,
                    return_dataframes=True,
                    tumor_sample=tumor_sample,
                    x_columns_name_dict=x_columns_name_dict,
                    columns_to_drop=columns_to_drop,
                    verbose=j == 0,  # only verbose in first iteration
                )
                .assign(signature=_get_sample_name_from_file_name(file_name, split_position=0))
                .reset_index()
                .set_index(["signature", "chrom", "pos"])
                for j, file_name in enumerate(np.unique(signature_vcf_files))
            )
        )

    else:
        entries = []
        logger.debug(f"Reading vcf file {signature_vcf_files}")
        x_columns_name_dict = {
            "X_CSS": "cycle_skip_status",
            "X_GCC": "gc_content",
            "X_LM": "left_motif",
            "X_RM": "right_motif",
        }
        columns_to_drop = ["X_IC", "X_HIL", "X_HIN", "X_IC", "X_IL"]

        signature_vcf_files = _safe_tabix_index(signature_vcf_files)  # make sure it's indexed
        with pysam.VariantFile(signature_vcf_files) as variant_file:
            info_keys = list(variant_file.header.info.keys())
            # get tumor sample name
            header = variant_file.header
            if tumor_sample is None:
                number_of_samples = len(list(header.samples))
                if number_of_samples >= 2:  # noqa: PLR2004
                    # mutect2 vcf
                    for x in str(header).split("\n"):
                        m = re.match(r"##tumor_sample=(.+)", x)  # good for mutect2
                        if m is not None:
                            tumor_sample = m.groups()[0]
                            logger.debug(f"Tumor sample name is {tumor_sample}")
                            break
                elif number_of_samples == 1:
                    # DV vcf
                    tumor_sample = list(header.samples)[0]
                    logger.debug(f"Tumor sample name is {tumor_sample}")
                elif number_of_samples == 0:
                    # synthetic signature
                    logger.debug("No samples found in vcf file")
            if tumor_sample is not None and tumor_sample not in header.samples:
                if raise_exception_on_sample_not_found:
                    raise ValueError(
                        f"Tumor sample name {tumor_sample} not in vcf sample names: {list(header.samples)}"
                    )
                tumor_sample = None
            if tumor_sample is None and raise_exception_on_sample_not_found:
                raise ValueError(f"Tumor sample name not found. vcf sample names: {list(header.samples)}")
            # not found but allowed to continue because raise_exception_on_sample_not_found = False
            # get INFO annotations and X_ variables
            genomic_region_annotations = [
                k for k in info_keys if variant_file.header.info[k].description.startswith("Genomic Region Annotation:")
            ]
            x_columns = [k for k in info_keys if k.startswith("X_") and k not in columns_to_drop]
            x_columns_name_dict = {k: x_columns_name_dict.get(k, k) for k in x_columns}

            af_field_type = match_vaf_field(header)

            logger.debug(f"Reading x_columns: {x_columns}")

            for j, rec in enumerate(variant_file):
                if j == 0 and tumor_sample is None:
                    samples = rec.samples.keys()
                    if len(samples) == 1:
                        tumor_sample = samples[0]
                entries.append(
                    tuple(
                        [
                            rec.chrom,
                            rec.pos,
                            rec.ref,
                            rec.alts[0],
                            rec.id,
                            rec.qual,
                            extract_vaf_val(rec, tumor_sample, af_field_type),
                            (
                                rec.samples[tumor_sample]["DP"]
                                if tumor_sample and "DP" in rec.samples[tumor_sample]
                                else np.nan
                            ),
                            (rec.info["LONG_HMER"] if "LONG_HMER" in rec.info else np.nan),
                            (
                                rec.info["TLOD"][0]
                                if "TLOD" in rec.info and isinstance(rec.info["TLOD"], Iterable)
                                else np.nan
                            ),
                            rec.info["SOR"] if "SOR" in rec.info else np.nan,
                        ]
                        + [_validate_info(rec.info.get(c, "False")) for c in genomic_region_annotations]
                        + [(rec.info[c][0] if isinstance(rec.info[c], tuple) else rec.info[c]) for c in x_columns]
                    )
                )
        logger.debug(f"Done reading vcf file {signature_vcf_files}" "Converting to dataframe")
        df_sig = (
            pd.DataFrame(
                entries,
                columns=[
                    "chrom",
                    "pos",
                    "ref",
                    "alt",
                    "id",
                    "qual",
                    "af",
                    "depth_tumor_sample",
                    "hmer",
                    "tlod",
                    "sor",
                ]
                + genomic_region_annotations
                + [x_columns_name_dict[c] for c in x_columns],
            )
            .reset_index(drop=True)
            .astype({"chrom": str, "pos": int})
            .set_index(["chrom", "pos"])
        )
        logger.debug("Done converting to dataframe")

    df_sig = df_sig.sort_index()

    if coverage_bed:
        df_sig = collect_coverage_per_locus_mosdepth(coverage_bed, df_sig)

    logger.debug("Calculating reference hmer")
    try:
        df_sig["hmer"] = pd.to_numeric(df_sig["hmer"], errors="coerce")
        df_sig.loc[:, "hmer"] = (
            df_sig[["hmer"]]
            .assign(
                hmer_calc=df_sig.apply(
                    lambda row: _get_hmer_length(row["ref"], row["left_motif"], row["right_motif"]),
                    axis=1,
                )
            )
            .astype(float)
            .max(axis=1)
        )
    except KeyError:
        logger.debug("Could not calculate hmer")

    logger.debug("Annotating with mutation type (ref->alt)")
    ref_is_c_or_t = df_sig["ref"].isin(["C", "T"])
    df_sig.loc[:, "mutation_type"] = (
        np.where(ref_is_c_or_t, df_sig["ref"], df_sig["ref"].apply(revcomp))
        + "->"
        + np.where(ref_is_c_or_t, df_sig["alt"], df_sig["alt"].apply(revcomp))
    )

    df_sig.columns = [x.replace("-", "_").lower() for x in df_sig.columns]

    if signature_type is not None and isinstance(signature_type, str):
        df_sig = df_sig.assign(signature_type=signature_type)

    if df_existing is not None:
        logger.debug(f"Concatenating to previous existing data in {output_parquet}")
        df_sig = pd.concat((df_existing.set_index(df_sig.index.names), df_sig))

    if output_parquet:
        logger.debug(f"Saving output signature/s to {output_parquet}")
        df_sig.reset_index().to_parquet(output_parquet)

    if return_dataframes:
        return df_sig
    return None


def read_intersection_dataframes(
    intersected_featuremaps_parquet,
    output_parquet=None,
    *,
    return_dataframes=False,
):
    """
    Read featuremap dataframes from several intersections of one featuremaps with several signatures, each is annotated
    with the signature name. Assumed to be the output of the intersection of featuremap with signature as generated by
    terra_pipeline/wdls/tasks/mrd.wdl, task FeatureMapIntersectWithSignatures
    Return a concatenated dataframe with the signature name as a column

    Parameters
    ----------
    intersected_featuremaps_parquet: list[str] or str
        list of featuremaps intesected with various signatures
    output_parquet: str
        File name to save result to, default None
    return_dataframes: bool
        Return dataframes

    Returns
    -------
    dataframe: pd.DataFrame
        concatenated intersection dataframes

    Raises
    ------
    ValueError
        may be raised

    """
    if output_parquet is None and not return_dataframes:
        raise ValueError("output_parquet is not None and return_dataframes is False - nothing to do")
    if isinstance(intersected_featuremaps_parquet, str):
        intersected_featuremaps_parquet = [intersected_featuremaps_parquet]
    logger.debug(f"Reading {len(intersected_featuremaps_parquet)} intersection featuremaps")
    df_int = pd.concat(
        pd.read_parquet(f, engine="fastparquet").assign(
            cfdna=_get_sample_name_from_file_name(f, split_position=0),
            signature=_get_sample_name_from_file_name(f, split_position=1),
            signature_type=_get_sample_name_from_file_name(f, split_position=2),
        )
        for f in intersected_featuremaps_parquet
    )
    if output_parquet is not None:
        df_int.reset_index().to_parquet(output_parquet)
    if return_dataframes:
        return df_int
    return None


def generate_synthetic_signatures(
    signature_vcf: str,
    db_vcf: str,
    n_synthetic_signatures: int,
    output_dir: str,
    ref_fasta: str = None,
) -> list[str]:
    """
    Generate synthetic signatures from a signature vcf file and a db vcf file

    Parameters
    ----------
    signature_vcf: str
        signature vcf file
    db_vcf: str
        db vcf file
    n_synthetic_signatures: int
        number of synthetic signatures to generate
    output_dir: str
        output directory
    ref_fasta: str, optional
        reference fasta file, default None. Required if input vcf is not annotated with left and right motifs
        X_LM and X_RM

    Returns
    -------
    synthetic_signatures: list[str]
        list of synthetic control vcf files

    """
    # parse trinuc substitution distribution from signature vcf and db vcf
    trinuc_signature = get_trinuc_substitution_dist(signature_vcf, ref_fasta)
    trinuc_db = get_trinuc_substitution_dist(db_vcf, ref_fasta)

    # allocate index and counter per synthetic signature
    trinuc_dict = {}
    # fix a seed for random.choice
    random_seed: int = 0
    rng = np.random.default_rng(random_seed)
    for trinucsub in trinuc_signature.keys():
        n_subs_signature = trinuc_signature[trinucsub]
        n_subs_db = trinuc_db[trinucsub]
        trinuc_dict[trinucsub] = {}
        trinuc_dict[trinucsub]["index_to_sample"] = {}
        trinuc_dict[trinucsub]["trinuc_counter"] = {}
        n_subs_signature = min(n_subs_db, n_subs_signature)  # limit options to the available loci
        for i in range(n_synthetic_signatures):
            trinuc_dict[trinucsub]["index_to_sample"][i] = (
                rng.choice(range(n_subs_db), n_subs_signature, replace=False) if n_subs_signature > 0 else ()
            )
            trinuc_dict[trinucsub]["trinuc_counter"][i] = 0

    # make output directory
    os.makedirs(output_dir, exist_ok=True)
    # db and signautre basename for output files
    db_basename = splitext(splitext(basename(db_vcf))[0])[0]
    signature_basename = splitext(splitext(basename(signature_vcf))[0])[0]
    # read db vcf file and allocate output files and file handles
    db_fh = pysam.VariantFile(db_vcf, "rb")
    synthetic_signatures = []
    file_handle_list = []
    for i in range(n_synthetic_signatures):
        output_vcf = pjoin(output_dir, f"syn{i}_{signature_basename}_{db_basename}.vcf.gz")
        synthetic_signatures.append(output_vcf)
        outfh = pysam.VariantFile(output_vcf, "w", header=db_fh.header)
        file_handle_list.append(outfh)

    # read db file, parse trinuc substiution and write to output files
    for rec in db_fh.fetch():
        db_trinucsub = parse_trinuc_sub(rec)
        for i in range(n_synthetic_signatures):
            if trinuc_dict[db_trinucsub]["trinuc_counter"][i] in trinuc_dict[db_trinucsub]["index_to_sample"][i]:
                file_handle_list[i].write(rec)
            trinuc_dict[db_trinucsub]["trinuc_counter"][i] += 1

    db_fh.close()

    for i in range(n_synthetic_signatures):
        file_handle_list[i].close()
        output_vcf = synthetic_signatures[i]
        pysam.tabix_index(output_vcf, preset="vcf", min_shift=0, force=True)

    return synthetic_signatures


def read_and_filter_features_parquet(
    features_file_parquet: str,
    read_filter_query: str,
    thresh_locus_filter_high_ratio_of_low_mapq_reads: float = 0.95,
    thresh_locus_filter_high_ratio_of_filtered_reads: float = 0.8,
    thresh_locus_filter_many_alt_reads: int = 2,
    thresh_locus_filter_many_non_ref_alt_reads: int = 2,
    thresh_locus_filter_many_indels: int = 4,
):
    """
    Read featuremap parquet file and filter by query

    Parameters
    ----------
    features_file_parquet: str
        featuremap parquet file
    read_filter_query: str
        query to filter the dataframe

    Returns
    -------
    df_features: pd.DataFrame
        original dataframe
    df_features_filt: pd.DataFrame
        filtered dataframe
    filtering_ratio: pd.DataFrame
        A dataframe that includes the ratio of filtered to total reads per variant
    """
    df_features = pd.read_parquet(features_file_parquet, engine="fastparquet")
    if "index" in df_features.columns:
        df_features = df_features.drop(columns="index")
    # rename columns to lowercase
    df_features = df_features.rename(columns=lambda x: x.lower()).set_index(["chrom", "pos"]).sort_index()
    # refine ad columns
    ad_cols = ["ad_a", "ad_c", "ad_g", "ad_t"]
    if ad_cols[0] in df_features.columns:
        df_features = df_features.assign(
            filtering_ratio=df_features["dp_filt"] / df_features["dp"],
            dp_mapq60_ratio=(df_features["dp_mapq60"]) / df_features["dp"],
            ad_ref=np.choose(
                df_features["ref"].str.lower().map({"a": 0, "c": 1, "g": 2, "t": 3}).to_numpy(),
                df_features[ad_cols].to_numpy().T,
            ),
            ad_alt=np.choose(
                df_features["alt"].str.lower().map({"a": 0, "c": 1, "g": 2, "t": 3}).to_numpy(),
                df_features[ad_cols].to_numpy().T,
            ),
        )
        df_features = df_features.assign(
            ad_non_ref_alt=df_features["dp"] - df_features["ad_ref"] - df_features["ad_alt"],
        )
    # assign mi_primary - the primary alignment's MI (read name)
    if "mi" in df_features.columns and "rn" in df_features.columns:
        df_features = df_features.assign(
            mi_primary=df_features["mi"].fillna(df_features["rn"]),
        )

    # assign locus filter
    df_features = (
        df_features.reset_index()
        .set_index(["chrom", "pos", "signature"])
        .join(
            df_features.groupby(["chrom", "pos", "signature"])
            .agg({"mi_primary": "nunique"})
            .rename(columns={"mi_primary": "unique_hq_supporting_reads"})["unique_hq_supporting_reads"],
            how="left",
        )
    ).reset_index(level="signature")
    df_features = df_features.assign(
        locus_filter_low_ratio_of_low_mapq_reads=df_features["dp_mapq60_ratio"]
        > thresh_locus_filter_high_ratio_of_low_mapq_reads,
        locus_filter_low_ratio_of_filtered_reads=df_features["filtering_ratio"]
        > thresh_locus_filter_high_ratio_of_filtered_reads,
        locus_filter_low_alt_reads=df_features["ad_alt"] <= thresh_locus_filter_many_alt_reads,
        locus_filter_only_one_hq_supporting_read=df_features["unique_hq_supporting_reads"] <= 1,
        locus_filter_low_non_ref_alt_reads=df_features["ad_non_ref_alt"] <= thresh_locus_filter_many_non_ref_alt_reads,
        locus_filter_few_indels=(df_features["ad_del"] + df_features["ad_ins"] <= thresh_locus_filter_many_indels),
    )
    df_features = df_features.assign(locus_filter=df_features.filter(regex="locus_filter_.*").all(axis=1))

    # kept for backwards compatibility - filtering ratio per locus
    filtering_ratio = (
        df_features.query("signature_type=='matched'").groupby(level=["chrom", "pos"]).agg({"filtering_ratio": "first"})
    )
    df_features_filt = df_features.query(read_filter_query)
    return df_features, df_features_filt, filtering_ratio


def read_and_filter_signatures_parquet(
    signatures_file_parquet: str,
    signature_filter_query: str,
    filtering_ratio: pd.DataFrame,
):
    """
    Read signature parquet file and filter by query

    Parameters
    ----------
    signatures_file_parquet: str
        signature parquet file
    signature_filter_query: str
        query to filter the dataframe
    filtering_ratio: pd.DataFrame
        A dataframe that includes the ratio of filtered to total reads per variant
    """
    df_signatures = (
        pd.read_parquet(signatures_file_parquet, engine="fastparquet")
        .astype({"ug_hcr": bool, "id": bool, "ug_mrd_blacklist": bool})
        .set_index(["chrom", "pos"])
    )

    nunique = (
        df_signatures.groupby(level=["chrom", "pos"])
        .agg({"signature": "nunique"})
        .rename(columns={"signature": "nunique"})
    )
    nunique.value_counts().rename("count").to_frame().join(nunique.value_counts(normalize=True).rename("norm"))

    x = df_signatures.filter(regex="coverage").sum(axis=1)
    norm_coverage = (x / x.median()).rename("norm_coverage").reset_index().drop_duplicates().set_index(["chrom", "pos"])
    df_signatures = (
        df_signatures.join(nunique)
        .join(
            filtering_ratio,
            how="left",
        )
        .join(norm_coverage, how="left")
        .fillna({"filtering_ratio": 1})
    )

    df_signatures_filt = df_signatures.query(signature_filter_query)
    return df_signatures, df_signatures_filt


def plot_signature_mutation_types(df_signatures_in: pd.DataFrame, signature_filter_query_in: pd.DataFrame):
    """
    Plot a histogram of substitution types for a signature dataframe

    Parameters
    ----------
    df_signatures_in: pd.DataFrame
        signature dataframe
    signature_filter_query_in: pd.DataFrame
        query to filter the dataframe
    """
    fig, axs = plt.subplots(1, 2, figsize=(18, 4))
    fig.suptitle(",".join(df_signatures_in["signature"].unique()), y=1.13)
    for ax, column, df_plot in zip(
        axs.flatten(),
        [
            "Unfiltered",
            "Filtered",
        ],
        [
            df_signatures_in,
            df_signatures_in.query(signature_filter_query_in),
        ],
        strict=False,
    ):
        x = df_plot["mutation_type"].value_counts(normalize=True).sort_index()
        all_muts = [
            "C->A",
            "C->G",
            "C->T",
            "T->A",
            "T->C",
            "T->G",
        ]
        x = x.reindex(index=all_muts, method="bfill", fill_value=0)
        tot_mutations = df_plot.shape[0]
        plt.sca(ax)
        plt.bar(range(6), x, color=["b", "g", "r", "y", "m", "c"])
        for px, py in zip(range(6), x, strict=False):
            plt.text(px, py + 0.01, f"{py:.1%}", ha="center", fontsize=16)
        plt.ylim(0, ax.get_ylim()[1] + 0.03)
        plt.yticks([])
        plt.xticks(range(6), x.index.values, fontsize=20, rotation=90)
        plt.title(f"{column}, total={tot_mutations:,}", fontsize=28)
    plt.show()


def plot_signature_allele_fractions(df_signatures_in: pd.DataFrame, signature_filter_query_in: pd.DataFrame):
    """
    Plot allele fraction histograms for a signature dataframe

    Parameters
    ----------
    df_signatures_in: pd.DataFrame
        signature dataframe
    signature_filter_query_in: pd.DataFrame
        query to filter the dataframe
    """
    bins = np.linspace(0, 1, 100)
    fig, axs = plt.subplots(1, 2, figsize=(18, 4), sharey=True)
    fig.suptitle(",".join(df_signatures_in["signature"].unique()), y=1.13)
    for ax, column, df_plot in zip(
        axs.flatten(),
        [
            "Unfiltered",
            "Filtered",
        ],
        [
            df_signatures_in,
            df_signatures_in.query(signature_filter_query_in),
        ],
        strict=False,
    ):
        plt.sca(ax)
        x = df_plot["af"].to_numpy()
        tot_mutations = df_plot.shape[0]
        h, bin_edges = np.histogram(x, bins=bins)
        bin_centers = (bin_edges[1:] + bin_edges[:-1]) / 2
        plt.fill_between(
            bin_centers,
            -10,
            h,
            label=f"Median = {np.median(x):.1%}\nMean = {np.mean(x):.1%}",
        )
        plt.legend()
        plt.xlim(0, 1)
        plt.ylim(-1, ax.get_ylim()[1])
        plt.xlabel("AF")
        plt.title(f"{column}, total={tot_mutations:,}", fontsize=28)
    plt.show()


def plot_tf(df_tf_in: pd.DataFrame, zero_tf_fill=1e-7, title=None, random_seed=3456):  # noqa: C901, PLR0912, PLR0915
    """
    Plot tumor fraction boxplot

    Parameters
    ----------
    df_tf_in: pd.DataFrame
        tumor fraction dataframe
    zero_tf_fill: float, optional
        fill zero values with this value, default 1e-7
    title: str, optional
        title of chioce, default None
    random_seed: int, optional
        random seed for plotting, default 3456
    """
    df_tf_in = df_tf_in.assign(tf=df_tf_in["tf"].clip(lower=zero_tf_fill))
    any_nonzero_tf = (df_tf_in["tf"] > 0).any()
    try:
        df_tf_matched = df_tf_in.loc[("matched", slice(None)), "tf"]
    except KeyError:
        df_tf_matched = pd.DataFrame({"signature_type": [np.nan], "signature": [np.nan], "tf": [np.nan]}).set_index(
            ["signature_type", "signature"]
        )["tf"]
    try:
        df_tf_control = df_tf_in.loc[("control", slice(None)), "tf"]
    except KeyError:
        df_tf_control = pd.DataFrame({"signature_type": [np.nan], "signature": [np.nan], "tf": [np.nan]}).set_index(
            ["signature_type", "signature"]
        )["tf"]
    try:
        df_tf_db_control = df_tf_in.loc[("db_control", slice(None)), "tf"]
    except KeyError:
        df_tf_db_control = pd.DataFrame({"signature_type": [np.nan], "signature": [np.nan], "tf": [np.nan]}).set_index(
            ["signature_type", "signature"]
        )["tf"]

    plt.figure(figsize=(8, 12))
    if title:
        plt.title(title, y=1.02, fontsize=28)

    if df_tf_matched.notna().any():
        x = 0.2 * np.ones(df_tf_matched.shape[0])
        y = df_tf_matched.to_numpy()
        labels = (
            df_tf_matched.index.get_level_values("signature")
            + df_tf_matched.apply(lambda x: f" (TF={x:.1e})").to_numpy()
        )
        hscat1 = plt.scatter(x, y, s=100, c="#D03020")
        for i, label in enumerate(labels):
            if not np.isnan(y[i]):
                plt.text(
                    x[i] + 0.015,
                    y[i],
                    label,
                    ha="left",
                    va="center",
                    fontsize=10,
                    alpha=0.3,
                )
    else:
        hscat1 = None

    hbp1 = plt.boxplot(
        df_tf_control,
        positions=[0],
        showfliers=False,
        patch_artist=True,
        boxprops={"facecolor": "b"},
        whiskerprops={"color": "b"},
        capprops={"color": "b"},
    )
    hbp2 = plt.boxplot(
        df_tf_db_control,
        positions=[0],
        showfliers=False,
        patch_artist=True,
        boxprops={"facecolor": "g"},
        whiskerprops={"color": "g"},
        capprops={"color": "g"},
    )
    rng = np.random.default_rng(random_seed)
    x = 0.2 + rng.uniform(-0.1, 0.1, size=df_tf_control.shape[0])
    y = df_tf_control.to_numpy()
    labels = df_tf_control.index.get_level_values("signature")
    hscat2 = plt.scatter(x, y, s=100, c="#3390DD")
    for i, label in enumerate(labels):
        if not np.isnan(y[i]):
            plt.text(
                x[i] + 0.015,
                y[i],
                label,
                ha="left",
                va="center",
                fontsize=10,
                alpha=0.3,
            )
    x = 0.2 + rng.uniform(-0.1, 0.1, size=df_tf_db_control.shape[0])
    y = df_tf_db_control.to_numpy()
    labels = df_tf_db_control.index.get_level_values("signature")
    hscat3 = plt.scatter(x, y, s=100, c="g")
    for i, label in enumerate(labels):
        if not np.isnan(y[i]):
            plt.text(
                x[i] + 0.015,
                y[i],
                label,
                ha="left",
                va="center",
                fontsize=10,
                alpha=0.3,
            )
    if any_nonzero_tf:  # if all values are nan or 0, we cannot set log scale
        plt.yscale("log")
    else:
        print("WARNING: Could not set plot to log scale")
    plt.xticks([])
    plt.xlim(-0.2, 0.5)
    plt.ylabel("Measured tumor fraction")
    plt.legend(
        [hscat1, hscat2, hbp1["boxes"][0], hscat3, hbp2["boxes"][0]],
        [
            "Matched",
            "Individual controls",
            "Control distribution",
            "db_controls",
            "db_control distribution",
        ],
        bbox_to_anchor=[1.01, 1],
    )
    for line in hbp1["medians"]:
        # get position data for median line
        x, y = line.get_xydata()[0]  # top of median line
        # overlay median value
        if not np.isnan(x) and not np.isnan(y):
            plt.text(
                x,
                y,
                f"{y:.1e}" if y > zero_tf_fill else "0",
                ha="right",
                va="center",
                color="b",
                fontsize=16,
            )  # draw above, centered
    for line in hbp2["medians"]:
        # get position data for median line
        x, y = line.get_xydata()[0]  # top of median line
        # overlay median value
        if not np.isnan(x) and not np.isnan(y):
            plt.text(
                x,
                y,
                f"{y:.1e}" if y > zero_tf_fill else "0",
                ha="right",
                va="center",
                color="g",
                fontsize=16,
            )  # draw above, centered


def get_tf_from_filtered_data(
    df_features_in: pd.DataFrame,
    df_signatures_in: pd.DataFrame,
    title=None,
    denom_ratio=None,
    *,
    plot_results=False,
):
    """
    Calculate tumor fraction from filtered dataframes
    """
    df_features_in_intersected = (
        df_features_in.join(
            df_signatures_in.groupby(level=["chrom", "pos"]).size().astype(bool).rename("locus_in_signature"),
            how="inner",
        )
        .dropna(subset=["locus_in_signature"])
        .query("locus_in_signature")
        .drop(columns=["locus_in_signature"])
    )  # retaun only loci that are in the signature df - they might have been filtered out
    df_supporting_reads_per_locus = (
        df_features_in_intersected.reset_index()
        .groupby(["chrom", "pos", "signature", "signature_type"])
        .size()
        .rename("supporting_reads")
        .reset_index(level=["signature", "signature_type"])
    )
    df_supporting_reads = (
        (df_supporting_reads_per_locus.groupby(["signature_type", "signature"]).sum()).fillna(0).astype(int)
    )
    # fill in coverage for signatures with zero supporting reads
    wanted_index = df_signatures_in.groupby(["signature_type", "signature"])["id"].first().index
    df_supporting_reads = df_supporting_reads.reindex(wanted_index, fill_value=0)

    df_coverage = (df_signatures_in.groupby("signature").agg({"coverage": "sum"})).fillna(0).astype(int)
    df_tf = df_supporting_reads.join(df_coverage).fillna(0)
    df_tf["corrected_coverage"] = df_tf["coverage"] * denom_ratio
    df_tf["corrected_coverage"] = np.ceil(df_tf["corrected_coverage"])
    df_tf = df_tf.assign(tf=df_tf["supporting_reads"] / df_tf["corrected_coverage"]).sort_index(ascending=False)

    if plot_results:
        plot_tf(df_tf, title=title)
        plt.show()

    return (df_tf, df_supporting_reads_per_locus)


def plot_vaf_matched_unmatched(
    df_supporting_reads_per_locus: pd.DataFrame,
    df_signatures: pd.DataFrame,
):
    """
    Plot histogram of allele frequencies of all, plasma-matched and unmatched variants
    """
    fig, ax = plt.subplots(3, 1, figsize=(10, 8))
    queries = {
        "all variants": df_supporting_reads_per_locus.index,
        "matched variants": df_supporting_reads_per_locus.query("signature_type == 'matched'").index,
        "control vairants": df_supporting_reads_per_locus.query("signature_type != 'matched'").index,
    }

    colors = ["blue", "red", "green"]

    bins = np.linspace(0, 1, 50)
    sns.set_style("whitegrid")
    for i, (quary_name, index_flt) in enumerate(queries.items()):
        sns.histplot(
            data=df_signatures.loc[index_flt]["af"],
            bins=bins,
            color=colors[i],
            ax=ax[i],
        )
        ax[i].set_title(quary_name)

    plt.tight_layout()
    plt.show()


def calc_tumor_fraction_denominator_ratio(featuremap_df_file: str, srsnv_metadata_json: str, read_filter_query: str):
    """
    Calculate the ratio of filtered to total reads from the single_read_snv training dataframe
    Parameters
    ----------
    featuremap_df_file: str
        featuremap df file (parquet file), training dataframe for single_read_snv
    srsnv_metadata_json: str
        single_read_snv metadata json file, includes true-positive filtering funnel
    read_filter_query: str
        query to filter the dataframe
    Returns
    -------
    denom_ratio: float
        ratio of filtered to total reads
    """
    df_model_data = pd.read_parquet(featuremap_df_file, engine="fastparquet")
    df_model_data = df_model_data.rename(columns=lambda x: x.lower())
    df_model_data["filt"] = 1  # by definition, all reads in the featuremap training dataset pass filter

    # read srsnv metadata filtering funnel
    with open(srsnv_metadata_json) as f:
        srsnv_metadata = json.load(f)
    tp_filtering = pd.DataFrame(srsnv_metadata["filtering_stats"]["positive"]["filters"])
    tp_filtering = tp_filtering.drop(
        index=tp_filtering[tp_filtering["type"] == "downsample"].index
    )  # remove downsampling step
    filt_denom = tp_filtering.loc[tp_filtering.query('type == "region"').index[-1], "rows"]  # last region step
    filt_numer = tp_filtering.loc[
        tp_filtering.index[-1], "rows"
    ]  # final number of true positives (before downsampling)
    filt_ratio = filt_numer / filt_denom

    read_filter_non_filt = df_model_data.query("label").eval(read_filter_query).mean()
    denom_ratio = filt_ratio * read_filter_non_filt
    return denom_ratio, filt_ratio, read_filter_non_filt
