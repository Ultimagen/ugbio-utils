import argparse
import logging
import math
import os
import subprocess
import sys
import tempfile
import warnings
from concurrent.futures import ThreadPoolExecutor
from os.path import basename, dirname
from os.path import join as pjoin

import numpy as np
import pandas as pd
import pysam
from ugbio_core.logger import logger
from ugbio_core.vcf_utils import VcfUtils
from ugbio_core.vcfbed import vcftools

from ugbio_featuremap import somatic_featuremap_inference_utils
from ugbio_featuremap.featuremap_utils import FeatureMapFields
from ugbio_featuremap.somatic_featuremap_utils import integrate_tandem_repeat_features

warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)


info_fields_for_training = ["TR_DISTANCE", "TR_LENGTH", "TR_SEQ_UNIT_LENGTH"]

format_fields_for_training = [
    FeatureMapFields.VAF.value,
    FeatureMapFields.RAW_VAF.value,
    FeatureMapFields.FILT.value,
    FeatureMapFields.DP_FILT.value,
    FeatureMapFields.ADJ_REF_DIFF.value,
    FeatureMapFields.DP_MAPQ60.value,
    FeatureMapFields.DUP.value,
    FeatureMapFields.REV.value,
    FeatureMapFields.SCST.value,
    FeatureMapFields.SCED.value,
]
format_mpileup_fields_for_training = ["ref_counts_pm_2", "nonref_counts_pm_2"]

added_format_features = {
    "ALT_READS": ["number of supporting reads for the alternative allele", "Integer"],
    "PASS_ALT_READS": ["number of passed supporting reads for the alternative allele", "Integer"],
    "MQUAL_MEAN": ["mean value of MQUAL", "Float"],
    "SNVQ_MEAN": ["mean value of SNVQ", "Float"],
    "MQUAL_MAX": ["mean value of MQUAL", "Float"],
    "SNVQ_MAX": ["mean value of SNVQ", "Float"],
    "MQUAL_MIN": ["mean value of MQUAL", "Float"],
    "SNVQ_MIN": ["mean value of SNVQ", "Float"],
    "COUNT_DUPLICATE": ["number of duplicate reads", "Integer"],
    "COUNT_NON_DUPLICATE": ["number of non-duplicate reads", "Integer"],
    "REVERSE_COUNT": ["number of reverse strand reads", "Integer"],
    "FORWARD_COUNT": ["number of forward strand reads", "Integer"],
    "EDIST_MEAN": ["mean value of EDIST", "Float"],
    "EDIST_MAX": ["max value of EDIST", "Float"],
    "EDIST_MIN": ["min value of EDIST", "Float"],
    "RL_MEAN": ["mean value of RL", "Float"],
    "RL_MAX": ["max value of RL", "Float"],
    "RL_MIN": ["min value of RL", "Float"],
    "SCST_NUM_READS": ["number of soft clip start reads", "Integer"],
    "SCED_NUM_READS": ["number of soft clip end reads", "Integer"],
    "MAP0_COUNT": ["number of reads with mapping quality 0", "Integer"],
}
added_info_features = {
    "REF_ALLELE": ["reference allele", "String"],
    "ALT_ALLELE": ["alternative allele", "String"],
}
columns_for_aggregation = [
    FeatureMapFields.MQUAL.value,
    FeatureMapFields.SNVQ.value,
    FeatureMapFields.MAPQ.value,
    FeatureMapFields.EDIST.value,
    FeatureMapFields.RL.value,
]
ORIGINAL_RECORD_INDEX_FIELD = "record_index"


def process_sample_columns(df_variants, prefix):  # noqa: C901
    """
    Process columns for a sample with given prefix (t_ or n_).

    This function processes various columns for tumor or normal samples,
    including alternative reads, aggregation features, duplicates, and
    padding reference counts.

    Parameters
    ----------
    df_variants : pd.DataFrame
        DataFrame containing variant information.
    prefix : str
        Sample prefix ('t_' for tumor, 'n_' for normal).

    Returns
    -------
    pd.DataFrame
        Modified DataFrame with processed sample columns.
    """

    def add_agg_features(df, feature_name, prefix):
        df[f"{prefix}{feature_name}_min"] = df[f"{prefix}{feature_name}"].apply(
            lambda x: min(x) if x is not None and len(x) > 0 and None not in x else float("nan")
        )
        df[f"{prefix}{feature_name}_max"] = df[f"{prefix}{feature_name}"].apply(
            lambda x: max(x) if x is not None and len(x) > 0 and None not in x else float("nan")
        )
        df[f"{prefix}{feature_name}_mean"] = df[f"{prefix}{feature_name}"].apply(
            lambda x: sum(x) / len(x) if x is not None and len(x) > 0 and None not in x else float("nan")
        )
        return df

    def parse_is_duplicate(df, dup_colname, prefix):
        df[f"{prefix}count_duplicate"] = df[dup_colname].apply(
            lambda x: sum(x) if x is not None and len(x) > 0 and None not in x else float("nan")
        )
        df[f"{prefix}count_non_duplicate"] = df[dup_colname].apply(
            lambda x: sum(1 for val in x if val == 0)
            if x is not None and len(x) > 0 and None not in x
            else float("nan")
        )
        return df

    def parse_padding_ref_counts(df_variants, ref_counts_colname, non_ref_counts_colname):
        if ref_counts_colname not in df_variants.columns:
            raise ValueError(f"Column {ref_counts_colname} not found in df_variants")

        if df_variants.empty:
            raise ValueError("df_variants is empty, cannot parse padding ref counts")

        if df_variants[ref_counts_colname].isna().all():
            raise ValueError(f"Column {ref_counts_colname} is empty")

        padding_counts_length = len(df_variants[ref_counts_colname].iloc[0])
        # Handle ref_count
        ref_df = pd.DataFrame(
            df_variants[ref_counts_colname].tolist(), columns=[f"{prefix}ref{i}" for i in range(padding_counts_length)]
        )
        for col in ref_df.columns:
            df_variants[col] = ref_df[col].to_numpy()

        # Handle nonref_count
        nonref_df = pd.DataFrame(
            df_variants[non_ref_counts_colname].tolist(),
            columns=[f"{prefix}nonref{i}" for i in range(padding_counts_length)],
        )
        for col in nonref_df.columns:
            df_variants[col] = nonref_df[col].to_numpy()

        return df_variants

    def count_num_sc_reads(df_variants, prefix):
        df_variants[f"{prefix}scst_num_reads"] = df_variants[f"{prefix}scst"].apply(
            lambda x: sum((v is not None) and (v > 0) for v in x) if isinstance(x, (tuple, list)) else np.nan  # noqa: UP038
        )
        df_variants[f"{prefix}sced_num_reads"] = df_variants[f"{prefix}sced"].apply(
            lambda x: sum((v is not None) and (v > 0) for v in x) if isinstance(x, (tuple, list)) else np.nan  # noqa: UP038
        )
        return df_variants

    def count_mapq0_reads(df_variants, prefix):
        df_variants[f"{prefix}map0_count"] = df_variants[f"{prefix}mapq"].apply(
            lambda x: sum(v == 0 for v in x) if x is not None else np.nan
        )
        return df_variants

    """Process columns for a sample with given prefix (t_ or n_)"""
    # Process alt_reads
    df_variants[f"{prefix}alt_reads"] = [tup[1] for tup in df_variants[f"{prefix}ad"]]
    # Process pass_alt_reads;
    df_variants[f"{prefix}pass_alt_reads"] = df_variants[f"{prefix}{FeatureMapFields.FILT.value.lower()}"].apply(
        lambda x: sum(x) if x is not None and len(x) > 0 and None not in x else float("nan")
    )
    # Process aggregations for each column
    for colname in columns_for_aggregation:
        colname_lower = colname.lower()
        df_variants = add_agg_features(df_variants, f"{colname_lower}", prefix)

    # Process duplicates
    df_variants = parse_is_duplicate(df_variants, f"{prefix}{FeatureMapFields.DUP.value.lower()}", prefix)
    df_variants = parse_padding_ref_counts(
        df_variants,
        f"{prefix}{format_mpileup_fields_for_training[0]}",
        f"{prefix}{format_mpileup_fields_for_training[1]}",
    )
    # Process forward/reverse counts
    df_variants[[f"{prefix}forward_count", f"{prefix}reverse_count"]] = df_variants[f"{prefix}rev"].apply(
        lambda x: pd.Series({"num0": 0, "num1": 0})
        if x is None or (isinstance(x, float) and pd.isna(x))
        else pd.Series({"num0": [v for v in x if v in (0, 1)].count(0), "num1": [v for v in x if v in (0, 1)].count(1)})
    )
    df_variants = count_num_sc_reads(df_variants, prefix)
    df_variants = count_mapq0_reads(df_variants, prefix)

    return df_variants


def df_sfm_fields_transformation(df_variants):  # noqa: C901
    """
    Transform somatic featuremap fields for both tumor and normal samples.

    Processes both tumor ('t_') and normal ('n_') sample columns and adds
    reference and alternative allele information.

    Parameters
    ----------
    df_variants : pd.DataFrame
        DataFrame containing variant information for tumor and normal samples.

    Returns
    -------
    pd.DataFrame
        Transformed DataFrame with processed fields for both samples and
        added allele information.
    """
    # Process both tumor and normal samples
    for prefix in ["t_", "n_"]:
        df_variants = process_sample_columns(df_variants, prefix)
    df_variants["ref_allele"] = [tup[0] for tup in df_variants["t_alleles"]]
    df_variants["alt_allele"] = [tup[1] for tup in df_variants["t_alleles"]]
    df_variants["n_dp"] = df_variants["n_dp"].fillna(df_variants["n_ref2"] + df_variants["n_nonref2"])

    return df_variants


def add_fields_to_header(hdr, added_format_features, added_info_features):
    """
    Add custom FORMAT and INFO fields to VCF header.

    Parameters
    ----------
    hdr : pysam.VariantHeader
        VCF header object to modify.
    added_format_features : dict
        Dictionary of FORMAT fields to add with their descriptions and types.
    added_info_features : dict
        Dictionary of INFO fields to add with their descriptions and types.
    """
    for field in added_format_features:
        field_type = added_format_features[field][1]
        field_description = added_format_features[field][0]
        hdr.formats.add(field, 1, field_type, field_description)
    for field in added_info_features:
        field_type = added_info_features[field][1]
        field_description = added_info_features[field][0]
        hdr.info.add(field, 1, field_type, field_description)


def process_vcf_records_serially(vcfin, df_variants, hdr, vcfout, write_agg_params):
    """
    Process VCF records serially by iterating through both VCF and DataFrame simultaneously.

    Parameters
    ----------
    vcfin : pysam.VariantFile
        Input VCF file object for reading.
    df_variants : pd.DataFrame
        DataFrame containing variant information with aggregated features.
    hdr : pysam.VariantHeader
        VCF header object for the output file.
    vcfout : pysam.VariantFile
        Output VCF file object for writing.
    write_agg_params : bool
        Whether to write aggregated parameters to the output VCF.
    """

    # Use original record order for processing
    df_variants_sorted = df_variants.sort_values(f"t_{ORIGINAL_RECORD_INDEX_FIELD}").reset_index(drop=True)
    df_iter = iter(df_variants_sorted.itertuples(index=False))
    current_df_record = next(df_iter, None)

    for vcf_row in vcfin:
        vcf_chrom = vcf_row.chrom
        vcf_pos = vcf_row.pos
        vcf_alt = vcf_row.alleles[1] if len(vcf_row.alleles) > 1 else None

        # Skip to matching DataFrame record or process without match
        while current_df_record is not None and (
            current_df_record.t_chrom < vcf_chrom  # case1: next chromosome
            or (current_df_record.t_chrom == vcf_chrom and current_df_record.t_pos < vcf_pos)  # case2: next position
            or (
                current_df_record.t_chrom == vcf_chrom
                and current_df_record.t_pos == vcf_pos
                and current_df_record.alt_allele != vcf_alt  # case3: alt allele for multi-allelic site
            )
        ):
            current_df_record = next(df_iter, None)

        # Check if we have a matching record
        if (
            current_df_record is not None
            and current_df_record.t_chrom == vcf_chrom
            and current_df_record.t_pos == vcf_pos
            and current_df_record.alt_allele == vcf_alt
        ):
            if write_agg_params:
                # Add INFO fields
                for key in added_info_features:
                    vcf_row.info[key.upper()] = getattr(current_df_record, key.lower())

                # Add FORMAT fields
                for key in added_format_features:
                    tumor_value = getattr(current_df_record, f"t_{key.lower()}")
                    normal_value = getattr(current_df_record, f"n_{key.lower()}")

                    if pd.notna(tumor_value):
                        vcf_row.samples[0][key.upper()] = tumor_value
                    else:
                        vcf_row.samples[0][key.upper()] = None

                    if pd.notna(normal_value):
                        vcf_row.samples[1][key.upper()] = normal_value
                    else:
                        vcf_row.samples[1][key.upper()] = None

                # Add XGBoost probability if available
                if "XGB_PROBA" in hdr.info and hasattr(current_df_record, "xgb_proba"):
                    vcf_row.info["XGB_PROBA"] = current_df_record.xgb_proba

        vcfout.write(vcf_row)


def read_merged_tumor_normal_vcf(
    vcf_file: str, custom_info_fields: list[str], fillna_dict: dict[str, object] = None, chrom: str = None
) -> "pd.DataFrame":
    """
    Reads a merged tumor-normal VCF file and returns a concatenated DataFrame with prefixed columns
    for tumor and normal samples.

    Parameters
    ----------
    vcf_file : str
        Path to the VCF file containing both tumor and normal samples.
    custom_info_fields : list of str
        List of custom INFO fields to extract from the VCF.
    fillna_dict : dict of str to object, optional
        Dictionary specifying values to fill missing data for each field.
        Defaults to None.
    chrom : str, optional
        Chromosome to filter the VCF records. If None, all chromosomes are
        included. Defaults to None.

    Returns
    -------
    pd.DataFrame
        A DataFrame with tumor columns prefixed by 't_' and normal columns
        prefixed by 'n_'. Missing values are filled according to `fillna_dict` if provided.
    """
    # Read to df
    if chrom is not None:
        df_tumor = vcftools.get_vcf_df(vcf_file, sample_id=0, custom_info_fields=custom_info_fields, chromosome=chrom)
        df_normal = vcftools.get_vcf_df(vcf_file, sample_id=1, custom_info_fields=custom_info_fields, chromosome=chrom)
    else:
        df_tumor = vcftools.get_vcf_df(vcf_file, sample_id=0, custom_info_fields=custom_info_fields)
        df_normal = vcftools.get_vcf_df(vcf_file, sample_id=1, custom_info_fields=custom_info_fields)

    # make all colnames in lowercase.
    for colname in df_tumor.columns:
        df_tumor[colname.lower()] = df_tumor[colname]
        df_normal[colname.lower()] = df_normal[colname]
        if colname != colname.lower():
            df_tumor = df_tumor.drop(columns=[colname])
            df_normal = df_normal.drop(columns=[colname])

    # keep original records order
    df_tumor[ORIGINAL_RECORD_INDEX_FIELD] = range(1, len(df_tumor) + 1)
    # merge dataframes
    df_tumor_normal = pd.concat([df_tumor.add_prefix("t_"), df_normal.add_prefix("n_")], axis=1)

    # create merged fillna dict
    if fillna_dict:
        fillna_dict_merged = {}
        for key in fillna_dict:  # noqa: PLC0206
            fillna_dict_merged[f"t_{key}"] = fillna_dict[key]
            fillna_dict_merged[f"n_{key}"] = fillna_dict[key]
        df_tumor_normal = df_tumor_normal.fillna(fillna_dict_merged)

    return df_tumor_normal


def featuremap_fields_aggregation(  # noqa: C901, PLR0915
    somatic_featuremap_vcf_file: str,
    output_vcf: str,
    filter_tags=None,
    genomic_region: str = None,
    xgb_model_file: str = None,
    write_agg_params: bool = True,  # noqa: FBT001, FBT002
    verbose: bool = True,  # noqa: FBT001, FBT002
) -> str:
    """
    Write the vcf file with the aggregated fields and the xgb probability.

    Parameters
    ----------
    somatic_featuremap_vcf_file : str
        Path to input somatic featuremap VCF file.
    output_vcf : str
        Path to output VCF file with aggregated fields.
    filter_tags : str, optional
        Filter tags to apply on the VCF file. Defaults to None.
    genomic_region : str, optional
        Specific genomic interval to process. Defaults to None.
    xgb_model_file : str, optional
        Path to XGBoost model file for inference. Defaults to None.
    write_agg_params : bool, optional
        Whether to write aggregated parameters to output. Defaults to True.
    verbose : bool, optional
        Whether to enable verbose logging. Defaults to True.

    Returns
    -------
    str
        Path to the output VCF file with aggregated fields and XGBoost probabilities.
    """
    if verbose:
        logger.setLevel(logging.DEBUG)
        for handler in logger.handlers:
            handler.setLevel(logging.DEBUG)

    vcf_utils = VcfUtils()
    # filter vcf file for the given filter tags and genomic interval
    filter_string = f"-f {filter_tags}" if filter_tags else ""
    interval_string = f"-r {genomic_region}" if genomic_region else ""
    extra_args = f"{filter_string} {interval_string}"
    temp_dir = tempfile.mkdtemp(dir=os.path.dirname(output_vcf))
    try:
        filtered_featuremap = pjoin(
            temp_dir, basename(somatic_featuremap_vcf_file).replace(".vcf.gz", ".sorted.filtered.vcf.gz")
        )
        vcf_utils.view_vcf(
            somatic_featuremap_vcf_file, filtered_featuremap, extra_args=extra_args, n_threads=os.cpu_count()
        )
        vcf_utils.index_vcf(filtered_featuremap)

        custom_info_fields = (
            format_fields_for_training
            + format_mpileup_fields_for_training
            + info_fields_for_training
            + columns_for_aggregation
        )
        df_variants = read_merged_tumor_normal_vcf(filtered_featuremap, custom_info_fields=custom_info_fields)
        if len(df_variants) > 0:
            df_variants = df_sfm_fields_transformation(df_variants)

            if xgb_model_file is not None:
                xgb_clf_es = somatic_featuremap_inference_utils.load_model(xgb_model_file)
                model_features = xgb_clf_es.get_booster().feature_names
                logger.info(f"loaded model. model features: {model_features}")
                df_variants["xgb_proba"] = somatic_featuremap_inference_utils.predict(xgb_clf_es, df_variants)
            # Write df_variants to parquet file
            parquet_output = output_vcf.replace(".vcf.gz", "_featuremap.parquet")
            df_variants.to_parquet(parquet_output, index=False)
            logger.info(f"Written feature map dataframe to {parquet_output}")

            # Serial processing to avoid expensive lookups for each record
            with pysam.VariantFile(filtered_featuremap) as vcfin:
                hdr = vcfin.header
                add_fields_to_header(hdr, added_format_features, added_info_features)
                if xgb_model_file is not None:
                    hdr.info.add("XGB_PROBA", 1, "Float", "XGBoost model predicted probability")
                with pysam.VariantFile(output_vcf, mode="w", header=hdr) as vcfout:
                    process_vcf_records_serially(vcfin, df_variants, hdr, vcfout, write_agg_params)
                vcfout.close()
                vcfin.close()
            pysam.tabix_index(output_vcf, preset="vcf", min_shift=0, force=True)
            return output_vcf
        else:
            logger.warning(f"No variants found in the filtered VCF: {filtered_featuremap}")
            # Create an empty VCF with the updated header
            with pysam.VariantFile(filtered_featuremap) as vcfin:
                hdr = vcfin.header
                add_fields_to_header(hdr, added_format_features, added_info_features)
                if xgb_model_file is not None:
                    hdr.info.add("XGB_PROBA", 1, "Float", "XGBoost model predicted probability")
                with pysam.VariantFile(output_vcf, mode="w", header=hdr) as vcfout:
                    pass  # Just create the file with header
                vcfout.close()
                vcfin.close()
            pysam.tabix_index(output_vcf, preset="vcf", min_shift=0, force=True)
            return output_vcf
    finally:
        # Clean up temporary directory
        if os.path.exists(temp_dir):
            for filename in os.listdir(temp_dir):
                file_path = os.path.join(temp_dir, filename)
                try:
                    if os.path.isfile(file_path) or os.path.islink(file_path):
                        os.unlink(file_path)
                except Exception as e:
                    logger.error(f"Error removing temporary file {file_path}: {e}")


def collapse_bed_by_chunks(bed_file: str, num_chunks: int) -> list[str]:
    """
    Collapse a sorted BED file into equal-sized chunks by number of rows.

    This function reads a BED file and divides it into approximately equal-sized chunks
    based on the number of rows. Each chunk is represented by a single genomic interval
    spanning from the first to the last position within that chunk.

    Parameters
    ----------
    bed_file : str
        Path to input sorted BED file containing columns (chrom, start, end).
    num_chunks : int
        Number of chunks to create. If the number of rows in the BED file
        is less than num_chunks, the actual number of chunks will be
        adjusted accordingly.

    Returns
    -------
    list of str
        List of genomic intervals in the format "chrom:start-end",
        where each interval represents a collapsed chunk.

    Notes
    -----
    - The function assumes the input BED file has no header and contains
      exactly 3 columns: chromosome, start position, and end position.
    - Chunks are created based on equal distribution of rows, not equal
      genomic distance.
    - Each chunk preserves the chromosome of its first row; this assumes
      chunks don't span multiple chromosomes.

    Examples
    --------
    >>> intervals = collapse_bed_by_chunks("input.bed", num_chunks=4)
    >>> print(intervals)
    ['chr1:1000-5000', 'chr1:5001-10000', 'chr2:1000-8000', 'chr2:8001-15000']
    """
    df_bed_regions = pd.read_csv(bed_file, sep="\t", header=None, names=["chrom", "start", "end"], usecols=[0, 1, 2])
    n = len(df_bed_regions)
    # Handle case where rows < chunks: adjust num_chunks
    num_chunks = min(num_chunks, n)
    # Compute chunk size (number of rows per chunk, roughly equal)
    chunk_size = math.ceil(n / num_chunks)
    # Collapse into chunks
    collapsed = []
    for i in range(0, n, chunk_size):
        chunk_df_bed_regions = df_bed_regions.iloc[i : i + chunk_size]
        chrom = chunk_df_bed_regions.iloc[0]["chrom"]
        start = chunk_df_bed_regions.iloc[0]["start"]
        end = chunk_df_bed_regions.iloc[-1]["end"]
        collapsed.append((chrom, start, end))

    # Write output
    genomic_regions = []
    for chrom, start, end in collapsed:
        genomic_regions.append(f"{chrom}:{start+1}-{end}")
    return genomic_regions


def featuremap_fields_aggregation_on_an_interval_list(
    featuremap_vcf_file: str,
    output_vcf: str,
    genomic_regions_bed_file: str,
    filter_tags=None,
    xgb_model_file: str = None,
    verbose: bool = True,  # noqa: FBT001, FBT002
) -> None:
    """
    Apply featuremap fields aggregation on an interval list.

    Parameters
    ----------
    featuremap_vcf_file : str
        The input featuremap VCF file.
    output_vcf : str
        The output pileup VCF file.
    genomic_regions_bed_file : str
        genomic regions list in BED file.
    filter_tags : str, optional
        Filter tags to apply. Defaults to None.
    xgb_model_file : str, optional
        Path to XGBoost model file for inference. Defaults to None.
    verbose : bool, optional
        Whether to enable verbose logging. Defaults to True

    Returns
    -------
    str
        The output VCF file including the aggregated fields and the XGBoost probability.
    """
    if not output_vcf.endswith(".vcf.gz"):
        logger.debug("adding .vcf.gz suffix to the output vcf file")
        output_vcf = output_vcf + ".vcf.gz"

    with tempfile.TemporaryDirectory(dir=dirname(output_vcf)):
        num_cpus = os.cpu_count()
        genomic_regions = collapse_bed_by_chunks(genomic_regions_bed_file, num_chunks=num_cpus)

        params = [
            (
                featuremap_vcf_file,
                f"{output_vcf}.{genomic_region}.int_list.vcf.gz",
                filter_tags,
                genomic_region,
                xgb_model_file,
                verbose,
            )
            for genomic_region in genomic_regions
        ]
        num_cpus = os.cpu_count()
        with ThreadPoolExecutor(max_workers=num_cpus) as executor:
            results = list(executor.map(lambda p: featuremap_fields_aggregation(*p), params))

        # Write each string to the file
        interval_list_file = pjoin(tempfile.gettempdir(), "interval_vcf_files.list")
        with open(interval_list_file, "w") as file:
            for interval_vcf_file in results:
                file.write(interval_vcf_file + "\n")

        cmd = (
            f"bcftools concat -f {interval_list_file} -a | "
            f"bcftools sort - -Oz -o {output_vcf} && "
            f"bcftools index -t {output_vcf}"
        )
        logger.debug(cmd)
        subprocess.check_call(cmd, shell=True)  # noqa: S602
    return output_vcf


def __parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="somatic featuremap fields transformation",
        description=run.__doc__,
    )
    parser.add_argument(
        "-sfm",
        "--somatic_featuremap",
        type=str,
        required=True,
        help="""somatic featuremap vcf file""",
    )
    parser.add_argument(
        "-filter_string",
        "--filter_string",
        type=str,
        required=False,
        default="PASS",
        help="""filter tags to apply on the somatic featuremap pileup vcf file""",
    )
    parser.add_argument(
        "-o",
        "--output_vcf",
        type=str,
        required=True,
        help="""Output pileup vcf file""",
    )
    parser.add_argument(
        "-i",
        "--genomic_regions_bed_file",
        type=str,
        required=True,
        help="""genomic regions BED file""",
    )
    parser.add_argument(
        "-g",
        "--genome_file",
        type=str,
        required=True,
        help="""Genome FASTA index (.fai) file""",
    )
    parser.add_argument(
        "-ref_tr",
        "--ref_tr_file",
        type=str,
        required=True,
        help="""Reference tandem repeat file in BED format""",
    )
    parser.add_argument(
        "-xgb_model",
        "--xgb_model_file",
        type=str,
        required=False,
        default=None,
        help="""XGBoost model file for inference (optional)""",
    )
    parser.add_argument(
        "-v",
        "--disable_verbose",
        action="store_false",
        default=True,
        help="""Disable verbose output and debug messages. When used, verbose mode is turned off
                (default: verbose enabled)""",
    )
    return parser.parse_args(argv[1:])


def run(argv):
    """
    Add aggregated parameters and xgb probability to the featuremap pileup vcf file.

    Parameters
    ----------
    argv : list of str
        Command line arguments including the script name.
    """
    args_in = __parse_args(argv)

    # add tandem repeat features
    out_dir = dirname(args_in.output_vcf)
    sfm_with_tr = integrate_tandem_repeat_features(
        args_in.somatic_featuremap, args_in.ref_tr_file, args_in.genome_file, out_dir
    )

    featuremap_fields_aggregation_on_an_interval_list(
        sfm_with_tr,
        args_in.output_vcf,
        args_in.genomic_regions_bed_file,
        args_in.filter_string,
        args_in.xgb_model_file,
        args_in.disable_verbose,
    )


def main():
    """
    Main entry point for the script.

    Calls run() with command line arguments from sys.argv.
    """
    run(sys.argv)


if __name__ == "__main__":
    main()
