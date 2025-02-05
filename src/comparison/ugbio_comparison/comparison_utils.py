from collections import defaultdict

import numpy as np
import pandas as pd
import pyfaidx
import pysam
import ugbio_core.concordance.flow_based_concordance as fbc
import ugbio_core.vcfbed.variant_annotation as annotation
from ugbio_core.consts import DEFAULT_FLOW_ORDER
from ugbio_core.logger import logger
from ugbio_core.vcfbed import bed_writer, vcftools


def _fix_errors(df: pd.DataFrame) -> pd.DataFrame:
    """Parses dataframe generated from the VCFEVAL concordance VCF and prepares it for
     classify/classify_gt functions that only consider the genotypes of the call and the base
    only rather than at the classification that VCFEVAL produced

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe

    Returns
    -------
    pd.DataFrame
        Output dataframe,
    """

    # remove genotypes of variants that were filtered out and thus are false negatives
    # (VCFEVAL outputs UG genotype for ignored genotypes too and they are classified downstream
    # as true positives if we do not make this modification)
    fix_tp_fn_loc = (df["call"] == "IGN") & ((df["base"] == "FN") | (df["base"] == "FN_CA"))
    replace = df.loc[fix_tp_fn_loc, "gt_ultima"].apply(lambda x: (None,))
    df.loc[replace.index, "gt_ultima"] = replace

    # fix all the places in which vcfeval returns a good result, but the genotype is not adequate
    # in these cases we change the genotype of the gt to be adequate with the classify function as follow:
    # (TP,TP), (TP,None) - should put the values of ultima in the gt
    df.loc[(df["call"] == "TP") & ((df["base"] == "TP") | (df["base"].isna())), "gt_ground_truth"] = df[
        (df["call"] == "TP") & ((df["base"] == "TP") | (df["base"].isna()))
    ]["gt_ultima"]

    # (FP_CA,FN_CA), (FP_CA,None) - Fake a genotype from ultima such that one of the alleles is the same (and only one)
    df.loc[(df["call"] == "FP_CA") & ((df["base"] == "FN_CA") | (df["base"].isna())), "gt_ground_truth"] = df[
        (df["call"] == "FP_CA") & ((df["base"] == "FN_CA") | (df["base"].isna()))
    ]["gt_ultima"].apply(
        lambda x: ((x[0], x[0]) if (len(x) < 2 or (x[1] == 0)) else ((x[1], x[1]) if (x[0] == 0) else (x[0], 0)))  # noqa PLR2004
    )
    return df


def __map_variant_to_dict(variant: pysam.VariantRecord) -> defaultdict:
    """Converts a line from vcfeval concordance VCF to a dictionary. The following fields are extracted
    call genotype, base genotype, qual, chromosome, position, ref, alt and all values from the INFO column

    Parameters
    ----------
    variant : pysam.VariantRecord
        VCF record

    Returns
    -------
    defaultdict
        Output dictionary
    """
    call_sample_ind = 1
    gtr_sample_ind = 0

    return defaultdict(
        lambda: None,
        variant.info.items()
        + [
            ("GT_ULTIMA", variant.samples[call_sample_ind]["GT"]),
            ("GT_GROUND_TRUTH", variant.samples[gtr_sample_ind]["GT"]),
            ("QUAL", variant.qual),
            ("CHROM", variant.chrom),
            ("POS", variant.pos),
            ("REF", variant.ref),
            ("ALLELES", variant.alleles),
        ],
    )


def vcf2concordance(  # noqa PLR0915 C901
    raw_calls_file: str,
    concordance_file: str,
    chromosome: str | None = None,
    scoring_field: str | None = None,
) -> pd.DataFrame:
    """Generates concordance dataframe

    Parameters
    ----------
    raw_calls_file : str
        File with GATK calls (.vcf.gz)
    concordance_file : str
        GenotypeConcordance/VCFEVAL output file (.vcf.gz)
    chromosome: str
        Fetch a specific chromosome (Default - all)
    scoring_field : str, optional
        The name of the INFO field that is used to score the variants.
        This value replaces the TREE_SCORE in the output data frame.
        When None TREE_SCORE is not replaced (default: None)

    No Longer Returned
    ------------------
    pd.DataFrame
    """

    if chromosome is None:
        concord_vcf = pysam.VariantFile(concordance_file)
    else:
        concord_vcf = pysam.VariantFile(concordance_file).fetch(chromosome)

    def call_filter(x):
        # Remove variants that were ignored (either outside of comparison intervals or
        # filtered out).
        return not (
            (x["CALL"] in {"IGN", "OUT"} and x["BASE"] is None)
            or (x["CALL"] in {"IGN", "OUT"} and x["BASE"] in {"IGN", "OUT"})
            or (x["CALL"] is None and x["BASE"] in {"IGN", "OUT"})
        )

    concord_vcf_extend = filter(call_filter, (__map_variant_to_dict(variant) for variant in concord_vcf))

    columns = [
        "CHROM",
        "POS",
        "QUAL",
        "REF",
        "ALLELES",
        "GT_ULTIMA",
        "GT_GROUND_TRUTH",
        "SYNC",
        "CALL",
        "BASE",
        "STR",
        "RU",
        "RPA",
    ]
    column_names = [x.lower() for x in columns]
    concordance_df = pd.DataFrame([[x[y] for y in columns] for x in concord_vcf_extend], columns=column_names)

    # make the gt_ground_truth compatible with GC
    concordance_df["gt_ground_truth"] = concordance_df["gt_ground_truth"].map(
        lambda x: (None, None) if x == (None,) else x
    )

    concordance_df["indel"] = concordance_df["alleles"].apply(lambda x: len({len(y) for y in x}) > 1)
    concordance_df = _fix_errors(concordance_df)

    def classify(x: defaultdict | pd.Series | dict) -> str:
        """Classify a record as true positive / false positive / false negative by matching the alleles.
        TP will be called if some of the alleles match

        Parameters
        ----------
        x : defaultdict, pd.Series or dict
            Input record

        Returns
        -------
        str
            classification
        """
        if x["gt_ultima"] == (None, None) or x["gt_ultima"] == (None,):
            return "fn"

        if x["gt_ground_truth"] == (None, None) or x["gt_ground_truth"] == (None,):
            return "fp"

        # If both gt_ultima and gt_ground_truth are not none:
        set_gtr = set(x["gt_ground_truth"]) - {0}
        set_ultima = set(x["gt_ultima"]) - {0}

        if len(set_gtr & set_ultima) > 0:
            return "tp"

        if len(set_ultima - set_gtr) > 0:
            return "fp"

        # If it is not tp or fp, then return fn:
        return "fn"

    concordance_df["classify"] = concordance_df.apply(classify, axis=1, result_type="reduce")

    def classify_gt(x: defaultdict | pd.Series | dict) -> str:
        """Classify a record as true positive / false negative / false positive. True positive requires
        match in the alleles and genotypes

        Parameters
        ----------
        x : defaultdict, pd.Series or dict
            Input record

        Returns
        -------
        str
            Classification
        """
        n_ref_gtr = len([y for y in x["gt_ground_truth"] if y == 0])
        n_ref_ultima = len([y for y in x["gt_ultima"] if y == 0])

        if x["gt_ultima"] == (None, None) or x["gt_ultima"] == (None,):
            return "fn"
        if x["gt_ground_truth"] == (None, None) or x["gt_ground_truth"] == (None,):
            return "fp"
        if n_ref_gtr < n_ref_ultima:
            return "fn"
        if n_ref_gtr > n_ref_ultima:
            return "fp"
        if x["gt_ultima"] != x["gt_ground_truth"]:
            return "fp"
        # If not fn or fp due to the reasons above:
        return "tp"

    concordance_df["classify_gt"] = concordance_df.apply(classify_gt, axis=1, result_type="reduce")
    concordance_df.loc[
        (concordance_df["classify_gt"] == "tp") & (concordance_df["classify"] == "fp"),
        "classify_gt",
    ] = "fp"

    # cases where we called wrong allele and then filtered out - are false negatives, not false positives
    called_fn = (concordance_df["base"] == "FN") | (concordance_df["base"] == "FN_CA")
    marked_fp = concordance_df["classify"] == "fp"
    concordance_df.loc[called_fn & marked_fp, "classify"] = "fn"
    marked_fp = concordance_df["classify_gt"] == "fp"
    concordance_df.loc[called_fn & marked_fp, "classify_gt"] = "fn"
    concordance_df.index = pd.Index(list(zip(concordance_df.chrom, concordance_df.pos, strict=False)))
    original = vcftools.get_vcf_df(raw_calls_file, chromosome=chromosome, scoring_field=scoring_field)

    concordance_df = concordance_df.drop("qual", axis=1)

    drop_candidates = ["chrom", "pos", "alleles", "indel", "ref", "str", "ru", "rpa"]
    if original.shape[0] > 0:
        concordance = concordance_df.join(
            original.drop(
                [x for x in drop_candidates if x in original.columns and x in concordance_df.columns],
                axis=1,
            )
        )
    else:
        concordance = concordance_df.copy()

        tmp = original.drop(
            [x for x in drop_candidates if x in original.columns and x in concordance_df.columns],
            axis=1,
        )
        for t in tmp.columns:
            concordance[t] = None

    only_ref = concordance["alleles"].apply(len) == 1
    concordance = concordance[~only_ref]

    # Marking as false negative the variants that appear in concordance but not in the
    # original VCF (even if they do show some genotype)
    missing_variants = concordance.index.difference(original.index)
    logger.info("Identified %i variants missing in the input VCF", len(missing_variants))
    missing_variants_non_fn = concordance.loc[missing_variants].query("classify!='fn'").index
    logger.warning(
        "Identified %i variants missing in the input VCF and not marked false negatives",
        len(missing_variants_non_fn),
    )
    concordance.loc[missing_variants_non_fn, "classify"] = "fn"
    concordance.loc[missing_variants_non_fn, "classify_gt"] = "fn"

    return concordance


def bed_file_length(input_bed: str) -> int:
    """Calc the number of bases in a bed file

    Parameters
    ----------
    input_bed : str
        Input Bed file

    Return
    ------
    int
        number of bases in a bed file
    """

    input_df = pd.read_csv(input_bed, sep="\t", header=None)
    input_df = input_df.iloc[:, [0, 1, 2]]
    input_df.columns = ["chr", "pos_start", "pos_end"]
    return np.sum(input_df["pos_end"] - input_df["pos_start"] + 1)


def close_to_hmer_run(
    df: pd.DataFrame,
    runfile: str,
    min_hmer_run_length: int = 10,
    max_distance: int = 10,
) -> pd.DataFrame:
    """Adds column is_close_to_hmer_run and inside_hmer_run that is T/F"""
    df["close_to_hmer_run"] = False
    df["inside_hmer_run"] = False
    run_df = bed_writer.parse_intervals_file(runfile, min_hmer_run_length)
    gdf = df.groupby("chrom")
    grun_df = run_df.groupby("chromosome")
    for chrom in gdf.groups.keys():
        gdf_ix = gdf.groups[chrom]
        grun_ix = grun_df.groups[chrom]
        pos1 = np.array(df.loc[gdf_ix, "pos"])
        pos2 = np.array(run_df.loc[grun_ix, "start"])
        pos1_closest_pos2_start = np.searchsorted(pos2, pos1) - 1
        close_dist = abs(pos1 - pos2[np.clip(pos1_closest_pos2_start, 0, None)]) < max_distance
        close_dist |= abs(pos2[np.clip(pos1_closest_pos2_start + 1, None, len(pos2) - 1)] - pos1) < max_distance
        pos2 = np.array(run_df.loc[grun_ix, "end"])
        pos1_closest_pos2_end = np.searchsorted(pos2, pos1)
        close_dist |= abs(pos1 - pos2[np.clip(pos1_closest_pos2_end - 1, 0, None)]) < max_distance
        close_dist |= abs(pos2[np.clip(pos1_closest_pos2_end, None, len(pos2) - 1)] - pos1) < max_distance
        is_inside = pos1_closest_pos2_start == pos1_closest_pos2_end
        df.loc[gdf_ix, "inside_hmer_run"] = is_inside
        df.loc[gdf_ix, "close_to_hmer_run"] = close_dist & (~is_inside)
    return df


def annotate_concordance(
    concordance_df: pd.DataFrame,
    fasta: str,
    bw_high_quality: list[str] | None = None,
    bw_all_quality: list[str] | None = None,
    annotate_intervals: list[str] | None = None,
    runfile: str | None = None,
    flow_order: str | None = DEFAULT_FLOW_ORDER,
    hmer_run_length_dist: tuple = (10, 10),
) -> tuple[pd.DataFrame, list]:
    """Annotates concordance data with information about SNP/INDELs and motifs

    Parameters
    ----------
    concordance_df : pd.DataFrame
        Concordance dataframe
    fasta : str
        Indexed FASTA of the reference genome
    bw_high_quality : list[str], optional
        Coverage bigWig file from high mapq reads  (Optional)
    bw_all_quality : list[str], optional
        Coverage bigWig file from all mapq reads  (Optional)
    annotate_intervals : list[str], optional
        Interval files for annotation
    runfile : str, optional
        bed file with positions of hmer runs (in order to mark homopolymer runs)
    flow_order : str, optional
        Flow order
    hmer_run_length_dist : tuple, optional
        tuple (min_hmer_run_length, max_distance) for marking variants near homopolymer runs

    No Longer Returned
    ------------------
    pd.DataFrame
        Annotated dataframe
    list
        list of the names of the annotations
    """

    if annotate_intervals is None:
        annotate_intervals = []

    logger.info("Marking SNP/INDEL")
    concordance_df = annotation.classify_indel(concordance_df)
    logger.info("Marking H-INDEL")
    concordance_df = annotation.is_hmer_indel(concordance_df, fasta)
    logger.info("Marking motifs")
    concordance_df = annotation.get_motif_around(concordance_df, 5, fasta)
    logger.info("Marking GC content")
    concordance_df = annotation.get_gc_content(concordance_df, 10, fasta)
    if bw_all_quality is not None and bw_high_quality is not None:
        logger.info("Calculating coverage")
        concordance_df = annotation.get_coverage(concordance_df, bw_high_quality, bw_all_quality)
    if runfile is not None:
        length, dist = hmer_run_length_dist
        logger.info("Marking homopolymer runs")
        concordance_df = close_to_hmer_run(concordance_df, runfile, min_hmer_run_length=length, max_distance=dist)
    annots = []
    if annotate_intervals is not None:
        for annotation_file in annotate_intervals:
            logger.info("Annotating intervals")
            concordance_df, annot = annotation.annotate_intervals(concordance_df, annotation_file)
            annots.append(annot)
    logger.debug("Filling filter column")  # debug since not interesting step
    concordance_df = annotation.fill_filter_column(concordance_df)

    logger.info("Filling filter column")
    if flow_order is not None:
        concordance_df = annotation.annotate_cycle_skip(concordance_df, flow_order=flow_order)
    return concordance_df, annots


def reinterpret_variants(
    concordance_df: pd.DataFrame,
    reference_fasta: str,
    *,
    ignore_low_quality_fps: bool = False,
) -> pd.DataFrame:
    """Reinterprets the variants by comparing the variant to the ground truth in flow space

    Parameters
    ----------
    concordance_df : pd.DataFrame
        Input dataframe
    reference_fasta : str
        Indexed FASTA
    ignore_low_quality_fps : bool, optional
        Shoud the low quality false positives be ignored in reinterpretation (True for mutect, default False)

    See Also
    --------
    `flow_based_concordance.py`

    No Longer Returned
    ------------------
    pd.DataFrame
        Reinterpreted dataframe
    """
    logger.info("Variants reinterpret")
    concordance_df_result = pd.DataFrame()
    fasta = pyfaidx.Fasta(reference_fasta, build_index=False, rebuild=False)
    for contig in concordance_df["chrom"].unique():
        concordance_df_contig = concordance_df.loc[concordance_df["chrom"] == contig]
        input_dict = _get_locations_to_work_on(concordance_df_contig, ignore_low_quality_fps=ignore_low_quality_fps)
        concordance_df_contig = fbc.reinterpret_variants(concordance_df_contig, input_dict, fasta)
        concordance_df_result = pd.concat([concordance_df_result, concordance_df_contig])
    return concordance_df_result


def _get_locations_to_work_on(input_df: pd.DataFrame, *, ignore_low_quality_fps: bool = False) -> dict:
    """Dictionary of  in the dataframe that we care about

    Parameters
    ----------
    input_df : pd.DataFrame
        Input
    ignore_low_quality_fps : bool, optional
        Should we ignore the low quality false positives

    Returns
    -------
    dict
        locations dictionary split between fps/fns/tps etc.

    """
    filtered_df = vcftools.FilterWrapper(input_df)
    fps = filtered_df.reset().get_fp().get_df()
    if "tree_score" in fps.columns and fps["tree_score"].dtype == np.float64 and ignore_low_quality_fps:
        cutoff = fps.tree_score.quantile(0.80)
        fps = fps.query(f"tree_score > {cutoff}")
    fns = filtered_df.reset().get_df().query('classify=="fn"')
    tps = filtered_df.reset().get_tp().get_df()
    gtr = (
        filtered_df.reset()
        .get_df()
        .loc[filtered_df.get_df()["gt_ground_truth"].apply(lambda x: x not in [(None, None), (None,)])]
        .copy()
    )
    gtr = gtr.sort_values("pos")
    ugi = (
        filtered_df.reset()
        .get_df()
        .loc[filtered_df.get_df()["gt_ultima"].apply(lambda x: x not in [(None, None), (None,)])]
        .copy()
    )
    ugi = ugi.sort_values("pos")

    pos_fps = np.array(fps.pos)
    pos_gtr = np.array(gtr.pos)
    pos_ugi = np.array(ugi.pos)
    pos_fns = np.array(fns.pos)

    result = {
        "fps": fps,
        "fns": fns,
        "tps": tps,
        "gtr": gtr,
        "ugi": ugi,
        "pos_fps": pos_fps,
        "pos_gtr": pos_gtr,
        "pos_ugi": pos_ugi,
        "pos_fns": pos_fns,
    }

    return result
