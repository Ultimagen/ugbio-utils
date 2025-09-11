import logging
import os
import re
import subprocess
import tempfile
from collections import defaultdict
from enum import Enum
from os.path import basename, dirname
from os.path import join as pjoin

import numpy as np
import pysam
from ugbio_core.consts import (
    AD,
    ALT,
    CHROM,
    DP,
    FILTER,
    GT,
    IS_CYCLE_SKIP,
    POS,
    QUAL,
    REF,
    VAF,
)
from ugbio_core.logger import logger
from ugbio_ppmseq.ppmSeq_consts import HistogramColumnNames

from ugbio_featuremap.featuremap_utils import FeatureMapFilters


class PileupFeatureMapFields(Enum):
    CHROM = CHROM
    POS = POS
    REF = REF
    ALT = ALT
    QUAL = QUAL
    FILTER = FILTER
    READ_COUNT = "X_READ_COUNT"
    FILTERED_COUNT = "X_FILTERED_COUNT"
    X_SCORE = "X_SCORE"
    X_EDIST = "X_EDIST"
    X_LENGTH = "X_LENGTH"
    X_MAPQ = "X_MAPQ"
    X_INDEX = "X_INDEX"
    X_FC1 = "X_FC1"
    X_FC2 = "X_FC2"
    X_QUAL = "X_QUAL"
    X_RN = "X_RN"
    TRINUC_CONTEXT_WITH_ALT = "trinuc_context_with_alt"
    HMER_CONTEXT_REF = "hmer_context_ref"
    HMER_CONTEXT_ALT = "hmer_context_alt"
    IS_CYCLE_SKIP = IS_CYCLE_SKIP
    IS_FORWARD = "is_forward"
    IS_DUPLICATE = "is_duplicate"
    MAX_SOFTCLIP_LENGTH = "max_softclip_length"
    X_FLAGS = "X_FLAGS"
    X_CIGAR = "X_CIGAR"
    PREV_1 = "prev_1"
    PREV_2 = "prev_2"
    PREV_3 = "prev_3"
    NEXT_1 = "next_1"
    NEXT_2 = "next_2"
    NEXT_3 = "next_3"


fields_to_collect_all_options = {
    "numeric_array_fields": [
        PileupFeatureMapFields.X_SCORE.value,
        PileupFeatureMapFields.X_EDIST.value,
        PileupFeatureMapFields.X_LENGTH.value,
        PileupFeatureMapFields.X_MAPQ.value,
        PileupFeatureMapFields.X_INDEX.value,
        PileupFeatureMapFields.X_FC1.value,
        PileupFeatureMapFields.X_FC2.value,
        PileupFeatureMapFields.MAX_SOFTCLIP_LENGTH.value,
        PileupFeatureMapFields.X_FLAGS.value,
        HistogramColumnNames.STRAND_RATIO_START.value,
        HistogramColumnNames.STRAND_RATIO_END.value,
        "ML_QUAL",
    ],
    "string_list_fields": [
        PileupFeatureMapFields.X_RN.value,
        PileupFeatureMapFields.X_CIGAR.value,
        "rq",
        "tm",
        HistogramColumnNames.STRAND_RATIO_CATEGORY_START.value,
        HistogramColumnNames.STRAND_RATIO_CATEGORY_END.value,
        "st",
        "et",
    ],
    "boolean_fields": [
        PileupFeatureMapFields.IS_FORWARD.value,
        PileupFeatureMapFields.IS_DUPLICATE.value,
    ],
    "fields_to_write_once": [
        PileupFeatureMapFields.READ_COUNT.value,
        PileupFeatureMapFields.FILTERED_COUNT.value,
        PileupFeatureMapFields.TRINUC_CONTEXT_WITH_ALT.value,
        PileupFeatureMapFields.HMER_CONTEXT_REF.value,
        PileupFeatureMapFields.HMER_CONTEXT_ALT.value,
        PileupFeatureMapFields.PREV_1.value,
        PileupFeatureMapFields.PREV_2.value,
        PileupFeatureMapFields.PREV_3.value,
        PileupFeatureMapFields.NEXT_1.value,
        PileupFeatureMapFields.NEXT_2.value,
        PileupFeatureMapFields.NEXT_3.value,
    ],
    "boolean_fields_to_write_once": [
        PileupFeatureMapFields.IS_CYCLE_SKIP.value,
    ],
}


def write_a_pileup_record(  # noqa: C901, PLR0912 #TODO: refactor. too complex
    record_dict: dict,
    rec_id: tuple,
    out_fh: pysam.VariantFile,
    header: pysam.VariantHeader,
    fields_to_collect: dict,
    min_qual: int,
    sample_name: str = "SAMPLE",
    qual_agg_func: callable = np.max,
):
    """
    Write a pileup record to a vcf file

    Inputs:
        record_dict (dict): A dictionary containing the record information
        rec_id (str): The record id
        out_fh (pysam.VariantFile): A pysam file handle of the output vcf file
        header (pysam.VariantHeader): A pysam VarianyHeader pbject of the output vcf file
        fields_to_collect (dict): A dictionary containing the fields to collect
        min_qual (int): The minimum quality threshold
        sample_name (str): The name of the sample (default: "SAMPLE")
        qual_agg_func (function): The function to aggregate the quality scores (default: np.max)
    Output:
        rec (pysam.VariantRecord): A pysam VariantRecord object

    """
    format_fields = [GT, AD, DP, VAF]
    rec = header.new_record()
    rec.chrom = rec_id[0]
    rec.pos = rec_id[1]
    rec.ref = rec_id[2]
    rec.alts = rec_id[3]
    rec.id = "."
    rec.qual = qual_agg_func(record_dict[PileupFeatureMapFields.X_QUAL.value])
    # INFO fields to aggregate
    # exceptions: X_QUAL
    rec.info[PileupFeatureMapFields.X_QUAL.value] = list(record_dict[PileupFeatureMapFields.X_QUAL.value])
    for field in fields_to_collect["numeric_array_fields"]:
        if field in record_dict:
            rec.info[field] = list(record_dict[field])
    for field in fields_to_collect["string_list_fields"]:
        if field in record_dict:
            rec.info[field] = "|".join(record_dict[field])
    for field in fields_to_collect["boolean_fields"]:
        if field in record_dict:
            rec.info[field] = "|".join(["T" if f else "F" for f in record_dict[field]])
    for field in fields_to_collect["fields_to_write_once"]:
        if field in record_dict:
            rec.info[field] = record_dict[field][0]
            if not all(f == record_dict[field][0] for f in record_dict[field]):
                raise ValueError(
                    f"Field {field} has multiple values, but expected to have only a single value. rec: {record_dict}"
                )
    for field in fields_to_collect["boolean_fields_to_write_once"]:
        if field in record_dict:
            rec.info[field] = record_dict[field][0]
            if not all(f == record_dict[field][0] for f in record_dict[field]):
                raise ValueError(
                    f"Field {field} has multiple values, but expected to have only a single value. rec: {record_dict}"
                )

    # FORMAT fields to aggregate
    record_dict[DP] = rec.info[PileupFeatureMapFields.FILTERED_COUNT.value]
    record_dict[VAF] = record_dict["rec_counter"] / record_dict[DP]
    record_dict[GT] = (0, 1)
    record_dict[AD] = [
        record_dict[DP] - record_dict["rec_counter"],
        record_dict["rec_counter"],
    ]
    for format_field in format_fields:
        rec.samples[sample_name][format_field] = record_dict[format_field]
    # FILTER field
    if rec.qual < min_qual:
        curr_filter = FeatureMapFilters.LOW_QUAL.value
    elif rec.qual >= min_qual and record_dict["rec_counter"] == 1:
        curr_filter = FeatureMapFilters.SINGLE_READ.value
    else:
        curr_filter = FeatureMapFilters.PASS.value
    rec.filter.add(curr_filter)
    # write to file
    out_fh.write(rec)
    return rec


def pileup_featuremap(  # noqa: C901, PLR0912, PLR0915 #TODO: refactor
    featuremap: str,
    output_vcf: str,
    genomic_interval: str = None,
    min_qual: int = 0,
    sample_name: str = "SAMPLE",
    qual_agg_func: callable = np.max,
    *,
    verbose: bool = True,
):
    """
    Pileup featuremap vcf to a regular vcf, save the aggregated quality and read count

    Inputs:
        featuremap (str): The input featuremap vcf file
        output_vcf (str): The output pileup vcf file
        genomic_interval (str): The genomic interval to pileup, format: chr:start-end
        min_qual (int): The minimum quality threshold
        sample_name (str): The name of the sample (default: "SAMPLE")
        qual_agg_func (callable): The function to aggregate the quality scores (default: np.max)
        verbose (bool): The verbosity level (default: True)
    Output:
        output_vcf (str): The output vcf file
    """
    if verbose:
        logger.setLevel(logging.DEBUG)
        for handler in logger.handlers:
            handler.setLevel(logging.DEBUG)
    # process genomic interval input
    if genomic_interval is None:
        chrom = None
        start = None
        end = None
    else:
        if not re.match(r"^\w+:\d+-\d+$", genomic_interval):
            raise ValueError("Input genomic_interval format should be 'chrom:start-end'")
        genomic_interval_list = genomic_interval.split(":")
        chrom = genomic_interval_list[0]
        start = int(genomic_interval_list[1].split("-")[0])
        end = int(genomic_interval_list[1].split("-")[1])

    # generate a new header
    # filter out the fields that are not present in the input featuremap vcf
    fields_to_collect = defaultdict(list)
    orig_header = pysam.VariantFile(featuremap).header
    header = pysam.VariantHeader()
    header.add_meta("fileformat", "VCFv4.2")
    header.filters.add(
        "SingleRead", None, None, "Aggregated quality above threshold, Only a single read agrees with alternative"
    )
    header.filters.add("LowQual", None, None, "Aggregated quality below threshold")

    for field in fields_to_collect_all_options["numeric_array_fields"]:
        if field in orig_header.info:
            header.info.add(field, ".", orig_header.info[field].type, orig_header.info[field].description)
            fields_to_collect["numeric_array_fields"] += [field]
        else:
            logger.debug(f"Field {field} not found in the input featuremap vcf")

    for field in fields_to_collect_all_options["string_list_fields"]:
        if field in orig_header.info:
            header.info.add(field, "1", orig_header.info[field].type, orig_header.info[field].description)
            fields_to_collect["string_list_fields"] += [field]
        else:
            logger.debug(f"Field {field} not found in the input featuremap vcf")

    for field in fields_to_collect_all_options["boolean_fields"]:
        if field in orig_header.info:
            header.info.add(field, "1", "String", orig_header.info[field].description)
            fields_to_collect["boolean_fields"] += [field]
        else:
            logger.debug(f"Field {field} not found in the input featuremap vcf")

    for field in fields_to_collect_all_options["fields_to_write_once"]:
        if field in orig_header.info:
            header.info.add(field, "1", orig_header.info[field].type, orig_header.info[field].description)
            fields_to_collect["fields_to_write_once"] += [field]
        else:
            logger.debug(f"Field {field} not found in the input featuremap vcf")

    for field in fields_to_collect_all_options["boolean_fields_to_write_once"]:
        if field in orig_header.info:
            header.info.add(field, "0", "Flag", orig_header.info[field].description)
            fields_to_collect["boolean_fields_to_write_once"] += [field]
        else:
            logger.debug(f"Field {field} not found in the input featuremap vcf")

    # exceptions: X_QUAL
    header.info.add(PileupFeatureMapFields.X_QUAL.value, ".", "Float", "Quality of reads containing this location")
    header.formats.add(GT, 1, "String", "Genotype")
    header.formats.add(DP, 1, "Integer", "Read depth")
    header.formats.add(AD, "R", "Integer", "Allelic depths")
    header.formats.add(VAF, 1, "Float", "Allele frequency")
    # copy contigs from original header
    for contig in orig_header.contigs:
        header.contigs.add(contig)
    header.add_sample(sample_name)

    # open an output file
    out_fh = pysam.VariantFile(output_vcf, "w", header=header)

    # sort the vcf file prior to reading
    with tempfile.TemporaryDirectory(dir=dirname(output_vcf)) as temp_dir:
        sorted_featuremap = pjoin(temp_dir, basename(featuremap).replace(".vcf.gz", ".sorted.vcf.gz"))
        if genomic_interval is None:
            # sort all featuremap
            sort_cmd = f"bcftools sort {featuremap} -Oz -o {sorted_featuremap} && bcftools index -t {sorted_featuremap}"
        else:
            # sort only the genomic interval of interest
            sort_cmd = f"bcftools view {featuremap} {genomic_interval} |\
                  bcftools sort - -Oz -o {sorted_featuremap} && bcftools index -t {sorted_featuremap}"
        logger.debug(sort_cmd)
        subprocess.check_call(sort_cmd, shell=True)  # noqa: S602

        cons_dict = defaultdict(dict)
        with pysam.VariantFile(sorted_featuremap) as f:
            prev_key = ()
            for rec in f.fetch(chrom, start, end):
                rec_id = (rec.chrom, rec.pos, rec.ref, rec.alts[0])
                if "rec_counter" not in cons_dict[rec_id]:
                    if len(cons_dict.keys()) > 1:
                        # write to file
                        write_a_pileup_record(
                            cons_dict[prev_key],
                            prev_key,
                            out_fh,
                            header,
                            fields_to_collect,
                            min_qual,
                            sample_name,
                            qual_agg_func,
                        )
                        cons_dict.pop(prev_key)
                    # initialize rec_counter
                    cons_dict[rec_id]["rec_counter"] = 0
                    # exceptions: X_QUAL
                    cons_dict[rec_id][PileupFeatureMapFields.X_QUAL.value] = np.array([])
                    for field in fields_to_collect["numeric_array_fields"]:
                        cons_dict[rec_id][field] = np.array([])
                    for field in fields_to_collect["string_list_fields"]:
                        cons_dict[rec_id][field] = []
                    for field in fields_to_collect["boolean_fields"]:
                        cons_dict[rec_id][field] = []
                    for field in fields_to_collect["fields_to_write_once"]:
                        cons_dict[rec_id][field] = []
                    for field in fields_to_collect["boolean_fields_to_write_once"]:
                        cons_dict[rec_id][field] = []

                # update the record
                cons_dict[rec_id]["rec_counter"] += 1
                # exceptions: X_QUAL
                cons_dict[rec_id][PileupFeatureMapFields.X_QUAL.value] = np.append(
                    cons_dict[rec_id][PileupFeatureMapFields.X_QUAL.value], rec.qual
                )
                for field in fields_to_collect["numeric_array_fields"]:
                    cons_dict[rec_id][field] = np.append(cons_dict[rec_id][field], rec.info.get(field, np.nan))
                for field in fields_to_collect["string_list_fields"]:
                    cons_dict[rec_id][field] += [rec.info.get(field, ".")]
                for field in fields_to_collect["boolean_fields"]:
                    cons_dict[rec_id][field] += [rec.info.get(field, False)]
                for field in fields_to_collect["fields_to_write_once"]:
                    cons_dict[rec_id][field] += [rec.info.get(field, None)]
                for field in fields_to_collect["boolean_fields_to_write_once"]:
                    cons_dict[rec_id][field] += [rec.info.get(field, False)]
                prev_key = rec_id

            # write last record
            if len(cons_dict.keys()) > 0:
                write_a_pileup_record(
                    cons_dict[prev_key],
                    prev_key,
                    out_fh,
                    header,
                    fields_to_collect,
                    min_qual,
                    sample_name,
                    qual_agg_func,
                )

        out_fh.close()
        pysam.tabix_index(output_vcf, preset="vcf", min_shift=0, force=True)
    return output_vcf


def pileup_featuremap_on_an_interval_list(
    featuremap: str,
    output_vcf: str,
    interval_list: str,
    min_qual: int = 0,
    sample_name: str = "SAMPLE",
    qual_agg_func: callable = np.max,
    *,
    verbose: bool = True,
):
    """
    Apply pileup featuremap on an interval list

    Inputs:
        featuremap (str): The input featuremap vcf file
        output_vcf (str): The output pileup vcf file
        interval_list (str): The interval list file
        min_qual (int): The minimum quality threshold
        sample_name (str): The name of the sample (default: "SAMPLE")
        qual_agg_func (callable): The function to aggregate the quality scores (default: np.max)
        verbose (bool): The verbosity level (default: True)
    Output:
        output_vcf (str): The output vcf file
    """
    with tempfile.TemporaryDirectory(dir=dirname(output_vcf)) as temp_dir:
        with open(interval_list, encoding="utf-8") as f:
            for line in f:
                # ignore header lines
                if line.startswith("@"):
                    continue
                # read genomic ineterval
                genomic_interval = line.strip()
                genomic_interval_list = genomic_interval.split("\t")
                chrom = genomic_interval_list[0]
                start = genomic_interval_list[1]
                end = genomic_interval_list[2]
                genomic_interval = chrom + ":" + str(start) + "-" + str(end)
                # run pileup_featuremap on the interval
                curr_output_vcf = pjoin(
                    temp_dir,
                    basename(output_vcf).replace(".vcf.gz", "")
                    + "."
                    + chrom
                    + "_"
                    + str(start)
                    + "_"
                    + str(end)
                    + ".int_list.vcf.gz",
                )
                pileup_featuremap(
                    featuremap, curr_output_vcf, genomic_interval, min_qual, sample_name, qual_agg_func, verbose=verbose
                )
        # merge the output vcfs
        vcfs = [pjoin(temp_dir, f) for f in os.listdir(temp_dir) if f.endswith(".vcf.gz")]
        vcf_str = " ".join(vcfs)
        cmd = f"bcftools concat {vcf_str} -a | bcftools sort - -Oz -o {output_vcf} && bcftools index -t {output_vcf}"
        logger.debug(cmd)
        subprocess.check_call(cmd, shell=True)  # noqa: S602
    return output_vcf
