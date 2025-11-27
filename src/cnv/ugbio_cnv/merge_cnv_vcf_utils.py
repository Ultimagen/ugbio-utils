import numpy as np
import pysam
import ugbio_core.misc_utils as mu
from ugbio_core.vcf_utils import VcfUtils
from ugbio_core.vcfbed import vcftools


def merge_cnvs_in_vcf(input_vcf: str, output_vcf: str, distance: int = 1000) -> None:
    """
    Merge CNV variants in a VCF file that are within a specified distance.

    Parameters
    ----------
    input_vcf : str
        Path to the input VCF file containing CNV variants.
    output_vcf : str
        Path to the output VCF file where merged variants will be written.
    distance : int, optional
        Maximum distance between CNV variants to consider them for merging, by default 1000.

    Returns
    -------
    None
        Writes the merged VCF to output_vcf and creates an index.
    """
    output_vcf_collapse = output_vcf + ".collapse.tmp.vcf.gz"
    temporary_files = [output_vcf_collapse]
    vu = VcfUtils()
    removed_vcf = vu.collapse_vcf(
        vcf=input_vcf,
        output_vcf=output_vcf_collapse,
        refdist=distance,
        pctseq=0.0,
        pctsize=0.0,
        ignore_filter=True,
        ignore_type=False,
        erase_removed=False,
    )
    temporary_files.append(str(removed_vcf))

    action_dictionary = {
        "weighted_avg": [
            "CNMOPS_COV_MEAN",
            "CNMOPS_COV_STDEV",
            "CNMOPS_COHORT_MEAN",
            "CNMOPS_COHORT_STDEV",
            "CopyNumber",
        ]
    }
    all_fields = sum(action_dictionary.values(), [])
    update_df = vcftools.get_vcf_df(str(removed_vcf), custom_info_fields=all_fields + ["SVLEN", "MatchId"]).sort_index()
    update_df["matchid"] = update_df["matchid"].apply(lambda x: x[0]).astype(float)
    update_df["end"] = update_df["pos"] + update_df["svlen"].apply(lambda x: x[0]) - 1
    with pysam.VariantFile(output_vcf_collapse) as vcf_in:
        hdr = vcf_in.header
        with pysam.VariantFile(output_vcf, "w", header=hdr) as vcf_out:
            for record in vcf_in:
                if "CollapseId" in record.info:
                    cid = float(record.info["CollapseId"])
                    update_records = update_df[update_df["matchid"] == cid]
                    record.stop = update_records["end"].max()
                    for action, fields in action_dictionary.items():
                        for field in fields:
                            values = np.array(list(update_records[field.lower()]) + [record.info[field]])
                            if action == "weighted_avg":
                                lengths = np.array(
                                    list(update_records["svlen"].apply(lambda x: x[0])) + [record.info["SVLEN"][0]]
                                )
                                weighted_avg = np.sum(values * lengths) / np.sum(lengths)
                                record.info[field] = round(weighted_avg, 3)

                    record.info["SVLEN"] = (record.stop - record.start,)
                vcf_out.write(record)
    mu.cleanup_temp_files(temporary_files)
