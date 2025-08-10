import argparse

import pysam
import tqdm.auto as tqdm
from ugbio_core.logger import logger


def filter_low_af_ratio_to_background(
    input_vcf: str,
    output_vcf: str,
    af_ratio_threshold: float = 10,
    af_ratio_threshold_h_indels: float = 0,
    t_vaf_threshold: float = 0,
    new_filter: str = "LowAFRatioToBackground",
):
    vcf_in = pysam.VariantFile(input_vcf)

    # Check if the new filter already exists in the header
    if new_filter in vcf_in.header.filters:
        logger.warning(f"Existing {new_filter} filter found in header. It will be replaced.")
        # NOTE: if there is an existing filter with the same name, the description will NOT be updated,
        # but the records will be updated with the new filter logic
    else:
        # Add a new FILTER definition to the header
        filter_desc = (
            f"Filter variants if AF ratio to background in GT ALT alleles < threshold. "
            f"For snps and non-h-indels: {af_ratio_threshold}, and "
            f"for h-indels (applied only if tumor VAF<{t_vaf_threshold}: {af_ratio_threshold_h_indels})"
        )
        vcf_in.header.filters.add(new_filter, None, None, filter_desc)

    vcf_out = pysam.VariantFile(output_vcf, "w", header=vcf_in.header)
    logger.info(f"Processing {input_vcf} and writing to {output_vcf}")
    filtered_count = 0
    for record in tqdm.tqdm(vcf_in.fetch()):
        # Skip if variant is marked RefCall
        if "RefCall" in list(record.filter.keys()):
            vcf_out.write(record)
            continue
        else:
            # Remove the AF ratio filter if it exists in the record
            if new_filter in record.filter.keys():
                # Get current filters
                current_filters = [str(x) for x in list(record.filter.keys())]
                # Reassign without the AF ratio filter
                record.filter.clear()
                for f in current_filters:
                    if f != new_filter:
                        record.filter.add(f)

            if len(record.filter) == 0:
                record.filter.clear()  # ensures FILTER=PASS in output

            threshold_to_use = (
                af_ratio_threshold_h_indels if record.info.get("VARIANT_TYPE") == "h-indel" else af_ratio_threshold
            )
            vaf_threshold_to_use = t_vaf_threshold if record.info.get("VARIANT_TYPE") == "h-indel" else 1

            failed = process_record(record, threshold_to_use, vaf_threshold_to_use)

            if failed:
                record.filter.add(new_filter)
                filtered_count += 1

        vcf_out.write(record)
    logger.info(f"Filtered {filtered_count} variants with {new_filter} filter.")
    vcf_in.close()
    vcf_out.close()

    pysam.tabix_index(output_vcf, preset="vcf", force=True)

    logger.info(f"Annotated VCF written to: {output_vcf}")


def process_record(record, af_ratio_threshold, t_vaf_threshold):
    failed = False
    for sample in record.samples:
        gt = record.samples[sample].get("GT")
        vaf = record.samples[sample].get("VAF")
        bg_vaf = record.samples[sample].get("BG_VAF")
        t_vaf = record.samples[sample].get("VAF")
        if gt is None or vaf is None or bg_vaf is None:
            continue

        for allele in gt:
            if (allele is None) or (allele == 0):
                continue  # skip REF or missing
            elif (allele - 1 >= len(vaf)) or (allele - 1 >= len(bg_vaf)):
                continue  # skip if allele index is out of bounds
            elif vaf[allele - 1] is None or vaf[allele - 1] == 0:
                # allele is not present in the sample, so filter this allele
                failed = True
                continue
            elif bg_vaf[allele - 1] is None or bg_vaf[allele - 1] == 0:
                # the allele is present in the sample but not in the background,
                # so do not filter the variant
                failed = False
                break
            else:  # bg_vaf[allele - 1] > 0
                af_ratio = vaf[allele - 1] / bg_vaf[allele - 1]
                if t_vaf is None or t_vaf[allele - 1] is None:
                    logger.warning("Tumor VAF is None for a GT allele!")
                elif (af_ratio >= af_ratio_threshold) or (t_vaf[allele - 1] >= t_vaf_threshold):
                    # there is an allele with AF ratio >= threshold,
                    # or there is an allele where t_vaf is above threshold
                    # so do not filter the variant
                    failed = False
                    break
                else:
                    # this allele has AF ratio < threshold (and tumor vaf < vaf_threshold for h-indels),
                    # so filter this allele
                    failed = True

    return failed


def main():
    parser = argparse.ArgumentParser(description="Annotate variants with low AF ratio")
    parser.add_argument("input_vcf", help="Input VCF file")
    parser.add_argument("output_vcf", help="Output VCF file")
    parser.add_argument(
        "--af_ratio_threshold",
        type=float,
        default=10,
        help="AF ratio threshold for snps and non-h-indels (default: 10)",
    )
    parser.add_argument(
        "--tumor_vaf_threshold_h_indels",
        type=float,
        default=0,
        help="Tumor VAF threshold for filtering (default: 0) - \
            any h-indel with VAF above this threshold will not be filtered",
    )
    parser.add_argument(
        "--af_ratio_threshold_h_indels", type=float, default=0, help="AF ratio threshold for h-indels (default: 0)"
    )

    parser.add_argument(
        "--new_filter",
        default="LowAFRatioToBackground",
        help="Name of new FILTER tag (default: LowAFRatioToBackground)",
    )

    args = parser.parse_args()

    filter_low_af_ratio_to_background(
        input_vcf=args.input_vcf,
        output_vcf=args.output_vcf,
        af_ratio_threshold=args.af_ratio_threshold,
        af_ratio_threshold_h_indels=args.af_ratio_threshold_h_indels,
        t_vaf_threshold=args.tumor_vaf_threshold_h_indels,
        new_filter=args.new_filter,
    )


if __name__ == "__main__":
    main()
