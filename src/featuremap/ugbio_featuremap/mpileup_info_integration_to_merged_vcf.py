import argparse
import logging
import os
import sys
from os.path import join as pjoin
from typing import Any

import pysam
from ugbio_core.logger import logger


def __parse_args(argv: list[str]) -> argparse.Namespace:
    """
    Parse command line arguments.

    Parameters
    ----------
    argv : list of str
        Command line arguments.

    Returns
    -------
    argparse.Namespace
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        prog="mpileup_info_integration_to_merged_vcf.py",
        description=run.__doc__,
    )
    parser.add_argument(
        "--sfmp_vcf",
        help="somatic featuremap pileup vcf with tumor as sample[0] and normal as sample[1]",
        required=True,
        type=str,
    )
    parser.add_argument("--tumor_mpileup_vcf", help="tumor mpileup vcf file", required=True, type=str)
    parser.add_argument("--normal_mpileup_vcf", help="normal mpileup vcf file", required=True, type=str)
    parser.add_argument(
        "--out_directory",
        help="out directory where intermediate and output files will be saved."
        " if not supplied all files will be written to current directory",
        required=False,
        type=str,
        default=".",
    )
    return parser.parse_args(argv[1:])


def build_lookup(vcf: pysam.VariantFile) -> dict:
    """
    Builds a lookup dictionary from a VCF file.

    Parameters
    ----------
    vcf : pysam.VariantFile
        A VariantFile object from which to fetch VCF records.

    Returns
    -------
    dict
        A dictionary mapping (chrom, pos, ref) tuples to their corresponding VCF record objects.
    """
    d = {}
    for rec in vcf.fetch():
        key = (rec.chrom, rec.pos, rec.ref)
        d[key] = rec
    return d


def create_new_header(
    main_vcf: pysam.VariantFile,
    vcf1: pysam.VariantFile,
    vcf2: pysam.VariantFile,
) -> pysam.VariantHeader:
    """
    Create a new VCF header by merging FORMAT fields from two additional VCF files into the main VCF header.

    Parameters
    ----------
    main_vcf : pysam.VariantFile
        The primary VCF file whose header will be copied and extended.
    vcf1 : pysam.VariantFile
        The first VCF file from which FORMAT fields will be merged if not already present.
    vcf2 : pysam.VariantFile
        The second VCF file from which FORMAT fields will be merged if not already present.

    Returns
    -------
    pysam.VariantHeader
        A new VCF header containing all FORMAT fields from the main VCF and any additional ones from vcf1 and vcf2.
    """
    header = main_vcf.header.copy()
    for src in (vcf1, vcf2):
        for fmt in src.header.formats:
            if fmt not in header.formats:
                header.formats.add(
                    fmt,
                    src.header.formats[fmt].number,
                    src.header.formats[fmt].type,
                    src.header.formats[fmt].description,
                )
    return header


def _n_genotypes(n_alleles: int, ploidy: int) -> int:
    """
    Calculate the number of possible genotypes given the number of alleles and ploidy.

    For diploid organisms (ploidy=2), uses a shortcut formula. For other ploidy levels,
    computes the number of combinations with repetition (multichoose).

    Parameters
    ----------
    n_alleles : int
        The number of distinct alleles at the locus.
    ploidy : int
        The ploidy level (number of sets of chromosomes).

    Returns
    -------
    int
        The number of possible genotypes.

    Examples
    --------
    >>> _n_genotypes(2, 2)
    3
    >>> _n_genotypes(3, 2)
    6
    >>> _n_genotypes(2, 3)
    4
    """
    # combinations with repetition: C(n_alleles + ploidy - 1, ploidy)
    # diploid shortcut: n*(n+1)//2
    diploid_ploidy = 2
    if ploidy == diploid_ploidy:
        return n_alleles * (n_alleles + 1) // 2
    # generic
    num = 1
    den = 1
    k = ploidy
    n = n_alleles + ploidy - 1
    for i in range(1, k + 1):
        num *= n - (i - 1)
        den *= i
    return num // den


def _as_tuple(x: list | tuple | Any) -> tuple:
    """
    Convert input to a tuple.

    Parameters
    ----------
    x : list, tuple, or any type
        The input to be converted to a tuple. If `x` is a list or tuple, it is converted to a tuple.
        Otherwise, `x` is wrapped in a single-element tuple.

    Returns
    -------
    tuple
        The input as a tuple.
    """
    return tuple(x) if isinstance(x, list | tuple) else (x,)


def copy_format_fields_between_pysam_records(  # noqa: C901, PLR0912, PLR0915
    record: pysam.VariantRecord,
    new_record: pysam.VariantRecord,
    header: pysam.VariantHeader,
):
    """
    Copy all FORMAT fields for overlapping samples from a source pysam VariantRecord to a target VariantRecord.

    This function transfers all FORMAT fields (sample-level data) from `record` (source) to `new_record` (target)
    for samples present in both records. It ensures that the FORMAT fields are compatible with the target record's
    header and allele structure, handling allele-dependent fields (Number=A/R/G) and ploidy appropriately.

    Parameters
    ----------
    record : pysam.VariantRecord
        The source VariantRecord from which to copy FORMAT fields.
    new_record : pysam.VariantRecord
        The target VariantRecord to which FORMAT fields will be copied.
    header : pysam.VariantHeader
        The header of the target VCF, used to check FORMAT field definitions.

    Notes
    -----
    - Only samples present in both `record` and `new_record` are processed.
    - FORMAT fields not declared in the target header are skipped.
    - For allele-dependent fields (Number=A/R/G), the function checks that the value lengths
      match the target record's alleles.
    - Fields with variable or fixed length >1 are passed as tuples; scalars are wrapped as needed.
    - If a FORMAT field's value is incompatible with the target record (e.g., wrong length),
      it is skipped for that sample.
    """
    nalt_tgt = len(new_record.alts or ())

    for sample in new_record.samples:
        if sample not in record.samples:
            continue

        # Determine ploidy from GT in the *source* (fallback to target if missing)
        gt_src = record.samples[sample].get("GT", None)
        gt_tgt = new_record.samples[sample].get("GT", None)
        ploidy = (
            len(gt_src)
            if isinstance(gt_src, list | tuple) and gt_src is not None
            else (len(gt_tgt) if isinstance(gt_tgt, list | tuple) and gt_tgt is not None else 2)
        )

        for fmt_field in record.format.keys():
            # Skip if not declared in target header
            fmt_def = header.formats.get(fmt_field)
            if fmt_def is None:
                continue

            # Do not try to write GT unless you really want to overwrite it.
            if fmt_field == "GT":
                val = record.samples[sample].get("GT", None)
                if val is None:
                    continue
                # pysam expects a tuple of ints/None, e.g. (0,1) or (None, None)
                if not isinstance(val, list | tuple):
                    # Single allele -> make tuple
                    val = (val,)
                new_record.samples[sample]["GT"] = tuple(val)
                continue

            # Get value from source; skip if unset
            if fmt_field not in record.samples[sample]:
                continue
            val = record.samples[sample][fmt_field]
            if val is None:
                continue

            number = fmt_def.number  # could be int, or one of 'A','R','G','.'
            # For Number=A/R/G we must ensure allele-dependent lengths match the *target* record
            if number == "A":
                expected = nalt_tgt
                seq = _as_tuple(val)
                if len(seq) != expected:
                    # If allele counts differ, safest is to skip to avoid invalid lengths
                    continue
                new_record.samples[sample][fmt_field] = tuple(seq)
            elif number == "R":
                expected = 1 + nalt_tgt
                seq = _as_tuple(val)
                if len(seq) != expected:
                    continue
                new_record.samples[sample][fmt_field] = tuple(seq)
            elif number == "G":
                n_alleles_tgt = 1 + nalt_tgt
                expected = _n_genotypes(n_alleles_tgt, ploidy)
                seq = _as_tuple(val)
                if len(seq) != expected:
                    continue
                new_record.samples[sample][fmt_field] = tuple(seq)
            elif number == 1:
                # Single value expected: if a tuple/list was provided, take first element
                if isinstance(val, list | tuple):
                    new_record.samples[sample][fmt_field] = val[0]
                else:
                    new_record.samples[sample][fmt_field] = val
            elif number == "." or (isinstance(number, int) and number > 1):
                # Variable or fixed-length >1: pass tuples as-is, scalars get wrapped once
                if isinstance(val, list | tuple):
                    new_record.samples[sample][fmt_field] = tuple(val)
                else:
                    new_record.samples[sample][fmt_field] = (val,)
            else:
                # Fallback: try best-effort without re-shaping
                new_record.samples[sample][fmt_field] = val


def run(argv):
    """
    Merge mpileup information (for tumor and normal samples) into somatic featuremap pileup vcf.

    The output VCF file will have all input somatic featuremap pileup vcf records with additional
    FORMAT fields from mpileup vcf files.

    Parameters
    ----------
    argv : list of str
        Command line arguments.

    Returns
    -------
    None
    """
    args = __parse_args(argv)
    logger.setLevel(logging.DEBUG)
    for handler in logger.handlers:
        handler.setLevel(logging.DEBUG)

    logger.info(f"Output directory: {args.out_directory}")

    # Create output directory if it doesn't exist
    if not os.path.exists(args.out_directory):
        os.makedirs(args.out_directory)
        logger.info(f"Created output directory: {args.out_directory}")

    sfmp_vcf = args.sfmp_vcf
    tumor_mpileup_vcf = args.tumor_mpileup_vcf
    normal_mpileup_vcf = args.normal_mpileup_vcf
    out_sfmp_vcf = pjoin(args.out_directory, os.path.basename(sfmp_vcf).replace(".vcf.gz", ".mpileup.vcf.gz"))

    # Open input VCFs
    main_vcf = pysam.VariantFile(sfmp_vcf)
    vcf1 = pysam.VariantFile(tumor_mpileup_vcf)
    vcf2 = pysam.VariantFile(normal_mpileup_vcf)

    # Copy header and extend with missing FORMAT definitions from vcf1/vcf2
    header = create_new_header(main_vcf, vcf1, vcf2)

    # Create lookup dicts for vcf1/vcf2 records
    lookup1 = build_lookup(vcf1)
    lookup2 = build_lookup(vcf2)

    # Open output VCF
    with pysam.VariantFile(out_sfmp_vcf, "wz", header=header) as vcf_out:
        for record in main_vcf.fetch():
            # Create a new record using the updated header
            new_record = vcf_out.new_record(
                contig=record.chrom,
                start=record.start,
                stop=record.stop,
                id=record.id,
                qual=record.qual,
                alleles=record.alleles,
                filter=record.filter.keys(),
            )

            # Copy all original INFO fields
            for k, v in record.info.items():
                if k in header.info:
                    new_record.info[k] = v

            copy_format_fields_between_pysam_records(record, new_record, header)

            # Add new format fields
            key = (record.chrom, record.pos, record.ref)
            if key in lookup1:
                rec1 = lookup1[key]
            # Copy FORMAT values from vcf1 into first sample
            for field in rec1.format.keys():
                # Copy FORMAT values from vcf1 into first sample
                for field in rec1.format.keys():
                    new_record.samples[0][field] = rec1.samples[0].get(field, None)

            if key in lookup2:
                rec2 = lookup2[key]
                # Copy FORMAT values from vcf2 into second sample
                for field in rec2.format.keys():
                    new_record.samples[1][field] = rec2.samples[0].get(field, None)

            vcf_out.write(new_record)

    main_vcf.close()
    vcf1.close()
    vcf2.close()
    logger.info(f"Merged VCF file with mpileup info created: {out_sfmp_vcf}")


def main():
    run(sys.argv)


if __name__ == "__main__":
    main()
