import argparse
import logging
import os
import sys
from os.path import join as pjoin
from pathlib import Path
from typing import Any

from ugbio_core.logger import logger


def parse_bases(bases):
    ref_count = 0
    nonref_count = 0
    i = 0
    while i < len(bases):
        c = bases[i]
        if c in ".,":
            ref_count += 1
        elif c in "ACGTNacgtn*":
            nonref_count += 1
        elif c == "^":
            i += 1
        elif c == "$":
            pass
        elif c in "+-":
            i += 1
            length = ""
            nonref_count += 1  # count the indel itself
            while i < len(bases) and bases[i].isdigit():
                length += bases[i]
                i += 1
            i += int(length) - 1
        i += 1
    return ref_count, nonref_count


def load_bed_regions(bed_file, distance_start_to_center=1):
    regions = []
    contigs = set()
    with open(bed_file) as f:
        for line in f:
            if line.strip() and not line.startswith("#"):
                chrom, start, end = line.strip().split()[:3]
                start = int(start)
                end = int(end)
                center = start + distance_start_to_center  # 1-based, half-open interval
                region = {
                    "chrom": chrom,
                    "start": start,
                    "end": end,
                    "center": center,
                    "ref_counts": {
                        i: float("nan") for i in range(-distance_start_to_center, distance_start_to_center + 1)
                    },
                    "nonref_counts": {
                        i: float("nan") for i in range(-distance_start_to_center, distance_start_to_center + 1)
                    },
                    "seen": {i: False for i in range(-distance_start_to_center, distance_start_to_center + 1)},
                    "ref_base": "N",
                }
                contigs.add(chrom)
                regions.append(region)
    return regions, sorted(contigs)


def build_region_index(regions, distance_start_to_center=1):
    index = {}
    for region in regions:
        chrom = region["chrom"]
        for rel in range(-distance_start_to_center, distance_start_to_center + 1):
            pos = region["center"] + rel
            index.setdefault((chrom, pos), []).append((region, rel))
    return index


def process_mpileup(mpileup_file, region_index):
    n_fields_mpileup = 5
    with open(mpileup_file) as f:
        for line in f:
            fields = line.strip().split("\t")
            if len(fields) < n_fields_mpileup:
                continue
            chrom, pos, ref, depth, bases = fields[:n_fields_mpileup]
            pos = int(pos) - 1  # mpileup is 1-based
            key = (chrom, pos)
            if key not in region_index:
                continue
            ref_ct, nonref_ct = parse_bases(bases)
            for region, rel in region_index[key]:
                region["ref_counts"][rel] = ref_ct
                region["nonref_counts"][rel] = nonref_ct
                region["seen"][rel] = True
                if rel == 0:
                    region["ref_base"] = ref.upper()


def write_vcf(output_dir, base_file_name, regions, contigs, distance_start_to_center):
    """Writes the processed regions to a VCF file.
    Args:
        output_dir (str): Path to the output directory.
        base_file_name (str): Base file name for output files.
        regions (list): List of processed regions with counts.
        contigs (set): Set of contig names.
        distance_start_to_center (int): Distance from start to center.
    """
    format_range = range(-distance_start_to_center, distance_start_to_center + 1)
    format_range_str = [str(i) for i in format_range]
    format_range_str = [n.replace("-", "M") for n in format_range_str]  # replace negative with M for VCF format
    # Define the format ID and fields
    format_id = ":".join([f"REF_{i}" for i in format_range_str] + [f"NONREF_{i}" for i in format_range_str])
    output_path = pjoin(output_dir, f"{base_file_name}.vcf")
    with open(output_path, "w") as out:
        out.write("##fileformat=VCFv4.2\n")
        for contig in contigs:
            out.write(f"##contig=<ID={contig}>\n")

        format_fields = [(f"REF_{i}", f"Reference counts at position {i}") for i in format_range_str] + [
            (f"NONREF_{i}", f"Non-reference counts at position {i}") for i in format_range_str
        ]
        for fmt_id, desc in format_fields:
            out.write(f'##FORMAT=<ID={fmt_id},Number=1,Type=Integer,Description="{desc}">\n')

        header_fields = ["#CHROM", "POS", "ID", "REF", "ALT", "QUAL", "FILTER", "INFO", "FORMAT", "sample1"]
        out.write("\t".join(header_fields) + "\n")

        for r in regions:
            chrom = r["chrom"]
            pos = r["center"] + 1  # VCF is 1-based
            ref = r["ref_base"]
            alt = "<NONREF>"
            qual = "."
            filt = "PASS"
            info = "."
            sample_value = ":".join(
                [
                    str(int(r["ref_counts"][i])) if r["seen"][i] else "."
                    for i in range(-distance_start_to_center, distance_start_to_center + 1)
                ]
                + [
                    str(int(r["nonref_counts"][i])) if r["seen"][i] else "."
                    for i in range(-distance_start_to_center, distance_start_to_center + 1)
                ]
            )
            out.write(f"{chrom}\t{pos}\t.\t{ref}\t{alt}\t{qual}\t{filt}\t{info}\t{format_id}\t{sample_value}\n")
    # bgzip and index vcf file
    cmd = f"bgzip {output_path} && bcftools index -t {output_path}.gz"
    os.system(cmd)  # noqa: S605
    return f"{output_path}.gz"


def __parse_args(argv: list[str]) -> argparse.Namespace:
    """
    Parse command line arguments.

    Parameters
    ----------
    argv : List[str]
        Command line arguments.

    Returns
    -------
    argparse.Namespace
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        prog="ref_nonref_per_base_window.py", description="Parse samtools mpileup to vcf", allow_abbrev=True
    )
    parser.add_argument("--input", required=True, help="Input samtools mpileup output file")
    parser.add_argument("--bed", required=True, help="Input BED file")
    parser.add_argument("--distance_start_to_center", type=int, default=1, help="Distance from start to center")
    parser.add_argument("--output_dir", required=True, help="Output directory")
    parser.add_argument("--base_file_name", required=True, help="Base file name for output files")
    return parser.parse_args(argv[1:])


def process_mpileup_to_vcf(
    mpileup_file: str, bed_file: str, output_dir: str, base_file_name: str, distance_start_to_center: int = 1
) -> str:
    """
    Python interface to process mpileup file and generate VCF output.

    This function provides a programmatic interface to the mpileup processing
    functionality, allowing it to be called from other Python scripts.

    Parameters
    ----------
    mpileup_file : str
        Path to the input samtools mpileup output file.
    bed_file : str
        Path to the input BED file containing regions of interest.
    output_dir : str
        Path to the output directory where VCF file will be written.
    base_file_name : str
        Base file name for output files (without extension).
    distance_start_to_center : int, optional
        Distance from start to center position for window analysis, by default 1.

    Returns
    -------
    str
        Path to the generated compressed VCF file (.vcf.gz).

    Examples
    --------
    >>> output_vcf = process_mpileup_to_vcf(
    ...     mpileup_file="sample.mpileup",
    ...     bed_file="regions.bed",
    ...     output_dir="output",
    ...     base_file_name="sample_results"
    ... )
    >>> print(f"VCF file created: {output_vcf}")
    """
    logger.info(f"Processing mpileup file: {mpileup_file}")
    logger.info(f"Using BED file: {bed_file}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Distance from start to center: {distance_start_to_center}")

    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Load BED regions
    regions, contigs = load_bed_regions(bed_file, distance_start_to_center=distance_start_to_center)
    logger.info(f"Loaded {len(regions)} regions from BED file")

    # Build region index
    region_index = build_region_index(regions, distance_start_to_center=distance_start_to_center)
    logger.info(f"Built region index with {len(region_index)} positions")

    # Process mpileup file
    process_mpileup(mpileup_file, region_index)
    logger.info("Processed mpileup file")

    # Write VCF output
    output_vcf_path = write_vcf(output_dir, base_file_name, regions, contigs, distance_start_to_center)
    logger.info(f"Generated VCF file: {output_vcf_path}")

    return output_vcf_path


def get_region_counts(mpileup_file: str, bed_file: str, distance_start_to_center: int = 1) -> list[dict[str, Any]]:
    """
    Python interface to get reference and non-reference counts for regions.

    This function processes the mpileup file and returns the count data as
    Python data structures without writing to VCF files.

    Parameters
    ----------
    mpileup_file : str
        Path to the input samtools mpileup output file.
    bed_file : str
        Path to the input BED file containing regions of interest.
    distance_start_to_center : int, optional
        Distance from start to center position for window analysis, by default 1.

    Returns
    -------
    List[Dict[str, Any]]
        List of dictionaries containing region information and counts.
        Each dictionary contains:
        - chrom: chromosome name
        - start: start position (0-based)
        - end: end position (0-based)
        - center: center position (1-based)
        - ref_counts: dict mapping relative positions to reference counts
        - nonref_counts: dict mapping relative positions to non-reference counts
        - seen: dict mapping relative positions to boolean indicating if data was found
        - ref_base: reference base at center position

    Examples
    --------
    >>> regions = get_region_counts("sample.mpileup", "regions.bed")
    >>> for region in regions:
    ...     print(f"Region {region['chrom']}:{region['start']}-{region['end']}")
    ...     print(f"Reference counts: {region['ref_counts']}")
    ...     print(f"Non-reference counts: {region['nonref_counts']}")
    """
    logger.info(f"Getting region counts from mpileup file: {mpileup_file}")
    logger.info(f"Using BED file: {bed_file}")
    logger.info(f"Distance from start to center: {distance_start_to_center}")

    # Load BED regions
    regions, contigs = load_bed_regions(bed_file, distance_start_to_center=distance_start_to_center)
    logger.info(f"Loaded {len(regions)} regions from BED file")

    # Build region index
    region_index = build_region_index(regions, distance_start_to_center=distance_start_to_center)
    logger.info(f"Built region index with {len(region_index)} positions")

    # Process mpileup file
    process_mpileup(mpileup_file, region_index)
    logger.info("Processed mpileup file")

    return regions


def run(argv: list[str]) -> None:
    """
    Main processing function that handles command line execution.

    Parameters
    ----------
    argv : List[str]
        Command line arguments.
    """
    args = __parse_args(argv)
    logger.setLevel(logging.DEBUG)
    for handler in logger.handlers:
        handler.setLevel(logging.DEBUG)

    output_vcf_path = process_mpileup_to_vcf(
        mpileup_file=args.input,
        bed_file=args.bed,
        output_dir=args.output_dir,
        base_file_name=args.base_file_name,
        distance_start_to_center=args.distance_start_to_center,
    )

    logger.info(f"Processing complete. Output VCF: {output_vcf_path}")
    return output_vcf_path


def main() -> None:
    """
    Entry point for command line execution.
    """
    run(sys.argv)


if __name__ == "__main__":  # pragma: no cover
    main()
