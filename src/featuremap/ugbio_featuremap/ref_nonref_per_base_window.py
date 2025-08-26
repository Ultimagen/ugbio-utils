import argparse
import logging
import os
import sys
import time
from os.path import join as pjoin
from pathlib import Path

import numpy as np
import pandas as pd
from ugbio_core.logger import logger

VERY_WIDE_PADDING = 10


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


def load_bed_regions(
    bed_file: str | Path, distance_start_to_center: int | None = 1
) -> tuple[pd.DataFrame, set[str], int]:
    """
    Load BED regions and create a memory-efficient DataFrame for pileup analysis.

    This function reads a BED file and creates a pandas DataFrame optimized for
    analyzing reference and non-reference counts within a specified window
    around each region's start position. The implementation uses vectorized
    operations and optimized data types for improved memory efficiency.

    Parameters
    ----------
    bed_file : str or Path
        Path to the input BED file containing genomic regions.
        Expected format: tab-separated with columns chrom, start, end.
        Lines starting with '#' are treated as comments and ignored.
    distance_start_to_center : int, default 1, can be None
        Distance from start position to center for window analysis.
        Must be less than 10 for optimal memory efficiency.
        Creates a window of size 2*distance_start_to_center + 1.
        If distance_start_to_center is None, the BED is assumed to be padded

    Returns
    -------
    tuple[pd.DataFrame, set[str], int]
        A tuple containing:
        - regions_df : pd.DataFrame
            DataFrame with MultiIndex (chrom, position) containing:
            - start: start position (0-based, int32)
            - end: end position (0-based, int32)
            - center: center position (1-based, int32)
            - relative_position: offset from center position (int32)
            - ref_count: reference base count (nullable Int32)
            - nonref_count: non-reference base count (nullable Int32)
            - seen: whether position has data (bool)
            - ref_base: reference base at center position (category)
            Note: Index may contain duplicate (chrom, position) pairs when regions overlap.
        - contigs : set of str
            Sorted set of unique chromosome/contig names
        - distance_start_to_center : int
            The distance from start to center used for window analysis, same as given or estimated from the BED

    Raises
    ------
    FileNotFoundError
        If the BED file does not exist.
    ValueError
        If distance_start_to_center is not positive or >= 10.
    pd.errors.EmptyDataError
        If the BED file is empty or contains no valid regions.

    Examples
    --------
    >>> regions_df, contigs = load_bed_regions("regions.bed", distance_start_to_center=2)
    >>> print(f"Loaded {len(regions_df.index.get_level_values('chrom').unique())} chromosomes")
    >>> print(f"Total positions: {len(regions_df.index.get_level_values('position').unique())}")
    >>> print(f"Contigs: {contigs}")
    >>> # Access data for a specific chromosome and position
    >>> chr1_data = regions_df.loc['chr1']  # All positions on chr1
    >>> position_data = regions_df.loc[('chr1', 12345)]

    Notes
    -----
    The function assumes distance_start_to_center < 10 for memory optimization.
    Each region creates 2*distance_start_to_center + 1 rows in the DataFrame.
    Memory usage is optimized through:
    - Categorical data types for repeated string values
    - Appropriate integer types (int32 vs int64)
    - Nullable integer types for missing count data
    - Pre-allocated DataFrame structure

    See Also
    --------
    build_region_index : Creates position-based index from regions DataFrame
    process_mpileup : Processes mpileup data using region index
    """
    start_time = time.time()

    # Input validation
    bed_path = Path(bed_file)
    if not bed_path.exists():
        raise FileNotFoundError(f"BED file not found: {bed_file}")

    logger.info(f"Loading BED regions from: {bed_file}")

    # Read BED file efficiently with pandas
    try:
        bed_df = pd.read_csv(
            bed_file,
            sep="\t",
            header=None,
            comment="#",
            usecols=[0, 1, 2],  # Only read first 3 columns
            names=["chrom", "start", "end"],
            dtype={
                "chrom": "string",  # Use string dtype for better memory efficiency
                "start": "int32",  # int32 sufficient for genomic coordinates
                "end": "int32",
            },
        )
        bed_df = bed_df.drop_duplicates()

    except pd.errors.EmptyDataError as e:
        raise pd.errors.EmptyDataError(f"BED file is empty or contains no valid regions: {bed_file}") from e

    if bed_df.empty:
        raise pd.errors.EmptyDataError(f"No valid regions found in BED file: {bed_file}")

    logger.info(f"Read {len(bed_df)} regions from BED file")

    if distance_start_to_center is None:
        distance_start_to_center_list = ((bed_df["end"] - bed_df["start"]) // 2).unique()
        if len(distance_start_to_center_list) != 1:
            raise ValueError("If distance_start_to_center is None, all regions must be same size")
        distance_start_to_center = int(distance_start_to_center_list[0])
    if distance_start_to_center <= 0:
        raise ValueError(f"distance_start_to_center must be positive, got: {distance_start_to_center}")

    if distance_start_to_center >= VERY_WIDE_PADDING:
        logger.warning(
            f"distance_start_to_center={distance_start_to_center} is large (>=10). "
            "This may impact memory efficiency."
        )

        logger.info(f"Window size: {2 * distance_start_to_center + 1} positions per region")

    # Calculate centers using vectorized operations
    bed_df["center"] = bed_df["start"] + distance_start_to_center

    # Get unique contigs and convert to categorical for memory efficiency
    contigs = sorted(bed_df["chrom"].unique())
    bed_df["chrom"] = bed_df["chrom"].astype(pd.CategoricalDtype(categories=contigs))

    logger.info(f"Found {len(contigs)} unique contigs: {contigs}")

    # Create relative position range
    rel_positions = list(range(-distance_start_to_center, distance_start_to_center + 1))
    window_size = len(rel_positions)
    n_regions = len(bed_df)
    total_rows = n_regions * window_size

    logger.debug(f"Creating DataFrame with {total_rows} rows ({n_regions} regions × {window_size} positions)")

    # Pre-allocate arrays for efficient DataFrame construction
    rel_pos_array = np.tile(rel_positions, n_regions)

    # Expand region data to match the MultiIndex structure
    expanded_data = {
        "chrom": np.repeat(bed_df["chrom"].to_numpy(), window_size),
        "start": np.repeat(bed_df["start"].to_numpy(), window_size),
        "end": np.repeat(bed_df["end"].to_numpy(), window_size),
        "center": np.repeat(bed_df["center"].to_numpy(), window_size),
        "relative_position": rel_pos_array,
    }

    # Calculate absolute positions for each relative position
    expanded_data["position"] = expanded_data["center"] + rel_pos_array

    # Initialize count and status columns with appropriate dtypes
    expanded_data["ref_count"] = np.full(total_rows, pd.NA, dtype=int)
    expanded_data["nonref_count"] = np.full(total_rows, pd.NA, dtype=int)
    expanded_data["seen"] = np.full(total_rows, fill_value=False, dtype=bool)

    # Initialize ref_base with categorical type for memory efficiency
    expanded_data["ref_base"] = np.full(total_rows, "N", dtype=str)

    # Create MultiIndex DataFrame
    multi_index = pd.MultiIndex.from_arrays(
        [expanded_data["chrom"], expanded_data["position"]], names=["chrom", "position"]
    )

    regions_df = pd.DataFrame(expanded_data, index=multi_index)

    # Log memory usage and performance statistics
    memory_usage_mb = regions_df.memory_usage(deep=True).sum() / 1024 / 1024
    elapsed_time = time.time() - start_time

    logger.info(f"Created regions DataFrame: {regions_df.shape[0]} rows × {regions_df.shape[1]} columns")
    logger.info(f"Memory usage: {memory_usage_mb:.2f} MB")
    logger.info(f"Processing time: {elapsed_time:.3f} seconds")
    logger.debug(f"DataFrame dtypes:\n{regions_df.dtypes}")
    regions_df = regions_df.sort_index()
    return regions_df, set(contigs), distance_start_to_center


def process_mpileup(mpileup_file, region_df):
    """Processes a samtools mpileup file and updates the region DataFrame with reference and non-reference counts.
    Parameters
    ----------
    mpileup_file : str
        Path to the input samtools mpileup output file.
    region_df : pd.DataFrame
        DataFrame with MultiIndex (chrom, position) containing regions to be updated.
        Must include columns: ref_counts, nonref_counts, seen, ref_base.

    Returns
    -------
    None
    """
    n_fields_mpileup = 5
    with open(mpileup_file) as f:
        for line in f:
            fields = line.strip().split("\t")
            if len(fields) < n_fields_mpileup:
                continue
            chrom, pos, ref, _, bases = fields[:n_fields_mpileup]
            pos = int(pos) - 1  # mpileup is 1-based
            key = (chrom, pos)
            if key not in region_df.index:
                continue
            ref_ct, nonref_ct = parse_bases(bases)
            region_df.loc[key, "ref_count"] = ref_ct
            region_df.loc[key, "nonref_count"] = nonref_ct
            region_df.loc[key, "seen"] = True
            mask = region_df.loc[key, "relative_position"] == 0
            region_df.loc[key, "ref_base"] = region_df.loc[key, "ref_base"].where(~mask, ref.upper())


def rearrange_pileup_df(region_df) -> pd.DataFrame:
    """Rearranges the region DataFrame to have one row per region with counts in columns.
    Parameters
    ----------
    region_df : pd.DataFrame
        DataFrame with MultiIndex (chrom, position) containing regions to be rearranged.
        Must include columns: start, end, center, relative_position, ref_counts, nonref_counts, seen, ref_base.

    Returns
    -------
    pd.DataFrame
        Rearranged DataFrame with one row per region and counts in columns.
    """
    # Reset index to work with regular columns
    df_reset = region_df.reset_index(drop=True)

    # Create pivot table for ref_count
    ref_pivot = df_reset.pivot_table(
        index=["chrom", "center"], columns="relative_position", values="ref_count", aggfunc="first"
    )

    # Create pivot table for nonref_count
    nonref_pivot = df_reset.pivot_table(
        index=["chrom", "center"], columns="relative_position", values="nonref_count", aggfunc="first"
    )
    seen_pivot = df_reset.pivot_table(
        index=["chrom", "center"], columns="relative_position", values="seen", aggfunc="first"
    )
    # Rename columns to desired format
    ref_pivot.columns = [f'ref_count_{"m" + str(abs(col)) if col < 0 else str(col)}' for col in ref_pivot.columns]
    nonref_pivot.columns = [
        f'nonref_count_{"m" + str(abs(col)) if col < 0 else str(col)}' for col in nonref_pivot.columns
    ]
    seen_pivot.columns = [f'seen_{"m" + str(abs(col)) if col < 0 else str(col)}' for col in seen_pivot.columns]
    # Combine the two pivot tables
    result_df = pd.concat([ref_pivot, nonref_pivot, seen_pivot], axis=1)
    result_df["ref_base"] = region_df.query("relative_position==0").loc[result_df.index, "ref_base"]
    result_df = result_df.reset_index()
    return result_df


def write_vcf(
    output_dir: str, base_file_name: str, regions: pd.DataFrame, contigs: set[str], distance_start_to_center: int
) -> None:
    """Writes the processed regions to a VCF file.
    Parameters
    ----------
    output_dir : str
        Path to the output directory.
    base_file_name : str
        Base file name for output files.
    regions : pd.DataFrame
        DataFrame with processed regions containing counts and reference bases.
    contigs : set
        Set of contig names.
    distance_start_to_center : int
        Distance from start to center for window analysis.

    Returns
    -------
        None, outputs an indexed VCF file under {output_dir}/{base_file_name}.vcf.gz
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
        ref_count_columns = [f"ref_count_{'m' + str(abs(i)) if i < 0 else str(i)}" for i in format_range]
        nonref_count_columns = [f"nonref_count_{'m' + str(abs(i)) if i < 0 else str(i)}" for i in format_range]
        seen_columns = [f"seen_{'m' + str(abs(i)) if i < 0 else str(i)}" for i in format_range]
        for _, r in regions.iterrows():
            chrom = r["chrom"]
            pos = r["center"] + 1  # VCF is 1-based
            ref = r["ref_base"]
            alt = "<NONREF>"
            qual = "."
            filt = "PASS"
            info = "."
            sample_value = ":".join(
                [
                    str(int(r[rc_col])) if r[seen_col] else "."
                    for rc_col, seen_col in zip(ref_count_columns, seen_columns, strict=False)
                ]
                + [
                    str(int(r[nrc_col])) if r[seen_col] else "."
                    for nrc_col, seen_col in zip(nonref_count_columns, seen_columns, strict=False)
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
    parser.add_argument(
        "--distance_start_to_center",
        type=int,
        help="Distance from start to center if bed is not padded",
        required=False,
    )
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
    regions, contigs, distance_start_to_center = load_bed_regions(
        bed_file, distance_start_to_center=distance_start_to_center
    )
    logger.info(f"Loaded {len(regions)} regions from BED file")

    # Process mpileup file
    process_mpileup(mpileup_file, regions)
    logger.info("Processed mpileup file")

    regions = rearrange_pileup_df(regions)
    logger.info("Rearranged pileup data")

    # Write VCF output
    output_vcf_path = write_vcf(output_dir, base_file_name, regions, contigs, distance_start_to_center)
    logger.info(f"Generated VCF file: {output_vcf_path}")

    return output_vcf_path


def run(argv: list[str]) -> str:
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
