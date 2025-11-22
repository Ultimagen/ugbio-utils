from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import warnings

from simppl.simple_pipeline import SimplePipeline
from ugbio_core.exec_utils import print_and_execute
from ugbio_core.logger import logger

warnings.filterwarnings("ignore")

bedtools = "bedtools"


def filter_by_bed_file(in_bed_file, filtration_cutoff, filtering_bed_file, prefix, tag):
    """Filter BED file by another BED file using bedtools subtract.
    Parameters
    ----------
    in_bed_file : str
        Input BED file to be filtered.
    filtration_cutoff : float
        Filtration cutoff (fraction of overlap required to filter).
    filtering_bed_file : str
        BED file containing regions to filter out.
    prefix : str
        Prefix for output files.
    tag : str
        Tag to append to output filenames.

    Returns
    -------
    str
        Path to the annotated BED file with filtered regions marked.
    """
    out_filename = os.path.basename(in_bed_file)
    out_filtered_bed_file = prefix + out_filename.rstrip(".bed") + "." + tag + ".bed"
    cmd = (
        bedtools
        + " subtract -N -f "
        + str(filtration_cutoff)
        + " -a "
        + in_bed_file
        + " -b "
        + filtering_bed_file
        + " > "
        + out_filtered_bed_file
    )

    os.system(cmd)  # noqa: S605
    filtered_out_records = prefix + out_filename.rstrip(".bed") + "." + tag + ".filtered_out.bed"
    cmd = (
        bedtools
        + " subtract -N -f "
        + str(filtration_cutoff)
        + " -a "
        + in_bed_file
        + " -b "
        + out_filtered_bed_file
        + " > "
        + filtered_out_records
    )

    os.system(cmd)  # noqa: S605
    out_annotate_file = prefix + out_filename.rstrip(".bed") + "." + tag + ".annotate.bed"
    cmd = "cat " + filtered_out_records + ' | awk \'{print $1"\t"$2"\t"$3"\t"$4"|' + tag + "\"}' > " + out_annotate_file
    os.system(cmd)  # noqa: S605

    return out_annotate_file


def filter_by_length(bed_file, length_cutoff, prefix):
    out_filename = os.path.basename(bed_file)
    out_len_file = prefix + out_filename.rstrip(".bed") + ".len.bed"
    cmd = "awk '$3-$2<" + str(length_cutoff) + "' " + bed_file + " > " + out_len_file
    os.system(cmd)  # noqa: S605
    out_len_annotate_file = prefix + out_filename.rstrip(".bed") + ".len.annotate.bed"
    cmd = "cat " + out_len_file + ' | awk \'{print $1"\t"$2"\t"$3"\t"$4"|LEN"}\' > ' + out_len_annotate_file
    os.system(cmd)  # noqa: S605

    return out_len_annotate_file


def intersect_bed_regions(  # noqa: C901, PLR0912, PLR0915 #TODO: refactor
    include_regions: list[str],
    exclude_regions: list[str] = None,
    output_bed: str = "output.bed",
    max_mem: int = None,
    sp: SimplePipeline = None,
    *,
    assume_input_sorted: bool = False,
):
    """
    Intersect BED regions with the option to subtract exclude regions,
    using bedops for the operations (must be installed).

    Parameters
    ----------
    include_regions : list of str
        List of paths to BED files to be intersected.
    exclude_regions : list of str, optional
        List of paths to BED or VCF files to be subtracted from the intersected result.
    output_bed : str, optional
        Path to the output BED file.
    assume_input_sorted : bool, optional
        If True, assume that the input files are already sorted. If False, the function will sort them on-the-fly.
    max_mem : int, optional
        Maximum memory in bytes allocated for the sort-bed operations.
        If not specified, the function will allocate 80% of the available system memory.
    sp : SimplePipeline, optional
        SimplePipeline object to be used for printing and executing commands.

    Returns
    -------
    None
        The function saves the intersected (and optionally subtracted) regions to the output_bed file.

    Raises
    ------
    FileNotFoundError
        If any of the input files do not exist.

    """

    # Checking if all input files exist
    for region_file in include_regions + (exclude_regions if exclude_regions else []):
        if not os.path.exists(region_file):
            raise FileNotFoundError(f"File '{region_file}' does not exist.")

    # Make sure bedops is installed
    if subprocess.call(["bedops", "--version"]) != 0:  # noqa: S607
        raise RuntimeError("bedops is not installed. Please install bedops and make sure it is in your PATH.")

    # If only one include region is provided and no exclude regions, just copy the file to the output
    if len(include_regions) == 1 and exclude_regions is None and assume_input_sorted:
        shutil.copy(include_regions[0], output_bed)
        return

    # If max_mem is not specified, set it to 80% of available memory
    total_memory = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    if max_mem:
        if max_mem < total_memory:
            logger.warning(
                f"max_mem ({max_mem}) cannot be larger than the total system memory ({total_memory}). "
                f"Using {int(total_memory * 0.8)}."
            )
            max_mem = int(total_memory * 0.8)
    else:
        max_mem = int(total_memory * 0.8)

    sort_bed_cmd = f"sort-bed --max-mem {max_mem}"
    with tempfile.TemporaryDirectory() as tempdir:
        # Function to get a temp file path within the tempdir
        def get_temp_file():
            return os.path.join(tempdir, next(tempfile._get_candidate_names()))

        # Process the include regions
        if len(include_regions) == 1:
            if not assume_input_sorted:
                sorted_include = get_temp_file()
                print_and_execute(
                    f"{sort_bed_cmd} {include_regions[0]} > {sorted_include}",
                    simple_pipeline=sp,
                    module_name=__name__,
                )
                intersected_include_file = sorted_include
            else:
                intersected_include_file = include_regions[0]
        else:
            sorted_includes = []
            for include_bed_or_vcf in include_regions:
                if not assume_input_sorted:
                    sorted_include = get_temp_file()
                    print_and_execute(
                        f"{sort_bed_cmd} {include_bed_or_vcf} > {sorted_include}",
                        simple_pipeline=sp,
                        module_name=__name__,
                    )
                    sorted_includes.append(sorted_include)
                else:
                    sorted_includes.append(include_bed_or_vcf)
            intersected_include = get_temp_file()
            print_and_execute(
                f"bedops --header --intersect {' '.join(sorted_includes)} > {intersected_include}",
                simple_pipeline=sp,
                module_name=__name__,
            )
            intersected_include_file = intersected_include

        # Process the exclude_regions similarly and get the subtracted regions
        if exclude_regions:
            excludes = []
            for exclude_bed_or_vcf in exclude_regions:
                sorted_exclude_bed = get_temp_file()
                if exclude_bed_or_vcf.endswith((".vcf", ".vcf.gz")):  # vcf file
                    if not assume_input_sorted:
                        print_and_execute(
                            f"bcftools view {exclude_bed_or_vcf} | "
                            "bcftools annotate -x INFO,FORMAT | "
                            f"vcf2bed --max-mem {max_mem} > {sorted_exclude_bed}",
                            simple_pipeline=sp,
                            module_name=__name__,
                        )
                    else:
                        print_and_execute(
                            f"bcftools view {exclude_bed_or_vcf} | "
                            "bcftools annotate -x INFO,FORMAT | "
                            f"vcf2bed --do-not-sort > {sorted_exclude_bed}",
                            simple_pipeline=sp,
                            module_name=__name__,
                        )
                    excludes.append(sorted_exclude_bed)
                elif not assume_input_sorted:
                    print_and_execute(
                        f"{sort_bed_cmd} {exclude_bed_or_vcf} > {sorted_exclude_bed}",
                        simple_pipeline=sp,
                        module_name=__name__,
                    )
                    excludes.append(sorted_exclude_bed)
                else:  # bed file and assume_input_sorted
                    excludes.append(exclude_bed_or_vcf)

            # Construct the final command
            cmd = f"bedops --header --difference {intersected_include_file} {' '.join(excludes)} > {output_bed}"
        else:
            cmd = f"mv {intersected_include_file} {output_bed}"

        # Execute the final command
        print_and_execute(cmd, simple_pipeline=sp, module_name=__name__)


def count_bases_in_bed_file(file_path: str) -> int:
    """
    Count the number of bases in a given region from a file.

    Parameters
    ----------
    file_path : str
        Path to the bed file containing region data. interval_list files are also supported.

    Returns
    -------
    int
        Total number of bases in the provided region.

    Raises
    ------
    FileNotFoundError
        If the provided file path does not exist.
    """

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    # count the # of bases in region
    n_bases_in_region = 0
    with open(file_path, encoding="utf-8") as fh:
        for line in fh:
            if not line.startswith("@") and not line.startswith("#"):  # handle handles and interval_list files
                spl = line.rstrip().split("\t")
                n_bases_in_region += int(spl[2]) - int(spl[1])

    return n_bases_in_region


def bedtools_map(
    a_bed: str,
    b_bed: str,
    output_bed: str,
    column: int | str = 4,
    operation: str = "mean",
    *,
    presort: bool = False,
    sp: SimplePipeline | None = None,
    additional_args: str = "",
    tempdir: str | None = None,
) -> None:
    """
    Run bedtools map to annotate intervals in file A with values from file B.

    Parameters
    ----------
    a_bed : str
        Path to BED file A (intervals to be annotated).
    b_bed : str
        Path to BED file B (annotation source).
    output_bed : str
        Path to output BED file with annotations.
    column : int or str, optional
        Column number from B file to map onto A (default: 4).
        Can be an integer or string representation of integer.
    operation : str, optional
        Operation to apply when multiple B intervals overlap an A interval.
        Options include: sum, mean, median, min, max, count, collapse, etc.
        Default: "mean". See bedtools map documentation for full list.
    presort : bool, optional
        If True, sort input files before running bedtools map (default: False).
        Bedtools map requires sorted input.
    sp : SimplePipeline, optional
        SimplePipeline object for command execution logging.
    additional_args : str, optional
        Additional arguments to pass to bedtools map (e.g., "-null 0").
    tempdir : str, optional
        Directory to use for temporary files when presort=True.
        If None, uses the system default temporary directory.

    Returns
    -------
    None
        The function saves the annotated regions to the output_bed file.

    Raises
    ------
    FileNotFoundError
        If either input file does not exist.
    RuntimeError
        If bedtools is not installed.

    Examples
    --------
    >>> bedtools_map("regions.bed", "scores.bed", "annotated.bed", column=5, operation="max")
    >>> bedtools_map("regions.bed", "scores.bed", "output.bed", presort=True, additional_args="-null 0")
    >>> bedtools_map("regions.bed", "scores.bed", "output.bed", presort=True, tempdir="/scratch")
    """
    # Check if input files exist
    if not os.path.exists(a_bed):
        raise FileNotFoundError(f"File A '{a_bed}' does not exist.")
    if not os.path.exists(b_bed):
        raise FileNotFoundError(f"File B '{b_bed}' does not exist.")

    # Check if bedtools is installed
    if shutil.which(bedtools) is None:
        raise RuntimeError("bedtools is not installed. Please install bedtools and make sure it is in your PATH.")

    # Convert column to string for command construction
    column_str = str(column)

    # Prepare file paths (sort if requested)
    if presort:
        with tempfile.TemporaryDirectory(dir=tempdir) as tmpdir:
            sorted_a = os.path.join(tmpdir, "sorted_a.bed")
            sorted_b = os.path.join(tmpdir, "sorted_b.bed")

            # Sort file A
            sort_a_cmd = f"{bedtools} sort -i {a_bed} > {sorted_a}"
            print_and_execute(sort_a_cmd, simple_pipeline=sp, module_name=__name__)

            # Sort file B
            sort_b_cmd = f"{bedtools} sort -i {b_bed} > {sorted_b}"
            print_and_execute(sort_b_cmd, simple_pipeline=sp, module_name=__name__)

            # Run bedtools map
            map_cmd = f"{bedtools} map -a {sorted_a} -b {sorted_b} " f"-c {column_str} -o {operation}"
            if additional_args:
                map_cmd += f" {additional_args}"
            map_cmd += f" > {output_bed}"

            print_and_execute(map_cmd, simple_pipeline=sp, module_name=__name__)
    else:
        # Run bedtools map directly on input files
        map_cmd = f"{bedtools} map -a {a_bed} -b {b_bed} " f"-c {column_str} -o {operation}"
        if additional_args:
            map_cmd += f" {additional_args}"
        map_cmd += f" > {output_bed}"

        print_and_execute(map_cmd, simple_pipeline=sp, module_name=__name__)
