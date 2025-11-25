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


class BedUtils:
    """Utilities for BED processing, mostly wrappers around shell scripts

    Attributes
    ----------
    sp : SimplePipeline
        Simple pipeline object
    """

    bedtools = "bedtools"
    bedops = "bedops"
    bcftools = "bcftools"
    vcf2bed = "vcf2bed"

    def __init__(self, simple_pipeline: SimplePipeline | None = None):
        """Working with BED files

        Parameters
        ----------
        simple_pipeline : SimplePipeline, optional
            Optional SimplePipeline object for executing shell commands
        """
        self.sp = simple_pipeline

    def __execute(self, command: str, output_file: str | None = None):
        """Summary

        Parameters
        ----------
        command : str
            Description
        output_file : str, optional
            Description
        """
        print_and_execute(command, output_file=output_file, simple_pipeline=self.sp, module_name=__name__)

    def filter_by_bed_file(
        self, in_bed_file: str, filtration_cutoff: float, filtering_bed_file: str, prefix: str, tag: str
    ) -> str:
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
        out_filtered_bed_file = prefix + os.path.splitext(out_filename)[0] + "." + tag + ".bed"
        cmd = (
            self.bedtools
            + " subtract -N -f "
            + str(filtration_cutoff)
            + " -a "
            + in_bed_file
            + " -b "
            + filtering_bed_file
            + " > "
            + out_filtered_bed_file
        )
        self.__execute(cmd)
        filtered_out_records = prefix + os.path.splitext(out_filename)[0] + "." + tag + ".filtered_out.bed"
        cmd = (
            self.bedtools
            + " subtract -N -f "
            + str(filtration_cutoff)
            + " -a "
            + in_bed_file
            + " -b "
            + out_filtered_bed_file
            + " > "
            + filtered_out_records
        )
        self.__execute(cmd)

        out_annotate_file = prefix + os.path.splitext(out_filename)[0] + "." + tag + ".annotate.bed"
        cmd = (
            "cat "
            + filtered_out_records
            + ' | awk \'{print $1"\t"$2"\t"$3"\t"$4"|'
            + tag
            + "\"}' > "
            + out_annotate_file
        )
        self.__execute(cmd)

        return out_annotate_file

    def filter_by_length(self, bed_file: str, length_cutoff: int, prefix: str) -> str:
        """Filter BED file by length using awk (|LEN is added to the "name" column).
        Parameters
        ----------
        bed_file : str
            Input BED file to be filtered.
        length_cutoff : int
            Length cutoff (regions shorter than this will be filtered).
        prefix : str
            Prefix for output files.
        Returns
        -------
        str
            Path to the annotated BED file with length-filtered regions marked.
        """
        out_filename = os.path.basename(bed_file)
        out_len_file = prefix + os.path.splitext(out_filename)[0] + ".len.bed"
        cmd = "awk '$3-$2<" + str(length_cutoff) + "' " + bed_file + " > " + out_len_file
        self.__execute(cmd)
        out_len_annotate_file = prefix + os.path.splitext(out_filename)[0] + ".len.annotate.bed"
        cmd = "cat " + out_len_file + ' | awk \'{print $1"\t"$2"\t"$3"\t"$4"|LEN"}\' > ' + out_len_annotate_file
        self.__execute(cmd)

        return out_len_annotate_file

    def __validate_intersect_bed_regions(
        self,
        include_regions: list[str],
        exclude_regions: list[str] | None,
        max_mem: int | None,
    ) -> int:
        """
        Validate inputs for intersect_bed_regions.

        Parameters
        ----------
        include_regions : list of str
            List of paths to BED files to be intersected.
        exclude_regions : list of str, optional
            List of paths to BED or VCF files to be subtracted from the intersected result.
        max_mem : int, optional
            Maximum memory in bytes allocated for the sort-bed operations.

        Returns
        -------
        int
            Validated max_mem value (in bytes).

        Raises
        ------
        FileNotFoundError
            If any of the input files do not exist.
        RuntimeError
            If bedops is not installed.
        """
        # Checking if all input files exist
        for region_file in include_regions + (exclude_regions if exclude_regions else []):
            if not os.path.exists(region_file):
                raise FileNotFoundError(f"File '{region_file}' does not exist.")

        # Make sure bedops is installed
        if subprocess.call([self.bedops, "--version"]) != 0:
            raise RuntimeError("bedops is not installed. Please install bedops and make sure it is in your PATH.")

        # If max_mem is not specified, set it to 80% of available memory
        total_memory = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
        if max_mem:
            if max_mem > total_memory:
                logger.warning(
                    f"max_mem ({max_mem}) cannot be larger than the total system memory ({total_memory}). "
                    f"Using {int(total_memory * 0.8)}."
                )
                max_mem = int(total_memory * 0.8)
        else:
            max_mem = int(total_memory * 0.8)

        return max_mem

    def __process_include_regions(
        self,
        include_regions: list[str],
        sort_bed_cmd: str,
        get_temp_file,
        *,
        assume_input_sorted: bool = False,
    ) -> str:
        """
        Process include regions by sorting and intersecting them.

        Parameters
        ----------
        include_regions : list of str
            List of paths to BED files to be intersected.
        sort_bed_cmd : str
            Command string for sorting BED files.
        get_temp_file : callable
            Function to generate temporary file paths.
        assume_input_sorted : bool, optional
            If True, assume input files are already sorted.

        Returns
        -------
        str
            Path to the file containing the intersected include regions.
        """
        if len(include_regions) == 1:
            if not assume_input_sorted:
                sorted_include = get_temp_file()
                self.__execute(f"{sort_bed_cmd} {include_regions[0]} > {sorted_include}")
                return sorted_include
            else:
                return include_regions[0]
        else:
            sorted_includes = []
            for include_bed_or_vcf in include_regions:
                if not assume_input_sorted:
                    sorted_include = get_temp_file()
                    self.__execute(f"{sort_bed_cmd} {include_bed_or_vcf} > {sorted_include}")
                    sorted_includes.append(sorted_include)
                else:
                    sorted_includes.append(include_bed_or_vcf)
            intersected_include = get_temp_file()
            self.__execute(f"{self.bedops} --header --intersect {' '.join(sorted_includes)} > {intersected_include}")
            return intersected_include

    def __process_exclude_regions(
        self,
        exclude_regions: list[str],
        sort_bed_cmd: str,
        max_mem: int,
        get_temp_file,
        *,
        assume_input_sorted: bool = False,
    ) -> list[str]:
        """
        Process exclude regions by converting VCFs to BED and sorting.

        Parameters
        ----------
        exclude_regions : list of str
            List of paths to BED or VCF files to be subtracted.
        sort_bed_cmd : str
            Command string for sorting BED files.
        max_mem : int
            Maximum memory in bytes for sort operations.
        get_temp_file : callable
            Function to generate temporary file paths.
        assume_input_sorted : bool, optional
            If True, assume input files are already sorted.

        Returns
        -------
        list of str
            List of paths to processed exclude region files.
        """
        excludes = []
        for exclude_bed_or_vcf in exclude_regions:
            sorted_exclude_bed = get_temp_file()
            if exclude_bed_or_vcf.endswith((".vcf", ".vcf.gz")):  # vcf file
                if not assume_input_sorted:
                    self.__execute(
                        f"{self.bcftools} view {exclude_bed_or_vcf} | "
                        f"{self.bcftools} annotate -x INFO,FORMAT | "
                        f"{self.vcf2bed} --max-mem {max_mem} > {sorted_exclude_bed}"
                    )
                else:
                    self.__execute(
                        f"{self.bcftools} view {exclude_bed_or_vcf} | "
                        f"{self.bcftools} annotate -x INFO,FORMAT | "
                        f"{self.vcf2bed} --do-not-sort > {sorted_exclude_bed}"
                    )
                excludes.append(sorted_exclude_bed)
            elif not assume_input_sorted:
                self.__execute(f"{sort_bed_cmd} {exclude_bed_or_vcf} > {sorted_exclude_bed}")
                excludes.append(sorted_exclude_bed)
            else:  # bed file and assume_input_sorted
                excludes.append(exclude_bed_or_vcf)
        return excludes

    def intersect_bed_regions(
        self,
        include_regions: list[str],
        exclude_regions: list[str] | None = None,
        output_bed: str = "output.bed",
        max_mem: int | None = None,
        tempdir_prefix: str | None = None,
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
        tempdir_prefix : str, optional
            Directory to use for temporary files. If not specified, the system default temporary directory will be used.
        assume_input_sorted : bool, optional
            If True, assume that the input files are already sorted. Default is False.

        Returns
        -------
        None
            The function saves the intersected (and optionally subtracted) regions to the output_bed file.

        Raises
        ------
        FileNotFoundError
            If any of the input files do not exist.

        """
        # Validate inputs
        max_mem = self.__validate_intersect_bed_regions(include_regions, exclude_regions, max_mem)

        # If only one include region is provided and no exclude regions, just copy the file to the output
        if len(include_regions) == 1 and exclude_regions is None and assume_input_sorted:
            shutil.copy(include_regions[0], output_bed)
            return

        sort_bed_cmd = f"sort-bed --max-mem {max_mem}"
        with tempfile.TemporaryDirectory(dir=tempdir_prefix):
            # Function to get a temp file path within the tempdir
            def get_temp_file():
                return tempfile.NamedTemporaryFile(delete=False, dir=tempdir_prefix).name

            # Process the include regions
            intersected_include_file = self.__process_include_regions(
                include_regions, sort_bed_cmd, get_temp_file, assume_input_sorted=assume_input_sorted
            )

            # Process the exclude_regions and construct the final command
            if exclude_regions:
                excludes = self.__process_exclude_regions(
                    exclude_regions, sort_bed_cmd, max_mem, get_temp_file, assume_input_sorted=assume_input_sorted
                )
                cmd = f"{self.bedops} --header --difference {intersected_include_file} \
                    {' '.join(excludes)} > {output_bed}"
            else:
                cmd = f"mv {intersected_include_file} {output_bed}"

            # Execute the final command
            self.__execute(cmd)

    def intersect_bed_files(
        self,
        input_bed1: str,
        input_bed2: str,
        bed_output: str,
        tempdir_prefix: str | None = None,
        *,
        assume_input_sorted: bool = False,
    ) -> None:
        """
        Simple intersection of two BED files.

        This is a convenience wrapper around intersect_bed_regions for the common
        case of intersecting exactly two BED files.

        Parameters
        ----------
        input_bed1 : str
            First input BED file
        input_bed2 : str
            Second input BED file
        bed_output : str
            Output BED file containing intersection
        tempdir_prefix : str, optional
            Directory to use for temporary files
        assume_input_sorted : bool, optional
            If True, assume input files are already sorted (default: False)

        See Also
        --------
        intersect_bed_regions : More general function supporting multiple includes/excludes
        """
        self.intersect_bed_regions(
            include_regions=[input_bed1, input_bed2],
            exclude_regions=None,
            output_bed=bed_output,
            tempdir_prefix=tempdir_prefix,
            assume_input_sorted=assume_input_sorted,
        )

    def count_bases_in_bed_file(self, file_path: str) -> int:
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
        self,
        a_bed: str,
        b_bed: str,
        output_bed: str,
        column: int | str = 5,
        operation: str = "mean",
        additional_args: str = "",
        tempdir_prefix: str | None = None,
        *,
        presort: bool = False,
    ):
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
            Column number from B file to map onto A (default: 5).
            Can be an integer or string representation of integer.
        operation : str, optional
            Operation to apply when multiple B intervals overlap an A interval.
            Options include: sum, mean, median, min, max, count, collapse, etc.
            Default: "mean". See bedtools map documentation for full list.
        presort : bool, optional
            If True, sort input files before running bedtools map (default: False).
            Bedtools map requires sorted input.
        additional_args : str, optional
            Additional arguments to pass to bedtools map (e.g., "-null 0").
        tempdir_prefix : str, optional
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
        if shutil.which(self.bedtools) is None:
            raise RuntimeError("bedtools is not installed. Please install bedtools and make sure it is in your PATH.")

        # Convert column to string for command construction
        column_str = str(column)

        # Prepare file paths (sort if requested)
        if presort:
            with tempfile.TemporaryDirectory(dir=tempdir_prefix) as tmpdir:
                sorted_a = os.path.join(tmpdir, "sorted_a.bed")
                sorted_b = os.path.join(tmpdir, "sorted_b.bed")

                # Sort file A
                sort_a_cmd = f"{self.bedtools} sort -i {a_bed} > {sorted_a}"
                self.__execute(sort_a_cmd)

                # Sort file B
                sort_b_cmd = f"{self.bedtools} sort -i {b_bed} > {sorted_b}"
                self.__execute(sort_b_cmd)

                # Run bedtools map
                map_cmd = f"{self.bedtools} map -a {sorted_a} -b {sorted_b} -c {column_str} -o {operation}"
                if additional_args:
                    map_cmd += f" {additional_args}"
                map_cmd += f" > {output_bed}"

                self.__execute(map_cmd)
        else:
            # Run bedtools map directly on input files
            map_cmd = f"{self.bedtools} map -a {a_bed} -b {b_bed} -c {column_str} -o {operation}"
            if additional_args:
                map_cmd += f" {additional_args}"
            map_cmd += f" > {output_bed}"

            self.__execute(map_cmd)
