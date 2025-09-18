import os
import os.path

import pysam
from simppl.simple_pipeline import SimplePipeline
from ugbio_core.exec_utils import print_and_execute


class VcfUtils:
    """Utilities of vcf pipeline, mostly wrappers around shell scripts

    Attributes
    ----------
    sp : SimplePipeline
        Simple pipeline object
    """

    def __init__(self, simple_pipeline: SimplePipeline | None = None):
        """Combines VCF in parts from GATK and indices the result

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

    def combine_vcf(self, n_parts: int, input_prefix: str, output_fname: str):
        """Combines VCF in parts from GATK and indices the result

        Parameters
        ----------
        n_parts : int
            Number of VCF parts (names will be 1-based)
        input_prefix : str
            Prefix of the VCF files (including directory) 1.vcf.gz ... will be added
        output_fname : str
            Name of the output VCF
        """
        input_files = [f"{input_prefix}.{x}.vcf" for x in range(1, n_parts + 1)] + [
            f"{input_prefix}.{x}.vcf.gz" for x in range(1, n_parts + 1)
        ]
        input_files = [x for x in input_files if os.path.exists(x)]
        self.__execute(f"bcftools concat -o {output_fname} -O z {input_files}")
        self.index_vcf(output_fname)

    def index_vcf(self, vcf: str):
        """Tabix index on VCF

        Parameters
        ----------
        vcf : str
            Input vcf.gz file
        """
        self.__execute(f"bcftools index -tf {vcf}")

    def sort_vcf(self, input_file: str, output_file: str):
        """Sort VCF file

        Parameters
        ----------
        input_file : str
            Input file name
        output_file : str
            Output file name

        Returns
        -------
        None
            Generates `output_file`.
        """
        self.__execute(f"bcftools sort -o {output_file} -O z {input_file}")

    def reheader_vcf(self, input_file: str, new_header: str, output_file: str):
        """Run bcftools reheader and index

        Parameters
        ----------
        input_file : str
            Input file name
        new_header : str
            Name of the new header
        output_file : str
            Name of the output file

        No Longer Returned
        ------------------
        None, generates `output_file`
        """
        self.__execute(f"bcftools reheader -h {new_header} {input_file}")
        self.index_vcf(output_file)

    def intersect_bed_files(self, input_bed1: str, input_bed2: str, bed_output: str) -> None:
        """Intersects bed files

        Parameters
        ----------
        input_bed1 : str
            Input Bed file
        input_bed2 : str
            Input Bed file
        bed_output : str
            Output bed intersected file

        Writes output_fn file
        """
        self.__execute(f"bedtools intersect -a {input_bed1} -b {input_bed2}", output_file=bed_output)

    def intersect_with_intervals(self, input_fn: str, intervals_fn: str, output_fn: str) -> None:
        """Intersects VCF with intervalList. Writes output_fn file

        Parameters
        ----------
        input_fn : str
            Input file
        intervals_fn : str
            Interval_list file
        output_fn : str
            Output file

        Writes output_fn file
        """
        self.__execute(f"gatk SelectVariants -V {input_fn} -L {intervals_fn} -O {output_fn}")

    def annotate_tandem_repeats(self, input_file: str, reference_fasta: str) -> str:
        """Runs VariantAnnotator on the input file to add tandem repeat annotations (maybe others)

        Parameters
        ----------
        input_file : str
            vcf.gz file
        reference_fasta : str
            Reference file (should have .dict file nearby)

        Creates a copy of the input_file with .annotated.vcf.gz and the index
        Returns
        -------
        path to output file: str
        """

        output_file = input_file.replace("vcf.gz", "annotated.vcf.gz")
        self.__execute(f"gatk VariantAnnotator -V {input_file} -O {output_file} -R {reference_fasta} -A TandemRepeat")
        return output_file

    @staticmethod
    def copy_vcf_record(rec: "pysam.VariantRecord", new_header: "pysam.VariantHeader") -> "pysam.VariantRecord":
        """
        Create a new VCF record with the same data as the input record, but using a new header.

        Parameters
        ----------
        rec : pysam.VariantRecord
            The original VCF record to copy.
        new_header : pysam.VariantHeader
            The new VCF header to use for the copied record.

        Returns
        -------
        pysam.VariantRecord
            A new VCF record with the same data as `rec`, but using `new_header`.
        """
        new_record = new_header.new_record(
            contig=rec.chrom,
            start=rec.start,
            stop=rec.stop,
            id=rec.id,
            qual=rec.qual,
            alleles=rec.alleles,
            filter=rec.filter.keys(),
        )

        # copy INFO fields
        for k, v in rec.info.items():
            if k in new_header.info:
                new_record.info[k] = v

        # copy FORMAT fields
        for sample in rec.samples:
            src = rec.samples[sample]
            tgt = new_record.samples[sample]
            for k, v in src.items():
                if v in (None, (None,)):
                    continue  # no need to assign missing values
                tgt[k] = v

        return new_record
