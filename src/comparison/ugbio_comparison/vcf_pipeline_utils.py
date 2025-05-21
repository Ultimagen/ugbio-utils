import os
import os.path
import shutil

import pandas as pd
import pysam
from simppl.simple_pipeline import SimplePipeline
from ugbio_core.exec_utils import print_and_execute
from ugbio_core.logger import logger


class VcfPipelineUtils:
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

    def run_vcfeval(  # noqa PLR0913
        self,
        vcf: str,
        gt: str,
        hcr: str,
        outdir: str,
        ref_genome: str,
        ev_region: str | None = None,
        output_mode: str = "split",
        samples: str | None = None,
        *,
        erase_outdir: bool = True,
        additional_args: str = "",
        score: str = "QUAL",
        all_records: bool = False,
    ):  # pylint: disable=too-many-arguments
        """
        Run vcfeval to evaluate the concordance between two VCF files
        Parameters
        ----------
        vcf : str
            Our variant calls
        gt : str
            GIAB (or other source truth file)
        hcr : str
            High confidence regions
        outdir : str
            Output directory
        ref_genome : str
            SDF reference file
        ev_region: str, optional
            Bed file of regions to evaluate (--bed-region)
        output_mode: str, optional
            Mode of vcfeval (default - split)
        samples: str, optional
            Sample names to compare (baseline,calls)
        erase_outdir: bool, optional
            Erase the output directory if it exists before running (otherwise vcfeval crashes)
        additional_args: str, optional
            Additional arguments to pass to vcfeval
        score: str, optional
            Score field to use for producing ROC curves in VCFEVAL
        all_records: bool, optional
            Include all records in the evaluation (default - False)
        """
        if erase_outdir and os.path.exists(outdir):
            shutil.rmtree(outdir)
        cmd = [
            "rtg",
            "RTG_MEM=12G",
            "vcfeval",
            "-b",
            gt,
            "-c",
            vcf,
            "-e",
            hcr,
            "-t",
            ref_genome,
            "-m",
            output_mode,
            "--decompose",
            "-f",
            score,
            "-o",
            outdir,
        ]
        if ev_region:
            cmd += ["--bed-regions", ev_region]
        if all_records:
            cmd += ["--all-records"]
        if additional_args:
            cmd += additional_args.split()
        if samples:
            cmd += ["--sample", samples]

        logger.info(" ".join(cmd))
        return self.__execute(" ".join(cmd))

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

    # pylint: disable=too-many-arguments
    def run_vcfeval_concordance(  # noqa PLR0913
        self,
        input_file: str,
        truth_file: str,
        output_prefix: str,
        ref_genome: str,
        evaluation_regions: str,
        comparison_intervals: str | None = None,
        input_sample: str | None = None,
        truth_sample: str | None = None,
        *,
        ignore_filter: bool = False,
        mode: str = "combine",
        ignore_genotype: bool = False,
    ) -> str:
        """Run vcfeval to evaluate concordance

        Parameters
        ----------
        input_file : str
            Our variant calls
        truth_file : str
            GIAB (or other source truth file)
        output_prefix : str
            Output prefix
        ref_genome : str
            Fasta reference file
        evaluation_regions: str
            Bed file of regions to evaluate (HCR)
        comparison_intervals: Optional[str]
            Picard intervals file to make the comparisons on. Default: None = all genome
        input_sample : str, optional
            Name of the sample in our input_file
        truth_sample : str, optional
            Name of the sample in the truth file
        ignore_filter : bool, optional
            Ignore status of the variant filter
        mode: str, optional
            Mode of vcfeval (default - combine)
        ignore_genotype: bool, optional
            Don't compare genotype information, only compare if allele is present in ground-truth
        Returns
        -------
        final concordance vcf file if the mode is "combine"
        otherwise - returns the output directory of vcfeval
        """

        output_dir = os.path.dirname(output_prefix)
        sdf_path = ref_genome + ".sdf"
        vcfeval_output_dir = os.path.join(output_dir, os.path.basename(output_prefix) + ".vcfeval_output")

        if os.path.isdir(vcfeval_output_dir):
            shutil.rmtree(vcfeval_output_dir)

        # filter the vcf to be only in the comparison_intervals.
        filtered_truth_file = os.path.join(output_dir, ".".join((os.path.basename(truth_file), "filtered", "vcf.gz")))
        if comparison_intervals is not None:
            self.intersect_with_intervals(truth_file, comparison_intervals, filtered_truth_file)
        else:
            shutil.copy(truth_file, filtered_truth_file)
            self.index_vcf(filtered_truth_file)

        if truth_sample is not None and input_sample is not None:
            samples = f"{truth_sample},{input_sample}"
        else:
            samples = None

        self.run_vcfeval(
            input_file,
            filtered_truth_file,
            evaluation_regions,
            vcfeval_output_dir,
            sdf_path,
            output_mode=mode,
            samples=samples,
            erase_outdir=True,
            additional_args="--squash_ploidy" if ignore_genotype else "",
            all_records=ignore_filter,
        )

        if mode == "combine":
            # fix the vcf file format
            self.fix_vcf_format(os.path.join(vcfeval_output_dir, "output"))

            # make the vcfeval output file without weird variants
            self.normalize_vcfeval_vcf(
                os.path.join(vcfeval_output_dir, "output.vcf.gz"),
                os.path.join(vcfeval_output_dir, "output.norm.vcf.gz"),
                ref_genome,
            )

            vcf_concordance_file = f'{output_prefix + ".vcfeval_concordance.vcf.gz"}'
            # move the file to be compatible with the output file of the genotype
            # concordance
            self.__execute(f'mv {os.path.join(vcfeval_output_dir, "output.norm.vcf.gz")} {vcf_concordance_file}')

            # generate index file for the vcf.gz file
            self.index_vcf(vcf_concordance_file)
            return vcf_concordance_file
        return vcfeval_output_dir

    def normalize_vcfeval_vcf(self, input_vcf: str, output_vcf: str, ref: str) -> None:
        """Combines monoallelic rows from VCFEVAL into multiallelic
        and combines the BASE/CALL annotations together. Mostly uses `bcftools norm`,
        but since it does not aggregate the INFO tags, they are aggregated using
        `bcftools annotate`.

        Parameters
        ----------
        input_vcf: str
            Input (output.vcf.gz from VCFEVAL)
        output_vcf: str
            Input (output.vcf.gz from VCFEVAL)
        ref: str
            Reference FASTA

        Returns
        -------
        None:
            Creates output_vcf
        """

        tempdir = f"{output_vcf}_tmp"
        os.mkdir(tempdir)

        # Step1 - bcftools norm

        self.__execute(f"bcftools norm -f {ref} -m+any -o {tempdir}/step1.vcf.gz -O z {input_vcf}")
        self.index_vcf(f"{tempdir}/step1.vcf.gz")
        self.__execute(
            f"bcftools annotate -a {input_vcf} -c CHROM,POS,CALL,BASE -Oz \
                -o {tempdir}/step2.vcf.gz {tempdir}/step1.vcf.gz"
        )
        self.index_vcf(f"{tempdir}/step2.vcf.gz")

        # Step2 - write out the annotation table. We use VariantsToTable from gatk, but remove
        # the mandatory header and replace NA by "."
        self.__execute(
            f"gatk VariantsToTable -V {input_vcf} -O {tempdir}/source.tsv -F CHROM -F POS -F REF -F CALL -F BASE"
        )
        self.__execute(
            f"gatk VariantsToTable -V {tempdir}/step2.vcf.gz -O {tempdir}/dest.tsv \
                -F CHROM -F POS -F REF -F CALL -F BASE"
        )

        # Step3 - identify lines that still need to be filled (where the CALL/BASE after the nomrmalization
        # are different from those that are expected from collapsing). This happens because plain bcftools annotate
        # requires match of reference allele that can change with normalization
        df1 = pd.read_csv(os.path.join(tempdir, "source.tsv"), sep="\t").set_index(["CHROM", "POS", "CALL"])
        df2 = pd.read_csv(os.path.join(tempdir, "dest.tsv"), sep="\t").set_index(["CHROM", "POS", "CALL"])

        difflines = df1.loc[df1.index.difference(df2.index)].reset_index().fillna(".")[["CHROM", "POS", "CALL"]]
        difflines.to_csv(os.path.join(tempdir, "step3.call.tsv"), sep="\t", header=False, index=False)

        df1 = pd.read_csv(os.path.join(tempdir, "source.tsv"), sep="\t").set_index(["CHROM", "POS", "BASE"])
        df2 = pd.read_csv(os.path.join(tempdir, "dest.tsv"), sep="\t").set_index(["CHROM", "POS", "BASE"])

        difflines = df1.loc[df1.index.difference(df2.index)].reset_index().fillna(".")[["CHROM", "POS", "BASE"]]
        difflines.to_csv(os.path.join(tempdir, "step3.base.tsv"), sep="\t", header=False, index=False)

        # Step 4 - annoate with the additional tsvs
        self.__execute(f"bgzip {tempdir}/step3.call.tsv")
        self.__execute(f"bgzip {tempdir}/step3.base.tsv")
        self.__execute(f"tabix -s1 -e2 -b2 {tempdir}/step3.call.tsv.gz")
        self.__execute(f"tabix -s1 -e2 -b2 {tempdir}/step3.base.tsv.gz")
        self.__execute(
            f"bcftools annotate -c CHROM,POS,CALL -a {tempdir}/step3.call.tsv.gz \
                -Oz -o {tempdir}/step4.vcf.gz {tempdir}/step2.vcf.gz"
        )
        self.index_vcf(f"{tempdir}/step4.vcf.gz")
        self.__execute(
            f"bcftools annotate -c CHROM,POS,CALL -a {tempdir}/step3.base.tsv.gz \
                -Oz -o {output_vcf} {tempdir}/step4.vcf.gz"
        )
        self.index_vcf(output_vcf)
        # shutil.rmtree(tempdir)

    def fix_vcf_format(self, output_prefix: str):
        """Legacy function to fix the PS field format in the old GIAB truth sets. The function overwrites the input file

        Parameters
        ----------
        output_prefix : str
            Prefix of the input and the output file (without the .vcf.gz)
        """
        self.__execute(f"gunzip -f {output_prefix}.vcf.gz")
        with open(f"{output_prefix}.vcf", encoding="utf-8") as input_file_handle:
            with open(f"{output_prefix}.tmp", "w", encoding="utf-8") as output_file_handle:
                for line in input_file_handle:
                    if line.startswith("##FORMAT=<ID=PS"):
                        output_file_handle.write(line.replace("Type=Integer", "Type=String"))
                    else:
                        output_file_handle.write(line)
        self.__execute(f"mv {output_file_handle.name} {input_file_handle.name}")
        self.__execute(f"bgzip {input_file_handle.name}")
        self.index_vcf(f"{input_file_handle.name}.gz")

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

    def transform_hom_calls_to_het_calls(self, input_file_calls: str, output_file_calls: str) -> None:
        """Reverse homozygous reference calls in deepVariant to filtered heterozygous so that max recall can be
        calculated

        Parameters
        ----------
        input_file_calls : str
            Input file name
        output_file_calls : str
            Output file name
        """

        with pysam.VariantFile(input_file_calls) as input_file:
            with pysam.VariantFile(output_file_calls, "w", header=input_file.header) as output_file:
                for rec in input_file:
                    if (
                        rec.samples[0]["GT"] == (0, 0)
                        or rec.samples[0]["GT"] == (None, None)
                        and "PASS" not in rec.filter
                    ):
                        rec.samples[0]["GT"] = (0, 1)
                    output_file.write(rec)
        self.index_vcf(output_file_calls)
