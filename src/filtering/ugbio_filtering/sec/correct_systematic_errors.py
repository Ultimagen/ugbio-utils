#!/env/python
# Copyright 2022 Ultima Genomics Inc.
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.
# DESCRIPTION
#    Filter general blacklist removing locaitons where the call statistics is different
#    from how false positive calls look like
# CHANGELOG in reverse chronological order

import argparse
import ast
import glob
import itertools
import os.path
import pickle
import subprocess
import sys
from collections import defaultdict
from enum import Enum
from pathlib import Path
from typing import TextIO

import pysam
from pysam import VariantFile, VariantRecord
from ugbio_core.logger import logger
from ugbio_core.vcfbed.bed_writer import BedWriter
from ugbio_core.vcfbed.buffered_variant_reader import BufferedVariantReader
from ugbio_core.vcfbed.pysam_utils import get_filtered_alleles_list, get_filtered_alleles_str, get_genotype

from ugbio_filtering.blacklist import Blacklist
from ugbio_filtering.sec.systematic_error_correction_call import SECCall, SECCallType
from ugbio_filtering.sec.systematic_error_correction_caller import SECCaller
from ugbio_filtering.sec.systematic_error_correction_record import SECRecord
from ugbio_filtering.variant_filtering_utils import VariantSelectionFunctions

BASE_PATH = Path(__file__).parent.parent
SCRIPTS_DIR = "scripts"
REMOVE_VCF_DUPLICATES_SHELL_SCRIPT = BASE_PATH / SCRIPTS_DIR / "remove_vcf_duplicates.sh"


def get_args(argv: list[str]):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--relevant_coords",
        help="path to bed file describing relevant genomic subset to analyze",
        required=True,
    )
    parser.add_argument(
        "--model",
        required=True,
        help="path to pkl file containing conditional allele distributions per position of interest"
        ", supports glob pattern for multiple pkl files, and multiple assignments",
        action="append",
    )
    parser.add_argument(
        "--gvcf",
        required=True,
        help="path to gvcf file, (for getting the raw aligned reads information)",
    )
    parser.add_argument("--output_file", help="path to output file (vcf/vcf.gz/pickle)")
    parser.add_argument(
        "--strand_enrichment_pval_thresh",
        default=0.00001,
        type=float,
        help="p-value threshold for strand enrichment of alt-alleles (forward_pval * reverse_pval)",
    )
    parser.add_argument(
        "--lesser_strand_enrichment_pval_thresh",
        default=0.05,
        type=float,
        help="p-value threshold for the strand with higher p-value max(forward_pval, reverse_pval)",
    )
    parser.add_argument(
        "--min_gt_correlation",
        default=0.99,
        type=float,
        help="don't call loci with lower correlation between ground-truth and observed genotypes",
    )
    parser.add_argument(
        "--noise_ratio_for_unobserved_snps",
        default=0.02,
        type=float,
        help="expected probability of seeing alternative in unobserved snp for noisy positions",
    )
    parser.add_argument(
        "--noise_ratio_for_unobserved_indels",
        default=0.05,
        type=float,
        help="expected probability of seeing alternative in unobserved indel for noisy positions",
    )
    parser.add_argument(
        "--novel_detection_only",
        default=False,
        action="store_true",
        help="do not use information on known variants",
    )
    parser.add_argument(
        "--replace_to_known_genotype",
        default=False,
        action="store_true",
        help="in case reads match known genotype from ground-truth, use this genotype",
    )
    parser.add_argument(
        "--filter_uncorrelated",
        default=False,
        action="store_true",
        help="filter variants in positions where ref and alt " "conditioned genotype have similar distributions",
    )
    args = parser.parse_args(argv[1:])
    return args


class SystematicErrorCorrector:
    """
    Given an observed gvcf file, and expected-distributions model, decide for each variant,
    if it should be kept (observation is significantly different from the expected systematic error)
    Or should be filtered (observation is similar to expected systematic error)

    Works in two modes
    1. Regular - use all available information from the expected distributions model
    2. Novel detection only - do not use information on expected distributions given known alternative genotypes.

    Sites where the model is ambiguous,
    (observation can be explained equally well by two different ground-truth genotypes)
    are identified as uncorrelated sites.
    The default behaviour is to keep such variants as they are, since the method has no info about them.


    --filter_uncorrelated will cause calling these sites as reference.
        Notice this will filter out only sites where one of the ground-truth genotypes is the reference genotype
        Since if both alternatives are non-ref, then either way the variant is probably real.
    """

    # pylint: disable=too-many-arguments
    def __init__(  # noqa PLR0913
        self,
        relevant_coords: TextIO,
        cad_files: list[str],
        gvcf_reader: BufferedVariantReader,
        strand_enrichment_pval_thresh: float,
        lesser_strand_enrichment_pval_thresh: float,
        min_gt_correlation: float,
        noise_ratio_for_unobserved_snps: float,
        noise_ratio_for_unobserved_indels: float,
        output_file: str,
        *,
        novel_detection_only: bool,
        replace_to_known_genotype: bool,
        filter_uncorrelated: bool,
    ):
        self.relevant_coords = relevant_coords
        # map chromosome name to cad file name
        self.cad_files_dict = {f.split(".")[-2]: f for f in cad_files}
        self.distributions_per_chromosome = None
        self.gvcf_reader = gvcf_reader
        self.output_file = output_file
        self.filter_uncorrelated = filter_uncorrelated

        self.output_type = None
        if output_file.endswith((".vcf", ".vcf.gz")):
            self.output_type = OutputType.VCF
        elif output_file.endswith((".pickle", ".pkl")):
            self.output_type = OutputType.PICKLE
        else:
            raise ValueError("output file must end with vcf/vcf.gz/pickle/pkl suffixes")

        self.caller = SECCaller(
            strand_enrichment_pval_thresh,
            lesser_strand_enrichment_pval_thresh,
            min_gt_correlation,
            novel_detection_only=novel_detection_only,
            replace_to_known_genotype=replace_to_known_genotype,
        )
        SECRecord.noise_ratio_for_unobserved_snps = noise_ratio_for_unobserved_snps
        SECRecord.noise_ratio_for_unobserved_indels = noise_ratio_for_unobserved_indels

    def correct_systematic_errors(self):  # noqa PLR0912 C901
        self.add_header_lines()

        # Initialize output writers
        vcf_writer = None
        bed_writer = BedWriter(f"{self.output_file}.bed")
        if self.output_type == OutputType.VCF:
            vcf_writer = pysam.libcbcf.VariantFile(self.output_file, mode="w", header=self.gvcf_reader.header)

        # Initialize (chr, pos) tuples, relevant only for pickle output
        chr_pos_tuples = []

        with open(f"{self.output_file}.log", "w", encoding="utf-8") as log_stream:
            progress_count = 0
            current_chr = ""
            call_counts = defaultdict(int)
            # For each position in the relevant-coords bed file, compare observation to expected distribution of alleles
            for line in self.relevant_coords:
                progress_count += 1
                fields = line.split("\t")
                chrom, start, end = fields[0], fields[1], fields[2]

                # Load a new chromosome every-time a new chromosome in relevant_coords is encountered
                # IMPORTANT - relevant_coords must be sorted by chr,pos
                if chrom != current_chr:
                    with open(self.cad_files_dict[chrom], "rb") as cad_fh:
                        self.distributions_per_chromosome = pickle.load(cad_fh)  # noqa S301
                        current_chr = chrom

                for pos in range(int(start) + 1, int(end) + 1):
                    observed_variant = self.gvcf_reader.get_variant(chrom, pos)

                    # skip position if no variant was observed
                    if observed_variant is None:
                        continue

                    sample_info = observed_variant.samples[0]

                    # Handle no-call
                    if None in sample_info["GT"]:
                        if self.output_type == OutputType.VCF:
                            vcf_writer.write(observed_variant)
                        continue

                    # Check if called alternative allele matches excluded (noise) allele

                    called_non_excluded_alleles = self.did_call_non_excluded_alleles(
                        fields,
                        observed_variant,
                        sample_info,
                    )
                    # Call a non_noise_allele in case alternative allele is not excluded
                    # e.g a SNP in a position where the noise in a hmer indel
                    if called_non_excluded_alleles:
                        observed_alleles = get_filtered_alleles_str(observed_variant)
                        call = SECCall(
                            SECCallType.NON_NOISE_ALLELE,
                            observed_alleles,
                            get_genotype(sample_info),
                            [],
                        )
                    # Execute SEC caller on the observed variant
                    # and expected_distribution loaded from the memDB (pickle)
                    else:
                        expected_distribution = self.distributions_per_chromosome.get(pos, None)
                        call = self.caller.call(observed_variant, expected_distribution)

                    for sec_record in call.sec_records:
                        log_stream.write(f"{sec_record}\n")

                    call_counts[call.call_type.value] += 1

                    if progress_count % 100000 == 0:
                        logger.info(f"processed {progress_count} records. call_counts: {dict(call_counts)}")

                    if self.output_type == OutputType.PICKLE:
                        self.__process_call_pickle_output(call, chr_pos_tuples, chrom, pos)

                    if self.output_type == OutputType.VCF:
                        self.__process_call_vcf_output(call, observed_variant, sample_info, vcf_writer)

                    self.__process_call_bed_output(bed_writer, call, chrom, pos)

                    log_stream.write(f"{chrom}\t{pos}\t{call}\n\n")

            self.__finalize(bed_writer, chr_pos_tuples, log_stream, vcf_writer)
        logger.info("Systematic error correction finished with success")

    @staticmethod
    def did_call_non_excluded_alleles(fields, observed_variant, sample_info):
        called_non_excluded_alleles = False
        if len(fields) > 3:  # noqa PLR2004
            excluded_refs = ast.literal_eval(fields[3])
            flat_excluded_refs = list(itertools.chain(*excluded_refs))
            all_excluded_alts = ast.literal_eval(fields[4])
            called_ref = observed_variant.ref
            called_alts = set(get_filtered_alleles_list(sample_info)).intersection(observed_variant.alts)
            if len(called_alts) > 0:
                called_non_excluded_alleles = not _are_all_called_alleles_excluded(
                    flat_excluded_refs,
                    all_excluded_alts,
                    called_ref,
                    called_alts,
                )
        return called_non_excluded_alleles

    def __finalize(self, bed_writer, chr_pos_tuples, log_stream, vcf_writer):
        if self.output_type == OutputType.VCF:
            vcf_writer.close()
            pysam.tabix_index(self.output_file, preset="vcf", force=True)
        if self.output_type == OutputType.PICKLE:
            exclusion_filter = Blacklist(
                blacklist=set(chr_pos_tuples),
                annotation="SEC",
                selection_fcn=VariantSelectionFunctions.ALL,
                description="Variant is filtered since alt alleles have high likelihood to be "
                "generated by a systematic error",
            )
            with open(self.output_file, "wb") as out_pickle_file:
                pickle.dump([exclusion_filter], out_pickle_file)
        bed_writer.close()
        log_stream.close()

    def add_header_lines(self):
        self.gvcf_reader.header.add_meta(
            "FORMAT",
            items=[
                ("ID", "ST"),
                ("Number", 1),
                ("Type", "String"),
                ("Description", "SECType"),
            ],
        )
        self.gvcf_reader.header.add_meta(
            "FORMAT",
            items=[
                ("ID", "SPV"),
                ("Number", 1),
                ("Type", "Float"),
                ("Description", "SECPValue"),
            ],
        )

    @staticmethod
    def __process_call_vcf_output(
        call: SECCall,
        observed_variant: VariantRecord,
        sample_info,
        vcf_writer: VariantFile,
    ):
        """
        Write all vcf lines with additional:
        1. ST field: (reference, novel, known, uncorrelated, non_noise_allele)
        2. updated GQ whenever appropriate (known variants)
        3. Corrected genotype, to ref genotype when filtered the variant
        """
        sample_info["ST"] = str(call.call_type.value)
        sample_info["SPV"] = call.novel_variant_p_value
        if call.call_type == SECCallType.REFERENCE:
            gt_indices = call.get_genotype_indices_tuple()
            sample_info["GT"] = gt_indices
        elif call.call_type == SECCallType.KNOWN:
            observed_variant.alleles = call.get_alleles_list()
            sample_info = observed_variant.samples[0]
            sample_info["GT"] = call.get_genotype_indices_tuple()
        if call.genotype_quality is not None:
            sample_info["GQ"] = call.genotype_quality
        vcf_writer.write(observed_variant)

    @staticmethod
    def __process_call_bed_output(bed_writer: BedWriter, call: SECCall, chrom: str, pos: int):
        """
        output all positions with sec_call_type, for assess_concordance_with_exclusion_h5.py
        """
        bed_writer.write(chrom, pos - 1, pos, call.call_type.value, call.novel_variant_p_value)

    def __process_call_pickle_output(self, call: SECCall, chr_pos_tuples: list[tuple], chrom: str, pos: int):
        """
        output only positions which were decided to have the reference genotype
        (or uncorrelated if directed to filter them)
        """
        if call.call_type == SECCallType.REFERENCE or (
            call.call_type == SECCallType.UNCORRELATED and self.filter_uncorrelated
        ):
            chr_pos_tuples.append((chrom, pos))


class OutputType(Enum):
    VCF = 1
    PICKLE = 2


def _are_all_called_alleles_excluded(
    excluded_refs: list[str],
    all_excluded_alts: list[list[str]],
    called_ref: str,
    called_alts: set[str],
) -> bool:
    """
    search for each called ref->alt pair in excluded alleles
    Return True iff ALL the called ref->alt pairs are excluded
    Example:
        if A->G is excluded, and called A->G return True
        if A->G is excluded, and called A->C return False
        if A->G is excluded, and called A->G/C return False
        if A->G, A->C are excluded, and called A->G/C return True
    """
    for excluded_ref, excluded_alts in zip(excluded_refs, all_excluded_alts, strict=False):
        if called_ref == excluded_ref:
            non_excluded_alts = called_alts.difference(excluded_alts)
            if len(non_excluded_alts) == 0:
                return True
    return False


def run(argv: list[str]):
    """
    filter out variants which appear like systematic-errors, while keeping those which are not well explained by errors
    """
    args = get_args(argv)
    out_file = args.output_file
    if os.path.exists(args.gvcf):
        dedup_input_vcf = f"{out_file}.input.nodup.vcf.gz"
        cmd = [
            "sh",
            REMOVE_VCF_DUPLICATES_SHELL_SCRIPT,
            args.gvcf,
            dedup_input_vcf,
        ]
        subprocess.call(cmd)
        gvcf_reader = BufferedVariantReader(dedup_input_vcf)
    else:
        logger.error("gvcf input does not exist")
        return

    with open(args.relevant_coords, encoding="utf-8") as relevant_coords:
        pickle_files = []
        for model_file_name in args.model:
            if "*" in model_file_name:
                pickle_files.extend(glob.glob(model_file_name))
            else:
                pickle_files.append(model_file_name)

        SystematicErrorCorrector(
            relevant_coords=relevant_coords,
            cad_files=pickle_files,
            gvcf_reader=gvcf_reader,
            strand_enrichment_pval_thresh=args.strand_enrichment_pval_thresh,
            lesser_strand_enrichment_pval_thresh=args.lesser_strand_enrichment_pval_thresh,
            min_gt_correlation=args.min_gt_correlation,
            noise_ratio_for_unobserved_snps=args.noise_ratio_for_unobserved_snps,
            noise_ratio_for_unobserved_indels=args.noise_ratio_for_unobserved_indels,
            output_file=out_file,
            novel_detection_only=args.novel_detection_only,
            replace_to_known_genotype=args.replace_to_known_genotype,
            filter_uncorrelated=args.filter_uncorrelated,
        ).correct_systematic_errors()


def main():
    run(sys.argv)


if __name__ == "__main__":
    main()
