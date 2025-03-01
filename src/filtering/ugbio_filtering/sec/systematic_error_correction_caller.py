import math

import numpy as np
from pysam import VariantRecord
from ugbio_core.vcfbed.genotype import Genotype
from ugbio_core.vcfbed.pysam_utils import get_genotype, get_genotype_indices, has_candidate_alternatives

from ugbio_filtering.sec.conditional_allele_distribution import ConditionalAlleleDistribution
from ugbio_filtering.sec.conditional_allele_distribution_correlator import correlate_sec_records
from ugbio_filtering.sec.evaluate_locus_observation import evaluate_observation
from ugbio_filtering.sec.systematic_error_correction_call import SECCall, SECCallType
from ugbio_filtering.sec.systematic_error_correction_record import SECRecord


# pylint: disable=too-few-public-methods
class SECCaller:
    """
    Calibrate call of an observed variant given expected allele distributions
    1. Evaluate each conditioned genotype, to produce sec_records.
    2. Decide upon a SECCall given sec_records

    If all sec_records are rejected (observed doesn't fit expected) call a novel genotype.
    Other-wise call the sec_record of the genotype with the maximum likelihood.

    sec_records should not be empty even if variant was never observed before.
    Such an "empty" sec_record considers the probability of a new error to reduce FP novel genotype calls

    The call function is immutable, it does not change any of its inputs, only returns a new SECCall
    """

    def __init__(
        self,
        strand_enrichment_pval_thresh: float,
        lesser_strand_enrichment_pval_thresh: float,
        min_gt_correlation: float,
        *,
        novel_detection_only: bool,
        replace_to_known_genotype: bool,
    ):
        self.stand_enrichment_pval_thresh = strand_enrichment_pval_thresh
        self.lesser_strand_enrichment_pval_thresh = lesser_strand_enrichment_pval_thresh
        self.min_gt_correlation = min_gt_correlation
        self.novel_detection_only = novel_detection_only
        self.replace_to_known_genotype = replace_to_known_genotype

    def call(  # noqa C901 PLR0912
        self,
        observed_variant: VariantRecord,
        expected_distribution: ConditionalAlleleDistribution,
    ) -> SECCall:
        sample_info = observed_variant.samples[0]
        observed_genotype = get_genotype_indices(sample_info)
        observed_alleles = ",".join(observed_variant.alleles)

        if not has_candidate_alternatives(observed_variant) or sum(observed_variant.samples[0]["SB"]) == 0:
            return SECCall(SECCallType.REFERENCE, observed_alleles, observed_genotype, [], None, 1)

        sec_records = evaluate_observation(observed_variant, expected_distribution)

        gt_correlation = None
        if len(sec_records) > 1:
            gt_correlation = correlate_sec_records(sec_records)
            if self.novel_detection_only and gt_correlation < self.min_gt_correlation:
                return SECCall(
                    SECCallType.UNCORRELATED,
                    observed_alleles,
                    observed_genotype,
                    sec_records,
                    novel_variant_p_value=None,
                    gt_correlation=gt_correlation,
                )

        if self.novel_detection_only:
            # remove alternative conditioned genotypes
            sec_records = [r for r in sec_records if Genotype(r.conditioned_genotype).is_reference()]

        likelihood_sorted_sec_records = sorted(sec_records, key=lambda r: r.likelihood, reverse=True)
        best_sec_record = likelihood_sorted_sec_records[0]
        genotype_quality = None

        # novel genotype (doesn't fit any known genotype)
        # pylint: disable=no-else-return
        if (
            best_sec_record.strand_enrichment_pval < self.stand_enrichment_pval_thresh
            and best_sec_record.lesser_strand_enrichment_pval < self.lesser_strand_enrichment_pval_thresh
        ):
            if "*" in get_genotype(sample_info) and 0 in sample_info.allele_indices:
                return SECCall(
                    SECCallType.REFERENCE,
                    observed_alleles,
                    Genotype(observed_genotype).convert_to_reference(),
                    sec_records,
                    novel_variant_p_value=best_sec_record.strand_enrichment_pval,
                )

            call_type = SECCallType.NOVEL if best_sec_record.were_observed_alleles_in_db else SECCallType.UNOBSERVED
            return SECCall(
                call_type,
                observed_alleles,
                observed_genotype,
                sec_records,
                novel_variant_p_value=best_sec_record.strand_enrichment_pval,
            )
        else:
            # locus has multiple alternative hypothesis (known variant with info on more than one genotype)
            if len(sec_records) > 1:
                gt_correlation = correlate_sec_records(likelihood_sorted_sec_records)
                reference_conditioned_sec_record = [
                    r for r in sec_records if Genotype(r.conditioned_genotype).is_reference()
                ]
                if len(reference_conditioned_sec_record) > 0:
                    reference_conditioned_sec_record = reference_conditioned_sec_record[0]
                ref_conditioned_likelihood_ratio = (
                    reference_conditioned_sec_record.likelihood / best_sec_record.likelihood
                )
                # ground-truth genotype is not correlated with observed data
                if gt_correlation < self.min_gt_correlation and ref_conditioned_likelihood_ratio > 1 / 100:
                    return SECCall(
                        SECCallType.UNCORRELATED,
                        observed_alleles,
                        observed_genotype,
                        sec_records,
                        novel_variant_p_value=best_sec_record.strand_enrichment_pval,
                        gt_correlation=gt_correlation,
                    )

                genotype_quality = self.__calc_genotype_quality(likelihood_sorted_sec_records)

            reference_conditioned_record = None
            for sec_record in sec_records:
                if Genotype(sec_record.conditioned_genotype).is_reference():
                    reference_conditioned_record = sec_record

            if self.replace_to_known_genotype:
                alleles = best_sec_record.conditioned_alleles
                called_genotype = best_sec_record.conditioned_genotype
            else:
                alleles = observed_alleles
                called_genotype = observed_genotype

            if Genotype(best_sec_record.conditioned_genotype).is_reference():
                call_type = SECCallType.REFERENCE
                called_genotype = Genotype(called_genotype).convert_to_reference()
            elif reference_conditioned_record is None or not reference_conditioned_record.were_observed_alleles_in_db:
                call_type = SECCallType.UNOBSERVED
            else:
                call_type = SECCallType.KNOWN

            return SECCall(
                call_type,
                alleles,
                called_genotype,
                sec_records,
                genotype_quality,
                novel_variant_p_value=best_sec_record.strand_enrichment_pval,
                gt_correlation=gt_correlation,
            )

    @staticmethod
    def __calc_genotype_quality(likelihood_sorted_sec_records: list[SECRecord]) -> int:
        best_likelihood = likelihood_sorted_sec_records[0].likelihood
        second_best_likelihood = likelihood_sorted_sec_records[1].likelihood
        if np.isnan(best_likelihood) or np.isnan(second_best_likelihood):
            return 0
        if best_likelihood > 0 and second_best_likelihood > 0:
            return 10 * int(math.log2(best_likelihood) - math.log2(second_best_likelihood))
        if best_likelihood > 0:
            return 100
        return 0
