import argparse
import logging
from array import array
from bisect import bisect_right
from math import log

import numpy as np
import pysam
import statsmodels.api as sm
import ugbio_core.flow_format.flow_based_read as fbr
import ugbio_core.math_utils as phred
from pyfaidx import Fasta

# Logger setup
logger = logging.getLogger(__name__)

# Global constants
BAM_CSOFT_CLIP = 4
HMER_NUM_POSSIBLE_SIZES = 21
SEQUENCE_CONTEXT_SIZE = 30
HMER_SPLIT_MIDPOINT = 0.5
EM_INITIAL_PI = 0.05
EM_ITERATIONS = 10
PI_MIN = 0.001
PI_MAX = 0.999
READS_THRESHOLD_CYCLE_LOW = 500
CYCLE_LOW = 200
READS_THRESHOLD_CYCLE_HIGH = 1500
CYCLE_HIGH = 5000
HIGH_CONF_THRESHOLD = 1.1
NORMAL_SCORE_MAX = 25.0
SCORE_EXPONENT = 0.25
TUMOR_SCORE_MAX = 40.0
TUMOR_MIXTURE_MULTIPLIER_FACTOR = 1000
GERMLINE_VCF_COUNT = 6
NUM_DIRECTIONS = 2
DIRECTION_PREFIXES = ["fw_", "bw_"]
LIKELIHOOD_LOG_BASE = 10
LOG_MIN_PVAL_BASE = 0.0000000001
SCORE_ADJUSTMENT_FACTOR = 0.6
MAX_LOG_LIKELIHOOD = -400.0
MAX_COMBINED_SCORE = 20.0
NUM_SETS = 3
SET_RANGE_START = 1
SET_RANGE_END = 4
READS_QUALITY_THRESHOLD = 0
MIN_MAPPING_QUALITY = 5


def apply_variant(ref_fasta: Fasta, ref_pos: list[str | int], variant: list[int | str]) -> tuple[int, str | None]:
    """Calculate hmer size at variant position.

    Args:
        ref_fasta: Reference FASTA file
        ref_pos: [chromosome, position]
        variant: [position, ref, alt]

    Returns:
        Tuple of (hmer_size, sequence) or (-1, None) if invalid
    """
    try:
        seq = (
            str(ref_fasta[ref_pos[0]][ref_pos[1] - SEQUENCE_CONTEXT_SIZE : variant[0] - 1])
            + variant[2]
            + str(ref_fasta[ref_pos[0]][variant[0] + len(variant[1]) - 1 : ref_pos[1] + SEQUENCE_CONTEXT_SIZE])
        )
        pos = SEQUENCE_CONTEXT_SIZE
        size = 0
        nuc = seq[pos]
        while seq[pos] == nuc:
            pos -= 1
        pos += 1
        while seq[pos] == nuc:
            pos += 1
            size += 1
        return size, seq
    except (IndexError, TypeError):
        return -1, None


def load_bed_intervals(bed_path: str) -> dict[str, list[tuple[int, int]]]:
    """Load and merge overlapping BED intervals by chromosome.

    Args:
        bed_path: Path to BED file

    Returns:
        Dictionary mapping chromosome names to sorted, merged interval lists
    """
    by_chrom = {}
    with open(bed_path) as f:
        for line in f:
            if not line.strip() or line.startswith(("#", "track", "browser")):
                continue
            chrom, start, end, *_ = line.rstrip("\n").split("\t")
            start, end = int(start), int(end)
            by_chrom.setdefault(chrom, []).append((start, end))

    # sort + merge per chrom
    merged = {}
    for chrom, ivs in by_chrom.items():
        ivs.sort()
        out = []
        for s, e in ivs:
            if not out or s > out[-1][1]:
                out.append([s, e])
            else:
                out[-1][1] = max(out[-1][1], e)
        merged[chrom] = [(s, e + 1) for s, e in out]
    return merged


def pos_in_bed(merged_intervals: dict[str, list[tuple[int, int]]], chrom: str, pos0: int) -> bool:
    """Check if position falls within any BED interval.

    Args:
        merged_intervals: Dictionary of merged intervals by chromosome
        chrom: Chromosome name
        pos0: 0-based coordinate (BED intervals are [start, end))

    Returns:
        True if position is in an interval, False otherwise
    """
    ivs = merged_intervals.get(chrom)
    if not ivs:
        return False
    # find rightmost interval with start <= pos0
    starts = [s for s, _ in ivs]
    i = bisect_right(starts, pos0) - 1
    if i < 0:
        return False
    s, e = ivs[i]
    return s <= pos0 < e


def calc_exp(frequencies: list[float]) -> float:
    """Calculate weighted expectation from frequency distribution."""
    return sum(i * freq for i, freq in enumerate(frequencies))


def find_best_mixture_hmer_sizes(results: list) -> tuple[int, int]:
    """Find best and second-best hmer sizes from mixture likelihood.

    Args:
        results: List of read results with probability distributions

    Returns:
        Tuple of (best_hmer_size, second_best_hmer_size)
    """
    hmer_size_to_likely = [[0, i] for i in range(HMER_NUM_POSSIBLE_SIZES)]
    for result in results:
        size_prob = result[1]
        for size in range(HMER_NUM_POSSIBLE_SIZES):
            hmer_size_to_likely[size][0] -= log(1 - size_prob[size])
    hmer_size_to_likely.sort()
    return hmer_size_to_likely[-1][1], hmer_size_to_likely[-2][1]


def get_max_nuc(nuc_list: list[str]) -> str | None:
    """Find the most frequent nucleotide in a list.

    Returns the nucleotide (base) that appears most frequently in the input list.
    Uses Counter from collections to efficiently count occurrences.

    Args:
        nuc_list: List of nucleotide characters (strings)

    Returns:
        The most common nucleotide as a string, or None if list is empty
    """
    if not nuc_list:
        return None
    from collections import Counter

    return Counter(nuc_list).most_common(1)[0][0]


def filter_reads(results: list, nuc: str, strand: int) -> list:
    """Filter reads by nucleotide and strand.

    Args:
        results: List of read results
        nuc: Nucleotide to filter by
        strand: Strand direction (0 or 1)

    Returns:
        Filtered list of reads matching criteria
    """
    return [res for res in results if res[0] == nuc and not res[4] and res[5] and bool(res[2]) == bool(strand)]


def process_reads(reads: list) -> list:
    """Process reads to extract expectation and confidence metrics.

    Args:
        reads: List of raw read results

    Returns:
        List of processed reads with computed metrics
    """
    processed_results = []
    for res in reads:
        expect = calc_exp(res[1])
        sorted_probs = sorted(res[1])
        max_conf = sorted_probs[-1]
        high_conf = sorted_probs[-3]
        processed_results.append((expect, max_conf, high_conf, res[2], res[3]))
    return processed_results


def calc_parameters(reads: list) -> list:
    """Calculate expectation split and confidence parameters from reads.

    This function computes statistical parameters from hmer read data to establish
    thresholds for quality filtering. It uses the 2/3 quantile of fractional expectation
    remainders to define adaptive split points for probability binning.

    Algorithm:
    1. Calculates fractional remainders (expectation - floor(expectation)) for all reads
    2. Sorts remainders and extracts the 2/3 quantile value
    3. Adapts split thresholds based on read coverage:
       - Low coverage: uses fixed thresholds
       - High coverage: uses 3-point or 5-point split based on 2/3 and 3/7 quantiles
    4. Determines confidence threshold and cycle count based on coverage level

    Args:
        reads: List of processed read tuples, each containing (expectation, max_confidence,
               high_confidence, ref_val, other_val) from process_reads output

    Returns:
        List containing:
        - [0]: exp_split tuple defining probability boundaries (2-5 elements representing
               fractional thresholds for hmer size binning)
        - [1]: (high_conf, cycle) tuple where:
            * high_conf: Quality confidence threshold for filtering (float)
            * cycle: Read cycle threshold indicating base quality criteria (int)
    """
    remainder_list = sorted([abs(x[0] - round(x[0])) for x in reads])
    idx_2_3 = len(remainder_list) * 2 // 3
    remainder = remainder_list[idx_2_3]
    exp_split = (remainder, HMER_SPLIT_MIDPOINT, 1 - remainder)

    if len(remainder_list) > READS_THRESHOLD_CYCLE_LOW:
        high_conf_list = sorted([x[2] for x in reads])
        high_conf = high_conf_list[len(high_conf_list) * 2 // 3]
        cycle = CYCLE_LOW
        if len(remainder_list) > READS_THRESHOLD_CYCLE_HIGH:
            remainder2 = remainder_list[len(remainder_list) * 3 // 7]
            exp_split = (remainder2, remainder, HMER_SPLIT_MIDPOINT, 1 - remainder, 1 - remainder2)
    else:
        cycle = CYCLE_HIGH
        high_conf = HIGH_CONF_THRESHOLD
    return [exp_split, (high_conf, cycle)]


def get_cell(read: tuple, params: list, cell_shift: int, exp_move: float = 0) -> int:
    """Calculate cell index for read based on parameters.

    Args:
        read: Read tuple with expectation and probability data
        params: Parameters from calc_parameters
        cell_shift: Cell shift value
        exp_move: Optional expectation move adjustment

    Returns:
        Cell index
    """
    split_values = params[0]
    expect = read[0] + exp_move
    floor_expect = int(expect)
    cell = floor_expect * cell_shift
    remainder = expect - floor_expect
    for i in split_values:
        if remainder < i:
            break
        cell += 2
    cell += int(read[2] < params[1][0] and read[4] < params[1][1])
    return cell


def get_pval(exps: list[float], groups: list[int], w: list[float]) -> float:
    """Calculate p-value testing for group effect using weighted least squares regression.

    Performs statistical hypothesis testing to detect significant differences in hmer
    expectation values between normal (group 0) and tumor (group 1) samples.

    Hypothesis Testing:
    - H0 (Null): No difference in expectation values between groups (coefficient = 0)
    - H1 (Alternative): Significant difference exists between groups (coefficient ≠ 0)

    Algorithm:
    1. Fits initial WLS regression: expectation ~ intercept + group_effect
    2. Extracts p-value for the group coefficient
    3. Tests alternative weight schemes to capture different aspects of the data:
       - added_weights_max: Emphasizes differences at extremes (higher values get higher weights)
       - added_weights_min: Emphasizes central values (lower values get higher weights)
    4. Returns minimum p-value across all weight schemes (conservative approach for detecting signal)

    Args:
        exps: List of expectation values (hmer sizes) from reads
        groups: List of group labels (0=normal samples, 1=tumor samples), same length as exps
        w: List of confidence weights for each read, same length as exps

    Returns:
        p-value (float): Minimum p-value from group coefficient across weight schemes.
                        Lower values indicate stronger evidence of group difference.
    """
    y = np.asarray(exps, dtype=float)
    g = np.asarray(groups, dtype=float)  # 0 or 1
    w = np.asarray(w, dtype=float)

    x_matrix = sm.add_constant(g)  # intercept + group effect
    res = sm.WLS(y, x_matrix, weights=w).fit()
    p_val = float(res.pvalues[1])

    length = len(exps)
    exp_position = sorted(enumerate(exps), key=lambda x: x[1])
    indices = [idx for idx, _ in exp_position]

    # Try weighted regressions with adjusted weights
    added_weights_max = np.asarray(w, dtype=float).copy()
    added_weights_min = np.asarray(w, dtype=float).copy()
    for i, idx in enumerate(indices):
        added_weights_max[idx] *= (((i + 1.0) / length) ** 2) + 1.0
        added_weights_min[idx] *= (((length - i) / length) ** 2) + 1.0

    res = sm.WLS(y, x_matrix, weights=added_weights_max).fit()
    p_val = min(p_val, float(res.pvalues[1]))
    res = sm.WLS(y, x_matrix, weights=added_weights_min).fit()
    p_val = min(p_val, float(res.pvalues[1]))
    return p_val


def get_machine_likelihood(filtered_results: list, best: int, second: int) -> tuple[float, float]:
    """Calculate machine likelihood ratio and mixture parameter.

    Args:
        filtered_results: List of filtered read results
        best: Index of best hmer size hypothesis
        second: Index of alternative hmer size hypothesis

    Returns:
        Tuple of (likelihood_ratio, mixture_parameter)
    """
    h0_probs = np.asarray([result[1][best] for result in filtered_results], dtype=float)
    h1_probs = np.asarray([result[1][second] for result in filtered_results], dtype=float)
    h0_likely = sum(log(prob) for prob in h0_probs)

    # EM algorithm to estimate mixture parameter
    pi = EM_INITIAL_PI
    for _ in range(EM_ITERATIONS):
        denom = pi * h1_probs + (1.0 - pi) * h0_probs
        r = (pi * h1_probs) / denom
        pi = min(max(float(r.mean()), PI_MIN), PI_MAX)

    h1_likely = sum(log(h1_probs[i] * pi + (1 - pi) * h0_probs[i]) for i in range(len(h0_probs)))
    return h1_likely - h0_likely, pi


def is_pass(rec: pysam.VariantRecord) -> bool:
    """Check if variant record passes filters.

    Args:
        rec: VCF record to check

    Returns:
        True if record has PASS filter or no filters, False otherwise
    """
    return "PASS" in rec.filter or len(rec.filter) == 0


def direction_score(normal_score: float, normal_mixture: float, tumor_score: float, tumor_mixture: float) -> float:
    """Calculate directional score for tumor vs normal comparison.

    Args:
        normal_score: Normal sample quality score
        normal_mixture: Normal sample mixture parameter
        tumor_score: Tumor sample quality score
        tumor_mixture: Tumor sample mixture parameter

    Returns:
        Computed directional score
    """
    normal_score_clamped = max(min(normal_score, NORMAL_SCORE_MAX), 0)
    normal_modified_mixture = normal_mixture * (normal_score_clamped**SCORE_EXPONENT)
    tumor_modified_mixture = tumor_mixture - normal_modified_mixture
    tumor_mixture_multiplier = max(0, min((max(0, tumor_modified_mixture) ** 2) * TUMOR_MIXTURE_MULTIPLIER_FACTOR, 1))
    tumor_modified_score = max(min(tumor_score, TUMOR_SCORE_MAX), 0) / 2
    return tumor_modified_score * tumor_mixture_multiplier


def combine_scores(
    ttest_score: float,
    likely_score: float,
    likely_mixture: float,
    normal_ml_score: float,
    normal_ml_mixture: float,
    tumor_ml_score: float,
    tumor_ml_mixture: float,
) -> float:
    """Combine multiple scoring metrics into single score.

    Args:
        ttest_score: Statistical t-test score
        likely_score: Likelihood score
        likely_mixture: Mixture parameter from likelihood
        normal_ml_score: Machine learning score for normal
        normal_ml_mixture: Machine learning mixture for normal
        tumor_ml_score: Machine learning score for tumor
        tumor_ml_mixture: Machine learning mixture for tumor

    Returns:
        Combined score from all metrics
    """
    score = ttest_score / 4
    likely_mixture_multiplier = min((likely_mixture**2) * TUMOR_MIXTURE_MULTIPLIER_FACTOR, 1)
    score += max(0, likely_score) * likely_mixture_multiplier
    ml_score = direction_score(normal_ml_score, normal_ml_mixture, tumor_ml_score, tumor_ml_mixture)
    score += ml_score
    return score


def fill_direction_results_with_error() -> dict:
    """Create error result dictionary with -1 for all fields.

    Returns:
        Dictionary with all fields set to -1
    """
    fields = [
        "normal_exp",
        "tumor_exp",
        "normal_cvg",
        "tumor_cvg",
        "ttest_score",
        "likely_score",
        "mixture",
        "normal_main_hmer",
        "tumor_main_hmer",
        "normal_second_hmer",
        "tumor_second_hmer",
        "normal_ml_score",
        "normal_ml_mixture",
        "tumor_ml_score",
        "tumor_ml_mixture",
        "tot_score",
    ]
    return {field: -1 for field in fields}


def get_hmer_qualities_from_pileup_element(  # noqa: C901
    pe: pysam.PileupRead, max_hmer: int = 20, min_call_prob: float = 0.1, *, soft_clipping_as_edge: bool = True
) -> tuple:
    """Return hmer length probabilities for a single PileupRead element.

    Parameters
    ----------
    pe : pysam.PileupRead
        PileupRead element
    max_hmer : int
        Maximum hmer length that we call
    min_call_prob : float
        Minimum probability for the called hmer length
    soft_clipping_as_edge : bool
        Whether to treat soft-clipped edges as edges for probability smearing

    Returns
    -------
    tuple:
        (nucleotide, probabilities, is_forward, cycle, is_edge, is_not_duplicate)
    """
    filler = 10 ** (-35 / 10) / (max_hmer + 1)

    qpos = pe.query_position_or_next
    hnuc = str(pe.alignment.query_sequence)[qpos]
    qstart = qpos
    while qstart > 0 and str(pe.alignment.query_sequence)[qstart - 1] == hnuc:
        qstart -= 1
    qend = qpos + 1
    while qend < len(str(pe.alignment.query_sequence)) and str(pe.alignment.query_sequence)[qend] == hnuc:
        qend += 1

    hmer_probs = np.zeros(max_hmer + 1)
    hmer_length = qend - qstart
    key = fbr.generate_key_from_sequence(str(pe.alignment.query_sequence), "TGCA")
    cumsum_key = np.cumsum(key)
    cycle = np.searchsorted(cumsum_key, pe.query_position_or_next)
    # smear probabilities
    is_soft_clipped = False
    if pe.alignment.cigartuples and soft_clipping_as_edge:
        # Check if base is adjacent to soft-clipping at start
        if pe.alignment.cigartuples[0][0] == BAM_CSOFT_CLIP:
            soft_clip_end = pe.alignment.cigartuples[0][1]
            if qstart - soft_clip_end <= 1:
                is_soft_clipped = True
        # Check if base is adjacent to soft-clipping at end
        if pe.alignment.cigartuples[-1][0] == BAM_CSOFT_CLIP:
            query_length = pe.alignment.query_length
            soft_clip_start = query_length - pe.alignment.cigartuples[-1][1]
            if qend - soft_clip_start <= 1:
                is_soft_clipped = True
    if qstart == 0 or qend == len(str(pe.alignment.query_sequence)) or is_soft_clipped:
        hmer_probs[:] = 1.0
        is_edge = True
    else:
        query_qualities = pe.alignment.query_qualities
        if query_qualities is None:
            raise ValueError("query_qualities is None")
        qual = query_qualities[qstart:qend]
        probs = phred.unphred(np.asarray(qual))
        tp_tag = pe.alignment.get_tag("tp")
        if not isinstance(tp_tag, list | np.ndarray | array):
            raise ValueError("tp tag must be a list or array, " + str(pe.alignment.query_name))
        tps = tp_tag[qstart:qend]
        for tpval, p in zip(tps, probs, strict=False):
            hmer_probs[tpval + hmer_length] += p
        hmer_probs = np.clip(hmer_probs, filler, None)
        hmer_probs[hmer_length] = 0
        hmer_probs[hmer_length] = max(1 - np.sum(hmer_probs), min_call_prob)
        is_edge = False
    hmer_probs /= np.sum(hmer_probs)
    return hnuc, hmer_probs, not pe.alignment.is_reverse, cycle, is_edge, not pe.alignment.is_duplicate


def _setup_vcf_headers(vcf_file, verbose):
    """Setup VCF headers and return file handle.

    Args:
        vcf_file: Path to input VCF file
        verbose: If True, add all fields; if False, only add mixture and tot_score

    Returns:
        pysam.VariantFile with headers configured
    """
    f = pysam.VariantFile(vcf_file)

    # Base fields for each direction
    base_fields = [
        ("normal_exp", "A", "Float", "Normal hmer size expectancy"),
        ("tumor_exp", "A", "Float", "Tumor hmer size expectancy"),
        ("normal_cvg", "A", "Integer", "Number of normal reads"),
        ("tumor_cvg", "A", "Integer", "Number of tumor reads"),
        ("ttest_score", "A", "Float", "Score from ttest"),
        ("likely_score", "A", "Float", "Score from likelihood"),
        ("mixture", "A", "Float", "Best mixture size"),
        ("normal_main_hmer", "A", "Integer", "Best hmer of normal"),
        ("tumor_main_hmer", "A", "Integer", "Best hmer of tumor"),
        ("normal_second_hmer", "A", "Integer", "Second best hmer of normal"),
        ("tumor_second_hmer", "A", "Integer", "Second best hmer of tumor"),
        ("normal_ml_score", "A", "Float", "Normal ML score"),
        ("normal_ml_mixture", "A", "Float", "Normal ML mixture"),
        ("tumor_ml_score", "A", "Float", "Tumor ML score"),
        ("tumor_ml_mixture", "A", "Float", "Tumor ML mixture"),
        ("tot_score", "A", "Float", "Combined ML score"),
    ]

    if verbose:
        # Verbose mode: Add all fields
        # Add non-prefixed fields (for backward compatibility)
        for name, number, typ, desc in base_fields:
            f.header.info.add(name, number=number, type=typ, description=desc)

        # Add prefixed fields for each direction
        for prefix in DIRECTION_PREFIXES:
            for name, number, typ, desc in base_fields:
                f.header.info.add(f"{prefix}{name}", number=number, type=typ, description=desc)

        # Add debugging fields (verbose mode only)
        f.header.info.add("ref_hmer_size", number=1, type="Integer", description="Reference hmer size")
        f.header.info.add("other_variant", number=1, type="Integer", description="Other variant")
    else:
        # Minimal mode: Only add mixture and tot_score
        f.header.info.add("mixture", number="A", type="Float", description="Average mixture across directions")
        f.header.info.add("tot_score", number="A", type="Float", description="Combined ML score")

    return f


def _check_normal_other_variants(
    rec,
    ref_fasta,
    chrom: str,
    pos: int,
    ref_hmer_size: int,
    normal_germline,
) -> int:
    """Check if record has conflicting variants in the region.

    Args:
        rec: VCF record
        ref_fasta: Reference FASTA
        chrom: Chromosome
        pos: Position
        ref_hmer_size: Reference homopolymer size
        normal_germline: Normal germline VCF

    Returns:
        1 if other conflicting variants found, 0 otherwise
    """

    other_variant = 0

    # Check normal germline variants
    germline_normal_vars = normal_germline.fetch(chrom, rec.pos - 2, rec.pos + rec.info["X_HIL"][0] + 4)
    for var in germline_normal_vars:
        if is_pass(var):
            size = apply_variant(ref_fasta, [chrom, pos], [var.pos, var.ref, var.alts[0]])[0]
            if size != ref_hmer_size:
                other_variant = 1

    return other_variant


def _get_variant_type(ref_len: int, alt_len: int) -> str:
    """Determine variant type based on reference and alternate lengths.

    Args:
        ref_len: Length of reference allele
        alt_len: Length of alternate allele

    Returns:
        'deletion', 'insertion', or 'snp'
    """
    if ref_len > alt_len:
        return "deletion"
    elif ref_len < alt_len:
        return "insertion"
    else:
        return "snp"


def _check_somatic_variants(
    rec, ref_fasta: object, chrom: str, pos: int, ref_hmer_size: int, vcf_file, rec_type: str
) -> bool:
    """Check for conflicting variants in somatic VCF with higher QUAL.

    Args:
        rec: VCF record
        ref_fasta: Reference FASTA
        chrom: Chromosome
        pos: Position
        ref_hmer_size: Reference homopolymer size
        vcf_file: Input VCF file for fetching variants
        rec_type: Variant type (insertion/deletion/snp)

    Returns:
        True if conflicting variant found, False otherwise
    """
    this_loc_variants = vcf_file.fetch(chrom, rec.pos - 2, rec.pos + rec.info["X_HIL"][0] + 4)
    for var in this_loc_variants:
        if var.qual > rec.qual:
            size = apply_variant(ref_fasta, [chrom, pos], [var.pos, var.ref, var.alts[0]])[0]
            # Check if variant changes hmer size and is same type as record
            if size != ref_hmer_size:
                var_type = _get_variant_type(len(var.ref), len(var.alts[0]))
                if var_type == rec_type:
                    logger.debug(
                        f"Found conflicting {var_type} at {var.contig}:{var.pos} "
                        f"({var.ref}→{var.alts[0]}) with QUAL={var.qual} > {rec.qual}"
                    )
                    return True
    return False


def _check_germline_variants(
    rec, ref_fasta: object, chrom: str, pos: int, ref_hmer_size: int, tumor_germline_handle, rec_type: str
) -> bool:
    """Check for conflicting variants in germline VCF with PASS filter.

    Args:
        rec: VCF record
        ref_fasta: Reference FASTA
        chrom: Chromosome
        pos: Position
        ref_hmer_size: Reference homopolymer size
        tumor_germline_handle: Tumor germline VCF
        rec_type: Variant type (insertion/deletion/snp)

    Returns:
        True if conflicting variant found, False otherwise
    """
    if not tumor_germline_handle:
        return False

    germline_tumor_vars = tumor_germline_handle.fetch(chrom, rec.pos - 2, rec.pos + rec.info["X_HIL"][0] + 4)
    for var in germline_tumor_vars:
        if is_pass(var):
            # Skip if it's the same variant as the somatic record
            if var.ref == rec.ref and var.alts == rec.alts and var.pos == rec.pos:
                continue
            size = apply_variant(ref_fasta, [chrom, pos], [var.pos, var.ref, var.alts[0]])[0]
            # Check if variant changes hmer size and is same type as record
            if size != ref_hmer_size:
                var_type = _get_variant_type(len(var.ref), len(var.alts[0]))
                if var_type == rec_type:
                    logger.debug(
                        f"Found conflicting germline {var_type} at {var.contig}:{var.pos} "
                        f"({var.ref}→{var.alts[0]}) matching record type"
                    )
                    return True
    return False


def _check_other_variants(variant_context: dict, config: dict) -> int:
    """Check if record has conflicting variants in the region.

    Detects if there are alternative variants (somatic with higher QUAL or germline with PASS)
    that change the hmer size AND are of the same type (insertion/deletion) as the record.

    Args:
        variant_context: Dictionary with keys:
            - rec: VCF record
            - ref_fasta: Reference FASTA
            - chrom: Chromosome
            - pos: Position
            - ref_hmer_size: Reference homopolymer size
            - vcf_file: Input VCF file for fetching variants
            - tumor_germline_handle: Tumor germline VCF (optional)
        config: Configuration dictionary

    Returns:
        1 if other conflicting variants found, 0 otherwise
    """
    rec = variant_context["rec"]
    ref_fasta = variant_context["ref_fasta"]
    chrom = variant_context["chrom"]
    pos = variant_context["pos"]
    ref_hmer_size = variant_context["ref_hmer_size"]
    vcf_file = variant_context["vcf_file"]
    tumor_germline_handle = variant_context["tumor_germline_handle"]

    # Determine the variant type of the current record
    rec_type = _get_variant_type(len(rec.ref), len(rec.alts[0]))

    # Check for conflicts in somatic or germline VCFs
    if _check_somatic_variants(rec, ref_fasta, chrom, pos, ref_hmer_size, vcf_file, rec_type):
        return 1

    if _check_germline_variants(rec, ref_fasta, chrom, pos, ref_hmer_size, tumor_germline_handle, rec_type):
        return 1

    return 0


def _collect_tumor_pileup_data(tumor_reads, chrom: str, pos: int):
    """Collect pileup data from tumor reads and determine nucleotide.

    Args:
        tumor_reads: Tumor pysam AlignmentFile
        chrom: Chromosome
        pos: Position

    Returns:
        Tuple of (tumor_read_data, nuc) or None if no pileup data
    """
    p = tumor_reads.pileup(
        chrom,
        pos - 1,
        pos,
        truncate=True,
        min_base_quality=READS_QUALITY_THRESHOLD,
        min_mapping_quality=MIN_MAPPING_QUALITY,
    )
    pileup_list = list(p)
    if not pileup_list:
        return None, None
    tumor_pileup = pileup_list[0]
    results = [get_hmer_qualities_from_pileup_element(x) for x in tumor_pileup.pileups]

    nuc = get_max_nuc([x[0] for x in results])
    return results, nuc


def _collect_normal_pileup_data(normal_reads, chrom: str, pos: int):
    """Collect pileup data from normal reads.

    Args:
        normal_reads: Normal pysam AlignmentFile
        chrom: Chromosome
        pos: Position

    Returns:
        List of normal read data or None if no pileup data
    """
    p = normal_reads.pileup(
        chrom,
        pos - 1,
        pos,
        truncate=True,
        min_base_quality=READS_QUALITY_THRESHOLD,
        min_mapping_quality=MIN_MAPPING_QUALITY,
    )
    pileup_list = list(p)
    if not pileup_list:
        return None

    normal_pileup = pileup_list[0]
    return [get_hmer_qualities_from_pileup_element(x) for x in normal_pileup.pileups]


def get_results_per_normal_sort_value(direction_results: dict, mixture_bound: float) -> float:
    """Calculate sorting value for normal results based on tot_score.

    For each allele, calculate mixture as average of fw_mixture and bw_mixture.
    If mixture is below mixture_bound, set tot_score to 0.
    Return the maximum tot_score across all alleles.

    Args:
        direction_results: Dictionary of results for each direction and allele
        mixture_bound: Mixture threshold for scoring

    Returns:
        Maximum tot_score across alleles (adjusted for mixture_bound)
    """
    return max(
        (
            0
            if (direction_results[alt_idx][0]["mixture"] + direction_results[alt_idx][1]["mixture"]) / 2.0
            < mixture_bound
            else min(direction_results[alt_idx][0]["tot_score"], direction_results[alt_idx][1]["tot_score"])
        )
        for alt_idx in range(len(direction_results))
    )


def _process_multiple_normals_median(
    normal_reads_list,
    tumor_read_data,
    nuc,
    chrom: str,
    pos: int,
    ref_fasta,
    rec,
    config: dict,
) -> dict:
    """Process multiple normal files and return median result by tot_score.

    Args:
        normal_reads_list: List of normal sample reads file paths
        tumor_read_data: Pre-computed tumor pileup read data
        nuc: Nucleotide determined from tumor reads
        chrom: Chromosome
        pos: Position
        ref_fasta: Reference FASTA file handle
        rec: VCF record for the variant being processed
        config: Configuration dictionary with parameters

    Returns:
        Dictionary with median results
    """
    results_per_normal = []
    normal_germline_files = config["normal_germline_files"]

    ref_hmer_size = apply_variant(ref_fasta, [chrom, pos], [rec.pos, "", ""])[0]
    for normal_reads, normal_germline_file in zip(normal_reads_list, normal_germline_files, strict=False):
        try:
            normal_germline = pysam.VariantFile(normal_germline_file)

            # Check for conflicting variants with this specific normal_germline
            other_variant = _check_normal_other_variants(rec, ref_fasta, chrom, pos, ref_hmer_size, normal_germline)

            if other_variant:
                logger.debug(f"Skipping normal file {normal_germline_file} due to conflicting variants")
                normal_germline.close()
                continue
            normal_germline.close()

            # Collect pileup data for this normal
            normal_read_data = _collect_normal_pileup_data(normal_reads, chrom, pos)
            if normal_read_data is None:
                continue

            # Check minimum normal coverage threshold
            min_normal_cvg_threshold = config["min_normal_cvg"]
            normal_cvg = len(normal_read_data)
            if min_normal_cvg_threshold > 0 and normal_cvg < min_normal_cvg_threshold:
                logger.debug(
                    f"Skipping normal file {normal_reads} due to low coverage:\n"
                    f"  {normal_cvg} < {min_normal_cvg_threshold}"
                )
                continue

            # Process all alternative alleles for this normal
            direction_results = {}
            for alt_idx, alt in enumerate(rec.alts):
                exp_shift_tries = [len(alt) - len(rec.ref)]
                direction_results[alt_idx] = {}

                # Process both directions
                for direction in range(NUM_DIRECTIONS):
                    result = _process_direction_for_allele(
                        direction,
                        exp_shift_tries,
                        rec.info.get("ref_hmer_size", -1)
                        if "ref_hmer_size" in rec.info
                        else apply_variant(ref_fasta, [chrom, pos], [rec.pos, "", ""])[0],
                        normal_read_data,
                        tumor_read_data,
                        nuc,
                        config["pseudocounts"],
                    )
                    direction_results[alt_idx][direction] = result

                    # Calculate combined score
                    is_ins = len(alt) > len(rec.ref)
                    pval_score = result.get("ttest_score", -1)
                    ttest_score = (
                        pval_score
                        if (is_ins ^ (result.get("tumor_exp", 0) < result.get("normal_exp", 0)))
                        else -pval_score
                    )

                    combined_score = combine_scores(
                        ttest_score,
                        result.get("likely_score", -1),
                        result.get("mixture", 0),
                        result.get("normal_ml_score", -1),
                        result.get("normal_ml_mixture", 0),
                        result.get("tumor_ml_score", -1),
                        result.get("tumor_ml_mixture", 0),
                    )
                    direction_results[alt_idx][direction]["tot_score"] = round(combined_score, 4)

            # Calculate tot_score for this normal
            tot_score = tuple(
                min(
                    direction_results[alt_idx][0]["tot_score"],
                    direction_results[alt_idx][1]["tot_score"],
                )
                for alt_idx in range(len(rec.alts))
            )

            results_per_normal.append((tot_score, direction_results))
        except Exception as e:
            logger.exception(f"Error processing normal file {normal_germline_file}: {e}")
            continue

    if not results_per_normal:
        return None

    # Sort by tot_score (use maximum score across alleles after mixture filtering)
    results_per_normal.sort(key=lambda x: get_results_per_normal_sort_value(x[1], config["mixture_bound"]))

    # Select median: for even count, select higher-scoring of the two middle elements
    num_normals = len(results_per_normal)
    median_idx = num_normals // 2

    return results_per_normal[median_idx][1]


def _validate_normal_data(ref_hmer_size: int, exp_shift_tries, normal_main_hmer: int) -> bool:
    """Validate normal read data parameters.

    Args:
        ref_hmer_size: Reference homopolymer size
        exp_shift_tries: List of possible hmer size differences
        normal_main_hmer: Best hmer size from normal reads

    Returns:
        True if data is valid, False otherwise
    """
    return not (
        ref_hmer_size + exp_shift_tries[0] > HMER_NUM_POSSIBLE_SIZES
        or ref_hmer_size + exp_shift_tries[0] < 0
        or ref_hmer_size > HMER_NUM_POSSIBLE_SIZES
        or normal_main_hmer != ref_hmer_size
    )


def _process_reads_for_direction(
    normal_read_data,
    tumor_read_data,
    nuc: str,
    direction: int,
    ref_hmer_size: int,
    exp_shift_tries,
) -> dict:
    """Filter and process normal and tumor reads for a direction.

    Args:
        normal_read_data: Normal pileup read data
        tumor_read_data: Tumor pileup read data
        nuc: Nucleotide
        direction: Direction index (0=forward, 1=backward)
        ref_hmer_size: Reference homopolymer size
        exp_shift_tries: List of possible hmer size differences

    Returns:
        Dictionary with processed read data or None if validation fails
    """
    # Process normal reads
    filtered_normal_reads = filter_reads(normal_read_data, nuc, direction)
    filtered_normal_results = process_reads(filtered_normal_reads)
    normal_cvg = len(filtered_normal_results)
    normal_main_hmer, normal_second_hmer = find_best_mixture_hmer_sizes(filtered_normal_reads)

    # Validate normal data
    if not _validate_normal_data(ref_hmer_size, exp_shift_tries, normal_main_hmer):
        return None

    # Get normal ML score
    normal_ml_score, normal_ml_mixture = get_machine_likelihood(
        filtered_normal_reads, ref_hmer_size, ref_hmer_size + exp_shift_tries[0]
    )

    # Process tumor reads
    filtered_tumor_reads = filter_reads(tumor_read_data, nuc, direction)
    filtered_tumor_results = process_reads(filtered_tumor_reads)
    tumor_cvg = len(filtered_tumor_results)
    tumor_main_hmer, tumor_second_hmer = find_best_mixture_hmer_sizes(filtered_tumor_reads)
    tumor_ml_score, tumor_ml_mixture = get_machine_likelihood(
        filtered_tumor_reads, ref_hmer_size, ref_hmer_size + exp_shift_tries[0]
    )

    if tumor_cvg == 0:
        return None

    return {
        "filtered_normal_reads": filtered_normal_reads,
        "filtered_normal_results": filtered_normal_results,
        "normal_cvg": normal_cvg,
        "normal_main_hmer": normal_main_hmer,
        "normal_second_hmer": normal_second_hmer,
        "normal_ml_score": normal_ml_score,
        "normal_ml_mixture": normal_ml_mixture,
        "filtered_tumor_reads": filtered_tumor_reads,
        "filtered_tumor_results": filtered_tumor_results,
        "tumor_cvg": tumor_cvg,
        "tumor_main_hmer": tumor_main_hmer,
        "tumor_second_hmer": tumor_second_hmer,
        "tumor_ml_score": tumor_ml_score,
        "tumor_ml_mixture": tumor_ml_mixture,
    }


def _calculate_em_parameters(
    filtered_normal_results,
    exp_shift_tries,
    pseudocounts: float,
) -> dict:
    """Calculate EM parameters and cell counts from filtered normal results.

    Args:
        filtered_normal_results: Processed normal read data
        exp_shift_tries: List of possible hmer size differences
        pseudocounts: Prior count for EM algorithm

    Returns:
        Dictionary with EM parameters and cell counts
    """
    params = calc_parameters(filtered_normal_results)
    cell_shift = 2 * len(params[0])
    num_cells = cell_shift * 21
    original_counts = [pseudocounts] * num_cells
    extra_counts = [[pseudocounts] * num_cells for _ in exp_shift_tries]

    exps = []
    groups = []
    weights = []

    # Process normal reads
    for read in filtered_normal_results:
        exps.append(read[0])
        groups.append(0)
        weights.append(read[1])
        cell = get_cell(read, params, cell_shift)
        if 0 <= cell < num_cells:
            original_counts[cell] += 1
        for x in range(len(exp_shift_tries)):
            curr_cell = cell + cell_shift * exp_shift_tries[x]
            if 0 <= curr_cell < num_cells:
                extra_counts[x][curr_cell] += 1

    return {
        "params": params,
        "cell_shift": cell_shift,
        "num_cells": num_cells,
        "original_counts": original_counts,
        "extra_counts": extra_counts,
        "exps": exps,
        "groups": groups,
        "weights": weights,
    }


def _calculate_statistical_scores(
    exps,
    groups,
    weights,
    original_counts,
    extra_counts,
    filtered_tumor_results,
    params,
    cell_shift,
    exp_shift_tries,
) -> dict:
    """Calculate statistical scores and mixture parameters using EM algorithm.

    Args:
        exps: List of expectation values from normal reads
        groups: Group labels (0=normal, 1=tumor)
        weights: Weights for each read
        original_counts: Cell counts for H0 hypothesis
        extra_counts: Cell counts for H1 hypotheses
        filtered_tumor_results: Processed tumor read data
        params: EM parameters
        cell_shift: Cell shift value
        exp_shift_tries: List of possible hmer size differences

    Returns:
        Dictionary with calculated scores
    """
    normal_exp = sum(exps) / len(exps) if exps else 0

    # Calculate original likelihood for H0
    sum_counts = sum(original_counts)
    original_probs = [x / sum_counts for x in original_counts]
    extra_probs = [[x / sum(count) for x in count] for count in extra_counts]

    original_likelihood = 0
    h0_probs = []
    h1_probs = [[] for _ in range(len(exp_shift_tries))]

    # Process tumor reads
    tot_exp = 0
    for read in filtered_tumor_results:
        tot_exp += read[0]
        exps.append(read[0])
        groups.append(1)
        weights.append(read[1])
        cell = get_cell(read, params, cell_shift)
        h0_probs.append(original_probs[cell])
        original_likelihood += log(original_probs[cell])
        for i in range(len(exp_shift_tries)):
            h1_probs[i].append(extra_probs[i][cell])

    tumor_exp = tot_exp / len(filtered_tumor_results) if filtered_tumor_results else 0

    # Calculate statistical scores
    ttest_pval = max(get_pval(exps, groups, weights), LOG_MIN_PVAL_BASE**2)
    h0_probs = np.asarray(h0_probs, dtype=float)

    # EM algorithm for mixture parameter
    h1_mixtures = []
    for i in range(len(exp_shift_tries)):
        try:
            h1_probs[i] = np.asarray(h1_probs[i], dtype=float)
            pi = EM_INITIAL_PI
            for _ in range(EM_ITERATIONS):
                denom = pi * h1_probs[i] + (1.0 - pi) * h0_probs
                r = (pi * h1_probs[i]) / denom
                pi = min(max(float(r.mean()), PI_MIN), PI_MAX)
            h1_mixtures.append(pi)
        except Exception:
            h1_mixtures.append(0.002)

    # Calculate likelihoods
    extra_likelihoods = []
    for i in range(len(exp_shift_tries)):
        likely = sum(
            log((1 - h1_mixtures[i]) * h0_probs[j] + h1_mixtures[i] * h1_probs[i][j]) for j in range(len(h0_probs))
        )
        extra_likelihoods.append(likely)

    extra_score = max(extra_likelihoods)
    extra_shift = extra_likelihoods.index(extra_score)
    mixture = h1_mixtures[extra_shift]

    # Compute combined score from ttest and likelihood
    tmp_score = (
        -SCORE_ADJUSTMENT_FACTOR
        * max(original_likelihood - extra_score + log(len(exp_shift_tries)), MAX_LOG_LIKELIHOOD)
    ) / log(LIKELIHOOD_LOG_BASE)
    tmp_score = min(MAX_COMBINED_SCORE, tmp_score)
    pval_score = -log(ttest_pval) / log(LIKELIHOOD_LOG_BASE)
    likely_score = pval_score + (tmp_score if tmp_score > 0 else 0)

    return {
        "normal_exp": normal_exp,
        "tumor_exp": tumor_exp,
        "pval_score": pval_score,
        "likely_score": likely_score,
        "mixture": mixture,
        "extra_shift": extra_shift,
    }


def _process_direction_for_allele(
    direction: int,
    exp_shift_tries,
    ref_hmer_size: int,
    normal_read_data,
    tumor_read_data,
    nuc: str,
    pseudocounts: float,
) -> dict:
    """Process a single direction for an allele.

    Args:
        direction: Direction index (0=forward, 1=backward)
        exp_shift_tries: List of possible hmer size differences
        ref_hmer_size: Reference homopolymer size
        normal_read_data: Normal pileup read data
        tumor_read_data: Tumor pileup read data
        nuc: Nucleotide
        pseudocounts: Prior count for EM algorithm

    Returns:
        Dict with direction results or error dict with -1 values
    """
    try:
        # Process and validate reads
        read_data = _process_reads_for_direction(
            normal_read_data,
            tumor_read_data,
            nuc,
            direction,
            ref_hmer_size,
            exp_shift_tries,
        )
        if read_data is None:
            return fill_direction_results_with_error()

        # Calculate EM parameters
        em_params = _calculate_em_parameters(
            read_data["filtered_normal_results"],
            exp_shift_tries,
            pseudocounts,
        )

        # Calculate statistical scores
        scores = _calculate_statistical_scores(
            em_params["exps"],
            em_params["groups"],
            em_params["weights"],
            em_params["original_counts"],
            em_params["extra_counts"],
            read_data["filtered_tumor_results"],
            em_params["params"],
            em_params["cell_shift"],
            exp_shift_tries,
        )

        return {
            "normal_exp": round(scores["normal_exp"], 4),
            "tumor_exp": round(scores["tumor_exp"], 4),
            "normal_cvg": read_data["normal_cvg"],
            "tumor_cvg": read_data["tumor_cvg"],
            "ttest_score": round(scores["pval_score"], 4),
            "likely_score": round(scores["likely_score"], 4),
            "mixture": round(scores["mixture"], 4),
            "normal_main_hmer": read_data["normal_main_hmer"],
            "tumor_main_hmer": read_data["tumor_main_hmer"],
            "normal_second_hmer": read_data["normal_second_hmer"],
            "tumor_second_hmer": read_data["tumor_second_hmer"],
            "normal_ml_score": round(read_data["normal_ml_score"], 4),
            "normal_ml_mixture": round(read_data["normal_ml_mixture"], 4),
            "tumor_ml_score": round(read_data["tumor_ml_score"], 4),
            "tumor_ml_mixture": round(read_data["tumor_ml_mixture"], 4),
        }
    except Exception as e:
        logger.exception(f"Error processing direction {direction}: {e}")
        return fill_direction_results_with_error()


def _should_skip_record(rec, min_hmer: int, max_hmer: int) -> bool:
    """Check if VCF record should be skipped from processing.

    Args:
        rec: VCF record
        min_hmer: Minimum homopolymer size threshold
        max_hmer: Maximum homopolymer size threshold

    Returns:
        True if record should be skipped, False otherwise
    """
    # Check for required fields; skip if missing
    if "VARIANT_TYPE" not in rec.info or "X_HIL" not in rec.info:
        logger.debug(f"Record at {rec.contig}:{rec.pos} missing required fields (VARIANT_TYPE or X_HIL). Skipping.")
        return True

    hmer_size = rec.info["X_HIL"][0]
    return rec.info["VARIANT_TYPE"] != "h-indel" or hmer_size < min_hmer or hmer_size > max_hmer or rec.qual == 0


def _write_results_to_record(
    rec,
    all_direction_results: dict,
    *,
    verbose: bool,
    score_bound: float,
    mixture_bound: float,
) -> None:
    """Write processing results to VCF record and update filters.

    Args:
        rec: VCF record to update
        all_direction_results: Dictionary with results for all directions and alleles
        verbose: If True, write all fields; if False, write only mixture and tot_score
        score_bound: Score threshold for PASS filter
        mixture_bound: Mixture threshold for PASS filter
    """
    prefixes = DIRECTION_PREFIXES

    if verbose:
        # Verbose mode: Write all fields for all alleles and directions
        for direction in range(NUM_DIRECTIONS):
            pre = prefixes[direction]

            # Collect tuples from all alleles
            for field in [
                "normal_exp",
                "tumor_exp",
                "normal_cvg",
                "tumor_cvg",
                "ttest_score",
                "likely_score",
                "mixture",
                "normal_main_hmer",
                "tumor_main_hmer",
                "normal_second_hmer",
                "tumor_second_hmer",
                "normal_ml_score",
                "normal_ml_mixture",
                "tumor_ml_score",
                "tumor_ml_mixture",
                "tot_score",
            ]:
                values = tuple(all_direction_results[alt_idx][direction][field] for alt_idx in range(len(rec.alts)))
                rec.info[pre + field] = values

    # Always write tot_score as minimum across directions
    tot_score_values = tuple(
        min(all_direction_results[alt_idx][d]["tot_score"] for d in range(NUM_DIRECTIONS))
        for alt_idx in range(len(rec.alts))
    )
    rec.info["tot_score"] = tot_score_values

    # Always write mixture as average across directions
    mixture_values = tuple(
        sum(all_direction_results[alt_idx][d]["mixture"] for d in range(NUM_DIRECTIONS)) / NUM_DIRECTIONS
        for alt_idx in range(len(rec.alts))
    )
    rec.info["mixture"] = mixture_values

    # Check if all alleles pass the score and mixture bounds
    all_pass = all(
        tot_score_values[alt_idx] > score_bound and mixture_values[alt_idx] > mixture_bound
        for alt_idx in range(len(rec.alts))
    )

    # Set PASS filter if all alleles pass bounds
    if all_pass:
        rec.filter.clear()  # Clear any existing filters
        rec.filter.add("PASS")


def _process_record(
    rec,
    ref_fasta,
    vcf_file,
    normal_reads,
    tumor_germline_handle,
    tumor_reads,
    merged_intervals,
    config: dict,
    tumor_read_data=None,
) -> dict:
    """Process a single VCF record and return results.

    Args:
        rec: VCF record to process
        ref_fasta: Reference FASTA file
        vcf_file: Input VCF file for variant checking
        normal_reads: List of normal sample reads file paths
        tumor_germline_handle: Tumor germline VCF file handle (optional)
        tumor_reads: Tumor pysam AlignmentFile
        merged_intervals: Merged BED intervals or None
        config: Configuration dictionary with all parameters
        tumor_read_data: Pre-collected tumor pileup data (optional)

    Returns:
        Dictionary with result or None if record should not be processed
    """
    # Validate required fields
    if "X_HIL" not in rec.info:
        logger.debug(f"Record at {rec.contig}:{rec.pos} missing X_HIL field. Skipping processing.")
        return None

    chrom = rec.contig
    pos = rec.pos + rec.info["X_HIL"][0] // 2
    ref_hmer_size = apply_variant(ref_fasta, [chrom, pos], [rec.pos, "", ""])[0]

    if config["verbose"]:
        rec.info["ref_hmer_size"] = ref_hmer_size

    # Check target intervals
    if merged_intervals:
        if not pos_in_bed(merged_intervals, chrom, pos):
            return None

    # Check for conflicting variants
    variant_context = {
        "rec": rec,
        "ref_fasta": ref_fasta,
        "chrom": chrom,
        "pos": pos,
        "ref_hmer_size": ref_hmer_size,
        "vcf_file": vcf_file,
        "tumor_germline_handle": tumor_germline_handle,
    }
    other_variant = _check_other_variants(variant_context, config)
    if config["verbose"]:
        rec.info["other_variant"] = other_variant
    if other_variant:
        return None

    # Collect tumor pileup data if not already provided
    if tumor_read_data is None:
        pileup_result = _collect_tumor_pileup_data(tumor_reads, chrom, pos)
        if pileup_result is None:
            return None
        tumor_read_data, nuc = pileup_result
    else:
        # Extract nucleotide from pre-collected tumor data
        nuc = get_max_nuc([x[0] for x in tumor_read_data])

    # Process multiple normals and select median by tot_score
    all_direction_results = _process_multiple_normals_median(
        normal_reads,
        tumor_read_data,
        nuc,
        chrom,
        pos,
        ref_fasta,
        rec,
        config,
    )

    return all_direction_results


def _validate_and_collect_tumor_data(rec, tumor_reads, config: dict) -> tuple | None:
    """Validate and collect tumor pileup data for a record.

    Collects tumor pileup data and checks against minimum coverage threshold.

    Args:
        rec: VCF record to process
        tumor_reads: Tumor pysam AlignmentFile
        config: Configuration dictionary with min_tumor_cvg parameter

    Returns:
        Tuple of (tumor_read_data, nuc) if valid, None if record should be skipped
    """
    # Calculate position for pileup
    chrom = rec.contig
    pos = rec.pos + rec.info["X_HIL"][0] // 2

    # Collect tumor pileup data
    tumor_pileup_data = _collect_tumor_pileup_data(tumor_reads, chrom, pos)
    if tumor_pileup_data is None:
        return None

    tumor_read_data, nuc = tumor_pileup_data

    # Check minimum tumor coverage threshold
    min_tumor_cvg_threshold = config["min_tumor_cvg"]
    tumor_cvg = len(tumor_read_data)
    if min_tumor_cvg_threshold > 0 and tumor_cvg < min_tumor_cvg_threshold:
        return None

    return tumor_read_data, nuc


def _process_vcf_records(
    vcf_file_handle,
    ref_fasta,
    normal_reads,
    tumor_germline_handle,
    tumor_reads,
    vcf_out_file_handle,
    merged_intervals,
    config: dict,
    vcf_handle_for_queries=None,
) -> None:
    """Process all VCF records and write results.

    Args:
        vcf_file_handle: Input VCF file handle
        ref_fasta: Reference FASTA file
        normal_reads: List of normal sample reads files
        tumor_germline_handle: Tumor germline VCF file handle (optional)
        tumor_reads: Tumor alignment file
        vcf_out_file_handle: Output VCF file handle
        merged_intervals: Merged BED intervals or None
        config: Configuration dictionary with all parameters
        vcf_handle_for_queries: Separate VCF handle for nested queries (optional)
    """
    for rec in vcf_file_handle.fetch():
        try:
            # Check if record should be skipped
            if _should_skip_record(rec, config["min_hmer"], config["max_hmer"]):
                vcf_out_file_handle.write(rec)
                continue

            # Validate and collect tumor data (includes coverage check)
            tumor_data = _validate_and_collect_tumor_data(rec, tumor_reads, config)
            if tumor_data is None:
                vcf_out_file_handle.write(rec)
                continue

            tumor_read_data, nuc = tumor_data

            # Process record with pre-collected tumor data
            all_direction_results = _process_record(
                rec,
                ref_fasta,
                vcf_handle_for_queries or vcf_file_handle,
                normal_reads,
                tumor_germline_handle,
                tumor_reads,
                merged_intervals,
                config,
                tumor_read_data=tumor_read_data,
            )

            if all_direction_results is None or len(all_direction_results) == 0:
                vcf_out_file_handle.write(rec)
                continue
            _write_results_to_record(
                rec,
                all_direction_results,
                verbose=config["verbose"],
                score_bound=config["score_bound"],
                mixture_bound=config["mixture_bound"],
            )

            # Write record after processing all alleles
            vcf_out_file_handle.write(rec)

        except Exception as e:
            # If processing fails, log error and write record as-is to ensure no data loss
            logger.warning(
                f"Error processing VCF record at {rec.contig}:{rec.pos}: {type(e).__name__}: {str(e)}. "
                f"Writing record without analysis results."
            )
            vcf_out_file_handle.write(rec)


def _close_files(
    vcf_handle, vcf_out_handle, tumor_reads_handle, normal_reads_handles=None, vcf_handle_for_queries=None
):
    """Close all open file handles.

    Args:
        vcf_handle: Input VCF file handle
        vcf_out_handle: Output VCF file handle
        tumor_reads_handle: Tumor reads BAM file handle
        normal_reads_handles: List of normal reads BAM file handles (optional)
        vcf_handle_for_queries: Separate VCF handle for nested queries (optional)
    """
    vcf_handle.close()
    vcf_out_handle.close()
    tumor_reads_handle.close()
    if normal_reads_handles:
        for handle in normal_reads_handles:
            if handle:
                handle.close()
    if vcf_handle_for_queries:
        vcf_handle_for_queries.close()


def _initialize_files(
    vcf_file, vcf_out_file, tumor_reads_file, normal_reads_files, ref_fasta_path, tumor_germline_file, verbose
):
    """Initialize and open all required file handles.

    Args:
        vcf_file: Input VCF file path
        vcf_out_file: Output VCF file path
        tumor_reads_file: Tumor reads BAM file path
        normal_reads_files: List of BAM files paths
        ref_fasta_path: Reference FASTA file path
        tumor_germline_file: Tumor germline VCF file path
        verbose: Verbose flag for VCF header setup

    Returns:
        Tuple of file handles: (vcf_handle, vcf_out_handle, ref_fasta, tumor_reads_handle,
                               normal_reads_handles, tumor_germline_handle, vcf_handle_for_queries)
    """
    # Setup VCF headers using helper function
    vcf_handle = _setup_vcf_headers(vcf_file, verbose)

    # Open a separate VCF handle for nested queries to avoid iterator corruption
    # (pysam does not support nested fetch() on same handle - see pysam issue #827)
    vcf_handle_for_queries = pysam.VariantFile(vcf_file)

    # Initialize tumor reads file
    tumor_reads_handle = pysam.AlignmentFile(tumor_reads_file, reference_filename=ref_fasta_path)
    normal_reads_handles = [
        pysam.AlignmentFile(normal_reads_file, reference_filename=ref_fasta_path)
        for normal_reads_file in normal_reads_files
    ]
    # Load reference FASTA
    ref_fasta = Fasta(ref_fasta_path)

    # Open output VCF file
    vcf_out_handle = pysam.VariantFile(vcf_out_file, "wz", header=vcf_handle.header.copy())

    if tumor_germline_file:
        tumor_germline_handle = pysam.VariantFile(tumor_germline_file)
    else:
        tumor_germline_handle = None

    return (
        vcf_handle,
        vcf_out_handle,
        ref_fasta,
        tumor_reads_handle,
        normal_reads_handles,
        tumor_germline_handle,
        vcf_handle_for_queries,
    )


def variant_calling(config):
    """Process VCF records and add hmer-based scoring.

    Main entry point for variant calling pipeline that processes VCF records
    with pileup data from normal and tumor samples. Supports multiple normal
    samples by accepting comma-separated file paths.

    Args:
        config: Configuration dictionary with keys:
            vcf_file: Input VCF file path
            normal_reads_files: Comma-separated normal sample reads file paths or single path
            tumor_reads_file: Tumor sample reads file path
            vcf_out_file: Output VCF file path
            min_hmer: Minimum homopolymer size threshold
            max_hmer: Maximum homopolymer size threshold
            min_tumor_cvg: Minimum tumor coverage threshold
            pseudocounts: Prior count for EM algorithm
            target_intervals_bed_file: Optional BED file for interval filtering
            tumor_germline_file: Optional tumor germline VCF file
            normal_germline_file: Comma-separated normal germline VCF file paths or single path
            normal_germline_files: Pre-parsed list of normal germline file paths
            verbose: If True, write all fields; if False, only write mixture and tot_score
            score_bound: Score threshold for PASS filter (default: 2.0)
            mixture_bound: Mixture threshold for PASS filter (default: 0.01)
            ref_fasta_path: Path to reference FASTA file
    """
    # Extract configuration parameters
    vcf_file = config["vcf_file"]
    normal_reads_files_list = config["normal_reads_files_list"]
    tumor_reads_file = config["tumor_reads_file"]
    vcf_out_file = config["vcf_out_file"]
    ref_fasta_path = config["ref_fasta_path"]
    tumor_germline_file = config["tumor_germline_file"]
    verbose = config["verbose"]
    target_intervals_bed_file = config["target_intervals_bed_file"]

    # Initialize and open files
    (
        vcf_handle,
        vcf_out_handle,
        ref_fasta,
        tumor_reads,
        normal_reads,
        tumor_germline_handle,
        vcf_handle_for_queries,
    ) = _initialize_files(
        vcf_file, vcf_out_file, tumor_reads_file, normal_reads_files_list, ref_fasta_path, tumor_germline_file, verbose
    )

    # Load BED intervals if provided
    merged_intervals = None
    if target_intervals_bed_file:
        merged_intervals = load_bed_intervals(target_intervals_bed_file)

    try:
        # Process VCF records
        _process_vcf_records(
            vcf_handle,
            ref_fasta,
            normal_reads,
            tumor_germline_handle,
            tumor_reads,
            vcf_out_handle,
            merged_intervals,
            config,
            vcf_handle_for_queries,
        )
    finally:
        # Close all file handles
        _close_files(vcf_handle, vcf_out_handle, tumor_reads, normal_reads, vcf_handle_for_queries)


def main() -> None:
    """Parse command-line arguments and run variant calling pipeline.

    This function creates an argument parser for the VCF hmer update
    tool and calls the variant_calling function with the parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Update VCF records with hmer-based variant calling " "scores and quality metrics.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Example usage:\n"
        "  python vcf_hmer_update.py input.vcf normal_pileup.txt "
        "tumor_pileup.txt output.vcf --min-hmer 4 --pseudocounts 0.5",
    )

    # Required positional arguments
    parser.add_argument(
        "input_vcf",
        help="Input VCF file path",
    )
    parser.add_argument(
        "normal_reads_files",
        help="Comma-separated normal sample reads file paths (or single path)",
    )
    parser.add_argument(
        "tumor_reads_file",
        help="Tumor sample reads file path",
    )
    parser.add_argument(
        "output_vcf",
        help="Output VCF file path",
    )

    # Optional arguments
    parser.add_argument(
        "--min_hmer",
        type=int,
        default=4,
        help="Minimum homopolymer size (default: 6)",
    )
    parser.add_argument(
        "--max_hmer",
        type=int,
        default=20,
        help="Maximum homopolymer size (default: 20)",
    )
    parser.add_argument(
        "--min_tumor_cvg",
        type=int,
        default=0,
        help="Minimum tumor coverage threshold (default: 0)",
    )
    parser.add_argument(
        "--min_normal_cvg",
        type=int,
        default=0,
        help="Minimum normal coverage threshold; normals below this are skipped (default: 0)",
    )
    parser.add_argument(
        "--pseudocounts",
        type=float,
        default=0.5,
        help="Pseudocounts prior for EM algorithm (default: 0.5)",
    )
    parser.add_argument(
        "--ref_fasta",
        type=str,
        default="/data/Runs/genomes/hg38/Homo_sapiens_assembly38.fasta",
        help="Path to reference FASTA file (default: hg38 path)",
    )

    parser.add_argument(
        "--bed_file",
        type=str,
        default=None,
        help="BED file with intervals to process",
    )
    parser.add_argument(
        "--tumor_germline",
        type=str,
        default=None,
        help="Tumor germline VCF file path",
    )
    parser.add_argument(
        "--normal_germline",
        type=str,
        default="/data/Runs/cloud_sync/s3/"
        "genomics-pipeline-concordanz-us-east-1/test/germline/"
        "417309-TN20-126568-Z0280-CGCACAATGCGAGAT.vcf.gz",
        help="Comma-separated normal germline VCF file paths or single path (default: provided path)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Write all fields to VCF output; if not set, only write mixture and tot_score",
    )
    parser.add_argument(
        "--score_bound",
        type=float,
        default=2.0,
        help="Score bound threshold (default: 2.0); "
        "records above this score with mixture above mixture_bound are marked PASS",
    )
    parser.add_argument(
        "--mixture_bound",
        type=float,
        default=0.01,
        help="Mixture bound threshold (default: 0.01); "
        "records above this mixture with score above score_bound are marked PASS",
    )

    args = parser.parse_args()

    # Parse comma-separated file paths
    normal_reads_files_list = [f.strip() for f in args.normal_reads_files.split(",")]
    if args.normal_germline:
        normal_germline_files_list = [f.strip() for f in args.normal_germline.split(",")]
        # Validate that both lists have the same size
        if len(normal_reads_files_list) != len(normal_germline_files_list):
            raise ValueError(
                f"Number of normal_reads_file ({len(normal_reads_files_list)}) must match "
                f"number of normal_germline_file ({len(normal_germline_files_list)})"
            )
    else:
        normal_germline_files_list = [None] * len(normal_reads_files_list)

    # Create config dictionary with all parameters
    config = {
        "vcf_file": args.input_vcf,
        "normal_reads_files": args.normal_reads_files,
        "normal_reads_files_list": normal_reads_files_list,
        "tumor_reads_file": args.tumor_reads_file,
        "vcf_out_file": args.output_vcf,
        "min_hmer": args.min_hmer,
        "max_hmer": args.max_hmer,
        "min_tumor_cvg": args.min_tumor_cvg,
        "min_normal_cvg": args.min_normal_cvg,
        "pseudocounts": args.pseudocounts,
        "target_intervals_bed_file": args.bed_file,
        "tumor_germline_file": args.tumor_germline,
        "normal_germline_file": args.normal_germline,
        "verbose": args.verbose,
        "score_bound": args.score_bound,
        "mixture_bound": args.mixture_bound,
        "ref_fasta_path": args.ref_fasta,
        "normal_germline_files": normal_germline_files_list,
    }

    variant_calling(config)


if __name__ == "__main__":
    main()
