import argparse
from bisect import bisect_right
from math import log

import numpy as np
import pysam
import statsmodels.api as sm
import ugbio_core.flow_format.flow_based_read as fbr
import ugbio_core.math_utils as phred
from pyfaidx import Fasta

# Global constants
BAM_CSOFT_CLIP = 4
MAX_HMER_SIZE = 21
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


def apply_variant(ref_fasta, ref_pos, variant):
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


def load_bed_intervals(bed_path: str):
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


def pos_in_bed(merged_intervals, chrom: str, pos0: int) -> bool:
    """pos0 is 0-based coordinate; BED intervals are [start,end)."""
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


def calc_exp(frequencies):
    """Calculate weighted expectation from frequency distribution."""
    return sum(i * freq for i, freq in enumerate(frequencies))


def find_best_mixture_hmer_sizes(results):
    hmer_size_to_likely = [[0, i] for i in range(MAX_HMER_SIZE)]
    for result in results:
        size_prob = result[1]
        for size in range(MAX_HMER_SIZE):
            hmer_size_to_likely[size][0] -= log(1 - size_prob[size])
    hmer_size_to_likely.sort()
    return hmer_size_to_likely[-1][1], hmer_size_to_likely[-2][1]


def get_max_nuc(nuc_list):
    """Find most common nucleotide in list."""
    if not nuc_list:
        return None
    from collections import Counter

    return Counter(nuc_list).most_common(1)[0][0]


def filter_reads(results, nuc, strand):
    """Filter reads by nucleotide and strand."""
    return [res for res in results if res[0] == nuc and not res[4] and res[5] and bool(res[2]) == bool(strand)]


def process_reads(reads):
    """Process reads to extract expectation and confidence metrics."""
    processed_results = []
    for res in reads:
        expect = calc_exp(res[1])
        sorted_probs = sorted(res[1])
        max_conf = sorted_probs[-1]
        high_conf = sorted_probs[-3]
        processed_results.append((expect, max_conf, high_conf, res[2], res[3]))
    return processed_results


def calc_parameters(reads):
    """Calculate expectation split and confidence parameters from reads."""
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


def get_cell(read, params, cell_shift, exp_move=0):
    """Calculate cell index for read based on parameters."""
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


def get_pval(exps, groups, w):
    """Calculate p-value using weighted least squares regression."""
    y = np.asarray(exps, dtype=float)
    g = np.asarray(groups, dtype=float)  # 0 or 1
    w = np.asarray(w, dtype=float)

    X = sm.add_constant(g)  # intercept + group effect
    res = sm.WLS(y, X, weights=w).fit()
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

    res = sm.WLS(y, X, weights=added_weights_max).fit()
    p_val = min(p_val, float(res.pvalues[1]))
    res = sm.WLS(y, X, weights=added_weights_min).fit()
    p_val = min(p_val, float(res.pvalues[1]))
    return p_val


def get_machine_likelihood(filtered_results, best, second):
    """Calculate machine likelihood ratio and mixture parameter."""
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


def is_pass(rec):
    """Check if variant record passes filters."""
    return "PASS" in rec.filter or len(rec.filter) == 0


def get_safe_pileup(alignment_file, chrom, start, end, **kwargs):
    """Safely get pileup, trying both chr-prefixed and non-prefixed contig names.

    Args:
        alignment_file: pysam AlignmentFile object
        chrom: Chromosome name
        start: Start position
        end: End position
        **kwargs: Additional arguments to pass to pileup()

    Returns:
        Pileup iterator or None if no reads found at position
    """
    try:
        # Try original format first
        return alignment_file.pileup(chrom, start, end, **kwargs)
    except ValueError:
        try:
            # Try with chr prefix if not present
            if not chrom.startswith("chr"):
                return alignment_file.pileup(f"chr{chrom}", start, end, **kwargs)
            # Try without chr prefix if present
            else:
                return alignment_file.pileup(chrom.replace("chr", ""), start, end, **kwargs)
        except ValueError:
            # Return empty pileup-like iterator
            return iter([])


def direction_score(normal_score, normal_mixture, tumor_score, tumor_mixture):
    """Calculate directional score for tumor vs normal comparison."""
    normal_score_clamped = max(min(normal_score, NORMAL_SCORE_MAX), 0)
    normal_modified_mixture = normal_mixture * (normal_score_clamped**SCORE_EXPONENT)
    tumor_modified_mixture = tumor_mixture - normal_modified_mixture
    tumor_mixture_multiplier = max(0, min((max(0, tumor_modified_mixture) ** 2) * TUMOR_MIXTURE_MULTIPLIER_FACTOR, 1))
    tumor_modified_score = max(min(tumor_score, TUMOR_SCORE_MAX), 0) / 2
    return tumor_modified_score * tumor_mixture_multiplier


def combine_scores(
    ttest_score, likely_score, likely_mixture, normal_fw_score, normal_fw_mixture, tumor_fw_score, tumor_fw_mixture
):
    """Combine multiple scoring metrics into single score."""
    score = ttest_score / 4
    likely_mixture_multiplier = min((likely_mixture**2) * TUMOR_MIXTURE_MULTIPLIER_FACTOR, 1)
    score += max(0, likely_score) * likely_mixture_multiplier
    ml_score = direction_score(normal_fw_score, normal_fw_mixture, tumor_fw_score, tumor_fw_mixture)
    score += ml_score
    return score


def fill_direction_results_with_error(direction_results: dict, prefixes: list) -> None:
    """Fill direction_results with -1 for all directions and fields.

    Args:
        direction_results: Dictionary to fill with error values
        prefixes: List of direction prefixes (e.g., ['fw_', 'bw_'])
    """
    fields = [
        "normal_exp",
        "tumor_exp",
        "normal_cvg",
        "tumor_cvg",
        "ins_size",
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
    for direction in range(len(prefixes)):
        direction_results[direction] = {field: -1 for field in fields}


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
        if not isinstance(tp_tag, list | np.ndarray):
            raise ValueError("tp tag must be a list or array")
        tps = tp_tag[qstart:qend]
        for tpval, p in zip(tps, probs, strict=False):
            hmer_probs[tpval + hmer_length] += p
        hmer_probs = np.clip(hmer_probs, filler, None)
        hmer_probs[hmer_length] = 0
        hmer_probs[hmer_length] = max(1 - np.sum(hmer_probs), min_call_prob)
        is_edge = False
    hmer_probs /= np.sum(hmer_probs)
    return hnuc, hmer_probs, not pe.alignment.is_reverse, cycle, is_edge, not pe.alignment.is_duplicate


def variant_calling(
    vcf_file,
    normal_reads_file,
    tumor_reads_file,
    vcf_out_file,
    min_hmer=4,
    zamir=0.5,
    target_intervals_bed_file=None,
    tumor_germline_file=None,
    normal_germline_file=None,
):
    ref_fasta_path = "/data/Runs/genomes/hg38/Homo_sapiens_assembly38.fasta"
    normal_reads = pysam.AlignmentFile(normal_reads_file, check_sq=False, reference_filename=ref_fasta_path)
    tumor_reads = pysam.AlignmentFile(tumor_reads_file, check_sq=False, reference_filename=ref_fasta_path)
    normal_germline = pysam.VariantFile(normal_germline_file)
    tumor_germline = None
    if tumor_germline_file:
        tumor_germline = pysam.VariantFile(tumor_germline_file)
    prefixes = ["fw_", "bw_"]
    f = pysam.VariantFile(vcf_file)
    f.header.info.add("normal_exp", number="A", type="Float", description="Normal hmer size expectancy")
    f.header.info.add("tumor_exp", number="A", type="Float", description="Tumor hmer sizeexpectancy")
    f.header.info.add("normal_cvg", number="A", type="Integer", description="Number of normal reads")
    f.header.info.add("tumor_cvg", number="A", type="Integer", description="Number of tumor reads")
    f.header.info.add("ins_size", number="A", type="Integer", description="Best insertion size")
    f.header.info.add("ttest_score", number="A", type="Float", description="Score from ttest")
    f.header.info.add("likely_score", number="A", type="Float", description="Score from likelyhood")
    f.header.info.add("mixture", number="A", type="Float", description="Best mixture size")
    f.header.info.add(
        "normal_main_hmer",
        number="A",
        type="Integer",
        description="Best hmer of normal according to machine probabilities",
    )
    f.header.info.add(
        "tumor_main_hmer",
        number="A",
        type="Integer",
        description="Best hmer of tumor according to machine probabilities",
    )
    f.header.info.add(
        "normal_second_hmer",
        number="A",
        type="Integer",
        description="Second best hmer of normal according to machine probabilities",
    )
    f.header.info.add(
        "tumor_second_hmer",
        number="A",
        type="Integer",
        description="Second best hmer of tumor according to machine probabilities",
    )
    f.header.info.add("normal_ml_score", number="A", type="Float", description="Normal forward score")
    f.header.info.add("normal_ml_mixture", number="A", type="Float", description="Normal forward mixture")
    f.header.info.add("tumor_ml_score", number="A", type="Float", description="Tumor forward score")
    f.header.info.add("tumor_ml_mixture", number="A", type="Float", description="Tumor forward mixture")

    fields = [
        ("normal_exp", "A", "Float", "Normal hmer size expectancy"),
        ("tumor_exp", "A", "Float", "Tumor hmer size expectancy"),
        ("normal_cvg", "A", "Integer", "Number of normal reads"),
        ("tumor_cvg", "A", "Integer", "Number of tumor reads"),
        ("ins_size", "A", "Integer", "Best insertion size"),
        ("ttest_score", "A", "Float", "Score from ttest"),
        ("likely_score", "A", "Float", "Score from likelihood"),
        ("mixture", "A", "Float", "Best mixture size"),
        ("normal_main_hmer", "A", "Integer", "Best hmer of normal according to machine probabilities"),
        ("tumor_main_hmer", "A", "Integer", "Best hmer of tumor according to machine probabilities"),
        ("normal_second_hmer", "A", "Integer", "Second best hmer of normal according to machine probabilities"),
        ("tumor_second_hmer", "A", "Integer", "Second best hmer of tumor according to machine probabilities"),
        ("normal_ml_score", "A", "Float", "Normal machine likelihood score"),
        ("normal_ml_mixture", "A", "Float", "Normal machine likelihood mixture"),
        ("tumor_ml_score", "A", "Float", "Tumor machine likelihood score"),
        ("tumor_ml_mixture", "A", "Float", "Tumor machine likelihood mixture"),
        ("tot_score", "A", "Float", "combined score for machine likelihoods"),
    ]

    for prefix in prefixes:
        for name, number, typ, desc in fields:
            f.header.info.add(f"{prefix}{name}", number=number, type=typ, description=desc)

    f.header.info.add("ref_hmer_size", number=1, type="Integer", description="Reference hmer size")
    f.header.info.add(
        "tot_score",
        number="A",
        type="Float",
        description="Heuristic combined score of all statistical somatic variant tests",
    )
    f.header.info.add("other_variant", number=1, type="Integer", description="Other variant")

    ref_fasta = Fasta(ref_fasta_path)

    merged_intervals = None
    if target_intervals_bed_file:
        merged_intervals = load_bed_intervals(target_intervals_bed_file)

    fo = pysam.VariantFile(vcf_out_file, "wz", header=f.header.copy())

    for rec in f.fetch():
        if rec.info["VARIANT_TYPE"] != "h-indel" or rec.info["X_HIL"][0] < min_hmer or rec.qual == 0:
            fo.write(rec)
            continue
        chrom = rec.contig
        pos = rec.pos + rec.info["X_HIL"][0] // 2
        ref_hmer_size = apply_variant(ref_fasta, [chrom, pos], [rec.pos, "", ""])[0]
        rec.info["ref_hmer_size"] = ref_hmer_size
        if merged_intervals:
            if not pos_in_bed(merged_intervals, chrom, pos):
                fo.write(rec)
                continue
        my_hmer_size = apply_variant(ref_fasta, [chrom, pos], [rec.pos, rec.ref, rec.alts[0]])[0]
        # filter with same effect variants in the region
        this_loc_variants = f.fetch(chrom, rec.pos - 2, rec.pos + rec.info["X_HIL"][0] + 4)
        other_variant = 0
        for var in this_loc_variants:
            if var.qual > rec.qual:
                size = apply_variant(ref_fasta, [chrom, pos], [var.pos, var.ref, var.alts[0]])[0]
                if size == my_hmer_size or size == -1:
                    other_variant = 1
        germline_normal_vars = normal_germline.fetch(chrom, rec.pos - 2, rec.pos + rec.info["X_HIL"][0] + 4)
        for var in germline_normal_vars:
            if is_pass(var):
                size = apply_variant(ref_fasta, [chrom, pos], [var.pos, var.ref, var.alts[0]])[0]
                if size != ref_hmer_size:
                    other_variant = 1
        if tumor_germline:
            germline_tumor_vars = tumor_germline.fetch(chrom, rec.pos - 2, rec.pos + rec.info["X_HIL"][0] + 4)
            for var in germline_tumor_vars:
                if is_pass(var):
                    size = apply_variant(ref_fasta, [chrom, pos], [var.pos, var.ref, var.alts[0]])[0]
                    if size == ref_hmer_size:
                        if var.ref == rec.ref and var.alts == rec.alts and var.pos == rec.pos:
                            continue
                        other_variant = 1
        rec.info["other_variant"] = other_variant
        if other_variant:
            fo.write(rec)
            continue

        # Normalize contig name for pileup (remove chr prefix if present)
        pileup_chrom = chrom.replace("chr", "") if chrom.startswith("chr") else chrom

        p = get_safe_pileup(
            normal_reads,
            chrom,
            pos - 1,
            pos,
            truncate=True,
            min_base_quality=READS_QUALITY_THRESHOLD,
            min_mapping_quality=MIN_MAPPING_QUALITY,
        )
        pileup_list = list(p)
        if not pileup_list:
            fo.write(rec)
            continue
        pileup = pileup_list[0]
        reads = pileup.pileups
        normal_read_data = [get_hmer_qualities_from_pileup_element(x) for x in reads]
        nuc = get_max_nuc([x[0] for x in normal_read_data])
        p = get_safe_pileup(
            tumor_reads,
            chrom,
            pos - 1,
            pos,
            truncate=True,
            min_base_quality=READS_QUALITY_THRESHOLD,
            min_mapping_quality=MIN_MAPPING_QUALITY,
        )
        pileup_list = list(p)
        if not pileup_list:
            fo.write(rec)
            continue
        pileup = pileup_list[0]
        reads = pileup.pileups
        tumor_read_data = [get_hmer_qualities_from_pileup_element(x) for x in reads]

        # Loop over alternative alleles
        all_direction_results = {}
        for alt_idx, alt in enumerate(rec.alts):
            exp_shift_tries = [len(alt) - len(rec.ref)]

            min_score = 50
            direction_results = {}
            all_direction_results[alt_idx] = direction_results

            for direction in range(2):
                exps = []
                groups = []
                weights = []
                filtered_normal_reads = filter_reads(normal_read_data, nuc, direction)
                filtered_normal_results = process_reads(filtered_normal_reads)
                normal_cvg = len(filtered_normal_results)
                normal_main_hmer, normal_second_hmer = find_best_mixture_hmer_sizes(filtered_normal_reads)
                if (
                    ref_hmer_size + exp_shift_tries[0] > MAX_HMER_SIZE
                    or ref_hmer_size + exp_shift_tries[0] < 0
                    or ref_hmer_size > MAX_HMER_SIZE
                    or normal_main_hmer != ref_hmer_size
                ):
                    min_score = 0
                    fill_direction_results_with_error(direction_results, prefixes)
                    continue
                normal_ml_score, normal_ml_mixture = get_machine_likelihood(
                    filtered_normal_reads, ref_hmer_size, ref_hmer_size + exp_shift_tries[0]
                )
                filtered_tumor_reads = filter_reads(tumor_read_data, nuc, direction)
                filtered_tumor_results = process_reads(filtered_tumor_reads)
                tumor_cvg = len(filtered_tumor_results)
                tumor_main_hmer, tumor_second_hmer = find_best_mixture_hmer_sizes(filtered_tumor_reads)
                tumor_ml_score, tumor_ml_mixture = get_machine_likelihood(
                    filtered_tumor_reads, ref_hmer_size, ref_hmer_size + exp_shift_tries[0]
                )
                params = calc_parameters(filtered_normal_results)
                cell_shift = 2 * len(params[0])
                num_cells = cell_shift * 21
                original_counts = [zamir] * num_cells
                extra_counts = [[zamir] * num_cells for x in exp_shift_tries]

                for read in filtered_normal_results:
                    exps.append(read[0])
                    groups.append(0)
                    certainty = read[1]
                    weights.append(certainty)
                    cell = get_cell(read, params, cell_shift)
                    if cell >= 0 and cell < num_cells:
                        original_counts[cell] += 1
                    for x in range(len(exp_shift_tries)):
                        curr_cell = cell + cell_shift * exp_shift_tries[x]
                        if curr_cell >= 0 and curr_cell < num_cells:
                            extra_counts[x][curr_cell] += 1
                sum_counts = sum(original_counts)
                normal_exp = sum(exps) / len(exps)
                original_probs = [x / sum_counts for x in original_counts]
                extra_probs = []
                for count in extra_counts:
                    sum_count = sum(count)
                    extra_probs.append([x / sum_count for x in count])
                original_likelyhood = 0
                h0_probs = []
                h1_probs = [[] for x in range(len(exp_shift_tries))]
                if tumor_cvg == 0:
                    print(f"failed on {chrom} position {pos} - tumor cvg is 0")
                    min_score = 0
                    fill_direction_results_with_error(direction_results, prefixes)
                    continue
                tot_number = 0
                tot_exp = 0
                for read in filtered_tumor_results:
                    tot_number += 1
                    tot_exp += read[0]
                    exps.append(read[0])
                    groups.append(1)
                    certainty = read[1]
                    weights.append(certainty)
                    cell = get_cell(read, params, cell_shift)
                    h0_probs.append(original_probs[cell])
                    original_likelyhood += log(original_probs[cell])
                    for i in range(len(exp_shift_tries)):
                        h1_probs[i].append(extra_probs[i][cell])
                tumor_exp = tot_exp / tot_number
                ttest_pval = max(get_pval(exps, groups, weights), LOG_MIN_PVAL_BASE**2)
                h0_probs = np.asarray(h0_probs, dtype=float)
                h1_mixtures = []
                for i in range(len(exp_shift_tries)):
                    try:
                        h1_probs[i] = np.asarray(h1_probs[i], dtype=float)
                        pi = EM_INITIAL_PI
                        for _ in range(EM_ITERATIONS):
                            denom = pi * h1_probs[i] + (1.0 - pi) * h0_probs
                            mon = pi * h1_probs[i]
                            r = mon / denom
                            pi_new = float(r.mean())
                            pi = min(max(pi_new, PI_MIN), PI_MAX)
                        h1_mixtures.append(pi)
                    except:
                        h1_mixtures.append(0.002)
                extra_likelyhoods = []
                for i in range(len(exp_shift_tries)):
                    likely = 0
                    for j in range(len(h0_probs)):
                        likely += log((1 - h1_mixtures[i]) * h0_probs[j] + h1_mixtures[i] * h1_probs[i][j])
                    extra_likelyhoods.append(likely)
                extra_score = max(extra_likelyhoods)
                extra_shift = extra_likelyhoods.index(extra_score)
                mixture = h1_mixtures[extra_shift]
                tmp_score = (
                    -SCORE_ADJUSTMENT_FACTOR
                    * max(original_likelyhood - extra_score + log(len(exp_shift_tries)), MAX_LOG_LIKELIHOOD)
                ) / log(LIKELIHOOD_LOG_BASE)
                tmp_score = min(MAX_COMBINED_SCORE, tmp_score)
                pval_score = -log(ttest_pval) / log(LIKELIHOOD_LOG_BASE)
                likely_score = -log(ttest_pval) / log(LIKELIHOOD_LOG_BASE)
                if tmp_score > 0:
                    likely_score += tmp_score
                pre = prefixes[direction]

                # Store all results for this direction and allele
                direction_results[direction] = {
                    "normal_exp": round(normal_exp, 4),
                    "tumor_exp": round(tumor_exp, 4),
                    "normal_cvg": normal_cvg,
                    "tumor_cvg": tumor_cvg,
                    "ins_size": exp_shift_tries[extra_shift],
                    "ttest_score": round(pval_score, 4),
                    "likely_score": round(tmp_score, 4),
                    "mixture": round(mixture, 4),
                    "normal_main_hmer": normal_main_hmer,
                    "tumor_main_hmer": tumor_main_hmer,
                    "normal_second_hmer": normal_second_hmer,
                    "tumor_second_hmer": tumor_second_hmer,
                    "normal_ml_score": round(normal_ml_score, 4),
                    "normal_ml_mixture": round(normal_ml_mixture, 4),
                    "tumor_ml_score": round(tumor_ml_score, 4),
                    "tumor_ml_mixture": round(tumor_ml_mixture, 4),
                }

                is_ins = len(alt) > len(rec.ref)
                ttest_score = pval_score if (is_ins ^ (tumor_exp < normal_exp)) else -pval_score
                combined_score = combine_scores(
                    ttest_score,
                    tmp_score,
                    mixture,
                    normal_ml_score,
                    normal_ml_mixture,
                    tumor_ml_score,
                    tumor_ml_mixture,
                )
                direction_results[direction]["tot_score"] = round(combined_score, 4)

                min_score = min(combined_score, min_score)

        # Write results as tuples for all alleles and directions
        for direction in range(2):
            pre = prefixes[direction]

            # Collect tuples from all alleles
            for field in [
                "normal_exp",
                "tumor_exp",
                "normal_cvg",
                "tumor_cvg",
                "ins_size",
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

        # Write tot_score as minimum of fw_tot_score and bw_tot_score for each allele
        tot_score_values = tuple(
            min(all_direction_results[alt_idx][0]["tot_score"], all_direction_results[alt_idx][1]["tot_score"])
            for alt_idx in range(len(rec.alts))
        )
        rec.info["tot_score"] = tot_score_values

        # Write once after processing all alleles
        fo.write(rec)
    f.close()
    fo.close()


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
        "tumor_pileup.txt output.vcf --min-hmer 4 --zamir 0.5",
    )

    # Required positional arguments
    parser.add_argument(
        "input_vcf",
        help="Input VCF file path",
    )
    parser.add_argument(
        "normal_reads_file",
        help="Normal sample reads file path",
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
        "--min-hmer",
        type=int,
        default=4,
        help="Minimum homopolymer size (default: 6)",
    )
    parser.add_argument(
        "--zamir",
        type=float,
        default=0.5,
        help="Zamir threshold parameter (default: 0.5)",
    )

    parser.add_argument(
        "--bed-file",
        type=str,
        default=None,
        help="BED file with intervals to process",
    )
    parser.add_argument(
        "--tumor-germline",
        type=str,
        default=None,
        help="Tumor germline VCF file path",
    )
    parser.add_argument(
        "--normal-germline",
        type=str,
        default="/data/Runs/cloud_sync/s3/"
        "genomics-pipeline-concordanz-us-east-1/test/germline/"
        "417309-TN20-126568-Z0280-CGCACAATGCGAGAT.vcf.gz",
        help="Normal germline VCF file path (default: provided path)",
    )

    args = parser.parse_args()

    variant_calling(
        args.input_vcf,
        args.normal_reads_file,
        args.tumor_reads_file,
        args.output_vcf,
        args.min_hmer,
        args.zamir,
        args.bed_file,
        args.tumor_germline,
        args.normal_germline,
    )


if __name__ == "__main__":
    main()
