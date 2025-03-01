import numpy as np
import pandas as pd
import pyfaidx
import pysam
import ugbio_core.concordance.flow_based_concordance as fbc
import ugbio_core.flow_format.flow_based_read as fbr
from ugbio_core.vcfbed import vcftools

import ugbio_filtering.training_prep as tprep
from ugbio_filtering.tprep_constants import SPAN_DEL


def select_overlapping_variants(df: pd.DataFrame, *, require_star_for_spandel: bool = True) -> list:
    """Selects lists of overlapping variants that need to be genotyped together. This
    can be multiallelic variants or variants with spanning deletion

    Parameters
    ----------
    df : pd.DataFrame
        Training set dataframe
    require_star_for_spandel: bool
        Should "*" appear in the alleles of the row to be considered as a spanning deletion (true)
        or just overlap with deletion is enough (false)
    Returns
    -------
    list
        List of list of indices that generate the co-genotyped sets
    """
    # first we generate a list of multiallelic locations
    multiallelic_locations = set(np.where(df["alleles"].apply(len) > 2)[0])  # noqa PLR2004
    del_length = df["alleles"].apply(lambda x: max(len(x[0]) - len(y) for y in x))
    current_span = 0
    results = []
    cluster = []
    for i in range(df.shape[0]):
        # The cluster is a long deletion with the variants that it spans (in which case they
        # should contain SPAN_DEL allele)

        # If there is no suspicious spanning deletion and the variant is not a deletion - skip
        if len(cluster) == 0 and del_length[i] == 0:
            continue
        # if we are out of the span of the current deletion, we need to close the cluster
        if df.iloc[i]["pos"] > current_span:
            if len(cluster) > 1:
                for c in cluster:
                    if c in multiallelic_locations:
                        multiallelic_locations.remove(c)
                results.append(cluster[:])
            cluster = []
        # if we start a new deletion
        if len(cluster) == 0:
            cluster.append(i)
        # or if we are in a deletion and the variant contains SPAN_DEL
        elif len(cluster) > 0 and ((SPAN_DEL in df.iloc[i]["alleles"]) or (not require_star_for_spandel)):
            cluster.append(i)
        # case when a deletion spans another deletion
        current_span = max(current_span, del_length[i] + df.iloc[i]["pos"])
    for m in multiallelic_locations:
        results.append([m])
    # sorting the list of lists according to the position. Not sure how critical this is.
    sorted_results = sorted(results)
    return sorted_results


def split_multiallelic_variants(
    multiallelic_variant: pd.Series,
    call_vcf_header: pysam.VariantHeader | pysam.VariantFile | str,
    ref: pyfaidx.FastaRecord,
) -> pd.DataFrame:
    """Splits multiallelic variants into multiple rows. It is hard to train a model that would predict
    a genotype of (1/2). So we split the multiallelic variant into two rows: one genotypes the REF with the
    first (strongest ALT) and basically asks if there are REF reads. The second genotypes the second ALT with the
    first ALT as the reference (basically asks if there are reads supporting the second ALT).

    Note that we do not treat cases when there are more than two alternative alleles.

    Parameters
    ----------
    multiallelic_variant : pd.Series
        A row from the training set dataframe
    call_vcf_header :
        Header of the VCF : pysam.VariantHeader | pysam.VariantFile | str
    ref : pyfaidx.Fasta
        Reference

    Returns
    -------
    pd.DataFrame
        A dataframe with the same columns as the input Series index (input Series are
        originally a row in the training set dataframe). The multiallielic variant will be
        split into multiple rows that will first genotype REF and ALT1, and then ALT1 and ALT2.
        We do not genotype more than two alleles of the multiallelic variant.
    """
    record_to_nbr_dict = vcftools.header_record_number(call_vcf_header)
    ignore_columns = ["gt_vcfeval", "alleles_vcfeval", "label", "sync"]

    for ic in ignore_columns:
        record_to_nbr_dict[ic] = 1

    alleles = multiallelic_variant["alleles"]

    # select the strongest alleles in order
    pls = np.array(
        [
            select_pl_for_allele_subset(multiallelic_variant["pl"], (0, i), normed=False)[-1]
            for i in range(1, len(alleles))
        ]
    )
    notingt = np.array([i not in multiallelic_variant["gt"] for i in range(1, len(alleles))])

    # taking care of the edge case when there are several equally strong alleles -
    # then the ones that are in GT should be first
    allele_order = np.argsort(pls + notingt * 1000) + 1

    # very rare case when there is a spanning deletion without an actual deletion (I guess GATK bug)
    allele_order = [x for x in allele_order if alleles[x] != SPAN_DEL]
    if len(allele_order) == 1:
        return pd.DataFrame(
            extract_allele_subset_from_multiallelic(multiallelic_variant, (0, allele_order[0]), record_to_nbr_dict, ref)
        ).T
    return pd.concat(
        (
            extract_allele_subset_from_multiallelic(multiallelic_variant, alleles, record_to_nbr_dict, ref)
            for alleles in ((0, allele_order[0]), (allele_order[0], allele_order[1]))
        ),
        axis=1,
    ).T


def extract_allele_subset_from_multiallelic(
    multiallelic_variant: pd.Series, alleles: tuple, record_to_nbr_dict: dict, ref: pyfaidx.FastaRecord
) -> pd.Series:
    """When analyzing multiallelic variants, we split them into pairs of alleles. Each pair of alleles
    need to have the updated columns (GT/PL etc.). This function updates those columns

    Parameters
    ----------
    multiallelic_variant : pd.Series
        The series that represent the multiallelic variant
    alleles : tuple
        The allele tuple to fetch from the multiallelic variant
    record_to_nbr_dict : dict
        Dictionary that says what is the encoding in the VCF of each column
    ref : pyfaidx.Fasta
        Reference FASTA

    Returns
    -------
    pd.Series
        Updated subsetted variant
    """
    pos = multiallelic_variant["pos"]
    special_treatment_columns = {
        "sb": lambda x: x,
        "pl": lambda x: select_pl_for_allele_subset(x, alleles),
        "gt": lambda x: encode_gt_for_allele_subset(x, alleles),
        "ref": lambda _: encode_ref_for_allele_subset(multiallelic_variant["alleles"], alleles),
        "indel": lambda _: is_indel_subset(multiallelic_variant["alleles"], alleles),
        "x_ic": lambda _: indel_classify_subset(multiallelic_variant["alleles"], alleles)[0],
        "x_il": lambda _: indel_classify_subset(multiallelic_variant["alleles"], alleles)[1],
        "x_hil": lambda _: (
            classify_hmer_indel_relative(multiallelic_variant["alleles"], alleles, ref, pos, flow_order="TGCA")[1],
        ),
        "x_hin": lambda _: (
            classify_hmer_indel_relative(multiallelic_variant["alleles"], alleles, ref, pos, flow_order="TGCA")[0],
        ),
        "label": lambda x: encode_label(x, alleles),
    }
    result = {}
    for col in multiallelic_variant.index:
        if col in special_treatment_columns:
            result[col] = special_treatment_columns[col](multiallelic_variant.at[col])  # noqa PD008
        elif isinstance(multiallelic_variant[col], tuple) and record_to_nbr_dict[col] != 1:
            result[col] = vcftools.subsample_to_alleles(multiallelic_variant.at[col], record_to_nbr_dict[col], alleles)  # noqa PD008
        else:
            result[col] = multiallelic_variant.at[col]  # noqa PD008
    return pd.Series(result)


def encode_ref_for_allele_subset(allele_tuple: tuple[str], allele_idcs: tuple) -> str:
    """Make reference allele for the indices tuple

    Parameters
    ----------
    allele_tuple: tuple
        Tuple of alleles
    allele_idcs:
        Indices to select (the first one will be considered "ref")

    Returns
    -------
    str
        Indices to select (the first one will be considered "ref")

    Returns
    -------
    str
        String that corresponds to the reference allele
    """
    return allele_tuple[allele_idcs[0]]


def encode_gt_for_allele_subset(original_gt: tuple, allele_idcs: tuple) -> tuple:
    """Encodes GT according to the subsampled allele indices.
    If only first allele is present in the original genotype: 0,0
    If only second allele is present in the original genotype: 1,1
    If both alleles are present: 0,1
    Requires: at least one allele is present

    Parameters
    ----------
    original_gt : tuple
        Original genotype tuple
    allele_idcs : tuple
        Indices of alleles that are being selected

    Returns
    -------
    tuple
        Resulting genotype, at least one allele should exist in the orginal genotype

    Raises
    ------
    RuntimeError
        if neither allele is present in the original genotype
    """
    assert (  # noqa S101
        allele_idcs[0] in original_gt or allele_idcs[1] in original_gt
    ), "One of the alleles should be present in the GT"
    if allele_idcs[0] in original_gt and allele_idcs[1] not in original_gt:
        return (0, 0)
    if allele_idcs[1] in original_gt and allele_idcs[0] not in original_gt:
        return (1, 1)
    if allele_idcs[1] in original_gt and allele_idcs[0] in original_gt:
        return (0, 1)
    raise RuntimeError("Neither allele found in the original genotype")


def get_pl_idx(tup: tuple) -> int:
    """Returns the index of the PL value in the tuple

    Parameters
    ----------
    tup : tuple
        PL tuple

    Returns
    -------
    int
        Index of the PL value
    """
    offset: int = max(tup) * (max(tup) + 1) // 2
    return offset + int(min(tup))


def get_gt_from_pl_idx(idx: int) -> tuple:
    """Returns GT given index in PL

    Parameters
    ----------
    idx : int
        index of the PL value

    Returns
    -------
    tuple
        Genotype
    """

    count = 0
    n_alleles = 0
    while count < idx + 1:
        count += n_alleles
        n_alleles += 1
    max_allele = n_alleles - 1
    min_allele = idx - (count - n_alleles)
    return (min_allele - 1, max_allele - 1)


def select_pl_for_allele_subset(original_pl: tuple, allele_idcs: tuple, *, normed: bool = True) -> tuple:
    """Selects Pl values according to the subsampled allele indices. Normalizes to the minimum

    Parameters
    ----------
    original_pl : tuple
        Original PL tuple
    allele_idcs : tuple
        Indices of alleles that are being selected
    normed: bool
        Should PL be normalized after subsetting so that the highest is zero, default: True

    Returns
    -------
    tuple
        Resulting genotype, at least one allele should exist in the orginal genotype
    """
    allele_idcs = tuple(allele_idcs)
    idcs = [
        get_pl_idx(x)
        for x in ((allele_idcs[0], allele_idcs[0]), (allele_idcs[0], allele_idcs[1]), (allele_idcs[1], allele_idcs[1]))
    ]
    pltake = [original_pl[x] for x in idcs]
    if normed:
        min_pl = min(pltake)
        return tuple(x - min_pl for x in pltake)
    if len(pltake) > 3:  # noqa PLR2004
        print(original_pl, allele_idcs)
    return tuple(pltake)


def is_indel_subset(alleles: tuple, allele_indices: tuple, spandel: pd.Series | None = None) -> bool:
    """Checks if the variant is an indel

    Parameters
    ----------
    alleles : tuple
        Alleles tuple
    allele_indices : tuple
        Indices of alleles that are being selected
    spandel : pd.Series, optional
        The variant at the position that generates the spanning deletion, by default None (not used), relevant only if
        the selected tuple of alleles contains spanning deletion.

    Returns
    -------
    bool
        True if indel, False otherwise

    Raises
    ------
    RuntimeError
        If alleles contain spanning deletion but the spanning deletion series is None
    """
    if spandel is not None and _contain_spandel(alleles, allele_indices):
        return True  # spanning deletion always is an indel
    if spandel is None and _contain_spandel(alleles, allele_indices):
        raise RuntimeError("Can't deal with spanning deletion allele without the spandel")
    return any(len(alleles[x]) != len(alleles[allele_indices[0]]) for x in allele_indices)


def indel_classify_subset(
    alleles: tuple, allele_indices: tuple, spandel: pd.Series | None = None
) -> tuple[tuple[str | None], tuple[int | None]]:
    """Checks if the variant is insertion or deletion

    Parameters
    ----------
    alleles : tuple
        Alleles tuple
    allele_indices : tuple
        Indices of alleles that are being selected
    spandel: pd.Series, optional
        The variant at the position that generates the spanning deletion, by default None (not used),
        relevant only if the alleles contains spanning deletion

    Returns
    -------
    tuple[tuple[str|None], tuple[int]]
        (('ins', ), (length,)) or (('del',), (length,)). The reason for the awkward
        output format is that this is the format of x_il and x_ic tags in the VCF

    Raises
    ------
    RuntimeError
        If alleles contain spanning deletion but the spanning deletion series is None

    """
    if not is_indel_subset(alleles, allele_indices, spandel):
        return (("NA",), (None,))
    ref_allele = alleles[allele_indices[0]]
    alt_allele = alleles[allele_indices[1]]
    if spandel is not None and _contain_spandel(alleles, allele_indices):
        return (
            ("del",),
            (spandel["x_il"][0],),
        )  # spanning deletion always is a deletion, length determined by the spanning deletion
    if spandel is None and _contain_spandel(alleles, allele_indices):
        raise RuntimeError("Unable to parse spanning deletion without the variant that contains the deletion allele")

    if len(ref_allele) > len(alt_allele):
        return (("del",), (len(ref_allele) - len(alt_allele),))
    return (("ins",), (len(alt_allele) - len(ref_allele),))


def classify_hmer_indel_relative(
    alleles: tuple,
    allele_indices: tuple,
    ref: pyfaidx.FastaRecord | str,
    pos: int,
    flow_order: str = "TGCA",
    spandel: pd.Series | None = None,
) -> tuple:
    """Checks if one allele is hmer indel relative to the other

    Parameters
    ----------
    alleles : tuple
        Tuple of all alleles
    allele_indices : tuple
        Pair of allele indices, the second one is measured relative to the first one
    ref : pyfaidx.Fasta or str
        Reference sequence
    pos: int
        Position of the variant (one-based)
    flow_order: str
        Flow order, default is TGCA
    spandel: pd.Series, optional
        The variant at the position that generates the spanning deletion, by default None (not used),
        relevant only if the alleles contain spanning deletion.

    Returns
    -------
    tuple
        Pair of hmer indel nucleotide and hmer indel length

    Raises
    ------
    RuntimeError
        If the allele indices point to spanning deletion but spandel variable is not given
    """
    refstr = fbc.get_reference_from_region(ref, (max(0, pos - 20), min(pos + 20, len(ref))))
    refstr = refstr.upper()
    # case of spanning deletion
    if spandel is not None and _contain_spandel(alleles, allele_indices):
        fake_series = pd.DataFrame(
            pd.Series(
                {
                    "alleles": alleles,
                    "gt": [a for a in allele_indices if alleles[a] != SPAN_DEL],
                    "pos": pos,
                    "ref": alleles[0],
                }
            )
        ).T
        haplotypes = fbc.apply_variants_to_reference(
            refstr, fake_series, pos - 20, genotype_col="gt", include_ref=False
        )
        fake_series = pd.DataFrame(
            pd.Series(
                {"alleles": spandel["alleles"][0:2], "gt": (0, 1), "pos": spandel["pos"], "ref": spandel["alleles"][0]}
            )
        ).T
        haplotypes = (
            haplotypes
            + fbc.apply_variants_to_reference(refstr, fake_series, pos - 20, genotype_col="gt", include_ref=False)[1:2]
        )
    elif spandel is None and _contain_spandel(alleles, allele_indices):
        raise RuntimeError(
            "when the alleles contain spanning deletion, the line containing the variant that is a deletion is required"
        )
    else:
        fake_series = pd.DataFrame(
            pd.Series({"alleles": alleles, "gt": allele_indices, "pos": pos, "ref": alleles[0]})
        ).T
        haplotypes = fbc.apply_variants_to_reference(
            refstr, fake_series, pos - 20, genotype_col="gt", include_ref=False
        )
    fhaplotypes = [fbr.generate_key_from_sequence(x, flow_order) for x in haplotypes]
    compare = fbc.compare_haplotypes(fhaplotypes[0:1], fhaplotypes[1:2])
    if compare[0] != 1:
        return (".", 0)
    flow_location = np.nonzero(fhaplotypes[0] - fhaplotypes[1])[0]
    nucleotide = flow_order[flow_location[0] % 4]
    length = max(int(fhaplotypes[0][flow_location]), int(fhaplotypes[1][flow_location]))
    return (nucleotide, length)


def encode_label(original_label: tuple, allele_indices: tuple) -> tuple:
    """Encodes a training label for the subset of alleles
    Parameters
    ----------
    original_label: tuple
        The current label
    allele_indices: tuple
        Pair of allele indices, the second one is measured relative to the first

    Returns
    -------
    tuple:
        label - (0,0),(0,1),(1,1). In case the label can't be encoded in the current alleles - for example when
        both alleles are not in the label - returns (-2,-2)

    Raises
    ------
    RuntimeError
        If there is some issue encoding the label
    """

    # TODO: check what MISS is coming from (maybe these should be false positives)?
    if tprep.MISS in original_label or tprep.IGNORE in original_label:
        return original_label
    if allele_indices[0] in original_label and allele_indices[1] in original_label:
        return (0, 1)
    if allele_indices[0] in original_label and allele_indices[1] not in original_label:
        return (0, 0)
    if allele_indices[1] in original_label and allele_indices[0] not in original_label:
        return (1, 1)
    if allele_indices[0] not in original_label and allele_indices[1] not in original_label:
        return (tprep.MISS, tprep.MISS)
    raise RuntimeError(f"can''t encode {original_label} with {allele_indices} Bug?")


def cleanup_multiallelics(training_df: pd.DataFrame) -> pd.DataFrame:
    """Fixes multiallelics in the training set dataframe. Converts non-h-indels that
        are hmer indels into h-indel variant type. Adjust the values of the RU/RPA/STR
    when the variant is convered to hmer indel. (I.e. A -> CCA or CA, CCA is hmer indel of CA).
    Recalculates gq, qual and qd

        Parameters
        ----------
        training_df : pd.DataFrame
           Input data frame

        Returns
        -------
        pd.DataFrame
            Output dataframe
    """
    training_df = training_df.copy()
    select = (training_df["variant_type"] == "snp") & (
        training_df["x_il"].apply(lambda x: x[0] is not None and x[0] != 0)
    )
    training_df.loc[select, "variant_type"] = "non-h-indel"

    select = (training_df["variant_type"] == "non-h-indel") & (
        training_df["x_hil"].apply(lambda x: x[0] is not None and x[0] > 0)
    )
    training_df.loc[select, "variant_type"] = "h-indel"

    fix_str = "str" in training_df.columns  # in some cases we do not have this annotation

    if fix_str:
        training_df.loc[select, "str"] = True
        training_df.loc[select, "ru"] = training_df.loc[select, "x_hin"].apply(lambda x: x[0])
        ins_or_del = training_df.loc[select, "x_ic"].apply(lambda x: x[0])
        training_df.loc[select, "ins_or_del"] = ins_or_del

        def _alleles_lengths(v: pd.Series) -> tuple:
            inslen = v["x_il"][0]
            if inslen is None:
                inslen = 0
            if v["ins_or_del"] == "ins":
                return (v["x_hil"][0], v["x_hil"][0] + inslen)
            return (v["x_hil"][0] + inslen, v["x_hil"][0])

        training_df.loc[select, "rpa"] = training_df.loc[select].apply(_alleles_lengths, axis=1)
        training_df = training_df.drop("ins_or_del", axis=1)
    select = (training_df["variant_type"] == "h-indel") & (
        training_df["x_hil"].apply(lambda x: x[0] is None or x[0] == 0)
    )
    training_df.loc[select, "variant_type"] = "non-h-indel"
    pls = training_df["pl"].apply(sorted)
    training_df["gq"] = np.clip(pls.apply(lambda x: x[1] - x[0]), 0, 99)
    mq = training_df["pl"].apply(lambda x: sorted(x[1:])[0])
    qref = training_df["pl"].apply(lambda x: x[0])
    training_df["qual"] = np.clip(mq - qref, 0, None)
    training_df["qd"] = training_df["qual"] / training_df["dp"]

    return training_df


def _contain_spandel(alleles: tuple, allele_indices: tuple) -> bool:
    """Returns if the allele_indices point to spanning deletion

    Parameters
    ----------
    alleles : tuple
        Tuple of alleles
    allele_indices : tuple
        Tuple of allele indices that are being queried (length = 2)

    Returns
    -------
    bool
        True if SPAN_DEL is in the alleles
    """
    return SPAN_DEL in (alleles[allele_indices[0]], alleles[allele_indices[1]])
