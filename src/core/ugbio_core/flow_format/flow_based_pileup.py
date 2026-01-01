# Set of classes for working with pileup in a flow space.
# Allows to fetch the flow probabilities for each hompolymer in the pileup.
import numpy as np
import pysam
import ugbio_core.flow_format.flow_based_read as fbr
import ugbio_core.math_utils as phred


class FlowBasedIteratorColumn:
    """Wrapper for pysam.IteratorColumn that allows to fetch flow probabilities for each homopolymer in the pileup.

    Attributes
    ----------
    pileup_iterator : pysam.IteratorColumn
        Original pysam.IteratorColumn object
    """

    MAX_READ_DICTIONARY_SIZE = 1000

    def __init__(self, pileup_iterator: pysam.IteratorColumn):
        self.pileup_iterator = pileup_iterator
        self.__flow_reads_dict = {}

    def __next__(self):
        result = next(self.pileup_iterator)
        pileups = result.pileups

        if len(self.__flow_reads_dict) > FlowBasedIteratorColumn.MAX_READ_DICTIONARY_SIZE:
            qnames = [x.alignment.query_name for x in pileups]
            delete_candidates = []
            for n in self.__flow_reads_dict:
                if n not in qnames:
                    delete_candidates.append(n)
            for n in delete_candidates:
                del self.__flow_reads_dict[n]
        for x in pileups:
            if x.alignment.query_name not in self.__flow_reads_dict:
                fr = fbr.FlowBasedRead.from_sam_record(x.alignment, max_hmer_size=20)
                self.__flow_reads_dict[x.alignment.query_name] = fr
        return FlowBasedPileupColumn(result, self.__flow_reads_dict)


class FlowBasedPileupColumn:
    """Wrapper for pysam.PileupColumn that allows to fetch flow probabilities for each homopolymer in the pileup.

    Attributes
    ----------
    pc : pysam.PileupColumn
        Original pysam.PileupColumn object
    flow_reads_dict : dict
        Dictionary of FlowBasedRead objects, indexed by query name

    Methods
    -------
    fetch_hmers():
        Returns a list of flow probabilities for each homopolymer in the pileup.
    """

    def __init__(self, pc: pysam.PileupColumn, flow_reads: dict):
        """Constructor - receives a pysam.PileupColumn object and a dictionary of FlowBasedRead objects.
        The dictionary is query_name: FlowBasedRead

        Parameters
        ----------
        pc : pysam.PileupColumn
            PileupColumn
        flow_reads : dict
            pysam.AlignedRead reads converted into FlowBasedRead objects
        """

        self.pc = pc
        self.flow_reads_dict = flow_reads

    def fetch_hmer_qualities(self) -> list[tuple]:
        """
        Return list of hmer length probabilities for every read in the PileupColumn

        Return
        ------
        list[tuple]:
            list of pairs (hmer,probabilities of hmer length) for every read  in the PileupColumn

        See also
        --------
        flow_based_read.FlowBasedRead.get_flow_matrix_column_for_base
        """
        qpos = [x.query_position_or_next for x in self.pc.pileups]
        qnames = [x.alignment.query_name for x in self.pc.pileups]
        hmers = [self.flow_reads_dict[x].get_flow_matrix_column_for_base(y) for x, y in zip(qnames, qpos, strict=False)]
        return hmers


class FlowBasedAlignmentFile(pysam.AlignmentFile):
    """Wrapper for pysam.AlignmentFile that returns FlowBasedPileupColumn objects.

    Methods
    -------
    pileup(contig, start, end, mq):
        similar to pysam.AlignmentFile.pileup, but returns FlowBasedPileupColumn objects.
        Works only in `truncate` mode and min_base_quality = 0

    See also
    --------
    pysam.AlignmentFile
    """

    def pileup(self, contig, start, end, mq) -> FlowBasedIteratorColumn:
        """Return a generator of FlowBasedPileupColumn objects.
        Parameters
        ----------
        contig : str
            Reference sequence name
        start : int
            Start position (1-based)
        end : int
            End position (1-based)
        mq : int
            Minimum mapping quality

        Returns
        -------
        FlowBasedIteratorColumn
            Iterator of FlowBasedPileupColumn objects
        """
        pup = super().pileup(
            contig, start, end, truncate=True, min_base_quality=0, flag_filter=3844, min_mapping_quality=mq
        )
        return FlowBasedIteratorColumn(pup)


def get_hmer_qualities_from_pileup_element(
    pe: pysam.PileupRead, max_hmer: int = 20, min_call_prob: float = 0.1
) -> tuple:
    """
    Return hmer length probabilities for a single PileupRead element

    Parameters
    ----------
    pe : pysam.PileupRead
        PileupRead element
    max_hmer : int
        Maximum hmer length that we call
    min_call_prob : float
        Minimum probability for the called hmer length

    Returns
    -------
    tuple:
        pair (hmer,probabilities of hmer length) for the read in the PileupRead element
    See also
    --------
    flow_based_read.FlowBasedRead.get_flow_matrix_column_for_base
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

    # smear probabilities
    if qstart == 0 or qend == len(str(pe.alignment.query_sequence)):
        hmer_probs[:] = 1.0
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

    hmer_probs /= np.sum(hmer_probs)
    return hnuc, hmer_probs
