import warnings

warnings.filterwarnings('ignore')

import ugbio_core.dna_sequence_utils as dna_sequence_utils


class TestDnaUtils:

    def test_revcomp(self):
        seq = "ATGCAGCTGTGTTACGCGAT"

        assert dna_sequence_utils.revcomp(seq) == "ATCGCGTAACACAGCTGCAT"

    def test_revcom_arr(self):
        seq = ['A', 'T', 'G', 'C', 'A', 'G', 'C', 'T', 'G', 'T', 'G', 'T', 'T', 'A', 'C', 'G', 'C', 'G', 'A', 'T']
        assert dna_sequence_utils.revcomp(list(seq)) == ['A', 'T', 'C', 'G', 'C', 'G', 'T', 'A', 'A', 'C', 'A', 'C', 'A', 'G',
                                                'C', 'T', 'G', 'C', 'A', 'T']
