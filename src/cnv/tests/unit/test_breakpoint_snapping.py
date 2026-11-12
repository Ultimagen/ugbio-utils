import pytest
from ugbio_cnv import breakpoint_snapping as bs


class TestJunctionParsing:
    def test_read_length_includes_clips(self):
        assert bs.read_length_from_cigar("20S100M30H") == 150
        assert bs.read_length_from_cigar("50M5I45M") == 100

    def test_piece_forward_and_reverse(self):
        # 100M50S on + strand: query 0-100, reference 1000-1100
        assert bs.alignment_piece(1000, "+", "100M50S", 150) == (0, 100, 1000, 1100)
        # same alignment on - strand: in read orientation the aligned part is the last 100 bases
        assert bs.alignment_piece(1000, "-", "100M50S", 150) == (50, 150, 1000, 1100)

    def test_deletion_junction_forward(self):
        # read: first 100 bases end at 10100, last 50 bases start at 15000 -> deletion 10100-15000
        js = bs.split_read_junctions("chr1", 10000, "100M50S", "chr1,15001,+,100S50M,60,0;", is_reverse=False)
        assert js == [(10100, 15000, "DEL")]

    def test_deletion_junction_reverse_strand(self):
        # on the minus strand the read runs right to left: the primary piece (end of the read) is on the right
        js = bs.split_read_junctions("chr1", 15000, "100S50M", "chr1,10001,-,100M50S,60,0;", is_reverse=True)
        assert js == [(10100, 15000, "DEL")]

    def test_tandem_duplication_junction(self):
        # read start maps to the duplication end (20100), its continuation maps back to the duplication start (12000)
        js = bs.split_read_junctions("chr1", 20000, "100M50S", "chr1,12001,+,100S50M,60,0;", is_reverse=False)
        assert js == [(12000, 20100, "DUP")]

    def test_microhomology_is_left_normalized(self):
        # 3 bases aligned in both pieces: read 0-100 -> 10000-10100 and read 97-150 -> 15000-15053;
        # the shared bases go to the second piece: deletion 10097-15000 (the left-normalized VCF position)
        js = bs.split_read_junctions("chr1", 10000, "100M50S", "chr1,15001,+,97S53M,60,0;", is_reverse=False)
        assert js == [(10097, 15000, "DEL")]
        # the same read on the minus strand
        js = bs.split_read_junctions("chr1", 14999, "97S53M", "chr1,10001,-,100M50S,60,0;", is_reverse=True)
        assert js == [(10097, 14999, "DEL")]  # left piece 10000-10100, right piece from 14999, 3 bases shared
        # tandem duplication: the end of the duplicated segment moves left by the overlap
        js = bs.split_read_junctions("chr1", 20000, "100M50S", "chr1,12001,+,97S53M,60,0;", is_reverse=False)
        assert js == [(12000, 20097, "DUP")]

    def test_inserted_bases_at_the_junction_keep_the_breakpoints(self):
        # 5 read bases in neither piece (an insertion at the junction): no overlap, breakpoints unchanged
        js = bs.split_read_junctions("chr1", 10000, "100M55S", "chr1,15001,+,105S50M,60,0;", is_reverse=False)
        assert js == [(10100, 15000, "DEL")]

    def test_other_chromosome_strand_or_low_mapq_ignored(self):
        assert bs.split_read_junctions("chr1", 10000, "100M50S", "chr2,15001,+,100S50M,60,0;", is_reverse=False) == []
        assert bs.split_read_junctions("chr1", 10000, "100M50S", "chr1,15001,-,100S50M,60,0;", is_reverse=False) == []
        assert bs.split_read_junctions("chr1", 10000, "100M50S", "chr1,15001,+,100S50M,0,0;", is_reverse=False) == []


class TestBestCluster:
    def test_median_of_largest_group(self):
        js = [(10100, 15000), (10102, 15003), (10098, 14999), (30000, 36000)]
        assert bs.best_junction_cluster(js, 5000, bs.SnapParams()) == (10100, 15000, 3)

    def test_needs_two_reads(self):
        assert bs.best_junction_cluster([(10100, 15000)], 5000, bs.SnapParams()) is None

    @pytest.mark.parametrize(("call_len", "expected"), [(2000, (1000, 1500, 2)), (20000, None)])
    def test_large_calls_need_large_junctions(self, call_len, expected):
        # a 500 bp junction may refine a 2 kb call but must not shrink a 20 kb call
        js = [(1000, 1500), (1001, 1500)]
        assert bs.best_junction_cluster(js, call_len, bs.SnapParams()) == expected

    def test_small_indels_ignored(self):
        assert bs.best_junction_cluster([(1000, 1050), (1000, 1050)], 1000, bs.SnapParams()) is None
