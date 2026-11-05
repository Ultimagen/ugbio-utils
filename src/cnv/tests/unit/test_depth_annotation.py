import numpy as np
import pandas as pd
import pysam
import pytest
from ugbio_cnv import depth_annotation as da


def test_contig_annotation_gc_n_and_beds():
    seq = "G" * 100 + "AT" * 50 + "N" * 50 + "C" * 50 + "A" * 30  # 330 bp -> 4 bins, last one padded with N
    lowmap = pd.DataFrame({"chrom": ["chr1"], "start": [150], "end": [250]})
    segdup = pd.DataFrame({"chrom": ["chr2"], "start": [0], "end": [100]})
    ann = da.contig_annotation(seq, "chr1", lowmap, segdup)
    assert list(ann["start"]) == [0, 100, 200, 300]
    assert ann["gc"].iloc[0] == pytest.approx(1.0)
    assert ann["gc"].iloc[1] == pytest.approx(0.0)
    assert ann["gc"].iloc[2] == pytest.approx(1.0)  # GC of the ACGT bases only
    assert list(ann["n_frac"]) == pytest.approx([0.0, 0.0, 0.5, 0.7])
    assert list(ann["lowmap"]) == pytest.approx([0.0, 0.5, 0.5, 0.0])
    assert list(ann["segdup"]) == [0.0, 0.0, 0.0, 0.0]


def test_overlapping_intervals_count_once():
    bed = pd.DataFrame({"chrom": ["chr1", "chr1"], "start": [0, 50], "end": [100, 150]})
    assert list(da.bed_fraction(bed, "chr1", 2)) == pytest.approx([1.0, 0.5])


def test_build_writes_parquet(tmp_path):
    fasta = tmp_path / "ref.fa"
    fasta.write_text(">chr1\n" + "GC" * 150 + "\n>chrUn\n" + "A" * 100 + "\n")
    pysam.faidx(str(fasta))
    bed = tmp_path / "empty.bed"
    bed.write_text("chr9\t0\t10\n")
    out = tmp_path / "ann.parquet"
    da.build_depth_annotation(str(fasta), str(bed), str(bed), str(out), contigs=["chr1", "chr2"])
    ann = pd.read_parquet(out)
    assert set(ann["chrom"].astype(str)) == {"chr1"}  # chr2 is not in the FASTA, chrUn not requested
    assert len(ann) == 3
    assert np.allclose(ann["gc"], 1.0)


def test_regions_restrict_the_annotation(tmp_path):
    fasta = tmp_path / "ref.fa"
    fasta.write_text(">chr1\n" + "GC" * 500 + "\n>9\n" + "AT" * 150 + "\n")
    pysam.faidx(str(fasta))
    bed = tmp_path / "empty.bed"
    bed.write_text("chr9\t0\t10\n")
    out = tmp_path / "ann.parquet"
    # 1-based inclusive 151-250 touches bins starting at 100 and 200; '9' (b37 naming) is a whole contig
    da.build_depth_annotation(str(fasta), str(bed), str(bed), str(out), regions=["chr1:151-250", "9"])
    ann = pd.read_parquet(out)
    ann["chrom"] = ann["chrom"].astype(str)
    assert ann[ann.chrom == "chr1"]["start"].tolist() == [100, 200]
    assert ann[ann.chrom == "9"]["start"].tolist() == [0, 100, 200]


@pytest.mark.parametrize(
    ("region", "expected"),
    [
        ("chr1:1-12,950,000", ("chr1", 0, 12950000)),
        ("9", ("9", 0, None)),
        ("HLA-A*01:01:01:01:100-200", ("HLA-A*01:01:01:01", 99, 200)),
    ],
)
def test_parse_region(region, expected):
    assert da.parse_region(region) == expected
