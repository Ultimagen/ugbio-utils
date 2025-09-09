import gzip
import tempfile
from pathlib import Path

from ugbio_srsnv.srsnv_training import _parse_interval_list


def test_parse_interval_list_resource() -> None:
    """
    Verify that _parse_interval_list correctly extracts contig sizes and order
    from the hg38 calling-regions fixture.
    """
    res_dir = Path(__file__).parent.parent / "resources"
    interval_path = res_dir / "wgs_calling_regions.without_encode_blacklist.hg38.interval_list.gz"
    assert interval_path.is_file(), "interval_list fixture is missing from resources"

    chrom_sizes, chrom_list = _parse_interval_list(str(interval_path))

    # Basic sanity checks
    assert chrom_list, "no chromosomes returned"
    assert set(chrom_list).issubset(chrom_sizes), "every chromosome must have a size entry"

    # First contig in data lines should match first in chrom_list
    first_data_chrom = None
    with gzip.open(interval_path, "rt") as fh:
        for line in fh:
            if not line.startswith("@"):
                first_data_chrom = line.split("\t", 1)[0]
                break
    assert chrom_list[0] == first_data_chrom


def test_parse_interval_list_gzipped() -> None:
    """
    Verify that _parse_interval_list correctly handles gzipped files.
    """
    # Create a minimal interval list content
    content = """@HD	VN:1.6	SO:coordinate
@SQ	SN:chr1	LN:248956422
@SQ	SN:chr2	LN:242193529
chr1	100	200	+	region1
chr2	300	400	+	region2
"""

    with tempfile.NamedTemporaryFile(suffix=".interval_list.gz", delete=False) as tmp_gz:
        # Write content as gzipped
        with gzip.open(tmp_gz.name, "wt", encoding="utf-8") as gz_file:
            gz_file.write(content)

        try:
            chrom_sizes, chrom_list = _parse_interval_list(tmp_gz.name)

            # Verify parsing results
            assert len(chrom_sizes) == 2
            assert chrom_sizes["chr1"] == 248956422
            assert chrom_sizes["chr2"] == 242193529
            assert chrom_list == ["chr1", "chr2"]

        finally:
            # Clean up temporary file
            Path(tmp_gz.name).unlink()


def test_parse_interval_list_plain_text() -> None:
    """
    Verify that _parse_interval_list still works with plain text files.
    """
    # Create a minimal interval list content
    content = """@HD	VN:1.6	SO:coordinate
@SQ	SN:chr1	LN:248956422
@SQ	SN:chr2	LN:242193529
chr1	100	200	+	region1
chr2	300	400	+	region2
"""

    with tempfile.NamedTemporaryFile(mode="w", suffix=".interval_list", delete=False) as tmp_txt:
        tmp_txt.write(content)
        tmp_txt.flush()

        try:
            chrom_sizes, chrom_list = _parse_interval_list(tmp_txt.name)

            # Verify parsing results
            assert len(chrom_sizes) == 2
            assert chrom_sizes["chr1"] == 248956422
            assert chrom_sizes["chr2"] == 242193529
            assert chrom_list == ["chr1", "chr2"]

        finally:
            # Clean up temporary file
            Path(tmp_txt.name).unlink()
