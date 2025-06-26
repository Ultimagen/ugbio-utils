from pathlib import Path

from ugbio_srsnv.srsnv_training import _parse_interval_list


def test_parse_interval_list_resource() -> None:
    """
    Verify that _parse_interval_list correctly extracts contig sizes and order
    from the hg38 calling-regions fixture.
    """
    res_dir = Path(__file__).parent.parent / "resources"
    interval_path = res_dir / "wgs_calling_regions.without_encode_blacklist.hg38.interval_list"
    assert interval_path.is_file(), "interval_list fixture is missing from resources"

    chrom_sizes, chrom_list = _parse_interval_list(str(interval_path))

    # Basic sanity checks
    assert chrom_list, "no chromosomes returned"
    assert set(chrom_list).issubset(chrom_sizes), "every chromosome must have a size entry"

    # First contig in data lines should match first in chrom_list
    first_data_chrom = None
    with interval_path.open() as fh:
        for line in fh:
            if not line.startswith("@"):
                first_data_chrom = line.split("\t", 1)[0]
                break
    assert chrom_list[0] == first_data_chrom
