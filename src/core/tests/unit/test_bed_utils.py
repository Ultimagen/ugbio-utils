import pytest
from ugbio_core.bed_utils import BedUtils


@pytest.mark.parametrize(
    "include_regions, exclude_regions, expected_output",
    [
        # Test with a single include region and no exclude regions
        (["chr1\t10\t20\n"], None, ["chr1\t10\t20\n"]),
        # Test with multiple include regions and no exclude regions (they overlap)
        (["chr1\t10\t20\n", "chr1\t15\t25\n"], None, ["chr1\t15\t20\n"]),
        # Test with include and exclude regions (overlap partially)
        (["chr1\t10\t20\n"], ["chr1\t15\t25\n"], ["chr1\t10\t15\n"]),
        # Test with multiple overlapping include regions and overlapping exclude regions
        (
            ["chr1\t10\t30\n", "chr1\t20\t40\n"],
            ["chr1\t25\t35\n"],
            ["chr1\t20\t25\n"],
        ),
        # Test where exclude regions completely remove include regions
        (
            ["chr1\t10\t20\n", "chr2\t30\t40\n"],
            ["chr1\t10\t20\n", "chr2\t30\t40\n"],
            [],
        ),
        # Test with include and exclude regions (overlap partially) - like the next test without headers
        (
            ["chr1\t5\t25\n", "chr1\t15\t40\n"],
            ["chr1\t10\t20\n"],
            ["chr1\t20\t25\n"],
        ),
        # Test bed files with headers
        (
            ["##header1\nchr1\t5\t25\n", "##header2\nchr1\t15\t40\n"],
            ["##header3\nchr1\t10\t20\n"],
            ["chr1\t20\t25\n"],
        ),
    ],
)
def test_intersect_bed_regions(tmpdir, include_regions, exclude_regions, expected_output):
    include_files = []
    for idx, content in enumerate(include_regions):
        file_path = tmpdir.join(f"include_{idx}.bed")
        file_path.write(content)
        include_files.append(str(file_path))

    exclude_files = None
    if exclude_regions:
        exclude_files = []
        for idx, content in enumerate(exclude_regions):
            file_path = tmpdir.join(f"exclude_{idx}.bed")
            file_path.write(content)
            exclude_files.append(str(file_path))

    output_file = str(tmpdir.join("output.bed"))

    BedUtils().intersect_bed_regions(
        include_regions=include_files, exclude_regions=exclude_files, output_bed=output_file
    )

    with open(output_file) as f:
        result = f.readlines()

    assert result == expected_output


@pytest.mark.parametrize(
    "a_content, b_content, column, operation, expected_lines, presort",
    [
        # Test basic mean operation on column 4 (scores)
        (
            "chr1\t100\t200\tregion1\n" "chr1\t300\t400\tregion2\n",
            "chr1\t150\t160\tscore1\t10\n" "chr1\t170\t180\tscore2\t20\n" "chr1\t350\t360\tscore3\t30\n",
            5,
            "mean",
            [
                "chr1\t100\t200\tregion1\t15\n",
                "chr1\t300\t400\tregion2\t30\n",
            ],
            False,
        ),
        # Test sum operation
        (
            "chr1\t100\t200\tregion1\n",
            "chr1\t150\t160\tscore1\t10\n" "chr1\t170\t180\tscore2\t20\n",
            5,
            "sum",
            ["chr1\t100\t200\tregion1\t30\n"],
            False,
        ),
        # Test max operation
        (
            "chr1\t100\t200\tregion1\n",
            "chr1\t150\t160\tscore1\t10\n" "chr1\t170\t180\tscore2\t25\n" "chr1\t190\t195\tscore3\t5\n",
            5,
            "max",
            ["chr1\t100\t200\tregion1\t25\n"],
            False,
        ),
        # Test count operation
        (
            "chr1\t100\t200\tregion1\n" "chr1\t300\t400\tregion2\n",
            "chr1\t150\t160\tscore1\t10\n" "chr1\t170\t180\tscore2\t20\n" "chr1\t350\t360\tscore3\t30\n",
            5,
            "count",
            [
                "chr1\t100\t200\tregion1\t2\n",
                "chr1\t300\t400\tregion2\t1\n",
            ],
            False,
        ),
        # Test with presort enabled
        (
            "chr1\t300\t400\tregion2\n" "chr1\t100\t200\tregion1\n",  # unsorted
            "chr1\t350\t360\tscore3\t30\n" "chr1\t150\t160\tscore1\t10\n",  # unsorted
            5,
            "mean",
            [
                "chr1\t100\t200\tregion1\t10\n",
                "chr1\t300\t400\tregion2\t30\n",
            ],
            True,
        ),
        # Test collapse operation (concatenates values)
        (
            "chr1\t100\t200\tregion1\n",
            "chr1\t150\t160\tscore1\t10\n" "chr1\t170\t180\tscore2\t20\n",
            5,
            "collapse",
            ["chr1\t100\t200\tregion1\t10,20\n"],
            False,
        ),
    ],
)
def test_bedtools_map(tmpdir, a_content, b_content, column, operation, expected_lines, presort):
    """Test bedtools_map function with various operations and parameters."""
    # Create input files
    a_file = tmpdir.join("a.bed")
    a_file.write(a_content)

    b_file = tmpdir.join("b.bed")
    b_file.write(b_content)

    output_file = str(tmpdir.join("output.bed"))

    # Run bedtools_map
    BedUtils().bedtools_map(
        a_bed=str(a_file),
        b_bed=str(b_file),
        output_bed=output_file,
        column=column,
        operation=operation,
        presort=presort,
    )

    # Read and verify output
    with open(output_file) as f:
        result = f.readlines()

    assert result == expected_lines


def test_bedtools_map_with_custom_tempdir(tmpdir):
    """Test bedtools_map with custom temporary directory location."""
    a_content = "chr1\t300\t400\tregion2\nchr1\t100\t200\tregion1\n"  # unsorted
    b_content = "chr1\t350\t360\tscore3\t30\nchr1\t150\t160\tscore1\t10\n"  # unsorted

    a_file = tmpdir.join("a.bed")
    a_file.write(a_content)

    b_file = tmpdir.join("b.bed")
    b_file.write(b_content)

    # Create custom temp directory
    custom_temp = tmpdir.join("custom_temp")
    custom_temp.mkdir()

    output_file = str(tmpdir.join("output.bed"))

    # Run with custom tempdir
    BedUtils().bedtools_map(
        a_bed=str(a_file),
        b_bed=str(b_file),
        output_bed=output_file,
        column=5,
        operation="mean",
        presort=True,
        tempdir_prefix=str(custom_temp),
    )

    # Read output
    with open(output_file) as f:
        result = f.readlines()

    # Verify results
    expected = [
        "chr1\t100\t200\tregion1\t10\n",
        "chr1\t300\t400\tregion2\t30\n",
    ]
    assert result == expected


def test_intersect_bed_files_simple(tmpdir):
    """Test the simple intersect_bed_files wrapper with two overlapping regions."""
    # Create first BED file
    bed1_content = "chr1\t100\t200\n" "chr1\t300\t400\n" "chr2\t500\t600\n"
    bed1_file = tmpdir.join("bed1.bed")
    bed1_file.write(bed1_content)

    # Create second BED file that overlaps with first
    bed2_content = "chr1\t150\t250\n" "chr1\t350\t450\n" "chr3\t700\t800\n"
    bed2_file = tmpdir.join("bed2.bed")
    bed2_file.write(bed2_content)

    output_file = str(tmpdir.join("output.bed"))

    # Run the intersection
    BedUtils().intersect_bed_files(input_bed1=str(bed1_file), input_bed2=str(bed2_file), bed_output=output_file)

    # Read and verify output - should only contain overlapping regions
    with open(output_file) as f:
        result = f.readlines()

    # Expected: intersection of the two BED files
    # chr1 100-200 intersects with chr1 150-250 = chr1 150-200
    # chr1 300-400 intersects with chr1 350-450 = chr1 350-400
    expected = ["chr1\t150\t200\n", "chr1\t350\t400\n"]
    assert result == expected


def test_intersect_bed_files_with_sorted_input(tmpdir):
    """Test intersect_bed_files with assume_input_sorted=True."""
    # Create sorted BED files
    bed1_content = "chr1\t100\t200\n" "chr1\t300\t400\n"
    bed1_file = tmpdir.join("bed1_sorted.bed")
    bed1_file.write(bed1_content)

    bed2_content = "chr1\t150\t250\n"
    bed2_file = tmpdir.join("bed2_sorted.bed")
    bed2_file.write(bed2_content)

    output_file = str(tmpdir.join("output.bed"))

    # Run with assume_input_sorted
    BedUtils().intersect_bed_files(
        input_bed1=str(bed1_file),
        input_bed2=str(bed2_file),
        bed_output=output_file,
        assume_input_sorted=True,
    )

    # Verify output exists
    with open(output_file) as f:
        result = f.readlines()

    expected = ["chr1\t150\t200\n"]
    assert result == expected


def test_intersect_bed_files_with_custom_tempdir(tmpdir):
    """Test intersect_bed_files with custom temporary directory."""
    bed1_content = "chr1\t100\t200\n"
    bed1_file = tmpdir.join("bed1.bed")
    bed1_file.write(bed1_content)

    bed2_content = "chr1\t150\t250\n"
    bed2_file = tmpdir.join("bed2.bed")
    bed2_file.write(bed2_content)

    # Create custom temp directory
    custom_temp = tmpdir.join("custom_temp")
    custom_temp.mkdir()

    output_file = str(tmpdir.join("output.bed"))

    # Run with custom tempdir
    BedUtils().intersect_bed_files(
        input_bed1=str(bed1_file),
        input_bed2=str(bed2_file),
        bed_output=output_file,
        tempdir_prefix=str(custom_temp),
    )

    # Verify output
    with open(output_file) as f:
        result = f.readlines()

    expected = ["chr1\t150\t200\n"]
    assert result == expected


def test_bedtools_sort(tmpdir):
    """Test bedtools_sort with unsorted BED file."""
    # Create unsorted BED file
    unsorted_content = (
        "chr2\t300\t400\tregion3\n" "chr1\t100\t200\tregion1\n" "chr1\t500\t600\tregion4\n" "chr1\t200\t300\tregion2\n"
    )
    input_file = tmpdir.join("unsorted.bed")
    input_file.write(unsorted_content)

    output_file = str(tmpdir.join("sorted.bed"))

    # Run bedtools_sort
    BedUtils().bedtools_sort(input_bed=str(input_file), output_bed=output_file)

    # Read and verify output is sorted
    with open(output_file) as f:
        result = f.readlines()

    # Expected: sorted by chromosome then by start position
    expected = [
        "chr1\t100\t200\tregion1\n",
        "chr1\t200\t300\tregion2\n",
        "chr1\t500\t600\tregion4\n",
        "chr2\t300\t400\tregion3\n",
    ]
    assert result == expected


def test_bedtools_coverage(tmpdir):
    """Test bedtools_coverage with basic coverage calculation."""
    # Create BED file A (regions to compute coverage for)
    a_content = "chr1\t100\t200\tregion1\n" "chr1\t300\t400\tregion2\n" "chr2\t500\t600\tregion3\n"
    a_file = tmpdir.join("a.bed")
    a_file.write(a_content)

    # Create BED file B (intervals for coverage)
    # region1 has 2 overlaps covering 30 bases out of 100
    # region2 has 1 overlap covering 20 bases out of 100
    # region3 has no overlaps
    b_content = "chr1\t120\t140\tfeature1\n" "chr1\t160\t170\tfeature2\n" "chr1\t350\t370\tfeature3\n"
    b_file = tmpdir.join("b.bed")
    b_file.write(b_content)

    output_file = str(tmpdir.join("output.bed"))

    # Run bedtools_coverage
    BedUtils().bedtools_coverage(a_bed=str(a_file), b_bed=str(b_file), output_bed=output_file)

    # Read and verify output
    with open(output_file) as f:
        result = f.readlines()

    # Expected columns: original A columns + count + covered_bases + length + fraction
    # region1: 2 overlaps, 30 bases covered, 100 bases total, 0.3 fraction
    # region2: 1 overlap, 20 bases covered, 100 bases total, 0.2 fraction
    # region3: 0 overlaps, 0 bases covered, 100 bases total, 0.0 fraction
    expected = [
        "chr1\t100\t200\tregion1\t2\t30\t100\t0.3000000\n",
        "chr1\t300\t400\tregion2\t1\t20\t100\t0.2000000\n",
        "chr2\t500\t600\tregion3\t0\t0\t100\t0.0000000\n",
    ]
    assert result == expected
