import filecmp
from os.path import join as pjoin
from pathlib import Path

import pytest
from ugbio_cnv.reformat_parascopy_bed import run


@pytest.fixture
def resources_dir():
    return Path(__file__).parent.parent / "resources"


class TestReformatParascopyBed:
    def test_cli_samples_file(self, tmpdir, resources_dir):
        """Test end-to-end conversion with real samples file, verifying info column expansion."""
        input_file = pjoin(resources_dir, "res.samples.bed.gz")
        expected_output = pjoin(resources_dir, "res.samples.expected.bed")
        output_file = pjoin(tmpdir, "output.bed")

        # Run conversion
        run(["prog", "--input_bed", input_file, "--output_bed", output_file])

        # Verify output matches expected
        assert filecmp.cmp(output_file, expected_output), "Output does not match expected"

        # Verify info column was expanded (not nested)
        with open(output_file) as f:
            lines = f.readlines()
            # Check first data line (line 2)
            first_data_line = lines[1]
            assert "group=02-01" in first_data_line, "Info tags should be expanded to top-level"
            assert "n_windows=165" in first_data_line, "Info tags should be expanded to top-level"
            assert ";info=" not in first_data_line, "Info column should not appear as its own tag"
            assert "psCN=2,2" in first_data_line, "Commas should be preserved in values"

    def test_cli_paralog_file(self, tmpdir, resources_dir):
        """Test end-to-end conversion with real paralog file."""
        input_file = pjoin(resources_dir, "res.paralog.bed.gz")
        expected_output = pjoin(resources_dir, "res.paralog.expected.bed")
        output_file = pjoin(tmpdir, "output.bed")

        # Run conversion
        run(["prog", "--input_bed", input_file, "--output_bed", output_file])

        # Verify output matches expected
        assert filecmp.cmp(output_file, expected_output), "Output does not match expected"
