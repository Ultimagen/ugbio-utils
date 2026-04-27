import re
from pathlib import Path

import pytest
from ugbio_core.calculate_library_complexity import run


@pytest.fixture
def csv_file():
    """Return path to the sorter CSV test file."""
    resources_dir = Path(__file__).parent.parent / "resources"
    return str(resources_dir / "026532-Lb_1866-Z0058-CATCTCAGTGCAATGAT.csv")


def parse_output_for_x(captured_output: str) -> int:
    """Extract the estimated library size X from stdout."""
    match = re.search(r"Estimated library size X = (\d+)", captured_output)
    if not match:
        raise ValueError("Could not find 'Estimated library size X' in output")
    return int(match.group(1))


class TestCalculateLibraryComplexity:
    """End-to-end tests for library complexity calculation."""

    # Expected values from CSV file
    PF_BARCODE_READS = 122191864
    PCT_PF_READS_ALIGNED = 99.85
    PCT_DUPLICATION = 7.72
    EXPECTED_X = 748995462

    def test_csv_mode(self, csv_file, capsys):
        """Test library complexity calculation from CSV file."""
        argv = ["calculate_library_complexity", "--csv", csv_file]
        run(argv)

        captured = capsys.readouterr()
        output = captured.out

        # Verify output contains expected messages
        assert "Processing CSV:" in output
        assert "Converged at iteration" in output

        # Parse and verify X
        x = parse_output_for_x(output)
        assert x == self.EXPECTED_X

    def test_direct_pf_metrics_mode(self, capsys):
        """Test library complexity calculation from direct PF metrics."""
        argv = [
            "calculate_library_complexity",
            "--PF_Barcode_reads",
            str(self.PF_BARCODE_READS),
            "--PCT_PF_Reads_aligned",
            str(self.PCT_PF_READS_ALIGNED),
            "--pct_duplication",
            str(self.PCT_DUPLICATION),
        ]
        run(argv)

        captured = capsys.readouterr()
        output = captured.out

        # Verify mode message
        assert "Using PF metrics to compute N and C" in output

        # Parse and verify X
        x = parse_output_for_x(output)
        assert x == self.EXPECTED_X

    def test_csv_and_direct_modes_equivalence(self, csv_file, capsys):
        """Verify CSV and direct PF metrics modes produce identical results."""
        # Run CSV mode
        argv_csv = ["calculate_library_complexity", "--csv", csv_file]
        run(argv_csv)
        x_csv = parse_output_for_x(capsys.readouterr().out)

        # Run direct mode with same values
        argv_direct = [
            "calculate_library_complexity",
            "--PF_Barcode_reads",
            str(self.PF_BARCODE_READS),
            "--PCT_PF_Reads_aligned",
            str(self.PCT_PF_READS_ALIGNED),
            "--pct_duplication",
            str(self.PCT_DUPLICATION),
        ]
        run(argv_direct)
        x_direct = parse_output_for_x(capsys.readouterr().out)

        # Both modes should produce identical results
        assert x_csv == x_direct == self.EXPECTED_X
