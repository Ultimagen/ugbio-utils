from pathlib import Path

import pytest
from ugbio_core import cram_utils


@pytest.fixture
def inputs_dir():
    inputs_dir = Path(__file__).parent.parent / "resources"
    return inputs_dir


@pytest.fixture
def chr9_bam(inputs_dir):
    return str(inputs_dir / "chr9.sample.bam")


@pytest.fixture
def trimmed_bam(inputs_dir):
    return str(inputs_dir / "trimmed_read.bam")


@pytest.fixture
def trimmed_bam_copy(inputs_dir):
    return str(inputs_dir / "trimmed_read.copy.bam")


def test_check_cram_samples_identical_success(chr9_bam):
    """Test that identical samples across files pass when require_unique=False."""
    # Should not raise when using the same file twice (same sample)
    cram_utils.check_cram_samples(
        [chr9_bam, chr9_bam],
        require_unique=False,
        raise_on_failure=True,
    )


def test_check_cram_samples_identical_failure(chr9_bam, trimmed_bam):
    """Test that different samples fail when require_unique=False."""
    with pytest.raises(ValueError, match="Multiple sample names detected"):
        cram_utils.check_cram_samples(
            [chr9_bam, trimmed_bam],
            require_unique=False,
            raise_on_failure=True,
        )


def test_check_cram_samples_unique_success(chr9_bam, trimmed_bam):
    """Test that unique samples across files pass when require_unique=True."""
    # Should not raise when files have different samples
    cram_utils.check_cram_samples(
        [chr9_bam, trimmed_bam],
        require_unique=True,
        raise_on_failure=True,
    )


def test_check_cram_samples_unique_failure(trimmed_bam, trimmed_bam_copy):
    """Test that duplicate samples fail when require_unique=True."""
    # Using two different files with the same sample name
    with pytest.raises(ValueError, match="Duplicate sample names detected"):
        cram_utils.check_cram_samples(
            [trimmed_bam, trimmed_bam_copy],
            require_unique=True,
            raise_on_failure=True,
        )


def test_check_cram_samples_no_raise(chr9_bam, trimmed_bam, capsys):
    """Test that errors are printed instead of raised when raise_on_failure=False."""
    # Should not raise, but print error
    cram_utils.check_cram_samples(
        [chr9_bam, trimmed_bam],
        require_unique=False,
        raise_on_failure=False,
    )

    captured = capsys.readouterr()
    assert "Multiple sample names detected" in captured.out


def test_parse_args_require_unique():
    """Test command-line argument parsing for --require-unique."""
    args = cram_utils._parse_args(["prog", "--require-unique", "file1.cram", "file2.cram"])
    assert args.require_unique is True
    assert args.require_identical is False


def test_parse_args_require_identical():
    """Test command-line argument parsing for --require-identical."""
    args = cram_utils._parse_args(["prog", "--require-identical", "file1.cram"])
    assert args.require_unique is False
    assert args.require_identical is True


def test_parse_args_default():
    """Test default behavior (require identical)."""
    args = cram_utils._parse_args(["prog", "file1.cram"])
    assert args.require_unique is False
    assert args.require_identical is False  # Default, not explicitly set
