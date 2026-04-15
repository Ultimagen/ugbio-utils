import importlib
import sys

import pytest


def test_calculate_library_complexity_can_be_imported_without_cli_args(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["pytest"])
    sys.modules.pop("ugbio_core.calculate_library_complexity", None)
    module = importlib.import_module("ugbio_core.calculate_library_complexity")
    assert hasattr(module, "run")


@pytest.mark.parametrize(("n", "c"), [(0, 0), (10, 0)])
def test_estimate_library_size_handles_zero_counts(n: int, c: int):
    module = importlib.import_module("ugbio_core.calculate_library_complexity")
    assert module.estimate_library_size(n, c) == 0.0
