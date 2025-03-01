import numpy as np
import pytest
from ugbio_core.array_utils import idx_last_nz, idx_next_nz


class TestArrayUtils:
    inputs = [[1, 0, 1, 0, 1], [1, 0, 0, 2, 0, 5], [1, 0, 0, 0, 0, 2, 5], [0, 0, 0, 1, 2, 3, 0, 5], [2, 0, 1, 0, 0]]

    @pytest.mark.parametrize(
        "inp,expected",
        zip(
            inputs,
            [[0, 0, 2, 2, 4], [0, 0, 0, 3, 3, 5], [0, 0, 0, 0, 0, 5, 6], [-1, -1, -1, 3, 4, 5, 5, 7], [0, 0, 2, 2, 2]],
            strict=False,
        ),
    )
    def test_idx_last_nz(self, inp, expected):
        assert np.all(idx_last_nz(inp) == expected)

    @pytest.mark.parametrize(
        "inp,expected",
        zip(
            inputs,
            [[0, 2, 2, 4, 4], [0, 3, 3, 3, 5, 5], [0, 5, 5, 5, 5, 5, 6], [3, 3, 3, 3, 4, 5, 7, 7], [0, 2, 2, 5, 5]],
            strict=False,
        ),
    )
    def test_idx_next_nz(self, inp, expected):
        assert np.all(idx_next_nz(inp) == expected)
