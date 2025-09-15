from io import StringIO
from os.path import exists
from os.path import join as pjoin

import numpy as np
import pytest
from ugbio_core.misc_utils import BufferedFileIterator, find_scripts_path, idx_last_nz, idx_next_nz


class TestMiscUtils:
    def test_find_scripts_path(self):
        assert exists(pjoin(find_scripts_path(), "run_ucsc_command.sh"))

    inputs = [[1, 0, 1, 0, 1], [1, 0, 0, 2, 0, 5], [1, 0, 0, 0, 0, 2, 5], [0, 0, 0, 1, 2, 3, 0, 5], [2, 0, 1, 0, 0]]

    @pytest.mark.parametrize(
        "inp,expected",
        zip(
            inputs,
            [[0, 0, 2, 2, 4], [0, 0, 0, 3, 3, 5], [0, 0, 0, 0, 0, 5, 6], [-1, -1, -1, 3, 4, 5, 5, 7], [0, 0, 2, 2, 2]],
        ),
    )
    def test_idx_last_nz(self, inp, expected):
        assert np.all(idx_last_nz(inp) == expected)

    @pytest.mark.parametrize(
        "inp,expected",
        zip(
            inputs,
            [[0, 2, 2, 4, 4], [0, 3, 3, 3, 5, 5], [0, 5, 5, 5, 5, 5, 6], [3, 3, 3, 3, 4, 5, 7, 7], [0, 2, 2, 5, 5]],
        ),
    )
    def test_idx_next_nz(self, inp, expected):
        assert np.all(idx_next_nz(inp) == expected)


class TestBufferedFileIterator:
    def test_basic_iteration_without_parse_func(self):
        """Test basic iteration without a parsing function."""
        test_data = "line1\nline2\nline3\nline4\nline5\n"
        file_obj = StringIO(test_data)

        iterator = BufferedFileIterator(file_obj, window_size=3)

        results = []
        for item in iterator:
            results.append(item)

        expected = ["line1", "line2", "line3", "line4", "line5"]
        assert results == expected

        # Check final buffer state
        buffer = iterator.get_buffer()
        assert list(buffer) == ["line3", "line4", "line5"]
        assert len(iterator) == 3

    def test_basic_iteration_with_parse_func(self):
        """Test iteration with a parsing function."""
        test_data = "1\n2\n3\n4\n5\n"
        file_obj = StringIO(test_data)

        def parse_int(line):
            return int(line.strip())

        iterator = BufferedFileIterator(file_obj, window_size=3, parse_func=parse_int)

        results = []
        for item in iterator:
            results.append(item)

        expected = [1, 2, 3, 4, 5]
        assert results == expected

        # Check final buffer state
        buffer = iterator.get_buffer()
        assert list(buffer) == [3, 4, 5]

    def test_buffer_size_limit(self):
        """Test that buffer respects the window_size limit."""
        test_data = "a\nb\nc\nd\ne\nf\ng\n"
        file_obj = StringIO(test_data)

        iterator = BufferedFileIterator(file_obj, window_size=3)

        # Consume all items
        _ = list(iterator)

        # Buffer should only contain the last 3 items
        buffer = iterator.get_buffer()
        assert list(buffer) == ["e", "f", "g"]
        assert len(buffer) == 3

    def test_buffer_smaller_than_window_size(self):
        """Test behavior when file has fewer lines than window_size."""
        test_data = "line1\nline2\n"
        file_obj = StringIO(test_data)

        iterator = BufferedFileIterator(file_obj, window_size=5)

        items = list(iterator)
        assert items == ["line1", "line2"]

        buffer = iterator.get_buffer()
        assert list(buffer) == ["line1", "line2"]
        assert len(buffer) == 2

    def test_empty_file(self):
        """Test behavior with empty file."""
        file_obj = StringIO("")

        iterator = BufferedFileIterator(file_obj, window_size=3)

        items = list(iterator)
        assert items == []

        buffer = iterator.get_buffer()
        assert list(buffer) == []
        assert len(buffer) == 0

    def test_clear_buffer(self):
        """Test buffer clearing functionality."""
        test_data = "a\nb\nc\nd\n"
        file_obj = StringIO(test_data)

        iterator = BufferedFileIterator(file_obj, window_size=3)

        # Consume some items
        next(iterator)  # a
        next(iterator)  # b
        next(iterator)  # c

        assert len(iterator) == 3

        # Clear buffer
        iterator.clear_buffer()
        assert len(iterator) == 0
        assert list(iterator.get_buffer()) == []

        # Continue iteration - buffer should start filling again
        next(iterator)  # d
        assert len(iterator) == 1

    def test_current_property(self):
        """Test the current property returns the most recent item."""
        test_data = "first\nsecond\nthird\n"
        file_obj = StringIO(test_data)

        iterator = BufferedFileIterator(file_obj, window_size=2)

        assert iterator.current is None

        item1 = next(iterator)
        assert iterator.current == "first"
        assert item1 == "first"

        item2 = next(iterator)
        assert iterator.current == "second"
        assert item2 == "second"

    def test_parse_function_with_complex_data(self):
        """Test parsing function with more complex data structure."""
        # Simulate mpileup-like data
        test_data = "chr1\t100\tA\t10\t5\nchr1\t101\tT\t8\t3\nchr2\t200\tG\t12\t7\n"
        file_obj = StringIO(test_data)

        def parse_mpileup_like(line):
            fields = line.strip().split("\t")
            return (fields[0], int(fields[1]), int(fields[3]), int(fields[4]))

        iterator = BufferedFileIterator(file_obj, window_size=2, parse_func=parse_mpileup_like)

        results = list(iterator)
        expected = [("chr1", 100, 10, 5), ("chr1", 101, 8, 3), ("chr2", 200, 12, 7)]
        assert results == expected

        # Check buffer contains parsed data
        buffer = iterator.get_buffer()
        assert list(buffer) == [("chr1", 101, 8, 3), ("chr2", 200, 12, 7)]

    def test_window_size_one(self):
        """Test behavior with window_size=1."""
        test_data = "a\nb\nc\n"
        file_obj = StringIO(test_data)

        iterator = BufferedFileIterator(file_obj, window_size=1)

        next(iterator)  # a
        assert list(iterator.get_buffer()) == ["a"]

        next(iterator)  # b
        assert list(iterator.get_buffer()) == ["b"]  # Should only keep the most recent

        next(iterator)  # c
        assert list(iterator.get_buffer()) == ["c"]

    def test_iteration_protocol(self):
        """Test that the iterator properly implements the iteration protocol."""
        test_data = "1\n2\n3\n"
        file_obj = StringIO(test_data)

        iterator = BufferedFileIterator(file_obj, window_size=2)

        # Test __iter__ returns self
        assert iter(iterator) is iterator

        # Test StopIteration is raised when exhausted
        items = []
        try:
            while True:
                items.append(next(iterator))
        except StopIteration:
            pass

        assert items == ["1", "2", "3"]

    def test_parse_function_exception_handling(self):
        """Test behavior when parse function raises an exception."""
        test_data = "1\ninvalid\n3\n"
        file_obj = StringIO(test_data)

        def parse_int_strict(line):
            return int(line.strip())

        iterator = BufferedFileIterator(file_obj, window_size=2, parse_func=parse_int_strict)

        assert next(iterator) == 1

        # This should raise ValueError due to "invalid"
        with pytest.raises(ValueError):
            next(iterator)
