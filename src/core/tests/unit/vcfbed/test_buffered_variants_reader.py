import unittest
from pathlib import Path

import pytest
from ugbio_core.vcfbed.buffered_variant_reader import BufferedVariantReader


@pytest.fixture
def resources_dir():
    return Path(__file__).parent.parent.parent / "resources"


class TestBufferedVariantReader(unittest.TestCase):
    def test_get_variant(self, resources_dir):
        reader = BufferedVariantReader(f"{resources_dir}/single_sample_example.vcf.gz")
        variant_1 = reader.get_variant("chr1", 930196)
        self.assertEqual(("T", "TG", "<NON_REF>"), variant_1.alleles)
        variant_2 = reader.get_variant("chr1", 1044019)
        self.assertEqual(("G", "GC", "<NON_REF>"), variant_2.alleles)
        variant_3 = reader.get_variant("chr1", 10)
        self.assertIsNone(variant_3)

    def test_header(self, resources_dir):
        reader = BufferedVariantReader(f"{resources_dir}/single_sample_example.vcf.gz")
        self.assertEqual("HG00239", reader.header.samples[0])
