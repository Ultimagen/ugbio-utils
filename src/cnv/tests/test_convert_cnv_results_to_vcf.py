import filecmp
import gzip
import shutil
import warnings
from os.path import join as pjoin
from pathlib import Path

import pytest

warnings.filterwarnings('ignore')

from ugbio_cnv import convert_cnv_results_to_vcf


@pytest.fixture
def resources_dir():
    return Path(__file__).parent / "resources"


def unzip_file(zipped_file_name):
    with gzip.open(zipped_file_name, 'rb') as f_in:
        out_file_name = zipped_file_name.rstrip('.gz')
        with open(out_file_name, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
    return out_file_name


def compare_zipped_files(a, b):
    a_unzipped = unzip_file(a)
    b_unzipped = unzip_file(b)
    assert filecmp.cmp(a_unzipped, b_unzipped)


class TestConvertCnvResultsToVcf:
    def test_convert_cnv_results_to_vcf(self, tmpdir, resources_dir):
        input_bed_file = pjoin(resources_dir, "EL-0059.cnvs.annotate.bed")
        genome_file = pjoin(resources_dir, "Homo_sapiens_assembly38.chr1-24.genome")
        expected_out_vcf = pjoin(resources_dir, 'EL-0059.cnv.vcf.gz')

        sample_name = 'EL-0059'
        out_dir = f"{tmpdir}"
        convert_cnv_results_to_vcf.run([
            "convert_cnv_results_to_vcf",
            "--cnv_annotated_bed_file",
            input_bed_file,
            "--out_directory",
            out_dir,
            "--sample_name",
            sample_name,
            "--fasta_index_file",
            genome_file
        ])

        out_vcf_file = pjoin(tmpdir, sample_name + '.cnv.vcf.gz')
        compare_zipped_files(out_vcf_file, expected_out_vcf)
