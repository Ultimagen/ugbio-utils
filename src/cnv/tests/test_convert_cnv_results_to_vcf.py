import subprocess
from os.path import join as pjoin
import hashlib
import warnings
import gzip
import shutil
import filecmp

warnings.filterwarnings('ignore')

from . import get_resource_dir

from ugbio_cnv import convert_cnv_results_to_vcf


def unzip_file(zipped_file_name):
    with gzip.open(zipped_file_name, 'rb') as f_in:
        out_file_name = zipped_file_name.rstrip('.gz')
        with open(out_file_name, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
    return out_file_name

def compare_zipped_files(a, b):
    # fileA = hashlib.sha256(open(a, 'rb').read()).digest()
    # fileB = hashlib.sha256(open(b, 'rb').read()).digest()
    # if fileA == fileB:
    #     return True
    # else:
    #     return False
    a_unzipped= unzip_file(a)
    b_unzipped = unzip_file(b)
    assert filecmp.cmp(a_unzipped, b_unzipped)


class TestConvertCnvResultsToVcf:
    inputs_dir = get_resource_dir(__file__)

    def test_convert_cnv_results_to_vcf(self, tmpdir):
        input_bed_file = pjoin(self.inputs_dir, "EL-0059.cnvs.annotate.bed")
        genome_file = pjoin(self.inputs_dir, "Homo_sapiens_assembly38.chr1-24.genome")
        expected_out_vcf = pjoin(self.inputs_dir, 'EL-0059.cnv.vcf.gz')

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

        # cmd = [
        #     "bcftools",
        #     "view",
        #     expected_out_vcf
        # ]
        # assert subprocess.check_call(cmd, cwd=tmpdir) == 0
        #
        # cmd1 = [
        #     "bcftools",
        #     "view",
        #     out_vcf_file
        # ]
        # assert subprocess.check_call(cmd1, cwd=tmpdir) == 0

        assert compare_zipped_files(out_vcf_file, expected_out_vcf)
