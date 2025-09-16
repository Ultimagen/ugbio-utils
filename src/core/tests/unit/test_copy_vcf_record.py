import os
import tempfile

import pysam
import pytest
from ugbio_core.vcf_utils import VcfUtils


def create_test_vcf(path):
    header = (
        "##fileformat=VCFv4.2\n"
        '##INFO=<ID=DP,Number=1,Type=Integer,Description="Total Depth">\n'
        '##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">\n'
        "##contig=<ID=1,length=249250621>\n"
        "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tsample1\n"
    )
    record = "1\t1000\trs1\tA\tG\t50\tPASS\tDP=10\tGT\t0/1\n"
    with open(path, "w") as f:
        f.write(header)
        f.write(record)


def test_copy_vcf_record():
    with tempfile.TemporaryDirectory() as tmpdir:
        vcf_path = os.path.join(tmpdir, "test.vcf")
        create_test_vcf(vcf_path)
        with pysam.VariantFile(vcf_path) as vcf:
            header = vcf.header.copy()
            rec = next(iter(vcf))
            # Use the static method to copy the record
            new_rec = VcfUtils.copy_vcf_record(rec, header)
            # Check that the new record matches the original
            assert new_rec.chrom == rec.chrom
            assert new_rec.pos == rec.pos
            assert new_rec.ref == rec.ref
            assert new_rec.alts == rec.alts
            assert new_rec.qual == rec.qual
            assert new_rec.id == rec.id
            assert dict(new_rec.info) == dict(rec.info)
            assert new_rec.samples[0]["GT"] == rec.samples[0]["GT"]


if __name__ == "__main__":
    pytest.main([__file__])
