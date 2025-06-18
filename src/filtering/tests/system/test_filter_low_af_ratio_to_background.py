import os
import tempfile
import pytest
import pysam
from ugbio_filtering.filter_low_af_ratio_to_background import filter_low_af_ratio_to_background

@pytest.fixture
def example_vcf(tmp_path):
    vcf_content = """##fileformat=VCFv4.2
##FILTER=<ID=RefCall,Description="Reference call">
##INFO=<ID=VARIANT_TYPE,Number=1,Type=String,Description="Type of variant">
##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">
##FORMAT=<ID=AD,Number=R,Type=Integer,Description="Allelic depths">
##FORMAT=<ID=DP,Number=1,Type=Integer,Description="Read depth">
##FORMAT=<ID=BG_AD,Number=R,Type=Integer,Description="Background allelic depths">
##FORMAT=<ID=BG_DP,Number=1,Type=Integer,Description="Background read depth">
##contig=<ID=chr1,length=248956422>
#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tsample1
chr1\t100\t.\tA\tT\t.\tPASS\tVARIANT_TYPE=snp\tGT:AD:DP:BG_AD:BG_DP\t0/1:5,10:15:1,1:2
chr1\t200\t.\tG\tC\t.\tPASS\tVARIANT_TYPE=snp\tGT:AD:DP:BG_AD:BG_DP\t0/1:2,2:4:1,1:2
chr1\t300\t.\tT\tG\t.\tRefCall\tVARIANT_TYPE=snp\tGT:AD:DP:BG_AD:BG_DP\t0/1:10,10:20:1,1:2
chr1\t400\t.\tC\tA\t.\tPASS\tVARIANT_TYPE=h-indel\tGT:AD:DP:BG_AD:BG_DP\t0/1:10,10:20:1,1:2
chr1\t400\t.\tC\tA,T\t.\tPASS\tVARIANT_TYPE=non-h-indel\tGT:AD:DP:BG_AD:BG_DP\t0/2:10,30,10:50:10,10,1:21
chr1\t400\t.\tC\tA,T\t.\tPASS\tVARIANT_TYPE=non-h-indel\tGT:AD:DP:BG_AD:BG_DP\t0/2:10,30,100:140:10,10,1:21
"""
    vcf_path = tmp_path / "input.vcf"
    with open(vcf_path, "w") as f:
        f.write(vcf_content)
    return str(vcf_path)

def test_filter_low_af_ratio_to_background_basic(example_vcf, tmp_path):
    output_vcf = tmp_path / "output.vcf"
    filter_low_af_ratio_to_background(
        input_vcf=example_vcf,
        output_vcf=str(output_vcf),
        af_ratio_threshold=10,
        new_filter="LowAFRatioToBackground"
    )
    with pysam.VariantFile(str(output_vcf)) as vcf:
        records = list(vcf.fetch())
        # First record: AF ratio = (10/15)/(1/2) = (0.6667)/(0.5) = 1.333 < 10, should be filtered
        assert "LowAFRatioToBackground" in records[0].filter.keys()
        # Second record: AF ratio = (2/4)/(1/2) = 0.5/0.5 = 1 < 10, should be filtered
        assert "LowAFRatioToBackground" in records[1].filter.keys()
        # Third record: FILTER=RefCall, should not be filtered
        assert "LowAFRatioToBackground" not in records[2].filter.keys()
        assert "RefCall" in records[2].filter.keys()
        # Fourth record: VARIANT_TYPE=h-indel, should not be filtered
        assert "LowAFRatioToBackground" not in records[3].filter.keys()
        # Fifth record: AF ratio in alt allele 2 = (10/50)/(1/21) = 4.2 < 10, should be filtered
        assert "LowAFRatioToBackground" in records[4].filter.keys()
        # Sixth record: AF ratio in alt allele 2 = (100/140)/(1/21) = 15 > 10, should not be filtered
        assert "LowAFRatioToBackground" not in records[5].filter.keys()

def test_filter_low_af_ratio_to_background_no_fail(example_vcf, tmp_path):
    output_vcf = tmp_path / "output2.vcf"
    # Use a low threshold so nothing is filtered
    filter_low_af_ratio_to_background(
        input_vcf=example_vcf,
        output_vcf=str(output_vcf),
        af_ratio_threshold=0.05,
        new_filter="LowAFRatioToBackground"
    )
    with pysam.VariantFile(str(output_vcf)) as vcf:
        for rec in vcf.fetch():
            assert "LowAFRatioToBackground" not in rec.filter.keys()

def test_filter_low_af_ratio_to_background_custom_filter(example_vcf, tmp_path):
    output_vcf = tmp_path / "output3.vcf"
    filter_low_af_ratio_to_background(
        input_vcf=example_vcf,
        output_vcf=str(output_vcf),
        af_ratio_threshold=10,
        new_filter="MyCustomFilter"
    )
    with pysam.VariantFile(str(output_vcf)) as vcf:
        records = list(vcf.fetch())
        assert "MyCustomFilter" in records[0].filter.keys()
        assert "MyCustomFilter" in records[1].filter.keys()