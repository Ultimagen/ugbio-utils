from pathlib import Path

import pysam
import pytest
from ugbio_filtering.filter_low_af_ratio_to_background import filter_low_af_ratio_to_background


@pytest.fixture
def example_vcf(tmp_path: Path):
    vcf_content = """##fileformat=VCFv4.2
##FILTER=<ID=RefCall,Description="Reference call">
##INFO=<ID=VARIANT_TYPE,Number=1,Type=String,Description="Type of variant">
##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">
##FORMAT=<ID=AD,Number=R,Type=Integer,Description="Allelic depths">
##FORMAT=<ID=DP,Number=1,Type=Integer,Description="Read depth">
##FORMAT=<ID=BG_AD,Number=R,Type=Integer,Description="Background allelic depths">
##FORMAT=<ID=BG_DP,Number=1,Type=Integer,Description="Background read depth">
##FORMAT=<ID=VAF,Number=A,Type=Float,Description="Variant allele fractions">
##FORMAT=<ID=BG_VAF,Number=A,Type=Float,Description="Background variant allele fractions">
##contig=<ID=chr1,length=248956422>
#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tsample1
chr1\t100\t.\tA\tT\t.\tPASS\tVARIANT_TYPE=snp\tGT:AD:DP:BG_AD:BG_DP:VAF:BG_VAF\t0/1:5,10:15:1,1:2:0.6667:0.5
chr1\t200\t.\tG\tC\t.\tPASS\tVARIANT_TYPE=snp\tGT:AD:DP:BG_AD:BG_DP:VAF:BG_VAF\t0/1:2,2:4:1,1:2:0.5:0.5
chr1\t300\t.\tT\tG\t.\tRefCall\tVARIANT_TYPE=snp\tGT:AD:DP:BG_AD:BG_DP:VAF:BG_VAF\t0/1:10,10:20:1,1:2:0.5:0.5
chr1\t400\t.\tCTA\tC\t.\tPASS\tVARIANT_TYPE=non-h-indel\tGT:AD:DP:BG_AD:BG_DP:VAF:BG_VAF\t0/1:10,10:20:2,1:3:0.5:0.33333
chr1\t500\t.\tC\tCA\t.\tPASS\tVARIANT_TYPE=h-indel\tGT:AD:DP:BG_AD:BG_DP:VAF:BG_VAF\t0/1:10,90:100:9,1:10:0.9:0.1
chr1\t600\t.\tC\tCTA,T\t.\tPASS\tVARIANT_TYPE=non-h-indel\tGT:AD:DP:BG_AD:BG_DP:VAF:BG_VAF\t0/2:10,30,10:50:10,10,1:21:0.6,0.2:0.47619,0.04762
chr1\t700\t.\tC\tCTA,T\t.\tPASS\tVARIANT_TYPE=non-h-indel\tGT:AD:DP:BG_AD:BG_DP:VAF:BG_VAF\t0/2:10,30,100:140:10,10,1:21:0.21428,0.71428:0.47619,0.04762
chr1\t700\t.\tC\tCTA,T\t.\tPASS\tVARIANT_TYPE=non-h-indel\tGT:AD:DP:BG_AD:BG_DP:VAF:BG_VAF\t0/1:10,30,100:140:10,10,1:21:0.21428,0.71428:0.47619,0.04762
chr1\t800\t.\tC\tCA,T\t.\tPASS\tVARIANT_TYPE=h-indel\tGT:AD:DP:BG_AD:BG_DP:VAF:BG_VAF\t0/1:10,30,100:140:10,30,100:140:0.21428,0.71428:0.21428,0.71428
chr1\t900\t.\tC\tCA\t.\tPASS\tVARIANT_TYPE=h-indel\tGT:AD:DP:BG_AD:BG_DP:VAF:BG_VAF\t0/1:1,10:11:1,10:11:0.0909:0.90909
"""
    vcf_path = tmp_path / "input.vcf.gz"
    with open(vcf_path, "w") as f:
        f.write(vcf_content)
    return str(vcf_path)


def test_filter_low_af_ratio_to_background_basic(example_vcf: str, tmp_path: Path):
    output_vcf = tmp_path / "output.vcf.gz"
    filter_low_af_ratio_to_background(
        input_vcf=example_vcf,
        output_vcf=str(output_vcf),
        af_ratio_threshold=10,
        af_ratio_threshold_h_indels=2,
        t_vaf_threshold=0.15,
        new_filter="LowAFRatioToBackground",
    )
    with pysam.VariantFile(str(output_vcf)) as vcf:
        records = list(vcf.fetch())
        # First record: VARIANT_TYPE=snp, AF ratio = 0.6667/0.5 = 1.333 < 10,
        # should be filtered
        assert "LowAFRatioToBackground" in records[0].filter.keys()
        # Second record: VARIANT_TYPE=snp, AF ratio = 0.5/0.5 = 1 < 10,
        # should be filtered
        assert "LowAFRatioToBackground" in records[1].filter.keys()
        # Third record: VARIANT_TYPE=snp, FILTER=RefCall,
        # should not be filtered
        assert "LowAFRatioToBackground" not in records[2].filter.keys()
        assert "RefCall" in records[2].filter.keys()
        # Fourth record: VARIANT_TYPE=non-h-indel, AF ratio = 0.5/0.33333 = 1.5 < 10,
        # should be filtered
        assert "LowAFRatioToBackground" in records[3].filter.keys()
        # Fifth record: VARIANT_TYPE=h-indel, AF ratio = 0.9/0.1 = 9 > 2, VAF > 0.15,
        # should not be filtered
        assert "LowAFRatioToBackground" not in records[4].filter.keys()
        # Sixth record: VARIANT_TYPE=non-h-indel, AF ratio in alt allele 2 = 0.2/0.04762 = 4.2 < 10,
        # should be filtered
        assert "LowAFRatioToBackground" in records[5].filter.keys()
        # Seventh record: VARIANT_TYPE=non-h-indel, AF ratio in alt allele 2 = 0.71428/0.04762 = 15 > 10,
        # should not be filtered
        assert "LowAFRatioToBackground" not in records[6].filter.keys()
        # Eighth record: VARIANT_TYPE=non-h-indel, AF ratio in alt allele 1 = 0.21428/0.47619 = 0.45 < 10,
        # should be filtered
        assert "LowAFRatioToBackground" in records[7].filter.keys()
        # Ninth record: VARIANT_TYPE=h-indel, AF ratio in alt allele 1 = 0.21428/0.21428 = 1 < 2, VAF > 0.15,
        # should not be filtered
        assert "LowAFRatioToBackground" not in records[8].filter.keys()
        # Tenth record: VARIANT_TYPE=h-indel, AF ratio = 0.0909/0.90909 = 0.1 < 2, VAF < 0.15,
        # should be filtered
        assert "LowAFRatioToBackground" in records[9].filter.keys()


def test_filter_low_af_ratio_to_background_no_fail(example_vcf: str, tmp_path: Path):
    output_vcf = tmp_path / "output2.vcf.gz"
    # Use a low threshold so nothing is filtered
    filter_low_af_ratio_to_background(
        input_vcf=example_vcf,
        output_vcf=str(output_vcf),
        af_ratio_threshold=0.05,
        af_ratio_threshold_h_indels=0.05,
        t_vaf_threshold=1,
        new_filter="LowAFRatioToBackground",
    )
    with pysam.VariantFile(str(output_vcf)) as vcf:
        for rec in vcf.fetch():
            assert "LowAFRatioToBackground" not in rec.filter.keys()


def test_filter_low_af_ratio_to_background_custom_filter(example_vcf: str, tmp_path: Path):
    output_vcf = tmp_path / "output3.vcf.gz"
    filter_low_af_ratio_to_background(
        input_vcf=example_vcf,
        output_vcf=str(output_vcf),
        af_ratio_threshold=10,
        af_ratio_threshold_h_indels=2,
        t_vaf_threshold=0.5,
        new_filter="MyCustomFilter",
    )
    with pysam.VariantFile(str(output_vcf)) as vcf:
        records = list(vcf.fetch())
        assert "MyCustomFilter" in records[0].filter.keys()
        assert "MyCustomFilter" in records[1].filter.keys()
