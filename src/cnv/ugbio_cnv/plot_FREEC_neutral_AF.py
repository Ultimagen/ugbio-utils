import pandas as pd
import numpy as np
import os
from os.path import join as pjoin
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
import argparse
import logging
import sys
import seaborn as sns
from ugbio_core.logger import logger
from ugbio_core.pileuptofreq import pileup_to_freq
import warnings
warnings.filterwarnings('ignore')

sns.set_context("talk")
sns.set_style("white")

def mpileup_to_freq(pileup_file,outfile):
    """
    Convert mpileup file to frequencies counts
    """
    # mpileup to frequencies counts
    df_freq = pileup_to_freq(pileup_file)
    #consider SNPs only
    df_freq = df_freq[~df_freq['column_name'].str.contains('Deletion|+|-', na=False)]
    df_freq = df_freq[df_freq['Base']!=df_freq['Ref']]
    #add Allele Frequency column
    df_freq['AF']=df_freq['Count']/df_freq['Depth']
    #convert to bed
    df_freq[['Chrom','Pos','Pos','AF']].to_csv(outfile,sep='\t',header=None,index=None)    

def get_neutral_AF(SNP_bed_file,cnvs_file,outfile)->pd.DataFrame:
    
    cmd = f"bedtools intersect -v -a {SNP_bed_file} -b {cnvs_file} > {outfile}"
    os.system(cmd)
    df_neutral_AF = pd.read_csv(outfile,header=None,sep='\t')
    df_neutral_AF.columns = ['Chrom','Start','End','AF']
    return df_neutral_AF

def plot_neutral_AF(df_neutral_AF,sample_name,outfile):
    plt.figure()
    df_neutral_AF['AF'].hist(bins=50)
    xlabels = np.arange(0, 1.1, 0.1)
    plt.xticks(xlabels)
    plt.xlabel('Allele Frequency')
    plt.ylabel('Count (logscale)')
    plt.title(f"{sample_name} : Allele Frequency distribution in neutral regions") 
    plt.yscale('log')
    plt.savefig(outfile, dpi=300,bbox_inches="tight")

def run(argv):
    """
    Runs the plot_FREEC_neutral_AF.py script to generate histogram of AF of neutral (non-CNV) locations across the sample.
    input arguments:
    --mpileup: input mpileup file.
    --cnvs_file: input bed file holding the called CNVs.
    --out_directory: output directory
    --sample_name: sample name    
    output files:
    AF histogram: <out_directory>/<mpileup basename>.neutralAF.hist.jpeg.jpeg 
        shows histogram of AFof neutral (non-CNV) locations across the sample. 
    AF bed file: <out_directory>/<mpileup basename>.freq.SNP.neutral.bed
    """
    parser = argparse.ArgumentParser(
        prog="plot_FREEC_neutral_AF.py", description="generate histogram of AF of neutral (non-CNV) locations across the sample"
    )
    
    parser.add_argument("--mpileup", help="input mpileup file as generated from controlFREEC pipeline.", required=True, type=str)
    parser.add_argument("--cnvs_file", help="input bed file holding the sample's called CNVs", required=True, type=str)
    parser.add_argument("--out_directory", help="output directory", required=True, type=str)
    parser.add_argument("--sample_name", help="sample name", required=True, type=str)
    parser.add_argument("--verbosity",help="Verbosity: ERROR, WARNING, INFO, DEBUG",required=False,default="INFO",)

    args = parser.parse_args(argv[1:])
    logger.setLevel(getattr(logging, args.verbosity))

    
    pileup_file = args.mpileup
    basename = os.path.basename(pileup_file)
    outdir = args.out_directory
    logger.info(f"file will be written to {outdir}")

    SNP_bed_file = pjoin(outdir,f"{basename}.freq.SNP.bed")
    mpileup_to_freq(pileup_file,SNP_bed_file)
    
    # mpileup to frequencies counts
    # freq_file = pjoin(outdir, f"{basename}.freq")
    # cmd = f"python -m ugbio_cnv.pileuptofreq -i {pileup_file} -o {freq_file}"
    # os.system(cmd)
    # logger.info(f"finished running pileuptofreq")

    #consider SNPs only
    # freq_SNP_file = pjoin(outdir, f"{basename}.freq.SNP")
    # cmd = f"cat {freq_file} | grep -v \"\[+-\]\" | grep -v \"Deletion\" | grep -v \"Insertion\" | awk -F \";\" \'$4!=$6\' > {freq_SNP_file}"
    # os.system(cmd)    

    #convert to bed
    # df_SNP = pd.read_csv(freq_SNP_file , sep=";")
    # df_SNP['AF']=df_SNP['Count']/df_SNP['Depth']

    # SNP_bed_file = pjoin(outdir,f"{basename}.freq.SNP.bed")
    # df_SNP[['Chrom','Pos','Pos','AF']].to_csv(SNP_bed_file,sep='\t',header=None,index=None)
    
    neutral_SNP_bed_file = pjoin(outdir,f"{basename}.freq.SNP.neutral.bed")
    df_neutral_AF = get_neutral_AF(SNP_bed_file,args.cnvs_file,neutral_SNP_bed_file)

    # cmd = f"bedtools intersect -v -a {SNP_bed_file} -b {args.cnvs_file} > {neutral_SNP_bed_file}"
    # os.system(cmd)

    # df_neutral_AF = pd.read_csv(neutral_SNP_bed_file,header=None,sep='\t')
    # df_neutral_AF.columns = ['Chrom','Start','End','AF']
    
    neutral_SNP_hist_file = pjoin(outdir,f"{basename}.freq.SNP.neutral.hist.jpeg")
    # plt.figure()
    # df_neutral_AF['AF'].hist(bins=50)
    # xlabels = np.arange(0, 1.1, 0.1)
    # plt.xticks(xlabels)
    # plt.xlabel('Allele Frequency')
    # plt.ylabel('Count')   
    # plt.title(f"{args.sample_name} : Allele Frequency distribution in neutral regions") 
    # plt.yscale('log')
    # plt.savefig(neutral_SNP_hist_file, dpi=300,bbox_inches="tight")
    
    plot_neutral_AF(df_neutral_AF,args.sample_name,neutral_SNP_hist_file)
    logger.info(f"out hist file : {neutral_SNP_hist_file}")

if __name__ == "__main__":
    run(sys.argv)