import pysam
import sys
import argparse

def filter_low_af_ratio_to_background(input_vcf, output_vcf, af_ratio_threshold=10, new_filter="LowAFRatioToBackground"):

    vcf_in = pysam.VariantFile(input_vcf)    

    # Add a new FILTER definition to the header
    if new_filter not in vcf_in.header.filters:
        filter_desc = f"For snps and non-h-indels: filter if AF ratio to background in GT ALT alleles < {af_ratio_threshold}"
        vcf_in.header.filters.add(new_filter, None, None, filter_desc)

    vcf_out = pysam.VariantFile(output_vcf, 'w', header=vcf_in.header)

    for record in vcf_in.fetch():
        
        # Skip if variant is marked RefCall
        if (record.filter.keys() == {"RefCall"}) | (record.info.get("VARIANT_TYPE")=='h-indel'):
            vcf_out.write(record)
            continue
        else:
            
            failed = False
            for sample in record.samples:
                            
                gt = record.samples[sample].get("GT")
                ad = record.samples[sample].get("AD")
                dp = record.samples[sample].get("DP")
                bg_ad = record.samples[sample].get("BG_AD")
                bg_dp = record.samples[sample].get("BG_DP")
    
                if gt is None or ad is None:
                    continue
    
                for allele in gt:
                    if allele is None or allele == 0:
                        continue  # skip REF or missing
                    elif dp==0:
                        #print(f"depth of {record.chrom}:{record.pos} sample {sample} allele {record.alts[allele-1]} is zero")
                        continue
                    elif bg_dp==0:
                        #print(f"background depth of {record.chrom}:{record.pos} sample {sample} allele {record.alts[allele-1]} is zero")
                        continue
                    else: 
                        if bg_ad[allele]>0:                        
                            af_ratio = (ad[allele]/dp) / (bg_ad[allele]/bg_dp)
                            if af_ratio >= af_ratio_threshold:
                                failed = False
                                break
                            else:
                                failed = True     
                            
    
            if failed:
                record.filter.add(new_filter)
        
        vcf_out.write(record)

    vcf_in.close()
    vcf_out.close()
    print(f"Annotated VCF written to: {output_vcf}")


def main():
    parser = argparse.ArgumentParser(description="Annotate variants with low AF ratio")
    parser.add_argument("input_vcf", help="Input VCF file")
    parser.add_argument("output_vcf", help="Output VCF file")
    parser.add_argument("--af_ratio_threshold", type=float, default=10, help="AF ratio threshold (default: 10)")
    parser.add_argument("--new_filter", default="LowAFRatioToBackground", help="Name of new FILTER tag (default: LowAFRatioToBackground)")

    args = parser.parse_args()

    filter_low_af_ratio_to_background(
        input_vcf=args.input_vcf,
        output_vcf=args.output_vcf,
        af_ratio_threshold=args.af_ratio_threshold,
        new_filter=args.new_filter
    )
    
if __name__ == "__main__":
    main()
