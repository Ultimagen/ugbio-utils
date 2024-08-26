use std::path::Path;

use anyhow::Result;
use clap::Parser;

mod variants;
use variants::Variants;

#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    /// Input VCF file (e.g. input.vcf.gz)
    #[arg(long)]
    vcf_in: String,

    /// Output VCF file (e.g. output.vcf.gz)
    #[arg(long)]
    vcf_out: String,
}
// Query (TODO: Currently hardcoded to 'GROUP BY chrom, pos HAVING COUNT(*) = 1')
// #[arg(long)]
// query: String,

fn main() -> Result<()> {
    let args = Args::parse();

    let vcf_in = Path::new(&args.vcf_in);
    let vcf_out = Path::new(&args.vcf_out);

    let vcf_db = vcf_out.with_extension("db");

    let variants = Variants::from_vcf(vcf_in, &vcf_db)?;
    variants.query(vcf_out)?;

    Ok(())
}
