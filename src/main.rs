use std::path::PathBuf;

use anyhow::Result;
use clap::{Parser, Subcommand};

mod variants;
use variants::Variants;

#[derive(Parser)]
#[command(version, about, long_about = None)]
struct Cli {
    // /// Optional name to operate on
    // name: Option<String>,

    // /// Sets a custom config file
    // #[arg(short, long, value_name = "FILE")]
    // config: Option<PathBuf>,

    // /// Turn debugging information on
    // #[arg(short, long, action = clap::ArgAction::Count)]
    // debug: u8,
    #[command(subcommand)]
    command: Commands,
}

const DEFAULT_DB: &str = "variants.db";

#[derive(Subcommand)]
enum Commands {
    /// Import VCF (possibly compressed) into a SQLite database
    Import {
        /// SQLite database file to import to
        #[arg(long, value_name = "FILE", default_value = DEFAULT_DB)]
        db: PathBuf,

        /// VCF file to read from
        #[arg(long, value_name = "FILE")]
        vcf_in: PathBuf,
    },
    /// Query SQLite database and export to VCF
    Query {
        /// SQLite database file to query from
        #[arg(long, value_name = "FILE", default_value = DEFAULT_DB)]
        db: PathBuf,

        /// VCF file to write to
        #[arg(long, value_name = "FILE")]
        vcf_out: PathBuf,
        // Query (TODO: Currently hardcoded to 'GROUP BY chrom, pos HAVING COUNT(*) = 1')
        // #[arg(long)]
        // query: String,
    },
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Commands::Import { vcf_in, db } => {
            let mut variants = Variants::new(&db)?;
            variants.import(&vcf_in)?;
        }
        Commands::Query { db, vcf_out } => {
            let variants = Variants::new(&db)?;
            variants.query(&vcf_out)?;
        }
    }

    Ok(())
}
