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

        /// Optional WHERE clause
        /// Examples:
        ///   key = 'rq' AND value <= 0.5
        ///   key = 'X_FILTERED_COUNT' AND value = 4

        #[arg(long, name = "where", value_name = "WHERE")]
        where_: Option<String>,

        /// Optional GROUP BY clause (columns: chrom, pos, id, ref, alt, qual, filter, info)
        /// Examples:
        ///   chrom, pos
        #[arg(long)]
        group_by: Option<String>,

        /// Optional HAVING clause
        /// Examples:
        ///   COUNT(*) = 1
        #[arg(long)]
        having: Option<String>,

        /// Optional LIMIT clause
        #[arg(long)]
        limit: Option<String>,
    },
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Commands::Import { vcf_in, db } => {
            let mut variants = Variants::new(&db)?;
            variants.import(&vcf_in)?;
        }
        Commands::Query {
            db,
            vcf_out,
            where_,
            group_by,
            having,
            limit,
        } => {
            let variants = Variants::new(&db)?;

            variants.query(
                &vcf_out,
                where_.as_deref(),
                group_by.as_deref(),
                having.as_deref(),
                limit.as_deref(),
            )?;
        }
    }

    Ok(())
}
