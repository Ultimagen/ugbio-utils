use std::fs::File;
use std::io;
use std::io::BufReader;
use std::io::Read;
use std::io::Write;
use std::time::Instant;

use anyhow::Result;

use noodles::bed;
use noodles::core::Position;
use noodles::vcf;

use rusqlite::Connection;

/*
fn vcf_read() -> Result<()> {
    // Open the database connection
    let conn = Connection::open("vcf.db")?;

    // Create the table if it doesn't exist
    conn.execute(
        "CREATE TABLE IF NOT EXISTS variants (
            chrom TEXT,
            pos INTEGER,
            id TEXT,
            ref TEXT,
            alt TEXT,
            qual REAL,
            filter TEXT
        )",
        [],
    )?;

    // Open the VCF file
    let file = File::open("path/to/your/file.vcf.gz").expect("Failed to open file");
    let reader = BufReader::new(file);
    let mut vcf_reader = vcf::record::Builder::default()
        .build_bgzf(reader)
        .expect("Failed to build VCF reader");

    // Read the VCF header
    let header = vcf_reader.read_header().expect("Failed to read VCF header");

    // Prepare the SQL statement
    let mut stmt = conn.prepare(
        "INSERT INTO variants (chrom, pos, id, ref, alt, qual, filter)
         VALUES (?, ?, ?, ?, ?, ?, ?)",
    )?;

    // Iterate over records in the VCF file
    for result in vcf_reader.records(&header) {
        let record = result.expect("Failed to read VCF record");

        let chrom = record.chromosome().to_string();
        let pos = record.position().get() as i64;
        let id = record.ids().join(",");
        let ref_allele = record.reference_bases().to_string();
        let alt_alleles = record
            .alternate_bases()
            .iter()
            .map(|allele| allele.to_string())
            .collect::<Vec<String>>()
            .join(",");
        let qual = record.quality_score().map(|q| q.get()).unwrap_or(0.0);
        let filter = record
            .filters(&header)
            .map(|filters| {
                filters
                    .iter()
                    .map(|f| f.to_string())
                    .collect::<Vec<String>>()
                    .join(";")
            })
            .unwrap_or_else(|| "PASS".to_string());

        // Insert the record into the database
        stmt.execute([
            &chrom,
            &pos.to_string(),
            &id,
            &ref_allele,
            &alt_alleles,
            &qual.to_string(),
            &filter,
        ])?;
    }

    println!("VCF data has been successfully imported into the database.");
    Ok(())
}
*/

/*
    TODO: Use the vcf.gz format directly instead of converting to bed and then filtering:
        Read the vcf data into the DB.

    bedtools groupby -c 3 -o count < all.bed > grouped.bed
*/
fn groupby(input: impl Read, output: impl Write) -> Result<()> {
    let mut conn = Connection::open("bed.db")?;

    let before = Instant::now();
    read_records(input, &mut conn)?;
    eprintln!("read_records: {:.2?}", before.elapsed());

    // todo!("perform the groupby query on the sqlite table");
    let before = Instant::now();
    write_records(output, &mut conn)?;
    eprintln!("write_records: {:.2?}", before.elapsed());

    Ok(())
}

fn read_records(input: impl Read, conn: &mut Connection) -> Result<()> {
    let buf_input = io::BufReader::new(input);
    let mut bed_input = bed::Reader::new(buf_input);

    conn.execute(
        "CREATE TABLE IF NOT EXISTS records (
           name TEXT NOT NULL,
           start INTEGER NOT NULL,
           end INTEGER NOT NULL
         )",
        (),
    )?;

    let tx = conn.transaction()?;
    {
        let mut stmt = tx.prepare("INSERT INTO records (name, start, end) VALUES (?1, ?2, ?3)")?;

        for rec in bed_input.records::<3>() {
            let record_in = rec?;

            // eprintln!("{record_in:?}");
            stmt.execute((
                record_in.reference_sequence_name(),
                record_in.start_position().get(),
                record_in.end_position().get(),
            ))?;
        }
    }
    tx.commit()?;

    Ok(())
}

fn write_records(output: impl Write, conn: &mut Connection) -> Result<()> {
    let buf_output = io::BufWriter::new(output);
    let mut bed_output = bed::Writer::new(buf_output);

    let mut stmt = conn.prepare(
        r"
            SELECT name, start, end
            FROM records
            GROUP BY end
            HAVING COUNT(*) = 1
        ",
    )?;

    let mut rows = stmt.query([])?;

    while let Some(row) = rows.next()? {
        let name: String = row.get(0)?;
        let start: usize = row.get(1)?;
        let end: usize = row.get(2)?;

        let count: usize = 1; // TODO: Unnecessary since it's always 1!
        let optional_fields = bed::record::OptionalFields::from(vec![count.to_string()]);

        let start_position = Position::try_from(start)?;
        let end_position = Position::try_from(end)?;

        let record_out = bed::Record::<3>::builder()
            .set_reference_sequence_name(name)
            .set_start_position(start_position)
            .set_end_position(end_position)
            .set_optional_fields(optional_fields)
            .build()
            .unwrap();

        bed_output.write_record(&record_out).unwrap();
    }

    Ok(())
}

fn main() -> Result<()> {
    groupby(io::stdin(), io::stdout())
}
