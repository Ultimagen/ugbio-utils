use std::collections::HashMap;
use std::io;
use std::path::PathBuf;
use std::time::Instant;

use anyhow::bail;
use anyhow::Context;
use anyhow::Result;

use noodles::core::Position;
use noodles::vcf;
use vcf::variant::io::Write as _;
use vcf::variant::record::Ids as _;
use vcf::variant::record_buf;

use rusqlite::named_params;
use rusqlite::Connection;

// TODO: Use [sqlite_zstd](https://github.com/phiresky/sqlite-zstd?tab=readme-ov-file#usage)
// TODO: Deconstruct the INFO field into its own table (note that Values can be Arrays)

pub(crate) struct Variants {
    conn: Connection,
}

impl Variants {
    pub(crate) fn new(db: &PathBuf) -> Result<Self> {
        let conn = Connection::open(db)
            .with_context(|| format!("Failed to open database: {}", db.display()))?;

        eprintln!("Opened database: {}", db.display());
        Ok(Self { conn })
    }

    pub(crate) fn import(&mut self, vcf_in: &PathBuf) -> Result<()> {
        let conn = &mut self.conn;

        eprintln!("Importing VCF: {}", vcf_in.display());

        conn.execute_batch(
            /*
               https://kerkour.com/sqlite-for-servers

               Notes:
               - There is no need for AUTOINCREMENT on the primary key (see https://www.sqlite.org/autoinc.html)
               - Pragmas: https://www.sqlite.org/pragma.html
            */
            "
            PRAGMA foreign_keys = ON;
            PRAGMA journal_mode=OFF;
            -- PRAGMA busy_timeout=2000;

            BEGIN;

            CREATE TABLE IF NOT EXISTS variants (
                variant_id INTEGER PRIMARY KEY,
                chrom TEXT,
                pos INTEGER,
                id TEXT,
                ref TEXT,
                alt TEXT,
                qual REAL,
                filter TEXT
            );

            CREATE TABLE IF NOT EXISTS info_keys (
                key_id INTEGER PRIMARY KEY,
                key TEXT UNIQUE
            );

            CREATE TABLE IF NOT EXISTS info (
                info_id INTEGER PRIMARY KEY,
                variant_id INTEGER,
                key_id INTEGER,
                value NUMERIC, -- Use dynamic data type (unlike TEXT) to save space
                FOREIGN KEY (variant_id) REFERENCES variants(variant_id),
                FOREIGN KEY (key_id) REFERENCES info_keys(key_id)
            );

            CREATE TABLE IF NOT EXISTS metadata (
                header TEXT
            );

            CREATE INDEX IF NOT EXISTS idx_variants_chrom_pos ON variants (chrom, pos);
            CREATE INDEX IF NOT EXISTS idx_info_variant_id ON info (variant_id);

            COMMIT;",
        )?;

        let mut reader = vcf::io::reader::Builder::default()
            .build_from_path(vcf_in)
            .with_context(|| format!("Failed to open VCF file: {}", vcf_in.display()))?;

        let header = reader.read_header()?;
        store_header(&header, conn)?;

        let mut key_ids: HashMap<String, i64> = HashMap::new();

        // Store unique INFO keys
        {
            let mut stmt_info_keys = conn.prepare(
                "INSERT INTO info_keys (key)
                 VALUES (:key)",
            )?;

            for (key, _) in header.infos() {
                let key_id = stmt_info_keys.insert(named_params! {
                    ":key": key,
                })?;
                key_ids.insert(key.to_string(), key_id);
            }
        }

        let before = Instant::now();
        let tx = conn.transaction()?;
        {
            let mut stmt_variants = tx.prepare(
                "INSERT INTO variants
                    (chrom, pos, id, ref, alt, qual, filter)
                 VALUES
                    (:chrom, :pos, :id, :ref, :alt, :qual, :filter)",
            )?;

            let mut stmt_info = tx.prepare(
                "INSERT INTO info
                    (variant_id, key_id, value)
                 VALUES
                    (:variant_id, :key_id, :value)",
            )?;

            for result in reader.records() {
                let record = result?;

                let pos: Option<usize> = record.variant_start().transpose()?.map(usize::from);
                let ids = record.ids();
                let id: Option<&str> = ids.iter().next(); // Take only first ID (if any)
                let qual: Option<f32> = record.quality_score().transpose()?;
                let info = record.info();

                let variant_id = stmt_variants.insert(named_params! {
                    ":chrom": record.reference_sequence_name(),
                    ":pos": pos,
                    ":id": id,
                    ":ref": record.reference_bases(),
                    ":alt": record.alternate_bases().as_ref(),
                    ":qual": qual,
                    ":filter": record.filters().as_ref(),
                })?;

                // Store INFO
                use rusqlite::ToSql as _;
                use vcf::variant::record::info::field::Value;

                for field in info.iter(&header) {
                    let s1: String;

                    match field? {
                        (key, Some(value)) => {
                            let value = match value {
                                Value::String(s) => s.to_sql(),
                                Value::Integer(ref i) => i.to_sql(),
                                Value::Float(ref f) => f.to_sql(),
                                Value::Flag => 1.to_sql(),
                                Value::Character(c) => {
                                    s1 = c.to_string();
                                    s1.to_sql()
                                }
                                Value::Array(a) => {
                                    s1 = format!("{:?}", a);
                                    s1.to_sql()
                                }
                            }?;

                            stmt_info.execute(named_params! {
                                ":variant_id": variant_id,
                                ":key_id": key_ids[key],
                                ":value": value,
                            })?;
                        }
                        (key, None) => {
                            bail!("Key without value: {key}");
                        }
                    }
                }
            }
        }
        tx.commit()?;

        // Optimize database
        // https://www.sqlite.org/lang_analyze.html
        conn.execute("PRAGMA optimize", [])?;
        conn.execute("VACUUM", [])?;

        eprintln!("Import took {:.2?}", before.elapsed());

        Ok(())
    }

    pub(crate) fn query(
        &self,
        vcf_out: &PathBuf,
        group_by: Option<&str>,
        having: Option<&str>,
    ) -> Result<()> {
        let conn = &self.conn;

        let sql = "SELECT chrom, pos, id, ref, alt, qual, filter, info FROM variants".to_string();

        let sql = if let Some(group_by) = group_by {
            format!("{sql} GROUP BY {group_by}")
        } else {
            sql
        };

        let sql = if let Some(having) = having {
            format!("{sql} HAVING {having}")
        } else {
            sql
        };

        eprintln!("Exporting to VCF: {}", vcf_out.display());
        eprintln!("Query: {sql};");

        let before = Instant::now();
        let mut stmt = conn.prepare(&sql)?;
        let mut rows = stmt.query([])?;

        let mut writer = vcf::io::writer::Builder::default().build_from_path(vcf_out)?;
        let header = load_header(conn)?;
        writer.write_header(&header)?;

        while let Some(row) = rows.next()? {
            let chrom: String = row.get("chrom")?;
            let pos: Option<usize> = row.get("pos")?;
            let id: Option<String> = row.get("id")?;
            let ref_: String = row.get("ref")?;
            let alt: String = row.get("alt")?;
            let qual: Option<f32> = row.get("qual")?;
            let filter: String = row.get("filter")?;
            let info: String = row.get("info")?;

            // eprintln!("{chrom} {pos:?} {id:?} {ref_} {alt} {qual:?} {filter} {info}");

            // let optional_fields = bed::record::OptionalFields::from(vec![count.to_string()]);

            let pos = Position::try_from(pos.unwrap_or_default())?;
            let ids: record_buf::Ids = id.map(String::from).into_iter().collect();
            let alternate_bases = record_buf::AlternateBases::from(vec![alt]);
            let filters: record_buf::Filters = [filter].into_iter().collect();

            let info = parse_info(&info, &header)?;

            let mut record = vcf::variant::RecordBuf::builder()
                .set_reference_sequence_name(chrom)
                .set_variant_start(pos)
                .set_ids(ids)
                .set_reference_bases(ref_)
                .set_alternate_bases(alternate_bases)
                .set_filters(filters)
                // .set_info("BAR=QUUX".parse()?)
                .set_info(info)
                .build();

            // The builder doesn't accept an Option<f32> for the quality score,
            // so we have to set it afterwards.
            *record.quality_score_mut() = qual;

            writer.write_variant_record(&header, &record)?;
        }

        eprintln!("Query/export took {:.2?}", before.elapsed());

        Ok(())
    }
}

fn parse_info(info: &str, header: &vcf::Header) -> Result<record_buf::Info> {
    // TODO: There seems to be no way to set the info from a raw string
    // (like the one we kept when reading the VCF).
    // It seems we must parse the string and reconstruct it here :shrug:
    //
    // Ideas:
    // - Use the original string parsing from the vcf reader to parse the INFO from the DB
    // - Store the entire record in the DB and export it here.
    //   The INFO field is the largest in any case, so this may not be a problem.

    use record_buf::info::field::Value;

    // let ns = (String::from("FOO"), Some(Value::String("BAR".to_string())));
    // let info: record_buf::Info = [ns].into_iter().collect();

    let info = vcf::record::Info::new(info);
    let info: io::Result<Vec<_>> = info.iter(header).collect();
    let info = info?;
    let info: Vec<(String, Option<Value>)> = info
        .into_iter()
        .map(|(k, v)| {
            let v: Option<Value> = v.map(|v| v.try_into().unwrap());
            (k.to_string(), v)
        })
        .collect();
    let info: record_buf::Info = info.into_iter().collect();

    // eprintln!("{info:#?}");
    // std::process::exit(0);

    // let info: std::io::Result<Vec<_>> = info.iter(&header).collect();
    // let info = record_buf::Info::from(info); // = info.iter(&header).collect();

    Ok(info)
}

fn store_header(header: &vcf::Header, conn: &Connection) -> Result<()> {
    let header = {
        let mut buf = io::Cursor::new(Vec::new());
        {
            let mut writer = vcf::io::writer::Builder::default().build_from_writer(&mut buf);
            writer.write_header(header)?;
        }
        String::from_utf8(buf.into_inner()).context("Failed to convert header to string")?
    };

    conn.execute(
        "INSERT INTO metadata (header) VALUES (:header)",
        named_params! {
            ":header": header,
        },
    )?;

    Ok(())
}

fn load_header(conn: &Connection) -> Result<vcf::Header> {
    let header: String = conn.query_row("SELECT header FROM metadata", [], |row| row.get(0))?;

    let mut reader =
        vcf::io::reader::Builder::default().build_from_reader(io::Cursor::new(header))?;

    let header = reader.read_header()?;

    Ok(header)
}
