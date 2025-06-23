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

pub struct QueryOpts<'a> {
    pub vcf_out: &'a PathBuf,
    pub where_: Option<&'a str>,
    pub group_by: Option<&'a str>,
    pub having: Option<&'a str>,
    pub limit: Option<&'a str>,
    pub skip_header: bool,
    pub explain: bool,
}
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
                key TEXT UNIQUE -- indexed automatically
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
            CREATE INDEX IF NOT EXISTS idx_info_key_id ON info (key_id);

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

    pub(crate) fn query(&self, options: QueryOpts) -> Result<()> {
        let conn = &self.conn;

        let sql = "
            SELECT
                variants.variant_id,
                chrom, pos, id, ref, alt, qual, filter,
                info_keys.key,
                info.value
            FROM
                variants
            LEFT JOIN
                info ON variants.variant_id = info.variant_id
            JOIN
                info_keys ON info.key_id = info_keys.key_id
            "
        .to_string();

        let sql = if let Some(where_) = options.where_ {
            format!("{sql} WHERE {where_}")
        } else {
            sql
        };

        let sql = if let Some(group_by) = options.group_by {
            format!("{sql} GROUP BY {group_by}")
        } else {
            sql
        };

        let sql = if let Some(having) = options.having {
            format!("{sql} HAVING {having}")
        } else {
            sql
        };

        let sql = if let Some(limit) = options.limit {
            format!("{sql} LIMIT {limit}")
        } else {
            sql
        };

        let sql = if options.explain {
            format!("EXPLAIN QUERY PLAN {sql}")
        } else {
            sql
        };

        eprintln!("Exporting to VCF: {}", options.vcf_out.display());
        eprintln!("Query: {sql};");

        let before = Instant::now();
        let mut stmt = conn.prepare(&sql)?;
        let mut rows = stmt.query([])?;

        let mut stmt_info = conn.prepare(
            "SELECT key, value
             FROM info
             JOIN info_keys ON info.key_id = info_keys.key_id
             WHERE variant_id = :variant_id",
        )?;

        let mut writer = vcf::io::writer::Builder::default().build_from_path(options.vcf_out)?;
        let header = load_header(conn)?;

        if !options.skip_header {
            writer.write_header(&header)?;
        }

        if options.explain {
            eprintln!("\nQuery plan:\n");
        }

        while let Some(row) = rows.next()? {
            if options.explain {
                let detail: String = row.get("detail")?;
                eprintln!("{detail}");
                continue;
            }

            let variant_id: usize = row.get("variant_id")?;
            let chrom: String = row.get("chrom")?;
            let pos: Option<usize> = row.get("pos")?;
            let id: Option<String> = row.get("id")?;
            let ref_: String = row.get("ref")?;
            let alt: String = row.get("alt")?;
            let qual: Option<f32> = row.get("qual")?;
            let filter: String = row.get("filter")?;

            let pos = Position::try_from(pos.unwrap_or_default())?;
            let ids: record_buf::Ids = id.map(String::from).into_iter().collect();
            let alternate_bases = record_buf::AlternateBases::from(vec![alt]);
            let filters: record_buf::Filters = [filter].into_iter().collect();

            let mut info = record_buf::Info::default();

            // Construct INFO field
            {
                use record_buf::info::field::Value;
                use rusqlite::types::ValueRef;

                let mut rows = stmt_info.query(named_params! {
                    ":variant_id": variant_id,
                })?;

                while let Some(row) = rows.next()? {
                    let key: String = row.get("key")?;
                    let value = row.get_ref("value")?;

                    let value: Option<Value> = match value {
                        ValueRef::Integer(i) => Some(Value::Integer(i as i32)),
                        ValueRef::Real(f) => Some(Value::Float(f as f32)),
                        ValueRef::Text(s) | ValueRef::Blob(s) => {
                            Some(Value::String(String::from_utf8(s.to_vec())?))
                        }
                        ValueRef::Null => None,
                    };

                    info.insert(key, value);
                }
            }

            let mut record = vcf::variant::RecordBuf::builder()
                .set_reference_sequence_name(chrom)
                .set_variant_start(pos)
                .set_ids(ids)
                .set_reference_bases(ref_)
                .set_alternate_bases(alternate_bases)
                .set_filters(filters)
                .set_info(info)
                .build();

            // The builder doesn't accept an Option<f32> for the quality score,
            // so we have to set it afterwards.
            *record.quality_score_mut() = qual;

            writer.write_variant_record(&header, &record)?;
        }

        if options.explain {
            return Ok(());
        }

        eprintln!("Query/export took {:.2?}", before.elapsed());

        Ok(())
    }
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
