use std::io;
use std::io::Read;
use std::io::Write;
use std::time::Instant;

use anyhow::Result;

use noodles::bgzf;
use noodles::core::Position;
use noodles::vcf;
use noodles::vcf::variant::io::Write as _;
use noodles::vcf::variant::record::Ids as _;
use noodles::vcf::variant::record_buf;

use rusqlite::named_params;
use rusqlite::Connection;

struct Variants {
    conn: Connection,
    header: vcf::Header,
}

impl Variants {
    fn from_vcf(input: impl Read) -> Result<Self> {
        let mut conn = Connection::open("vcf.db")?;

        conn.execute_batch(
            "BEGIN;

             CREATE TABLE variants (
                chrom TEXT,
                pos INTEGER,
                id TEXT,
                ref TEXT,
                alt TEXT,
                qual REAL,
                filter TEXT,
                info TEXT
            );

            CREATE INDEX idx_variants_chrom_pos ON variants (chrom, pos);

            COMMIT; ",
        )?;

        // let reader = Builder::default().build_from_path("sample.vcf")?;
        let mut reader = vcf::io::reader::Builder::default()
            .set_compression_method(vcf::io::CompressionMethod::Bgzf)
            .build_from_reader(input)?;

        let header = reader.read_header()?;
        // eprintln!("Header: {header:#?}");

        let before = Instant::now();

        let tx = conn.transaction()?;
        {
            let mut stmt = tx.prepare(
                "INSERT INTO variants
                    (chrom, pos, id, ref, alt, qual, filter, info)
                 VALUES
                    (:chrom, :pos, :id, :ref, :alt, :qual, :filter, :info)",
            )?;

            for result in reader.records() {
                let record = result?;

                // eprintln!("Records: {record:#?}");
                let pos: Option<usize> = record.variant_start().transpose()?.map(usize::from);
                let ids = record.ids();
                let id: Option<&str> = ids.iter().next(); // Take only first ID (if any)
                let qual: Option<f32> = record.quality_score().transpose()?;

                stmt.execute(named_params! {
                    ":chrom": record.reference_sequence_name(),
                    ":pos": pos,
                    ":id": id,
                    ":ref": record.reference_bases(),
                    ":alt": record.alternate_bases().as_ref(),
                    ":qual": qual,
                    ":filter": record.filters().as_ref(),
                    ":info": record.info().as_ref(),
                })?;
            }
        }
        tx.commit()?;

        conn.execute("ANALYZE", [])?;

        eprintln!("vcf_read: {:.2?}", before.elapsed());

        Ok(Self { conn, header })
    }

    fn query(&self, output: impl Write) -> Result<()> {
        // let buf_output = io::BufWriter::new(output);
        // let mut bed_output = bed::Writer::new(buf_output);
        let conn = &self.conn;

        let mut stmt = conn.prepare(
            "
            SELECT chrom, pos, id, ref, alt, qual, filter, info
            FROM variants
            GROUP BY chrom, pos
            HAVING COUNT(*) = 1
            ",
        )?;

        // let mut limit = 0;
        let mut rows = stmt.query([])?;

        let mut writer = vcf::io::Writer::new(bgzf::Writer::new(output));

        writer.write_header(&self.header)?;

        while let Some(row) = rows.next()? {
            // limit += 1;
            // if limit > 10 {
            //     break;
            // };

            let chrom: String = row.get("chrom")?;
            let pos: Option<usize> = row.get("pos")?;
            let id: Option<String> = row.get("id")?;
            let ref_: String = row.get("ref")?;
            let alt: String = row.get("alt")?;
            let qual: Option<f32> = row.get("qual")?;
            let filter: String = row.get("filter")?;
            let info: String = row.get("info")?;

            // eprintln!("{chrom} {pos:?} {id:?} {ref_} {alt} {qual:?} {filter} {info}");

            // let count: usize = 1; // TODO: Unnecessary since it's always 1!
            // let optional_fields = bed::record::OptionalFields::from(vec![count.to_string()]);

            let pos = Position::try_from(pos.unwrap_or_default())?;
            let ids: record_buf::Ids = id.map(String::from).into_iter().collect();
            let alternate_bases = record_buf::AlternateBases::from(vec![alt]);
            let filters: record_buf::Filters = [filter].into_iter().collect();

            // TODO: There seems to be no way to set the info from a raw string (like we kept when reading the VCF).
            // It seems we must parse the string and reconstruct it here :shrug:
            // let info = record_buf::Info::new(&info);

            let builder = vcf::variant::RecordBuf::builder()
                .set_reference_sequence_name(chrom)
                .set_variant_start(pos)
                .set_ids(ids)
                .set_reference_bases(ref_)
                .set_alternate_bases(alternate_bases)
                .set_filters(filters)
                // .set_info(info)
                ;

            let mut record = builder.build();

            // The builder doesn't accept an Option<f32> for the quality score,
            // so we have to set it afterwards.
            *record.quality_score_mut() = qual;

            writer.write_variant_record(&self.header, &record)?;
        }

        Ok(())
    }
}

fn main() -> Result<()> {
    let variants = Variants::from_vcf(io::stdin())?;
    variants.query(io::stdout())?;

    Ok(())
}
