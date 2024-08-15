use std::io;
use std::io::Read;
use std::io::Write;

use noodles::bed;
use noodles::bed::record::OptionalFields;
use noodles::core::Position;

/*
  bedtools groupby -c 3 -o count < all.bed > grouped.bed
*/
fn groupby(input: impl Read, output: impl Write) {
    read_records(input);

    // ....

    write_records(output);
}

fn read_records(input: impl Read) {
    let buf_input = io::BufReader::new(input);
    let mut bed_input = bed::Reader::new(buf_input);

    for rec in bed_input.records::<3>() {
        let record_in = rec.unwrap();

        eprintln!("{record_in:?}");
    }
}

fn write_records(output: impl Write) {
    let buf_output = io::BufWriter::new(output);
    let mut bed_output = bed::Writer::new(buf_output);

    /* loop... */
    {
        let count = 42;
        let optional_fields = OptionalFields::from(vec![count.to_string()]);

        let name = "chr1";
        let start_position = Position::try_from(10107).unwrap();
        let end_position = Position::try_from(10108).unwrap();

        let record_out = bed::Record::<3>::builder()
            .set_reference_sequence_name(name)
            .set_start_position(start_position)
            .set_end_position(end_position)
            .set_optional_fields(optional_fields)
            .build()
            .unwrap();

        bed_output.write_record(&record_out).unwrap();
    }
}

fn main() {
    groupby(io::stdin(), io::stdout())
}
