# vcflite

A very early stab at implementing VCF querying and filtering using [noodles](https://crates.io/crates/noodles) and [SQLite](https://www.sqlite.org).

The goal is to be simple and performant: After all SQL is highly specialized in set operations, and Rust with `noodles` provides a performant environment for handling VCF files.


## Installation

The easiest way to install `vcflite` is to use the Docker image:

    docker run -ti ugbio_vcflite vcflite --help


## Usage

    Usage: vcflite <COMMAND>

    Commands:
      import  Import VCF (possibly compressed) into a SQLite database
      query   Query SQLite database and export to VCF
      help    Print this message or the help of the given subcommand(s)

    Options:
      -h, --help     Print help
      -V, --version  Print version

There are two commands: `import` and `query`. The `import` command reads a VCF file and imports it into a SQLite database.  The `query` command queries the database and exports the results to a VCF file. Both input and output files may be compressed, this is determined automatically by the file extension (`.vcf` or `.vcf.gz`).

After importing the data once, multiple queries can be performed on the database.

If `import` is run on an existing database file, the data will be appended to the existing data. Take care when using this, since the headers should be identical.


### Importing

    Import VCF (possibly compressed) into a SQLite database

    Usage: vcflite import [OPTIONS] --vcf-in <FILE>

    Options:
        --db <FILE>      SQLite database file to import to [default: variants.db]
        --vcf-in <FILE>  VCF file to read from
    -h, --help           Print help


### Querying

    Query SQLite database and export to VCF

    Usage: vcflite query [OPTIONS] --vcf-out <FILE>

    Options:
        --db <FILE>            SQLite database file to query from [default: variants.db]
        --vcf-out <FILE>       VCF file to write to
        --group-by <GROUP_BY>  [default: "chrom, pos"]
        --no-group-by          Do not provide a GROUP BY clause (override the default)
        --having <HAVING>      [default: "COUNT(*) = 1"]
        --no-having            Do not provide a HAVING clause (override the default)
    -h, --help                 Print help


The `query` command supports two options for filtering the data: `--group-by` and `--having`. The `--group-by` option is used to group the data by one or more columns. The `--having` option is used to filter the groups based on a condition. The default is to group by `chrom` and `pos` and to filter for groups that have exactly one member. This is useful for removing duplicate variants.


### Performance

The main performance gain when querying comes from the use of SQLite indexes.
Currently, a single index is created when importing:

    CREATE INDEX idx_variants_chrom_pos ON variants (chrom, pos);

Since the import and query stages are separate, it's easy to add more indexes from the command line by using the `sqlite3` command line tool after the database is created, without having to change the code. After an index is created, it will be used automatically by following queries.

The SQLite [EXPLAIN QUERY PLAN](https://www.sqlite.org/eqp.html) command can be used to see how the query is executed.

For example, the following query will use the index:

    $ sqlite3 variants.db
    SQLite version 3.46.1 2024-08-13 09:16:08
    Enter ".help" for usage hints.
    sqlite> EXPLAIN QUERY PLAN SELECT * FROM variants GROUP BY chrom, pos;
    QUERY PLAN
    `--SCAN variants USING INDEX idx_variants_chrom_pos

### Limitations

- Only a single CPU core is used. Resolving this would make the import much faster.
- The database is not compressed, which means that the disk space usage can be roughly 10x the size of the VCF file.
- There is an experimental `zstd` branch in this repo that uses [sqlite-zstd](https://github.com/phiresky/sqlite-zstd) to compress the database, but it is not yet ready for use. Its main limitation is that it is slow, but it greatly reduces the database size (about 4-5x). This can also be optimized further.
