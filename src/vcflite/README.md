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
        --where <WHERE>        Optional WHERE clause
        --group-by <GROUP_BY>  Optional GROUP BY clause (columns: chrom, pos, id, ref, alt, qual, filter, info)
        --having <HAVING>      Optional HAVING clause
        --limit <LIMIT>        Optional LIMIT clause
    -h, --help                 Print help

The `query` command supports two options for filtering the data: `--group-by` and `--having`. The `--group-by` option is used to group the data by one or more columns. The `--having` option is used to filter the groups based on a condition. The default is to group by `chrom` and `pos` and to filter for groups that have exactly one member. This is useful for removing duplicate variants.

### Examples

The `info` field can be filtered on by specifying `key` and `value` constraints:

Get variants with an `rq` value less than 0.5:

    --where "key = 'rq' AND value <= 0.5"

Get variants with an `X_READ_COUNT` value of at least 20:

    --where "key = 'X_READ_COUNT' AND value >= 20"

Grouping by `chrom` and `pos` and filtering for groups that have exactly one member is useful for removing duplicate variants:

    --group-by "chrom, pos" --having "COUNT(*) = 1"

When using `--group-by`, [aggregate functions](https://www.sqlite.org/draft/lang_aggfunc.html) such as `COUNT(*)` can be used to filter the groups.

A query that makes use of grouping as well as filtering:

    --where "key = 'X_READ_COUNT' AND value >= 20"
    --group-by "chrom, pos"
    --having "CAST(COUNT(*) AS FLOAT)/value > 0.04"

Note: The `WHERE` clause is applied _before_ the `GROUP BY` clause, so it's possible to filter the data before grouping it. The `HAVING` clause is applied _after_ the `GROUP BY` clause, so it's possible to filter the groups based on the results of the aggregation (such as the `COUNT(*)` function).

The `CAST` function is used to convert the integer result of `COUNT(*)` to a float, so that the division by `value` will result in a float.


### Debugging

To send the VCF output to the terminal without the header:

    vcflite query --vcf-out /dev/stdout --skip-header --limit 10


### Performance

The main performance gain when querying comes from the use of SQLite indexes.

Since the import and query stages are separate, it's easy to add more indexes from the command line by using the `sqlite3` command line tool after the database is created, without having to change the code. After an index is created, it will be used automatically by following queries.

The `--explain` option can be used to see how the query is executed according to SQLite's [EXPLAIN QUERY PLAN](https://www.sqlite.org/eqp.html), and can help decide in indexes should be added.

Example output:

    Query plan:

    SEARCH info_keys USING COVERING INDEX sqlite_autoindex_info_keys_1 (key=?)
    SEARCH info USING INDEX idx_info_key_id (key_id=?)
    SEARCH variants USING INTEGER PRIMARY KEY (rowid=?)
    USE TEMP B-TREE FOR GROUP BY

### Limitations

- Only a single CPU core is used. Resolving this would make the import much faster.
- The database is not compressed, which means that the disk space usage can be roughly 10x the size of the VCF file.
- There is an experimental `zstd` branch in this repo that uses [sqlite-zstd](https://github.com/phiresky/sqlite-zstd) to compress the database, but it is not yet ready for use. Its main limitation is that it is slow, but it greatly reduces the database size (about 4-5x). This can also be optimized further.
