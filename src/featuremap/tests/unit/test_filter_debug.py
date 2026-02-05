#!/usr/bin/env python3
"""Quick test to verify filter logic with null ID values"""

import polars as pl
from ugbio_featuremap.filter_dataframe import (
    _create_filter_columns,
    _create_final_filter_column,
)

# Create test dataframe with NULL ID values (simulating Polars parsing with null_values=["."])
df_test = pl.DataFrame(
    {
        "ID": [None, "rs123", None, "rs456", None],  # None = "." from VCF
        "gnomAD_AF": [None, 0.5, 0.00001, 0.01, None],
        "POS": [100, 200, 300, 400, 500],
    }
)

print("Original DataFrame (ID=None represents '.' from VCF):")
print(df_test)
print()

# Define filters
filters = [
    {"name": "not_in_dbsnp", "type": "region", "field": "ID", "op": "eq", "value": "."},
    {"name": "not_in_gnomad", "type": "region", "field": "gnomAD_AF", "op": "le", "value": 0.0001},
]

# Apply filters
lazy_frame = df_test.lazy()
lazy_frame, filter_cols = _create_filter_columns(lazy_frame, filters)

# Add ID null handling (treating null as ".")
id_col_name = "__filter_not_in_dbsnp"
print("Adding null handling for ID filter (null = '.')")
lazy_frame = lazy_frame.with_columns((pl.col(id_col_name) | pl.col("ID").is_null()).alias(id_col_name))

# Add gnomAD null handling
gnomad_col_name = "__filter_not_in_gnomad"
lazy_frame = lazy_frame.with_columns((pl.col(gnomad_col_name) | pl.col("gnomAD_AF").is_null()).alias(gnomad_col_name))

lazy_frame = _create_final_filter_column(lazy_frame, filter_cols, None)

# Collect and show results
result = lazy_frame.collect()
print("\nDataFrame with filter columns:")
print(result)
print()

# Show filtered result
filtered = result.filter(pl.col("__filter_final")).select(pl.exclude("^__filter_.*$"))
print("Filtered DataFrame:")
print(filtered)
print()
print(f"Original rows: {df_test.height}, Filtered rows: {filtered.height}")
print("\nExpected: 3 rows (POS 100, 300, 500 - all with ID=None and low/null gnomAD_AF)")
