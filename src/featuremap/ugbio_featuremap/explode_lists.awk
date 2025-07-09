#!/usr/bin/awk -f
# AWK script for exploding list format fields in VCF TSV output
# Usage: awk -v list_indices="3,4,5" -f explode_lists.awk input.tsv

BEGIN {
    # Parse list_indices parameter (0-based column indices, convert to 1-based for AWK)
    split(list_indices, indices, ",")
    for (i in indices) {
        list_cols[indices[i] + 1] = 1  # Convert to 1-based
    }
}

{
    # Store the current row
    for (i = 1; i <= NF; i++) {
        row[i] = $i
    }

    # Find the maximum list length among all list columns
    max_len = 1
    for (col_idx in list_cols) {
        if (col_idx <= NF) {
            # Split the list (comma-separated values)
            n = split(row[col_idx], values, ",")
            if (n > max_len) {
                max_len = n
            }
            # Store the split values
            for (j = 1; j <= n; j++) {
                lists[col_idx][j] = values[j]
            }
            # Fill missing values with "."
            for (j = n + 1; j <= max_len; j++) {
                lists[col_idx][j] = "."
            }
        }
    }

    # Output exploded rows
    for (i = 1; i <= max_len; i++) {
        for (j = 1; j <= NF; j++) {
            if (j in list_cols) {
                printf "%s", lists[j][i]
            } else {
                printf "%s", row[j]
            }
            if (j < NF) printf "\t"
        }
        printf "\n"
    }

    # Clear arrays for next row
    delete lists
}
