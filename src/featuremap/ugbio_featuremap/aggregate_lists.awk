#!/usr/bin/awk -f
# AWK script for computing aggregation metrics (mean, min, max, count, count_zero) for list format fields in VCF TSV output
# Also supports expanding fixed-size columns (e.g., AD -> AD_0, AD_1)
#
# Usage:
#   awk -v list_indices="3,4,5" -f aggregate_lists.awk input.tsv
#   awk -v list_indices="3,4" -v expand_indices="5" -v expand_sizes="2" -f aggregate_lists.awk input.tsv

BEGIN {
    # Parse list_indices parameter (0-based column indices, convert to 1-based for AWK)
    num_list_cols = split(list_indices, indices, ",")
    for (i = 1; i <= num_list_cols; i++) {
        col_idx = indices[i] + 1  # Convert to 1-based
        list_cols[col_idx] = 1
        ordered_list_cols[i] = col_idx
    }

    # Parse expand_indices and sizes (for expanding fixed-size columns)
    if (expand_indices != "") {
        num_expand_cols = split(expand_indices, expand_idx_list, ",")
        split(expand_sizes, expand_size_list, ",")
        for (i = 1; i <= num_expand_cols; i++) {
            col_idx = expand_idx_list[i] + 1  # Convert to 1-based
            expand_cols[col_idx] = expand_size_list[i]
        }
    }
}

function compute_aggregations(col_idx, values_str, values, n, i, val, sum, count, min_val, max_val, mean, count_zero) {
    # Split the list (comma-separated values)
    n = split(values_str, values, ",")

    sum = 0
    count = 0
    count_zero = 0
    min_val = ""
    max_val = ""

    # Process each value in the list
    for (i = 1; i <= n; i++) {
        # Trim whitespace and try to convert to number
        val = values[i]
        gsub(/^[ \t]+|[ \t]+$/, "", val)

        # Skip empty values, ".", or non-numeric values
        if (val == "" || val == "." || val !~ /^-?[0-9]+\.?[0-9]*$/) {
            continue
        }

        val = val + 0  # Convert to number

        if (val == 0) count_zero++

        if (count == 0) {
            min_val = val
            max_val = val
        } else {
            if (val < min_val) min_val = val
            if (val > max_val) max_val = val
        }

        sum += val
        count++
    }

    # Return results as string: "mean\tmin\tmax\tcount\tcount_zero"
    if (count == 0) {
        return ".\t.\t.\t0\t0"
    }

    mean = sum / count
    return sprintf("%.6f\t%.6f\t%.6f\t%d\t%d", mean, min_val, max_val, count, count_zero)
}

function expand_column(values_str, size, values, n, i, result) {
    # Expand the column and output individual elements
    n = split(values_str, values, ",")
    result = ""

    for (i = 1; i <= size; i++) {
        if (i > 1) result = result "\t"
        if (i <= n && values[i] != "" && values[i] != ".") {
            result = result values[i]
        } else {
            result = result "."
        }
    }

    return result
}

{
    # Store the current row
    for (i = 1; i <= NF; i++) {
        row[i] = $i
    }

    # Output columns, replacing list columns with aggregated metrics or expanded columns
    first = 1
    for (j = 1; j <= NF; j++) {
        if (j in list_cols) {
            # Replace list column with mean, min, max, count, count_zero
            if (!first) printf "\t"
            aggs = compute_aggregations(j, row[j])
            printf "%s", aggs
            first = 0
        } else if (j in expand_cols) {
            # Expand fixed-size column into individual columns
            if (!first) printf "\t"
            expanded_vals = expand_column(row[j], expand_cols[j])
            printf "%s", expanded_vals
            first = 0
        } else {
            # Output non-list column as-is
            if (!first) printf "\t"
            printf "%s", row[j]
            first = 0
        }
    }
    printf "\n"
}
