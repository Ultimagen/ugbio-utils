#!/usr/bin/awk -f
# AWK script for computing aggregation metrics (mean, min, max, count) for list format fields in VCF TSV output
# Usage: awk -v list_indices="3,4,5" -f aggregate_lists.awk input.tsv

BEGIN {
    # Parse list_indices parameter (0-based column indices, convert to 1-based for AWK)
    num_list_cols = split(list_indices, indices, ",")
    for (i = 1; i <= num_list_cols; i++) {
        col_idx = indices[i] + 1  # Convert to 1-based
        list_cols[col_idx] = 1
        ordered_list_cols[i] = col_idx
    }
}

function compute_aggregations(col_idx, values_str, values, n, i, val, sum, count, min_val, max_val, mean) {
    # Split the list (comma-separated values)
    n = split(values_str, values, ",")
    
    sum = 0
    count = 0
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
    
    # Return results as string: "mean\tmin\tmax\tcount"
    if (count == 0) {
        return ".\t.\t.\t0"
    }
    
    mean = sum / count
    return sprintf("%.6f\t%.6f\t%.6f\t%d", mean, min_val, max_val, count)
}

{
    # Store the current row
    for (i = 1; i <= NF; i++) {
        row[i] = $i
    }

    # Output columns, replacing list columns with aggregated metrics
    first = 1
    for (j = 1; j <= NF; j++) {
        if (j in list_cols) {
            # Replace list column with mean, min, max, count
            if (!first) printf "\t"
            aggs = compute_aggregations(j, row[j])
            printf "%s", aggs
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
