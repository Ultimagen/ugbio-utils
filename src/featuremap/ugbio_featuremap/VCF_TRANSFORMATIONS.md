# VCF Transformations in Somatic Featuremap Fields Transformation

This document lists all transformations applied to the input VCF file, with references to the exact locations in the code where each transformation occurs.

## Overview

The script transforms a somatic featuremap VCF through multiple stages:
1. Tandem repeat annotation
2. Filtering
3. Field extraction and aggregation
4. XGBoost inference (optional)
5. Writing transformed fields back to VCF

---

## 1. Tandem Repeat Features Integration

**Location:** `run()` function, lines 669-671  
**Function:** `integrate_tandem_repeat_features()` (imported from `somatic_featuremap_utils`)

### Added INFO Fields:
- **TR_START** - Start position of the closest tandem repeat
- **TR_END** - End position of the closest tandem repeat  
- **TR_SEQ** - Sequence of the tandem repeat unit
- **TR_LENGTH** - Total length of the tandem repeat region
- **TR_SEQ_UNIT_LENGTH** - Length of the tandem repeat unit
- **TR_DISTANCE** - Distance from variant to the closest tandem repeat

**Reference:** Lines 669-671, and see `somatic_featuremap_utils.integrate_tandem_repeat_features()` for implementation details.

---

## 2. VCF Filtering

**Location:** `featuremap_fields_aggregation()` function, lines 388-398

### Transformations:
- **Filter by filter tags** - Applies `-f {filter_tags}` filter using `vcf_utils.view_vcf()`
  - Line 388: `filter_string = f"-f {filter_tags}" if filter_tags else ""`
  - Line 396-397: `vcf_utils.view_vcf()` with filter_string
  
- **Filter by genomic region** - Applies `-r {genomic_region}` interval filter
  - Line 389: `interval_string = f"-r {genomic_region}" if genomic_region else ""`
  - Line 396-397: `vcf_utils.view_vcf()` with interval_string

- **Index VCF** - Creates tabix index for filtered VCF
  - Line 399: `vcf_utils.index_vcf(filtered_featuremap)`

**Reference:** Lines 387-399

---

## 3. VCF to DataFrame Conversion

**Location:** `read_merged_tumor_normal_vcf()` function, lines 289-344

### Transformations:

#### 3.1 Column Name Lowercasing
- **Action:** Converts all column names to lowercase
- **Location:** Lines 324-329
- **Code:**
  ```python
  for colname in df_tumor.columns:
      df_tumor[colname.lower()] = df_tumor[colname]
      df_normal[colname.lower()] = df_normal[colname]
      if colname != colname.lower():
          df_tumor = df_tumor.drop(columns=[colname])
          df_normal = df_normal.drop(columns=[colname])
  ```

#### 3.2 Sample Prefix Addition
- **Action:** Adds `t_` prefix to tumor columns and `n_` prefix to normal columns
- **Location:** Line 334
- **Code:** `pd.concat([df_tumor.add_prefix("t_"), df_normal.add_prefix("n_")], axis=1)`

#### 3.3 Record Index Addition
- **Action:** Adds `record_index` field to preserve original VCF record order
- **Location:** Line 332
- **Code:** `df_tumor[ORIGINAL_RECORD_INDEX_FIELD] = range(1, len(df_tumor) + 1)`

#### 3.4 Missing Value Filling
- **Action:** Fills missing values if `fillna_dict` is provided
- **Location:** Lines 337-342
- **Code:** Creates merged fillna_dict with t_ and n_ prefixes and applies `fillna()`

**Reference:** Lines 289-344

---

## 4. Field Transformations in DataFrame

**Location:** `df_sfm_fields_transformation()` function, lines 164-189

### 4.1 Sample Column Processing
- **Location:** Line 184 (called for both `t_` and `n_` prefixes)
- **Function:** `process_sample_columns()` (lines 63-161)

#### 4.1.1 Alternative Reads Extraction
- **Action:** Extracts alternative read count from AD (Allele Depth) field
- **Location:** Line 137
- **Code:** `df_variants[f"{prefix}alt_reads"] = [tup[1] for tup in df_variants[f"{prefix}ad"]]`
- **Output Field:** `{prefix}alt_reads` (e.g., `t_alt_reads`, `n_alt_reads`)

#### 4.1.2 Pass Alternative Reads Calculation
- **Action:** Sums filtered reads from FILT field
- **Location:** Lines 139-141
- **Code:** 
  ```python
  df_variants[f"{prefix}pass_alt_reads"] = df_variants[f"{prefix}{FeatureMapFields.FILT.value.lower()}"].apply(
      lambda x: sum(x) if x is not None and len(x) > 0 and None not in x else float("nan")
  )
  ```
- **Output Field:** `{prefix}pass_alt_reads`

#### 4.1.3 Quality Score Aggregations
- **Action:** Calculates min, max, and mean for MQUAL, SNVQ, and MAPQ fields
- **Location:** Lines 143-145 (calls `add_agg_features()`, lines 84-94)
- **Aggregated Fields:**
  - `{prefix}mqual_min`, `{prefix}mqual_max`, `{prefix}mqual_mean`
  - `{prefix}snvq_min`, `{prefix}snvq_max`, `{prefix}snvq_mean`
  - `{prefix}mapq_min`, `{prefix}mapq_max`, `{prefix}mapq_mean`
- **Reference:** Lines 84-94 (add_agg_features function)

#### 4.1.4 Duplicate Read Counting
- **Action:** Counts duplicate and non-duplicate reads from DUP field
- **Location:** Line 148 (calls `parse_is_duplicate()`, lines 96-105)
- **Output Fields:**
  - `{prefix}count_duplicate` - Sum of duplicate flags
  - `{prefix}count_non_duplicate` - Count of non-duplicate reads (where value == 0)
- **Reference:** Lines 96-105

#### 4.1.5 Padding Reference Counts Expansion
- **Action:** Expands `ref_counts_pm_2` and `nonref_counts_pm_2` arrays into individual columns
- **Location:** Lines 149-153 (calls `parse_padding_ref_counts()`, lines 107-133)
- **Output Fields:**
  - `{prefix}ref0`, `{prefix}ref1`, `{prefix}ref2`, ... (one per padding position)
  - `{prefix}nonref0`, `{prefix}nonref1`, `{prefix}nonref2`, ... (one per padding position)
- **Reference:** Lines 107-133

#### 4.1.6 Forward/Reverse Strand Counting
- **Action:** Counts forward (0) and reverse (1) strand reads from REV field
- **Location:** Lines 155-159
- **Code:**
  ```python
  df_variants[[f"{prefix}forward_count", f"{prefix}reverse_count"]] = df_variants[f"{prefix}rev"].apply(
      lambda x: pd.Series({"num0": 0, "num1": 0})
      if x is None or (isinstance(x, float) and pd.isna(x))
      else pd.Series({"num0": [v for v in x if v in (0, 1)].count(0), "num1": [v for v in x if v in (0, 1)].count(1)})
  )
  ```
- **Output Fields:** `{prefix}forward_count`, `{prefix}reverse_count`

### 4.2 Allele Extraction
- **Action:** Extracts reference and alternative alleles from `t_alleles` tuple
- **Location:** Lines 185-186
- **Code:**
  ```python
  df_variants["ref_allele"] = [tup[0] for tup in df_variants["t_alleles"]]
  df_variants["alt_allele"] = [tup[1] for tup in df_variants["t_alleles"]]
  ```
- **Output Fields:** `ref_allele`, `alt_allele`

### 4.3 Normal Depth Calculation
- **Action:** Fills missing `n_dp` values with sum of `n_ref2` and `n_nonref2`
- **Location:** Line 187
- **Code:** `df_variants["n_dp"] = df_variants["n_dp"].fillna(df_variants["n_ref2"] + df_variants["n_nonref2"])`

**Reference:** Lines 164-189

---

## 5. XGBoost Inference (Optional)

**Location:** `featuremap_fields_aggregation()` function, lines 411-415

### Transformations:
- **Action:** Loads XGBoost model and predicts probabilities for each variant
- **Location:** Lines 411-415
- **Code:**
  ```python
  if xgb_model_file is not None:
      xgb_clf_es = somatic_featuremap_inference_utils.load_model(xgb_model_file)
      model_features = xgb_clf_es.get_booster().feature_names
      logger.info(f"loaded model. model features: {model_features}")
      df_variants["xgb_proba"] = somatic_featuremap_inference_utils.predict(xgb_clf_es, df_variants)
  ```
- **Output Field:** `xgb_proba` (in DataFrame)

**Reference:** Lines 411-415

---

## 6. Parquet File Export

**Location:** `featuremap_fields_aggregation()` function, lines 417-419

### Transformations:
- **Action:** Exports transformed DataFrame to Parquet format
- **Location:** Lines 417-419
- **Code:**
  ```python
  parquet_output = output_vcf.replace(".vcf.gz", "_featuremap.parquet")
  df_variants.to_parquet(parquet_output, index=False)
  ```

**Reference:** Lines 417-419

---

## 7. VCF Header Modifications

**Location:** `add_fields_to_header()` function, lines 192-212

### Added FORMAT Fields:
All fields defined in `added_format_features` dictionary (lines 41-54):
- **ALT_READS** - Number of supporting reads for the alternative allele (Integer)
- **PASS_ALT_READS** - Number of passed supporting reads for the alternative allele (Integer)
- **MQUAL_MEAN** - Mean value of MQUAL (Float)
- **SNVQ_MEAN** - Mean value of SNVQ (Float)
- **MQUAL_MAX** - Maximum value of MQUAL (Float)
- **SNVQ_MAX** - Maximum value of SNVQ (Float)
- **MQUAL_MIN** - Minimum value of MQUAL (Float)
- **SNVQ_MIN** - Minimum value of SNVQ (Float)
- **COUNT_DUPLICATE** - Number of duplicate reads (Integer)
- **COUNT_NON_DUPLICATE** - Number of non-duplicate reads (Integer)
- **REVERSE_COUNT** - Number of reverse strand reads (Integer)
- **FORWARD_COUNT** - Number of forward strand reads (Integer)

**Location:** Lines 205-208

### Added INFO Fields:
All fields defined in `added_info_features` dictionary (lines 55-58):
- **REF_ALLELE** - Reference allele (String)
- **ALT_ALLELE** - Alternative allele (String)

**Location:** Lines 209-212

### Optional XGB_PROBA INFO Field:
- **XGB_PROBA** - XGBoost model predicted probability (Float)
- **Location:** Lines 425-426, 439-440 (conditional on xgb_model_file)

**Reference:** Lines 192-212, 424, 425-426, 438-440

---

## 8. VCF Record Writing with New Fields

**Location:** `process_vcf_records_serially()` function, lines 215-286

### Transformations:

#### 8.1 INFO Field Addition
- **Action:** Adds REF_ALLELE and ALT_ALLELE to each VCF record's INFO field
- **Location:** Lines 264-265
- **Code:**
  ```python
  for key in added_info_features:
      vcf_row.info[key.upper()] = getattr(current_df_record, key.lower())
  ```
- **Fields Added:** `REF_ALLELE`, `ALT_ALLELE`

#### 8.2 FORMAT Field Addition
- **Action:** Adds aggregated FORMAT fields to both tumor (sample 0) and normal (sample 1) samples
- **Location:** Lines 268-280
- **Code:**
  ```python
  for key in added_format_features:
      tumor_value = getattr(current_df_record, f"t_{key.lower()}")
      normal_value = getattr(current_df_record, f"n_{key.lower()}")
      
      if pd.notna(tumor_value):
          vcf_row.samples[0][key.upper()] = tumor_value
      else:
          vcf_row.samples[0][key.upper()] = None
      
      if pd.notna(normal_value):
          vcf_row.samples[1][key.upper()] = normal_value
      else:
          vcf_row.samples[1][key.upper()] = None
  ```
- **Fields Added:** All 12 FORMAT fields listed in section 7, for both tumor and normal samples

#### 8.3 XGB_PROBA Addition (Optional)
- **Action:** Adds XGBoost probability to INFO field if model was used
- **Location:** Lines 283-284
- **Code:**
  ```python
  if "XGB_PROBA" in hdr.info and hasattr(current_df_record, "xgb_proba"):
      vcf_row.info["XGB_PROBA"] = current_df_record.xgb_proba
  ```

**Reference:** Lines 215-286

---

## 9. Final VCF Processing

**Location:** `featuremap_fields_aggregation()` function, line 431

### Transformations:
- **Action:** Creates tabix index for the output VCF
- **Location:** Line 431
- **Code:** `pysam.tabix_index(output_vcf, preset="vcf", min_shift=0, force=True)`

**Reference:** Line 431

---

## 10. Interval Merging (Parallel Processing)

**Location:** `featuremap_fields_aggregation_on_an_interval_list()` function, lines 579-585

### Transformations:
- **Action:** Concatenates interval-specific VCF files, sorts, and indexes
- **Location:** Lines 579-585
- **Code:**
  ```python
  cmd = (
      f"bcftools concat -f {interval_list_file} -a | "
      f"bcftools sort - -Oz -o {output_vcf} && "
      f"bcftools index -t {output_vcf}"
  )
  ```

**Reference:** Lines 579-585

---

## Summary of All VCF Modifications

### INFO Fields Added:
1. TR_START, TR_END, TR_SEQ, TR_LENGTH, TR_SEQ_UNIT_LENGTH, TR_DISTANCE (from tandem repeat integration)
2. REF_ALLELE, ALT_ALLELE (extracted from alleles)
3. XGB_PROBA (optional, from XGBoost inference)

### FORMAT Fields Added:
1. ALT_READS (extracted from AD)
2. PASS_ALT_READS (summed from FILT)
3. MQUAL_MEAN, MQUAL_MAX, MQUAL_MIN (aggregated from MQUAL)
4. SNVQ_MEAN, SNVQ_MAX, SNVQ_MIN (aggregated from SNVQ)
5. COUNT_DUPLICATE, COUNT_NON_DUPLICATE (counted from DUP)
6. REVERSE_COUNT, FORWARD_COUNT (counted from REV)

### VCF Structure Changes:
- Filtering applied (filter_tags and genomic_region)
- Records sorted and indexed
- Header modified to include new field definitions

### Data Transformations (in DataFrame, not directly in VCF):
- Column names lowercased
- Sample prefixes added (t_/n_)
- Padding counts expanded into individual columns
- Quality scores aggregated (min/max/mean)
- Normal depth calculated from ref/nonref counts


