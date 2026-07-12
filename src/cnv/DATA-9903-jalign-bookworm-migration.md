# CNV Docker: Bookworm Migration — Blocked by libparasail8

**Ticket:** DATA-9903
**Status:** Reverted — CNV pinned to `ugbio_base:1.7.0` (bullseye)
**Date:** 2026-07-12

## Decision

The CNV module remains on bullseye (`ugbio_base:1.7.0`) while all other modules migrate to bookworm (`ugbio_base:1.8.3`). The migration is blocked by incompatibilities between `para_jalign` and bookworm's `libparasail8`.

---

## Background

Migrating bullseye → bookworm changes the parasail library from `libparasail3` (soname 3) to `libparasail8` (soname 8). The `para_jalign` binary calls parasail's trace/alignment functions which behave differently between these library versions.

---

## Issues Encountered

### Issue 1: ABI Mismatch Segfault

**Symptom:** `para_jalign` (compiled on bullseye) segfaults with signal -11 when running on bookworm.

**Cause:** The bullseye-compiled binary links against `libparasail3`. On bookworm, it dynamically resolves to `libparasail8` which has different struct layouts and internal state, causing undefined behavior.

**Attempted fix:** Built a native bookworm binary (`para_jalign-1.4.1.1-debian-bookworm-x64`) compiled against `libparasail8`. This resolves the ABI mismatch for most regions but introduced new issues (see below).

---

### Issue 2: NO_CIGARS Env Var Corrupts Alignment Scores

**Symptom:** With `ENV NO_CIGARS=1`, the `align1.score` column contained `-32755` (uninitialized int16) for 129/150 rows. The `*.begin` fields were all zero.

**Cause:** The `NO_CIGARS` feature (commit `4b49a18` in jalign repo) skips parasail trace calls entirely. But `score1`/`score2` and `begin` fields are populated FROM the trace result inside `format_cigar()`. When trace is skipped, these remain uninitialized. The CNV scoring logic (`_evaluate_alignment_scores()`) uses these fields for read classification — so NO_CIGARS breaks CNV calling entirely.

**Resolution:** NO_CIGARS approach abandoned. Created jalign release 1.4.1.1 based on 1.4.1 (excluding the NO_CIGARS commit), with only the bookworm build target added.

---

### Issue 3: CIGAR and Query Sequence Lengths Differ

**Symptom:** CI tests fail with:
```
[E::bam_read1] CIGAR and query sequence lengths differ for 036742_1-Z0027-1180341078
OSError: error -4 while reading file
```

**Cause:** `libparasail8` produces CIGARs whose query-consuming operation lengths (M+I+S+=+X) do not always sum to the actual read length. This is a behavioral difference from `libparasail3`. When pysam writes such records to BAM and htslib reads them back, it rejects them as malformed.

**Attempted fix:** Added `_validate_cigar()` in `jalign.py` to check CIGAR length against query length and fall back to `"*"`. This fixed the immediate error but caused Issue 4.

---

### Issue 4: samtools sort Silently Drops Records

**Symptom:** WDL task on AWS HealthOmics processes 1400/1402 regions successfully, but post-processing fails:
```
samtools index: failed to create index for "*.jalign.sort.bam": No such file or directory
```

**Cause:** Records with `cigarstring="*"` but FLAG indicating **mapped** violate the SAM specification. `samtools sort` silently drops these invalid records, producing an empty BAM. The subsequent `samtools index` fails because the sorted BAM doesn't exist.

**Attempted fix:** Mark records with invalid CIGARs as unmapped (`FLAG=0x4`, `MAPQ=0`). This is SAM-spec compliant and samtools handles them correctly.

---

### Issue 5: Residual Segfaults (2/1402 Regions)

**Symptom:** Even with the native bookworm binary, `para_jalign` segfaults on 2 specific regions:
```
ERROR - Alignment tool failed with return code -11: ../src/para_jalign.cpp:399
```
Affected: `9:141152500-141215000`, `9:141153500-141214000`

**Cause:** Edge-case inputs that trigger a bug in parasail8's alignment path. The error message `"ERR failed to reference no 1 on lineno: 1"` indicates a bounds check issue in the C++ code.

**Status:** Not fixed — requires debugging the C++ alignment code in para_jalign.

---

## Steps Required to Complete the Migration

For a jalign specialist to enable bookworm support, the following steps are needed:

### 1. Fix libparasail8 CIGAR Generation

The core issue is that `libparasail8` generates CIGARs that don't match query length. Options:
- **A)** Debug why parasail8's trace function produces mismatched CIGARs and fix in para_jalign
- **B)** Report upstream to parasail if this is a library bug
- **C)** Pin a specific parasail version that produces correct CIGARs
- **D)** Accept the CIGAR workaround (validate + mark unmapped) — this is safe for CNV calling since only alignment scores are used, but produces BAM output with many unmapped records

### 2. Fix Residual Segfaults

Debug `para_jalign.cpp:399` for the 2 failing regions:
- Reproduce with inputs from regions `9:141152500-141215000` and `9:141153500-141214000`
- Likely a bounds check or null pointer issue in the alignment path

### 3. Validate Score Correctness

Verify that alignment scores from the bookworm binary match bullseye:
- Run jalign on the same input with both binaries
- Compare `align1.score`, `align2.score`, `*.begin` columns
- Confirm CNV classification (DUP_SUPPORT, DEL_SUPPORT) matches

### 4. Apply the Docker Migration

Once the above are resolved, the CNV Dockerfile changes are:

```dockerfile
# Build stage: bullseye → bookworm
FROM python:3.11-bookworm AS build

# Runtime: remove BASE_IMAGE pin (will use DEFAULT_BASE_IMAGE = ugbio_base:1.8.3)
ARG BASE_IMAGE

# Package changes:
# - libncurses5-dev + libncursesw5-dev → libncurses-dev
# - libstdc++-10-dev → libstdc++-12-dev
# - libparasail3 → libparasail8
# - R version: 4.5.1-1~bullseyecran.0 → 4.5.3-1~bookwormcran.0
# - CRAN repo: bullseye-cran40 → bookworm-cran40

# Jalign: update to version with bookworm binary
ARG JALIGN_VERSION=1.4.1.1
# Asset: para_jalign-1.4.1.1-debian-bookworm-x64
```

And in `jalign.py`, add CIGAR validation (if option D above is chosen):
```python
import re

_QUERY_CONSUMING_OPS = set("MISX=")
_CIGAR_RE = re.compile(r"(\d+)([MIDNSHP=X])")

def _validate_cigar(cigar: str, query_length: int) -> str:
    """Return cigar unchanged if valid, '*' otherwise."""
    if cigar == "*" or not cigar:
        return "*"
    try:
        consumed = sum(
            int(length) for length, op in _CIGAR_RE.findall(cigar)
            if op in _QUERY_CONSUMING_OPS
        )
        return cigar if consumed == query_length else "*"
    except Exception:
        return "*"
```

Then in `create_bam_record_from_alignment()`, mark invalid-CIGAR records as unmapped (`FLAG=0x4`, `MAPQ=0`).

### 5. Run Full Validation

- CI tests: `uv run pytest src/cnv/tests/`
- Docker build: `gh workflow run build-ugbio-member-docker.yml --ref <branch> -f member=cnv`
- WDL test on AWS HealthOmics with real sample

---

## Jalign Release History

| Version | Base | Status | Notes |
|---------|------|--------|-------|
| 1.4.1 | bullseye | **Current (production)** | Stable, used with ugbio_base:1.7.0 |
| 1.4.1.1 | bullseye + bookworm | Available | Clean build from 1.4.1 + bookworm target only |
| 1.4.2 | — | **Deleted** | Included NO_CIGARS commit that corrupts scores |

---

## Key Insight

The jalign scoring/classification logic uses **only alignment scores** (not CIGARs or begin positions). The CIGAR output in BAM is for visualization/downstream tools, not for CNV calling. This means option D (validate + mark unmapped) is functionally safe — but produces aesthetically incomplete BAM output. The ideal fix is at the parasail/para_jalign level.
