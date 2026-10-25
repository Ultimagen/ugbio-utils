# CNV Docker: Bookworm Migration

**Tickets:** DATA-9903, BIOIN-2945
**Status:** Migrated. CNV runs on bookworm `ugbio_base` with the distro `libparasail8` and a native bookworm `para_jalign` (no bullseye libraries).

## History

1. **2026-07-12:** The first bookworm attempt was reverted. CNV stayed on `ugbio_base:1.7.0` (bullseye), because `para_jalign` produced broken output with bookworm's `libparasail8`.
2. **Interim:** CNV moved to a bookworm base image but kept a workaround: it copied `libparasail.so*` and the headers from a `debian:bullseye` stage (parasail 2.4.3) and ran the bullseye `para_jalign` 1.4.1.
3. **BIOIN-2945:** The root cause was found and fixed in jalign. The bullseye stage was removed.

## Root cause

Bookworm ships parasail **2.6** (`libparasail8`); bullseye ships parasail **2.4.3** (`libparasail3`). `para_jalign` used the `parasail_sw_table_diag_16` and `parasail_sw_trace_diag_16` kernels. In parasail 2.6, **every diag kernel** is broken: 16- and 32-bit, SSE4.1 and AVX2, through the dispatcher or called directly. These kernels never update `score`, `end_query` or `end_ref` and return INT_MIN-based sentinels:

| | parasail 2.4.3 diag | parasail 2.6 diag | parasail 2.6 striped/scan |
|---|---|---|---|
| `score, end_query, end_ref` | `630, 339, 574` | `-32755, -32755, -32755` | `630, 339, 574` |
| cigar (`align1`) | `124M1I83M1D42M1I89M` | `-32754S33094S` | `124M1I83M1D42M1I89M` |

The DP score table itself is correct, so jump scores matched. Only the per-alignment score, begin position and cigar were corrupted. This one bug explains every symptom from the first attempt:

- **`align1.score == -32755`:** This was attributed to `NO_CIGARS`. The real cause is the diag kernel sentinel. The `NO_CIGARS` env var is unrelated.
- **`CIGAR and query sequence lengths differ`:** Cigars such as `-32754S33094S` came from the bogus end positions.
- **`samtools sort` dropping records:** A downstream effect of the cigar workaround (cigar `*` on mapped reads).

### "Residual segfaults" (`9:141152500-141215000`) are unrelated

The error `para_jalign.cpp:399 ERR failed to reference no 1 on lineno: 1` happens when a CNV's breakpoint window lies past the contig end. The window's reference is then empty, so the input line has no 4th field. jalign 1.4.1 also segfaults on such input on **bullseye**. This is fixed on the Python side, where `_fetch_reads_at_breakpoints` clamps windows to the contig length. Since jalign 1.4.2, para_jalign also exits with an error instead of crashing on malformed input.

## Fix (jalign 1.4.2, Ultimagen/jalign#20)

- Call the `parasail_sw_{table,trace}_striped_16` kernels instead of diag. They are correct on parasail 2.6 and about 30% faster.
- There is no 32-bit fallback. 16-bit scores only saturate for reads of about 16 kb at match=2, so para_jalign exits with an error if `match × read length > INT16_MAX`.
- Harden input parsing. para_jalign exits with an error on over-long lines or sequences, and on an `=` reference with no previous line.
- Build cigars in `std::string`. The fixed 4 KB buffer used to truncate long cigars.
- Fix undefined behavior and leaks: `bzero` of a struct holding `std::string`, `free()` instead of `parasail_result_free` on trace results, and leaked cigar and reference strings.
- Replace the `debian-bullseye` build/release target with `debian-bookworm`, linked dynamically against the distro `libparasail8`.

The CNV Dockerfile fetches release asset `para_jalign-${JALIGN_VERSION}-debian-bookworm-x64` (`JALIGN_VERSION=1.4.2`).

## Validation

Every output field (scores, readlens, begins, cigars) was compared between production jalign (diag kernels, parasail 2.4.3, bullseye) and the new jalign on bookworm (striped kernels, parasail 2.6):

| Input | Reads | Differences |
|---|---|---|
| `src/cnv/tests/resources/test.jalign.cram`, 4 CNV regions | 2,228 | none |
| Randomized: random, tandem-repeat and homopolymer refs (50–3000 bp), jump and non-jump reads (5–600 bp), indels and mismatches, 3 scoring schemes | 6,000 | none |

Other checks:

- Full `run_jalign` on the test CRAM gives VCF and CSV identical to `expected_test.jalign.{vcf.gz,csv}`.
- `uv run pytest src/cnv/tests/` passes, including `test_run_jalign_with_real_para_jalign_single_record`.
- The striped kernels are about 30% faster than diag (17 s vs 23 s on 2,763 reads).
