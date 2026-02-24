from __future__ import annotations

from collections import Counter, defaultdict

import pysam


def discover_bam_schema(bam_paths: list[str], sample_reads_per_bam: int = 20000) -> dict:
    tag_counts = Counter()
    tag_types = defaultdict(Counter)
    category_values = defaultdict(Counter)
    mapq_counts = Counter()
    read_len_counts = Counter()

    for bam_path in bam_paths:
        bam = pysam.AlignmentFile(bam_path, "rb")
        n = 0
        for rec in bam.fetch(until_eof=True):
            if rec.is_unmapped:
                continue
            n += 1
            read_len_counts[min(rec.query_length or 0, 400)] += 1
            mapq_counts[min(rec.mapping_quality, 60)] += 1
            for tag, value in rec.get_tags(with_value_type=False):
                tag_counts[tag] += 1
                tag_types[tag][type(value).__name__] += 1
                if isinstance(value, str) and len(value) <= 10:  # noqa: PLR2004
                    category_values[tag][value] += 1
            if n >= sample_reads_per_bam:
                break
        bam.close()

    tag_types_out = {k: dict(v) for k, v in tag_types.items()}
    category_values_out = {k: [x[0] for x in v.most_common()] for k, v in category_values.items()}
    schema = {
        "schema_version": 1,
        "bam_paths": bam_paths,
        "sample_reads_per_bam": sample_reads_per_bam,
        "tag_counts": dict(tag_counts),
        "tag_types": tag_types_out,
        "category_values": category_values_out,
        "st_distribution": dict(category_values.get("st", {})),
        "et_distribution": dict(category_values.get("et", {})),
        "tag_semantics": {
            "tm": "Trimming reasons: A=Adapter, Q=Quality, Z=Three Zeroes",
            "st": "Name of pattern matched in start loop segment (ppmSeq)",
            "et": "Name of pattern matched in end loop segment (ppmSeq)",
            "rq": "Read quality float tag",
            "tp": "Per-position integer-like array",
            "t0": "Per-position quality-like string",
        },
        "mapq_distribution": dict(mapq_counts),
        "read_len_distribution": dict(read_len_counts),
    }
    return schema
