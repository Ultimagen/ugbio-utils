"""
End-to-end run of the depth CNV caller on HG002 chr19:19.45-19.95 Mb (612365, 40x PE), which holds a 5.2 kb
heterozygous deletion (T2T-Q100 truth: POS 19,723,634, SVLEN 5,219; also called by the SV caller).

Chain as in the pipeline: call_depth_cnvs -> split_read_cnv_candidates -> analyze_cnv_breakpoint_reads ->
snap_cnv_breakpoints -> filter_nested_cnv_candidates, then merge_cnv_sv with the SV calls; with the optional cn.mops
calls, cnmops_cnv_candidates after the split-read candidates.
"""

from pathlib import Path

import pysam
import pytest
from ugbio_cnv import analyze_cnv_breakpoint_reads, breakpoint_snapping, depth_cnv_caller
from ugbio_cnv import split_read_candidates as src
from ugbio_cnv.merge_cnv_sv import merge_cnv_sv_vcfs

SAMPLE = "NA24385"  # as in gridss_sv_example.vcf.gz (merge_cnv_sv needs the same sample)
TRUE_DEL = (19723635, 19728853)  # split-read junction, left-normalized: the deleted bases of the truth record


@pytest.fixture
def resources_dir():
    return Path(__file__).parent.parent / "resources"


@pytest.fixture
def inputs(resources_dir):
    d = resources_dir / "depth_cnv"
    prefix = d / "HG002.chr19_19.45-19.95M"
    return {
        "bam": str(prefix) + ".bam",
        "q0": str(prefix) + ".q0.regions.bed.gz",
        "q20": str(prefix) + ".q20.regions.bed.gz",
        "sv": str(resources_dir / "gridss_sv_example.vcf.gz"),
        "annotation": str(d / "annotation.chr19_19.4-20.0M.parquet"),
        "fai": str(resources_dir / "chr19.fasta.fai"),
        "fasta": str(resources_dir / "chr19.fasta"),
    }


def run_chain(inputs: dict, tmp_path: Path) -> str:
    t = str(tmp_path)
    depth_cnv_caller.run(
        [
            "call_depth_cnvs",
            "--depth-mq0",
            inputs["q0"],
            "--depth-mq20",
            inputs["q20"],
            "--annotation",
            inputs["annotation"],
            "--sample-name",
            SAMPLE,
            "--fasta-index",
            inputs["fai"],
            "--output-vcf",
            f"{t}/depth.vcf.gz",
        ]
    )
    src.run(
        [
            "split_read_cnv_candidates",
            "--alignment",
            inputs["bam"],
            "--reference-fasta",
            inputs["fasta"],
            "--depth-vcf",
            f"{t}/depth.vcf.gz",
            "--depth-mq0",
            inputs["q0"],
            "--depth-mq20",
            inputs["q20"],
            "--annotation",
            inputs["annotation"],
            "--output-vcf",
            f"{t}/cand.vcf.gz",
            "--threads",
            "1",
        ]
    )
    analyze_cnv_breakpoint_reads.main(
        [
            "--bam-file",
            inputs["bam"],
            "--vcf-file",
            f"{t}/cand.vcf.gz",
            "--reference-fasta",
            inputs["fasta"],
            "--cushion",
            "1500",
            "--output-file",
            f"{t}/ann.vcf.gz",
            "--output-bam",
            f"{t}/ev.bam",
        ]
    )
    pysam.sort("-o", f"{t}/ev.sort.bam", f"{t}/ev.bam")
    pysam.index(f"{t}/ev.sort.bam")
    breakpoint_snapping.run(
        [
            "snap_cnv_breakpoints",
            "--input-vcf",
            f"{t}/ann.vcf.gz",
            "--alignment",
            inputs["bam"],
            "--reference-fasta",
            inputs["fasta"],
            "--output-vcf",
            f"{t}/snap.vcf.gz",
            "--threads",
            "1",
        ]
    )
    src.run_nested(
        ["filter_nested_cnv_candidates", "--input-vcf", f"{t}/snap.vcf.gz", "--output-vcf", f"{t}/final.vcf.gz"]
    )
    return f"{t}/final.vcf.gz"


def test_depth_cnv_pipeline_finds_and_snaps_the_deletion(inputs, tmp_path):
    final = run_chain(inputs, tmp_path)
    with pysam.VariantFile(final) as v:
        assert list(v.header.samples) == [SAMPLE]
        meta = {r.key: r.value for r in v.header.records if r.type == "GENERIC"}
        assert meta["depth_cnv_karyotype"] == "not inferred (no chrX)"  # chr19 only: no sex chromosomes to decide
        dels = [r for r in v if r.info["SVTYPE"] == "DEL"]
    assert len(dels) == 1
    d = dels[0]
    assert (d.pos, d.stop) == TRUE_DEL  # snapped to the split-read junction
    assert d.info["SNAP"] == "SR"
    assert d.info["CNV_SOURCE"] == ("depth",)
    assert d.info["CNV_DEL_READS"] > 0
    assert d.info["CNV_DEL_PAIRS"] > 0
    assert abs(d.info["DEPTH_POS"] - TRUE_DEL[0]) <= 1000
    assert d.samples[SAMPLE]["GT"] == (0, 1)
    assert "PASS" in d.filter.keys()


def test_output_merges_with_sv_calls(inputs, tmp_path):
    final = run_chain(inputs, tmp_path)
    # FilterVCF writes the filter score to QUAL (--overwrite_qual_tag); emulate it so the merge can keep max QUAL
    scored = str(tmp_path / "scored.vcf.gz")
    with pysam.VariantFile(final) as vin, pysam.VariantFile(scored, "wz", header=vin.header) as vout:
        for r in vin:
            r.qual = 40.0
            vout.write(r)
    pysam.tabix_index(scored, preset="vcf", force=True)
    # the SV example is genome-wide; keep the test region (the reference index here is chr19 only)
    sv_region = str(tmp_path / "sv.region.vcf.gz")
    with pysam.VariantFile(inputs["sv"]) as vin, pysam.VariantFile(sv_region, "wz", header=vin.header) as vout:
        for r in vin.fetch("chr19", 19_450_000, 19_950_000):
            vout.write(r)
    pysam.tabix_index(sv_region, preset="vcf", force=True)
    merged = str(tmp_path / "merged.vcf.gz")
    merge_cnv_sv_vcfs(
        scored, sv_region, merged, inputs["fai"], min_sv_length=1000, max_sv_length=5_000_000, min_sv_qual=600
    )
    with pysam.VariantFile(merged) as v:
        dels = [
            r
            for r in v
            if r.info.get("SVTYPE") == "DEL"
            and "PASS" in r.filter.keys()
            and abs(r.stop - r.start) >= 1000
            and r.start < TRUE_DEL[1]
            and r.stop > TRUE_DEL[0]
        ]
    assert len(dels) == 1  # the CNV call and the SV call collapse into one record
    assert "depth" in dels[0].info["CNV_SOURCE"]


def test_pair_candidates_find_the_deletion_from_pairs(inputs, tmp_path):
    from ugbio_cnv import pair_candidates

    t = str(tmp_path)
    depth_cnv_caller.run(
        ["call_depth_cnvs", "--depth-mq0", inputs["q0"], "--depth-mq20", inputs["q20"],
         "--annotation", inputs["annotation"], "--sample-name", SAMPLE, "--fasta-index", inputs["fai"],
         "--output-vcf", f"{t}/depth.vcf.gz"]
    )  # fmt: skip
    # without the depth call of the deletion, the pairs alone must find it
    with (
        pysam.VariantFile(f"{t}/depth.vcf.gz") as vin,
        pysam.VariantFile(f"{t}/no_del.vcf.gz", "wz", header=vin.header) as vo,
    ):
        for r in vin:
            if not (r.info["SVTYPE"] == "DEL" and r.start < TRUE_DEL[1] and r.stop > TRUE_DEL[0]):
                vo.write(r)
    pysam.tabix_index(f"{t}/no_del.vcf.gz", preset="vcf", force=True)
    args = [inputs["bam"], inputs["fasta"]]
    tracks = [inputs["q0"], inputs["q20"], inputs["annotation"]]
    n = pair_candidates.pair_cnv_candidates(*args, f"{t}/no_del.vcf.gz", *tracks, f"{t}/pair.vcf.gz", ["chr19"], 1)
    assert n >= 1
    with pysam.VariantFile(f"{t}/pair.vcf.gz") as v:
        found = [r for r in v if r.id.startswith("PAIR_") and r.info["SVTYPE"] == "DEL"]
    hit = [r for r in found if abs(r.start - TRUE_DEL[0]) <= 600 and abs(r.stop - TRUE_DEL[1]) <= 600]
    assert len(hit) == 1
    assert hit[0].info["CNV_SOURCE"] == ("pair",) and hit[0].info["PAIR_READS"] >= 3
    # with the depth call present, the deletion is not added a second time
    pair_candidates.pair_cnv_candidates(*args, f"{t}/depth.vcf.gz", *tracks, f"{t}/pair2.vcf.gz", ["chr19"], 1)
    with pysam.VariantFile(f"{t}/pair2.vcf.gz") as v:
        again = [r for r in v if r.id.startswith("PAIR_") and r.start < TRUE_DEL[1] and r.stop > TRUE_DEL[0]]
    assert not again


def test_cnmops_calls_not_called_are_added(inputs, tmp_path):
    """A cn.mops call of the deletion is already called; one elsewhere becomes a candidate without a depth change."""
    run_chain(inputs, tmp_path)
    cnmops = str(tmp_path / "cnmops.vcf.gz")
    with pysam.VariantFile(str(tmp_path / "cand.vcf.gz")) as v:
        header = pysam.VariantHeader()
        for c in v.header.contigs.values():
            header.contigs.add(c.name, c.length)
        header.info.add("SVTYPE", 1, "String", "type")
        header.info.add("END", 1, "Integer", "end")
        header.add_sample(SAMPLE)
        with pysam.VariantFile(cnmops, "wz", header=header) as out:
            for s, e in ((19_600_000, 19_605_000), (TRUE_DEL[0] - 300, TRUE_DEL[1] + 700)):
                rec = out.new_record(contig="chr19", start=s, stop=e, alleles=("N", "<DEL>"))
                rec.info["SVTYPE"] = "DEL"
                out.write(rec)
    pysam.tabix_index(cnmops, preset="vcf", force=True)
    out = str(tmp_path / "with_cnmops.vcf.gz")
    src.run_cnmops(
        ["cnmops_cnv_candidates", "--input-vcf", str(tmp_path / "cand.vcf.gz"), "--cnmops-vcf", cnmops]
        + ["--depth-mq0", inputs["q0"], "--depth-mq20", inputs["q20"], "--annotation", inputs["annotation"]]
        + ["--output-vcf", out]
    )
    with pysam.VariantFile(out) as v:
        added = [(r.start, r.stop) for r in v if r.info["CNV_SOURCE"] == ("cn.mops",)]
    assert added == [(19_600_000, 19_605_000)]
