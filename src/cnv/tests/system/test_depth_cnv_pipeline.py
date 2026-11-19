"""
End-to-end run of the depth CNV caller on HG002 chr19:49.4-51.4 Mb (612365, 40x PE). T2T-Q100 truth in the region:
- a 4.6 kb heterozygous deletion (POS 50,574,632, SVLEN 4,622; also called by the SV caller), refined by split reads;
- a 1.6 kb homozygous deletion (POS 50,903,499, SVLEN 1,576), refined by read pairs;
- tandem duplications / insertions of 2-97 kb at 50.09-50.14 Mb (copy number about 5), called from depth;
- tandem-repeat expansions at 49.46 Mb (0.9-1.7 kb), found as duplications from read pairs only.
The depth caller also calls two deletions without truth (50.50 and 51.32 Mb), left for the filter model.

Chain as in the pipeline: call_depth_cnvs -> split_read_cnv_candidates -> analyze_cnv_breakpoint_reads ->
refine_cnv_breakpoints -> filter_nested_cnv_candidates, then merge_cnv_sv with the SV calls; with the optional cn.mops
calls, cnmops_cnv_candidates after the split-read candidates.
"""

from pathlib import Path

import pysam
import pytest
from ugbio_cnv import analyze_cnv_breakpoint_reads, breakpoint_refinement, depth_cnv_caller
from ugbio_cnv import split_read_candidates as src
from ugbio_cnv.merge_cnv_sv import merge_cnv_sv_vcfs

SAMPLE = "NA24385"  # as in gridss_sv_example.vcf.gz (merge_cnv_sv needs the same sample)
TRUE_DEL = (50574633, 50579254)  # split-read junction, left-normalized: the deleted bases of the truth record
HOM_DEL = (50903556, 50905041)  # pair junction of the homozygous deletion
DUP_REGION = (50_090_000, 50_140_000)  # the duplications called from depth
PAIR_DUPS = ((49461052, 49463638), (49461062, 49461589))  # the tandem-repeat expansions found by read pairs


@pytest.fixture
def resources_dir():
    return Path(__file__).parent.parent / "resources"


@pytest.fixture
def inputs(resources_dir):
    d = resources_dir / "depth_cnv"
    prefix = d / "HG002.chr19_49.4-51.4M"
    return {
        "bam": str(prefix) + ".bam",
        "q0": str(prefix) + ".q0.regions.bed.gz",
        "q20": str(prefix) + ".q20.regions.bed.gz",
        "sv": str(resources_dir / "gridss_sv_example.vcf.gz"),
        "annotation": str(d / "annotation.chr19_49.35-51.45M.parquet"),
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
    breakpoint_refinement.run(
        [
            "refine_cnv_breakpoints",
            "--input-vcf",
            f"{t}/ann.vcf.gz",
            "--alignment",
            inputs["bam"],
            "--reference-fasta",
            inputs["fasta"],
            "--output-vcf",
            f"{t}/refined.vcf.gz",
            "--threads",
            "1",
        ]
    )
    src.run_nested(
        ["filter_nested_cnv_candidates", "--input-vcf", f"{t}/refined.vcf.gz", "--output-vcf", f"{t}/final.vcf.gz"]
    )
    return f"{t}/final.vcf.gz"


def test_depth_cnv_pipeline_finds_and_refines_the_deletions(inputs, tmp_path):
    final = run_chain(inputs, tmp_path)
    with pysam.VariantFile(final) as v:
        assert list(v.header.samples) == [SAMPLE]
        meta = {r.key: r.value for r in v.header.records if r.type == "GENERIC"}
        assert meta["depth_cnv_karyotype"] == "not inferred (no chrX)"  # chr19 only: no sex chromosomes to decide
        calls = list(v)
    dels = [r for r in calls if r.info["SVTYPE"] == "DEL"]
    assert len(dels) == 4
    d = next(r for r in dels if r.start < TRUE_DEL[1] and r.stop > TRUE_DEL[0])
    assert (d.pos, d.stop) == TRUE_DEL  # refined to the split-read junction
    assert d.info["BREAKPOINT_SOURCE"] == "split_read"
    assert d.info["CNV_SOURCE"] == ("depth",)
    assert d.info["CNV_DEL_READS"] > 0
    assert d.info["CNV_DEL_PAIRS"] > 0
    assert abs(d.info["DEPTH_START"] - TRUE_DEL[0]) <= 1000
    assert d.samples[SAMPLE]["GT"] == (0, 1)
    assert "PASS" in d.filter.keys()
    h = next(r for r in dels if r.start < HOM_DEL[1] and r.stop > HOM_DEL[0])
    assert (h.pos, h.stop) == HOM_DEL  # refined to the read-pair junction
    assert h.info["BREAKPOINT_SOURCE"] == "pair"
    assert h.samples[SAMPLE]["GT"] == (1, 1)
    dups = [r for r in calls if r.info["SVTYPE"] == "DUP"]
    assert dups and all(DUP_REGION[0] <= r.start and r.stop <= DUP_REGION[1] for r in dups)
    assert all(r.info["CNV_SOURCE"] == ("depth",) and r.info["CN_MEAN"] > 4 for r in dups)
    assert sum(r.stop - r.start for r in dups) > 40_000


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
        for r in vin.fetch("chr19", 49_400_000, 51_400_000):
            vout.write(r)
    pysam.tabix_index(sv_region, preset="vcf", force=True)
    merged = str(tmp_path / "merged.vcf.gz")
    merge_cnv_sv_vcfs(
        scored, sv_region, merged, inputs["fai"], min_sv_length=500, max_sv_length=5_000_000, min_sv_qual=600
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


def test_pair_candidates_find_the_deletion_and_duplications_from_pairs(inputs, tmp_path):
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
    assert hit[0].info["CNV_SOURCE"] == ("pair",) and hit[0].info["CLUSTER_SUPPORT"] >= 3
    # the tandem-repeat expansions have no depth call: only the pairs find them
    dups = sorted((r.start, r.stop) for r in v_dups(f"{t}/pair.vcf.gz"))
    assert len(dups) == len(PAIR_DUPS)
    assert all(abs(a - b) <= 100 for got, exp in zip(dups, sorted(PAIR_DUPS)) for a, b in zip(got, exp))
    # with the depth call present, the deletion is not added a second time
    pair_candidates.pair_cnv_candidates(*args, f"{t}/depth.vcf.gz", *tracks, f"{t}/pair2.vcf.gz", ["chr19"], 1)
    with pysam.VariantFile(f"{t}/pair2.vcf.gz") as v:
        again = [r for r in v if r.id.startswith("PAIR_") and r.start < TRUE_DEL[1] and r.stop > TRUE_DEL[0]]
    assert not again


def v_dups(path: str) -> list:
    """Duplications added by the pair candidates."""
    with pysam.VariantFile(path) as v:
        return [r for r in v if r.id.startswith("PAIR_") and r.info["SVTYPE"] == "DUP"]


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
            for s, e in ((TRUE_DEL[0] - 300, TRUE_DEL[1] + 700), (51_100_000, 51_105_000)):
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
    assert added == [(51_100_000, 51_105_000)]
