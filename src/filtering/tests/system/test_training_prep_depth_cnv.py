import pickle

import numpy as np
import pandas as pd
import pysam
import pytest
from ugbio_filtering import train_models_pipeline
from ugbio_filtering import training_prep_depth_cnv_pipeline as tp

CONTIGS = {"chr1": 5_000_000, "chr2": 5_000_000}
INFO = [
    ("SVTYPE", "1", "String"),
    ("SVLEN", "1", "Integer"),
    ("END", "1", "Integer"),
    ("CN_MEAN", "1", "Float"),
    ("BASE_CN", "1", "Integer"),
    ("Z", "1", "Float"),
    ("N_MERGED", "1", "Integer"),
    ("SEGDUP", "1", "Float"),
    ("UNIQ_FRAC", "1", "Float"),
    ("LOW_CONF", "0", "Flag"),
    ("CNV_SOURCE", ".", "String"),
    ("SR_READS", "1", "Integer"),
    ("SNAP", "1", "String"),
    ("SNAP_N", "1", "Integer"),
    ("DEPTH_POS", "1", "Integer"),
    ("DEPTH_END", "1", "Integer"),
    ("CNV_DEL_READS", "1", "Integer"),
    ("CNV_DUP_READS", "1", "Integer"),
    ("CNV_TOTAL_READS", "1", "Integer"),
    ("CNV_DEL_FRAC", "1", "Float"),
    ("CNV_DUP_FRAC", "1", "Float"),
    ("CNV_DEL_PAIRS", "1", "Integer"),
    ("CNV_DUP_PAIRS", "1", "Integer"),
    ("CNV_DEL_PAIRS_FRAC", "1", "Float"),
    ("CNV_DUP_PAIRS_FRAC", "1", "Float"),
    ("DEL_READS_MEDIAN_INSERT_SIZE", "1", "Float"),
    ("DUP_READS_MEDIAN_INSERT_SIZE", "1", "Float"),
    ("DEL_PAIRS_MEDIAN_SPAN", "1", "Float"),
    ("DUP_PAIRS_MEDIAN_SPAN", "1", "Float"),
]


def header() -> pysam.VariantHeader:
    h = pysam.VariantHeader()
    for c, n in CONTIGS.items():
        h.contigs.add(c, length=n)
    for tag, number, kind in INFO:
        h.info.add(tag, number, kind, tag)
    h.add_line('##ALT=<ID=DEL,Description="Deletion">')
    h.formats.add("GT", 1, "String", "Genotype")
    h.add_sample("S")
    return h


def write_vcf(path, records):
    with pysam.VariantFile(path, "wz", header=header()) as out:
        for k, (chrom, start, end, extra) in enumerate(records):
            r = out.new_record(contig=chrom, start=start, stop=end, alleles=("N", "<DEL>"), id=f"C{k}")
            r.filter.add("PASS")
            r.info["SVTYPE"], r.info["SVLEN"] = "DEL", -(end - start)
            for tag, value in extra.items():
                r.info[tag] = value
            r.samples["S"]["GT"] = (0, 1)
            out.write(r)
    pysam.tabix_index(path, preset="vcf", force=True)
    return path


@pytest.fixture
def inputs(tmp_path):
    rng = np.random.default_rng(0)
    truth, calls = [], []
    for chrom in CONTIGS:
        for k in range(30):
            start = 100_000 + k * 100_000
            end = start + int(rng.integers(1000, 20000))
            real = k % 2 == 0
            if real:
                truth.append((chrom, start, end, {}))
            calls.append(
                (
                    chrom,
                    start,
                    end,
                    {
                        "CN_MEAN": 1.0 if real else 1.7,
                        "BASE_CN": 2,
                        "Z": -15.0 if real else -5.5,
                        "N_MERGED": 1,
                        "SEGDUP": 0.0,
                        "UNIQ_FRAC": 0.95,
                        "CNV_SOURCE": ("depth",),
                        "SNAP": "SR" if real else "none",
                        "SNAP_N": 8 if real else 0,
                        "DEPTH_POS": start + 1,
                        "DEPTH_END": end,
                        "CNV_DEL_READS": int(rng.integers(5, 15)) if real else 0,
                        "CNV_DUP_READS": 0,
                        "CNV_TOTAL_READS": 500,
                        "CNV_DEL_FRAC": 0.02 if real else 0.0,
                        "CNV_DUP_FRAC": 0.0,
                        "CNV_DEL_PAIRS": int(rng.integers(3, 10)) if real else 0,
                        "CNV_DUP_PAIRS": 0,
                        "CNV_DEL_PAIRS_FRAC": 0.01 if real else 0.0,
                        "CNV_DUP_PAIRS_FRAC": 0.0,
                    },
                )
            )
    bed = tmp_path / "benchmark.bed"
    bed.write_text("".join(f"{c}\t0\t{n}\n" for c, n in CONTIGS.items()))
    return {
        "calls": write_vcf(str(tmp_path / "calls.vcf.gz"), calls),
        "truth": write_vcf(str(tmp_path / "truth.vcf.gz"), truth),
        "bed": str(bed),
    }


def test_labels_split_and_train(inputs, tmp_path):
    prefix = str(tmp_path / "depth_cnv")
    train, test = tp.training_prep_depth_cnv(inputs["calls"], inputs["truth"], inputs["bed"], ["chr1"], prefix)
    assert len(train) == 30 and len(test) == 30  # chr1 / chr2
    assert set(train["chrom"]) == {"chr1"} and set(test["chrom"]) == {"chr2"}
    assert train["label"].sum() == 15  # every second call matches the truth
    assert (train.set_index("pos")["label"] == (train.set_index("pos")["cnv_del_reads"] > 0).astype(int)).all()
    assert pd.read_hdf(f"{prefix}.train.h5").shape[0] == 30

    out = str(tmp_path / "model")
    train_models_pipeline.run(
        ["train_models_pipeline", "--train_dfs", f"{prefix}.train.h5", "--test_dfs", f"{prefix}.test.h5",
         "--output_file_prefix", out, "--vcf_type", "cnv_depth", "--gt_type", "approximate"]
    )  # fmt: skip
    with open(f"{out}.pkl", "rb") as f:
        model = pickle.load(f)
    assert {"xgb", "transformer", "decision_threshold"} <= set(model)
    assert isinstance(model["decision_threshold"], int)
    scores = model["xgb"].predict_proba(model["transformer"].transform(test))[:, 1]
    assert ((scores > 0.5) == test["label"].astype(bool)).mean() >= 0.9


def test_known_event_positives(inputs, tmp_path):
    events = tmp_path / "events.bed"
    events.write_text("chr2\t100000\t200000\tDEL\nchr2\t300000\t400000\tDUP\n")  # one DEL event, one DUP event
    pos = tp.known_event_positives(inputs["calls"], str(events))
    assert list(pos["chrom"]) == ["chr2"] and list(pos["pos"]) == [100_001]  # the DUP event has no DEL call
    assert set(pos["label"]) == {1}

    prefix = str(tmp_path / "with_events")
    train, _ = tp.training_prep_depth_cnv(
        inputs["calls"], inputs["truth"], inputs["bed"], ["chr1"], prefix, known_events=[(inputs["calls"], str(events))]
    )
    assert len(train) == 31

    # a given decision threshold is stored instead of the out-of-bag F1 one
    out26 = str(tmp_path / "model26")
    train_models_pipeline.run(
        ["train_models_pipeline", "--train_dfs", f"{prefix}.train.h5", "--test_dfs", f"{prefix}.test.h5",
         "--output_file_prefix", out26, "--vcf_type", "cnv_depth", "--gt_type", "approximate",
         "--decision_threshold", "26"]
    )  # fmt: skip
    with open(f"{out26}.pkl", "rb") as f:
        assert pickle.load(f)["decision_threshold"] == 26


def test_calls_sharing_a_start_keep_their_own_end(tmp_path):
    calls = write_vcf(str(tmp_path / "calls.vcf.gz"), [("chr1", 100_000, 105_000, {}), ("chr1", 100_000, 160_000, {})])
    read = tp.read_calls(calls).sort_values("id")
    assert list(read["id"]) == ["C0", "C1"] and list(read["end"]) == [105_000, 160_000]
