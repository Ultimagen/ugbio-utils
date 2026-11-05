"""Coverage-oriented unit tests for ugbio_srsnv.trinuc_histogram_plotting (small synthetic inputs)."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402
from ugbio_srsnv.trinuc_histogram_plotting import (  # noqa: E402
    _get_trinuc_with_alt_in_order,
    _resolve_qual_group_specs,
    calc_and_plot_trinuc_hist,
    calc_trinuc_stats,
    plot_trinuc_hist,
    plot_trinuc_hist_and_qual_panels,
    plot_trinuc_hist_panels,
    plot_trinuc_qual,
    reverse_engineer_hist_stats,
)

GROUPS = ["simplex SE", "simplex PE", "duplex SE", "duplex PE"]


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


@pytest.fixture
def group_df():
    rng = np.random.default_rng(0)
    trinucs = _get_trinuc_with_alt_in_order("symmetric")[0]
    n = 2000
    return pd.DataFrame(
        {
            "tcwa": rng.choice(trinucs, n),
            "label": rng.choice([False, True], n),
            "is_forward": rng.choice([True, False], n),
            "SNVQ": rng.normal(50, 10, n).clip(0, 90),
            "read_group": pd.Categorical(rng.choice(GROUPS, n), categories=GROUPS, ordered=True),
            "is_cycle_skip": rng.choice([True, False], n),
        }
    )


def test_trinuc_order_validation_and_reverse_layout():
    with pytest.raises(ValueError, match="order must be"):
        _get_trinuc_with_alt_in_order("bogus")
    ref_alt, index, snv_labels = _get_trinuc_with_alt_in_order("reverse")
    assert len(ref_alt) == len(index) == 192
    assert len(set(ref_alt)) == 192
    # in "reverse" order position i+96 is the reverse complement of position i
    comp = {"A": "T", "C": "G", "G": "C", "T": "A"}
    for i in (0, 17, 95):
        c1, r, c2, a = ref_alt[i]
        assert ref_alt[i + 96] == comp[c2] + comp[r] + comp[c1] + comp[a]
    assert snv_labels[0] == "A > C" and len(snv_labels) == 12


def test_resolve_qual_group_specs_all_and_empty():
    all_only = pd.DataFrame({"mixed=all median_qual": [1.0]})
    assert _resolve_qual_group_specs(all_only) == [("mixed=all", "All reads", "tab:blue")]
    assert _resolve_qual_group_specs(all_only, None, {"mixed=all": "k"}) == [("mixed=all", "All reads", "k")]
    assert _resolve_qual_group_specs(pd.DataFrame({"x": [1]})) == []
    # explicit specs are filtered to the prefixes actually present
    specs = [("mixed=a", "A", "r"), ("mixed=b", "B", "g")]
    assert _resolve_qual_group_specs(pd.DataFrame({"mixed=b median_qual": [1.0]}), specs) == [("mixed=b", "B", "g")]


def test_plot_trinuc_hist_defaults_create_axis():
    stats = {False: np.ones(192), True: np.arange(192, dtype=float)}
    ax, bars = plot_trinuc_hist(stats, add_annotations=False)
    assert set(bars) == {False, True}
    assert len(bars[True].patches) == 96
    assert ax.get_ylim()[1] == pytest.approx(1.05 * 191)
    assert len(ax.texts) == 0  # no SNV annotations


def test_plot_trinuc_qual_defaults_create_axis(group_df):
    stats = calc_trinuc_stats(group_df, trinuc_col="tcwa", labels=[True], include_quality=True, collapsed=False)
    ax, lines = plot_trinuc_qual(stats, add_annotations=False)
    assert "mixed=all" in lines
    assert len(ax.texts) == 0


def test_calc_trinuc_stats_cycle_skip_and_no_labels(group_df):
    stats = calc_trinuc_stats(
        group_df,
        trinuc_col="tcwa",
        labels=None,
        include_quality=False,
        include_cycle_skip=True,
        collapsed=True,
        motif_orientation="ref_dir",
    )
    assert stats.shape[0] == 96
    assert "is_cycle_skip" in stats.columns
    assert any(c.startswith("all reads (") for c in stats.columns)
    # missing cycle-skip column -> filled with None
    stats2 = calc_trinuc_stats(
        group_df.drop(columns=["is_cycle_skip"]),
        trinuc_col="tcwa",
        labels=[False, True],
        include_cycle_skip=True,
        collapsed=False,
        motif_orientation="fwd_only",
    )
    assert stats2.shape[0] == 192
    assert stats2["is_cycle_skip"].isna().all()
    # fwd_only keeps only forward reads
    n_fwd = int(group_df["is_forward"].sum())
    count_cols = [c for c in stats2.columns if c.endswith(")")]
    assert sum(int(c.split("(")[1].rstrip(")")) for c in count_cols) == n_fwd


def test_reverse_engineer_hist_stats_rejects_bad_columns():
    with pytest.raises(ValueError, match="expected format"):
        reverse_engineer_hist_stats(pd.DataFrame({"bad column": [1.0]}))


def test_calc_and_plot_with_group_specs_legend_labels(group_df):
    """N-group split: quality legend shows the group_specs display labels in order (not mixed=... names)."""
    specs = [(f"mixed={g}", g, c) for g, c in zip(GROUPS, ["gray", "purple", "orange", "green"], strict=True)]
    fig, stats = calc_and_plot_trinuc_hist(
        group_df,
        trinuc_col="tcwa",
        labels=[True, False],
        include_quality=True,
        collapsed=False,
        split_col="read_group",
        group_specs=specs,
        motif_orientation="ref_dir",
    )
    for g in GROUPS:
        assert f"mixed={g} median_qual" in stats.columns
    qual_legend = fig.legends[1]
    assert [t.get_text() for t in qual_legend.get_texts()] == GROUPS
    hist_legend = [t.get_text() for t in fig.legends[0].get_texts()]
    n_tp = int(group_df["label"].sum())
    assert f"TP ({n_tp})" in hist_legend
    assert len(fig.axes) == 4  # 2 panel pairs (non-collapsed)


def test_hist_and_qual_panels_unlabeled_reads(group_df):
    stats = calc_trinuc_stats(
        group_df, trinuc_col="tcwa", labels=None, include_quality=False, include_cycle_skip=False, collapsed=True
    )
    fig = plot_trinuc_hist_and_qual_panels(stats, collapsed=None, motif_orientation="fwd_only")
    labels = [t.get_text() for t in fig.legends[0].get_texts()]
    assert labels == [f"all reads ({len(group_df)})"]  # non TP/FP label kept verbatim
    assert len(fig.axes) == 2  # collapsed inferred from 96 rows


def test_hist_panels_inferred_collapse_and_labels(group_df):
    stats = calc_trinuc_stats(
        group_df,
        trinuc_col="tcwa",
        labels=[True, False],
        include_quality=False,
        include_cycle_skip=False,
        collapsed=False,
    )
    fig = plot_trinuc_hist_panels(stats, collapsed=None, motif_orientation="ref_dir", figsize=(18, 12))
    assert len(fig.axes) == 2
    texts = [t.get_text() for t in fig.legends[0].get_texts()]
    assert any(t.startswith("TP (") for t in texts) and any(t.startswith("FP (") for t in texts)
    fig2 = plot_trinuc_hist_panels(
        stats.rename(columns={c: c.replace("True (", "all reads (") for c in stats.columns}),
        collapsed=False,
        motif_orientation="fwd_only",
    )
    assert any(t.get_text().startswith("all reads (") for t in fig2.legends[0].get_texts())
