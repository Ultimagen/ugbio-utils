import re
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import gridspec

from ugbio_srsnv.srsnv_utils import FLOW_ORDER, get_trinuc_context_with_alt_fwd_vectorized, is_cycle_skip

# Constants
IS_FORWARD = "is_forward"
TRINUC_FORWARD_COUNT = 96
X_VALUES = list(range(TRINUC_FORWARD_COUNT))
HIST_COLORS = {
    "False": "tab:blue",
    "True": "tab:orange",
    False: "tab:blue",
    True: "tab:orange",
    "all reads": "tab:blue",
}


def _get_trinuc_with_alt_in_order(order: str = "symmetric"):
    """Get trinuc_with_context in right order, so that each SNV in position i
    has the opposite SNV in position i+96.
    E.g., in position 0 we have A[A>C]A and in posiiton 96 we have A[C>A]A
    """
    if order not in {"symmetric", "reverse"}:
        raise ValueError(f'order must be either "symmetric" or "reverse". Got {order}')
    if order == "symmetric":
        trinuc_ref_alt = [
            c1 + r + c2 + a
            for r, a in (("A", "C"), ("A", "G"), ("A", "T"), ("C", "G"), ("C", "T"), ("G", "T"))
            for c1 in ("A", "C", "G", "T")
            for c2 in ("A", "C", "G", "T")
            if r != a
        ] + [
            c1 + r + c2 + a
            for a, r in (("A", "C"), ("A", "G"), ("A", "T"), ("C", "G"), ("C", "T"), ("G", "T"))
            for c1 in ("A", "C", "G", "T")
            for c2 in ("A", "C", "G", "T")
            if r != a
        ]
    elif order == "reverse":
        trinuc_ref_alt = [
            c1 + r + c2 + a
            for r, a in (("A", "C"), ("A", "G"), ("A", "T"), ("C", "G"), ("C", "T"), ("G", "T"))
            for c1 in ("A", "C", "G", "T")
            for c2 in ("A", "C", "G", "T")
            if r != a
        ] + [
            c1 + r + c2 + a
            for r, a in (("T", "G"), ("T", "C"), ("T", "A"), ("G", "C"), ("G", "A"), ("C", "A"))
            for c2 in ("T", "G", "C", "A")
            for c1 in ("T", "G", "C", "A")
            if r != a
        ]
    trinuc_index = np.array([f"{t[0]}[{t[1]}>{t[3]}]{t[2]}" for t in trinuc_ref_alt])
    snv_labels = [" ".join(trinuc_snv[2:5]) for trinuc_snv in trinuc_index[np.arange(0, 16 * 12, 16)]]
    return trinuc_ref_alt, trinuc_index, snv_labels


def plot_trinuc_hist(  # noqa: C901
    hist_stats,
    labels=None,
    panel_num=0,
    ax=None,
    ylim=None,
    x_values=None,
    hist_colors=None,
    xtick_fontsize=10,
    order="symmetric",
    *,
    add_annotations=True,
):
    if labels is None:
        labels = [False, True]
    if x_values is None:
        x_values = X_VALUES
    if hist_colors is None:
        hist_colors = HIST_COLORS

    trinuc_symmetric_ref_alt, symmetric_index, snv_labels = _get_trinuc_with_alt_in_order(order=order)
    trinuc_is_cycle_skip = np.array([is_cycle_skip(tcwa, flow_order=FLOW_ORDER) for tcwa in trinuc_symmetric_ref_alt])
    if ax is None:
        fig, ax = plt.subplots(figsize=(16, 3))
    inds = np.array(x_values) + 96 * panel_num
    bars = {}
    # Plot histogram bars
    for label in labels:
        bars[label] = ax.bar(x_values, hist_stats[label][inds], color=hist_colors[f"{label}"], width=1.0, alpha=0.5)
    if ylim is None:
        ylim = 1.05 * max([hist_stats[label].max() for label in labels])
        # _, ylim = ax.get_ylim()
    # Plot separators for ref, alt pair-groups
    for j in range(5):
        ax.plot([(j + 1) * 16 - 0.5] * 2, [0, ylim], "k--")
    ax.set_xticks(x_values, symmetric_index[inds], rotation=90, fontsize=xtick_fontsize)
    tick_label_colors = ["green" if trinuc_is_cycle_skip[j] else "red" for j in inds]
    for j in x_values:
        ax.get_xticklabels()[j].set_color(tick_label_colors[j])
    ax.tick_params(axis="x", pad=-2)
    ax.grid(visible=True)
    ax.set_xlim([-0.5, 95.5])
    ax.set_ylim([0, ylim])
    ax.grid(visible=True, axis="both", alpha=0.75, linestyle=":")

    # Add ref > alt annotations (optional)
    if add_annotations:
        snv_positions = [8, 24, 40, 56, 72, 88]  # Midpoint for each SNV titles in plot
        i = panel_num
        for label, pos in zip(snv_labels[6 * i : 6 * i + 6], snv_positions, strict=False):
            ax.annotate(
                label,
                xy=(pos, ylim),  # Position at the top of the plot
                xytext=(-2, 6),  # Offset from the top of the plot
                textcoords="offset points",
                ha="center",
                fontsize=12,
                fontweight="bold",
            )
    return ax, bars


def plot_trinuc_qual(  # noqa: C901, PLR0912, PLR0915
    stats_df,
    panel_num=0,
    ax=None,
    x_values=None,
    xtick_fontsize=10,
    order="symmetric",
    qual_colors=None,
    *,
    add_annotations=True,
):
    """Plot trinucleotide quality statistics.

    Parameters
    ----------
    stats_df : pd.DataFrame
        DataFrame with quality statistics in new format (mixed=all, mixed=True, mixed=False)
    panel_num : int
        Panel number (0 or 1 for forward/reverse)
    ax : matplotlib.Axes, optional
        Axis to plot on
    x_values : list, optional
        X-axis values
    xtick_fontsize : int
        Font size for x-tick labels
    order : str
        Trinucleotide ordering
    qual_colors : dict, optional
        Colors for quality lines

    Returns
    -------
    ax : matplotlib.Axes
        The axis
    lines : dict
        Dictionary of line objects for legend
    """
    if x_values is None:
        x_values = X_VALUES
    if qual_colors is None:
        qual_colors = {"mixed=all": "tab:blue", "mixed=True": "tab:green", "mixed=False": "tab:red"}

    trinuc_symmetric_ref_alt, symmetric_index, snv_labels = _get_trinuc_with_alt_in_order(order=order)
    trinuc_is_cycle_skip = np.array([is_cycle_skip(tcwa, flow_order=FLOW_ORDER) for tcwa in trinuc_symmetric_ref_alt])

    if ax is None:
        fig, ax = plt.subplots(figsize=(16, 3))

    inds = np.array(x_values) + 96 * panel_num
    lines = {}

    # Determine which mixed categories to plot
    # Check if we have mixed=True and mixed=False columns
    has_mixed_true = any("mixed=True median_qual" in col for col in stats_df.columns)
    has_mixed_false = any("mixed=False median_qual" in col for col in stats_df.columns)
    has_mixed_all = any("mixed=all median_qual" in col for col in stats_df.columns)

    # Determine which categories to plot based on priority
    if has_mixed_true and has_mixed_false:
        # If we have both True and False, plot only these (not 'all')
        mixed_categories = ["mixed=True", "mixed=False"]
    elif has_mixed_all:
        # If we only have 'all', plot only this
        mixed_categories = ["mixed=all"]
    else:
        # No quality data available
        mixed_categories = []

    # Create extended x_values for step plotting (like in calc_and_plot_trinuc_plot)
    x_values_ext = (
        [x_values[0] - (x_values[1] - x_values[0]) / 2] + x_values + [x_values[-1] + (x_values[-1] - x_values[-2]) / 2]
    )
    inds_ext = [inds[0]] + list(inds) + [inds[-1]]

    # Plot quality lines and shaded areas for each mixed category
    for mixed_cat in mixed_categories:
        median_col = f"{mixed_cat} median_qual"
        q1_col = f"{mixed_cat} q1_qual"
        q2_col = f"{mixed_cat} q2_qual"

        if median_col in stats_df.columns:
            color = qual_colors.get(mixed_cat, "black")

            # Get data for this panel (extended for step plotting)
            median_data = stats_df[median_col].iloc[inds_ext]
            q1_data = stats_df[q1_col].iloc[inds_ext] if q1_col in stats_df.columns else median_data
            q2_data = stats_df[q2_col].iloc[inds_ext] if q2_col in stats_df.columns else median_data

            # Plot shaded area for quantile range
            ax.fill_between(x_values_ext, q1_data, q2_data, step="mid", color=color, alpha=0.2)

            # Plot median line using step plot (like in reference implementation)
            (line,) = ax.step(
                x_values_ext,
                median_data,
                where="mid",
                color=color,
                alpha=0.7,
                label=(
                    mixed_cat.replace("mixed=", "")
                    .replace("all", "All reads")
                    .replace("True", "Mixed reads")
                    .replace("False", "Non-mixed reads")
                ),
            )
            lines[mixed_cat] = line

    # Set x-tick labels and colors based on cycle skip
    ax.set_xticks(x_values)
    ax.set_xticklabels(symmetric_index[inds], rotation=90, fontsize=xtick_fontsize)
    tick_label_colors = ["green" if trinuc_is_cycle_skip[j] else "red" for j in inds]
    for j, _ in enumerate(x_values):
        ax.get_xticklabels()[j].set_color(tick_label_colors[j])

    ax.tick_params(axis="x", pad=-2)
    ax.grid(visible=True, axis="both", alpha=0.75, linestyle=":")
    ax.set_xlim([-0.5, 95.5])

    # Set y-limits based on data range
    if mixed_categories:
        all_q1s = []
        all_q2s = []
        for mixed_cat in mixed_categories:
            q1_col = f"{mixed_cat} q1_qual"
            q2_col = f"{mixed_cat} q2_qual"
            if q1_col in stats_df.columns:
                all_q1s.extend(stats_df[q1_col].dropna().values)
            if q2_col in stats_df.columns:
                all_q2s.extend(stats_df[q2_col].dropna().values)
        if all_q1s and all_q2s:
            snvq_min = min(all_q1s)
            snvq_max = max(all_q2s)
            ymin = snvq_min - 0.03 * (snvq_max - snvq_min)
            ymax = snvq_max + 0.03 * (snvq_max - snvq_min)
            ax.set_ylim([ymin, ymax])

    # Plot separators for ref, alt pair-groups
    ylim_min, ylim_max = ax.get_ylim()
    for j in range(5):
        ax.plot([(j + 1) * 16 - 0.5] * 2, [ylim_min, ylim_max], "k--")

    # Add ref > alt annotations (optional)
    if add_annotations:
        snv_positions = [8, 24, 40, 56, 72, 88]
        i = panel_num
        for label, pos in zip(snv_labels[6 * i : 6 * i + 6], snv_positions, strict=False):
            ylim = ax.get_ylim()[1]
            ax.annotate(
                label,
                xy=(pos, ylim),
                xytext=(-2, 6),
                textcoords="offset points",
                ha="center",
                fontsize=12,
                fontweight="bold",
            )

    return ax, lines


def calc_trinuc_stats(  # noqa: C901, PLR0912, PLR0913, PLR0915
    plot_df,
    trinuc_col="tcwa_fwd",
    label_col="label",
    is_forward_col="is_forward",
    qual_col="SNVQ",
    cycle_skip_col="is_cycle_skip",
    is_mixed_col="is_mixed",
    order: str = "symmetric",
    labels=None,
    q1: float = 0.1,
    q2: float = 0.9,
    motif_orientation: str = "seq_dir",
    *,
    collapsed: bool = False,
    include_quality: bool = True,
    include_cycle_skip: bool = True,
):
    """Calculate trinuc stats (histogram and optionally quality) for each label, return as DataFrame.

    Parameters
    ----------
    plot_df : pd.DataFrame
        Input dataframe with trinucleotide and quality data
    trinuc_col : str
        Column name containing trinucleotide context with alt
    label_col : str
        Column name for grouping labels
    is_forward_col : str
        Column name for forward/reverse orientation
    qual_col : str
        Column name for quality values
    cycle_skip_col : str
        Column name for cycle skip information
    order : str
        Order of trinucleotides ("symmetric" or "reverse")
    labels : list, optional
        List of labels to include
    q1, q2 : float
        Quantiles for quality statistics
    collapsed : bool
        Whether to collapse forward/reverse into single values
    motif_orientation : str
        "seq_dir": as read in sequencing direction,
        "ref_dir": aligned to reference direction by reverse-complementing reverse-strand reads, or
        "fwd_only": using only forward-strand reads.
    include_quality : bool
        Whether to include quality statistics
    include_cycle_skip : bool
        Whether to include cycle skip statistics

    Returns
    -------
    pd.DataFrame
        DataFrame with histogram counts, quality stats, and cycle skip info indexed by trinucleotide context
    """
    trinuc_symmetric_ref_alt, symmetric_index, snv_labels = _get_trinuc_with_alt_in_order(order=order)
    plot_df = plot_df.copy()
    if motif_orientation not in {"seq_dir", "ref_dir", "fwd_only"}:
        raise ValueError(f"{motif_orientation=} is not one of the allowed values {{'seq_dir', 'ref_dir', 'fwd_only'}}.")
    if motif_orientation == "fwd_only":
        plot_df = plot_df.loc[plot_df[is_forward_col], :]
    elif motif_orientation == "seq_dir":
        plot_df[trinuc_col] = get_trinuc_context_with_alt_fwd_vectorized(plot_df[trinuc_col], plot_df[is_forward_col])
    plot_df["trinuc_context_with_alt_int"] = plot_df[trinuc_col].map(
        {trinuc: i for i, trinuc in enumerate(trinuc_symmetric_ref_alt)}
    )

    if labels is None:
        plot_df[label_col] = "all reads"
        labels = ["all reads"]

    stats_data = {}
    total_snvs_per_label = {}

    for label in labels:
        hist_cond = plot_df[label_col] == label

        # Calculate histogram stats
        counts = (
            plot_df.loc[hist_cond, "trinuc_context_with_alt_int"]
            .value_counts()
            .reindex(range(2 * TRINUC_FORWARD_COUNT), fill_value=0)
            .sort_index()
            .to_numpy()
        )
        total_snvs_per_label[label] = counts.sum()
        normed = counts / counts.sum() if counts.sum() > 0 else counts
        if collapsed:
            normed = normed[:TRINUC_FORWARD_COUNT] + normed[TRINUC_FORWARD_COUNT:]

        # Store histogram data
        hist_col_name = f"{label} ({total_snvs_per_label[label]})"
        stats_data[hist_col_name] = normed

    # Calculate quality stats if requested
    if include_quality and qual_col in plot_df.columns:
        qual_cond = plot_df[label_col]
        qual_subset = plot_df.loc[qual_cond].copy()

        if not qual_subset.empty:
            stats_to_calculate = {
                "median_qual": "median",
                "q1_qual": lambda x: x.quantile(q1),
                "q2_qual": lambda x: x.quantile(q2),
                "count": "count",
            }

            # Calculate stats on all data first
            all_qual_stats = (
                qual_subset.groupby("trinuc_context_with_alt_int")[qual_col]
                .agg(list(stats_to_calculate.values()))
                .reindex(range(TRINUC_FORWARD_COUNT * 2), fill_value=np.nan)
            )

            # Store all data stats with 'all' prefix
            for i, (stat_key, _) in enumerate(stats_to_calculate.items()):
                stat_array = all_qual_stats.iloc[:, i].to_numpy()
                if collapsed:
                    # Average forward and reverse for collapsed mode
                    stat_array = (stat_array[:TRINUC_FORWARD_COUNT] + stat_array[TRINUC_FORWARD_COUNT:]) / 2
                stats_data[f"mixed=all {stat_key}"] = stat_array

            if is_mixed_col in qual_subset.columns:
                # Calculate stats for each is_mixed value slice
                for is_mixed in qual_subset[is_mixed_col].unique():
                    mixed_subset = qual_subset[qual_subset[is_mixed_col] == is_mixed]
                    if not mixed_subset.empty:
                        mixed_qual_stats = (
                            mixed_subset.groupby("trinuc_context_with_alt_int")[qual_col]
                            .agg(list(stats_to_calculate.values()))
                            .reindex(range(TRINUC_FORWARD_COUNT * 2), fill_value=np.nan)
                        )

                        # Store mixed-specific stats
                        for i, (stat_key, _) in enumerate(stats_to_calculate.items()):
                            stat_array = mixed_qual_stats.iloc[:, i].to_numpy()
                            if collapsed:
                                # Average forward and reverse for collapsed mode
                                stat_array = (stat_array[:TRINUC_FORWARD_COUNT] + stat_array[TRINUC_FORWARD_COUNT:]) / 2
                            stats_data[f"mixed={is_mixed} {stat_key}"] = stat_array

    # Calculate cycle skip stats if requested
    if include_cycle_skip and cycle_skip_col in plot_df.columns:
        cycle_skip_stats = (
            plot_df.groupby("trinuc_context_with_alt_int")[cycle_skip_col].mean().reindex(range(192), fill_value=np.nan)
        )
        if collapsed:
            cycle_skip_stats = (
                cycle_skip_stats.iloc[:TRINUC_FORWARD_COUNT].to_numpy()
                + cycle_skip_stats.iloc[TRINUC_FORWARD_COUNT:].to_numpy()
            ) / 2
        stats_data["is_cycle_skip"] = cycle_skip_stats.to_numpy() if not collapsed else cycle_skip_stats
    elif include_cycle_skip:
        # Column doesn't exist, fill with None/NaN
        length = TRINUC_FORWARD_COUNT if collapsed else TRINUC_FORWARD_COUNT * 2
        stats_data["is_cycle_skip"] = [None] * length

    # Set index
    if collapsed:
        index = symmetric_index[:TRINUC_FORWARD_COUNT]
    else:
        index = symmetric_index

    # was:
    # hist_stats_df = pd.DataFrame(
    #     {f'{label} ({total_snvs_per_label[label]})': hist_stats[label] for label in labels},
    #     index=index
    # )
    # return hist_stats_df
    stats_df = pd.DataFrame(stats_data, index=index)
    return stats_df


def reverse_engineer_hist_stats(hist_stats_df: pd.DataFrame):
    labels = []
    hist_stats = {}
    total_snvs_per_label = {}

    for col in hist_stats_df.columns:
        # Extract label and count using regex
        match = re.match(r"^(.*?)\s*\((\d+)\)$", col)
        if match:
            label, count = match.group(1), int(match.group(2))
        else:
            raise ValueError(f"Column name '{col}' is not in the expected format 'label (count)'")

        labels.append(label)
        hist_stats[label] = hist_stats_df[col].to_numpy()
        total_snvs_per_label[label] = count

    return labels, hist_stats, total_snvs_per_label


def plot_trinuc_hist_and_qual_panels(  # noqa: C901, PLR0912, PLR0913, PLR0915
    stats_df,
    q1=0.1,
    q2=0.9,
    order="symmetric",
    suptitle=None,
    figsize=None,
    collapsed=None,
    ytick_fontsize=12,
    xtick_fontsize=10,
    ylabel_fontsize=16,
    hspace=0.1,
    bottom_scale=1.0,
    hist_to_qual_height_ratio=2.0,
    motif_orientation="seq_dir",
):
    """Plot trinuc histogram and quality panels from stats_df DataFrame.

    Layout:
    - For collapsed mode: Quality panel above histogram panel
    - For non-collapsed mode: Quality1, Hist1, [space], Quality2, Hist2
    """
    # Extract histogram labels from columns
    hist_labels = []
    hist_stats = {}
    total_snvs_per_label = {}

    for col in stats_df.columns:
        if not col.endswith(("_median_qual", "_q1_qual", "_q2_qual")) and col != "is_cycle_skip":
            # Extract label and count using regex
            match = re.match(r"^(.*?)\s*\((\d+)\)$", col)
            if match:
                label, count = match.group(1), int(match.group(2))
                hist_labels.append(label)
                hist_stats[label] = stats_df[col].to_numpy()
                total_snvs_per_label[label] = count

    # Infer collapsed if not given
    hist_len = stats_df.shape[0]
    if collapsed is None:
        collapsed = hist_len == TRINUC_FORWARD_COUNT

    # Determine layout
    if figsize is None:
        figsize = (18, 5) if collapsed else (18, 10)
    fig = plt.figure(figsize=figsize)
    height_ratios = [1, hist_to_qual_height_ratio]  # Configurable ratio

    if collapsed:
        # Simple case: Quality above histogram with no space
        gs = gridspec.GridSpec(2, 1, height_ratios=height_ratios, hspace=0.0)
    else:
        # Complex case: Need custom spacing between panels
        # Create two separate gridspecs for the two panel pairs
        gs_top = gridspec.GridSpec(
            2, 1, height_ratios=height_ratios, hspace=0.0, top=0.95, bottom=0.5 + hspace / 2
        )  # Top half of figure
        gs_bottom = gridspec.GridSpec(
            2, 1, height_ratios=height_ratios, hspace=0.0, top=0.5 - hspace / 2, bottom=0.05
        )  # Bottom half of figure

    # Create panels
    axes = []
    bars_for_legend = {}
    lines_for_legend = {}

    panels_to_plot = 1 if collapsed else 2

    for panel_idx in range(panels_to_plot):
        if collapsed:
            qual_ax = fig.add_subplot(gs[0])
            hist_ax = fig.add_subplot(gs[1], sharex=qual_ax)
        elif panel_idx == 0:
            # First panel pair (top)
            qual_ax = fig.add_subplot(gs_top[0])
            hist_ax = fig.add_subplot(gs_top[1], sharex=qual_ax)
        else:
            # Second panel pair (bottom)
            qual_ax = fig.add_subplot(gs_bottom[0])
            hist_ax = fig.add_subplot(gs_bottom[1], sharex=qual_ax)

        # Plot quality panel
        _, lines = plot_trinuc_qual(
            stats_df, panel_num=panel_idx, ax=qual_ax, order=order, xtick_fontsize=xtick_fontsize, add_annotations=True
        )
        qual_ax.set_ylabel("SNVQ", fontsize=ylabel_fontsize)
        qual_ax.tick_params(axis="y", labelsize=ytick_fontsize)

        # Remove x-tick labels from quality panel since it shares x-axis
        qual_ax.set_xticklabels([])

        # Plot histogram panel
        _, bars = plot_trinuc_hist(
            hist_stats,
            labels=hist_labels,
            panel_num=panel_idx,
            ax=hist_ax,
            order=order,
            xtick_fontsize=xtick_fontsize,
            add_annotations=False,
        )
        hist_ax.set_ylabel("Density", fontsize=ylabel_fontsize)
        hist_ax.tick_params(axis="y", labelsize=ytick_fontsize)

        # Store for legend (only need one set)
        if panel_idx == 0:
            bars_for_legend = bars
            lines_for_legend = lines

        axes.extend([qual_ax, hist_ax])

    # Create legends with improved labels
    handles_hist, labels_hist = [], []
    handles_qual, labels_qual = [], []

    for label in hist_labels:
        if label in bars_for_legend:
            handles_hist.append(bars_for_legend[label].patches[0])
            # Determine TP/FP based on label
            if str(label).lower() in ["true", "1", 1, True]:
                label_name = "TP"
            elif str(label).lower() in ["false", "0", 0, False]:
                label_name = "FP"
            else:
                label_name = str(label)
            labels_hist.append(f"{label_name} ({total_snvs_per_label[label]})")

    for mixed_cat in lines_for_legend:
        handles_qual.append(lines_for_legend[mixed_cat])
        clean_label = (
            mixed_cat.replace("mixed=", "")
            .replace("all", "All reads")
            .replace("True", "Mixed reads")
            .replace("False", "Non-mixed reads")
        )
        labels_qual.append(clean_label)

    # Apply bottom_scale to layout
    bottom_margin = 0.15 * bottom_scale
    fig.tight_layout(rect=[0, bottom_margin, 1, 1])

    # Position legends with bottom_scale
    legend_y_offset = (-0.08 if collapsed else -0.165) * bottom_scale  # text_y
    # legend_y_offset = text_y - 0.04* bottom_scale  # Adjusted for legend position
    # legend_y_offset = -0.08 * bottom_scale
    if collapsed:
        hist_bbox = (0.24, legend_y_offset)
        qual_bbox = (0.5, legend_y_offset)
    else:
        hist_bbox = (0.24, legend_y_offset * 0.75)
        qual_bbox = (0.5, legend_y_offset * 0.75)

    fig.legend(
        handles_hist,
        labels_hist,
        title="Histogram: Labels (# SNVs total)",
        fontsize=14,
        title_fontsize=14,
        loc="lower center",
        bbox_to_anchor=hist_bbox,
        ncol=1,
        frameon=False,
    )

    fig.legend(
        handles_qual,
        labels_qual,
        title=f"SNVQ on TP (median + {int(q1*100)}%-{int(q2*100)}% range)",
        fontsize=14,
        title_fontsize=14,
        loc="lower center",
        bbox_to_anchor=qual_bbox,
        ncol=1,
        frameon=False,
    )

    # Explanatory text with bottom_scale
    if motif_orientation == "fwd_only":
        is_forward_text = "(forward reads only)"
    elif motif_orientation == "seq_dir":
        is_forward_text = "(sequencing direction)"
    elif motif_orientation == "ref_dir":
        is_forward_text = "(reference direction)"
    # For collapsed=False case, align bottom of explanatory text with bottom of legends
    if not collapsed:
        # Calculate text positions so bottom aligns with legend bottom
        legend_bottom = legend_y_offset * 0.75  # Same as legend position
        text_y_top = legend_bottom + 0.075  # Top line position (bottom + 3 lines * line_spacing)
        text_y_middle = legend_bottom + 0.045  # Middle line position
        text_y_bottom = legend_bottom + 0.02  # Bottom line position (aligned with legend bottom)
    else:
        # Keep original positioning for collapsed case (to be adjusted later)
        legend_bottom = legend_y_offset * 0.75
        text_y_top = legend_bottom + 0.12  # Top line position (bottom + 3 lines * line_spacing)
        text_y_middle = legend_bottom + 0.07  # Middle line position
        text_y_bottom = legend_bottom + 0.02  # Bottom line position (aligned with legend bottom)

    fig.text(
        0.7,  # x-position (slightly to the right of the second legend)
        text_y_top,  # y-position (adjust as necessary)
        f"Trinuc-SNV {is_forward_text}:",
        ha="left",  # Horizontal alignment
        fontsize=14,  # Font size
        color="black",  # Text color
    )
    fig.text(
        0.73,  # x-position (slightly to the right of the second legend)
        text_y_middle,  # y-position (adjust as necessary)
        "Green: Cycle skip",
        ha="left",  # Horizontal alignment
        fontsize=14,  # Font size
        color="green",  # Text color
    )
    fig.text(
        0.73,  # x-position (slightly to the right of the second legend)
        text_y_bottom,  # y-position (adjust as necessary)
        "Red: No cycle skip",
        ha="left",  # Horizontal alignment
        fontsize=14,  # Font size
        color="red",  # Text color
    )
    if suptitle:
        fig.suptitle(suptitle, fontsize=20)

    return fig


def plot_trinuc_hist_panels(  # noqa: C901, PLR0912, PLR0915
    hist_stats_df,
    order="symmetric",
    suptitle=None,
    figsize=None,
    collapsed=None,
    ytick_fontsize=12,
    xtick_fontsize=10,
    ylabel_fontsize=16,
    bottom_scale=1.0,
    motif_orientation: str = "seq_dir",
):
    """Plot trinuc histogram panels from hist_stats_df DataFrame."""
    # Reverse engineer labels, hist_stats dict, and total_snvs_per_label
    labels, hist_stats, total_snvs_per_label = reverse_engineer_hist_stats(hist_stats_df)
    # Infer collapsed if not given
    hist_len = hist_stats_df.shape[0]  # len(next(iter(hist_stats.values())))
    if collapsed is None:
        collapsed = hist_len == TRINUC_FORWARD_COUNT
    figrows = 1 if collapsed else 2
    if figsize is None:
        figsize = (18, 4) if collapsed else (18, 8)
    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(figrows, 1, height_ratios=[1] * figrows, hspace=0.4)
    ax_hist_orig = fig.add_subplot(gs[0])
    _, bars = plot_trinuc_hist(
        hist_stats, labels=labels, panel_num=0, ax=ax_hist_orig, order=order, xtick_fontsize=xtick_fontsize
    )
    ax_hist_orig.set_ylabel("Density", fontsize=ylabel_fontsize)  # <-- Add ylabel
    ax_hist_orig.tick_params(axis="y", labelsize=ytick_fontsize)  # <-- Set ytick label size
    if not collapsed:
        ax_hist_rev = fig.add_subplot(gs[1])
        plot_trinuc_hist(
            hist_stats, labels=labels, panel_num=1, ax=ax_hist_rev, order=order, xtick_fontsize=xtick_fontsize
        )
        ax_hist_rev.set_ylabel("Density", fontsize=ylabel_fontsize)  # <-- Add ylabel
        ax_hist_rev.tick_params(axis="y", labelsize=ytick_fontsize)  # <-- Set ytick label size
    # Legend
    handles_frac, labels_frac = [], []
    for label in labels:
        if str(label).lower() in ["true", "1", 1, True]:
            label_name = "TP"
        elif str(label).lower() in ["false", "0", 0, False]:
            label_name = "FP"
        else:
            label_name = str(label)
        handles_frac.append(bars[label].patches[0])
        labels_frac.append(f"{label_name} ({total_snvs_per_label[label]})")
    fig.tight_layout(rect=[0, 0.1, 1, 1])
    # Calculate scaling factor based on height/width ratio
    width, height = figsize
    # Reference aspect ratio (looks good at 18x8)
    ref_width, ref_height = (18, 4) if collapsed else (18, 8)
    ref_aspect = ref_height / ref_width
    current_aspect = height / width
    # Scale bottom margin: higher aspect ratio (taller) needs less bottom margin
    aspect_scale = (ref_aspect / current_aspect) * bottom_scale

    # Use scaled bottom margin
    base_bottom = 0.10 * (1.5 * int(collapsed) + 1)
    plt.subplots_adjust(bottom=base_bottom * aspect_scale, top=1 - 0.10 * (2 * int(collapsed)))
    # Also scale bbox_to_anchor for the legend
    # For collapsed=True case, use less aggressive scaling to avoid too much white space
    if collapsed:
        legend_y = -0.12  # Fixed position for collapsed case to reduce white space
    else:
        legend_y = -0.08 * aspect_scale  # Original scaling for non-collapsed case

    fig.legend(
        handles_frac,
        labels_frac,
        title="Histogram: Labels (# SNVs total)",
        fontsize=14,
        title_fontsize=14,
        loc="lower center",
        bbox_to_anchor=(0.33, legend_y),
        ncol=2,
        frameon=False,
    )
    # Explanatory text
    # Explanatory text with bottom_scale
    if motif_orientation == "fwd_only":
        is_forward_text = "(forward reads only)"
    elif motif_orientation == "seq_dir":
        is_forward_text = "(sequencing direction)"
    elif motif_orientation == "ref_dir":
        is_forward_text = "(reference direction)"

    # Adjust text positioning for collapsed case
    if collapsed:
        # Fixed positioning for collapsed case to align with adjusted legend
        text_base_y = -0.075  # Closer to the plot
        text_height = 0.05  # Reduced spacing between lines
    else:
        # Original positioning for non-collapsed case
        text_base_y = -0.06
        text_height = 0.025

    fig.text(
        0.63, text_base_y + 2 * text_height, f"Trinuc-SNV {is_forward_text}:", ha="left", fontsize=14, color="black"
    )
    fig.text(0.66, text_base_y + text_height, "Green: Cycle skip", ha="left", fontsize=14, color="green")
    fig.text(0.66, text_base_y, "Red: No cycle skip", ha="left", fontsize=14, color="red")
    if suptitle:
        fig.suptitle(suptitle, fontsize=20)
    return fig


def calc_and_plot_trinuc_hist(  # noqa: PLR0913
    plot_df,
    trinuc_col,
    label_col="label",
    is_forward_col="is_forward",
    qual_col="SNVQ",
    order: str = "symmetric",
    hspace=0.1,
    suptitle=None,
    figsize=None,
    labels=None,
    hist_to_qual_height_ratio=2.0,
    bottom_scale=1.0,
    q1: float = 0.1,
    q2: float = 0.9,
    motif_orientation: str = "seq_dir",
    *,
    collapsed=False,
    include_quality: bool = False,
):
    """Calculate and plot trinuc histogram, optionally with quality panels.

    Parameters
    ----------
    plot_df : pd.DataFrame
        Input dataframe
    trinuc_col : str
        Column name for trinucleotide context
    label_col : str
        Column name for labels
    is_forward_col : str
        Column name for forward/reverse orientation
    qual_col : str
        Column name for quality values
    order : str
        Trinucleotide ordering
    suptitle : str, optional
        Figure title
    figsize : tuple, optional
        Figure size
    labels : list, optional
        Labels to include
    collapsed : bool
        Whether to collapse forward/reverse
    motif_orientation : str
        "seq_dir": as read in sequencing direction,
        "ref_dir": aligned to reference direction by reverse-complementing reverse-strand reads, or
        "fwd_only": using only forward-strand reads.
    include_quality : bool
        Whether to include quality panels

    Returns
    -------
    fig : matplotlib.Figure
        The figure
    stats_df : pd.DataFrame
        Statistics dataframe
    """
    stats_df = calc_trinuc_stats(
        plot_df,
        trinuc_col=trinuc_col,
        label_col=label_col,
        is_forward_col=is_forward_col,
        qual_col=qual_col,
        order=order,
        labels=labels,
        q1=q1,
        q2=q2,
        collapsed=collapsed,
        motif_orientation=motif_orientation,
        include_quality=include_quality,
    )

    if collapsed and (order != "reverse"):
        # raise warning that collapsing not by motif and its reverse complement
        warnings.warn(
            "Collapsing trinuc motifs with their symmetricly reversed partners W[X>Y]Z with W[Y>Z]X "
            "rather than with their reversed complements. Results may not be as expected. "
            "Consider using 'reverse' order.",
            stacklevel=2,
        )

    if include_quality:
        fig = plot_trinuc_hist_and_qual_panels(
            stats_df,
            q1=q1,
            q2=q2,
            order=order,
            suptitle=suptitle,
            figsize=figsize,
            collapsed=collapsed,
            motif_orientation=motif_orientation,
            hspace=hspace,
            hist_to_qual_height_ratio=hist_to_qual_height_ratio,
            bottom_scale=bottom_scale,
        )
    else:
        # Extract only histogram columns for backward compatibility
        hist_cols = [
            col
            for col in stats_df.columns
            if not col.endswith(("_median_qual", "_q1_qual", "_q2_qual")) and col != "is_cycle_skip"
        ]
        hist_stats_df = stats_df[hist_cols]

        fig = plot_trinuc_hist_panels(
            hist_stats_df,
            order=order,
            suptitle=suptitle,
            figsize=figsize,
            collapsed=collapsed,
            motif_orientation=motif_orientation,
            bottom_scale=bottom_scale,
        )

    return fig, stats_df
