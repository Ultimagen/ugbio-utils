"""
Parse the consensus tool's stdout log for performance metrics.

The ``read_fuser`` consensus step writes a progress/stats log (the
``consensus_stdout`` output). It has two informative parts:

* periodic progress lines::

      total=<n_reads>, <n> unhandled, <n> dup_set_members

  the **last** of which gives the cumulative totals for the whole run; and

* a final summary line::

      stats=Stats { alt_prob_too_high: 0, hmer_too_long: 5712685,
                    no_clear_hmer_choice: 14009063, not_same_contig: 4,
                    other_error: 1006093, dup_sets_merged: 216196151,
                    unchanged_segments: 297230665 }

  counting why segments were merged / left unchanged / skipped.

This module extracts both into a flat dict of integers plus a couple of derived
rates (unhandled fraction, merge fraction), so the report can show how the
consensus tool performed.
"""

from __future__ import annotations

import re

_TOTAL_RE = re.compile(r"total=(\d+),\s*(\d+)\s*unhandled,\s*(\d+)\s*dup_set_members")
_STATS_RE = re.compile(r"stats=Stats\s*\{([^}]*)\}")
_KV_RE = re.compile(r"(\w+):\s*(\d+)")


def parse_consensus_log(log_path: str) -> dict:
    """Extract consensus-tool performance metrics from a stdout log.

    Parameters
    ----------
    log_path : str
        Local path to the ``consensus_stdout`` log.

    Returns
    -------
    dict
        Flat metrics. Progress totals: ``total_reads``, ``unhandled``,
        ``dup_set_members``. Stats-line counters are passed through verbatim
        (e.g. ``hmer_too_long``, ``no_clear_hmer_choice``, ``dup_sets_merged``,
        ``unchanged_segments`` ...). Derived: ``PCT_unhandled`` (unhandled /
        total_reads * 100) and ``PCT_dup_set_members`` (dup_set_members /
        total_reads * 100). Missing pieces are simply absent from the dict.
    """
    metrics: dict = {}
    last_total: tuple[int, int, int] | None = None

    with open(log_path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            m = _TOTAL_RE.search(line)
            if m:
                last_total = (int(m.group(1)), int(m.group(2)), int(m.group(3)))
                continue
            s = _STATS_RE.search(line)
            if s:
                for key, value in _KV_RE.findall(s.group(1)):
                    metrics[key] = int(value)

    if last_total is not None:
        total_reads, unhandled, dup_set_members = last_total
        metrics["total_reads"] = total_reads
        metrics["unhandled"] = unhandled
        metrics["dup_set_members"] = dup_set_members
        if total_reads:
            metrics["PCT_unhandled"] = 100 * unhandled / total_reads
            metrics["PCT_dup_set_members"] = 100 * dup_set_members / total_reads

    return metrics
