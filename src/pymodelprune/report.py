"""Render model profiles as a terminal table."""

from __future__ import annotations

from collections.abc import Sequence

from rich.console import Console
from rich.table import Table

from pymodelprune.profile import ModelProfile

SPARSE_COLUMNS_FROM = 0.05
"""Minimum sparsity of any profile for the Sparse and gz MB columns to be shown."""


def _human_count(value: int) -> str:
    for threshold, suffix in ((1e9, "B"), (1e6, "M"), (1e3, "K")):
        if value >= threshold:
            return f"{value / threshold:.1f}{suffix}"
    return str(value)


def build_table(profiles: Sequence[ModelProfile], compare: bool = True) -> Table:
    """Build a table of profiles.

    With `compare`, the first profile is the baseline and the others get size and
    speed ratios against it. Turn it off when the rows are unrelated models.
    Optional columns (type, FLOPs, accuracy, fidelity) appear only when some profile
    has a value for them. Sparse and gz MB appear only when some profile is at least
    5% sparse: they tell the story of unstructured pruning and are noise otherwise.
    Size and Speed are ratios against the baseline: Size below 1 is smaller, Speed
    above 1 is faster.
    """
    compare = compare and len(profiles) > 1
    show_accuracy = any(profile.accuracy is not None for profile in profiles)
    show_fidelity = any(profile.fidelity is not None for profile in profiles)

    show_dtype = any(profile.dtype != "fp32" for profile in profiles)
    show_sparse = any(profile.sparsity >= SPARSE_COLUMNS_FROM for profile in profiles)
    show_compressed = show_sparse and any(p.compressed_mb is not None for p in profiles)
    show_flops = any(profile.flops is not None for profile in profiles)

    # Borderless and narrow on purpose. A dense comparison with every optional column
    # (Type, FLOPs, Acc, Fid, Size, Speed) fits an 80 column terminal; the two sparsity
    # columns on top of all of those need about 95.
    table = Table(box=None, pad_edge=False, padding=(0, 1), header_style="bold cyan")
    table.add_column("Model", style="bold", no_wrap=True)
    headers = ["Params"]
    headers += ["Sparse"] if show_sparse else []
    headers += ["Type"] if show_dtype else []
    headers += ["MB"]
    headers += ["gz MB"] if show_compressed else []
    headers += ["FLOPs"] if show_flops else []
    headers += ["p50 ms"]
    headers += ["Acc"] if show_accuracy else []
    headers += ["Fid"] if show_fidelity else []
    headers += ["Size", "Speed"] if compare else []
    for header in headers:
        table.add_column(header, justify="right", no_wrap=True)

    baseline = profiles[0]
    for profile in profiles:
        row = [profile.name, _human_count(profile.total_params)]
        if show_sparse:
            row.append(f"{profile.sparsity:.1%}")
        if show_dtype:
            row.append(profile.dtype)
        row.append(f"{profile.size_mb:.2f}")
        if show_compressed:
            row.append("-" if profile.compressed_mb is None else f"{profile.compressed_mb:.2f}")
        if show_flops:
            row.append("-" if profile.flops is None else _human_count(profile.flops))
        row.append(f"{profile.latency.median_ms:.3f}")
        if show_accuracy:
            row.append("-" if profile.accuracy is None else f"{profile.accuracy:.1%}")
        if show_fidelity:
            row.append("-" if profile.fidelity is None else f"{profile.fidelity.agreement:.1%}")
        if compare:
            row.append(f"{profile.size_mb / baseline.size_mb:.2f}x")
            row.append(f"{baseline.latency.median_ms / profile.latency.median_ms:.2f}x")
        table.add_row(*row)
    return table


def render_profiles(
    profiles: Sequence[ModelProfile],
    console: Console | None = None,
    compare: bool = True,
) -> None:
    if not profiles:
        raise ValueError("at least one profile is required")
    (console or Console()).print(build_table(profiles, compare=compare))
