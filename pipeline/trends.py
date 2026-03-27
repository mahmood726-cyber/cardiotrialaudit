"""Temporal trend analysis — yearly and binned aggregations."""
import numpy as np
import pandas as pd

from pipeline.composite import DETECTOR_NAMES


def compute_yearly_trends(df: pd.DataFrame) -> pd.DataFrame:
    """Compute per-year summary statistics from trial-level results.

    Parameters
    ----------
    df : pd.DataFrame
        Trial-level results with start_year, flaw_count, composite_severity,
        and {name}_detected / {name}_severity columns.

    Returns
    -------
    pd.DataFrame
        One row per year with columns:
        - year, n_trials
        - {name}_rate (fraction detected), {name}_mean_severity
        - composite_mean, mean_flaw_count
    """
    if "start_year" not in df.columns:
        raise ValueError("DataFrame must contain 'start_year' column")

    work = df.copy()
    work["start_year"] = pd.to_numeric(work["start_year"], errors="coerce")
    work = work.dropna(subset=["start_year"])
    work["start_year"] = work["start_year"].astype(int)

    present = [
        name for name in DETECTOR_NAMES
        if f"{name}_detected" in work.columns
    ]

    agg_dict = {}
    agg_dict["n_trials"] = ("start_year", "count")

    for name in present:
        col_det = f"{name}_detected"
        col_sev = f"{name}_severity"
        agg_dict[f"{name}_rate"] = (col_det, "mean")
        agg_dict[f"{name}_mean_severity"] = (col_sev, "mean")

    if "composite_severity" in work.columns:
        agg_dict["composite_mean"] = ("composite_severity", "mean")
    if "flaw_count" in work.columns:
        agg_dict["mean_flaw_count"] = ("flaw_count", "mean")

    grouped = work.groupby("start_year")
    trends = grouped.agg(**agg_dict).reset_index()
    trends = trends.rename(columns={"start_year": "year"})
    trends = trends.sort_values("year").reset_index(drop=True)

    return trends


def compute_binned_trends(
    df: pd.DataFrame, bin_size: int = 3
) -> pd.DataFrame:
    """Compute trends in N-year bins (e.g., '2005-2007').

    Parameters
    ----------
    df : pd.DataFrame
        Trial-level results with start_year.
    bin_size : int
        Number of years per bin (default 3).

    Returns
    -------
    pd.DataFrame
        Same columns as compute_yearly_trends but with year_bin instead of year.
    """
    if "start_year" not in df.columns:
        raise ValueError("DataFrame must contain 'start_year' column")

    work = df.copy()
    work["start_year"] = pd.to_numeric(work["start_year"], errors="coerce")
    work = work.dropna(subset=["start_year"])
    work["start_year"] = work["start_year"].astype(int)

    if work.empty:
        return pd.DataFrame()

    min_year = int(work["start_year"].min())
    max_year = int(work["start_year"].max())

    # Create bin labels
    def _year_to_bin(y):
        bin_start = min_year + ((y - min_year) // bin_size) * bin_size
        bin_end = min(bin_start + bin_size - 1, max_year)
        return f"{bin_start}-{bin_end}"

    work["year_bin"] = work["start_year"].apply(_year_to_bin)

    # Create sort key for bins
    work["_bin_start"] = work["start_year"].apply(
        lambda y: min_year + ((y - min_year) // bin_size) * bin_size
    )

    present = [
        name for name in DETECTOR_NAMES
        if f"{name}_detected" in work.columns
    ]

    agg_dict = {}
    agg_dict["n_trials"] = ("year_bin", "count")
    agg_dict["_bin_start"] = ("_bin_start", "first")

    for name in present:
        col_det = f"{name}_detected"
        col_sev = f"{name}_severity"
        agg_dict[f"{name}_rate"] = (col_det, "mean")
        agg_dict[f"{name}_mean_severity"] = (col_sev, "mean")

    if "composite_severity" in work.columns:
        agg_dict["composite_mean"] = ("composite_severity", "mean")
    if "flaw_count" in work.columns:
        agg_dict["mean_flaw_count"] = ("flaw_count", "mean")

    grouped = work.groupby("year_bin")
    binned = grouped.agg(**agg_dict).reset_index()
    binned = binned.sort_values("_bin_start").reset_index(drop=True)
    binned = binned.drop(columns=["_bin_start"])

    return binned
