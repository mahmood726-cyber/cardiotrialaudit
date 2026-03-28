"""Temporal trend analysis — yearly and binned aggregations.

P1-1: Added maturity_filter parameter for right-truncation correction
in time-dependent detectors (ghost_protocols, results_delay).
"""
import numpy as np
import pandas as pd

from pipeline.composite import DETECTOR_NAMES
from pipeline.detectors.base import AACT_SNAPSHOT_DATE

# Detectors that require elapsed time and are subject to right-truncation bias
_TIME_DEPENDENT_DETECTORS = {"ghost_protocols", "results_delay"}
# Minimum months since primary_completion_date for a trial to be eligible
# for time-dependent detection (13 months = FDAAA 801 default grace)
_MATURITY_CUTOFF_DATE = pd.Timestamp("2025-01-19")  # 13 months before snapshot


def compute_yearly_trends(
    df: pd.DataFrame,
    maturity_filter: bool = False,
) -> pd.DataFrame:
    """Compute per-year summary statistics from trial-level results.

    Parameters
    ----------
    df : pd.DataFrame
        Trial-level results with start_year, flaw_count, composite_severity,
        and {name}_detected / {name}_severity columns.
    maturity_filter : bool
        If True, for time-dependent detectors (ghost_protocols, results_delay),
        only include trials with primary_completion_date before the maturity
        cutoff in the denominator. Adds {name}_eligible_n columns.

    Returns
    -------
    pd.DataFrame
        One row per year with columns:
        - year, n_trials
        - {name}_rate, {name}_severity_when_detected (mean severity among detected)
        - composite_mean, mean_flaw_count
        - {name}_eligible_n (if maturity_filter=True, for time-dependent detectors)
    """
    if "start_year" not in df.columns:
        raise ValueError("DataFrame must contain 'start_year' column")

    work = df.copy()
    work["start_year"] = pd.to_numeric(work["start_year"], errors="coerce")
    work = work.dropna(subset=["start_year"])
    work["start_year"] = work["start_year"].astype(int)

    # P1-1: Compute maturity eligibility mask
    if maturity_filter and "primary_completion_date" in work.columns:
        pcd = pd.to_datetime(work["primary_completion_date"], errors="coerce")
        work["_mature"] = pcd.notna() & (pcd < _MATURITY_CUTOFF_DATE)
    else:
        work["_mature"] = True

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
        agg_dict[f"{name}_severity_when_detected"] = (col_sev, "mean")

    if "composite_severity" in work.columns:
        agg_dict["composite_mean"] = ("composite_severity", "mean")
    if "flaw_count" in work.columns:
        agg_dict["mean_flaw_count"] = ("flaw_count", "mean")

    grouped = work.groupby("start_year")
    trends = grouped.agg(**agg_dict).reset_index()
    trends = trends.rename(columns={"start_year": "year"})

    # P1-1: Add eligible_n for time-dependent detectors
    if maturity_filter:
        for name in present:
            if name in _TIME_DEPENDENT_DETECTORS:
                col_det = f"{name}_detected"
                eligible = work[work["_mature"]].groupby("start_year").agg(
                    **{
                        f"{name}_eligible_n": ("start_year", "count"),
                        f"_{name}_eligible_rate": (col_det, "mean"),
                    }
                ).reset_index().rename(columns={"start_year": "year"})
                trends = trends.merge(eligible[["year", f"{name}_eligible_n"]], on="year", how="left")
                trends[f"{name}_eligible_n"] = trends[f"{name}_eligible_n"].fillna(0).astype(int)

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
        label = str(int(bin_start)) if bin_start == bin_end else f"{int(bin_start)}-{int(bin_end)}"
        return label

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
        agg_dict[f"{name}_severity_when_detected"] = (col_sev, "mean")

    if "composite_severity" in work.columns:
        agg_dict["composite_mean"] = ("composite_severity", "mean")
    if "flaw_count" in work.columns:
        agg_dict["mean_flaw_count"] = ("flaw_count", "mean")

    grouped = work.groupby("year_bin")
    binned = grouped.agg(**agg_dict).reset_index()
    binned = binned.sort_values("_bin_start").reset_index(drop=True)
    binned = binned.drop(columns=["_bin_start"])

    return binned
