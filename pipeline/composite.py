"""Composite flaw scoring — aggregate detector results per trial."""
import numpy as np
import pandas as pd

DETECTOR_NAMES = [
    "ghost_protocols",
    "outcome_switching",
    "population_distortion",
    "sample_size_decay",
    "sponsor_concentration",
    "geographic_shifts",
    "results_delay",
    "endpoint_softening",
    "comparator_manipulation",
    "statistical_fragility",
]


def compute_composite_scores(df: pd.DataFrame) -> pd.DataFrame:
    """Add composite flaw columns to a DataFrame with detector results.

    Adds three columns:
      - flaw_count (int): number of detectors that flagged the trial
      - composite_severity (float): mean severity across detected flaws (0 if none)
      - flaw_categories (str): comma-separated names of detected flaws

    Parameters
    ----------
    df : pd.DataFrame
        Must contain {name}_detected (bool) and {name}_severity (float)
        columns for each detector in DETECTOR_NAMES.

    Returns
    -------
    pd.DataFrame
        Input df with the three new columns appended.
    """
    result = df.copy()

    # Identify which detector columns are present
    present = [
        name for name in DETECTOR_NAMES
        if f"{name}_detected" in result.columns
    ]

    detected_cols = [f"{name}_detected" for name in present]
    severity_cols = [f"{name}_severity" for name in present]

    # Build boolean matrix of detections
    det_matrix = result[detected_cols].fillna(False).astype(bool)

    # flaw_count = sum of detected columns
    result["flaw_count"] = det_matrix.sum(axis=1).astype(int)

    # composite_severity = mean severity where detected, 0 if no flaws
    sev_matrix = result[severity_cols].copy()
    # Mask out severities where not detected
    sev_masked = sev_matrix.values.copy().astype(float)
    det_flags = det_matrix.values
    sev_masked[~det_flags] = np.nan

    with np.errstate(all="ignore"):
        mean_sev = np.nanmean(sev_masked, axis=1)
    # nanmean returns nan when all are nan (no flaws) — replace with 0
    mean_sev = np.where(np.isnan(mean_sev), 0.0, mean_sev)
    result["composite_severity"] = mean_sev

    # flaw_categories = comma-separated detector names where detected
    def _categories(row):
        cats = []
        for name in present:
            if row.get(f"{name}_detected", False):
                cats.append(name)
        return ", ".join(cats)

    result["flaw_categories"] = result.apply(_categories, axis=1)

    return result
