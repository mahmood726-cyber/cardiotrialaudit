"""Build a one-row-per-trial master table from AACT + cardiology filter."""
import re
from typing import Optional

import numpy as np
import pandas as pd

from pipeline.ingest import load_aact_table
from pipeline.cardio_filter import filter_cardiology_trials


def build_master_table(
    nrows_studies: int | None = None,
    cv_studies: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build a single one-row-per-trial DataFrame for all detectors.

    Joins cardiology-filtered studies with designs, eligibilities, and
    sponsors tables from AACT, then derives convenience columns.

    Parameters
    ----------
    nrows_studies : int | None
        Row limit passed to filter_cardiology_trials (for testing).
    cv_studies : pd.DataFrame | None
        Pre-filtered CV studies DataFrame. If None, runs the filter.

    Returns
    -------
    pd.DataFrame
        One row per trial with all columns needed by detectors.
    """
    if cv_studies is None:
        cv_studies = filter_cardiology_trials(nrows_studies=nrows_studies)

    nct_ids = set(cv_studies["nct_id"].unique())
    mt = cv_studies.copy()

    # Join designs
    designs = load_aact_table(
        "designs",
        usecols=["nct_id", "allocation", "intervention_model", "primary_purpose",
                 "masking", "subject_masked", "caregiver_masked",
                 "investigator_masked", "outcomes_assessor_masked"],
    )
    designs = designs[designs["nct_id"].isin(nct_ids)].drop_duplicates(subset="nct_id")
    mt = mt.merge(designs, on="nct_id", how="left")

    # Join eligibilities
    elig = load_aact_table(
        "eligibilities",
        usecols=["nct_id", "gender", "minimum_age", "maximum_age",
                 "criteria", "healthy_volunteers", "adult", "child", "older_adult"],
    )
    elig = elig[elig["nct_id"].isin(nct_ids)].drop_duplicates(subset="nct_id")
    elig["min_age_years"] = elig["minimum_age"].apply(_parse_age)
    elig["max_age_years"] = elig["maximum_age"].apply(_parse_age)
    mt = mt.merge(elig, on="nct_id", how="left")

    # Join sponsors (lead only)
    sponsors = load_aact_table(
        "sponsors",
        usecols=["nct_id", "agency_class", "lead_or_collaborator", "name"],
    )
    lead_sponsors = sponsors[
        (sponsors["nct_id"].isin(nct_ids))
        & (sponsors["lead_or_collaborator"] == "lead")
    ].drop_duplicates(subset="nct_id")
    lead_sponsors = lead_sponsors.rename(columns={
        "name": "lead_sponsor_name",
        "agency_class": "lead_sponsor_class",
    })[["nct_id", "lead_sponsor_name", "lead_sponsor_class"]]
    mt = mt.merge(lead_sponsors, on="nct_id", how="left")

    # Derived columns
    mt["has_results"] = mt["results_first_posted_date"].notna()
    mt["is_interventional"] = (
        mt.get("study_type", pd.Series(["INTERVENTIONAL"] * len(mt)))
        .str.upper() == "INTERVENTIONAL"
    )
    mt["is_randomized"] = (
        mt["allocation"].str.upper().eq("RANDOMIZED")
        if "allocation" in mt.columns
        else False
    )

    mt["primary_completion_date"] = pd.to_datetime(
        mt["primary_completion_date"], errors="coerce"
    )
    mt["results_first_posted_date"] = pd.to_datetime(
        mt["results_first_posted_date"], errors="coerce"
    )
    mt["results_delay_days"] = (
        mt["results_first_posted_date"] - mt["primary_completion_date"]
    ).dt.days

    mt = mt.drop_duplicates(subset="nct_id")
    return mt.reset_index(drop=True)


def _parse_age(age_str) -> Optional[float]:
    """Parse AACT age strings like '65 Years' into numeric years.

    Handles Years, Months, Days, Hours. Returns None for missing/unparseable.
    """
    if pd.isna(age_str) or not age_str:
        return None
    age_str = str(age_str).strip().lower()
    if age_str in ("n/a", ""):
        return None
    m = re.match(r"(\d+)", age_str)
    if not m:
        return None
    val = float(m.group(1))
    if "month" in age_str:
        val = val / 12.0
    elif "day" in age_str:
        val = val / 365.25
    elif "hour" in age_str:
        val = val / (365.25 * 24)
    return val
