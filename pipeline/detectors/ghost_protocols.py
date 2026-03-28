"""Ghost Protocols detector — trials that vanish from the evidence record.

Type A: NOT_YET_RECRUITING / WITHDRAWN, registered >= 3 years ago.
       P1-3: Uses study_first_posted_date instead of start_date when available.
Type B: UNKNOWN / SUSPENDED, no update >= 2 years.
Type C: COMPLETED / TERMINATED, no results >= 13 months after primary_completion_date.
       P0-3: Extended grace period (36 months) when result_agreements indicate extensions.

P1-2: Uses AACT_SNAPSHOT_DATE from base module.
P1-12: Vectorized implementation (no iterrows).
"""
import logging

import numpy as np
import pandas as pd

from pipeline.detectors.base import AACT_SNAPSHOT_DATE, BaseDetector, DetectorResult

logger = logging.getLogger(__name__)

# Default grace period for results posting (13 months per FDAAA 801)
_DEFAULT_GRACE_MONTHS = 13.0
# Extended grace when result_agreements indicate delayed certification
_EXTENDED_GRACE_MONTHS = 36.0


class GhostProtocolsDetector(BaseDetector):
    """Trials that vanish from the evidence record without results or updates.

    Note: Type C overlaps with results_delay detector for COMPLETED trials with
    no results. Both measure different constructs (evidence disappearance vs
    regulatory non-compliance). Documented in manuscript methods.
    """

    name = "ghost_protocols"
    description = "Trials that vanish from the evidence record without results or updates"
    aact_tables = ("result_agreements",)

    def detect(
        self, master_df: pd.DataFrame, raw_tables: dict | None = None
    ) -> DetectorResult:
        n = len(master_df)
        nct_ids = master_df["nct_id"].tolist()
        flags = [False] * n
        severities = [0.0] * n
        details = [""] * n

        # P0-3: Load result_agreements for grace period extensions
        extended_ncts = self._load_extended_grace_ncts(master_df, raw_tables)

        # Vectorized column extraction
        status = master_df["overall_status"].fillna("").str.upper() if "overall_status" in master_df.columns else pd.Series([""] * n)
        start_date = pd.to_datetime(master_df["start_date"], errors="coerce") if "start_date" in master_df.columns else pd.Series([pd.NaT] * n)
        # P1-3: Prefer study_first_posted_date for Type A
        if "study_first_posted_date" in master_df.columns:
            study_first_posted = pd.to_datetime(master_df["study_first_posted_date"], errors="coerce")
        else:
            study_first_posted = pd.Series([pd.NaT] * n)
        last_update = pd.to_datetime(master_df["last_update_posted_date"], errors="coerce") if "last_update_posted_date" in master_df.columns else pd.Series([pd.NaT] * n)
        pcd = pd.to_datetime(master_df["primary_completion_date"], errors="coerce") if "primary_completion_date" in master_df.columns else pd.Series([pd.NaT] * n)
        has_results = master_df["has_results"].fillna(False).astype(bool) if "has_results" in master_df.columns else pd.Series([False] * n)

        # Vectorized date computations
        # P1-3: For Type A, use study_first_posted_date, fall back to start_date
        type_a_date = study_first_posted.where(study_first_posted.notna(), start_date)
        years_since_reg = (AACT_SNAPSHOT_DATE - type_a_date).dt.days / 365.25
        years_silent = (AACT_SNAPSHOT_DATE - last_update).dt.days / 365.25
        months_overdue = (AACT_SNAPSHOT_DATE - pcd).dt.days / 30.44

        # Vectorized status masks
        type_a_statuses = status.isin(
            {"NOT YET RECRUITING", "NOT_YET_RECRUITING", "WITHDRAWN"}
        ).values
        type_b_statuses = status.isin(
            {"UNKNOWN STATUS", "UNKNOWN", "SUSPENDED"}
        ).values
        type_c_statuses = status.isin(
            {"COMPLETED", "TERMINATED"}
        ).values

        for i in range(n):
            nct = nct_ids[i]
            flagged = False
            sev = 0.0
            det = ""

            # Type A: zombie registrations
            if type_a_statuses[i] and pd.notna(type_a_date.iloc[i]):
                ys = years_since_reg.iloc[i]
                if ys >= 3.0:
                    flagged = True
                    sev = min(ys / 5.0, 1.0)
                    det = f"Type A: {status.iloc[i]}, registered {ys:.1f}y ago"

            # Type B: silent trials
            if not flagged and type_b_statuses[i] and pd.notna(last_update.iloc[i]):
                ys = years_silent.iloc[i]
                if ys >= 2.0:
                    flagged = True
                    sev = min(ys / 5.0, 1.0)
                    det = f"Type B: {status.iloc[i]}, silent {ys:.1f}y"

            # Type C: results-free completed trials
            if not flagged and type_c_statuses[i] and not has_results.iloc[i] and pd.notna(pcd.iloc[i]):
                mo = months_overdue.iloc[i]
                # P0-3: Check for extended grace period
                has_extension = nct in extended_ncts
                grace = _EXTENDED_GRACE_MONTHS if has_extension else _DEFAULT_GRACE_MONTHS
                if mo >= grace:
                    years_od = mo / 12.0
                    flagged = True
                    sev = min(years_od / 5.0, 1.0)
                    det = f"Type C: {status.iloc[i]}, no results {mo:.0f}m after completion"
                    if has_extension:
                        det += " (has results agreement — extended grace)"

            flags[i] = flagged
            severities[i] = round(sev, 4)
            details[i] = det

        return DetectorResult(
            nct_ids=nct_ids,
            flaw_detected=flags,
            severity=severities,
            detail=details,
        )

    def _load_extended_grace_ncts(
        self, master_df: pd.DataFrame, raw_tables: dict | None
    ) -> set[str]:
        """Load NCT IDs that have result_agreements with delay/extension language.

        P0-3: These trials get a 36-month grace period instead of 13 months.
        """
        try:
            if raw_tables is not None:
                ra = raw_tables.get("result_agreements")
                if ra is None:
                    return set()
            else:
                from pipeline.ingest import load_aact_table
                ra = load_aact_table(
                    "result_agreements",
                    usecols=["nct_id", "restrictive_agreement"],
                )
        except (KeyError, FileNotFoundError) as e:
            logger.debug("result_agreements table not available: %s", e)
            return set()

        if ra is None or ra.empty:
            return set()

        nct_set = set(master_df["nct_id"])
        ra = ra[ra["nct_id"].isin(nct_set)]

        if "restrictive_agreement" not in ra.columns:
            # If restrictive_agreement column missing, any entry implies agreement
            return set(ra["nct_id"].unique())

        # Any non-empty restrictive_agreement indicates a results agreement
        # that may include delay/extension provisions
        ra["has_agreement"] = ra["restrictive_agreement"].fillna("").astype(str).str.strip().ne("")
        return set(ra[ra["has_agreement"]]["nct_id"].unique())
