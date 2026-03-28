"""Results Delay detector — time from primary completion to results posting.

Severity tiers:
- <= 12 months: compliant (no flag)
- 12-24 months: severity 0.3
- 24-36 months: severity 0.6
- > 36 months: scaled up to 1.0
- No results on completed trial: severity based on months since completion

P0-2: FDAAA temporal stratification applied:
- Pre-FDAAA (PCD < 2008-01-01): severity *= 0.1
- FDAAA era (2008-2016): severity *= 0.5
- Final Rule era (2017+): full severity
- fdaaa801_violation = True: force severity to 1.0

P1-2: Uses AACT_SNAPSHOT_DATE from base module.
P1-12: Vectorized implementation (no iterrows).
"""
import numpy as np
import pandas as pd

from pipeline.detectors.base import AACT_SNAPSHOT_DATE, BaseDetector, DetectorResult

# FDAAA 801 temporal boundaries
_FDAAA_START = pd.Timestamp("2008-01-01")  # FDAAA 801 became law Sept 2007
_FINAL_RULE_START = pd.Timestamp("2017-01-01")  # Final Rule effective Jan 2017


class ResultsDelayDetector(BaseDetector):
    name = "results_delay"
    description = "Excessive delay between trial completion and results posting"
    aact_tables: tuple[str, ...] = ()

    def detect(
        self, master_df: pd.DataFrame, raw_tables: dict | None = None
    ) -> DetectorResult:
        n = len(master_df)
        nct_ids = master_df["nct_id"].tolist()
        flags = [False] * n
        severities = [0.0] * n
        details = [""] * n

        # Vectorized column extraction — use column-in-columns check to avoid
        # DataFrame.get() returning None instead of Series
        pcd = pd.to_datetime(master_df["primary_completion_date"], errors="coerce") if "primary_completion_date" in master_df.columns else pd.Series([pd.NaT] * n)
        results_date = pd.to_datetime(master_df["results_first_posted_date"], errors="coerce") if "results_first_posted_date" in master_df.columns else pd.Series([pd.NaT] * n)
        has_results = master_df["has_results"].fillna(False).astype(bool) if "has_results" in master_df.columns else pd.Series([False] * n)
        status = master_df["overall_status"].fillna("").str.upper() if "overall_status" in master_df.columns else pd.Series([""] * n)
        fdaaa_viol = master_df["fdaaa801_violation"].fillna(False).astype(bool) if "fdaaa801_violation" in master_df.columns else pd.Series([False] * n)

        # Masks
        pcd_valid = pcd.notna().values
        results_valid = results_date.notna().values
        has_res = has_results.values
        completed_statuses = status.isin(
            {"COMPLETED", "TERMINATED", "ACTIVE, NOT RECRUITING", "ACTIVE_NOT_RECRUITING"}
        ).values

        # Compute delay months for those with results
        delay_days = (results_date - pcd).dt.days
        delay_months = delay_days / 30.44

        # Compute months since completion for those without results
        months_since = (AACT_SNAPSHOT_DATE - pcd).dt.days / 30.44

        for i in range(n):
            if not pcd_valid[i]:
                continue

            sev = 0.0
            det = ""
            flagged = False

            if has_res[i] and results_valid[i]:
                dm = delay_months.iloc[i]
                if dm <= 12.0:
                    continue
                elif dm <= 24.0:
                    sev = 0.3
                    det = f"Results delay: {dm:.0f} months"
                    flagged = True
                elif dm <= 36.0:
                    sev = 0.6
                    det = f"Results delay: {dm:.0f} months"
                    flagged = True
                else:
                    sev = min(0.6 + (dm - 36.0) / 48.0, 1.0)
                    det = f"Results delay: {dm:.0f} months"
                    flagged = True
            elif not has_res[i] and completed_statuses[i]:
                ms = months_since.iloc[i]
                if ms > 12.0:
                    sev = min(ms / 60.0, 1.0)
                    det = f"No results {ms:.0f}m after completion"
                    flagged = True

            if flagged:
                # P0-2: FDAAA temporal stratification
                pcd_val = pcd.iloc[i]
                if fdaaa_viol[i]:
                    # Explicit FDAAA violation recorded — force max severity
                    sev = 1.0
                elif pcd_val < _FDAAA_START:
                    sev *= 0.1  # Pre-mandate era
                elif pcd_val < _FINAL_RULE_START:
                    sev *= 0.5  # FDAAA era but rarely enforced

                flags[i] = True
                severities[i] = round(sev, 4)
                details[i] = det

        return DetectorResult(
            nct_ids=nct_ids,
            flaw_detected=flags,
            severity=severities,
            detail=details,
        )
