"""Sample Size Decay detector — enrollment shortfalls and attrition.

Flags:
- Early termination (status=TERMINATED) -> severity 0.5
- Enrollment type still ANTICIPATED despite completion -> 0.3
- High attrition (>20% dropouts) from drop_withdrawals table
"""
import logging

import pandas as pd

from pipeline.detectors.base import BaseDetector, DetectorResult

logger = logging.getLogger(__name__)


class SampleSizeDecayDetector(BaseDetector):
    name = "sample_size_decay"
    description = "Enrollment shortfalls, early termination, or excessive attrition"
    aact_tables = ("drop_withdrawals",)

    def detect(
        self, master_df: pd.DataFrame, raw_tables: dict | None = None
    ) -> DetectorResult:
        nct_ids: list[str] = []
        flags: list[bool] = []
        severities: list[float] = []
        details: list[str] = []

        # Load attrition data
        dropouts = self._load_dropouts(master_df, raw_tables)

        for _, row in master_df.iterrows():
            nct = row["nct_id"]
            nct_ids.append(nct)
            score = 0.0
            issues: list[str] = []

            status = str(row.get("overall_status", "") or "").upper()
            enrollment = row.get("enrollment")
            enrollment_type = str(row.get("enrollment_type", "") or "").upper()

            # 1. Early termination
            if status == "TERMINATED":
                score += 0.5
                why = str(row.get("why_stopped", "") or "")
                issues.append(f"Terminated{': ' + why[:80] if why else ''}")

            # 2. Still anticipated enrollment on completed trial
            if enrollment_type == "ANTICIPATED" and status in (
                "COMPLETED", "TERMINATED", "ACTIVE, NOT RECRUITING",
                "ACTIVE_NOT_RECRUITING",
            ):
                score += 0.3
                issues.append("Enrollment still ANTICIPATED after completion")

            # 3. Attrition from drop_withdrawals
            if dropouts is not None and enrollment is not None and not pd.isna(enrollment):
                enrollment_val = float(enrollment) if float(enrollment) > 0 else 0
                if enrollment_val > 0:
                    total_drops = dropouts.get(nct, 0)
                    attrition_rate = total_drops / enrollment_val
                    if attrition_rate > 0.20:
                        score += min(attrition_rate, 1.0)
                        issues.append(
                            f"Attrition {attrition_rate:.0%} ({total_drops}/{int(enrollment_val)})"
                        )

            flagged = score > 0
            flags.append(flagged)
            severities.append(round(min(score, 1.0), 4))
            details.append("; ".join(issues) if flagged else "")

        return DetectorResult(
            nct_ids=nct_ids,
            flaw_detected=flags,
            severity=severities,
            detail=details,
        )

    def _load_dropouts(
        self, master_df: pd.DataFrame, raw_tables: dict | None
    ) -> dict[str, int] | None:
        """Load drop_withdrawals and compute total dropouts per trial."""
        try:
            if raw_tables is not None:
                dw = raw_tables.get("drop_withdrawals")
                if dw is None:
                    return None
            else:
                from pipeline.ingest import load_aact_table
                dw = load_aact_table("drop_withdrawals")
        except (KeyError, FileNotFoundError) as e:
            logger.warning("Could not load drop_withdrawals: %s", e)
            return None

        if dw is None or dw.empty:
            return None

        nct_set = set(master_df["nct_id"])
        dw = dw[dw["nct_id"].isin(nct_set)].copy()

        count_col = "count" if "count" in dw.columns else None
        if count_col is None:
            # Try alternate column names
            for col in dw.columns:
                if "count" in col.lower() or "drop" in col.lower():
                    count_col = col
                    break
        if count_col is None:
            return None

        dw[count_col] = pd.to_numeric(dw[count_col], errors="coerce").fillna(0)
        return dw.groupby("nct_id")[count_col].sum().to_dict()
