"""Outcome Switching detector — registered vs. reported primary outcomes mismatch.

Loads design_outcomes (registered) and outcomes (reported) tables from AACT.
Uses fuzzy matching (rapidfuzz) to compare registered vs reported primaries.
"""
import logging

import pandas as pd

from pipeline.detectors.base import BaseDetector, DetectorResult

logger = logging.getLogger(__name__)


def _fuzzy_best_match(query: str, candidates: list[str]) -> float:
    """Return the best token_sort_ratio for query against candidates (0-100)."""
    if not query or not candidates:
        return 0.0
    try:
        from rapidfuzz import fuzz
    except ImportError:
        # Fallback: exact substring match
        query_lower = query.lower()
        for c in candidates:
            if query_lower in c.lower() or c.lower() in query_lower:
                return 80.0
        return 0.0
    best = 0.0
    for c in candidates:
        score = fuzz.token_sort_ratio(query, c)
        if score > best:
            best = score
    return best


class OutcomeSwitchingDetector(BaseDetector):
    name = "outcome_switching"
    description = "Registered vs reported primary outcomes do not match"
    aact_tables = ("design_outcomes", "outcomes")

    def detect(
        self, master_df: pd.DataFrame, raw_tables: dict | None = None
    ) -> DetectorResult:
        nct_ids: list[str] = []
        flags: list[bool] = []
        severities: list[float] = []
        details: list[str] = []

        # Load auxiliary tables
        design_outcomes = self._load_table("design_outcomes", raw_tables)
        outcomes = self._load_table("outcomes", raw_tables)

        if design_outcomes is None or outcomes is None:
            # Cannot assess — return all-clean
            return self._empty_result(master_df)

        # Filter to primary outcomes
        registered = design_outcomes[
            design_outcomes["outcome_type"].str.upper().eq("PRIMARY")
        ].copy() if "outcome_type" in design_outcomes.columns else pd.DataFrame()

        reported = outcomes[
            outcomes["outcome_type"].str.upper().eq("PRIMARY")
        ].copy() if "outcome_type" in outcomes.columns else pd.DataFrame()

        # Build lookup dicts: nct_id -> list of measure/title strings
        reg_by_nct: dict[str, list[str]] = {}
        if not registered.empty:
            measure_col = "measure" if "measure" in registered.columns else "title"
            for nct, grp in registered.groupby("nct_id"):
                reg_by_nct[nct] = [
                    str(m) for m in grp[measure_col].dropna().tolist()
                ]

        rep_by_nct: dict[str, list[str]] = {}
        if not reported.empty:
            title_col = "title" if "title" in reported.columns else "measure"
            for nct, grp in reported.groupby("nct_id"):
                rep_by_nct[nct] = [
                    str(t) for t in grp[title_col].dropna().tolist()
                ]

        for _, row in master_df.iterrows():
            nct = row["nct_id"]
            nct_ids.append(nct)

            if not row.get("has_results", False):
                flags.append(False)
                severities.append(0.0)
                details.append("")
                continue

            reg_list = reg_by_nct.get(nct, [])
            rep_list = rep_by_nct.get(nct, [])

            if not reg_list and not rep_list:
                flags.append(False)
                severities.append(0.0)
                details.append("")
                continue

            issues: list[str] = []
            max_sev = 0.0

            # Check for unregistered new primaries in reported
            for rep_title in rep_list:
                if not reg_list:
                    issues.append(f"Unregistered primary: {rep_title[:60]}")
                    max_sev = max(max_sev, 1.0)
                else:
                    best = _fuzzy_best_match(rep_title, reg_list)
                    if best < 60:
                        issues.append(
                            f"Possible switch (score={best:.0f}): {rep_title[:60]}"
                        )
                        max_sev = max(max_sev, 1.0 - best / 100.0)

            # Check for missing registered primaries
            for reg_measure in reg_list:
                if not rep_list:
                    issues.append(f"Missing registered primary: {reg_measure[:60]}")
                    max_sev = max(max_sev, 0.7)
                else:
                    best = _fuzzy_best_match(reg_measure, rep_list)
                    if best < 60:
                        issues.append(
                            f"Registered primary not reported (score={best:.0f}): {reg_measure[:60]}"
                        )
                        max_sev = max(max_sev, 0.7)

            flagged = len(issues) > 0
            flags.append(flagged)
            severities.append(round(min(max_sev, 1.0), 4))
            details.append("; ".join(issues) if issues else "")

        return DetectorResult(
            nct_ids=nct_ids,
            flaw_detected=flags,
            severity=severities,
            detail=details,
        )

    # _load_table inherited from BaseDetector

    def _empty_result(self, master_df: pd.DataFrame) -> DetectorResult:
        n = len(master_df)
        return DetectorResult(
            nct_ids=master_df["nct_id"].tolist(),
            flaw_detected=[False] * n,
            severity=[0.0] * n,
            detail=[""] * n,
        )
