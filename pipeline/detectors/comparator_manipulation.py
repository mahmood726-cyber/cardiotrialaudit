"""Comparator Manipulation detector — placebo-controlled when SOC exists.

Loads interventions and design_groups tables from AACT.
Detects placebo/sham use in conditions where standard-of-care exists.
"""
import logging
import re

import pandas as pd

from pipeline.detectors.base import BaseDetector, DetectorResult

logger = logging.getLogger(__name__)

# CV conditions with established standard of care
# P1-8: Added "prevention" and "vascular" subdomains
# - Prevention: SOC = statin + antiplatelet for secondary prevention
# - Vascular: SOC = antiplatelet + statin + risk factor management
_SOC_SUBDOMAINS = frozenset({
    "HF", "CAD", "arrhythmia", "hypertension", "VTE", "structural",
    "prevention", "vascular",
})

_PLACEBO_PATTERN = re.compile(
    r"(?i)\b(placebo|sham|inactive|sugar\s+pill|dummy)\b"
)

_SOC_COMPARATOR_PATTERN = re.compile(
    r"(?i)\b(standard\s+of\s+care|usual\s+care|best\s+medical|guideline.directed"
    r"|optimal\s+medical|active\s+comparator|active\s+control)\b"
)


class ComparatorManipulationDetector(BaseDetector):
    name = "comparator_manipulation"
    description = "Placebo-controlled design when standard-of-care exists for the condition"
    aact_tables = ("interventions", "design_groups")

    def detect(
        self, master_df: pd.DataFrame, raw_tables: dict | None = None
    ) -> DetectorResult:
        nct_ids: list[str] = []
        flags: list[bool] = []
        severities: list[float] = []
        details: list[str] = []

        # Load intervention and arm data
        interv_info = self._load_intervention_info(master_df, raw_tables)
        group_info = self._load_group_info(master_df, raw_tables)

        for _, row in master_df.iterrows():
            nct = row["nct_id"]
            nct_ids.append(nct)

            # Check if trial has a subdomain with established SOC
            subdomains = row.get("cv_subdomains", [])
            if isinstance(subdomains, str):
                subdomains = [s.strip().strip("'\"") for s in subdomains.strip("[]").split(",")]

            has_soc_condition = any(sd in _SOC_SUBDOMAINS for sd in subdomains)

            if not has_soc_condition:
                flags.append(False)
                severities.append(0.0)
                details.append("")
                continue

            # Check for placebo in interventions
            interventions = interv_info.get(nct, [])
            groups = group_info.get(nct, [])
            combined_text = " ".join(interventions + groups)

            has_placebo = bool(_PLACEBO_PATTERN.search(combined_text))
            has_soc_comparator = bool(_SOC_COMPARATOR_PATTERN.search(combined_text))

            if has_placebo and not has_soc_comparator:
                # Placebo-only comparator in SOC condition
                flags.append(True)
                sev = 0.5
                severities.append(sev)
                matching = [sd for sd in subdomains if sd in _SOC_SUBDOMAINS]
                details.append(
                    f"Placebo-controlled with SOC available ({', '.join(matching)})"
                )
            elif has_placebo and has_soc_comparator:
                # Placebo + SOC (add-on design) — lower concern
                flags.append(True)
                sev = 0.2
                severities.append(sev)
                details.append("Placebo + SOC (add-on design)")
            else:
                flags.append(False)
                severities.append(0.0)
                details.append("")

        return DetectorResult(
            nct_ids=nct_ids,
            flaw_detected=flags,
            severity=severities,
            detail=details,
        )

    def _load_intervention_info(
        self, master_df: pd.DataFrame, raw_tables: dict | None
    ) -> dict[str, list[str]]:
        """Load intervention names per trial."""
        try:
            if raw_tables is not None:
                interv = raw_tables.get("interventions")
                if interv is None:
                    return {}
            else:
                from pipeline.ingest import load_aact_table
                interv = load_aact_table(
                    "interventions", usecols=["nct_id", "intervention_type", "name"]
                )
        except (KeyError, FileNotFoundError) as e:
            logger.warning("Could not load interventions: %s", e)
            return {}

        nct_set = set(master_df["nct_id"])
        interv = interv[interv["nct_id"].isin(nct_set)]
        result: dict[str, list[str]] = {}
        for nct, grp in interv.groupby("nct_id"):
            texts = []
            for _, r in grp.iterrows():
                texts.append(str(r.get("name", "") or ""))
                texts.append(str(r.get("intervention_type", "") or ""))
            result[nct] = texts
        return result

    def _load_group_info(
        self, master_df: pd.DataFrame, raw_tables: dict | None
    ) -> dict[str, list[str]]:
        """Load design group descriptions per trial."""
        try:
            if raw_tables is not None:
                groups = raw_tables.get("design_groups")
                if groups is None:
                    return {}
            else:
                from pipeline.ingest import load_aact_table
                groups = load_aact_table("design_groups")
        except (KeyError, FileNotFoundError) as e:
            logger.warning("Could not load design_groups: %s", e)
            return {}

        nct_set = set(master_df["nct_id"])
        groups = groups[groups["nct_id"].isin(nct_set)]
        result: dict[str, list[str]] = {}
        for nct, grp in groups.groupby("nct_id"):
            texts = []
            for _, r in grp.iterrows():
                texts.append(str(r.get("group_type", "") or ""))
                texts.append(str(r.get("title", "") or ""))
                texts.append(str(r.get("description", "") or ""))
            result[nct] = texts
        return result
