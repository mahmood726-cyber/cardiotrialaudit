"""Endpoint Softening detector — primary endpoints that are all-surrogate or mixed.

Loads outcomes table, classifies each primary outcome as:
- hard: mortality, MI, stroke, hospitalization, MACE
- surrogate: biomarkers, imaging, BP, lab values, 6MWT, NYHA
- pro: QoL, patient-reported outcomes, symptom scores
- other: unclassifiable

P0-4: 6MWT and NYHA moved from PRO to surrogate (they are functional assessments,
not patient-reported outcomes).

P1-4: Differentiated severity:
- surrogate-only = 1.0
- PRO-only = 0.7
- mixed hard+surrogate = 0.5
- mixed hard+PRO = 0.3
"""
import logging
import re

import pandas as pd

from pipeline.detectors.base import BaseDetector, DetectorResult

logger = logging.getLogger(__name__)

# Endpoint classification patterns
_HARD_PATTERNS = re.compile(
    r"(?i)\b("
    r"mortality|death|all.cause.death|cardiovascular.death|cv.death"
    r"|myocardial.infarction|heart.attack|acute.mi"
    r"|stroke|cerebrovascular.accident"
    r"|hospitalization|hospital.admission|rehospitalization"
    r"|mace|major.adverse.cardiovascular"
    r"|cardiac.arrest|sudden.cardiac.death"
    r"|stent.thrombosis|revascularization"
    r"|urgent.?(?:heart failure|hf).?visit|worsening.?heart.?failure"
    r"|cardiovascular.?hospitalization"
    r")\b"
)

# P0-4: 6MWT and NYHA moved here from PRO (functional assessments, not patient-reported)
_SURROGATE_PATTERNS = re.compile(
    r"(?i)\b("
    r"blood.pressure|systolic|diastolic|bp.reduction"
    r"|ejection.fraction|lvef|left.ventricular"
    r"|nt.pro.?bnp|bnp|troponin|biomarker"
    r"|ldl|hdl|cholesterol|triglyceride|lipid"
    r"|hba1c|glycated|fasting.glucose"
    r"|egfr|creatinine|albuminuria|uacr"
    r"|lumen|stenosis|plaque|intima.media|imt"
    r"|coronary.flow|fractional.flow|ffr"
    r"|peak.vo2|exercise.capacity"
    r"|arterial.stiffness|pulse.wave"
    r"|cardiac.output|cardiac.index"
    r"|infarct.size"
    r"|6.?minute.walk|6mwd|6mwt"
    r"|nyha|functional.class"
    r"|uacr|albumin.to.creatinine|galectin|sst2"
    r"|cardiac.?(?:mri|magnetic)|t1.?mapping|ecv"
    r")\b"
)

_PRO_PATTERNS = re.compile(
    r"(?i)\b("
    r"quality.of.life|qol|eq.?5d|sf.?36|sf.?12"
    r"|kccq|kansas.city.cardiomyopathy"
    r"|symptom.score|symptom.burden"
    r"|patient.reported|patient.global"
    r"|rose.dyspnea|seattle.angina|saq"
    r"|euroqol|visual.analog|vas"
    r")\b"
)


def classify_endpoint(title: str) -> str:
    """Classify an outcome title into hard/surrogate/pro/other."""
    if not title or not isinstance(title, str):
        return "other"
    if _HARD_PATTERNS.search(title):
        return "hard"
    if _SURROGATE_PATTERNS.search(title):
        return "surrogate"
    if _PRO_PATTERNS.search(title):
        return "pro"
    return "other"


class EndpointSofteningDetector(BaseDetector):
    name = "endpoint_softening"
    description = "Primary endpoints are all-surrogate or exclude hard clinical outcomes"
    aact_tables = ("outcomes",)

    def detect(
        self, master_df: pd.DataFrame, raw_tables: dict | None = None
    ) -> DetectorResult:
        nct_ids: list[str] = []
        flags: list[bool] = []
        severities: list[float] = []
        details: list[str] = []

        # Load outcomes
        outcomes_by_nct = self._load_primary_outcomes(master_df, raw_tables)

        for _, row in master_df.iterrows():
            nct = row["nct_id"]
            nct_ids.append(nct)

            primaries = outcomes_by_nct.get(nct, [])
            if not primaries:
                flags.append(False)
                severities.append(0.0)
                details.append("")
                continue

            classes = [classify_endpoint(title) for title in primaries]
            has_hard = "hard" in classes
            has_surrogate = "surrogate" in classes
            has_pro = "pro" in classes

            # P1-4: Differentiated severity levels
            if not has_hard and has_surrogate and not has_pro:
                # All-surrogate primaries
                sev = 1.0
                det = f"Surrogate-only primaries: {', '.join(primaries[:3])}"
                flags.append(True)
                severities.append(sev)
                details.append(det[:200])
            elif not has_hard and has_pro and not has_surrogate:
                # PRO-only primaries
                sev = 0.7
                det = f"PRO-only primaries: {', '.join(primaries[:3])}"
                flags.append(True)
                severities.append(sev)
                details.append(det[:200])
            elif not has_hard and (has_surrogate or has_pro):
                # Mixed surrogate+PRO, no hard
                sev = 1.0 if has_surrogate else 0.7
                det = f"All-surrogate/PRO primaries: {', '.join(primaries[:3])}"
                flags.append(True)
                severities.append(sev)
                details.append(det[:200])
            elif has_hard and has_surrogate:
                # Mixed hard+surrogate
                sev = 0.5
                det = f"Mixed primaries ({', '.join(set(classes))}): {', '.join(primaries[:3])}"
                flags.append(True)
                severities.append(sev)
                details.append(det[:200])
            elif has_hard and has_pro and not has_surrogate:
                # Mixed hard+PRO — lower concern
                sev = 0.3
                det = f"Mixed hard+PRO primaries: {', '.join(primaries[:3])}"
                flags.append(True)
                severities.append(sev)
                details.append(det[:200])
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

    def _load_primary_outcomes(
        self, master_df: pd.DataFrame, raw_tables: dict | None
    ) -> dict[str, list[str]]:
        """Load primary outcome titles per trial."""
        try:
            if raw_tables is not None:
                outcomes = raw_tables.get("outcomes")
                if outcomes is None:
                    outcomes = raw_tables.get("design_outcomes")
                if outcomes is None:
                    return {}
            else:
                from pipeline.ingest import load_aact_table
                try:
                    outcomes = load_aact_table("design_outcomes")
                except (KeyError, FileNotFoundError):
                    outcomes = load_aact_table("outcomes")
        except (KeyError, FileNotFoundError) as e:
            logger.warning("Could not load outcomes: %s", e)
            return {}

        if outcomes is None or outcomes.empty:
            return {}

        nct_set = set(master_df["nct_id"])
        outcomes = outcomes[outcomes["nct_id"].isin(nct_set)].copy()

        # Filter to primaries
        if "outcome_type" in outcomes.columns:
            outcomes = outcomes[
                outcomes["outcome_type"].str.upper().eq("PRIMARY")
            ]

        title_col = "title" if "title" in outcomes.columns else "measure"
        if title_col not in outcomes.columns:
            return {}

        result: dict[str, list[str]] = {}
        for nct, grp in outcomes.groupby("nct_id"):
            result[nct] = [str(t) for t in grp[title_col].dropna().tolist()]
        return result
