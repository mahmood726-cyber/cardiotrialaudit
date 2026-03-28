"""Population Distortion detector — overly restrictive eligibility criteria.

Flags trials with:
- Age restrictions below expected median for CV subdomain
- Excessive comorbidity exclusions
- Single-gender restrictions
"""
import re

import pandas as pd

from pipeline.detectors.base import BaseDetector, DetectorResult

# Expected median ages by CV subdomain (from epidemiology)
# P1-10: HF set to 72 as compromise between HFrEF (~67) and HFpEF (~76).
# HFrEF trials (PARADIGM-HF, DAPA-HF) enrol younger patients (~67);
# HFpEF trials (EMPEROR-Preserved, DELIVER) enrol older (~76).
# 72 is the weighted average assuming ~55% HFrEF / ~45% HFpEF trial mix.
_SUBDOMAIN_MEDIAN_AGE: dict[str, float] = {
    "HF": 72.0,
    "CAD": 68.0,
    "arrhythmia": 72.0,
    "hypertension": 65.0,
    "structural": 75.0,
    "vascular": 70.0,
    "prevention": 60.0,
    "VTE": 65.0,
    "other-CV": 68.0,
}

# Comorbidity exclusion patterns (search in exclusion criteria section)
# P1-9: Added COPD and frailty patterns
_COMORBIDITY_PATTERNS = [
    (re.compile(r"(?i)\b(chronic kidney disease|ckd|renal insufficiency|egfr\s*<)\b"), "CKD"),
    (re.compile(r"(?i)\b(diabetes|diabetic|hba1c)\b"), "diabetes"),
    (re.compile(r"(?i)\b(hepatic|liver disease|cirrhosis|alt\s*>|ast\s*>)\b"), "liver"),
    (re.compile(r"(?i)\b(cognitive impairment|dementia|alzheimer)\b"), "cognitive"),
    (re.compile(r"(?i)\b(malignancy|cancer|neoplasm|tumor)\b"), "cancer"),
    (re.compile(r"(?i)\b(anemia|hemoglobin\s*<|hgb\s*<)\b"), "anemia"),
    (re.compile(r"(?i)\b(obesity|bmi\s*>|body mass index\s*>)\b"), "obesity"),
    (re.compile(r"(?i)\b(copd|chronic obstructive|pulmonary disease|emphysema)\b"), "COPD"),
    (re.compile(r"(?i)\b(frail|frailty|poor functional status|bedridden)\b"), "frailty"),
]


def _extract_exclusion_section(criteria: str) -> str:
    """Extract text after 'Exclusion Criteria' header, if present."""
    if not criteria or not isinstance(criteria, str):
        return ""
    # AACT criteria field typically has "Inclusion Criteria:...Exclusion Criteria:..."
    m = re.search(r"(?i)exclusion\s+criteria", criteria)
    if m:
        return criteria[m.start():]
    return ""


def _get_primary_subdomain(cv_subdomains) -> str:
    """Get the first subdomain from cv_subdomains (list or string)."""
    if isinstance(cv_subdomains, list) and cv_subdomains:
        return cv_subdomains[0]
    if isinstance(cv_subdomains, str) and cv_subdomains:
        # May be stored as string repr of list
        cleaned = cv_subdomains.strip("[]' \"")
        return cleaned.split(",")[0].strip().strip("'\"")
    return "other-CV"


class PopulationDistortionDetector(BaseDetector):
    name = "population_distortion"
    description = "Overly restrictive eligibility excluding real-world patients"
    aact_tables: tuple[str, ...] = ()

    def detect(
        self, master_df: pd.DataFrame, raw_tables: dict | None = None
    ) -> DetectorResult:
        nct_ids: list[str] = []
        flags: list[bool] = []
        severities: list[float] = []
        details: list[str] = []

        for _, row in master_df.iterrows():
            nct_ids.append(row["nct_id"])
            score = 0.0
            issues: list[str] = []

            # 1. Age restriction check
            max_age = row.get("max_age_years")
            if max_age is not None and not pd.isna(max_age):
                subdomain = _get_primary_subdomain(row.get("cv_subdomains", "other-CV"))
                expected = _SUBDOMAIN_MEDIAN_AGE.get(subdomain, 68.0)
                if float(max_age) < expected:
                    gap = (expected - float(max_age)) / expected
                    score += gap  # 0-1 proportional to how far below expected
                    issues.append(
                        f"Max age {max_age:.0f}y < expected {expected:.0f}y for {subdomain}"
                    )

            # 2. Comorbidity exclusions
            criteria = str(row.get("criteria", "") or "")
            exclusion_text = _extract_exclusion_section(criteria)
            if exclusion_text:
                excluded = []
                for pattern, label in _COMORBIDITY_PATTERNS:
                    if pattern.search(exclusion_text):
                        excluded.append(label)
                if excluded:
                    # Each excluded comorbidity adds ~0.1 severity
                    score += len(excluded) * 0.1
                    issues.append(f"Excludes: {', '.join(excluded)}")

            # 3. Gender restriction
            gender = str(row.get("gender", "") or "").upper()
            if gender in ("MALE", "FEMALE"):
                score += 0.1
                issues.append(f"Gender restricted: {gender}")

            flagged = score >= 0.2
            flags.append(flagged)
            severities.append(round(min(score, 1.0), 4))
            details.append("; ".join(issues) if flagged else "")

        return DetectorResult(
            nct_ids=nct_ids,
            flaw_detected=flags,
            severity=severities,
            detail=details,
        )
