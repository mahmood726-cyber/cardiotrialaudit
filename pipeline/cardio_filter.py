"""Filter AACT studies to cardiology trials and tag sub-domains.

Strategy (C+D from design):
  1. Seed from CardioOracle CV condition patterns + drug class map
  2. Expand with full MeSH cardiovascular terms + intervention matching
  3. Union: trial matches if ANY condition OR ANY intervention is CV
  4. Tag each trial with sub-domains: HF, CAD, arrhythmia, hypertension,
     structural, vascular, prevention, VTE, other-CV

P0-1 fix: CKD/nephropathy conditions now require a CV intervention co-match
to avoid capturing pure nephrology trials.
"""
import re
from typing import Optional

import pandas as pd

from pipeline.ingest import load_aact_table

# ── Condition patterns (regex, case-insensitive) ────────────────────────
# Core CV conditions (always included regardless of intervention)
_CV_CORE_CONDITION_PART = (
    r"heart failure|cardiac failure|cardiomyopathy|myocardial"
    r"|coronary artery disease|coronary heart disease|ischemic heart"
    r"|acute coronary syndrome|angina|myocardial infarction|heart attack"
    r"|atrial fibrillation|atrial flutter|arrhythmia|tachycardia|bradycardia"
    r"|ventricular fibrillation|supraventricular"
    r"|hypertension|blood pressure|hypertensive"
    r"|aortic stenosis|mitral regurgitation|valvular|valve disease|endocarditis"
    r"|pulmonary hypertension|pulmonary arterial hypertension"
    r"|peripheral artery disease|peripheral arterial disease|peripheral vascular"
    r"|aortic aneurysm|aortic dissection|aortic disease"
    r"|atherosclerosis|carotid stenosis|cerebrovascular"
    r"|stroke|transient ischemic attack|cerebral infarction"
    r"|venous thromboembolism|pulmonary embolism|deep vein thrombosis"
    r"|cardiac arrest|sudden cardiac death|cardiac death"
    r"|pericarditis|myocarditis|cardiac tamponade"
    r"|congenital heart|tetralogy|septal defect"
    r"|cardiac rehabilitation|cardiac surgery|cardiopulmonary"
    r"|cardiovascular"
)

# CKD/nephropathy patterns — only included when paired with CV intervention
_CKD_CONDITION_PART = (
    r"chronic kidney disease|diabetic nephropathy|diabetic kidney"
    r"|cardiorenal"
)

# Combined pattern for any CV match (core + CKD)
CV_CONDITION_PATTERNS = re.compile(
    r"(?i)\b(" + _CV_CORE_CONDITION_PART + r"|" + _CKD_CONDITION_PART + r")\b"
)

# Core-only pattern (excludes CKD/nephropathy)
_CV_CORE_CONDITION_PATTERNS = re.compile(
    r"(?i)\b(" + _CV_CORE_CONDITION_PART + r")\b"
)

# CKD-only pattern (for identifying trials matched ONLY on CKD conditions)
_CKD_ONLY_PATTERNS = re.compile(
    r"(?i)\b(" + _CKD_CONDITION_PART + r")\b"
)

# ── Intervention drug patterns ──────────────────────────────────────────
_CV_DRUGS = [
    "empagliflozin", "dapagliflozin", "canagliflozin", "ertugliflozin", "sotagliflozin",
    "spironolactone", "eplerenone", "finerenone", "esaxerenone",
    "sacubitril", "entresto",
    "valsartan", "losartan", "candesartan", "irbesartan", "telmisartan", "olmesartan",
    "enalapril", "ramipril", "lisinopril", "perindopril", "captopril",
    "carvedilol", "bisoprolol", "metoprolol", "nebivolol", "atenolol",
    "semaglutide", "liraglutide", "dulaglutide", "exenatide", "tirzepatide",
    "evolocumab", "alirocumab", "inclisiran",
    "atorvastatin", "rosuvastatin", "simvastatin", "pravastatin",
    "apixaban", "rivaroxaban", "edoxaban", "dabigatran", "warfarin",
    "ticagrelor", "prasugrel", "clopidogrel", "cangrelor",
    "amlodipine", "diltiazem", "verapamil", "nifedipine",
    "amiodarone", "dronedarone", "flecainide", "sotalol",
    "nitroglycerin", "isosorbide", "hydralazine", "digoxin", "ivabradine",
    "ranolazine", "trimetazidine", "vernakalant",
]
CV_INTERVENTION_PATTERNS = re.compile(
    r"(?i)\b(?:" + "|".join(re.escape(d) for d in _CV_DRUGS) + r")\b"
)

_CV_DEVICES = [
    "percutaneous coronary intervention", "pci", "coronary stent", "drug-eluting stent",
    "bare-metal stent", "coronary angioplasty", "balloon angioplasty",
    "transcatheter aortic valve", "tavr", "tavi", "savr",
    "implantable cardioverter defibrillator", "icd",
    "cardiac resynchronization", "crt", "pacemaker",
    "left ventricular assist device", "lvad",
    "catheter ablation", "radiofrequency ablation", "cryoablation",
    "coronary artery bypass", "cabg",
    "intra-aortic balloon pump", "iabp", "ecmo",
    "mitral clip", "mitraclip", "watchman", "left atrial appendage",
]
CV_DEVICE_PATTERNS = re.compile(
    r"(?i)\b(?:" + "|".join(re.escape(d) for d in _CV_DEVICES) + r")\b"
)

# ── Sub-domain tagging rules ────────────────────────────────────────────
SUBDOMAIN_RULES = {
    "HF": re.compile(r"(?i)\b(heart failure|cardiac failure|cardiomyopathy|lvad|sacubitril|entresto|cardiac resynchronization|crt)\b"),
    "CAD": re.compile(r"(?i)\b(coronary artery disease|coronary heart|acute coronary syndrome|angina|myocardial infarction|heart attack|pci|coronary stent|drug-eluting stent|cabg|coronary bypass|coronary angioplasty)\b"),
    "arrhythmia": re.compile(r"(?i)\b(atrial fibrillation|atrial flutter|arrhythmia|tachycardia|ventricular fibrillation|supraventricular|catheter ablation|radiofrequency ablation|cryoablation|amiodarone|dronedarone|flecainide|icd|pacemaker|watchman|left atrial appendage)\b"),
    "hypertension": re.compile(r"(?i)\b(hypertension|hypertensive|blood pressure)\b"),
    "structural": re.compile(r"(?i)\b(aortic stenosis|mitral regurgitation|valvular|valve disease|tavr|tavi|savr|mitral clip|mitraclip|endocarditis)\b"),
    "vascular": re.compile(r"(?i)\b(peripheral artery|peripheral arterial|peripheral vascular|aortic aneurysm|aortic dissection|carotid stenosis|atherosclerosis|cerebrovascular|stroke|transient ischemic)\b"),
    "prevention": re.compile(r"(?i)\b(primary prevention|secondary prevention|cardiac rehabilitation|cardiovascular risk|risk reduction)\b"),
    "VTE": re.compile(r"(?i)\b(venous thromboembolism|pulmonary embolism|deep vein thrombosis|anticoagul)\b"),
}


def is_cv_condition(condition_text: str) -> bool:
    """Return True if condition text matches any cardiovascular pattern."""
    if not condition_text or not isinstance(condition_text, str):
        return False
    return bool(CV_CONDITION_PATTERNS.search(condition_text))


def is_ckd_only_condition(condition_text: str) -> bool:
    """Return True if condition matches CKD/nephropathy but NOT any core CV pattern.

    Used to identify trials that need a CV intervention co-match to be included.
    """
    if not condition_text or not isinstance(condition_text, str):
        return False
    has_ckd = bool(_CKD_ONLY_PATTERNS.search(condition_text))
    has_core_cv = bool(_CV_CORE_CONDITION_PATTERNS.search(condition_text))
    return has_ckd and not has_core_cv


def is_cv_intervention(name: str, intervention_type: str = "") -> bool:
    """Return True if intervention name matches any CV drug or device pattern."""
    if not name or not isinstance(name, str):
        return False
    if not isinstance(intervention_type, str):
        intervention_type = ""
    if intervention_type.upper() == "DEVICE":
        return bool(CV_DEVICE_PATTERNS.search(name))
    # For drugs, check drug patterns; for devices named as non-DEVICE type, also check
    return bool(CV_INTERVENTION_PATTERNS.search(name)) or bool(CV_DEVICE_PATTERNS.search(name))


def tag_subdomain(conditions: list[str], interventions: list[str]) -> list[str]:
    """Tag a trial with one or more CV sub-domains based on conditions and interventions."""
    combined = " ".join(str(x) for x in conditions + interventions if isinstance(x, str))
    tags = []
    for domain, pattern in SUBDOMAIN_RULES.items():
        if pattern.search(combined):
            tags.append(domain)
    if not tags:
        tags.append("other-CV")
    return tags


def filter_cardiology_trials(nrows_studies: int | None = None) -> pd.DataFrame:
    """Filter AACT studies to cardiology trials with sub-domain tags.

    P0-1: CKD/nephropathy-only condition matches are only included if they
    also have a CV intervention match. This prevents pure nephrology trials
    from inflating the cohort.

    Parameters
    ----------
    nrows_studies : int | None
        Optional row limit for the studies table (for testing).

    Returns
    -------
    pd.DataFrame
        Filtered DataFrame with cv_subdomains and cv_match_source columns.
    """
    study_cols = [
        "nct_id", "overall_status", "phase", "enrollment", "enrollment_type",
        "start_date", "primary_completion_date", "completion_date",
        "results_first_posted_date", "results_first_submitted_date",
        "study_first_posted_date", "last_update_posted_date",
        "study_type", "source", "source_class", "number_of_arms",
        "has_dmc", "why_stopped", "brief_title", "acronym",
        "is_fda_regulated_drug", "is_fda_regulated_device",
        "fdaaa801_violation", "number_of_groups",
    ]
    studies = load_aact_table("studies", nrows=nrows_studies, usecols=study_cols)
    studies["start_date"] = pd.to_datetime(studies["start_date"], errors="coerce")
    studies["start_year"] = studies["start_date"].dt.year
    studies = studies[(studies["start_year"] >= 2005) & (studies["start_year"] <= 2026)].copy()
    nct_ids_in_window = set(studies["nct_id"].unique())

    conditions = load_aact_table("conditions", usecols=["nct_id", "name"])
    conditions = conditions[conditions["nct_id"].isin(nct_ids_in_window)]
    conditions["is_cv"] = conditions["name"].apply(is_cv_condition)
    cv_by_condition = set(conditions[conditions["is_cv"]]["nct_id"].unique())

    # P0-1: Split condition matches into core CV vs CKD-only
    # A trial is "CKD-only" if ALL its matching conditions are CKD/nephropathy
    # (no core CV condition match).
    conditions["is_ckd_only"] = conditions["name"].apply(is_ckd_only_condition)
    conditions["is_core_cv"] = conditions["name"].apply(
        lambda x: bool(_CV_CORE_CONDITION_PATTERNS.search(str(x))) if isinstance(x, str) and x else False
    )

    # Trials that have at least one core CV condition match
    cv_core_conditions = set(
        conditions[conditions["is_core_cv"]]["nct_id"].unique()
    )
    # Trials matched ONLY on CKD/nephropathy (in cv_by_condition but not cv_core)
    cv_ckd_only = cv_by_condition - cv_core_conditions

    interventions = load_aact_table("interventions", usecols=["nct_id", "intervention_type", "name"])
    interventions = interventions[interventions["nct_id"].isin(nct_ids_in_window)]
    # P2-11: Vectorized intervention matching (replaces row-wise apply)
    name_match = interventions["name"].str.contains(CV_INTERVENTION_PATTERNS, na=False)
    device_match = interventions["name"].str.contains(CV_DEVICE_PATTERNS, na=False)
    interventions["is_cv"] = name_match | device_match
    cv_by_intervention = set(interventions[interventions["is_cv"]]["nct_id"].unique())

    # P0-1: Final set — CKD-only trials require a CV intervention co-match
    cv_nct_ids = cv_core_conditions | (cv_ckd_only & cv_by_intervention) | cv_by_intervention
    cv_studies = studies[studies["nct_id"].isin(cv_nct_ids)].copy()

    def _match_source(nct_id):
        in_cond = nct_id in cv_by_condition
        in_interv = nct_id in cv_by_intervention
        if in_cond and in_interv:
            return "both"
        elif in_cond:
            return "condition"
        else:
            return "intervention"

    cv_studies["cv_match_source"] = cv_studies["nct_id"].apply(_match_source)

    cond_by_nct = (
        conditions[conditions["nct_id"].isin(cv_nct_ids)]
        .groupby("nct_id")["name"]
        .apply(list)
        .to_dict()
    )
    interv_by_nct = (
        interventions[interventions["nct_id"].isin(cv_nct_ids)]
        .groupby("nct_id")["name"]
        .apply(list)
        .to_dict()
    )
    cv_studies["cv_subdomains"] = cv_studies["nct_id"].apply(
        lambda nct: tag_subdomain(cond_by_nct.get(nct, []), interv_by_nct.get(nct, []))
    )

    return cv_studies.reset_index(drop=True)
