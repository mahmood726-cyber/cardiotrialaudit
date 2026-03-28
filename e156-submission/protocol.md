# CardioTrialAudit — Study Protocol

## Title
Structural Integrity of Cardiology Randomised Controlled Trials: A Systematic Audit of ClinicalTrials.gov 2005-2026

## Registration
Protocol pre-registered: 2026-03-28
Target journal: BMJ (primary), PLoS Medicine / F1000Research (companion data paper)

## Objective
To quantify the prevalence and temporal trends of structural flaws in cardiology RCTs registered on ClinicalTrials.gov between 2005 and 2026 using automated registry-level flaw detection.

## Design
Cross-sectional analysis of the AACT (Aggregate Analysis of ClinicalTrials.gov) database snapshot from February 19, 2026.

## Data Source
- **Primary**: AACT pipe-delimited export (51 tables, 571,976 studies total)
- **Supplementary**: CT.gov API v2 for outcome version history (deferred to Phase B)
- **No patient-level data**: All data is registry-level metadata

## Eligibility Criteria

### Inclusion
- Registered on ClinicalTrials.gov with start_date between 2005-01-01 and 2026-12-31
- Matches cardiovascular condition patterns (heart failure, CAD, atrial fibrillation, hypertension, valvular disease, cardiomyopathy, acute coronary syndrome, peripheral artery disease, pulmonary hypertension, aortic disease, VTE, cardiac arrest, stroke, cardiorenal) OR cardiovascular intervention patterns (62 CV drugs across 14 classes + 30 CV device types)
- CKD/nephropathy-only trials included only if they also match a CV intervention

### Exclusion
- Non-cardiovascular trials
- CKD/nephropathy-only conditions without CV intervention co-match

### Expected yield
~50,000-55,000 cardiology trials

## Sub-domain Tagging
Each trial tagged with one or more: HF, CAD, arrhythmia, hypertension, structural, vascular, prevention, VTE, other-CV

## Primary Outcomes
Prevalence rate (proportion of trials flagged) for each of 10 structural flaw detectors:

### 1. Ghost Protocols
- Type A: Registered >=3 years, never started recruiting (uses study_first_posted_date)
- Type B: Status Unknown/Suspended, no update >=2 years
- Type C: Completed/Terminated, no results posted >=13 months after primary completion (extended to 36 months for trials with result_agreements)
- Severity: years overdue / 5.0, capped at 1.0

### 2. Outcome Switching
- Fuzzy text matching (rapidfuzz token_sort_ratio) between design_outcomes (registered) and outcomes (reported results)
- Score <60 = likely switch; unregistered new primaries = severity 1.0
- Known limitation: AACT design_outcomes reflects current registration, not original

### 3. Population Distortion
- Age restriction: max_age below expected median for subdomain (HF:72, CAD:68, arrhythmia:72, hypertension:65, structural:78, vascular:70, VTE:65, prevention:55)
- Comorbidity exclusions: CKD, diabetes, liver disease, cognitive impairment, cancer, anemia, obesity, COPD, frailty
- Gender restriction: single-gender adds 0.1
- Threshold: composite score >= 0.2

### 4. Sample Size Decay
- Early termination (status=TERMINATED)
- Enrollment still "ANTICIPATED" despite completion
- Attrition rate >20% (from drop_withdrawals table)

### 5. Sponsor Concentration
- Industry sponsorship + top-5 sponsor per year
- Industry share >40% of year's trials
- Severity: composite of industry status + market share

### 6. Geographic Shifts
- LMIC site proportion from facilities table
- World Bank 2024 income classification (Poland, Hungary = HIC; Romania, Bulgaria = UMIC)
- Flag: >50% LMIC sites with >=3 total sites

### 7. Results Delay
- Days from primary_completion_date to results_first_posted_date
- FDAAA temporal stratification:
  - Pre-2008: severity x0.1 (no legal mandate)
  - 2008-2016: severity x0.5 (FDAAA existed, low enforcement)
  - 2017+: full severity (Final Rule enforcement)
- fdaaa801_violation=True forces severity to 1.0

### 8. Endpoint Softening
- Classification: hard (mortality, MI, stroke, hospitalization, MACE), surrogate (biomarkers, imaging, BP, 6MWT, NYHA), PRO (KCCQ, EQ-5D, symptom scores)
- Surrogate-only primary = severity 1.0; PRO-only = 0.7; mixed hard+surrogate = 0.5; mixed hard+PRO = 0.3

### 9. Comparator Manipulation
- Placebo/sham detection in interventions and design_groups
- Cross-reference with SOC conditions: HF, CAD, arrhythmia, hypertension, VTE, structural, prevention, vascular
- Placebo when SOC exists = severity 0.5; add-on design = 0.2

### 10. Statistical Fragility
- Fisher exact test fragility index for binary primary outcomes
- Modifies fewer-events arm only (Walsh et al. 2014)
- FI <=3 = severity 0.8, FI 4-7 = 0.5, FI 8-15 = 0.3
- Coverage limited to trials with extractable 2x2 tables

## Composite Flaw Score
- `flaw_count` = number of detectors flagging the trial
- `composite_severity` = mean severity across flagged detectors (0 if no flaws)

## Temporal Analysis
- Time window: 2005-2026
- Yearly prevalence rates for each detector
- 3-year binned rates for manuscript figures
- Right-truncation correction: time-dependent detectors exclude trials with insufficient observation time
- Regulatory milestone annotations: 2007 FDAAA, 2017 Final Rule, 2020 COVID

## Statistical Methods
- Detection rates: proportion with Wilson 95% CI
- Trends: rates per year/bin; regulatory breakpoints annotated
- Sub-domain comparison: rates stratified by CV subdomain
- Sponsor analysis: industry vs academic/other comparison

## Sensitivity Analyses
1. Including vs excluding CKD-only trials
2. 13-month vs 24-month ghost protocol grace period
3. With vs without result_agreements extensions
4. FDAAA-applicable trials only (2017+)

## Software
- Python 3.13, pandas, numpy, scipy, statsmodels, rapidfuzz
- 162 automated tests
- 5-persona code review (statistical, security, UX, software engineering, domain expert)
- All code at C:\Models\CardioTrialAudit\

## Outputs
1. BMJ manuscript: "Structural Integrity of Cardiology Randomised Controlled Trials"
2. Companion data paper (PLoS Medicine / F1000Research) with full dataset
3. Interactive HTML dashboard (7 tabs, Chart.js)
4. E156 micro-paper

## Ethics
No patient-level data. All data from publicly accessible ClinicalTrials.gov registry. No ethics approval required.

## Timeline
- Phase A (complete): Pipeline + 10 detectors + dashboard + review
- Phase B: CT.gov API for outcome switching, Joinpoint regression, subtherapeutic dose detection
- Phase C: Manuscript drafting and submission

## Authors
Mahmood M (corresponding)

## Version History
- v1.0 (2026-03-28): Initial protocol
