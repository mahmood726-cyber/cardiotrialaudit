# E156 Protocol — `CardioTrialAudit`

This repository is the source code and dashboard backing an E156 micro-paper on the [E156 Student Board](https://mahmood726-cyber.github.io/e156/students.html).

---

## `[15]` Structural Flaw Prevalence in 52,765 Cardiology Trials: An Automated Registry Audit

**Type:** methods  |  ESTIMAND: Structural flaw prevalence  
**Data:** AACT ClinicalTrials.gov export (Feb 2026), 52,765 cardiology trials

### 156-word body

Among 52,765 cardiology trials on ClinicalTrials.gov from 2005 to 2026, what is the prevalence of structural design flaws detectable by automated registry auditing? We applied ten rule-based detectors to the AACT February 2026 snapshot screening protocol fields, posting dates, eligibility criteria, endpoint types, and comparator arms. The primary estimand was structural flaw prevalence computed as a proportion of screened entries. The overall prevalence of any structural flaw was 67.3 percent with a 95% CI of 66.1 to 68.5, ghost protocols affected 61.7 percent, and mean flaws per entry were 1.81. Ghost rates peaked at 79 percent in 2014 then declined, results delay fell from 80 to 40 percent between 2008 and 2022, and endpoint softening dropped from 26 to 16 percent. Nearly two thirds of registered cardiology trials carry at least one structural flaw detectable by automated screening before peer review. This analysis covers registered metadata and cannot detect unreported amendments, selective reporting, or unregistered studies.

### Submission metadata

```
Corresponding author: Mahmood Ahmad <mahmood.ahmad2@nhs.net>
ORCID: 0000-0001-9107-3704
Affiliation: Tahir Heart Institute, Rabwah, Pakistan

Links:
  Code:      https://github.com/mahmood726-cyber/CardioTrialAudit
  Protocol:  https://github.com/mahmood726-cyber/CardioTrialAudit/blob/main/E156-PROTOCOL.md
  Dashboard: https://mahmood726-cyber.github.io/cardiotrialaudit/

References (topic pack: CT.gov / ClinicalTrials.gov audit):
  1. Zarin DA, Tse T, Williams RJ, Rajakannan T. 2017. Update on trial registration 11 years after the ICMJE policy was established. N Engl J Med. 376(4):383-391. doi:10.1056/NEJMsr1601330
  2. Anderson ML, Chiswell K, Peterson ED, Tasneem A, Topping J, Califf RM. 2015. Compliance with results reporting at ClinicalTrials.gov. N Engl J Med. 372(11):1031-1039. doi:10.1056/NEJMsa1409364

Data availability: No patient-level data used. Analysis derived exclusively
  from publicly available aggregate records. All source identifiers are in
  the protocol document linked above.

Ethics: Not required. Study uses only publicly available aggregate data; no
  human participants; no patient-identifiable information; no individual-
  participant data. No institutional review board approval sought or required
  under standard research-ethics guidelines for secondary methodological
  research on published literature.

Funding: None.

Competing interests: MA serves on the editorial board of Synthēsis (the
  target journal); MA had no role in editorial decisions on this
  manuscript, which was handled by an independent editor of the journal.

Author contributions (CRediT):
  [STUDENT REWRITER, first author] — Writing – original draft, Writing –
    review & editing, Validation.
  [SUPERVISING FACULTY, last/senior author] — Supervision, Validation,
    Writing – review & editing.
  Mahmood Ahmad (middle author, NOT first or last) — Conceptualization,
    Methodology, Software, Data curation, Formal analysis, Resources.

AI disclosure: Computational tooling (including AI-assisted coding via
  Claude Code [Anthropic]) was used to develop analysis scripts and assist
  with data extraction. The final manuscript was human-written, reviewed,
  and approved by the author; the submitted text is not AI-generated. All
  quantitative claims were verified against source data; cross-validation
  was performed where applicable. The author retains full responsibility for
  the final content.

Preprint: Not preprinted.

Reporting checklist: PRISMA 2020 (methods-paper variant — reports on review corpus).

Target journal: ◆ Synthēsis (https://www.synthesis-medicine.org/index.php/journal)
  Section: Methods Note — submit the 156-word E156 body verbatim as the main text.
  The journal caps main text at ≤400 words; E156's 156-word, 7-sentence
  contract sits well inside that ceiling. Do NOT pad to 400 — the
  micro-paper length is the point of the format.

Manuscript license: CC-BY-4.0.
Code license: MIT.

SUBMITTED: [ ]
```


---

_Auto-generated from the workbook by `C:/E156/scripts/create_missing_protocols.py`. If something is wrong, edit `rewrite-workbook.txt` and re-run the script — it will overwrite this file via the GitHub API._