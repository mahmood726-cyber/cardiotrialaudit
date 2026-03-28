## Multi-Persona Review: CardioTrialAudit (Full Pipeline + Dashboard)
### Date: 2026-03-28
### Summary: 8 P0, 17 P1, 19 P2

---

#### P0 -- Critical (Must Fix)

- **P0-1** [Domain]: CKD/nephropathy over-inclusion inflates denominator to 54K (expected 15-30K). CKD-only trials are not cardiovascular. (`cardio_filter.py:38-39`)
  - Fix: Require CKD trials to also match a CV intervention OR have a CV primary endpoint. Run sensitivity analysis.

- **P0-2** [Domain]: FDAAA applicability period not enforced — results delay applied to 2005-2007 trials that had NO legal obligation to post. (`results_delay.py`)
  - Fix: Stratify by pre-FDAAA (2005-07), FDAAA (2008-16), Final Rule (2017+). Use `fdaaa801_violation` field.

- **P0-3** [Domain]: Ghost Type C 13-month grace ignores statutory extensions (2+ years). 61% rate likely inflated by 15-20% of trials with legitimate delay certifications. (`ghost_protocols.py:58-66`)
  - Fix: Check `result_agreements` table. Report both 13-month and 24-month rates.

- **P0-4** [Domain]: 6MWT and NYHA misclassified as PRO — both are objective/physician-assessed, should be surrogates. (`endpoint_softening.py:50-61`)
  - Fix: Move `6.?minute.walk|6mwd|6mwt` and `nyha|functional.class` from `_PRO_PATTERNS` to `_SURROGATE_PATTERNS`.

- **P0-5** [Stats]: `_extract_2x2_tables` produces invalid 2x2 tables — arm totals = sum of outcome counts, not arm sizes. Detector never fires for single-outcome trials. (`statistical_fragility.py:196-226`)
  - Fix: Get arm sizes from `participant_flows` or `result_groups`, not outcome_counts sum.

- **P0-6** [UX]: `--text-muted` (#718096) fails WCAG AA 4.5:1 contrast on ALL dark backgrounds (3.54:1 on bg-card). Affects 9+ components. (`index.html:19-20`)
  - Fix: Change `--text-muted: #718096` to `#8a9bb5` (5.0:1).

- **P0-7** [UX]: No `aria-live` region for filter/pagination updates — screen readers get zero notification of content changes. (`index.html:527,637,1526`)
  - Fix: Add `aria-live="polite"` region, announce counts on filter/page change.

- **P0-8** [UX]: `<canvas>` charts have no text alternative — 10 canvases with zero accessible content. (`index.html:532-728`)
  - Fix: Add `role="img"` + dynamic `aria-label` to each canvas after rendering.

#### P1 -- Important (Should Fix)

- **P1-1** [Stats]: No right-truncation correction in trends — 2024-2026 ghost/delay rates artificially depressed because trials haven't had enough time to fail. (`trends.py`, `ghost_protocols.py`, `results_delay.py`)
  - Fix: Exclude trials without sufficient observation time from denominator, or truncate trend charts.

- **P1-2** [Stats]: Hardcoded `_NOW = pd.Timestamp("2026-02-19")` — becomes stale on data refresh. (`ghost_protocols.py:11`, `results_delay.py:14`)
  - Fix: Derive from AACT snapshot metadata or pass as parameter.

- **P1-3** [Stats]: Ghost Type A uses `start_date` (planned) instead of `study_first_posted_date` (registration date). (`ghost_protocols.py:41`)
  - Fix: Use `study_first_posted_date`.

- **P1-4** [Stats]: PRO-only trials get severity 1.0 same as surrogates — FDA-endorsed PRO endpoints (KCCQ) should be differentiated. (`endpoint_softening.py:109-115`)
  - Fix: Surrogate-only = 1.0, PRO-only = 0.7, mixed = 0.5.

- **P1-5** [Security]: CSV formula injection — no sanitization of `=+@\t\r` prefixes per lessons.md. (`export.py:191,221`)
  - Fix: Add `_sanitize_csv_cell` to prepend `'` to dangerous starting characters.

- **P1-6** [Security]: Chart.js CDN lacks SRI integrity hash. (`index.html:7`)
  - Fix: Add `integrity="sha256-..."` and `crossorigin="anonymous"`.

- **P1-7** [Security]: `?data=` allows arbitrary URL fetch. (`index.html:876-881`)
  - Fix: Restrict to relative paths starting with `./` or `../`.

- **P1-8** [Domain]: "prevention" and "vascular" excluded from SOC conditions — secondary prevention placebo trials not flagged. (`comparator_manipulation.py:16-18`)
  - Fix: Add to `_SOC_SUBDOMAINS`.

- **P1-9** [Domain]: Missing COPD/frailty comorbidity exclusion patterns — COPD affects ~30% of HF patients. (`population_distortion.py:28-36`)
  - Fix: Add COPD and frailty regex patterns.

- **P1-10** [Domain]: HF median age 76 doesn't distinguish HFrEF (~65-70) vs HFpEF (~76). (`population_distortion.py:15-25`)
  - Fix: Split HF benchmark or use 72 as compromise.

- **P1-11** [Domain]: Poland/Hungary classified HIC, Romania/Bulgaria should be UMIC per World Bank 2024. (`geographic_shifts.py:28`)
  - Fix: Update classification per formal World Bank list.

- **P1-12** [SWE]: `iterrows()` on 54K+ rows in 7 detectors — ~100x slower than vectorized pandas. (`ghost_protocols.py:27`, etc.)
  - Fix: Vectorize with pandas boolean operations for simple detectors.

- **P1-13** [SWE]: Runner `merge()` in loop creates 10 DataFrame copies. (`runner.py:83`)
  - Fix: Collect detector DFs, use single `pd.concat(axis=1)`.

- **P1-14** [UX]: Heatmap cells at rates 40-50% — white text on orange fails 4.5:1. (`index.html:822-845`)
  - Fix: Lower `textColorForBg` threshold or use darker text.

- **P1-15** [UX]: Explorer table rows lack focus-visible styling. (`index.html:1499`)
  - Fix: Add `:focus-visible` CSS for `tr[data-idx]`.

- **P1-16** [UX]: Sort state not communicated — no `aria-sort` on column headers. (`index.html:1394-1484`)
  - Fix: Set `aria-sort="ascending/descending"` on active sort header.

- **P1-17** [UX]: Loading overlay not announced — no `role="status"` or `aria-live`. (`index.html:497-500`)
  - Fix: Add `role="status" aria-live="polite"` to overlay.

#### P2 -- Minor (19 items)

- **P2-1** [Stats]: Ghost Type C and Results Delay overlap — inflates flaw_count for same trials
- **P2-2** [Stats]: mean_severity semantics differ between composite.py and trends.py
- **P2-3** [Stats]: Sponsor detector flags nearly all industry trials (threshold too low)
- **P2-4** [Stats]: RuntimeWarning on zero-flaw rows in composite.py
- **P2-5** [Stats]: Last bin labeled "2026-2026" instead of just "2026"
- **P2-6** [Stats]: eGFR exclusion not threshold-aware (< 30 vs < 60)
- **P2-7** [Security]: Auxiliary table loads without `usecols` (memory optimization)
- **P2-8** [Security]: Heatmap year values unescaped (safe since numeric)
- **P2-9** [SWE]: Duplicated `_load_table` pattern across 5 detectors
- **P2-10** [SWE]: DETECTOR_NAMES duplicated in composite.py and runner.py
- **P2-11** [SWE]: `cardio_filter.py` uses apply() on 967K intervention rows
- **P2-12** [SWE]: Missing spec features (Joinpoint regression, CT.gov API)
- **P2-13** [SWE]: BaseDetector mutable class-level default `aact_tables: list = []`
- **P2-14** [Domain]: Missing COPD/frailty/AF in comorbidity exclusions
- **P2-15** [Domain]: Subtherapeutic dose detection not implemented (spec gap)
- **P2-16** [Domain]: Endpoint softening missing "urgent HF visit" / "worsening HF event"
- **P2-17** [UX]: No `<main>` landmark wrapper
- **P2-18** [UX]: Tabpanels lack `aria-labelledby`
- **P2-19** [UX]: No skip-to-content link

#### False Positive Watch
- Fragility Index modifying ONE arm only — CORRECT per Walsh et al.
- `is_cv_intervention` matching both drug AND device patterns — CORRECT by design
- `cv_subdomains` as list (multi-tag) — CORRECT for overlapping conditions
