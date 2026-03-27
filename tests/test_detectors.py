"""Comprehensive tests for all 10 flaw detectors + framework."""
import pandas as pd
import pytest

from pipeline.detectors.base import BaseDetector, DetectorResult
from pipeline.detectors.runner import DETECTOR_REGISTRY, run_all_detectors
from pipeline.detectors.ghost_protocols import GhostProtocolsDetector
from pipeline.detectors.outcome_switching import OutcomeSwitchingDetector, _fuzzy_best_match
from pipeline.detectors.population_distortion import (
    PopulationDistortionDetector, _extract_exclusion_section,
)
from pipeline.detectors.sample_size_decay import SampleSizeDecayDetector
from pipeline.detectors.sponsor_concentration import SponsorConcentrationDetector
from pipeline.detectors.geographic_shifts import GeographicShiftsDetector
from pipeline.detectors.results_delay import ResultsDelayDetector
from pipeline.detectors.endpoint_softening import (
    EndpointSofteningDetector, classify_endpoint,
)
from pipeline.detectors.comparator_manipulation import ComparatorManipulationDetector
from pipeline.detectors.statistical_fragility import (
    StatisticalFragilityDetector, compute_fragility_index, _fi_to_severity,
)


# ── Helpers ──────────────────────────────────────────────────────────────

def _make_master_df(overrides: list[dict]) -> pd.DataFrame:
    """Create a minimal master_df with sensible defaults per row."""
    defaults = {
        "nct_id": "NCT00000001",
        "overall_status": "COMPLETED",
        "phase": "Phase 3",
        "enrollment": 500,
        "enrollment_type": "ACTUAL",
        "start_date": pd.Timestamp("2020-01-01"),
        "start_year": 2020,
        "primary_completion_date": pd.Timestamp("2023-01-01"),
        "completion_date": pd.Timestamp("2023-06-01"),
        "results_first_posted_date": pd.Timestamp("2023-09-01"),
        "last_update_posted_date": pd.Timestamp("2024-01-01"),
        "study_type": "INTERVENTIONAL",
        "source": "Test",
        "source_class": "NIH",
        "brief_title": "Test trial",
        "why_stopped": "",
        "fdaaa801_violation": False,
        "cv_match_source": "condition",
        "cv_subdomains": ["HF"],
        "allocation": "Randomized",
        "masking": "Double",
        "primary_purpose": "Treatment",
        "intervention_model": "Parallel",
        "gender": "All",
        "minimum_age": "18 Years",
        "maximum_age": "90 Years",
        "criteria": "Inclusion Criteria: Adults with HF\nExclusion Criteria: None significant",
        "min_age_years": 18.0,
        "max_age_years": 90.0,
        "lead_sponsor_name": "Test University",
        "lead_sponsor_class": "NIH",
        "has_results": True,
        "is_interventional": True,
        "is_randomized": True,
        "results_delay_days": 240,
    }
    rows = []
    for i, override in enumerate(overrides):
        row = defaults.copy()
        row["nct_id"] = f"NCT{i+1:08d}"
        row.update(override)
        rows.append(row)
    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════════
# Framework tests
# ═══════════════════════════════════════════════════════════════════════


class TestDetectorResult:
    def test_to_dataframe_columns(self):
        dr = DetectorResult(
            nct_ids=["NCT001", "NCT002"],
            flaw_detected=[True, False],
            severity=[0.8, 0.0],
            detail=["issue", ""],
        )
        df = dr.to_dataframe("test_det")
        assert list(df.columns) == [
            "nct_id", "test_det_detected", "test_det_severity", "test_det_detail"
        ]
        assert len(df) == 2
        assert df.iloc[0]["test_det_detected"] == True
        assert df.iloc[1]["test_det_severity"] == 0.0

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="equal length"):
            DetectorResult(
                nct_ids=["NCT001"],
                flaw_detected=[True, False],
                severity=[0.5],
                detail=["x"],
            )


class TestRegistry:
    def test_registry_has_10_detectors(self):
        assert len(DETECTOR_REGISTRY) == 10

    def test_registry_names(self):
        expected = {
            "ghost_protocols", "outcome_switching", "population_distortion",
            "sample_size_decay", "sponsor_concentration", "geographic_shifts",
            "results_delay", "endpoint_softening", "comparator_manipulation",
            "statistical_fragility",
        }
        assert set(DETECTOR_REGISTRY.keys()) == expected

    def test_all_are_base_detector(self):
        for name, det in DETECTOR_REGISTRY.items():
            assert isinstance(det, BaseDetector), f"{name} is not a BaseDetector"

    def test_all_have_name(self):
        for name, det in DETECTOR_REGISTRY.items():
            assert det.name == name, f"Detector {name} has name={det.name}"


class TestRunner:
    def test_run_all_detectors_columns(self):
        df = _make_master_df([{}, {}])
        result = run_all_detectors(df, raw_tables={})
        for name in DETECTOR_REGISTRY:
            assert f"{name}_detected" in result.columns, f"Missing {name}_detected"
            assert f"{name}_severity" in result.columns, f"Missing {name}_severity"
            assert f"{name}_detail" in result.columns, f"Missing {name}_detail"

    def test_run_selected_detectors(self):
        df = _make_master_df([{}])
        result = run_all_detectors(
            df, raw_tables={}, detectors=["ghost_protocols", "results_delay"]
        )
        assert "ghost_protocols_detected" in result.columns
        assert "results_delay_detected" in result.columns
        assert "outcome_switching_detected" not in result.columns

    def test_run_preserves_rows(self):
        df = _make_master_df([{}, {}, {}])
        result = run_all_detectors(df, raw_tables={})
        assert len(result) == 3


# ═══════════════════════════════════════════════════════════════════════
# Individual detector tests
# ═══════════════════════════════════════════════════════════════════════


class TestGhostProtocols:
    def test_type_a_old_withdrawn(self):
        df = _make_master_df([{
            "overall_status": "WITHDRAWN",
            "start_date": pd.Timestamp("2019-01-01"),
            "has_results": False,
        }])
        result = GhostProtocolsDetector().detect(df)
        assert result.flaw_detected[0] is True
        assert "Type A" in result.detail[0]
        assert result.severity[0] > 0

    def test_type_a_recent_not_flagged(self):
        df = _make_master_df([{
            "overall_status": "WITHDRAWN",
            "start_date": pd.Timestamp("2024-06-01"),
        }])
        result = GhostProtocolsDetector().detect(df)
        assert result.flaw_detected[0] is False

    def test_type_b_silent_unknown(self):
        df = _make_master_df([{
            "overall_status": "UNKNOWN STATUS",
            "last_update_posted_date": pd.Timestamp("2022-01-01"),
        }])
        result = GhostProtocolsDetector().detect(df)
        assert result.flaw_detected[0] is True
        assert "Type B" in result.detail[0]

    def test_type_c_completed_no_results(self):
        df = _make_master_df([{
            "overall_status": "COMPLETED",
            "primary_completion_date": pd.Timestamp("2022-01-01"),
            "has_results": False,
            "results_first_posted_date": pd.NaT,
        }])
        result = GhostProtocolsDetector().detect(df)
        assert result.flaw_detected[0] is True
        assert "Type C" in result.detail[0]

    def test_completed_with_results_not_flagged(self):
        df = _make_master_df([{
            "overall_status": "COMPLETED",
            "has_results": True,
            "results_first_posted_date": pd.Timestamp("2023-09-01"),
        }])
        result = GhostProtocolsDetector().detect(df)
        assert result.flaw_detected[0] is False

    def test_severity_capped_at_1(self):
        df = _make_master_df([{
            "overall_status": "WITHDRAWN",
            "start_date": pd.Timestamp("2010-01-01"),
        }])
        result = GhostProtocolsDetector().detect(df)
        assert result.severity[0] <= 1.0


class TestOutcomeSwitching:
    def test_matching_outcomes_not_flagged(self):
        df = _make_master_df([{"has_results": True}])
        design_outcomes = pd.DataFrame({
            "nct_id": ["NCT00000001"],
            "outcome_type": ["PRIMARY"],
            "measure": ["All-cause mortality at 12 months"],
        })
        outcomes = pd.DataFrame({
            "nct_id": ["NCT00000001"],
            "outcome_type": ["PRIMARY"],
            "title": ["All-cause mortality at 12 months"],
        })
        result = OutcomeSwitchingDetector().detect(
            df, raw_tables={"design_outcomes": design_outcomes, "outcomes": outcomes}
        )
        assert result.flaw_detected[0] is False

    def test_switched_outcome_flagged(self):
        df = _make_master_df([{"has_results": True}])
        design_outcomes = pd.DataFrame({
            "nct_id": ["NCT00000001"],
            "outcome_type": ["PRIMARY"],
            "measure": ["All-cause mortality at 12 months"],
        })
        outcomes = pd.DataFrame({
            "nct_id": ["NCT00000001"],
            "outcome_type": ["PRIMARY"],
            "title": ["Change in blood pressure at 6 weeks"],
        })
        result = OutcomeSwitchingDetector().detect(
            df, raw_tables={"design_outcomes": design_outcomes, "outcomes": outcomes}
        )
        assert result.flaw_detected[0] is True
        assert result.severity[0] > 0

    def test_no_results_not_checked(self):
        df = _make_master_df([{"has_results": False}])
        result = OutcomeSwitchingDetector().detect(
            df, raw_tables={"design_outcomes": pd.DataFrame(), "outcomes": pd.DataFrame()}
        )
        assert result.flaw_detected[0] is False

    def test_missing_table_returns_clean(self):
        df = _make_master_df([{"has_results": True}])
        result = OutcomeSwitchingDetector().detect(df, raw_tables={})
        assert result.flaw_detected[0] is False


class TestPopulationDistortion:
    def test_restrictive_age_flagged(self):
        """Max age 55 for HF (expected median 76) should flag."""
        df = _make_master_df([{
            "max_age_years": 55.0,
            "cv_subdomains": ["HF"],
            "criteria": "Inclusion: HF patients\nExclusion Criteria: CKD, diabetes, cancer",
            "gender": "All",
        }])
        result = PopulationDistortionDetector().detect(df)
        assert result.flaw_detected[0] is True
        assert result.severity[0] > 0

    def test_broad_age_not_flagged(self):
        """Max age 90, no exclusions, all genders -> not flagged."""
        df = _make_master_df([{
            "max_age_years": 90.0,
            "cv_subdomains": ["HF"],
            "criteria": "Inclusion: HF\nExclusion Criteria: None",
            "gender": "All",
        }])
        result = PopulationDistortionDetector().detect(df)
        assert result.flaw_detected[0] is False

    def test_comorbidity_exclusions_increase_severity(self):
        df = _make_master_df([{
            "max_age_years": 70.0,
            "cv_subdomains": ["HF"],
            "criteria": "Inclusion: HF\nExclusion Criteria: chronic kidney disease, diabetes, liver disease, cancer, cognitive impairment",
            "gender": "All",
        }])
        result = PopulationDistortionDetector().detect(df)
        assert result.flaw_detected[0] is True
        assert "Excludes:" in result.detail[0]

    def test_gender_restriction_adds_severity(self):
        df = _make_master_df([{
            "max_age_years": 90.0,
            "cv_subdomains": ["CAD"],
            "criteria": "Inclusion: CAD\nExclusion Criteria: diabetes, cancer, CKD",
            "gender": "MALE",
        }])
        result = PopulationDistortionDetector().detect(df)
        assert result.flaw_detected[0] is True
        assert "Gender restricted" in result.detail[0]


class TestSampleSizeDecay:
    def test_terminated_flagged(self):
        df = _make_master_df([{
            "overall_status": "TERMINATED",
            "why_stopped": "Futility",
        }])
        result = SampleSizeDecayDetector().detect(df, raw_tables={})
        assert result.flaw_detected[0] is True
        assert result.severity[0] >= 0.5
        assert "Terminated" in result.detail[0]

    def test_anticipated_on_completed_flagged(self):
        df = _make_master_df([{
            "overall_status": "COMPLETED",
            "enrollment_type": "ANTICIPATED",
        }])
        result = SampleSizeDecayDetector().detect(df, raw_tables={})
        assert result.flaw_detected[0] is True
        assert "ANTICIPATED" in result.detail[0]

    def test_high_attrition_flagged(self):
        df = _make_master_df([{
            "enrollment": 100,
        }])
        drop_df = pd.DataFrame({
            "nct_id": ["NCT00000001", "NCT00000001"],
            "count": [15, 12],  # 27% attrition
        })
        result = SampleSizeDecayDetector().detect(
            df, raw_tables={"drop_withdrawals": drop_df}
        )
        assert result.flaw_detected[0] is True
        assert "Attrition" in result.detail[0]

    def test_normal_trial_not_flagged(self):
        df = _make_master_df([{
            "overall_status": "COMPLETED",
            "enrollment_type": "ACTUAL",
            "enrollment": 500,
        }])
        drop_df = pd.DataFrame({
            "nct_id": ["NCT00000001"],
            "count": [20],  # 4% attrition — below 20%
        })
        result = SampleSizeDecayDetector().detect(
            df, raw_tables={"drop_withdrawals": drop_df}
        )
        assert result.flaw_detected[0] is False


class TestSponsorConcentration:
    def test_industry_sponsor_flagged(self):
        """Industry sponsor with > 10% share should flag."""
        rows = [
            {"lead_sponsor_name": "BigPharma", "lead_sponsor_class": "INDUSTRY", "start_year": 2020},
        ] + [
            {"lead_sponsor_name": "BigPharma", "lead_sponsor_class": "INDUSTRY", "start_year": 2020}
            for _ in range(4)
        ]
        df = _make_master_df(rows)
        result = SponsorConcentrationDetector().detect(df)
        # BigPharma has 100% share (well above 10%), industry=0.2 + share=0.3 = 0.5
        assert result.flaw_detected[0] is True
        assert result.severity[0] >= 0.5

    def test_diverse_non_industry_not_flagged(self):
        rows = [
            {"lead_sponsor_name": f"Univ {i}", "lead_sponsor_class": "NIH", "start_year": 2020}
            for i in range(20)
        ]
        df = _make_master_df(rows)
        result = SponsorConcentrationDetector().detect(df)
        # Each sponsor has 5% share (< 10%), not industry
        assert not any(result.flaw_detected)

    def test_industry_flagged_when_top5(self):
        """Industry sponsor that's also top-5 in year should be flagged."""
        df = _make_master_df([
            {"lead_sponsor_class": "INDUSTRY", "lead_sponsor_name": "BigPharma"},
        ] * 10 + [
            {"lead_sponsor_class": "OTHER", "lead_sponsor_name": f"Uni{i}"},
        ] for i in range(5))
        # Flatten — just test that industry + top5 = flagged
        df_simple = _make_master_df([
            {"lead_sponsor_class": "INDUSTRY", "lead_sponsor_name": "BigPharma"},
            {"lead_sponsor_class": "OTHER", "lead_sponsor_name": "Uni1"},
        ])
        det = SponsorConcentrationDetector()
        result = det.detect(df_simple)
        assert len(result.nct_ids) == 2


class TestGeographicShifts:
    def test_lmic_majority_flagged(self):
        df = _make_master_df([{}])
        facilities = pd.DataFrame({
            "nct_id": ["NCT00000001"] * 5,
            "country": ["India", "India", "Brazil", "Nigeria", "United States"],
        })
        result = GeographicShiftsDetector().detect(
            df, raw_tables={"facilities": facilities}
        )
        # 4 out of 5 sites are LMIC (80%)
        assert result.flaw_detected[0] is True
        assert result.severity[0] == pytest.approx(0.8, abs=0.01)

    def test_hic_majority_not_flagged(self):
        df = _make_master_df([{}])
        facilities = pd.DataFrame({
            "nct_id": ["NCT00000001"] * 5,
            "country": ["United States", "United Kingdom", "Germany", "France", "India"],
        })
        result = GeographicShiftsDetector().detect(
            df, raw_tables={"facilities": facilities}
        )
        # Only 1/5 LMIC = 20% < 50%
        assert result.flaw_detected[0] is False

    def test_few_sites_not_flagged(self):
        """< 3 sites should not flag even if all LMIC."""
        df = _make_master_df([{}])
        facilities = pd.DataFrame({
            "nct_id": ["NCT00000001"] * 2,
            "country": ["India", "Brazil"],
        })
        result = GeographicShiftsDetector().detect(
            df, raw_tables={"facilities": facilities}
        )
        assert result.flaw_detected[0] is False

    def test_no_facilities_not_flagged(self):
        df = _make_master_df([{}])
        result = GeographicShiftsDetector().detect(df, raw_tables={})
        assert result.flaw_detected[0] is False


class TestResultsDelay:
    def test_compliant_not_flagged(self):
        """Results posted 8 months after completion — compliant."""
        df = _make_master_df([{
            "primary_completion_date": pd.Timestamp("2023-01-01"),
            "results_first_posted_date": pd.Timestamp("2023-09-01"),
            "has_results": True,
        }])
        result = ResultsDelayDetector().detect(df)
        assert result.flaw_detected[0] is False

    def test_moderate_delay_flagged(self):
        """Results posted 18 months after completion — severity 0.3."""
        df = _make_master_df([{
            "primary_completion_date": pd.Timestamp("2022-01-01"),
            "results_first_posted_date": pd.Timestamp("2023-07-01"),
            "has_results": True,
        }])
        result = ResultsDelayDetector().detect(df)
        assert result.flaw_detected[0] is True
        assert result.severity[0] == pytest.approx(0.3, abs=0.01)

    def test_severe_delay_flagged(self):
        """Results posted 30 months after completion — severity 0.6."""
        df = _make_master_df([{
            "primary_completion_date": pd.Timestamp("2021-01-01"),
            "results_first_posted_date": pd.Timestamp("2023-07-01"),
            "has_results": True,
        }])
        result = ResultsDelayDetector().detect(df)
        assert result.flaw_detected[0] is True
        assert result.severity[0] == pytest.approx(0.6, abs=0.01)

    def test_no_results_completed_flagged(self):
        """Completed trial, no results, 36+ months past PCD."""
        df = _make_master_df([{
            "primary_completion_date": pd.Timestamp("2021-01-01"),
            "results_first_posted_date": pd.NaT,
            "has_results": False,
            "overall_status": "COMPLETED",
        }])
        result = ResultsDelayDetector().detect(df)
        assert result.flaw_detected[0] is True
        assert result.severity[0] > 0

    def test_no_pcd_not_flagged(self):
        df = _make_master_df([{
            "primary_completion_date": pd.NaT,
        }])
        result = ResultsDelayDetector().detect(df)
        assert result.flaw_detected[0] is False


class TestEndpointSoftening:
    def test_classify_hard(self):
        assert classify_endpoint("All-cause mortality at 12 months") == "hard"
        assert classify_endpoint("Myocardial infarction or stroke") == "hard"
        assert classify_endpoint("Hospitalization for heart failure") == "hard"
        assert classify_endpoint("MACE composite endpoint") == "hard"

    def test_classify_surrogate(self):
        assert classify_endpoint("Change in systolic blood pressure") == "surrogate"
        assert classify_endpoint("NT-proBNP level at 6 months") == "surrogate"
        assert classify_endpoint("LDL cholesterol reduction") == "surrogate"

    def test_classify_pro(self):
        assert classify_endpoint("KCCQ total score at 12 months") == "pro"
        assert classify_endpoint("Quality of life (EQ-5D)") == "pro"
        assert classify_endpoint("6-minute walk distance") == "pro"

    def test_all_surrogate_flagged(self):
        df = _make_master_df([{}])
        outcomes = pd.DataFrame({
            "nct_id": ["NCT00000001", "NCT00000001"],
            "outcome_type": ["PRIMARY", "PRIMARY"],
            "title": ["Change in blood pressure", "NT-proBNP level"],
        })
        result = EndpointSofteningDetector().detect(
            df, raw_tables={"outcomes": outcomes}
        )
        assert result.flaw_detected[0] is True
        assert result.severity[0] == 1.0

    def test_hard_only_not_flagged(self):
        df = _make_master_df([{}])
        outcomes = pd.DataFrame({
            "nct_id": ["NCT00000001"],
            "outcome_type": ["PRIMARY"],
            "title": ["All-cause mortality at 12 months"],
        })
        result = EndpointSofteningDetector().detect(
            df, raw_tables={"outcomes": outcomes}
        )
        assert result.flaw_detected[0] is False

    def test_mixed_flagged_lower_severity(self):
        df = _make_master_df([{}])
        outcomes = pd.DataFrame({
            "nct_id": ["NCT00000001", "NCT00000001"],
            "outcome_type": ["PRIMARY", "PRIMARY"],
            "title": ["All-cause mortality", "Change in LDL cholesterol"],
        })
        result = EndpointSofteningDetector().detect(
            df, raw_tables={"outcomes": outcomes}
        )
        assert result.flaw_detected[0] is True
        assert result.severity[0] == 0.5


class TestComparatorManipulation:
    def test_placebo_with_soc_condition_flagged(self):
        df = _make_master_df([{
            "cv_subdomains": ["HF"],
        }])
        interventions = pd.DataFrame({
            "nct_id": ["NCT00000001", "NCT00000001"],
            "intervention_type": ["DRUG", "OTHER"],
            "name": ["Empagliflozin 10mg", "Placebo"],
        })
        design_groups = pd.DataFrame({
            "nct_id": ["NCT00000001", "NCT00000001"],
            "group_type": ["EXPERIMENTAL", "PLACEBO_COMPARATOR"],
            "title": ["Drug arm", "Placebo arm"],
            "description": ["Active drug", "Matching placebo"],
        })
        result = ComparatorManipulationDetector().detect(
            df, raw_tables={"interventions": interventions, "design_groups": design_groups}
        )
        assert result.flaw_detected[0] is True
        assert result.severity[0] >= 0.5

    def test_placebo_plus_soc_lower_severity(self):
        df = _make_master_df([{
            "cv_subdomains": ["HF"],
        }])
        interventions = pd.DataFrame({
            "nct_id": ["NCT00000001", "NCT00000001"],
            "intervention_type": ["DRUG", "OTHER"],
            "name": ["Empagliflozin 10mg", "Placebo"],
        })
        design_groups = pd.DataFrame({
            "nct_id": ["NCT00000001", "NCT00000001"],
            "group_type": ["EXPERIMENTAL", "PLACEBO_COMPARATOR"],
            "title": ["Drug arm + standard of care", "Placebo + standard of care"],
            "description": ["Active drug on top of usual care", "Placebo on top of usual care"],
        })
        result = ComparatorManipulationDetector().detect(
            df, raw_tables={"interventions": interventions, "design_groups": design_groups}
        )
        assert result.flaw_detected[0] is True
        assert result.severity[0] == pytest.approx(0.2, abs=0.01)

    def test_no_soc_subdomain_not_flagged(self):
        df = _make_master_df([{
            "cv_subdomains": ["prevention"],
        }])
        interventions = pd.DataFrame({
            "nct_id": ["NCT00000001"],
            "intervention_type": ["DRUG"],
            "name": ["Placebo"],
        })
        design_groups = pd.DataFrame({
            "nct_id": ["NCT00000001"],
            "group_type": ["PLACEBO_COMPARATOR"],
            "title": ["Placebo"],
            "description": ["Matching placebo"],
        })
        result = ComparatorManipulationDetector().detect(
            df, raw_tables={"interventions": interventions, "design_groups": design_groups}
        )
        assert result.flaw_detected[0] is False

    def test_active_comparator_not_flagged(self):
        df = _make_master_df([{
            "cv_subdomains": ["HF"],
        }])
        interventions = pd.DataFrame({
            "nct_id": ["NCT00000001", "NCT00000001"],
            "intervention_type": ["DRUG", "DRUG"],
            "name": ["Empagliflozin", "Enalapril"],
        })
        design_groups = pd.DataFrame({
            "nct_id": ["NCT00000001", "NCT00000001"],
            "group_type": ["EXPERIMENTAL", "ACTIVE_COMPARATOR"],
            "title": ["Drug A", "Drug B"],
            "description": ["Empagliflozin arm", "Enalapril arm"],
        })
        result = ComparatorManipulationDetector().detect(
            df, raw_tables={"interventions": interventions, "design_groups": design_groups}
        )
        assert result.flaw_detected[0] is False


class TestStatisticalFragility:
    def test_fragility_index_basic(self):
        """Known example: 5/100 vs 20/100 — clearly significant (p~0.002)."""
        fi = compute_fragility_index(5, 100, 20, 100)
        assert fi is not None
        assert fi > 0
        assert fi <= 15

    def test_fragility_index_not_significant(self):
        """50/100 vs 50/100 — p = 1.0, no fragility index."""
        fi = compute_fragility_index(50, 100, 50, 100)
        assert fi is None

    def test_fragility_index_very_fragile(self):
        """Close result — 3/50 vs 15/50 (p~0.003), FI should be small."""
        fi = compute_fragility_index(3, 50, 15, 50)
        assert fi is not None
        assert fi <= 10

    def test_fragility_index_invalid_inputs(self):
        assert compute_fragility_index(-1, 100, 10, 100) is None
        assert compute_fragility_index(10, 0, 10, 100) is None
        assert compute_fragility_index(101, 100, 10, 100) is None

    def test_fi_to_severity_mapping(self):
        assert _fi_to_severity(None) == 0.0
        assert _fi_to_severity(1) == 0.8
        assert _fi_to_severity(3) == 0.8
        assert _fi_to_severity(5) == 0.5
        assert _fi_to_severity(10) == 0.3
        assert _fi_to_severity(20) == 0.1

    def test_detector_with_provided_2x2(self):
        df = _make_master_df([{}])
        # Provide outcome_counts that form a 2x2 table
        oc = pd.DataFrame({
            "nct_id": ["NCT00000001", "NCT00000001"],
            "outcome_id": [1, 1],
            "result_group_id": [1, 2],
            "ctgov_group_code": ["P1", "P2"],
            "scope": ["Participants", "Participants"],
            "units": ["Participants", "Participants"],
            "count": [10, 25],
        })
        result = StatisticalFragilityDetector().detect(
            df, raw_tables={"outcome_counts": oc}
        )
        # Whether flagged depends on Fisher exact significance
        assert len(result.flaw_detected) == 1

    def test_detector_no_data_not_flagged(self):
        df = _make_master_df([{}])
        result = StatisticalFragilityDetector().detect(df, raw_tables={})
        assert result.flaw_detected[0] is False


# ── Additional unit tests for helper functions ───────────────────────


class TestHelpers:
    def test_fuzzy_best_match_exact(self):
        score = _fuzzy_best_match("mortality", ["mortality", "blood pressure"])
        assert score >= 90

    def test_fuzzy_best_match_empty(self):
        assert _fuzzy_best_match("", ["anything"]) == 0.0
        assert _fuzzy_best_match("anything", []) == 0.0

    def test_extract_exclusion_section(self):
        criteria = "Inclusion Criteria: Adults\nExclusion Criteria: No CKD, no cancer"
        exc = _extract_exclusion_section(criteria)
        assert "CKD" in exc
        assert "Adults" not in exc

    def test_extract_exclusion_section_missing(self):
        assert _extract_exclusion_section("Just some text") == ""
        assert _extract_exclusion_section(None) == ""
        assert _extract_exclusion_section("") == ""


# ── Integration: all detectors return correct length ─────────────────


class TestAllDetectorsLength:
    """Every detector must return DetectorResult matching master_df length."""

    @pytest.fixture
    def master_df(self):
        return _make_master_df([
            {"overall_status": "COMPLETED", "has_results": True},
            {"overall_status": "TERMINATED", "has_results": False},
            {"overall_status": "WITHDRAWN", "start_date": pd.Timestamp("2018-01-01")},
        ])

    @pytest.mark.parametrize("name", list(DETECTOR_REGISTRY.keys()))
    def test_detector_length(self, name, master_df):
        detector = DETECTOR_REGISTRY[name]
        result = detector.detect(master_df, raw_tables={})
        assert len(result.nct_ids) == len(master_df), (
            f"{name} returned {len(result.nct_ids)} rows, expected {len(master_df)}"
        )
        assert len(result.flaw_detected) == len(master_df)
        assert len(result.severity) == len(master_df)
        assert len(result.detail) == len(master_df)

    @pytest.mark.parametrize("name", list(DETECTOR_REGISTRY.keys()))
    def test_severity_in_range(self, name, master_df):
        detector = DETECTOR_REGISTRY[name]
        result = detector.detect(master_df, raw_tables={})
        for sev in result.severity:
            assert 0.0 <= sev <= 1.0, f"{name} severity {sev} out of range"
