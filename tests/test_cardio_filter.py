"""Tests for cardiology trial filtering and sub-domain tagging."""
import pandas as pd
import pytest

from tests.conftest import skip_if_no_aact
from pipeline.cardio_filter import (
    CV_CONDITION_PATTERNS,
    CV_INTERVENTION_PATTERNS,
    CV_DEVICE_PATTERNS,
    SUBDOMAIN_RULES,
    is_cv_condition,
    is_ckd_only_condition,
    is_cv_intervention,
    tag_subdomain,
    filter_cardiology_trials,
)


class TestConditionMatching:
    def test_heart_failure_matches(self):
        assert is_cv_condition("Heart Failure")
        assert is_cv_condition("Congestive Heart Failure")
        assert is_cv_condition("heart failure with preserved ejection fraction")

    def test_cad_matches(self):
        assert is_cv_condition("Coronary Artery Disease")
        assert is_cv_condition("Acute Coronary Syndrome")
        assert is_cv_condition("Myocardial Infarction")

    def test_non_cv_does_not_match(self):
        assert not is_cv_condition("Breast Cancer")
        assert not is_cv_condition("Asthma")
        assert not is_cv_condition("Depression")

    def test_edge_cases(self):
        assert is_cv_condition("Hypertension, Pulmonary")
        assert is_cv_condition("Atrial Fibrillation and Flutter")
        assert is_cv_condition("Peripheral Arterial Disease")


class TestInterventionMatching:
    def test_drug_matches(self):
        assert is_cv_intervention("Empagliflozin", "DRUG")
        assert is_cv_intervention("Ticagrelor 90mg", "DRUG")

    def test_device_matches(self):
        assert is_cv_intervention("Percutaneous Coronary Intervention", "DEVICE")
        assert is_cv_intervention("Implantable Cardioverter Defibrillator", "DEVICE")

    def test_non_cv_does_not_match(self):
        assert not is_cv_intervention("Ibuprofen", "DRUG")
        assert not is_cv_intervention("Cognitive Behavioral Therapy", "BEHAVIORAL")


class TestSubdomainTagging:
    def test_hf_tagged(self):
        tags = tag_subdomain(conditions=["Heart Failure"], interventions=["Sacubitril/Valsartan"])
        assert "HF" in tags

    def test_cad_tagged(self):
        tags = tag_subdomain(conditions=["Coronary Artery Disease"], interventions=["Ticagrelor"])
        assert "CAD" in tags

    def test_multiple_tags(self):
        tags = tag_subdomain(conditions=["Heart Failure", "Atrial Fibrillation"], interventions=["Apixaban"])
        assert "HF" in tags
        assert "arrhythmia" in tags


class TestCKDOnlyCondition:
    """P0-1: CKD/nephropathy conditions without core CV terms are CKD-only."""

    def test_ckd_alone_is_ckd_only(self):
        assert is_ckd_only_condition("Chronic Kidney Disease")
        assert is_ckd_only_condition("Diabetic Nephropathy")
        assert is_ckd_only_condition("Diabetic Kidney Disease")

    def test_cardiorenal_is_ckd_only(self):
        # "cardiorenal" matches CKD pattern but not core CV pattern
        assert is_ckd_only_condition("Cardiorenal Syndrome")

    def test_hf_with_ckd_is_not_ckd_only(self):
        # Has both CKD and core CV — not CKD-only
        assert not is_ckd_only_condition("Heart Failure with Chronic Kidney Disease")

    def test_core_cv_is_not_ckd_only(self):
        assert not is_ckd_only_condition("Heart Failure")
        assert not is_ckd_only_condition("Coronary Artery Disease")
        assert not is_ckd_only_condition("Atrial Fibrillation")

    def test_non_cv_is_not_ckd_only(self):
        assert not is_ckd_only_condition("Breast Cancer")
        assert not is_ckd_only_condition("Asthma")


class TestFilterPipeline:
    @skip_if_no_aact
    def test_filter_on_small_sample(self):
        """Run filter on first 1000 studies — should find at least some CV trials."""
        result = filter_cardiology_trials(nrows_studies=1000)
        assert isinstance(result, pd.DataFrame)
        assert "nct_id" in result.columns
        assert "cv_subdomains" in result.columns
        assert "cv_match_source" in result.columns
