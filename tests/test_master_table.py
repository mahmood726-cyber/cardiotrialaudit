"""Tests for master table construction."""
import pandas as pd
import pytest
from pipeline.master_table import build_master_table


def test_master_table_has_required_columns():
    """Master table must have all columns needed by detectors."""
    mt = build_master_table(nrows_studies=200)
    required = [
        "nct_id", "overall_status", "phase", "enrollment", "enrollment_type",
        "start_date", "start_year", "primary_completion_date",
        "results_first_posted_date", "source", "source_class",
        "brief_title", "why_stopped", "fdaaa801_violation",
        "cv_match_source", "cv_subdomains",
        "allocation", "masking", "primary_purpose", "intervention_model",
        "gender", "minimum_age", "maximum_age", "criteria",
        "lead_sponsor_name", "lead_sponsor_class",
        "has_results", "is_interventional", "is_randomized",
        "results_delay_days",
    ]
    for col in required:
        assert col in mt.columns, f"Missing column: {col}"


def test_master_table_one_row_per_trial():
    mt = build_master_table(nrows_studies=500)
    assert mt["nct_id"].is_unique


def test_master_table_date_types():
    mt = build_master_table(nrows_studies=100)
    assert pd.api.types.is_datetime64_any_dtype(mt["start_date"])
