"""Tests for AACT ZIP ingestion."""
from pathlib import Path

import pandas as pd
import pytest

from pipeline.ingest import load_aact_table, AACT_ZIP_PATH

# Use test fixture ZIP for all tests
FIXTURE_ZIP = Path(__file__).parent / "fixtures" / "test_aact.zip"


def test_load_studies_returns_dataframe():
    """Loading 'studies' from AACT ZIP returns a DataFrame with expected columns."""
    df = load_aact_table("studies", nrows=10, zip_path=FIXTURE_ZIP)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 10
    assert "nct_id" in df.columns
    assert "overall_status" in df.columns
    assert "enrollment" in df.columns
    assert "phase" in df.columns


def test_load_conditions_returns_dataframe():
    df = load_aact_table("conditions", nrows=10, zip_path=FIXTURE_ZIP)
    assert isinstance(df, pd.DataFrame)
    assert "nct_id" in df.columns
    assert "name" in df.columns


def test_load_nonexistent_table_raises():
    with pytest.raises(KeyError):
        load_aact_table("nonexistent_table_xyz", zip_path=FIXTURE_ZIP)


def test_load_studies_date_parsing():
    """Date columns should be parsed as datetime."""
    df = load_aact_table("studies", nrows=100, zip_path=FIXTURE_ZIP)
    assert pd.api.types.is_datetime64_any_dtype(df["start_date"]) or df["start_date"].isna().all()
