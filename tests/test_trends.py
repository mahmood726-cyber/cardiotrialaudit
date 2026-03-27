"""Tests for temporal trend analysis."""
import pandas as pd
import pytest

from pipeline.composite import DETECTOR_NAMES
from pipeline.trends import compute_yearly_trends, compute_binned_trends


def _make_trend_df(n_years: int = 5, base_year: int = 2015, trials_per_year: int = 10) -> pd.DataFrame:
    """Build a fake results DataFrame spanning n_years for trend tests."""
    rows = []
    for yr in range(base_year, base_year + n_years):
        for i in range(trials_per_year):
            row = {
                "nct_id": f"NCT{yr}{i:04d}",
                "start_year": yr,
                "flaw_count": i % 3,
                "composite_severity": (i % 3) * 0.3,
            }
            for name in DETECTOR_NAMES:
                row[f"{name}_detected"] = (i % (DETECTOR_NAMES.index(name) + 2) == 0)
                row[f"{name}_severity"] = 0.5 if row[f"{name}_detected"] else 0.0
            rows.append(row)
    return pd.DataFrame(rows)


class TestYearlyTrendsShape:
    """test_yearly_trends_shape: has year, n_trials, *_rate columns."""

    def test_has_required_columns(self):
        df = _make_trend_df(n_years=5)
        trends = compute_yearly_trends(df)
        assert "year" in trends.columns
        assert "n_trials" in trends.columns
        assert "composite_mean" in trends.columns
        assert "mean_flaw_count" in trends.columns
        for name in DETECTOR_NAMES:
            assert f"{name}_rate" in trends.columns
            assert f"{name}_mean_severity" in trends.columns

    def test_correct_number_of_years(self):
        df = _make_trend_df(n_years=5, base_year=2015)
        trends = compute_yearly_trends(df)
        assert len(trends) == 5
        assert trends["year"].min() == 2015
        assert trends["year"].max() == 2019

    def test_n_trials_correct(self):
        df = _make_trend_df(n_years=3, trials_per_year=20)
        trends = compute_yearly_trends(df)
        assert (trends["n_trials"] == 20).all()

    def test_rates_between_0_and_1(self):
        df = _make_trend_df(n_years=3)
        trends = compute_yearly_trends(df)
        for name in DETECTOR_NAMES:
            col = f"{name}_rate"
            assert (trends[col] >= 0.0).all()
            assert (trends[col] <= 1.0).all()

    def test_single_year(self):
        df = _make_trend_df(n_years=1, base_year=2020, trials_per_year=5)
        trends = compute_yearly_trends(df)
        assert len(trends) == 1
        assert trends["year"].iloc[0] == 2020


class TestBinnedTrendsShape:
    """test_binned_trends_shape: has year_bin column, bins for 2005-2026."""

    def test_has_year_bin_column(self):
        df = _make_trend_df(n_years=6, base_year=2005)
        binned = compute_binned_trends(df, bin_size=3)
        assert "year_bin" in binned.columns
        assert "n_trials" in binned.columns

    def test_bin_count_for_range(self):
        # 2005-2026 = 22 years, bin_size=3 -> ceil(22/3) = 8 bins
        df = _make_trend_df(n_years=22, base_year=2005, trials_per_year=5)
        binned = compute_binned_trends(df, bin_size=3)
        assert len(binned) == 8  # 2005-2007, 2008-2010, ..., 2023-2025, 2026-2026
        assert binned["year_bin"].iloc[0] == "2005-2007"

    def test_bin_count_exact_division(self):
        # 6 years / 3 = 2 bins exactly
        df = _make_trend_df(n_years=6, base_year=2010, trials_per_year=5)
        binned = compute_binned_trends(df, bin_size=3)
        assert len(binned) == 2
        assert binned["year_bin"].iloc[0] == "2010-2012"
        assert binned["year_bin"].iloc[1] == "2013-2015"

    def test_bin_size_5(self):
        df = _make_trend_df(n_years=10, base_year=2010, trials_per_year=3)
        binned = compute_binned_trends(df, bin_size=5)
        assert len(binned) == 2
        assert binned["year_bin"].iloc[0] == "2010-2014"
        assert binned["year_bin"].iloc[1] == "2015-2019"

    def test_has_detector_rate_columns(self):
        df = _make_trend_df(n_years=6, base_year=2005)
        binned = compute_binned_trends(df, bin_size=3)
        for name in DETECTOR_NAMES:
            assert f"{name}_rate" in binned.columns

    def test_empty_df(self):
        df = pd.DataFrame(columns=["start_year"])
        binned = compute_binned_trends(df, bin_size=3)
        assert len(binned) == 0
