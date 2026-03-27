"""Tests for composite flaw scoring."""
import numpy as np
import pandas as pd
import pytest

from pipeline.composite import compute_composite_scores, DETECTOR_NAMES


def _make_results_df(rows: list[dict]) -> pd.DataFrame:
    """Build a minimal results DataFrame with detector columns."""
    defaults = {"nct_id": "NCT00000001", "start_year": 2020}
    # Add detector columns defaulting to no flaws
    for name in DETECTOR_NAMES:
        defaults[f"{name}_detected"] = False
        defaults[f"{name}_severity"] = 0.0
        defaults[f"{name}_detail"] = ""

    records = []
    for i, override in enumerate(rows):
        row = defaults.copy()
        row["nct_id"] = f"NCT{i+1:08d}"
        row.update(override)
        records.append(row)
    return pd.DataFrame(records)


class TestCompositeScoreCalculation:
    """test_composite_score_calculation: 2 flaws -> mean severity, flaw_count=2."""

    def test_two_flaws_mean_severity(self):
        df = _make_results_df([{
            "ghost_protocols_detected": True,
            "ghost_protocols_severity": 0.8,
            "results_delay_detected": True,
            "results_delay_severity": 0.4,
        }])
        result = compute_composite_scores(df)
        assert result["flaw_count"].iloc[0] == 2
        assert abs(result["composite_severity"].iloc[0] - 0.6) < 1e-9
        assert "ghost_protocols" in result["flaw_categories"].iloc[0]
        assert "results_delay" in result["flaw_categories"].iloc[0]

    def test_single_flaw(self):
        df = _make_results_df([{
            "endpoint_softening_detected": True,
            "endpoint_softening_severity": 0.5,
        }])
        result = compute_composite_scores(df)
        assert result["flaw_count"].iloc[0] == 1
        assert abs(result["composite_severity"].iloc[0] - 0.5) < 1e-9
        assert result["flaw_categories"].iloc[0] == "endpoint_softening"

    def test_all_ten_flaws(self):
        overrides = {}
        for name in DETECTOR_NAMES:
            overrides[f"{name}_detected"] = True
            overrides[f"{name}_severity"] = 0.5
        df = _make_results_df([overrides])
        result = compute_composite_scores(df)
        assert result["flaw_count"].iloc[0] == 10
        assert abs(result["composite_severity"].iloc[0] - 0.5) < 1e-9

    def test_varying_severities(self):
        df = _make_results_df([{
            "ghost_protocols_detected": True,
            "ghost_protocols_severity": 1.0,
            "outcome_switching_detected": True,
            "outcome_switching_severity": 0.0,
            "sample_size_decay_detected": True,
            "sample_size_decay_severity": 0.5,
        }])
        result = compute_composite_scores(df)
        assert result["flaw_count"].iloc[0] == 3
        expected = (1.0 + 0.0 + 0.5) / 3
        assert abs(result["composite_severity"].iloc[0] - expected) < 1e-9


class TestZeroFlaws:
    """test_zero_flaws: no flaws -> flaw_count=0, composite_severity=0."""

    def test_no_flaws_detected(self):
        df = _make_results_df([{}])  # all defaults = no flaws
        result = compute_composite_scores(df)
        assert result["flaw_count"].iloc[0] == 0
        assert result["composite_severity"].iloc[0] == 0.0
        assert result["flaw_categories"].iloc[0] == ""

    def test_multiple_rows_mixed(self):
        df = _make_results_df([
            {},  # no flaws
            {"ghost_protocols_detected": True, "ghost_protocols_severity": 0.7},
            {},  # no flaws
        ])
        result = compute_composite_scores(df)
        assert result["flaw_count"].iloc[0] == 0
        assert result["flaw_count"].iloc[1] == 1
        assert result["flaw_count"].iloc[2] == 0
        assert result["composite_severity"].iloc[0] == 0.0
        assert abs(result["composite_severity"].iloc[1] - 0.7) < 1e-9

    def test_severity_ignored_when_not_detected(self):
        """Even if severity > 0, if not detected, it should not count."""
        df = _make_results_df([{
            "ghost_protocols_detected": False,
            "ghost_protocols_severity": 0.9,
        }])
        result = compute_composite_scores(df)
        assert result["flaw_count"].iloc[0] == 0
        assert result["composite_severity"].iloc[0] == 0.0
        assert result["flaw_categories"].iloc[0] == ""
