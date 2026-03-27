"""Tests for export module (dashboard JSON + manuscript CSV)."""
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pipeline.composite import DETECTOR_NAMES
from pipeline.export import export_dashboard_json, export_manuscript_csv, _NumpyEncoder


def _make_export_df(n: int = 10) -> pd.DataFrame:
    """Create a minimal results DataFrame for export tests."""
    rows = []
    for i in range(n):
        row = {
            "nct_id": f"NCT{i:08d}",
            "start_year": 2015 + (i % 10),
            "flaw_count": i % 3,
            "composite_severity": (i % 3) * 0.3,
            "flaw_categories": "ghost_protocols" if i % 3 > 0 else "",
        }
        for name in DETECTOR_NAMES:
            row[f"{name}_detected"] = (i % 5 == 0)
            row[f"{name}_severity"] = 0.6 if (i % 5 == 0) else 0.0
            row[f"{name}_detail"] = "flagged" if (i % 5 == 0) else ""
        rows.append(row)
    return pd.DataFrame(rows)


def _make_trends_df() -> pd.DataFrame:
    """Create a minimal trends DataFrame."""
    return pd.DataFrame({
        "year": [2015, 2016, 2017],
        "n_trials": [100, 120, 130],
        "composite_mean": [0.3, 0.35, 0.28],
        "mean_flaw_count": [1.5, 1.8, 1.2],
    })


class TestDashboardJsonStructure:
    """test_dashboard_json_structure: JSON has trials, trends, meta keys."""

    def test_has_required_keys(self, tmp_path):
        results_df = _make_export_df(20)
        trends_df = _make_trends_df()
        out = tmp_path / "dashboard.json"

        export_dashboard_json(results_df, trends_df, out)
        assert out.exists()

        with open(out, "r", encoding="utf-8") as f:
            data = json.load(f)

        assert "meta" in data
        assert "trials" in data
        assert "trends" in data
        assert data["meta"]["n_trials"] == 20
        assert isinstance(data["meta"]["year_range"], list)
        assert len(data["trials"]) == 20
        assert len(data["trends"]) == 3

    def test_binned_trends_included(self, tmp_path):
        results_df = _make_export_df(10)
        trends_df = _make_trends_df()
        binned_df = pd.DataFrame({
            "year_bin": ["2015-2017", "2018-2020"],
            "n_trials": [50, 60],
        })
        out = tmp_path / "dashboard.json"

        export_dashboard_json(results_df, trends_df, out, binned_df=binned_df)

        with open(out, "r", encoding="utf-8") as f:
            data = json.load(f)

        assert "binned_trends" in data
        assert len(data["binned_trends"]) == 2

    def test_no_binned_trends_when_none(self, tmp_path):
        results_df = _make_export_df(5)
        trends_df = _make_trends_df()
        out = tmp_path / "dashboard.json"

        export_dashboard_json(results_df, trends_df, out)

        with open(out, "r", encoding="utf-8") as f:
            data = json.load(f)

        assert "binned_trends" not in data

    def test_numpy_types_serialized(self, tmp_path):
        """Ensure numpy int64, float64, bool_ serialize correctly."""
        df = pd.DataFrame({
            "nct_id": ["NCT001"],
            "start_year": [np.int64(2020)],
            "value": [np.float64(0.5)],
            "flag": [np.bool_(True)],
        })
        trends = _make_trends_df()
        out = tmp_path / "dashboard.json"

        export_dashboard_json(df, trends, out)

        with open(out, "r", encoding="utf-8") as f:
            data = json.load(f)

        trial = data["trials"][0]
        assert trial["start_year"] == 2020
        assert trial["value"] == 0.5
        assert trial["flag"] is True

    def test_nan_becomes_null(self, tmp_path):
        df = pd.DataFrame({
            "nct_id": ["NCT001"],
            "start_year": [2020],
            "value": [float("nan")],
        })
        trends = _make_trends_df()
        out = tmp_path / "dashboard.json"

        export_dashboard_json(df, trends, out)

        with open(out, "r", encoding="utf-8") as f:
            data = json.load(f)

        assert data["trials"][0]["value"] is None

    def test_creates_parent_dirs(self, tmp_path):
        results_df = _make_export_df(3)
        trends_df = _make_trends_df()
        out = tmp_path / "nested" / "deep" / "dashboard.json"

        export_dashboard_json(results_df, trends_df, out)
        assert out.exists()


class TestManuscriptCsv:
    """test_manuscript_csv: creates trial_level_results.csv and detector_summary.csv."""

    def test_creates_trial_level_csv(self, tmp_path):
        results_df = _make_export_df(15)
        paths = export_manuscript_csv(results_df, tmp_path)

        assert "trial_level_results.csv" in paths
        csv_path = paths["trial_level_results.csv"]
        assert csv_path.exists()

        loaded = pd.read_csv(csv_path)
        assert len(loaded) == 15
        assert "nct_id" in loaded.columns

    def test_creates_detector_summary_csv(self, tmp_path):
        results_df = _make_export_df(20)
        paths = export_manuscript_csv(results_df, tmp_path)

        assert "detector_summary.csv" in paths
        csv_path = paths["detector_summary.csv"]
        assert csv_path.exists()

        loaded = pd.read_csv(csv_path)
        assert "detector" in loaded.columns
        assert "n_detected" in loaded.columns
        assert "detection_rate" in loaded.columns
        assert "mean_severity_when_detected" in loaded.columns
        assert len(loaded) == len(DETECTOR_NAMES)

    def test_creates_output_dir_if_missing(self, tmp_path):
        results_df = _make_export_df(5)
        out_dir = tmp_path / "new_dir" / "tables"
        paths = export_manuscript_csv(results_df, out_dir)
        assert out_dir.exists()
        for p in paths.values():
            assert p.exists()


class TestNumpyEncoder:
    """Unit tests for the custom JSON encoder."""

    def test_numpy_int(self):
        assert json.dumps({"v": np.int64(42)}, cls=_NumpyEncoder) == '{"v": 42}'

    def test_numpy_float(self):
        result = json.loads(json.dumps({"v": np.float64(3.14)}, cls=_NumpyEncoder))
        assert abs(result["v"] - 3.14) < 1e-9

    def test_numpy_nan(self):
        result = json.loads(json.dumps({"v": np.float64("nan")}, cls=_NumpyEncoder))
        assert result["v"] is None

    def test_numpy_bool(self):
        result = json.loads(json.dumps({"v": np.bool_(True)}, cls=_NumpyEncoder))
        assert result["v"] is True

    def test_numpy_array(self):
        result = json.loads(json.dumps({"v": np.array([1, 2, 3])}, cls=_NumpyEncoder))
        assert result["v"] == [1, 2, 3]
