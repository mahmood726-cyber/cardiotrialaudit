"""Export pipeline results for dashboard (JSON) and manuscript (CSV)."""
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


class _NumpyEncoder(json.JSONEncoder):
    """Handle numpy/pandas types in JSON serialization.

    Note: numpy float64 subclasses Python float, so json.dumps handles it
    natively — but NaN/Inf serialize as invalid JSON tokens. We override
    iterencode to intercept those via _sanitize_value.
    """

    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            v = float(obj)
            if math.isnan(v) or math.isinf(v):
                return None
            return v
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, pd.Timestamp):
            if pd.isna(obj):
                return None
            return obj.isoformat()
        if isinstance(obj, (pd.NaT.__class__,)):
            return None
        return super().default(obj)

    def encode(self, o):
        return super().encode(self._sanitize(o))

    def _sanitize(self, o):
        """Recursively convert numpy types and NaN/Inf to JSON-safe values."""
        if isinstance(o, dict):
            return {k: self._sanitize(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [self._sanitize(v) for v in o]
        if isinstance(o, float):
            if math.isnan(o) or math.isinf(o):
                return None
            return o
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            v = float(o)
            if math.isnan(v) or math.isinf(v):
                return None
            return v
        if isinstance(o, (np.bool_,)):
            return bool(o)
        if isinstance(o, np.ndarray):
            return [self._sanitize(x) for x in o.tolist()]
        if isinstance(o, pd.Timestamp):
            if pd.isna(o):
                return None
            return o.isoformat()
        return o


def _sanitize_for_json(df: pd.DataFrame) -> list[dict]:
    """Convert DataFrame to list of dicts with NaN -> None and numpy -> native."""
    records = df.to_dict(orient="records")
    sanitized = []
    for rec in records:
        clean = {}
        for k, v in rec.items():
            if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
                clean[k] = None
            elif isinstance(v, (np.integer,)):
                clean[k] = int(v)
            elif isinstance(v, (np.floating,)):
                if np.isnan(v) or np.isinf(v):
                    clean[k] = None
                else:
                    clean[k] = float(v)
            elif isinstance(v, (np.bool_,)):
                clean[k] = bool(v)
            elif isinstance(v, pd.Timestamp):
                clean[k] = v.isoformat() if not pd.isna(v) else None
            elif pd.isna(v) if not isinstance(v, (list, dict, str, bool)) else False:
                clean[k] = None
            else:
                clean[k] = v
        sanitized.append(clean)
    return sanitized


def export_dashboard_json(
    results_df: pd.DataFrame,
    trends_df: pd.DataFrame,
    output_path: str | Path,
    binned_df: pd.DataFrame | None = None,
) -> Path:
    """Write dashboard JSON with trial-level results and trend data.

    JSON structure:
    {
      "meta": { "generated_at", "n_trials", "year_range" },
      "trials": [ ... ],
      "trends": [ ... ],
      "binned_trends": [ ... ]  (if binned_df provided)
    }

    Parameters
    ----------
    results_df : pd.DataFrame
        Trial-level results with composite scores.
    trends_df : pd.DataFrame
        Yearly trend aggregations.
    output_path : str or Path
        Output file path for JSON.
    binned_df : pd.DataFrame | None
        Optional binned trend aggregations.

    Returns
    -------
    Path
        The output file path.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Build year range from trends or results
    if "year" in trends_df.columns and len(trends_df) > 0:
        year_min = int(trends_df["year"].min())
        year_max = int(trends_df["year"].max())
    elif "start_year" in results_df.columns:
        year_min = int(results_df["start_year"].min())
        year_max = int(results_df["start_year"].max())
    else:
        year_min, year_max = None, None

    payload = {
        "meta": {
            "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "n_trials": len(results_df),
            "year_range": [year_min, year_max],
        },
        "trials": _sanitize_for_json(results_df),
        "trends": _sanitize_for_json(trends_df),
    }

    if binned_df is not None and len(binned_df) > 0:
        payload["binned_trends"] = _sanitize_for_json(binned_df)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, cls=_NumpyEncoder, indent=2, ensure_ascii=False)

    return output_path


def export_manuscript_csv(
    results_df: pd.DataFrame,
    output_dir: str | Path,
) -> dict[str, Path]:
    """Write CSVs for manuscript tables.

    Produces:
      - trial_level_results.csv: full trial-level data
      - detector_summary.csv: per-detector summary stats

    Parameters
    ----------
    results_df : pd.DataFrame
        Trial-level results with composite scores.
    output_dir : str or Path
        Directory to write CSV files.

    Returns
    -------
    dict[str, Path]
        Mapping of filename to output path.
    """
    from pipeline.composite import DETECTOR_NAMES

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1) Trial-level results
    trial_path = output_dir / "trial_level_results.csv"
    results_df.to_csv(trial_path, index=False, encoding="utf-8")

    # 2) Detector summary
    present = [
        name for name in DETECTOR_NAMES
        if f"{name}_detected" in results_df.columns
    ]

    summary_rows = []
    for name in present:
        det_col = f"{name}_detected"
        sev_col = f"{name}_severity"
        n_detected = int(results_df[det_col].sum())
        n_total = len(results_df)
        rate = n_detected / n_total if n_total > 0 else 0.0
        mean_sev = (
            float(results_df.loc[results_df[det_col], sev_col].mean())
            if n_detected > 0
            else 0.0
        )
        summary_rows.append({
            "detector": name,
            "n_detected": n_detected,
            "n_total": n_total,
            "detection_rate": round(rate, 4),
            "mean_severity_when_detected": round(mean_sev, 4),
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_path = output_dir / "detector_summary.csv"
    summary_df.to_csv(summary_path, index=False, encoding="utf-8")

    return {
        "trial_level_results.csv": trial_path,
        "detector_summary.csv": summary_path,
    }
