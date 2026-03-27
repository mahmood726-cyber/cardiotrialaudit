"""Main pipeline orchestrator — run the full CardioTrialAudit pipeline."""
import logging
import sys
import time
from pathlib import Path

import pandas as pd

from pipeline.cardio_filter import filter_cardiology_trials
from pipeline.master_table import build_master_table
from pipeline.detectors.runner import run_all_detectors
from pipeline.composite import compute_composite_scores
from pipeline.trends import compute_yearly_trends, compute_binned_trends
from pipeline.export import export_dashboard_json, export_manuscript_csv

logger = logging.getLogger(__name__)

# Output directories (relative to project root)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_RESULTS_DIR = PROJECT_ROOT / "data" / "results"
MANUSCRIPT_TABLES_DIR = PROJECT_ROOT / "manuscript" / "tables"


def main(nrows: int | None = None) -> dict:
    """Run the full CardioTrialAudit pipeline.

    Steps:
      1. Filter cardiology trials from AACT
      2. Build one-row-per-trial master table
      3. Run all 10 flaw detectors
      4. Compute composite scores
      5. Compute yearly + binned trends
      6. Export dashboard JSON + manuscript CSVs

    Parameters
    ----------
    nrows : int | None
        Row limit for studies table (for testing/smoke tests).

    Returns
    -------
    dict
        Keys: cv_studies, master, results, trends, binned_trends, output_files
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        stream=sys.stderr,
    )

    t0 = time.time()

    # Step 1: Filter cardiology trials
    logger.info("Step 1: Filtering cardiology trials (nrows=%s)...", nrows)
    cv_studies = filter_cardiology_trials(nrows_studies=nrows)
    logger.info("  -> %d CV trials found", len(cv_studies))

    # Step 2: Build master table
    logger.info("Step 2: Building master table...")
    master = build_master_table(cv_studies=cv_studies)
    logger.info("  -> %d rows in master table", len(master))

    # Step 3: Run all detectors
    logger.info("Step 3: Running all 10 flaw detectors...")
    results = run_all_detectors(master)
    det_cols = [c for c in results.columns if c.endswith("_detected")]
    logger.info("  -> %d detector columns added", len(det_cols))

    # Step 4: Composite scores
    logger.info("Step 4: Computing composite scores...")
    results = compute_composite_scores(results)
    mean_flaws = results["flaw_count"].mean()
    logger.info("  -> Mean flaw count: %.2f", mean_flaws)

    # Step 5: Trends
    logger.info("Step 5: Computing temporal trends...")
    trends = compute_yearly_trends(results)
    binned = compute_binned_trends(results, bin_size=3)
    logger.info("  -> %d yearly rows, %d binned rows", len(trends), len(binned))

    # Step 6: Export
    logger.info("Step 6: Exporting results...")
    DATA_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    MANUSCRIPT_TABLES_DIR.mkdir(parents=True, exist_ok=True)

    json_path = export_dashboard_json(
        results_df=results,
        trends_df=trends,
        output_path=DATA_RESULTS_DIR / "dashboard.json",
        binned_df=binned,
    )

    csv_paths = export_manuscript_csv(
        results_df=results,
        output_dir=MANUSCRIPT_TABLES_DIR,
    )

    elapsed = time.time() - t0
    logger.info("Pipeline complete in %.1fs", elapsed)
    logger.info("  Dashboard JSON: %s", json_path)
    for name, path in csv_paths.items():
        logger.info("  %s: %s", name, path)

    return {
        "cv_studies": cv_studies,
        "master": master,
        "results": results,
        "trends": trends,
        "binned_trends": binned,
        "output_files": {
            "dashboard_json": str(json_path),
            **{k: str(v) for k, v in csv_paths.items()},
        },
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="CardioTrialAudit pipeline")
    parser.add_argument(
        "--nrows", type=int, default=None,
        help="Row limit for studies table (for testing)",
    )
    args = parser.parse_args()
    main(nrows=args.nrows)
