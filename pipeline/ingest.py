"""Load AACT pipe-delimited tables directly from ZIP into pandas DataFrames."""
import zipfile
from io import TextIOWrapper
from pathlib import Path

import pandas as pd

# Candidate AACT ZIP paths, searched in order
_AACT_ZIP_CANDIDATES = [
    Path(r"C:\Users\user\Pairwise70\hfpef_registry_calibration\data\aact\20260219_export_ctgov.zip"),
    Path(r"C:\Users\user\projects\transportability-meta-frontier\data\aact_downloads\20260327_export_ctgov.zip"),
]


def _find_aact_zip() -> Path:
    """Return the first existing AACT ZIP path from candidates."""
    for p in _AACT_ZIP_CANDIDATES:
        if p.exists():
            return p
    raise FileNotFoundError(
        f"AACT ZIP not found at any candidate location:\n"
        + "\n".join(f"  - {p}" for p in _AACT_ZIP_CANDIDATES)
    )


# Resolve at import time; tests override via zip_path parameter
try:
    AACT_ZIP_PATH = _find_aact_zip()
except FileNotFoundError:
    # Allow import even if no AACT ZIP exists (tests provide their own)
    AACT_ZIP_PATH = _AACT_ZIP_CANDIDATES[0]

# Date columns to parse per table
_DATE_COLUMNS = {
    "studies": [
        "study_first_submitted_date",
        "results_first_submitted_date",
        "study_first_posted_date",
        "results_first_posted_date",
        "last_update_posted_date",
        "start_date",
        "verification_date",
        "completion_date",
        "primary_completion_date",
    ],
}


def load_aact_table(
    table_name: str,
    nrows: int | None = None,
    usecols: list[str] | None = None,
    zip_path: Path = AACT_ZIP_PATH,
) -> pd.DataFrame:
    """Load a single AACT table from the ZIP archive.

    Parameters
    ----------
    table_name : str
        Table name without extension (e.g., 'studies', 'conditions').
    nrows : int | None
        Optional row limit for testing / sampling.
    usecols : list[str] | None
        Optional column subset to load (reduces memory).
    zip_path : Path
        Path to AACT ZIP file.

    Returns
    -------
    pd.DataFrame
    """
    filename = f"{table_name}.txt"
    date_cols = _DATE_COLUMNS.get(table_name, None)

    # If usecols specified and date_cols exist, only parse dates that are in usecols
    if usecols is not None and date_cols is not None:
        date_cols = [c for c in date_cols if c in usecols]
        if not date_cols:
            date_cols = None

    with zipfile.ZipFile(zip_path, "r") as zf:
        names = zf.namelist()
        match = [n for n in names if n.endswith(filename)]
        if not match:
            raise KeyError(
                f"Table '{table_name}' not found in ZIP. "
                f"Available: {[n.split('/')[-1].replace('.txt', '') for n in names if n.endswith('.txt')]}"
            )
        with zf.open(match[0]) as f:
            text = TextIOWrapper(f, encoding="utf-8")
            df = pd.read_csv(
                text,
                sep="|",
                nrows=nrows,
                usecols=usecols,
                parse_dates=date_cols if date_cols else False,
                low_memory=False,
            )
    return df
