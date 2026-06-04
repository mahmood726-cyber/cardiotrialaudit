"""Load AACT pipe-delimited tables from a ZIP into pandas DataFrames.

The actual loading is delegated to the shared ``aact-kit`` package
(``aact_kit.load_table``), which handles the ZIP backend (and four others)
uniformly. This module keeps cardiotrialaudit's ZIP-path discovery and the
``load_aact_table`` signature so the rest of the pipeline is unchanged.

Install: ``pip install aact-kit`` (or ``pip install -e C:/Projects/aact-kit``).
"""
import zipfile  # noqa: F401  (retained for backward-compat imports)
from pathlib import Path

import pandas as pd

from aact_kit import AACTBackend, AACTLocation, load_table

# Candidate AACT ZIP paths, searched in order
_AACT_ZIP_CANDIDATES = [
    Path(r"C:\Projects\hfpef_registry_calibration\data\aact\20260219_export_ctgov.zip"),
    Path(r"C:\Users\user\Pairwise70\hfpef_registry_calibration\data\aact\20260219_export_ctgov.zip"),
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


def load_aact_table(
    table_name: str,
    nrows: int | None = None,
    usecols: list[str] | None = None,
    zip_path: Path = AACT_ZIP_PATH,
) -> pd.DataFrame:
    """Load a single AACT table from the ZIP archive.

    Thin wrapper over :func:`aact_kit.load_table` with a ZIP location. Date
    columns for ``studies`` are parsed automatically (via aact-kit's
    ``DEFAULT_DATE_COLUMNS``), an absent table raises ``KeyError``, and the
    inner-zip member is matched by suffix with exact-filename preference —
    identical to the previous local implementation.

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
    location = AACTLocation(AACTBackend.ZIP, str(zip_path))
    return load_table(table_name, location=location, columns=usecols, nrows=nrows)
