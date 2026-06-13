"""Load AACT pipe-delimited tables from a ZIP into pandas DataFrames.

The actual loading is delegated to the shared ``aact-kit`` package
(``aact_kit.load_table``), which handles the ZIP backend (and four others)
uniformly. This module keeps cardiotrialaudit's ZIP-path discovery and the
``load_aact_table`` signature so the rest of the pipeline is unchanged.

Install: ``pip install aact-kit`` (or an editable local checkout of aact-kit).

The AACT ZIP location is supplied entirely by the environment so no machine
path is baked into the source:

* ``CARDIOTRIALAUDIT_AACT_ZIP`` — single ZIP path (highest precedence), and/or
* ``CARDIOTRIALAUDIT_AACT_ZIP_FALLBACKS`` — additional candidate paths,
  separated by ``os.pathsep`` (``;`` on Windows, ``:`` elsewhere).

If neither is set (e.g. in CI, where tests pass ``zip_path`` explicitly) the
candidate list is empty and resolution fails closed with a clear message.
"""
import os
import zipfile  # noqa: F401  (retained for backward-compat imports)
from pathlib import Path

import pandas as pd

from aact_kit import AACTBackend, AACTLocation, load_table


def _aact_zip_candidates() -> list[Path]:
    """Candidate AACT ZIP paths from the environment, in search order."""
    candidates: list[Path] = []
    primary = os.environ.get("CARDIOTRIALAUDIT_AACT_ZIP")
    if primary:
        candidates.append(Path(primary))
    fallbacks = os.environ.get("CARDIOTRIALAUDIT_AACT_ZIP_FALLBACKS")
    if fallbacks:
        candidates.extend(
            Path(p) for p in fallbacks.split(os.pathsep) if p.strip()
        )
    return candidates


_AACT_ZIP_CANDIDATES = _aact_zip_candidates()


def _find_aact_zip() -> Path:
    """Return the first existing AACT ZIP path from candidates."""
    for p in _AACT_ZIP_CANDIDATES:
        if p.exists():
            return p
    locations = "\n".join(f"  - {p}" for p in _AACT_ZIP_CANDIDATES) or "  (none configured)"
    raise FileNotFoundError(
        "AACT ZIP not found. Set CARDIOTRIALAUDIT_AACT_ZIP (and optionally "
        "CARDIOTRIALAUDIT_AACT_ZIP_FALLBACKS) to a valid export.\n"
        f"Searched:\n{locations}"
    )


# Resolve at import time; tests override via zip_path parameter.
try:
    AACT_ZIP_PATH = _find_aact_zip()
except FileNotFoundError:
    # Allow import even if no AACT ZIP is configured (tests provide their own
    # via the zip_path parameter). The sentinel below resolves on first use.
    AACT_ZIP_PATH = Path(
        os.environ.get("CARDIOTRIALAUDIT_AACT_ZIP", "AACT_ZIP_NOT_CONFIGURED")
    )


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
