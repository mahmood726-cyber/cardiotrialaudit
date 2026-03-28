"""Base class and result container for all flaw detectors."""
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass

import pandas as pd

logger = logging.getLogger(__name__)

# Snapshot date of the AACT data used by all time-dependent detectors.
# All detectors should import this instead of defining their own _NOW.
AACT_SNAPSHOT_DATE = pd.Timestamp("2026-02-19")


@dataclass
class DetectorResult:
    """Container for detector output — one entry per trial in master_df."""

    nct_ids: list[str]
    flaw_detected: list[bool]
    severity: list[float]  # 0.0 -- 1.0
    detail: list[str]

    def __post_init__(self):
        n = len(self.nct_ids)
        if len(self.flaw_detected) != n or len(self.severity) != n or len(self.detail) != n:
            raise ValueError(
                f"All lists must have equal length (nct_ids={n}, "
                f"flaw_detected={len(self.flaw_detected)}, "
                f"severity={len(self.severity)}, detail={len(self.detail)})"
            )

    def to_dataframe(self, detector_name: str) -> pd.DataFrame:
        """Convert to a DataFrame with detector-specific column names."""
        return pd.DataFrame({
            "nct_id": self.nct_ids,
            f"{detector_name}_detected": self.flaw_detected,
            f"{detector_name}_severity": self.severity,
            f"{detector_name}_detail": self.detail,
        })


class BaseDetector(ABC):
    """Abstract base class for all flaw detectors."""

    name: str = ""
    description: str = ""
    aact_tables: tuple[str, ...] = ()

    def _load_table(
        self, table_name: str, raw_tables: dict | None, usecols: list[str] | None = None
    ) -> pd.DataFrame | None:
        """Load an AACT table from raw_tables dict or from disk via ingest.

        Parameters
        ----------
        table_name : str
            Name of the AACT table (e.g. 'outcomes', 'facilities').
        raw_tables : dict | None
            Pre-loaded tables dict, or None to load from disk.
        usecols : list[str] | None
            Columns to select when loading from disk.

        Returns
        -------
        pd.DataFrame | None
            The table, or None if unavailable.
        """
        if raw_tables is not None:
            return raw_tables.get(table_name)
        try:
            from pipeline.ingest import load_aact_table
            return load_aact_table(table_name, usecols=usecols)
        except (KeyError, FileNotFoundError):
            return None

    @abstractmethod
    def detect(
        self, master_df: pd.DataFrame, raw_tables: dict | None = None
    ) -> DetectorResult:
        """Run detection on the master table.

        Parameters
        ----------
        master_df : pd.DataFrame
            One-row-per-trial table from build_master_table().
        raw_tables : dict | None
            Optional dict of {table_name: DataFrame} for AACT tables.
            If None, detectors needing extra tables will load them via ingest.

        Returns
        -------
        DetectorResult
            Must have length == len(master_df).
        """
        ...
