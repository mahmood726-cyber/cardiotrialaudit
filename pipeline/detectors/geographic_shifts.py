"""Geographic Shifts detector — trials with disproportionate LMIC site concentration.

Loads facilities table from AACT to determine country distribution.
Flags trials with >50% LMIC sites and >= 3 total sites.
"""
import logging

import pandas as pd

from pipeline.detectors.base import BaseDetector, DetectorResult

logger = logging.getLogger(__name__)

# High-income countries (World Bank classification, ISO 3166 names as in AACT)
_HIGH_INCOME_COUNTRIES = frozenset({
    "United States", "Canada",
    "United Kingdom", "Germany", "France", "Italy", "Spain", "Netherlands",
    "Belgium", "Austria", "Switzerland", "Sweden", "Norway", "Denmark",
    "Finland", "Ireland", "Luxembourg", "Iceland",
    "Portugal", "Greece", "Czech Republic", "Czechia",
    "Estonia", "Latvia", "Lithuania", "Slovenia", "Slovakia", "Croatia",
    "Malta", "Cyprus",
    "Japan", "South Korea", "Korea, Republic of",
    "Australia", "New Zealand",
    "Israel", "Singapore", "Hong Kong",
    "Taiwan", "United Arab Emirates", "Qatar", "Kuwait", "Bahrain",
    "Saudi Arabia", "Oman",
    "Poland", "Hungary",  # EU/OECD with EMA oversight — keep as HIC
    # P1-11: Romania and Bulgaria removed — classified as UMIC by World Bank
    "Chile", "Uruguay", "Panama",
    "Puerto Rico", "Guam",
})


class GeographicShiftsDetector(BaseDetector):
    name = "geographic_shifts"
    description = "Trials with disproportionate enrollment from LMIC sites"
    aact_tables = ("facilities",)

    def detect(
        self, master_df: pd.DataFrame, raw_tables: dict | None = None
    ) -> DetectorResult:
        nct_ids: list[str] = []
        flags: list[bool] = []
        severities: list[float] = []
        details: list[str] = []

        # Load facilities
        site_stats = self._load_site_stats(master_df, raw_tables)

        for _, row in master_df.iterrows():
            nct = row["nct_id"]
            nct_ids.append(nct)

            stats = site_stats.get(nct)
            if stats is None:
                flags.append(False)
                severities.append(0.0)
                details.append("")
                continue

            total_sites = stats["total"]
            lmic_sites = stats["lmic"]
            lmic_prop = lmic_sites / total_sites if total_sites > 0 else 0.0

            if total_sites >= 3 and lmic_prop > 0.5:
                flags.append(True)
                severities.append(round(lmic_prop, 4))
                countries_str = ", ".join(sorted(stats.get("lmic_countries", [])))
                details.append(
                    f"LMIC {lmic_prop:.0%} ({lmic_sites}/{total_sites} sites); "
                    f"countries: {countries_str}"
                )
            else:
                flags.append(False)
                severities.append(0.0)
                details.append("")

        return DetectorResult(
            nct_ids=nct_ids,
            flaw_detected=flags,
            severity=severities,
            detail=details,
        )

    def _load_site_stats(
        self, master_df: pd.DataFrame, raw_tables: dict | None
    ) -> dict[str, dict]:
        """Compute per-trial site distribution from facilities table."""
        try:
            if raw_tables is not None:
                fac = raw_tables.get("facilities")
                if fac is None:
                    return {}
            else:
                from pipeline.ingest import load_aact_table
                fac = load_aact_table(
                    "facilities", usecols=["nct_id", "country"]
                )
        except (KeyError, FileNotFoundError) as e:
            logger.warning("Could not load facilities: %s", e)
            return {}

        nct_set = set(master_df["nct_id"])
        fac = fac[fac["nct_id"].isin(nct_set)].copy()
        fac["country"] = fac["country"].fillna("Unknown")

        result: dict[str, dict] = {}
        for nct, grp in fac.groupby("nct_id"):
            countries = grp["country"].tolist()
            total = len(countries)
            lmic_countries = set()
            lmic_count = 0
            for c in countries:
                if c not in _HIGH_INCOME_COUNTRIES:
                    lmic_count += 1
                    lmic_countries.add(c)
            result[nct] = {
                "total": total,
                "lmic": lmic_count,
                "lmic_countries": lmic_countries,
            }
        return result
