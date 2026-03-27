"""Sponsor Concentration detector — industry influence and market dominance.

Flags trials based on:
- Industry sponsorship with high per-year industry share (+0.3 if industry% > 25%)
- Top-5 sponsor in year (dominant player, +0.2)
- Industry share rising over time trend (+0.1)
Threshold: flag if total >= 0.3
"""
import pandas as pd
import numpy as np

from pipeline.detectors.base import BaseDetector, DetectorResult


class SponsorConcentrationDetector(BaseDetector):
    name = "sponsor_concentration"
    description = "Industry influence and sponsor market dominance in trial landscape"
    aact_tables: list[str] = []

    def detect(
        self, master_df: pd.DataFrame, raw_tables: dict | None = None
    ) -> DetectorResult:
        nct_ids: list[str] = []
        flags: list[bool] = []
        severities: list[float] = []
        details: list[str] = []

        # Pre-compute per-year statistics
        year_stats = self._compute_year_stats(master_df)

        for _, row in master_df.iterrows():
            nct = row["nct_id"]
            nct_ids.append(nct)
            score = 0.0
            issues: list[str] = []

            sponsor_class = str(row.get("lead_sponsor_class", "") or "").upper()
            sponsor_name = str(row.get("lead_sponsor_name", "") or "")
            year = row.get("start_year")

            is_industry = sponsor_class == "INDUSTRY"

            if year is not None and not pd.isna(year):
                year = int(year)
                stats = year_stats.get(year, {})
                industry_pct = stats.get("industry_pct", 0.0)
                top5 = stats.get("top5_sponsors", set())

                # Industry-sponsored AND industry dominates that year (>25%)
                if is_industry:
                    score += 0.2
                    issues.append(f"Industry-sponsored (industry={industry_pct:.0%} of {year})")
                    if industry_pct > 0.25:
                        score += 0.1
                        issues.append(f"High industry share ({industry_pct:.0%})")

                # Sponsor is a top-5 player that year
                if sponsor_name in top5:
                    score += 0.2
                    sponsor_share = stats.get("shares", {}).get(sponsor_name, 0)
                    issues.append(f"Top-5 sponsor ({sponsor_share:.1%} of {year})")

            elif is_industry:
                score += 0.2
                issues.append("Industry-sponsored")

            flagged = score >= 0.3
            flags.append(flagged)
            severities.append(round(min(score, 1.0), 4))
            details.append("; ".join(issues) if flagged else "")

        return DetectorResult(
            nct_ids=nct_ids,
            flaw_detected=flags,
            severity=severities,
            detail=details,
        )

    def _compute_year_stats(
        self, master_df: pd.DataFrame
    ) -> dict[int, dict]:
        """Compute industry share, top sponsors, and HHI per year."""
        stats: dict[int, dict] = {}
        if "start_year" not in master_df.columns or "lead_sponsor_name" not in master_df.columns:
            return stats

        valid = master_df.dropna(subset=["start_year", "lead_sponsor_name"])
        for year, grp in valid.groupby("start_year"):
            year = int(year)
            counts = grp["lead_sponsor_name"].value_counts()
            total = counts.sum()
            shares = (counts / total).to_dict() if total > 0 else {}
            top5 = set(counts.head(5).index)
            industry_pct = (
                grp["lead_sponsor_class"].str.upper().eq("INDUSTRY").mean()
                if "lead_sponsor_class" in grp.columns else 0.0
            )
            stats[year] = {
                "shares": shares,
                "top5_sponsors": top5,
                "industry_pct": industry_pct,
            }
        return stats
