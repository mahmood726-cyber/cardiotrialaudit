"""Statistical Fragility detector — fragility index for 2x2 trial results.

Implements compute_fragility_index using Fisher exact test.
For positive trials (p < 0.05): modify fewer-events arm, add events
one at a time until p >= 0.05.

FI <= 3: severity 0.8; FI 4-7: 0.5; FI 8-15: 0.3
Coverage: ~5-15% of trials (only those with extractable 2x2 tables).

P0-5: Rewritten _extract_2x2_tables to use result_groups, outcome_measurements,
and baseline_measurements for arm sizes, with reported_events as fallback.
"""
import logging

import pandas as pd

from pipeline.detectors.base import BaseDetector, DetectorResult

logger = logging.getLogger(__name__)


def compute_fragility_index(
    events_a: int, total_a: int, events_b: int, total_b: int
) -> int | None:
    """Compute the Fragility Index for a 2x2 table.

    Modifies the fewer-events arm only, adding events one at a time
    until Fisher exact p >= 0.05.

    Parameters
    ----------
    events_a, total_a : int
        Events and total in arm A.
    events_b, total_b : int
        Events and total in arm B.

    Returns
    -------
    int | None
        Fragility index, or None if initial p >= 0.05 (not significant).
    """
    from scipy.stats import fisher_exact

    # Validate inputs
    if (
        total_a <= 0 or total_b <= 0
        or events_a < 0 or events_b < 0
        or events_a > total_a or events_b > total_b
    ):
        return None

    # Build initial 2x2 table
    table = [
        [events_a, total_a - events_a],
        [events_b, total_b - events_b],
    ]
    _, p = fisher_exact(table)
    if p >= 0.05:
        return None  # Not significant — no fragility index

    # Determine which arm has fewer events (that's the one we modify)
    fi = 0
    # Work on copies
    ea, na = events_a, total_a
    eb, nb = events_b, total_b

    # We add events to the arm with fewer events
    modify_a = ea <= eb

    max_iter = 1000  # safety guard
    while fi < max_iter:
        if modify_a:
            if ea >= na:
                break
            ea += 1
        else:
            if eb >= nb:
                break
            eb += 1

        table = [[ea, na - ea], [eb, nb - eb]]
        _, p = fisher_exact(table)
        fi += 1

        if p >= 0.05:
            return fi

    return fi if fi > 0 else None


def _fi_to_severity(fi: int | None) -> float:
    """Map fragility index to severity score."""
    if fi is None:
        return 0.0
    if fi <= 3:
        return 0.8
    if fi <= 7:
        return 0.5
    if fi <= 15:
        return 0.3
    return 0.1


class StatisticalFragilityDetector(BaseDetector):
    name = "statistical_fragility"
    description = "Trial results that would change with very few event reassignments"
    aact_tables = (
        "outcome_counts", "outcomes", "result_groups",
        "outcome_measurements", "baseline_measurements", "reported_events",
    )

    def detect(
        self, master_df: pd.DataFrame, raw_tables: dict | None = None
    ) -> DetectorResult:
        nct_ids: list[str] = []
        flags: list[bool] = []
        severities: list[float] = []
        details: list[str] = []

        # Try to load outcome data for 2x2 extraction
        tables_2x2 = self._extract_2x2_tables(master_df, raw_tables)

        for _, row in master_df.iterrows():
            nct = row["nct_id"]
            nct_ids.append(nct)

            data = tables_2x2.get(nct)
            if data is None:
                flags.append(False)
                severities.append(0.0)
                details.append("")
                continue

            fi = compute_fragility_index(
                data["events_a"], data["total_a"],
                data["events_b"], data["total_b"],
            )
            if fi is not None:
                sev = _fi_to_severity(fi)
                flags.append(True)
                severities.append(sev)
                details.append(
                    f"FI={fi} (events: {data['events_a']}/{data['total_a']} "
                    f"vs {data['events_b']}/{data['total_b']})"
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

    def _extract_2x2_tables(
        self, master_df: pd.DataFrame, raw_tables: dict | None
    ) -> dict[str, dict]:
        """Extract 2x2 tables from AACT tables.

        P0-5: Rewritten to properly extract arm sizes:
        1. Get arm groups from result_groups (P1/P2)
        2. Get event counts from outcome_measurements for PRIMARY outcomes
        3. Get arm sizes from baseline_measurements ("Number Analyzed")
        4. Fallback: reported_events subjects_at_risk for N per arm
        5. Final fallback: outcome_counts (original method)
        """
        nct_set = set(master_df["nct_id"])
        result: dict[str, dict] = {}

        # Try the improved extraction first
        improved = self._extract_from_measurements(master_df, raw_tables, nct_set)
        result.update(improved)

        # Fallback to outcome_counts for trials not already extracted
        remaining = nct_set - set(result.keys())
        if remaining:
            fallback = self._extract_from_outcome_counts(
                master_df, raw_tables, remaining
            )
            result.update(fallback)

        return result

    def _extract_from_measurements(
        self, master_df: pd.DataFrame, raw_tables: dict | None,
        nct_set: set[str],
    ) -> dict[str, dict]:
        """Extract 2x2 from outcome_measurements + baseline_measurements."""
        result: dict[str, dict] = {}

        # Load result_groups
        rg = self._load_table("result_groups", raw_tables,
                              usecols=["nct_id", "ctgov_group_code", "title", "description"])
        if rg is None or rg.empty:
            return result

        # Load outcome_measurements
        om = self._load_table("outcome_measurements", raw_tables,
                              usecols=["nct_id", "outcome_id", "ctgov_group_code",
                                    "param_type", "param_value_num"])
        if om is None or om.empty:
            return result

        # Load baseline_measurements for arm sizes
        bm = self._load_table("baseline_measurements", raw_tables,
                              usecols=["nct_id", "ctgov_group_code", "title",
                                    "param_type", "param_value_num"])

        # Load reported_events as fallback for arm sizes
        re_df = self._load_table("reported_events", raw_tables,
                                 usecols=["nct_id", "ctgov_group_code",
                                       "subjects_at_risk"])

        # Filter to our trials
        rg = rg[rg["nct_id"].isin(nct_set)]
        om = om[om["nct_id"].isin(nct_set)]

        # For each trial, try to build a 2x2
        om["param_value_num"] = pd.to_numeric(om["param_value_num"], errors="coerce")

        for nct in nct_set:
            try:
                nct_rg = rg[rg["nct_id"] == nct]
                # Need exactly P1 and P2 groups
                group_codes = nct_rg["ctgov_group_code"].unique()
                if not {"P1", "P2"}.issubset(set(group_codes)):
                    continue

                nct_om = om[om["nct_id"] == nct]
                if nct_om.empty:
                    continue

                # Get first outcome's data as proxy for primary
                outcome_ids = nct_om["outcome_id"].unique()
                if len(outcome_ids) == 0:
                    continue

                first_om = nct_om[nct_om["outcome_id"] == outcome_ids[0]]
                p1_data = first_om[first_om["ctgov_group_code"] == "P1"]
                p2_data = first_om[first_om["ctgov_group_code"] == "P2"]

                if p1_data.empty or p2_data.empty:
                    continue

                events_a = p1_data["param_value_num"].iloc[0]
                events_b = p2_data["param_value_num"].iloc[0]

                if pd.isna(events_a) or pd.isna(events_b):
                    continue

                events_a = int(events_a)
                events_b = int(events_b)

                # Get arm sizes — try baseline_measurements first
                total_a, total_b = self._get_arm_sizes(
                    nct, bm, re_df
                )
                if total_a is None or total_b is None:
                    continue

                # Sanity check
                if (total_a > 0 and total_b > 0
                        and 0 <= events_a <= total_a
                        and 0 <= events_b <= total_b):
                    result[nct] = {
                        "events_a": events_a,
                        "total_a": total_a,
                        "events_b": events_b,
                        "total_b": total_b,
                    }
            except Exception:
                continue

        return result

    def _get_arm_sizes(
        self, nct: str,
        bm: pd.DataFrame | None,
        re_df: pd.DataFrame | None,
    ) -> tuple[int | None, int | None]:
        """Get arm sizes (N per arm) from baseline or reported_events.

        Tries:
        1. baseline_measurements with title containing "Number Analyzed" or
           "Overall Number of Participants"
        2. reported_events subjects_at_risk (max per group)
        """
        # Try baseline_measurements
        if bm is not None and not bm.empty:
            nct_bm = bm[bm["nct_id"] == nct]
            if not nct_bm.empty:
                # Look for participant count rows
                analyzer_mask = nct_bm["title"].fillna("").str.contains(
                    r"(?i)(number\s*(of\s*)?(analyz|participant|enroll|baseline))",
                    regex=True
                )
                count_rows = nct_bm[analyzer_mask]
                if not count_rows.empty:
                    count_rows = count_rows.copy()
                    count_rows["param_value_num"] = pd.to_numeric(
                        count_rows["param_value_num"], errors="coerce"
                    )
                    p1 = count_rows[count_rows["ctgov_group_code"] == "P1"]
                    p2 = count_rows[count_rows["ctgov_group_code"] == "P2"]
                    if not p1.empty and not p2.empty:
                        ta = p1["param_value_num"].iloc[0]
                        tb = p2["param_value_num"].iloc[0]
                        if pd.notna(ta) and pd.notna(tb) and ta > 0 and tb > 0:
                            return int(ta), int(tb)

        # Fallback: reported_events subjects_at_risk
        if re_df is not None and not re_df.empty:
            nct_re = re_df[re_df["nct_id"] == nct]
            if not nct_re.empty:
                nct_re = nct_re.copy()
                nct_re["subjects_at_risk"] = pd.to_numeric(
                    nct_re["subjects_at_risk"], errors="coerce"
                )
                p1_sar = nct_re[nct_re["ctgov_group_code"] == "P1"]["subjects_at_risk"].max()
                p2_sar = nct_re[nct_re["ctgov_group_code"] == "P2"]["subjects_at_risk"].max()
                if pd.notna(p1_sar) and pd.notna(p2_sar) and p1_sar > 0 and p2_sar > 0:
                    return int(p1_sar), int(p2_sar)

        return None, None

    def _extract_from_outcome_counts(
        self, master_df: pd.DataFrame, raw_tables: dict | None,
        nct_set: set[str],
    ) -> dict[str, dict]:
        """Fallback: extract 2x2 from outcome_counts (original method)."""
        oc = self._load_table("outcome_counts", raw_tables,
                              usecols=None)  # load all columns
        if oc is None or oc.empty:
            return {}

        oc = oc[oc["nct_id"].isin(nct_set)].copy()

        if "count" not in oc.columns or "scope" not in oc.columns:
            return {}

        oc["count"] = pd.to_numeric(oc["count"], errors="coerce")

        result: dict[str, dict] = {}
        for nct, grp in oc.groupby("nct_id"):
            participants = grp[grp["scope"].str.upper().eq("PARTICIPANTS")] if "scope" in grp.columns else grp

            if participants.empty:
                continue

            if "ctgov_group_code" not in participants.columns:
                continue

            group_counts = participants.groupby("ctgov_group_code")["count"].sum()
            if len(group_counts) != 2:
                continue

            groups = sorted(group_counts.index.tolist())
            total_a = int(group_counts[groups[0]])
            total_b = int(group_counts[groups[1]])

            if total_a <= 0 or total_b <= 0:
                continue

            outcome_ids = participants["outcome_id"].unique()
            if len(outcome_ids) == 0:
                continue

            first_outcome = participants[participants["outcome_id"] == outcome_ids[0]]
            events_by_group = first_outcome.groupby("ctgov_group_code")["count"].sum()

            if len(events_by_group) == 2:
                events_a = int(events_by_group.get(groups[0], 0))
                events_b = int(events_by_group.get(groups[1], 0))

                if 0 <= events_a <= total_a and 0 <= events_b <= total_b:
                    result[nct] = {
                        "events_a": events_a,
                        "total_a": total_a,
                        "events_b": events_b,
                        "total_b": total_b,
                    }

        return result

    # _load_table inherited from BaseDetector
