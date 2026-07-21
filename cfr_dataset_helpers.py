# Not useful 

"""Helpers for adapting outbreak datasets into CFR input formats.

The module provides:
- delay-distribution estimation from individual-level records
- standardizers for count tables and line lists
- running daily CFR estimators for the six methods described in the prompt

Expected CFR estimator signatures from the earlier notebook/code:
    cfr_naive(df, deaths_col="deaths", cases_col="cases")
    cfr_resolved_cohort(df, deaths_col="deaths", recovered_col="recovered")
    cfr_delay_adjusted_nishiura(df, delay_distribution, date_col="date", cases_col="cases", deaths_col="deaths")
    cfr_competing_risks(df, time_col="time", event_col="event", ...)
    cfr_kaplan_meier(df, time_col="time", event_col="event", ...)
    cfr_parametric_mixture(df, time_col="time", event_col="event", ...)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------
# Generic utilities
# ---------------------------------------------------------------------

def _to_datetime(series: pd.Series, dayfirst: bool = False) -> pd.Series:
    return pd.to_datetime(series, errors="coerce", dayfirst=dayfirst)


def _numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")



def derive_first_available_date(
    df: pd.DataFrame,
    candidate_cols: Sequence[str],
    dayfirst: bool = False,
    out_col: str = "analysis_origin_date",
) -> pd.Series:
    """
    Return the first non-missing parsed date across a list of candidate columns.

    Useful when symptom onset is sparse and you want a fallback time origin such
    as first consultation or confirmation date.
    """
    if not candidate_cols:
        raise ValueError("candidate_cols cannot be empty")

    parsed = []
    for c in candidate_cols:
        if c in df.columns:
            parsed.append(_to_datetime(df[c], dayfirst=dayfirst))
    if not parsed:
        raise ValueError("None of the candidate columns were found in the dataframe")

    out = parsed[0].copy()
    for ser in parsed[1:]:
        out = out.fillna(ser)
    out.name = out_col
    return out


def _drop_metadata_rows(df: pd.DataFrame, date_cols: Sequence[str], dayfirst: bool = False) -> pd.DataFrame:
    """Drop rows that are clearly header/metadata rows in CSVs."""
    work = df.copy()
    if not date_cols:
        return work

    # Keep rows where at least one of the candidate date cols parses.
    parsed_any = pd.Series(False, index=work.index)
    for c in date_cols:
        if c in work.columns:
            parsed_any = parsed_any | pd.to_datetime(work[c], errors="coerce", dayfirst=dayfirst).notna()
    return work.loc[parsed_any].copy()


def _first_existing(columns: Sequence[str], candidates: Sequence[str]) -> Optional[str]:
    cols = set(columns)
    for c in candidates:
        if c in cols:
            return c
    return None


def _clean_outcome_value(x: object) -> str:
    if pd.isna(x):
        return ""
    s = str(x).strip().lower()
    return s


def _event_from_outcome_value(outcome: object,
                             death_labels: Sequence[str],
                             recovery_labels: Sequence[str]) -> str:
    s = _clean_outcome_value(outcome)
    if s in {x.lower() for x in death_labels}:
        return "death"
    if s in {x.lower() for x in recovery_labels}:
        return "recovery"
    return "censored"


# ---------------------------------------------------------------------
# Delay distributions
# ---------------------------------------------------------------------

def estimate_delay_distribution(
    df: pd.DataFrame,
    onset_col: str,
    outcome_date_col: str,
    outcome_filter_col: Optional[str] = None,
    outcome_value: Optional[object] = None,
    dayfirst: bool = False,
    name: str = "delay",
) -> Dict[str, object]:
    """
    Estimate an empirical delay distribution from individual-level data.

    Returns a dict with:
      support, pmf, cdf, mean_delay, n, delays, name

    If outcome_filter_col/outcome_value are provided, only rows matching that
    outcome are used (useful for onset-to-death or onset-to-recovery delays).
    """
    work = df.copy()

    if onset_col not in work.columns or outcome_date_col not in work.columns:
        raise ValueError(f"Need columns {onset_col!r} and {outcome_date_col!r}")

    onset = _to_datetime(work[onset_col], dayfirst=dayfirst)
    outcome_date = _to_datetime(work[outcome_date_col], dayfirst=dayfirst)

    mask = onset.notna() & outcome_date.notna()
    if outcome_filter_col is not None and outcome_value is not None and outcome_filter_col in work.columns:
        mask = mask & (work[outcome_filter_col].astype(str).str.lower() == str(outcome_value).lower())

    delays = (outcome_date.loc[mask] - onset.loc[mask]).dt.days
    delays = delays.dropna()
    delays = delays[delays >= 0].astype(int).to_numpy()

    if delays.size == 0:
        raise ValueError(f"No non-negative delays available for {name!r}")

    support = np.arange(0, int(delays.max()) + 1)
    counts = np.bincount(delays, minlength=len(support)).astype(float)
    pmf = counts / counts.sum()
    cdf = np.cumsum(pmf)

    return {
        "name": name,
        "delays": delays,
        "support": support,
        "pmf": pmf,
        "cdf": cdf,
        "mean_delay": float(delays.mean()),
        "n": int(delays.size),
    }


def estimate_delay_distributions_from_line_list(
    df: pd.DataFrame,
    onset_col: str,
    outcome_col: str,
    outcome_date_col: Optional[str] = None,
    death_labels: Sequence[str] = ("death", "dead"),
    recovery_labels: Sequence[str] = ("recovery", "recovered", "alive", "discharged"),
    dayfirst: bool = False,
) -> Dict[str, Dict[str, object]]:
    """
    Estimate separate delay distributions for death and recovery outcomes.

    You can either supply:
      - outcome_date_col: a single date column holding the outcome date
      - or set outcome_col values and use outcome_date_col as the same date source

    The returned dict has keys 'death' and/or 'recovery' when enough data exists.
    """
    if outcome_date_col is None:
        outcome_date_col = outcome_col

    work = df.copy()
    if outcome_col not in work.columns:
        raise ValueError(f"Missing outcome column {outcome_col!r}")

    outcome_norm = work[outcome_col].map(_clean_outcome_value)
    out: Dict[str, Dict[str, object]] = {}

    for label, aliases in [("death", death_labels), ("recovery", recovery_labels)]:
        sub = work.loc[outcome_norm.isin({x.lower() for x in aliases})].copy()
        if sub.empty:
            continue
        try:
            out[label] = estimate_delay_distribution(
                sub,
                onset_col=onset_col,
                outcome_date_col=outcome_date_col,
                outcome_filter_col=outcome_col,
                outcome_value=aliases[0],
                dayfirst=dayfirst,
                name=f"{label}_delay",
            )
        except ValueError:
            # Try without the single-value filter in case aliases are mixed in the source.
            onset = _to_datetime(sub[onset_col], dayfirst=dayfirst)
            outcome_date = _to_datetime(sub[outcome_date_col], dayfirst=dayfirst)
            mask = onset.notna() & outcome_date.notna()
            delays = (outcome_date.loc[mask] - onset.loc[mask]).dt.days
            delays = delays.dropna()
            delays = delays[delays >= 0].astype(int).to_numpy()
            if delays.size == 0:
                continue
            support = np.arange(0, int(delays.max()) + 1)
            counts = np.bincount(delays, minlength=len(support)).astype(float)
            pmf = counts / counts.sum()
            cdf = np.cumsum(pmf)
            out[label] = {
                "name": f"{label}_delay",
                "delays": delays,
                "support": support,
                "pmf": pmf,
                "cdf": cdf,
                "mean_delay": float(delays.mean()),
                "n": int(delays.size),
            }

    return out


# ---------------------------------------------------------------------
# Standardized daily count table
# ---------------------------------------------------------------------

def standardize_count_table(
    df: pd.DataFrame,
    date_col: str,
    cases_col: str,
    deaths_col: str,
    recovered_col: Optional[str] = None,
    group_cols: Optional[Sequence[str]] = None,
    cumulative: bool = True,
    dayfirst: bool = False,
) -> pd.DataFrame:
    """
    Convert a count table into a standardized daily dataframe with columns:
      date, cases, deaths, recovered (optional), plus group columns if provided.

    If cumulative=True, missing values are forward-filled within each group.
    """
    work = df.copy()

    needed = [date_col, cases_col, deaths_col] + ([recovered_col] if recovered_col else [])
    needed = [c for c in needed if c is not None]
    missing = [c for c in needed if c not in work.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    group_cols = list(group_cols or [])
    keep_cols = group_cols + [date_col, cases_col, deaths_col] + ([recovered_col] if recovered_col else [])
    work = work[keep_cols].copy()

    work[date_col] = _to_datetime(work[date_col], dayfirst=dayfirst)
    work = work.dropna(subset=[date_col]).copy()
    work = work.sort_values(group_cols + [date_col] if group_cols else [date_col]).reset_index(drop=True)

    work["cases"] = _numeric(work[cases_col])
    work["deaths"] = _numeric(work[deaths_col])
    if recovered_col:
        work["recovered"] = _numeric(work[recovered_col])
    else:
        work["recovered"] = np.nan

    if cumulative:
        if group_cols:
            work[["cases", "deaths", "recovered"]] = (
                work.groupby(group_cols, dropna=False)[["cases", "deaths", "recovered"]]
                .ffill()
            )
        else:
            work[["cases", "deaths", "recovered"]] = work[["cases", "deaths", "recovered"]].ffill()

    out_cols = group_cols + [date_col, "cases", "deaths", "recovered"]
    out = work[out_cols].rename(columns={date_col: "date"}).copy()
    return out


# ---------------------------------------------------------------------
# Standardized line list table
# ---------------------------------------------------------------------

def standardize_line_list(
    df: pd.DataFrame,
    onset_col: str,
    outcome_col: Optional[str] = None,
    outcome_date_col: Optional[str] = None,
    death_labels: Sequence[str] = ("death", "dead"),
    recovery_labels: Sequence[str] = ("recovery", "recovered", "alive", "discharged"),
    censor_date_col: Optional[str] = None,
    dayfirst: bool = False,
    keep_cols: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """
    Standardize a line list into columns:
      id (optional), onset_date, outcome_date, event, censor_date

    event is one of: death, recovery, censored
    """
    work = df.copy()
    if onset_col not in work.columns:
        raise ValueError(f"Missing onset column {onset_col!r}")

    if outcome_col is not None and outcome_col not in work.columns:
        raise ValueError(f"Missing outcome column {outcome_col!r}")

    if outcome_date_col is None:
        outcome_date_col = outcome_col

    cols = []
    if keep_cols:
        cols.extend([c for c in keep_cols if c in work.columns])
    cols = list(dict.fromkeys(cols + [onset_col] + ([outcome_col] if outcome_col else []) + ([outcome_date_col] if outcome_date_col else []) + ([censor_date_col] if censor_date_col else [])))
    work = work[cols].copy() if cols else work.copy()

    work["onset_date"] = _to_datetime(work[onset_col], dayfirst=dayfirst)

    if outcome_date_col and outcome_date_col in work.columns:
        work["outcome_date"] = _to_datetime(work[outcome_date_col], dayfirst=dayfirst)
    else:
        work["outcome_date"] = pd.NaT

    if censor_date_col and censor_date_col in work.columns:
        work["censor_date"] = _to_datetime(work[censor_date_col], dayfirst=dayfirst)
    else:
        work["censor_date"] = pd.NaT

    if outcome_col and outcome_col in work.columns:
        work["event"] = work[outcome_col].map(
            lambda x: _event_from_outcome_value(x, death_labels=death_labels, recovery_labels=recovery_labels)
        )
    else:
        work["event"] = "censored"

    # If event was censored but outcome date exists and outcome text is absent,
    # keep as censored. The running-snapshot builder will handle censoring.
    out = work.copy()
    return out


# ---------------------------------------------------------------------
# Running daily aggregates from a line list
# ---------------------------------------------------------------------

def build_daily_count_snapshot_from_line_list(
    line_list: pd.DataFrame,
    analysis_date: pd.Timestamp,
    onset_date_col: str = "onset_date",
    outcome_date_col: str = "outcome_date",
    event_col: str = "event",
) -> pd.Series:
    """
    At a given analysis date, build daily aggregate counts:
      cases = cases with onset <= analysis_date
      deaths = deaths observed by analysis_date among those cases
      recovered = recoveries observed by analysis_date among those cases

    Useful for methods 1-3.
    """
    analysis_date = pd.to_datetime(analysis_date)
    onset = _to_datetime(line_list[onset_date_col])
    outcome = _to_datetime(line_list[outcome_date_col])

    at_risk = onset.notna() & (onset <= analysis_date)
    cases = int(at_risk.sum())

    deaths = int(((line_list[event_col].astype(str) == "death") & outcome.notna() & (outcome <= analysis_date) & at_risk).sum())
    recovered = int(((line_list[event_col].astype(str) == "recovery") & outcome.notna() & (outcome <= analysis_date) & at_risk).sum())

    return pd.Series({"date": analysis_date.normalize(), "cases": cases, "deaths": deaths, "recovered": recovered})


def build_daily_snapshots_from_line_list(
    line_list: pd.DataFrame,
    onset_date_col: str = "onset_date",
    outcome_date_col: str = "outcome_date",
    event_col: str = "event",
    analysis_dates: Optional[Sequence[pd.Timestamp]] = None,
) -> pd.DataFrame:
    """
    Create a daily count table from a line list, one row per analysis date.
    """
    work = line_list.copy()
    onset = _to_datetime(work[onset_date_col])
    valid_dates = onset.dropna().dt.normalize().unique()
    valid_dates = np.array(sorted(valid_dates))

    if analysis_dates is None:
        analysis_dates = valid_dates
    else:
        analysis_dates = pd.to_datetime(pd.Index(analysis_dates)).normalize().to_numpy()

    rows = [
        build_daily_count_snapshot_from_line_list(
            work,
            ad,
            onset_date_col=onset_date_col,
            outcome_date_col=outcome_date_col,
            event_col=event_col,
        )
        for ad in analysis_dates
    ]
    return pd.DataFrame(rows)


def build_running_individual_dataset(
    line_list: pd.DataFrame,
    analysis_date: Union[str, pd.Timestamp],
    onset_date_col: str = "onset_date",
    outcome_date_col: str = "outcome_date",
    event_col: str = "event",
) -> pd.DataFrame:
    """
    Build the individual-level dataset needed for KM / competing risks / mixture
    at a single analysis date.

    Rows with onset after the analysis date are excluded.
    Outcome rows that occur after the analysis date are censored at the analysis date.
    """
    analysis_date = pd.to_datetime(analysis_date).normalize()
    work = line_list.copy()

    onset = _to_datetime(work[onset_date_col])
    outcome = _to_datetime(work[outcome_date_col])

    keep = onset.notna() & (onset <= analysis_date)
    work = work.loc[keep].copy()
    onset = onset.loc[keep]
    outcome = outcome.loc[keep]

    # observation end is the event date if it happened by analysis_date, else censor at analysis_date
    obs_end = outcome.copy()
    observed_event = work[event_col].astype(str).str.lower().isin(["death", "recovery"]) & outcome.notna() & (outcome <= analysis_date)
    obs_end = obs_end.where(observed_event, analysis_date)

    work["time"] = (obs_end - onset).dt.days.astype(float)

    event = np.where(
        observed_event & (work[event_col].astype(str).str.lower() == "death"),
        "death",
        np.where(
            observed_event & (work[event_col].astype(str).str.lower() == "recovery"),
            "recovery",
            "censored",
        ),
    )
    work["event"] = event
    return work




def running_cfr_estimates_from_count_table(
    count_table: pd.DataFrame,
    cfr_naive_fn,
    cfr_resolved_fn,
    cfr_delay_fn,
    delay_distribution: Optional[Dict[str, object]] = None,
    date_col: str = "date",
    cases_col: str = "cases",
    deaths_col: str = "deaths",
    recovered_col: str = "recovered",
) -> pd.DataFrame:
    """
    Compute daily running estimates from a standardized count table.

    This supports methods 1-3 only. The count table should already contain a
    time series of cumulative or daily counts with columns date/cases/deaths/(recovered).
    """
    work = count_table.copy()
    work[date_col] = pd.to_datetime(work[date_col], errors="coerce")
    work = work.dropna(subset=[date_col]).sort_values(date_col).reset_index(drop=True)

    rows = []
    for _, row in work.iterrows():
        snapshot = pd.DataFrame([{
            "date": row[date_col],
            "cases": row.get(cases_col, np.nan),
            "deaths": row.get(deaths_col, np.nan),
            "recovered": row.get(recovered_col, np.nan),
        }])

        out = {
            "date": pd.to_datetime(row[date_col]).normalize(),
            "cases": float(row.get(cases_col, np.nan)) if pd.notna(row.get(cases_col, np.nan)) else np.nan,
            "deaths": float(row.get(deaths_col, np.nan)) if pd.notna(row.get(deaths_col, np.nan)) else np.nan,
            "recovered": float(row.get(recovered_col, np.nan)) if pd.notna(row.get(recovered_col, np.nan)) else np.nan,
        }

        try:
            out["cfr_naive"] = cfr_naive_fn(snapshot)
        except Exception:
            out["cfr_naive"] = np.nan

        try:
            out["cfr_resolved"] = cfr_resolved_fn(snapshot)
        except Exception:
            out["cfr_resolved"] = np.nan

        if delay_distribution is not None:
            try:
                out["cfr_delay_adjusted"] = cfr_delay_fn(snapshot, delay_distribution=delay_distribution)
            except Exception:
                out["cfr_delay_adjusted"] = np.nan
        else:
            out["cfr_delay_adjusted"] = np.nan

        rows.append(out)

    return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)


def running_cfr_estimates_from_line_list(
    line_list: pd.DataFrame,
    analysis_dates: Sequence[Union[str, pd.Timestamp]],
    cfr_naive_fn,
    cfr_resolved_fn,
    cfr_delay_fn,
    cfr_competing_fn,
    cfr_km_fn,
    cfr_mixture_fn,
    delay_distribution: Optional[Dict[str, object]] = None,
    onset_date_col: str = "onset_date",
    outcome_date_col: str = "outcome_date",
    event_col: str = "event",
) -> pd.DataFrame:
    """
    Compute daily running estimates from a standardized line list.
    For methods 4-6, each analysis date creates a censored snapshot.
    For methods 1-3, each analysis date creates aggregate counts.
    """
    rows = []

    for ad in pd.to_datetime(pd.Index(analysis_dates)).normalize():
        counts_df = build_daily_count_snapshot_from_line_list(
            line_list,
            ad,
            onset_date_col=onset_date_col,
            outcome_date_col=outcome_date_col,
            event_col=event_col,
        )

        indiv_df = build_running_individual_dataset(
            line_list,
            ad,
            onset_date_col=onset_date_col,
            outcome_date_col=outcome_date_col,
            event_col=event_col,
        )

        row = {
            "date": ad,
            "n_cases": int(counts_df["cases"]),
            "n_deaths": int(counts_df["deaths"]),
            "n_recovered": int(counts_df["recovered"]),
        }

        # 1-2 always available if the needed counts exist.
        try:
            row["cfr_naive"] = cfr_naive_fn(pd.DataFrame([counts_df]))
        except Exception:
            row["cfr_naive"] = np.nan

        try:
            row["cfr_resolved"] = cfr_resolved_fn(pd.DataFrame([counts_df]))
        except Exception:
            row["cfr_resolved"] = np.nan

        # 3 requires delay distribution
        if delay_distribution is not None:
            try:
                row["cfr_delay_adjusted"] = cfr_delay_fn(pd.DataFrame([counts_df]), delay_distribution=delay_distribution)
            except Exception:
                row["cfr_delay_adjusted"] = np.nan
        else:
            row["cfr_delay_adjusted"] = np.nan

        # 4-6 need individual-level snapshot
        try:
            row["cfr_competing_risks"] = cfr_competing_fn(indiv_df)
        except Exception:
            row["cfr_competing_risks"] = np.nan

        try:
            row["cfr_kaplan_meier"] = cfr_km_fn(indiv_df)
        except Exception:
            row["cfr_kaplan_meier"] = np.nan

        try:
            row["cfr_parametric_mixture"] = cfr_mixture_fn(indiv_df)
            if isinstance(row["cfr_parametric_mixture"], dict):
                row["cfr_parametric_mixture"] = row["cfr_parametric_mixture"].get("cfr", np.nan)
        except Exception:
            row["cfr_parametric_mixture"] = np.nan

        rows.append(row)

    return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)


# ---------------------------------------------------------------------
# Dataset-specific loaders
# ---------------------------------------------------------------------

def load_drc_total(path: str) -> pd.DataFrame:
    """DRC 2018 MOH total table -> standardized count table."""
    raw = pd.read_csv(path)
    raw = _drop_metadata_rows(raw, ["report_date"], dayfirst=False)
    return standardize_count_table(
        raw,
        date_col="report_date",
        cases_col="total_cases",
        deaths_col="total_deaths",
        recovered_col="total_cured" if "total_cured" in raw.columns else None,
        cumulative=True,
        dayfirst=False,
    )


def load_drc_by_health_zone(path: str) -> pd.DataFrame:
    """DRC 2018 MOH by-health-zone table -> standardized count table with grouping."""
    raw = pd.read_csv(path)
    raw = _drop_metadata_rows(raw, ["report_date"], dayfirst=False)
    return standardize_count_table(
        raw,
        date_col="report_date",
        cases_col="total_cases",
        deaths_col="total_deaths",
        recovered_col="total_cured" if "total_cured" in raw.columns else None,
        group_cols=[c for c in ["province", "health_zone"] if c in raw.columns],
        cumulative=True,
        dayfirst=False,
    )


def load_drc_consolidated(path: str) -> pd.DataFrame:
    """
    DRC consolidated line table -> standardized count table at the location/date level.

    This table does not contain recovered counts, so method 2 cannot be used directly.
    Method 3 can still be used if an external delay distribution is supplied.
    """
    raw = pd.read_csv(path)
    raw = _drop_metadata_rows(raw, ["reference_date"], dayfirst=False)
    # aggregate to a daily national-style table by date if multiple rows exist
    work = raw.copy()
    work["reference_date"] = pd.to_datetime(work["reference_date"], errors="coerce")
    work["value"] = _numeric(work["value"])
    work = work.dropna(subset=["reference_date", "measure", "value"])

    # pivot to get cases/deaths totals
    piv = (
        work.loc[work["measure"].isin(["cases", "deaths"])]
        .pivot_table(
            index=["reference_date"],
            columns=["measure"],
            values="value",
            aggfunc="sum",
        )
        .reset_index()
    )
    piv = piv.rename(columns={"reference_date": "date", "cases": "cases", "deaths": "deaths"})
    if "cases" not in piv.columns:
        piv["cases"] = np.nan
    if "deaths" not in piv.columns:
        piv["deaths"] = np.nan
    piv["recovered"] = np.nan
    piv = piv.sort_values("date").reset_index(drop=True)
    return piv


def load_rosello2015(path: str) -> pd.DataFrame:
    """
    Rosello supplementary line list -> standardized line list.

    Outcome: Dead/Alive. Date_of_Death and Date_hospital_discharge can be used
    as outcome dates. Date_of_onset_symp is the preferred time origin.
    """
    raw = pd.read_csv(path)
    raw = _drop_metadata_rows(raw, ["Date_of_onset_symp", "Date_of_Death", "Date_hospital_discharge", "Date_of_notification", "Date_of_Hospitalisation", "Date_disease_ended"], dayfirst=True)
    out = standardize_line_list(
        raw,
        onset_col="Date_of_onset_symp",
        outcome_col="Outcome",
        outcome_date_col="Date_of_Death",
        death_labels=("Dead",),
        recovery_labels=("Alive",),
        dayfirst=True,
        keep_cols=[c for c in ["Outbreak", "Person_ID", "Outcome", "Date_of_onset_symp", "Date_of_Death", "Date_hospital_discharge", "Date_of_notification", "Date_of_Hospitalisation", "Date_disease_ended"] if c in raw.columns],
    )
    out["analysis_origin_date"] = derive_first_available_date(
        raw,
        candidate_cols=[c for c in ["Date_of_onset_symp", "Date_of_Hospitalisation", "Date_of_notification", "Date_disease_ended"] if c in raw.columns],
        dayfirst=True,
        out_col="analysis_origin_date",
    )
    # For recoveries, discharge date is a better outcome date than disease-ended where available.
    if "Date_hospital_discharge" in raw.columns:
        discharge = _to_datetime(raw["Date_hospital_discharge"], dayfirst=True)
        alive_mask = out["event"].eq("recovery") & discharge.notna()
        out.loc[alive_mask, "outcome_date"] = discharge.loc[alive_mask].values
    return out


def load_uganda_2022(path: str) -> pd.DataFrame:
    """
    Uganda 2022 Global.health line list -> standardized line list.

    Outcome: Death / Recovery. Date_onset is preferred for onset.
    """
    raw = pd.read_csv(path)
    raw = _drop_metadata_rows(raw, ["Date_onset", "Date_Death", "Date_Recovered", "Date_confirmation", "Date_of_first_consult", "Date_hospitalisation"], dayfirst=False)
    out = standardize_line_list(
        raw,
        onset_col="Date_onset",
        outcome_col="Outcome",
        outcome_date_col="Date_Death",
        death_labels=("Death",),
        recovery_labels=("Recovery",),
        dayfirst=False,
        keep_cols=[c for c in ["ID", "Case_status", "Date_onset", "Date_Death", "Date_Recovered", "Outcome", "Location_District", "Country", "Date_confirmation", "Date_of_first_consult"] if c in raw.columns],
    )
    out["analysis_origin_date"] = derive_first_available_date(
        raw,
        candidate_cols=[c for c in ["Date_onset", "Date_of_first_consult", "Date_confirmation"] if c in raw.columns],
        dayfirst=False,
        out_col="analysis_origin_date",
    )
    # Recovery outcome dates
    if "Date_Recovered" in raw.columns:
        recovered_dates = _to_datetime(raw["Date_Recovered"], dayfirst=False)
        rec_mask = out["event"].eq("recovery") & recovered_dates.notna()
        out.loc[rec_mask, "outcome_date"] = recovered_dates.loc[rec_mask].values
    return out


def load_sierra_leone_line_lists(path_confirmed: str, path_suspected: Optional[str] = None) -> pd.DataFrame:
    """
    Sierra Leone 2014 confirmed/suspected line lists.

    These tables have onset and sample test dates, but no outcome column, so
    they are NOT directly sufficient for CFR methods 1-6. They can, however,
    be standardized here if you want to study onset-to-testing delays.
    """
    frames = []
    for path in [path_confirmed, path_suspected]:
        if path is None:
            continue
        raw = pd.read_csv(path)
        raw = _drop_metadata_rows(raw, ["Date of symptom onset ", "Date of sample tested"], dayfirst=True)
        tmp = raw.copy()
        tmp["onset_date"] = _to_datetime(tmp["Date of symptom onset "], dayfirst=True)
        tmp["analysis_origin_date"] = tmp["onset_date"]
        tmp["sample_tested_date"] = _to_datetime(tmp["Date of sample tested"], dayfirst=True)
        tmp["delay_to_test"] = (tmp["sample_tested_date"] - tmp["onset_date"]).dt.days
        if "Outcome" not in tmp.columns:
            tmp["Outcome"] = np.nan
        frames.append(tmp)
    if not frames:
        raise ValueError("No input files provided.")
    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------
# Convenience: method-availability checks
# ---------------------------------------------------------------------

def available_methods_for_dataset(df: pd.DataFrame) -> Dict[str, bool]:
    """
    Heuristic compatibility check based on standardized columns.
    """
    cols = set(df.columns)
    has_counts = {"cases", "deaths"}.issubset(cols)
    has_recovered = "recovered" in cols
    has_individual = {"time", "event"}.issubset(cols) or {"onset_date", "event"}.issubset(cols)

    return {
        "naive": has_counts,
        "resolved_cohort": has_counts and has_recovered,
        "delay_adjusted_nishiura": has_counts,
        "competing_risks": has_individual,
        "kaplan_meier": has_individual,
        "parametric_mixture": has_individual,
    }
