
"""Exact CFR estimators and dataset adapters.

This module provides:
- dataset loaders for the attached CSV files
- helpers to standardize count tables and line lists
- delay-distribution estimation from individual-level data
- CFR estimators following the original published definitions
- a running_cfr() wrapper for daily expanding-window estimates

Supported estimator families:
1. Naive D/C
2. Resolved cohort D/(D+R)
3. Delay-adjusted confirmed CFR (Nishiura et al. / Epiverse cfr_static)
4. Competing risks cumulative incidence (Aalen-Johansen)
5. Kaplan-Meier adapted for death vs recovery (Ghani et al.)
6. Parametric mixture/cure model (Ghani et al.)

The point-estimate formulas are implemented directly from the published
descriptions and the Epiverse reference implementation docs.

Note that this code is made with AI and has not been fully checked. 
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import math
import warnings

import numpy as np
import pandas as pd
from scipy.special import expit, logit
from scipy import stats
from scipy.stats import beta, chi2, norm
from functools import lru_cache
from scipy.optimize import minimize, root_scalar

# ---------------------------------------------------------------------------
# Generic utilities
# ---------------------------------------------------------------------------

def _to_datetime(series: pd.Series, dayfirst: bool = False) -> pd.Series:
    return pd.to_datetime(series, errors="coerce", dayfirst=dayfirst)


def _coerce_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _require_columns(df: pd.DataFrame, cols: Sequence[str], *, label: str = "data") -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{label} is missing required columns: {missing}")


def _first_existing_column(df: pd.DataFrame, candidates: Sequence[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _safe_lower(value: Any) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip().lower()


def _ensure_sorted_unique_dates(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
    out = df.copy()
    out[date_col] = pd.to_datetime(out[date_col], errors="coerce")
    out = out.dropna(subset=[date_col]).sort_values(date_col).reset_index(drop=True)
    return out


def _daily_complete_dates(df: pd.DataFrame, date_col: str, start: Optional[pd.Timestamp] = None,
                          end: Optional[pd.Timestamp] = None) -> pd.DatetimeIndex:
    if df.empty:
        return pd.DatetimeIndex([])
    if start is None:
        start = pd.to_datetime(df[date_col].min())
    if end is None:
        end = pd.to_datetime(df[date_col].max())
    return pd.date_range(start=start, end=end, freq="D")



def fit_parametric_delay_distribution(delays, family="gamma"):
    x = pd.Series(delays).astype(float)
    x = x[np.isfinite(x) & (x >= 0)].to_numpy()
    if x.size == 0:
        raise ValueError("No valid non-negative delays available.")

    # avoid exact zeros for some scipy fits
    x = np.maximum(x, 1e-8)
    family = family.lower()

    if family == "gamma":
        shape, loc, scale = stats.gamma.fit(x, floc=0)
        return {
            "family": "gamma",
            "shape": float(shape),
            "loc": float(loc),
            "scale": float(scale),
            "cdf": lambda ages: stats.gamma.cdf(np.asarray(ages, dtype=float), a=shape, loc=0, scale=scale),
            "n": int(x.size),
        }

    if family == "weibull":
        shape, loc, scale = stats.weibull_min.fit(x, floc=0)
        return {
            "family": "weibull",
            "shape": float(shape),
            "loc": float(loc),
            "scale": float(scale),
            "cdf": lambda ages: stats.weibull_min.cdf(np.asarray(ages, dtype=float), c=shape, loc=0, scale=scale),
            "n": int(x.size),
        }

    if family == "lognormal":
        sigma, loc, scale = stats.lognorm.fit(x, floc=0)
        return {
            "family": "lognormal",
            "sigma": float(sigma),
            "loc": float(loc),
            "scale": float(scale),
            "cdf": lambda ages: stats.lognorm.cdf(np.asarray(ages, dtype=float), s=sigma, loc=0, scale=scale),
            "n": int(x.size),
        }

    raise ValueError("family must be one of {'gamma', 'weibull', 'lognormal'}")


def _delay_cdf_at_ages(delay_distribution: Any, ages: np.ndarray) -> np.ndarray:
    """Return F(age) for integer ages >=0."""
    ages = np.asarray(ages, dtype=int)
    if callable(delay_distribution):
        vals = np.asarray(delay_distribution(ages), dtype=float)
        return np.clip(vals, 0.0, 1.0)

    if isinstance(delay_distribution, Mapping):
        if "cdf" in delay_distribution:
            cdf = np.asarray(delay_distribution["cdf"], dtype=float)
        elif "pmf" in delay_distribution:
            cdf = np.cumsum(np.asarray(delay_distribution["pmf"], dtype=float))
        else:
            raise ValueError("Delay distribution mapping must contain 'cdf' or 'pmf'.")
    else:
        arr = np.asarray(delay_distribution, dtype=float)
        if arr.ndim != 1:
            raise ValueError("Delay distribution must be 1D.")
        # interpret as PMF when not explicitly labeled
        cdf = np.cumsum(arr)

    out = np.zeros_like(ages, dtype=float)
    out[ages < 0] = 0.0
    out[ages >= len(cdf)] = 1.0
    mask = (ages >= 0) & (ages < len(cdf))
    out[mask] = cdf[ages[mask]]
    return np.clip(out, 0.0, 1.0)


def _analysis_origin_from_line_list(
    df: pd.DataFrame,
    onset_col: str = "analysis_origin_date",
    fallback_cols: Sequence[str] = ("Date_onset", "Date_of_onset_symp", "Date_of_first_consult", "Date_confirmation", "Date_of_notification"),
    dayfirst: bool = True,
) -> pd.Series:
    if onset_col in df.columns:
        s = _to_datetime(df[onset_col], dayfirst=dayfirst)
        if s.notna().any():
            return s

    cols = [c for c in fallback_cols if c in df.columns]
    if not cols:
        raise ValueError("No suitable onset/origin date columns found.")
    out = pd.Series(pd.NaT, index=df.index, dtype="datetime64[ns]")
    for c in cols:
        cand = _to_datetime(df[c], dayfirst=dayfirst)
        out = out.fillna(cand)
    return out


def _outcome_date_from_line_list(
    df: pd.DataFrame,
    outcome_col: str = "event",
    death_label: str = "death",
    recovery_label: str = "recovery",
    death_date_candidates: Sequence[str] = ("Date_Death", "Date_of_Death"),
    recovery_date_candidates: Sequence[str] = ("Date_Recovered", "Date_hospital_discharge", "Date_disease_ended"),
    dayfirst: bool = True,
) -> Tuple[pd.Series, pd.Series]:
    """Return (event_type, event_date)."""
    if outcome_col not in df.columns and not any(c in df.columns for c in death_date_candidates + recovery_date_candidates):
        raise ValueError("Could not infer outcome/event information.")

    event = None
    event_date = pd.Series(pd.NaT, index=df.index, dtype="datetime64[ns]")

    if outcome_col in df.columns:
        event = df[outcome_col].map(_safe_lower)
    else:
        event = pd.Series("", index=df.index, dtype="object")

    death_date = None
    rec_date = None
    for c in death_date_candidates:
        if c in df.columns:
            death_date = _to_datetime(df[c], dayfirst=dayfirst)
            break
    for c in recovery_date_candidates:
        if c in df.columns:
            rec_date = _to_datetime(df[c], dayfirst=dayfirst)
            break

    if death_date is not None:
        event_date = event_date.fillna(death_date)
    if rec_date is not None:
        # only use rec date where no death date is available
        event_date = event_date.fillna(rec_date)

    if outcome_col in df.columns:
        # normalize common labels
        death_mask = event.isin({"death", "dead", "died", "deceased"})
        rec_mask = event.isin({"recovery", "recovered", "alive", "discharged", "discharge"})
        event = pd.Series(np.where(death_mask, death_label, np.where(rec_mask, recovery_label, event)), index=df.index)
    else:
        # infer from event dates
        death_mask = death_date.notna() if death_date is not None else pd.Series(False, index=df.index)
        rec_mask = rec_date.notna() if rec_date is not None else pd.Series(False, index=df.index)
        event = pd.Series(np.where(death_mask, death_label, np.where(rec_mask, recovery_label, "censored")), index=df.index)

    return event.astype(str), event_date


def _analysis_origin_from_line_list_generic(
    df: pd.DataFrame,
    dayfirst: bool = True,
) -> pd.Series:
    return _analysis_origin_from_line_list(df, dayfirst=dayfirst)


# ---------------------------------------------------------------------------
# Dataset loaders
# ---------------------------------------------------------------------------

def load_drc_consolidated(path: Union[str, Path]) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "reference_date" not in df.columns:
        raise ValueError("Expected a reference_date column in DRC consolidated data.")
    df["reference_date"] = _to_datetime(df["reference_date"])
    for c in ["value"]:
        if c in df.columns:
            df[c] = _coerce_numeric(df[c])
    return df


def load_drc_by_health_zone(path: Union[str, Path]) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "report_date" not in df.columns:
        raise ValueError("Expected a report_date column in DRC by health zone data.")
    df["report_date"] = _to_datetime(df["report_date"])
    for c in df.columns:
        if c not in {"publication_date", "report_date", "country", "province", "health_zone", "source"}:
            df[c] = _coerce_numeric(df[c])
    return df


def load_drc_total(path: Union[str, Path]) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "report_date" not in df.columns:
        raise ValueError("Expected a report_date column in DRC total data.")
    df["report_date"] = _to_datetime(df["report_date"])
    for c in df.columns:
        if c not in {"publication_date", "report_date", "country", "source"}:
            df[c] = _coerce_numeric(df[c])
    return df


def load_rosello2015(path: Union[str, Path]) -> pd.DataFrame:
    df = pd.read_csv(path)
    for c in ["Date_of_onset_symp", "Date_hospital_discharge", "Date_disease_ended", "Date_of_Death", "Date_of_notification", "Date_of_Hospitalisation"]:
        if c in df.columns:
            df[c] = _to_datetime(df[c], dayfirst=True)
    return df


def load_uganda_2022(path: Union[str, Path]) -> pd.DataFrame:
    df = pd.read_csv(path)
    for c in ["Date_onset", "Date_confirmation", "Date_of_first_consult", "Date_hospitalisation", "Date_discharge_hospital", "Date_Death", "Date_Recovered", "Date_isolation", "Date_entry", "Date_last_modified"]:
        if c in df.columns:
            df[c] = _to_datetime(df[c])
    return df



# ---------------------------------------------------------------------------
# Standardization helpers for count tables and line lists
# ---------------------------------------------------------------------------


def standardize_count_table(
    df: pd.DataFrame,
    *,
    date_col: str,
    cases_col: str,
    deaths_col: str,
    recovered_col: Optional[str] = None,
    group_cols: Optional[Sequence[str]] = None,
    fill_missing_dates: bool = True,
    is_cumulative: bool = True,
) -> pd.DataFrame:
    """Standardize a daily count table.

    If is_cumulative=True, missing dates are forward-filled and the supplied
    values are treated as cumulative totals.
    If is_cumulative=False, missing dates are filled with zeros and the supplied
    values are treated as incident counts.
    """
    _require_columns(df, [date_col, cases_col, deaths_col])
    cols = [date_col, cases_col, deaths_col]
    if recovered_col and recovered_col in df.columns:
        cols.append(recovered_col)
    if group_cols:
        cols.extend([c for c in group_cols if c in df.columns])

    out = df[cols].copy()
    out[date_col] = _to_datetime(out[date_col])
    out[cases_col] = _coerce_numeric(out[cases_col])
    out[deaths_col] = _coerce_numeric(out[deaths_col])
    if recovered_col and recovered_col in out.columns:
        out[recovered_col] = _coerce_numeric(out[recovered_col])

    fill_value = np.nan if is_cumulative else 0.0

    if group_cols:
        gcols = [c for c in group_cols if c in out.columns]
        if gcols:
            out = out.sort_values(gcols + [date_col]).reset_index(drop=True)
            if fill_missing_dates:
                chunks = []
                for _, g in out.groupby(gcols, dropna=False, sort=False):
                    g = g.sort_values(date_col)
                    full = pd.DataFrame({date_col: _daily_complete_dates(g, date_col)})
                    full = full.merge(g, on=date_col, how="left")
                    for c in [cases_col, deaths_col, recovered_col]:
                        if c and c in full.columns:
                            full[c] = full[c].fillna(fill_value)
                    if is_cumulative:
                        for c in [cases_col, deaths_col, recovered_col]:
                            if c and c in full.columns:
                                full[c] = full[c].ffill().fillna(0.0)
                    else:
                        for c in [cases_col, deaths_col, recovered_col]:
                            if c and c in full.columns:
                                full[c] = full[c].fillna(0.0)
                    for gc in gcols:
                        full[gc] = g[gc].iloc[0]
                    chunks.append(full)
                out = pd.concat(chunks, ignore_index=True)
    else:
        out = out.sort_values(date_col).reset_index(drop=True)
        if fill_missing_dates:
            full = pd.DataFrame({date_col: _daily_complete_dates(out, date_col)})
            full = full.merge(out, on=date_col, how="left")
            for c in [cases_col, deaths_col, recovered_col]:
                if c and c in full.columns:
                    full[c] = full[c].fillna(fill_value)
            if is_cumulative:
                for c in [cases_col, deaths_col, recovered_col]:
                    if c and c in full.columns:
                        full[c] = full[c].ffill().fillna(0.0)
            else:
                for c in [cases_col, deaths_col, recovered_col]:
                    if c and c in full.columns:
                        full[c] = full[c].fillna(0.0)
            out = full

    return out
def standardize_line_list(
    df: pd.DataFrame,
    *,
    onset_col: Optional[str] = None,
    outcome_col: Optional[str] = None,
    outcome_date_col: Optional[str] = None,
    death_label: str = "death",
    recovery_label: str = "recovery",
    dayfirst: bool = True,
) -> pd.DataFrame:
    """Return a standardized linelist with start_date, outcome_date, event."""
    work = df.copy()

    # Already-standardized input
    if {"start_date", "outcome_date", "event"}.issubset(work.columns):
        out = work[["start_date", "outcome_date", "event"]].copy()
        out["start_date"] = _to_datetime(out["start_date"], dayfirst=dayfirst)
        out["outcome_date"] = _to_datetime(out["outcome_date"], dayfirst=dayfirst)
        out["event"] = out["event"].map(_safe_lower)
        death_mask = out["event"].isin({"death", "dead", "died", "deceased"})
        rec_mask = out["event"].isin({"recovery", "recovered", "alive", "discharged", "discharge"})
        out.loc[death_mask, "event"] = death_label
        out.loc[rec_mask, "event"] = recovery_label
        out.loc[~(death_mask | rec_mask), "event"] = "censored"

        # Exclude impossible records: outcome before onset
        valid_outcome = out["outcome_date"].isna() | (out["outcome_date"] >= out["start_date"])
        return out.loc[valid_outcome].dropna(subset=["start_date"]).copy()

    if onset_col is None:
        onset_col = _first_existing_column(work, [
            "analysis_origin_date",
            "Date_onset",
            "Date_of_onset_symp",
            "onset_date",
            "date_onset",
            "Date_of_first_consult",
            "Date_confirmation",
            "Date_of_notification",
        ])
    if onset_col is None:
        raise ValueError("Could not identify an onset/start date column.")

    start = _to_datetime(work[onset_col], dayfirst=dayfirst)

    if outcome_col is not None and outcome_col in work.columns:
        event = work[outcome_col].map(_safe_lower)
    else:
        event = pd.Series("", index=work.index, dtype="object")

    if outcome_date_col is not None and outcome_date_col in work.columns:
        outcome_date = _to_datetime(work[outcome_date_col], dayfirst=dayfirst)
    else:
        # infer from common fields
        event, outcome_date = _outcome_date_from_line_list(
            work,
            outcome_col=outcome_col or "event",
            death_label=death_label,
            recovery_label=recovery_label,
            dayfirst=dayfirst,
        )

    # If event is missing, infer from dates
    if outcome_col is None or outcome_col not in work.columns:
        event = pd.Series(np.where(outcome_date.notna(), "resolved", "censored"), index=work.index)

    standardized = pd.DataFrame(
        {
            "start_date": start,
            "outcome_date": outcome_date,
            "event": event.astype(str),
        }
    )
    standardized = standardized.dropna(subset=["start_date"]).copy()

    # normalize labels
    standardized["event"] = standardized["event"].map(_safe_lower)
    death_mask = standardized["event"].isin({"death", "dead", "died", "deceased"})
    rec_mask = standardized["event"].isin({"recovery", "recovered", "alive", "discharged", "discharge"})
    standardized.loc[death_mask, "event"] = death_label
    standardized.loc[rec_mask, "event"] = recovery_label
    standardized.loc[~(death_mask | rec_mask), "event"] = "censored"

    # Exclude impossible records: outcome before onset
    valid_outcome = standardized["outcome_date"].isna() | (standardized["outcome_date"] >= standardized["start_date"])
    standardized = standardized.loc[valid_outcome].copy()

    return standardized


# ---------------------------------------------------------------------------
# Delay distributions from individual-level data
# ---------------------------------------------------------------------------



def estimate_delay_distribution_from_dates(
    onset_dates: Sequence[Any],
    outcome_dates: Sequence[Any],
    *,
    dayfirst: bool = True,
    family: str = "gamma",
) -> Dict[str, Any]:
    onset = pd.to_datetime(pd.Series(list(onset_dates)), errors="coerce", dayfirst=dayfirst)
    outcome = pd.to_datetime(pd.Series(list(outcome_dates)), errors="coerce", dayfirst=dayfirst)

    delays = (outcome - onset).dt.days.to_numpy()
    delays = delays[np.isfinite(delays) & (delays >= 0)]

    return fit_parametric_delay_distribution(delays, family=family)


def estimate_delay_distributions_from_individual_data(
    df: pd.DataFrame,
    *,
    onset_col: str,
    outcome_date_col: str,
    outcome_col: str,
    death_label: str = "death",
    recovery_label: str = "recovery",
    dayfirst: bool = True,
    family: str = "gamma",
) -> Dict[str, Dict[str, Any]]:
    """Estimate parametric delay distributions for death and recovery."""
    _require_columns(df, [onset_col, outcome_date_col, outcome_col])

    work = df[[onset_col, outcome_date_col, outcome_col]].copy()
    work[onset_col] = _to_datetime(work[onset_col], dayfirst=dayfirst)
    work[outcome_date_col] = _to_datetime(work[outcome_date_col], dayfirst=dayfirst)
    work[outcome_col] = work[outcome_col].map(_safe_lower)

    out = {}
    for label in [death_label, recovery_label]:
        sub = work.loc[work[outcome_col].eq(label)].dropna(subset=[onset_col, outcome_date_col]).copy()
        if sub.empty:
            continue

        out[label] = estimate_delay_distribution_from_dates(
            sub[onset_col],
            sub[outcome_date_col],
            dayfirst=dayfirst,
            family=family,
        )

    if not out:
        raise ValueError("No valid delays found to estimate any distribution.")

    return out
def _estimate_delay_distribution_for_analysis_date(
    df: pd.DataFrame,
    *,
    analysis_date: pd.Timestamp,
    onset_col: str = "start_date",
    outcome_date_col: str = "outcome_date",
    outcome_col: str = "event",
    death_label: str = "death",
    recovery_label: str = "recovery",
    dayfirst: bool = True,
    family: str = "gamma",
) -> Dict[str, Dict[str, Any]]:
    """Estimate delay distributions using only information available by analysis_date."""
    work = df[[onset_col, outcome_date_col, outcome_col]].copy()
    work[onset_col] = _to_datetime(work[onset_col], dayfirst=dayfirst)
    work[outcome_date_col] = _to_datetime(work[outcome_date_col], dayfirst=dayfirst)
    work[outcome_col] = work[outcome_col].map(_safe_lower)

    cutoff = pd.to_datetime(analysis_date)
    observed = work.loc[
        work[onset_col].notna()
        & (work[onset_col] <= cutoff)
        & work[outcome_date_col].notna()
        & (work[outcome_date_col] <= cutoff)
    ].copy()

    if observed.empty:
        raise ValueError("No resolved outcomes observed by analysis_date.")

    return estimate_delay_distributions_from_individual_data(
        observed,
        onset_col=onset_col,
        outcome_date_col=outcome_date_col,
        outcome_col=outcome_col,
        death_label=death_label,
        recovery_label=recovery_label,
        dayfirst=dayfirst,
        family=family,
    )



# ---------------------------------------------------------------------------
# Estimators
# ---------------------------------------------------------------------------

def cfr_naive(deaths: Union[pd.Series, np.ndarray, float, int], cases: Union[pd.Series, np.ndarray, float, int]) -> float:
    deaths = float(np.nansum(deaths))
    cases = float(np.nansum(cases))
    (lo,hi) = clopper_pearson_ci(deaths,cases)
    return {"estimate": deaths / cases, 
            "lower_ci":lo,
            "upper_ci": hi} if cases > 0 else np.nan


def cfr_resolved_cohort(deaths: Union[pd.Series, np.ndarray, float, int], recovered: Union[pd.Series, np.ndarray, float, int]) -> float:
    deaths = float(np.nansum(deaths))
    recovered = float(np.nansum(recovered))
    denom = deaths + recovered
    (lo,hi) = clopper_pearson_ci(deaths,denom)
    return {"estimate": deaths / denom, 
            "lower_ci":lo,
            "upper_ci": hi} if denom > 0 else np.nan




def cfr_delay_adjusted_nishiura(
    deaths: Sequence[float],
    cases: Sequence[float],
    delay_distribution: None,
    *,
    poisson_threshold: int = 1000,
) -> Dict[str, float]:
    """
    Profile-likelihood CI wrapper for delay-adjusted Nishiura CFR.

    This uses:
      total_cases   = sum(cases)
      total_deaths  = sum(deaths)
      total_outcomes = sum(cases * F(delay_age))

    where F(delay_age) is the delay CDF at each age.
    
    Calculates the estimate and upper and lower boundaries of confidence intervals.

    Estimates and confidence intervals 
    """
    deaths = np.asarray(deaths, dtype=float).reshape(-1)
    cases = np.asarray(cases, dtype=float).reshape(-1)

    if deaths.size != cases.size:
        raise ValueError("deaths and cases must have the same length.")
    if deaths.size == 0:
        return {
            "estimate": np.nan,
            "lower_ci": np.nan,
            "upper_ci": np.nan,
        }

    deaths = np.nan_to_num(deaths, nan=0.0)
    cases = np.nan_to_num(cases, nan=0.0)

    deaths = np.maximum(deaths, 0.0)
    cases = np.maximum(cases, 0.0)

    ages = np.arange(cases.size - 1, -1, -1, dtype=int)
    
    delay_dist_here = delay_distribution
    known_outcome_prob = _delay_cdf_at_ages(delay_dist_here, ages)

    total_cases = float(np.sum(cases))
    total_deaths = float(np.sum(deaths))
    total_outcomes = float(np.sum(cases * known_outcome_prob))

    p_mid = (total_deaths / round(total_outcomes)) if round(total_outcomes) > 0 else np.nan

    return estimate_severity_profile_likelihood(
        total_cases=total_cases,
        total_deaths=total_deaths,
        total_outcomes=total_outcomes,
        poisson_threshold=poisson_threshold,
        p_mid=p_mid,
    )




def _prepare_individual_time_data(
    df: pd.DataFrame,
    *,
    start_col: str = "start_date",
    outcome_date_col: str = "outcome_date",
    event_col: str = "event",
    analysis_date: Optional[pd.Timestamp] = None,
    death_label: str = "death",
    recovery_label: str = "recovery",
) -> pd.DataFrame:
    work = df[[start_col, outcome_date_col, event_col]].copy()
    work[start_col] = pd.to_datetime(work[start_col], errors="coerce")
    work[outcome_date_col] = pd.to_datetime(work[outcome_date_col], errors="coerce")
    work[event_col] = work[event_col].map(_safe_lower)
    work = work.dropna(subset=[start_col]).copy()

    if analysis_date is None:
        analysis_date = work[outcome_date_col].max()
        if pd.isna(analysis_date):
            analysis_date = work[start_col].max()

    analysis_date = pd.to_datetime(analysis_date)

    # retain only subjects whose start date is on/before analysis date
    work = work.loc[work[start_col] <= analysis_date].copy()

    # event by analysis date only if observed by then
    observed = work[outcome_date_col].notna() & (work[outcome_date_col] <= analysis_date)
    work["time"] = (np.minimum(
        work[outcome_date_col].fillna(analysis_date).values.astype("datetime64[ns]"),
        np.datetime64(analysis_date)
    ) - work[start_col].values.astype("datetime64[ns]")) / np.timedelta64(1, "D")
    work["time"] = work["time"].astype(float)

    event = np.where(observed, work[event_col].to_numpy(), "censored")
    event = pd.Series(event, index=work.index).map(_safe_lower)
    event = event.where(event.isin({death_label, recovery_label}), "censored")
    work["event"] = event
    return work[["time", "event"]].copy()

# options for CIs: bootstrap, wald, wald after log log transformation, https://link.springer.com/article/10.1007/s10985-018-09458-6#Sec13
def cfr_competing_risks(
    df: pd.DataFrame,
    *,
    time_col: str = "time",
    event_col: str = "event",
    death_label: str = "death",
    recovery_label: str = "recovery",
    alpha: float = 0.05,
    untrans: bool = False,
    return_ci: bool = True,
) -> Union[float, Dict[str, Any]]:
    """
    Exact Aalen-Johansen cumulative incidence for death in the presence of recovery
    as a competing event.

    The point estimate is the Aalen-Johansen CIF for death. When return_ci=True,
    a Greenwood-type Wald interval is returned using a delta-method covariance
    recursion for the state probabilities (survival, death CIF, recovery CIF).
    """
    _require_columns(df, [time_col, event_col])
    work = df[[time_col, event_col]].copy()
    work[time_col] = _coerce_numeric(work[time_col])
    work[event_col] = work[event_col].map(_safe_lower)
    work = work.dropna(subset=[time_col]).copy()

    times = work[time_col].to_numpy(dtype=float)
    events = work[event_col].to_numpy(dtype=str)

    valid = np.isfinite(times)
    times = times[valid]
    events = events[valid]

    if times.size == 0:
        return np.nan if not return_ci else {
            "estimate": np.nan,
            "lower_ci": np.nan,
            "upper_ci": np.nan,
            "se_cfr": np.nan,
            "n_cases": 0,
            "n_deaths": 0,
            "n_recoveries": 0,
            "variance_method": "greenwood",
            "ci_method": "raw" if untrans else "logit",
        }

    # Only event times (death or recovery)
    event_times = np.unique(times[np.isin(events, [death_label, recovery_label])])
    if event_times.size == 0:
        return np.nan if not return_ci else {
           "estimate": np.nan,
            "lower_ci": np.nan,
            "upper_ci": np.nan,
            "se_cfr": np.nan,
            "n_cases": int(times.size),
            "n_deaths": 0,
            "n_recoveries": 0,
            "variance_method": "greenwood",
            "ci_method": "raw" if untrans else "logit",
        }

    # State vector: [S, F_death, F_recovery]
    x = np.array([1.0, 0.0, 0.0], dtype=float)
    Sigma = np.zeros((3, 3), dtype=float)

    n_deaths = 0
    n_recoveries = 0

    for t in np.sort(event_times):
        at_risk = float(np.sum(times >= t))
        if at_risk <= 0:
            continue

        d_death = float(np.sum((times == t) & (events == death_label)))
        d_rec = float(np.sum((times == t) & (events == recovery_label)))
        d_all = d_death + d_rec
        if d_all <= 0:
            continue

        n_deaths += int(d_death)
        n_recoveries += int(d_rec)

        a = d_death / at_risk
        b = d_rec / at_risk
        c = max(0.0, 1.0 - a - b)
        S_prev = x[0]

        # Jacobian wrt previous state probabilities
        A = np.array(
            [
                [c, 0.0, 0.0],
                [a, 1.0, 0.0],
                [b, 0.0, 1.0],
            ],
            dtype=float,
        )

        # Jacobian wrt current step event probabilities [a, b]
        B = np.array(
            [
                [-S_prev, -S_prev],
                [S_prev, 0.0],
                [0.0, S_prev],
            ],
            dtype=float,
        )

        # Multinomial Greenwood-type covariance for (a, b)
        V = np.array(
            [
                [a * (1.0 - a), -a * b],
                [-a * b, b * (1.0 - b)],
            ],
            dtype=float,
        ) / at_risk

        # Recursion for covariance of [S, CIF_death, CIF_recovery]
        Sigma = A @ Sigma @ A.T + B @ V @ B.T

        # Update state probabilities
        x = np.array(
            [
                S_prev * c,
                x[1] + S_prev * a,
                x[2] + S_prev * b,
            ],
            dtype=float,
        )

    estimate = float(x[1])
    variance = float(Sigma[1, 1])
    se = float(np.sqrt(max(variance, 0.0))) if np.isfinite(variance) else np.nan

    if not return_ci:
        return estimate

    z = float(stats.norm.ppf(1.0 - alpha / 2.0))

    if untrans:
        lower = estimate - z * se if np.isfinite(se) else np.nan
        upper = estimate + z * se if np.isfinite(se) else np.nan
        lower = float(np.clip(lower, 0.0, 1.0)) if np.isfinite(lower) else np.nan
        upper = float(np.clip(upper, 0.0, 1.0)) if np.isfinite(upper) else np.nan
        ci_method = "raw"
    else:
        eps = 1e-12
        p = float(np.clip(estimate, eps, 1.0 - eps))
        if np.isfinite(se):
            var_logit = variance / ((p * (1.0 - p)) ** 2)
            se_logit = float(np.sqrt(max(var_logit, 0.0)))
            lower = float(expit(logit(p) - z * se_logit))
            upper = float(expit(logit(p) + z * se_logit))
        else:
            lower = np.nan
            upper = np.nan
        ci_method = "logit"

    return {
        "estimate": estimate,
        "lower_ci": lower,
        "upper_ci": upper,
        "se_cfr": se,
        "n_cases": int(times.size),
        "n_deaths": int(n_deaths),
        "n_recoveries": int(n_recoveries),
        "variance_method": "greenwood",
        "ci_method": ci_method,
    }

def cfr_competing_risk1(
    df: pd.DataFrame,
    *,
    time_col: str = "time",
    event_col: str = "event",
    death_label: str = "death",
    recovery_label: str = "recovery",
) -> float:
    """
    Exact Aalen-Johansen cumulative incidence for death in the presence of recovery
    as a competing event.
    """
    _require_columns(df, [time_col, event_col])
    work = df[[time_col, event_col]].copy()
    work[time_col] = _coerce_numeric(work[time_col])
    work[event_col] = work[event_col].map(_safe_lower)
    work = work.dropna(subset=[time_col]).copy()

    times = work[time_col].to_numpy(dtype=float)
    events = work[event_col].to_numpy(dtype=str)

    valid = np.isfinite(times)
    times = times[valid]
    events = events[valid]

    if times.size == 0:
        return np.nan

    # Only event times (death or recovery)
    event_times = np.unique(times[np.isin(events, [death_label, recovery_label])])
    if event_times.size == 0:
        return np.nan

    s = 1.0
    cif_death = 0.0

    for t in np.sort(event_times):
        at_risk = np.sum(times >= t)
        if at_risk <= 0:
            continue
        d_death = np.sum((times == t) & (events == death_label))
        d_rec = np.sum((times == t) & (events == recovery_label))
        d_all = d_death + d_rec
        cif_death += s * (d_death / at_risk)
        s *= (1.0 - d_all / at_risk)

    return float(cif_death)



def _greenwood_var_survival(y, d_all, s):
    """
    Greenwood variance for the composite survival curve S(t).
    y     : number at risk at each event time
    d_all : total failures (death + recovery) at each event time
    s     : survival values S(t) at each event time
    """
    cum = 0.0
    var = np.empty_like(s, dtype=float)

    for i, (yi, di, si) in enumerate(zip(y, d_all, s)):
        if yi <= 0:
            var[i] = np.nan
            continue
        if di > 0:
            if yi > di:
                cum += di / (yi * (yi - di))
                var[i] = (si ** 2) * cum
            else:
                # Boundary case: all at risk fail at this time.
                # Greenwood becomes unstable here; return NaN.
                var[i] = np.nan
        else:
            var[i] = (si ** 2) * cum

    return var


def cfr_ghani_2005_km(
    df: pd.DataFrame,
    *,
    time_col: str = "time",
    event_col: str = "event",
    death_label: str = "death",
    recovery_label: str = "recovery",
    greenwood: bool = False,
    untrans: bool = False,
    alpha: float = 0.05,
) -> dict:
    """
    Ghani et al. KM-like CFR estimator with Stata-style confidence intervals.

    Returns
    -------
    dict with keys:
      estimate, lower_ci, upper_ci, se_cfr, theta0, theta1,
      n_cases, n_dead, n_recovered, ci_method
    """
    _require_columns(df, [time_col, event_col])

    work = df[[time_col, event_col]].copy()
    work[time_col] = _coerce_numeric(work[time_col])
    work[event_col] = work[event_col].map(_safe_lower)
    work = work.dropna(subset=[time_col]).copy()

    if work.empty:
        return {
            "estimate": np.nan,
            "lower_ci": np.nan,
            "upper_ci": np.nan,
            "se_cfr": np.nan,
            "theta0": np.nan,
            "theta1": np.nan,
            "n_cases": 0,
            "n_dead": 0,
            "n_recovered": 0,
            "ci_method": "greenwood" if greenwood else "alt",
        }

    times = work[time_col].to_numpy(dtype=float)
    events = work[event_col].to_numpy(dtype=str)

    valid = np.isfinite(times)
    times = times[valid]
    events = events[valid]

    # Unique times where either death or recovery occurs
    event_times = np.sort(np.unique(times[np.isin(events, [death_label, recovery_label])]))
    if event_times.size == 0:
        return {
            "estimate": np.nan,
            "lower_ci": np.nan,
            "upper_ci": np.nan,
            "se_cfr": np.nan,
            "theta0": np.nan,
            "theta1": np.nan,
            "n_cases": int(len(times)),
            "n_dead": 0,
            "n_recovered": 0,
            "ci_method": "greenwood" if greenwood else "alt",
        }

    y = []
    d_death = []
    d_rec = []
    d_all = []
    S = []
    h_dead = []
    h_rec = []

    theta0 = 0.0
    theta1 = 0.0
    s_prev = 1.0

    for t in event_times:
        yi = float(np.sum(times >= t))
        dd = float(np.sum((times == t) & (events == death_label)))
        dr = float(np.sum((times == t) & (events == recovery_label)))
        da = dd + dr

        y.append(yi)
        d_death.append(dd)
        d_rec.append(dr)
        d_all.append(da)

        hd = dd / yi if yi > 0 else np.nan
        hr = dr / yi if yi > 0 else np.nan
        h_dead.append(hd)
        h_rec.append(hr)

        # cumulative incidence contributions
        theta0 += s_prev * hd if np.isfinite(hd) else 0.0
        theta1 += s_prev * hr if np.isfinite(hr) else 0.0

        s_t = s_prev * (1.0 - da / yi) if yi > 0 else np.nan
        S.append(s_t)
        s_prev = s_t if np.isfinite(s_t) else s_prev

    y = np.asarray(y, dtype=float)
    d_death = np.asarray(d_death, dtype=float)
    d_rec = np.asarray(d_rec, dtype=float)
    d_all = np.asarray(d_all, dtype=float)
    S = np.asarray(S, dtype=float)
    h_dead = np.asarray(h_dead, dtype=float)
    h_rec = np.asarray(h_rec, dtype=float)

    denom = theta0 + theta1
    cfr = theta0 / denom if denom > 0 else np.nan

    # Stata's nstar = (Ntot + Nevent) / 2
    n_total = float(len(times))
    n_event = float(np.sum(d_all))
    nstar = (n_total + n_event) / 2.0 if n_total > 0 else np.nan

    if greenwood:
        # Approximation to Stata's sts gen se(s) for the composite survival
        varS = _greenwood_var_survival(y, d_all, S)
        OM = np.diag(varS)
        for j in range(len(event_times)):
            for k in range(j):
                if np.isfinite(varS[k]) and np.isfinite(S[j]) and S[k] > 0:
                    OM[j, k] = varS[k] * S[j] / S[k]
                    OM[k, j] = OM[j, k]
                else:
                    OM[j, k] = np.nan
                    OM[k, j] = np.nan
        ci_method = "greenwood_logit" if not untrans else "greenwood_raw"
    else:
        T = S
        OM = np.outer(T, 1.0 - T) / nstar
        for k in range(len(event_times)):
            for j in range(k + 1, len(event_times)):
                OM[k, j] = T[k] * (1.0 - T[j]) / nstar
                OM[j, k] = OM[k, j]
        ci_method = "alt_logit" if not untrans else "alt_raw"

    hv_dead = h_dead.reshape(-1, 1)
    hv_rec = h_rec.reshape(-1, 1)

    B_dead = float(hv_dead.T @ OM @ hv_dead)
    B_rec = float(hv_rec.T @ OM @ hv_rec)
    cov01 = float(hv_dead.T @ OM @ hv_rec)

    A_dead = (S ** 2) * h_dead / y
    A_rec = (S ** 2) * h_rec / y

    var_dead = float(np.nansum(A_dead) + B_dead)
    var_rec = float(np.nansum(A_rec) + B_rec)

    var_cfr = (
        (theta1 ** 2) * var_dead
        + (theta0 ** 2) * var_rec
        - 2.0 * theta0 * theta1 * cov01
    ) / ((theta0 + theta1) ** 4) if denom > 0 else np.nan

    se_cfr = float(np.sqrt(var_cfr)) if np.isfinite(var_cfr) and var_cfr >= 0 else np.nan

    z = float(norm.ppf(1.0 - alpha / 2.0))

    if untrans:
        lower = cfr - z * se_cfr if np.isfinite(se_cfr) else np.nan
        upper = cfr + z * se_cfr if np.isfinite(se_cfr) else np.nan
    else:
        # Stata default: logit transform
        eps = 1e-12
        p = float(np.clip(cfr, eps, 1.0 - eps))
        if theta0 > 0 and theta1 > 0 and np.isfinite(var_dead) and np.isfinite(var_rec):
            var_logit = (
                var_dead / (theta0 ** 2)
                + var_rec / (theta1 ** 2)
                - 2.0 * cov01 / (theta0 * theta1)
            )
            se_logit = float(np.sqrt(var_logit)) if np.isfinite(var_logit) and var_logit >= 0 else np.nan
            if np.isfinite(se_logit):
                lower = float(expit(logit(p) - z * se_logit))
                upper = float(expit(logit(p) + z * se_logit))
            else:
                lower = np.nan
                upper = np.nan
        else:
            lower = np.nan
            upper = np.nan

    return {
        "estimate": float(cfr),
        "lower_ci": float(lower) if np.isfinite(lower) else np.nan,
        "upper_ci": float(upper) if np.isfinite(upper) else np.nan,
        "se_cfr": float(se_cfr) if np.isfinite(se_cfr) else np.nan,
        "theta0": float(theta0),
        "theta1": float(theta1),
        "n_cases": int(n_total),
        "n_dead": int(np.sum(d_death)),
        "n_recovered": int(np.sum(d_rec)),
        "ci_method": ci_method,
    }


def cfr_ghani_2005_km1(
    df: pd.DataFrame,
    *,
    time_col: str = "time",
    event_col: str = "event",
    death_label: str = "death",
    recovery_label: str = "recovery",
) -> float:
    """
    Ghani et al. (2005) adapted Kaplan-Meier CFR estimator.

    Returns a scalar CFR estimate.
    """
    _require_columns(df, [time_col, event_col])

    work = df[[time_col, event_col]].copy()
    work[time_col] = _coerce_numeric(work[time_col])
    work[event_col] = work[event_col].map(_safe_lower)
    work = work.dropna(subset=[time_col]).copy()

    if work.empty:
        return np.nan

    times = work[time_col].to_numpy(dtype=float)
    events = work[event_col].to_numpy(dtype=str)

    valid = np.isfinite(times)
    times = times[valid]
    events = events[valid]

    event_times = np.unique(times[np.isin(events, [death_label, recovery_label])])
    if event_times.size == 0:
        return np.nan

    s = 1.0          # composite survival Θ
    theta0 = 0.0     # death probability contribution
    theta1 = 0.0     # recovery probability contribution

    for t in np.sort(event_times):
        at_risk = np.sum(times >= t)
        if at_risk <= 0:
            continue

        d_death = np.sum((times == t) & (events == death_label))
        d_rec = np.sum((times == t) & (events == recovery_label))
        d_all = d_death + d_rec

        theta0 += s * (d_death / at_risk)
        theta1 += s * (d_rec / at_risk)

        s *= (1.0 - d_all / at_risk)

    denom = theta0 + theta1
    return float(theta0 / denom) if denom > 0 else np.nan


def _mixture_negloglik(params: np.ndarray, times: np.ndarray, events: np.ndarray, family: str) -> float:
    """
    Negative log-likelihood for the Ghani et al. mixture/cure model.

    Parameterization:
      p = expit(logit_p)   [probability of death]
      outcome-specific time-to-event distributions conditioned on death/recovery
      censored likelihood: p*S_d(t) + (1-p)*S_r(t)
    """
    logit_p = params[0]
    p = expit(logit_p)

    if family == "gamma":
        log_kd, log_thetad, log_kr, log_thetar = params[1:]
        kd, thetad = np.exp(log_kd), np.exp(log_thetad)
        kr, thetar = np.exp(log_kr), np.exp(log_thetar)

        def pdf_d(x):
            return stats.gamma.pdf(x, a=kd, scale=thetad)

        def sf_d(x):
            return stats.gamma.sf(x, a=kd, scale=thetad)

        def pdf_r(x):
            return stats.gamma.pdf(x, a=kr, scale=thetar)

        def sf_r(x):
            return stats.gamma.sf(x, a=kr, scale=thetar)

    elif family == "weibull":
        log_kd, log_lamd, log_kr, log_lamr = params[1:]
        kd, lamd = np.exp(log_kd), np.exp(log_lamd)
        kr, lamr = np.exp(log_kr), np.exp(log_lamr)

        def pdf_d(x):
            return stats.weibull_min.pdf(x, c=kd, scale=lamd)

        def sf_d(x):
            return stats.weibull_min.sf(x, c=kd, scale=lamd)

        def pdf_r(x):
            return stats.weibull_min.pdf(x, c=kr, scale=lamr)

        def sf_r(x):
            return stats.weibull_min.sf(x, c=kr, scale=lamr)

    elif family == "lognormal":
        log_sigmad, mud, log_sigmar, mur = params[1:]
        sigmad = np.exp(log_sigmad)
        sigmar = np.exp(log_sigmar)

        def pdf_d(x):
            return stats.lognorm.pdf(x, s=sigmad, scale=np.exp(mud))

        def sf_d(x):
            return stats.lognorm.sf(x, s=sigmad, scale=np.exp(mud))

        def pdf_r(x):
            return stats.lognorm.pdf(x, s=sigmar, scale=np.exp(mur))

        def sf_r(x):
            return stats.lognorm.sf(x, s=sigmar, scale=np.exp(mur))

    else:
        raise ValueError("family must be one of {'gamma', 'weibull', 'lognormal'}")

    eps = 1e-300
    ll = 0.0
    death_mask = events == "death"
    rec_mask = events == "recovery"
    cens_mask = ~(death_mask | rec_mask)

    if np.any(death_mask):
        td = np.maximum(times[death_mask], 1e-12)
        ll += np.sum(np.log(p + eps) + np.log(pdf_d(td) + eps))
    if np.any(rec_mask):
        tr = np.maximum(times[rec_mask], 1e-12)
        ll += np.sum(np.log(1.0 - p + eps) + np.log(pdf_r(tr) + eps))
    if np.any(cens_mask):
        tc = np.maximum(times[cens_mask], 1e-12)
        mix = p * sf_d(tc) + (1.0 - p) * sf_r(tc)
        ll += np.sum(np.log(mix + eps))

    return -float(ll)



def _nuisance_bounds_for_family(family: str):
    if family in {"gamma", "weibull"}:
        return [(-8, 8), (-8, 8), (-8, 8), (-8, 8)]
    if family == "lognormal":
        return [(-8, 8), (-10, 10), (-8, 8), (-10, 10)]
    raise ValueError("family must be one of {'gamma', 'weibull', 'lognormal'}")


def _default_nuisance_start(full_x: np.ndarray) -> np.ndarray:
    # full_x = [logit_p, nuisance...]
    return np.asarray(full_x[1:], dtype=float).copy()


def _profile_loglik_for_p(times, events, family, p_fixed, nuisance_start, maxiter=5000):
    """
    Profile log-likelihood at a fixed CFR value p_fixed:
    maximize over nuisance parameters only.
    """
    p_fixed = float(np.clip(p_fixed, 1e-12, 1 - 1e-12))
    logit_p_fixed = float(logit(p_fixed))
    bounds = _nuisance_bounds_for_family(family)

    def obj(nuis):
        params = np.concatenate(([logit_p_fixed], np.asarray(nuis, dtype=float)))
        return _mixture_negloglik(params, times, events, family)

    res = minimize(
        obj,
        x0=np.asarray(nuisance_start, dtype=float),
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": maxiter},
    )
    return -float(res.fun), res


def _lr_ci_for_mixture(times, events, family, full_res, alpha=0.05, maxiter=5000):
    """
    Likelihood-ratio CI for the CFR p in a parametric mixture model.
    """
    p_hat = float(expit(full_res.x[0]))
    ll_max = -float(full_res.fun)
    crit = float(chi2.ppf(1.0 - alpha, df=1))  # 3.84 for alpha=0.05
    target = ll_max - 0.5 * crit

    nuisance_start = _default_nuisance_start(full_res.x)
    cache = {}

    def prof_ll(p):
        p = float(np.clip(p, 1e-12, 1 - 1e-12))
        key = round(p, 12)
        if key not in cache:
            ll, _ = _profile_loglik_for_p(
                times, events, family, p, nuisance_start, maxiter=maxiter
            )
            cache[key] = ll
        return cache[key]

    def g(p):
        return prof_ll(p) - target

    # If the optimum itself is below target something is badly wrong,
    # but guard anyway.
    if g(p_hat) < 0:
        return np.nan, np.nan

    eps = 1e-8

    # Search lower side
    lower = np.nan
    if p_hat > eps:
        grid = np.linspace(eps, p_hat, 25)
        vals = [g(p) for p in grid]
        lo_idx = None
        for i in range(len(grid) - 1):
            if vals[i] < 0 <= vals[i + 1]:
                lo_idx = i
                break
        if lo_idx is not None:
            sol = root_scalar(lambda p: g(p), bracket=(grid[lo_idx], grid[lo_idx + 1]), method="brentq")
            lower = float(sol.root)
        else:
            lower = eps if vals[0] >= 0 else np.nan

    # Search upper side
    upper = np.nan
    if p_hat < 1 - eps:
        grid = np.linspace(p_hat, 1 - eps, 25)
        vals = [g(p) for p in grid]
        hi_idx = None
        for i in range(len(grid) - 1):
            if vals[i] >= 0 > vals[i + 1]:
                hi_idx = i
                break
        if hi_idx is not None:
            sol = root_scalar(lambda p: g(p), bracket=(grid[hi_idx], grid[hi_idx + 1]), method="brentq")
            upper = float(sol.root)
        else:
            upper = 1 - eps if vals[-1] >= 0 else np.nan

    return lower, upper


def cfr_parametric_mixture(
    df: pd.DataFrame,
    *,
    time_col: str = "time",
    event_col: str = "event",
    family: str = "gamma",
    death_label: str = "death",
    recovery_label: str = "recovery",
    start_params=None,
    maxiter: int = 5000,
    alpha: float = 0.05,
) -> dict:
    _require_columns(df, [time_col, event_col])
    work = df[[time_col, event_col]].copy()
    work[time_col] = _coerce_numeric(work[time_col])
    work[event_col] = work[event_col].map(_safe_lower)
    work = work.dropna(subset=[time_col]).copy()

    times = work[time_col].to_numpy(dtype=float)
    events = work[event_col].to_numpy(dtype=str)
    valid = np.isfinite(times)
    times = np.maximum(times[valid], 1e-12)
    events = events[valid]

    if np.sum(events == death_label) + np.sum(events == recovery_label) == 0:
        return {
            "estimate": np.nan,
            "lower_ci": np.nan,
            "upper_ci": np.nan,
            "family": family,
            "success": False,
            "message": "No resolved outcomes available for mixture model.",
            "result": None,
        }

    p0 = np.mean(events == death_label)
    p0 = min(max(p0, 1e-4), 1 - 1e-4)
    med = float(np.median(times)) if np.isfinite(np.median(times)) else 1.0
    med = max(med, 1e-3)

    if start_params is None:
        if family in {"gamma", "weibull"}:
            start_params = [np.log(p0 / (1.0 - p0)), 0.0, np.log(med), 0.0, np.log(med)]
        elif family == "lognormal":
            start_params = [np.log(p0 / (1.0 - p0)), np.log(0.5), np.log(med), np.log(0.5), np.log(med)]
        else:
            raise ValueError("family must be one of {'gamma', 'weibull', 'lognormal'}")

    x0 = np.asarray(start_params, dtype=float)
    bounds = [(-10, 10), *_nuisance_bounds_for_family(family)]

    res = minimize(
        _mixture_negloglik,
        x0,
        args=(times, events, family),
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": maxiter},
    )

    p_hat = float(expit(res.x[0]))
    lower, upper = _lr_ci_for_mixture(times, events, family, res, alpha=alpha, maxiter=maxiter)

    out = {
        "estimate": p_hat,
        "lower_ci": lower,
        "upper_ci": upper,
        "family": family,
        "success": bool(res.success),
        "message": str(res.message),
        "result": res,
    }

    return out

def cfr_parametric_mixture1(
    df: pd.DataFrame,
    *,
    time_col: str = "time",
    event_col: str = "event",
    family: str = "gamma",
    death_label: str = "death",
    recovery_label: str = "recovery",
    start_params: Optional[Sequence[float]] = None,
    maxiter: int = 5000,
) -> Dict[str, Any]:
    """
    Ghani et al. parametric mixture model for death/recovery.

    Default family is gamma, matching the example distribution in the paper;
    family may also be 'weibull' or 'lognormal' to match the alternative
    distributions discussed in the original article.
    """
    _require_columns(df, [time_col, event_col])
    work = df[[time_col, event_col]].copy()
    work[time_col] = _coerce_numeric(work[time_col])
    work[event_col] = work[event_col].map(_safe_lower)
    work = work.dropna(subset=[time_col]).copy()

    times = work[time_col].to_numpy(dtype=float)
    events = work[event_col].to_numpy(dtype=str)
    valid = np.isfinite(times)
    times = np.maximum(times[valid], 1e-12)
    events = events[valid]

    if np.sum(events == death_label) + np.sum(events == recovery_label) == 0:
        return {
            "cfr": np.nan,
            "success": False,
            "message": "No resolved outcomes available for mixture model.",
            "result": None,
        }

    p0 = np.mean(events == death_label)
    p0 = min(max(p0, 1e-4), 1 - 1e-4)
    med = float(np.median(times)) if np.isfinite(np.median(times)) else 1.0
    med = max(med, 1e-3)

    if start_params is None:
        if family == "gamma":
            # logit_p, log_kd, log_thetad, log_kr, log_thetar
            start_params = [
                math.log(p0 / (1.0 - p0)),
                0.0,
                math.log(med),
                0.0,
                math.log(med),
            ]
        elif family == "weibull":
            start_params = [
                math.log(p0 / (1.0 - p0)),
                0.0,
                math.log(med),
                0.0,
                math.log(med),
            ]
        elif family == "lognormal":
            start_params = [
                math.log(p0 / (1.0 - p0)),
                math.log(0.5),
                math.log(med),
                math.log(0.5),
                math.log(med),
            ]
        else:
            raise ValueError("family must be one of {'gamma', 'weibull', 'lognormal'}")

    x0 = np.asarray(start_params, dtype=float)

    if family in {"gamma", "weibull"}:
        bounds = [(-10, 10), (-8, 8), (-8, 8), (-8, 8), (-8, 8)]
    elif family == "lognormal":
        bounds = [(-10, 10), (-8, 8), (-10, 10), (-8, 8), (-10, 10)]
    else:
        bounds = None

    res = minimize(
        _mixture_negloglik,
        x0,
        args=(times, events, family),
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": maxiter},
    )

    p = float(expit(res.x[0]))
    result: Dict[str, Any] = {
        "cfr": p,
        "family": family,
        "success": bool(res.success),
        "message": str(res.message),
        "result": res,
    }

    if family == "gamma":
        result.update(
            {
                "death_shape": float(np.exp(res.x[1])),
                "death_scale": float(np.exp(res.x[2])),
                "recovery_shape": float(np.exp(res.x[3])),
                "recovery_scale": float(np.exp(res.x[4])),
            }
        )
    elif family == "weibull":
        result.update(
            {
                "death_shape": float(np.exp(res.x[1])),
                "death_scale": float(np.exp(res.x[2])),
                "recovery_shape": float(np.exp(res.x[3])),
                "recovery_scale": float(np.exp(res.x[4])),
            }
        )
    elif family == "lognormal":
        result.update(
            {
                "death_sigma": float(np.exp(res.x[1])),
                "death_mu": float(res.x[2]),
                "recovery_sigma": float(np.exp(res.x[3])),
                "recovery_mu": float(res.x[4]),
            }
        )

    return result


# ---------------------------------------------------------------------------
# Confidence intervals
# ---------------------------------------------------------------------------


def clopper_pearson_ci(x: float, n: float, alpha: float = 0.05) -> Tuple[float, float]:
    """
    Exact Clopper-Pearson confidence interval for a binomial proportion. These are used for the naive and resolved estimates.

    Parameters
    ----------
    x : int or float
        Number of "successes" (e.g. deaths).
    n : int or float
        Number of trials (e.g. cases, or resolved cases).
    alpha : float
        Significance level. 0.05 gives a 95% CI.

    Returns
    -------
    (lo, hi)
        Lower and upper confidence limits.
    """
    x = int(x)
    n = int(n)

    if n <= 0:
        return (np.nan, np.nan)

    if x < 0 or x > n:
        raise ValueError("Need 0 <= x <= n for Clopper-Pearson CI.")

    if x == 0:
        lo = 0.0
    else:
        lo = beta.ppf(alpha / 2.0, x, n - x + 1)

    if x == n:
        hi = 1.0
    else:
        hi = beta.ppf(1.0 - alpha / 2.0, x + 1, n - x)

    return float(lo), float(hi)




def _binom_logchoose(n: int, k: int) -> float:
    if k < 0 or k > n:
        return -np.inf
    return math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)


def _select_func_likelihood(total_cases: float, poisson_threshold: int, p_mid: float) -> Callable[[float, float, np.ndarray], np.ndarray]:
    """
    Python translation of the R .select_func_likelihood() helper.

    Returns a function f(total_outcomes, total_deaths, pp_grid) -> log-likelihood values.
    """

    total_cases = float(total_cases)
    p_mid = float(p_mid)

    # Binomial approximation
    if total_cases < poisson_threshold or p_mid >= 0.05:
        def func_likelihood(total_outcomes: float, total_deaths: float, pp: np.ndarray) -> np.ndarray:
            n = int(round(total_outcomes))
            k = int(round(total_deaths))
            pp = np.asarray(pp, dtype=float)

            out = np.full(pp.shape, -np.inf, dtype=float)
            valid = (pp > 0.0) & (pp < 1.0) & np.isfinite(pp)
            if not np.any(valid):
                return out

            out[valid] = (
                _binom_logchoose(n, k)
                + k * np.log(pp[valid])
                + (n - k) * np.log1p(-pp[valid])
            )
            return out

        return func_likelihood

    # Poisson approximation
    if total_cases >= poisson_threshold and p_mid < 0.05:
        warnings.warn(
            f"Total cases = {total_cases} and p = {p_mid:.3g}: using Poisson approximation to binomial likelihood.",
            RuntimeWarning,
        )

        def func_likelihood(total_outcomes: float, total_deaths: float, pp: np.ndarray) -> np.ndarray:
            mu = np.asarray(pp, dtype=float) * float(round(total_outcomes))
            k = int(round(total_deaths))

            out = np.full(mu.shape, -np.inf, dtype=float)
            valid = (mu > 0.0) & np.isfinite(mu)
            if not np.any(valid):
                return out

            # log Poisson pmf:
            # log P(K=k | mu) = k log(mu) - mu - log(k!)
            log_k_fact = math.lgamma(k + 1)
            out[valid] = k * np.log(mu[valid]) - mu[valid] - log_k_fact
            return out

        return func_likelihood

    # Fallback should not be reached, but keep it safe
    def func_likelihood(total_outcomes: float, total_deaths: float, pp: np.ndarray) -> np.ndarray:
        n = int(round(total_outcomes))
        k = int(round(total_deaths))
        pp = np.asarray(pp, dtype=float)

        out = np.full(pp.shape, -np.inf, dtype=float)
        valid = (pp > 0.0) & (pp < 1.0) & np.isfinite(pp)
        if not np.any(valid):
            return out

        out[valid] = (
            _binom_logchoose(n, k)
            + k * np.log(pp[valid])
            + (n - k) * np.log1p(-pp[valid])
        )
        return out

    return func_likelihood


def estimate_severity_profile_likelihood(
    total_cases: float,
    total_deaths: float,
    total_outcomes: float,
    poisson_threshold: int = 1000,
    p_mid: Optional[float] = None,
) -> Dict[str, float]:
    """
    Python translation of the R .estimate_severity() function.

    Parameters
    ----------
    total_cases
        Total cases observed.
    total_deaths
        Total deaths observed.
    total_outcomes
        Total expected outcomes (for the delay-adjusted method, this is the
        expected number of cases with known outcomes, i.e. sum(cases * F(delay))).
    poisson_threshold
        Threshold above which the Poisson approximation may be used.
    p_mid
        Initial severity estimate used to choose the approximation.
        If omitted, defaults to total_deaths / round(total_outcomes), matching the R code.

    Returns
    -------
    dict
        {"estimate", "lower_ci", "upper_ci"}
    """
    total_cases = float(total_cases)
    total_deaths = float(total_deaths)
    total_outcomes = float(total_outcomes)

    # Special case: when any two are zero, return NA-like values
    if sum(v == 0 for v in [total_cases, total_deaths, total_outcomes]) >= 2:
        return {
            "estimate": np.nan,
            "lower_ci": np.nan,
            "upper_ci": np.nan,
        }

    if p_mid is None:
        rounded_outcomes = int(round(total_outcomes))
        p_mid = (total_deaths / rounded_outcomes) if rounded_outcomes > 0 else np.nan

    # If expected outcomes are fewer than deaths, the R code returns NA
    if total_outcomes < total_deaths:
        warnings.warn(
            f"Total deaths = {total_deaths} and expected outcomes = {round(total_outcomes)}; returning NaN.",
            RuntimeWarning,
        )
        return {
            "estimate": np.nan,
            "lower_ci": np.nan,
            "upper_ci": np.nan,
        }

    func_likelihood = _select_func_likelihood(total_cases, poisson_threshold, p_mid)

    # Profile over severity grid
    p_grid = np.arange(1e-4, 1.0000 + 1e-4, 1e-4)
    lik = func_likelihood(total_outcomes, total_deaths, p_grid)

    if not np.any(np.isfinite(lik)):
        return {
            "estimate": np.nan,
            "lower_ci": np.nan,
            "upper_ci": np.nan,
        }

    max_lik = np.nanmax(lik)
    best_idx = np.nanargmax(lik)
    estimate = float(p_grid[best_idx])

    # 95% profile likelihood CI: logL >= max(logL) - 1.92
    keep = np.where(lik >= (max_lik - 1.92))[0]
    if keep.size == 0:
        lower_ci = np.nan
        upper_ci = np.nan
    else:
        lower_ci = float(p_grid[keep[0]])
        upper_ci = float(p_grid[keep[-1]])

    return {
        "estimate": estimate,
        "lower_ci": lower_ci,
        "upper_ci": upper_ci,
    }



# ---------------------------------------------------------------------------
# Daily running estimates
# ---------------------------------------------------------------------------
def _unpack_est_ci(x):
    if isinstance(x, dict):
        return (
            x.get("estimate", x.get("severity_estimate", np.nan)),
            x.get("lower_ci", x.get("severity_low", np.nan)),
            x.get("upper_ci", x.get("severity_high", np.nan)),
        )
    return float(x), np.nan, np.nan

def running_cfr_from_count_table(
    df: pd.DataFrame,
    *,
    date_col: str,
    cases_col: str,
    deaths_col: str,
    recovered_col: Optional[str] = None,
    delay_distribution: Any = None,
    methods: Optional[Sequence[str]] = None,
    group_cols: Optional[Sequence[str]] = None,
    is_cumulative: bool = True,
) -> pd.DataFrame:
    """
    Daily expanding-window CFR estimates for count tables.

    The supplied columns are treated as cumulative totals when is_cumulative=True.
    """
    work = standardize_count_table(
        df,
        date_col=date_col,
        cases_col=cases_col,
        deaths_col=deaths_col,
        recovered_col=recovered_col,
        group_cols=group_cols,
        fill_missing_dates=True,
        is_cumulative=is_cumulative,
    )

    if methods is None:
        methods = ["naive", "resolved", "delay_adjusted"]
    methods = list(methods)

    group_cols = [c for c in (group_cols or []) if c in work.columns]
    rows = []

    def _apply_pseudo_realtime_downward_correction(series: np.ndarray) -> np.ndarray:
        """
        Retroactively cap earlier cumulative values whenever a later lower value appears.
        This preserves the pseudo-real-time logic: once a revision is observed, all prior
        days are revised downward to the minimum seen so far.
        """
        arr = np.asarray(series, dtype=float).copy()
        if arr.size == 0:
            return arr

        last_seen = np.nan
        for i in range(arr.size):
            if not np.isfinite(arr[i]):
                continue

            if np.isfinite(last_seen) and arr[i] < last_seen:
                arr[:i] = np.where(np.isfinite(arr[:i]), np.minimum(arr[:i], arr[i]), arr[:i])

            last_seen = arr[i]

        return arr

    group_iter = [(None, work)] if not group_cols else list(work.groupby(group_cols, dropna=False, sort=False))

    for _, g in group_iter:
        g = g.sort_values(date_col).reset_index(drop=True)

        cases_raw = g[cases_col].ffill().fillna(0.0).to_numpy(dtype=float) if is_cumulative else g[cases_col].fillna(0.0).to_numpy(dtype=float)
        deaths_raw = g[deaths_col].ffill().fillna(0.0).to_numpy(dtype=float) if is_cumulative else g[deaths_col].fillna(0.0).to_numpy(dtype=float)

        recovered_raw = None
        if recovered_col and recovered_col in g.columns:
            recovered_raw = g[recovered_col].ffill().fillna(0.0).to_numpy(dtype=float) if is_cumulative else g[recovered_col].fillna(0.0).to_numpy(dtype=float)

        # Pseudo-real-time revision handling for cumulative series
        if is_cumulative:
            cases = _apply_pseudo_realtime_downward_correction(cases_raw)
            deaths = _apply_pseudo_realtime_downward_correction(deaths_raw)
            recovered = _apply_pseudo_realtime_downward_correction(recovered_raw) if recovered_raw is not None else None

            # Daily incidence after retrospective correction
            cases_inc = np.diff(np.r_[0.0, cases])
            deaths_inc = np.diff(np.r_[0.0, deaths])
            recovered_inc = np.diff(np.r_[0.0, recovered]) if recovered is not None else None
        else:
            cases = cases_raw
            deaths = deaths_raw
            recovered = recovered_raw
            cases_inc = cases
            deaths_inc = deaths
            recovered_inc = recovered

        for i, dt in enumerate(g[date_col]):
            row = {"date": pd.to_datetime(dt)}
            if group_cols:
                for c in group_cols:
                    row[c] = g[c].iloc[0]

            if "naive" in methods:
                est, lo, hi = _unpack_est_ci(cfr_naive(deaths[i], cases[i]))
                row["naive"] = est
                row["naive_lower"] = lo
                row["naive_upper"] = hi

            if "resolved" in methods:
                if recovered is None:
                    row["resolved"] = np.nan
                else:
                    est, lo, hi = _unpack_est_ci(cfr_resolved_cohort(deaths[i], recovered[i]))
                    row["resolved"] = est
                    row["resolved_lower"] = lo
                    row["resolved_upper"] = hi

            if "delay_adjusted" in methods:
                if delay_distribution is None:
                    row["delay_adjusted"] = np.nan
                else:
                    est, lo, hi = _unpack_est_ci(
                        cfr_delay_adjusted_nishiura(
                            deaths=deaths_inc[: i + 1],
                            cases=cases_inc[: i + 1],
                            delay_distribution=delay_distribution,
                        )
                    )
                    row["delay_adjusted"] = est
                    row["delay_adjusted_lower"] = lo
                    row["delay_adjusted_upper"] = hi

            rows.append(row)

    return pd.DataFrame(rows)




def running_cfr_from_line_list(
    df: pd.DataFrame,
    *,
    onset_col: Optional[str] = None,
    outcome_col: str = "event",
    outcome_date_col: Optional[str] = None,
    death_label: str = "death",
    recovery_label: str = "recovery",
    delay_distribution_death: Any = None,
    delay_distribution_recovery: Any = None,
    methods: Optional[Sequence[str]] = None,
    dayfirst: bool = True,
) -> pd.DataFrame:
    """
    Daily expanding-window CFR estimates for individual-level data.

    For each analysis date, cases with start_date <= analysis_date enter the cohort;
    outcomes after analysis_date are treated as censored at that date.
    """
    linelist = standardize_line_list(
        df,
        onset_col=onset_col,
        outcome_col=outcome_col if outcome_col in df.columns else None,
        outcome_date_col=outcome_date_col,
        death_label=death_label,
        recovery_label=recovery_label,
        dayfirst=dayfirst,
    )

    if methods is None:
        methods = ["naive", "resolved", "delay_adjusted", "competing_risks", "kaplan_meier_ghani", "parametric_mixture"]
    methods = list(methods)

    dates = pd.DatetimeIndex(sorted(pd.unique(linelist["start_date"].dropna())))
    rows = []
    for cutoff in dates:
        current = _prepare_individual_time_data(
            linelist,
            analysis_date=cutoff,
            death_label=death_label,
            recovery_label=recovery_label,
        )
        row = {"date": cutoff, "n": int(len(current))}

        if "naive" in methods or "resolved" in methods or "delay_adjusted" in methods:
            observed_start = linelist.loc[linelist["start_date"] <= cutoff].copy()
            cases = float(len(observed_start))
            deaths = float(np.sum((observed_start["event"] == death_label) & (observed_start["outcome_date"].notna()) & (observed_start["outcome_date"] <= cutoff)))
            recovered = float(np.sum((observed_start["event"] == recovery_label) & (observed_start["outcome_date"].notna()) & (observed_start["outcome_date"] <= cutoff)))

            if "naive" in methods:
                naive = cfr_naive(deaths, cases)
                row["naive"] = naive["estimate"]
                row["naive_lower"] = naive["lower_ci"]
                row["naive_upper"] = naive["upper_ci"]
                # row["naive"] = deaths / cases if cases > 0 else np.nan
            if "resolved" in methods:
                est, lo, hi = _unpack_est_ci(cfr_resolved_cohort(deaths, recovered))
                
                row["resolved"] = est
                row["resolved_lower"] = lo
                row["resolved_upper"] = hi
                # denom = deaths + recovered
                # row["resolved"] = deaths / denom if denom > 0 else np.nan
            if "delay_adjusted" in methods:
                if observed_start.empty or pd.isna(observed_start["start_date"].min()):
                    row["delay_adjusted"] = np.nan
                else:
                    # Build daily incidence by start date and daily death incidence by outcome date.
                    cohort_dates = pd.date_range(observed_start["start_date"].min(), cutoff, freq="D")
                    cases_daily = (
                        observed_start.groupby("start_date").size().reindex(cohort_dates, fill_value=0).to_numpy(dtype=float)
                    )
                    deaths_daily = (
                        observed_start.loc[
                            (observed_start["event"] == death_label)
                            & (observed_start["outcome_date"].notna())
                            & (observed_start["outcome_date"] <= cutoff)
                        ]
                        .groupby("outcome_date")
                        .size()
                        .reindex(cohort_dates, fill_value=0)
                        .to_numpy(dtype=float)
                    )

                    delay_dist_for_day = delay_distribution_death
                    if delay_dist_for_day is None:
                        try:
                            delay_dist_for_day = _estimate_delay_distribution_for_analysis_date(
                                linelist,
                                analysis_date=cutoff,
                                onset_col="start_date",
                                outcome_date_col="outcome_date",
                                outcome_col="event",
                                death_label=death_label,
                                recovery_label=recovery_label,
                                dayfirst=dayfirst,
                            )[death_label]["cdf"]
                        except Exception:
                            delay_dist_for_day = None

                    if delay_dist_for_day is None:
                        row["delay_adjusted"] = np.nan
                    else:
                        est, lo, hi = _unpack_est_ci(cfr_delay_adjusted_nishiura(
                            deaths=deaths_daily,
                            cases=cases_daily,
                            delay_distribution=delay_dist_for_day,
                        ))
                        row["delay_adjusted"] = est
                        row["delay_adjusted_lower"] = lo
                        row["delay_adjusted_upper"] = hi

        if "competing_risks" in methods:
            est, lo, hi = _unpack_est_ci(cfr_competing_risks(current))
            row["competing_risks"] = est
            row["competing_risks_lower"] = lo
            row["competing_risks_upper"] = hi
        if "kaplan_meier_ghani" in methods:
            est, lo, hi = _unpack_est_ci(cfr_ghani_2005_km(current))
            row["kaplan_meier_ghani"] = est
            row["kaplan_meier_ghani_upper"] = hi
            row["kaplan_meier_ghani_lower"] = lo
        if "parametric_mixture" in methods:
            mix = cfr_parametric_mixture(current, family="gamma")
            row["parametric_mixture"] = mix["estimate"]
            row["parametric_mixture_lower"] = mix["lower_ci"]
            row["parametric_mixture_upper"] = mix["upper_ci"]
            row["parametric_mixture_success"] = mix["success"]

        rows.append(row)

    return pd.DataFrame(rows)



def detect_dataset_kind(df: pd.DataFrame) -> str:
    cols = set(df.columns)
    count_signals = {"cases", "deaths", "report_date", "reference_date", "total_cases", "total_deaths"}
    line_signals = {
        "Date_onset", "Date_of_onset_symp", "Date_Death", "Date_Recovered",
        "Date_confirmation", "Date_of_first_consult", "Date_of_Death", "Date_disease_ended",
        "Outcome"
    }

    if len(cols & line_signals) >= 2:
        return "line_list"
    if len(cols & count_signals) >= 2:
        return "count_table"
    return "unknown"


def running_cfr(
    df: pd.DataFrame,
    *,
    dataset_kind: str = "auto",
    methods: Optional[Sequence[str]] = None,
    date_col: Optional[str] = None,
    cases_col: Optional[str] = None,
    deaths_col: Optional[str] = None,
    recovered_col: Optional[str] = None,
    onset_col: Optional[str] = None,
    outcome_col: str = "event",
    outcome_date_col: Optional[str] = None,
    delay_distribution_death: Any = None,
    delay_distribution_recovery: Any = None,
    death_label: str = "death",
    recovery_label: str = "recovery",
    dayfirst: bool = True,
) -> pd.DataFrame:
    if dataset_kind == "auto":
        dataset_kind = detect_dataset_kind(df)

    if dataset_kind == "count_table":
        if date_col is None:
            date_col = "report_date" if "report_date" in df.columns else "reference_date"
        if cases_col is None:
            cases_col = _first_existing_column(
                df,
                [
                    "confirmed_cases",
                    "total_confirmed_cases",
                    "total_cases",
                    "cases",
                ],
            )

        if deaths_col is None:
            deaths_col = _first_existing_column(
                df,
                [
                    "confirmed_deaths",
                    "total_confirmed_deaths",
                    "total_deaths",
                    "deaths",
                ],
            )

        if recovered_col is None:
            recovered_col = _first_existing_column(
                df,
                [
                    "confirmed_recovered",
                    "total_confirmed_cured",
                    "total_cured",
                    "recovered",
                ],
            )
        return running_cfr_from_count_table(
            df,
            date_col=date_col,
            cases_col=cases_col,
            deaths_col=deaths_col,
            recovered_col=recovered_col,
            delay_distribution=delay_distribution_death,
            methods=methods,
            is_cumulative=True,
        )

    if dataset_kind == "line_list":
        return running_cfr_from_line_list(
            df,
            onset_col=onset_col,
            outcome_col=outcome_col,
            outcome_date_col=outcome_date_col,
            death_label=death_label,
            recovery_label=recovery_label,
            delay_distribution_death=delay_distribution_death,
            delay_distribution_recovery=delay_distribution_recovery,
            methods=methods,
            dayfirst=dayfirst,
        )

    raise ValueError(f"Could not determine dataset_kind={dataset_kind!r}.")


__all__ = [
    "load_drc_consolidated",
    "load_drc_by_health_zone",
    "load_drc_total",
    "load_rosello2015",
    "load_uganda_2022",
    "standardize_count_table",
    "standardize_line_list",
    "estimate_delay_distributions_from_individual_data",
    "estimate_delay_distribution_from_dates",
    "cfr_naive",
    "cfr_resolved_cohort",
    "cfr_delay_adjusted_nishiura",
    "cfr_competing_risks",
    "cfr_ghani_2005_km",
    "cfr_parametric_mixture",
    "running_cfr_from_count_table",
    "running_cfr_from_line_list",
    "running_cfr",
    "detect_dataset_kind",
    "adapt_drc_total_to_counts",
    "adapt_drc_health_zone_to_counts",
    "adapt_drc_consolidated_to_counts",
    "adapt_rosello_to_linelist",
    "adapt_uganda_to_linelist",
]


def adapt_drc_total_to_counts(df: pd.DataFrame) -> pd.DataFrame:
    return standardize_count_table(
        df,
        date_col="report_date",
        cases_col="total_cases",
        deaths_col="total_deaths",
        recovered_col="total_cured",
        is_cumulative=True,
    )


def adapt_drc_health_zone_to_counts(df: pd.DataFrame) -> pd.DataFrame:
    return standardize_count_table(
        df,
        date_col="report_date",
        cases_col="total_cases",
        deaths_col="total_deaths",
        recovered_col="total_cured",
        group_cols=["health_zone"],
        is_cumulative=True,
    )



def adapt_drc_consolidated_to_counts(
    df: pd.DataFrame,
    *,
    group_cols: Optional[Sequence[str]] = None,
    prefer_confirmed: bool = True,
) -> pd.DataFrame:
    required = {"reference_date", "measure", "value"}
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Unexpected DRC consolidated schema. Missing: {missing}")

    work = df.copy()
    work["reference_date"] = _to_datetime(work["reference_date"])
    work["measure"] = work["measure"].astype(str).str.strip().str.lower()

    if "case_classification" in work.columns:
        work["case_classification"] = work["case_classification"].astype(str).str.strip().str.lower()
    else:
        work["case_classification"] = ""

    if "time_period" in work.columns:
        tp = work["time_period"].astype(str).str.strip().str.lower()
        work = work.loc[tp.eq("cumulative")].copy()

    allowed_measures = {
        "cases",
        "deaths",
        "recoveries",
        "recovered",
        "cured",
        "total_cases",
        "total_deaths",
        "total_cured",
    }
    work = work.loc[work["measure"].isin(allowed_measures)].copy()

    work["measure_key"] = np.where(
        work["case_classification"].ne("") & work["case_classification"].ne("nan"),
        work["case_classification"] + "_" + work["measure"],
        work["measure"],
    )

    index_cols = ["reference_date"]
    if group_cols:
        index_cols += [c for c in group_cols if c in work.columns]

    pivot = (
        work.pivot_table(
            index=index_cols,
            columns="measure_key",
            values="value",
            aggfunc="sum",
        )
        .reset_index()
        .rename(columns={"reference_date": "report_date"})
    )
    pivot.columns.name = None

    def _first_non_missing(cols: Sequence[str]) -> pd.Series:
        out = pd.Series(np.nan, index=pivot.index, dtype="float64")
        for c in cols:
            if c in pivot.columns:
                out = out.combine_first(pd.to_numeric(pivot[c], errors="coerce"))
        return out

    if prefer_confirmed:
        pivot["cases"] = _first_non_missing(
            ["confirmed_cases", "probable_cases", "suspected_cases", "cases", "total_cases"]
        )
        pivot["deaths"] = _first_non_missing(
            ["confirmed_deaths", "probable_deaths", "suspected_deaths", "deaths", "total_deaths"]
        )
        pivot["recovered"] = _first_non_missing(
            [
                "confirmed_recoveries", "confirmed_recovered",
                "probable_recoveries", "probable_recovered",
                "suspected_recoveries", "suspected_recovered",
                "recoveries", "recovered", "cured", "total_cured"
            ]
        )
    else:
        pivot["cases"] = _first_non_missing(["cases", "total_cases"])
        pivot["deaths"] = _first_non_missing(["deaths", "total_deaths"])
        pivot["recovered"] = _first_non_missing(["recoveries", "recovered", "cured", "total_cured"])

    if "total_cases" not in pivot.columns:
        pivot["total_cases"] = pivot["cases"]
    if "total_deaths" not in pivot.columns:
        pivot["total_deaths"] = pivot["deaths"]
    if "total_cured" not in pivot.columns:
        pivot["total_cured"] = pivot["recovered"]

    sort_cols = ["report_date"]
    if group_cols:
        sort_cols += [c for c in group_cols if c in pivot.columns]

    return pivot.sort_values(sort_cols).reset_index(drop=True)

def adapt_drc_consolidated_to_counts1(df: pd.DataFrame) -> pd.DataFrame:
    # This table is not a true cases/deaths time series, but it can be pivoted if desired.
    if "reference_date" not in df.columns or "measure" not in df.columns:
        raise ValueError("Unexpected DRC consolidated schema.")
    work = df.copy()
    work["reference_date"] = _to_datetime(work["reference_date"])
    work["measure"] = work["measure"].astype(str).str.lower()
    pivot = (
        work.pivot_table(index=["reference_date"], columns="measure", values="value", aggfunc="sum")
        .reset_index()
        .rename(columns={"reference_date": "report_date"})
    )
    for c in ["cases", "deaths", "recoveries", "total_cases", "total_deaths", "total_cured"]:
        if c not in pivot.columns:
            pivot[c] = np.nan
    if "cases" not in pivot.columns and "total_cases" in pivot.columns:
        pivot["cases"] = pivot["total_cases"]
    if "deaths" not in pivot.columns and "total_deaths" in pivot.columns:
        pivot["deaths"] = pivot["total_deaths"]
    return pivot



def adapt_rosello_to_linelist(
    df: pd.DataFrame,
    start_date_col: str = "Date_of_onset_symp",
) -> pd.DataFrame:
    work = df.copy()
    work["analysis_origin_date"] = _to_datetime(
    work[start_date_col],
    dayfirst=True,
)

    outcome = work["Outcome"].map(_safe_lower)
    event = pd.Series(
        np.where(
            outcome.isin({"dead", "death"}),
            "death",
            np.where(outcome.isin({"alive", "recovered", "discharged"}), "recovery", "censored"),
        ),
        index=work.index,
    )

    outcome_date = pd.Series(pd.NaT, index=work.index, dtype="datetime64[ns]")
    if "Date_of_Death" in work.columns:
        outcome_date = outcome_date.fillna(_to_datetime(work["Date_of_Death"], dayfirst=True))
    if "Date_hospital_discharge" in work.columns:
        outcome_date = outcome_date.fillna(_to_datetime(work["Date_hospital_discharge"], dayfirst=True))
    if "Date_disease_ended" in work.columns:
        outcome_date = outcome_date.fillna(_to_datetime(work["Date_disease_ended"], dayfirst=True))

    work = pd.DataFrame(
        {
            "start_date": work["analysis_origin_date"],
            "outcome_date": outcome_date,
            "event": event,
        }
    )
    return work.dropna(subset=["start_date"]).copy()


def adapt_rosello_to_linelist_by_outbreak(
    df: pd.DataFrame,
    outbreak_col: str = "Outbreak",
) -> Dict[str, pd.DataFrame]:
    if outbreak_col not in df.columns:
        raise ValueError(f"Missing required column: {outbreak_col}")

    out = {}
    for outbreak_value, group in df.groupby(outbreak_col, dropna=False):
        key = "missing" if pd.isna(outbreak_value) else str(outbreak_value)
        ll = adapt_rosello_to_linelist(group).copy()
        ll[outbreak_col] = outbreak_value
        out[key] = ll.reset_index(drop=True)

    return out

def adapt_uganda_to_linelist(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    work["analysis_origin_date"] = _to_datetime(work["Date_onset"], dayfirst=False)
    fallback = work["analysis_origin_date"].copy()
    for c in ["Date_of_first_consult", "Date_confirmation", "Date_hospitalisation", "Date_isolation"]:
        if c in work.columns:
            fallback = fallback.fillna(_to_datetime(work[c], dayfirst=False))
    outcome = work["Outcome"].map(_safe_lower)
    event = pd.Series(
        np.where(
            outcome.isin({"death", "dead", "died", "deceased"}),
            "death",
            np.where(outcome.isin({"recovery", "recovered", "alive", "discharged"}), "recovery", "censored"),
        ),
        index=work.index,
    )
    outcome_date = pd.Series(pd.NaT, index=work.index, dtype="datetime64[ns]")
    if "Date_Death" in work.columns:
        outcome_date = outcome_date.fillna(_to_datetime(work["Date_Death"], dayfirst=False))
    if "Date_Recovered" in work.columns:
        outcome_date = outcome_date.fillna(_to_datetime(work["Date_Recovered"], dayfirst=False))
    work = pd.DataFrame({"start_date": fallback, "outcome_date": outcome_date, "event": event})
    work = work.dropna(subset=["start_date"]).copy()
    return work


