
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

Note that this code is made with AI and has not been checked yet. 
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import math
import warnings

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit
from scipy import stats

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


def load_sierra_leone_confirmed(path: Union[str, Path]) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "Date of symptom onset " in df.columns:
        df["Date of symptom onset "] = _to_datetime(df["Date of symptom onset "], dayfirst=True)
    if "Date of sample tested" in df.columns:
        df["Date of sample tested"] = _to_datetime(df["Date of sample tested"], dayfirst=True)
    return df


def load_sierra_leone_suspected(path: Union[str, Path]) -> pd.DataFrame:
    return load_sierra_leone_confirmed(path)


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
        return out.dropna(subset=["start_date"]).copy()

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


# ---------------------------------------------------------------------------
# Estimators
# ---------------------------------------------------------------------------

def cfr_naive(deaths: Union[pd.Series, np.ndarray, float, int], cases: Union[pd.Series, np.ndarray, float, int]) -> float:
    deaths = float(np.nansum(deaths))
    cases = float(np.nansum(cases))
    return deaths / cases if cases > 0 else np.nan


def cfr_resolved_cohort(deaths: Union[pd.Series, np.ndarray, float, int], recovered: Union[pd.Series, np.ndarray, float, int]) -> float:
    deaths = float(np.nansum(deaths))
    recovered = float(np.nansum(recovered))
    denom = deaths + recovered
    return deaths / denom if denom > 0 else np.nan


def cfr_delay_adjusted_nishiura_1(
    deaths: Union[pd.Series, np.ndarray, float, int],
    cases: Union[pd.Series, np.ndarray, float, int],
    delay_distribution: Any,
) -> float:
    """
    Delay-adjusted static CFR per Nishiura et al. and the Epiverse cfr_static docs.

    Point estimate:
        CFR_hat = D_t / sum_i c_i * F(t - i)
    where F is the CDF of the onset-to-death delay distribution.
    """
    deaths = np.asarray(deaths, dtype=float)
    cases = np.asarray(cases, dtype=float)
    if deaths.size != cases.size:
        raise ValueError("deaths and cases must have the same length.")
    ages = np.arange(len(cases) - 1, -1, -1)
    known_outcome_prob = _delay_cdf_at_ages(delay_distribution, ages)
    estimated_known_outcomes = float(np.sum(cases * known_outcome_prob))
    total_deaths = float(np.nansum(deaths))
    return total_deaths / estimated_known_outcomes if estimated_known_outcomes > 0 else np.nan

def cfr_delay_adjusted_nishiura(
    deaths: Union[pd.Series, np.ndarray, float, int],
    cases: Union[pd.Series, np.ndarray, float, int],
    delay_distribution: Any,
) -> float:
    """
    Delay-adjusted confirmed CFR using daily incidence counts.

    This estimates p_t by maximizing the binomial log-likelihood

        D_t ~ Binomial(u_t * C_t, p_t)

    where:
        D_t = total deaths observed up to time t
        C_t = incident cases by day up to time t
        u_t = fraction of cases expected to have known outcomes by time t,
              computed from the delay distribution.

    Parameters
    ----------
    deaths:
        Daily incident deaths up to time t.
    cases:
        Daily incident confirmed cases up to time t.
    delay_distribution:
        Delay from onset/confirmation to death. May be:
        - a callable CDF: f(ages) -> probabilities
        - a mapping with key "cdf" or "pmf"
        - a 1D PMF/CDF-like array

    Returns
    -------
    float
        Maximum-likelihood estimate of the delay-adjusted CFR.
    """

    deaths = np.asarray(deaths, dtype=float).reshape(-1)
    cases = np.asarray(cases, dtype=float).reshape(-1)
    # print(deaths)
    # print(cases)

    if deaths.size != cases.size:
        raise ValueError("deaths and cases must have the same length.")
    if deaths.size == 0:
        return np.nan

    deaths = np.nan_to_num(deaths, nan=0.0)
    cases = np.nan_to_num(cases, nan=0.0)

    # Replace negative incidence caused by data revisions with zero
    deaths = np.maximum(deaths, 0.0)
    cases = np.maximum(cases, 0.0)

    if np.any(deaths < 0) or np.any(cases < 0):
        raise ValueError("deaths and cases must be non-negative incidence counts.")

    ages = np.arange(cases.size - 1, -1, -1, dtype=int)
    known_outcome_prob = _delay_cdf_at_ages(delay_distribution, ages)

    u_t_c_t = float(np.sum(cases * known_outcome_prob))
    d_t = float(np.sum(deaths))

    if u_t_c_t <= 0:
        return np.nan

    # Numerical guard: the model requires d_t <= u_t_c_t
    if d_t > u_t_c_t:
        u_t_c_t = d_t

    def neg_log_likelihood(x: np.ndarray) -> float:
        p = float(x[0])
        if p <= 0.0 or p >= 1.0:
            return np.inf

        return -(
            d_t * np.log(p) +
            (u_t_c_t - d_t) * np.log1p(-p)
        )

    # Start at the closed-form estimate for stability
    p0 = np.clip(d_t / u_t_c_t, 1e-12, 1 - 1e-12)

    res = minimize(
        neg_log_likelihood,
        x0=np.array([p0], dtype=float),
        method="L-BFGS-B",
        bounds=[(1e-12, 1 - 1e-12)],
    )

    if not res.success:
        return float(p0)

    return float(res.x[0])


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


def cfr_competing_risks(
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


def cfr_kaplan_meier(
    df: pd.DataFrame,
    *,
    time_col: str = "time",
    event_col: str = "event",
    death_label: str = "death",
    recovery_label: str = "recovery",
) -> float:
    """
    Ghani et al. adapted Kaplan-Meier estimator:
        CFR = P(death) / (P(death) + P(recovery))
    with each outcome estimated by a separate KM curve, treating the other
    outcome and censoring as censored.
    """
    _require_columns(df, [time_col, event_col])
    work = df[[time_col, event_col]].copy()
    work[time_col] = _coerce_numeric(work[time_col])
    work[event_col] = work[event_col].map(_safe_lower)
    work = work.dropna(subset=[time_col]).copy()

    times = work[time_col].to_numpy(dtype=float)
    events = work[event_col].to_numpy(dtype=str)

    def _km_survival(event_of_interest: str) -> float:
        # treat competing outcomes and censoring as censored
        observed = events == event_of_interest
        unique_times = np.unique(times[observed])
        s = 1.0
        for t in np.sort(unique_times):
            at_risk = np.sum(times >= t)
            d = np.sum((times == t) & observed)
            if at_risk > 0:
                s *= (1.0 - d / at_risk)
        return s

    s_death = _km_survival(death_label)
    s_recovery = _km_survival(recovery_label)
    p_death = 1.0 - s_death
    p_recovery = 1.0 - s_recovery
    denom = p_death + p_recovery
    return float(p_death / denom) if denom > 0 else np.nan


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


def cfr_parametric_mixture(
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
# Daily running estimates
# ---------------------------------------------------------------------------


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

    group_iter = [(None, work)] if not group_cols else list(work.groupby(group_cols, dropna=False, sort=False))

    for _, g in group_iter:
        g = g.sort_values(date_col).reset_index(drop=True)

        cases = g[cases_col].ffill().fillna(0.0).to_numpy(dtype=float) if is_cumulative else g[cases_col].fillna(0.0).to_numpy(dtype=float)
        deaths = g[deaths_col].ffill().fillna(0.0).to_numpy(dtype=float) if is_cumulative else g[deaths_col].fillna(0.0).to_numpy(dtype=float)
        recovered = None
        if recovered_col and recovered_col in g.columns:
            recovered = g[recovered_col].ffill().fillna(0.0).to_numpy(dtype=float) if is_cumulative else g[recovered_col].fillna(0.0).to_numpy(dtype=float)

        # incident series for delay adjustment
        if is_cumulative:
            cases_inc = np.diff(np.r_[0.0, cases])
            deaths_inc = np.diff(np.r_[0.0, deaths])
        else:
            cases_inc = cases
            deaths_inc = deaths

        for i, dt in enumerate(g[date_col]):
            row = {"date": pd.to_datetime(dt)}
            if group_cols:
                for c in group_cols:
                    row[c] = g[c].iloc[0]

            if "naive" in methods:
                row["naive"] = cfr_naive(deaths[i], cases[i])
            if "resolved" in methods:
                if recovered is None:
                    row["resolved"] = np.nan
                else:
                    row["resolved"] = cfr_resolved_cohort(deaths[i], recovered[i])
            if "delay_adjusted" in methods:
                if delay_distribution is None:
                    row["delay_adjusted"] = np.nan
                else:
                    row["delay_adjusted"] = cfr_delay_adjusted_nishiura(
                        deaths=deaths_inc[: i + 1],
                        cases=cases_inc[: i + 1],
                        delay_distribution=delay_distribution,
                    )
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
        methods = ["naive", "resolved", "delay_adjusted", "competing_risks", "kaplan_meier", "parametric_mixture"]
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
                row["naive"] = deaths / cases if cases > 0 else np.nan
            if "resolved" in methods:
                denom = deaths + recovered
                row["resolved"] = deaths / denom if denom > 0 else np.nan
            if "delay_adjusted" in methods:
                if delay_distribution_death is None:
                    row["delay_adjusted"] = np.nan
                else:
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
                        row["delay_adjusted"] = cfr_delay_adjusted_nishiura(
                            deaths=deaths_daily,
                            cases=cases_daily,
                            delay_distribution=delay_distribution_death,
                        )

        if "competing_risks" in methods:
            row["competing_risks"] = cfr_competing_risks(current)
        if "kaplan_meier" in methods:
            row["kaplan_meier"] = cfr_kaplan_meier(current)
        if "parametric_mixture" in methods:
            mix = cfr_parametric_mixture(current, family="gamma")
            row["parametric_mixture"] = mix["cfr"]
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
    "load_sierra_leone_confirmed",
    "load_sierra_leone_suspected",
    "standardize_count_table",
    "standardize_line_list",
    "estimate_delay_distributions_from_individual_data",
    "estimate_delay_distribution_from_dates",
    "cfr_naive",
    "cfr_resolved_cohort",
    "cfr_delay_adjusted_nishiura",
    "cfr_competing_risks",
    "cfr_kaplan_meier",
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
    "adapt_sierra_leone_to_linelist",
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


def adapt_drc_consolidated_to_counts(df: pd.DataFrame) -> pd.DataFrame:
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


from typing import Dict
import pandas as pd
import numpy as np

def adapt_rosello_to_linelist(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    work["analysis_origin_date"] = _to_datetime(work["Date_of_onset_symp"], dayfirst=True)

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


def adapt_sierra_leone_to_linelist(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    start = _to_datetime(work.get("Date of symptom onset "), dayfirst=True)
    return pd.DataFrame({"start_date": start, "outcome_date": pd.NaT, "event": "censored"})
