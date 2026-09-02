
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
import re
import warnings

import numpy as np
import pandas as pd
import patsy
import statsmodels.api as sm
import statsmodels.formula.api as smf
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


_AGE_BAND_RE = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*-\s*(\d+(?:\.\d+)?)\s*$")
_AGE_OPEN_RE = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*\+\s*$")


def parse_age_bands(
    series: pd.Series,
    *,
    method: str = "midpoint",
    open_bin_width: float = 10.0,
    rng: Optional[np.random.Generator] = None,
) -> pd.Series:
    """Convert age-band strings (e.g. "20-24", "80+") to a single numeric age.

    method="midpoint" uses the bin midpoint. This is a standard, simple
    approximation, but it treats an interval-censored age as if it were
    known exactly, which understates uncertainty and can attenuate a fitted
    age effect -- the wider/more inconsistent the bins, the bigger the
    concern. Check that bin widths are actually consistent before trusting
    this for a given dataset.

    method="uniform_sample" instead draws one age uniformly at random within
    each bin. It isn't better as a single call (one arbitrary draw vs. one
    arbitrary midpoint), but is meant to be called once per replicate in a
    multiple-imputation loop, so the within-bin uncertainty shows up as
    extra spread across replicates rather than being silently discarded.

    Entries that are already numeric, or don't match a "lo-hi"/"lo+"
    pattern, are coerced with pandas.to_numeric, so a column mixing exact
    ages and bands is handled sensibly.
    """
    if method not in {"midpoint", "uniform_sample"}:
        raise ValueError(f"Unknown method: {method!r}")
    if rng is None:
        rng = np.random.default_rng()

    def _parse_one(raw: Any) -> float:
        if pd.isna(raw):
            return np.nan
        text = str(raw).strip()

        m = _AGE_BAND_RE.match(text)
        if m:
            lo, hi = float(m.group(1)), float(m.group(2))
            if method == "midpoint":
                return (lo + hi) / 2.0
            return float(rng.uniform(lo, hi + 1.0))  # bins are integer-inclusive

        m = _AGE_OPEN_RE.match(text)
        if m:
            lo = float(m.group(1))
            if method == "midpoint":
                return lo + open_bin_width / 2.0
            return float(rng.uniform(lo, lo + open_bin_width))

        return pd.to_numeric(raw, errors="coerce")

    return series.map(_parse_one).astype(float)


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

def load_kenema_2014(path: Union[str, Path]) -> pd.DataFrame:
    df = pd.read_csv(path)
    for c in ["Date of admission", "Date of discharge"]:
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
    age_col: Optional[str] = None,
    death_label: str = "death",
    recovery_label: str = "recovery",
    dayfirst: bool = True,
) -> pd.DataFrame:
    """Return a standardized linelist with start_date, outcome_date, event.

    If `age_col` is given, or the input already has an `age`/`Age` column,
    it is carried through as a numeric `age` column for use by the
    age-specific estimators (cfr_naive_by_age, cfr_resolved_by_age).
    """
    work = df.copy()

    if age_col is None:
        age_col = _first_existing_column(work, ["age", "Age"])

    # Already-standardized input
    if {"start_date", "outcome_date", "event"}.issubset(work.columns):
        cols = ["start_date", "outcome_date", "event"]
        out = work[cols].copy()
        out["start_date"] = _to_datetime(out["start_date"], dayfirst=dayfirst)
        out["outcome_date"] = _to_datetime(out["outcome_date"], dayfirst=dayfirst)
        out["event"] = out["event"].map(_safe_lower)
        death_mask = out["event"].isin({"death", "dead", "died", "deceased"})
        rec_mask = out["event"].isin({"recovery", "recovered", "alive", "discharged", "discharge"})
        out.loc[death_mask, "event"] = death_label
        out.loc[rec_mask, "event"] = recovery_label
        out.loc[~(death_mask | rec_mask), "event"] = "censored"
        if age_col is not None:
            out["age"] = _coerce_numeric(work[age_col])

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
    if age_col is not None:
        standardized["age"] = _coerce_numeric(work[age_col])
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


# ---------------------------------------------------------------------------
# Age-specific (continuous) naive and resolved-cohort CFR
#
# Individual-level counterparts of cfr_naive / cfr_resolved_cohort: instead
# of a single pooled ratio, death is modelled as a logistic GAM with a
# B-spline term on age, died ~ bs(age). This is the direct regression
# analogue of the two pooled ratios (naive keeps every case in the
# denominator, resolved restricts to cases with a known outcome), giving a
# smooth CFR(age) curve and age-to-age odds ratios instead of a single
# stratified estimate.
# ---------------------------------------------------------------------------


_MAX_ABS_LOG_LINEAR_COEF = 15.0  # exp(15) =~ 3.3M -- already far beyond any plausible
# OR/HR; anything past this is quasi-separation that stopped short of literally
# diverging to inf/nan, not a real effect. See the Yambuku recovery model that
# motivated this: 11 events on 4 spline params converged (finite params/cov) to
# coefficients in the hundreds (HR ~ exp(400)), which passed a finite-only check
# but produced nonsense hazard ratios and CIF artifacts. Shared between the GLM
# (log-odds) and PHReg (log-hazard) checks below -- same failure mode, same fix,
# regardless of which scale the linear predictor lives on.


def _check_glm_converged(result) -> None:
    params = np.asarray(result.params)
    cov = np.asarray(result.cov_params())
    max_abs_coef = float(np.max(np.abs(params))) if params.size else 0.0
    if (
        not getattr(result, "converged", True)
        or not np.all(np.isfinite(params))
        or not np.all(np.isfinite(cov))
        or max_abs_coef > _MAX_ABS_LOG_LINEAR_COEF
    ):
        raise ValueError(
            f"GLM failed to converge stably (solver converged={getattr(result, 'converged', None)}, "
            f"|coef| up to {max_abs_coef:.3g} indicating possible quasi-separation)."
        )


def _spline_logistic_fit(
    data: pd.DataFrame,
    age_col: str,
    died_col: str,
    *,
    spline_df: int = 4,
    degree: int = 3,
    age_bounds: Optional[Tuple[float, float]] = None,
):
    """Fit died ~ bs(age) as a binomial GLM (a logistic GAM with a B-spline basis on age).

    `age_bounds`, if given, fixes the spline's boundary knots instead of
    deriving them from this fit's own data range. Needed whenever multiple
    fits (e.g. successive time snapshots) must share one basis/domain so
    they're comparable and so predictions at ages outside a given snapshot's
    own observed range don't raise (patsy's bs() doesn't extrapolate past
    its training knots by default).
    """
    n_unique_ages = data[age_col].nunique()
    if n_unique_ages <= degree:
        raise ValueError(
            f"Only {n_unique_ages} distinct age value(s) available; need more than "
            f"degree={degree} to fit a spline. Reduce spline_df/degree or supply more data."
        )
    bounds_arg = f", lower_bound={age_bounds[0]}, upper_bound={age_bounds[1]}" if age_bounds is not None else ""
    formula = f"{died_col} ~ bs({age_col}, df={spline_df}, degree={degree}{bounds_arg})"
    model = smf.glm(formula, data=data, family=sm.families.Binomial())
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        result = model.fit()
    _check_glm_converged(result)
    return result


def _aicc(aic: float, k: int, n: int) -> float:
    """Finite-sample-corrected AIC: AICc = AIC + 2k(k+1)/(n-k-1).

    Malloy, Spiegelman & Eisen (2009, Computational Statistics & Data
    Analysis 53(7):2605-2616) ran a dedicated simulation study on exactly
    this problem -- choosing spline complexity in Cox models -- and found
    plain AIC has a real, elevated false-positive rate for detecting
    spurious nonlinearity, worst in smaller samples; they recommend AICc,
    which selects fewer degrees of freedom than plain AIC while retaining
    power to detect real nonlinearity. The correction term is largest when
    n is small relative to k and vanishes as n grows, so it mainly changes
    behavior exactly where plain AIC is known to misbehave.

    Returns +inf (so this candidate can never win a min-AICc comparison)
    when n-k-1<=0 -- too little data relative to the parameter count for
    the correction to even be well-defined, let alone trustworthy.
    """
    denom = n - k - 1
    if denom <= 0:
        return float("inf")
    return aic + (2.0 * k * (k + 1)) / denom


def _fit_glm_with_model_selection(
    data: pd.DataFrame,
    age_col: str,
    died_col: str,
    *,
    spline_df: int = 4,
    degree: int = 3,
    age_bounds: Optional[Tuple[float, float]] = None,
) -> Tuple[Any, int]:
    """Fit died ~ bs(age) at several complexities, and keep whichever
    converges cleanly with the lowest AICc -- model selection among the
    candidates that fit properly, rather than a fallback ladder that just
    stops at the first one that happens to converge (that answers "what's
    the most flexible thing I could get away with," not "what does the data
    actually support"). Uses AICc (finite-sample-corrected AIC) rather than
    plain AIC -- see _aicc's docstring for why: a dedicated study of this
    exact problem (spline complexity selection in Cox/survival-style
    regression) found plain AIC has an elevated false-positive rate for
    detecting spurious nonlinearity, especially in smaller samples.

    Candidates: cubic splines (degree=3) from spline_df down to 3 basis
    functions (i.e. down to 0 interior knots -- a single global cubic);
    quadratic (degree=2, df=2, 0 interior knots -- a single global
    parabola, capable of one bend but not an inflection); and linear
    (degree=1, df=1, no bend at all). Quadratic matters as its own
    candidate, not just a stepping stone: it's the cheapest curve that can
    show a single dip-then-rise shape, and without it the search can only
    choose between "no curvature" and "a whole cubic's worth of
    flexibility" -- confirmed concretely on this project's own data, where
    a outbreak's AIC-best model turned out to be quadratic once it was
    actually offered, beating both the linear and cubic candidates that had
    been the only options before.

    Returns (fit, df_used) where df_used is the number of parameters used
    (1=linear, 2=quadratic, 3=cubic/0 knots, 4=cubic/1 knot, ...) --
    unambiguous here since every candidate in this fixed ladder has a
    distinct parameter count. Raises ValueError, with every attempt's
    failure reason, only if nothing converges (including linear).
    """
    n_obs = int(len(data))
    candidates = []
    attempts = []
    for df_candidate in range(spline_df, degree - 1, -1):
        try:
            fit = _spline_logistic_fit(
                data, age_col, died_col, spline_df=df_candidate, degree=degree, age_bounds=age_bounds
            )
            aicc = _aicc(float(fit.aic), len(fit.params), n_obs)
            candidates.append((fit, df_candidate, aicc))
        except ValueError as exc:
            attempts.append(f"spline df={df_candidate}: {exc}")

    # Quadratic (degree=2, df=2, 0 interior knots): the cheapest curve that
    # can show a single bend, sitting between linear and the cubic family.
    try:
        fit = _spline_logistic_fit(data, age_col, died_col, spline_df=2, degree=2, age_bounds=age_bounds)
        aicc = _aicc(float(fit.aic), len(fit.params), n_obs)
        candidates.append((fit, 2, aicc))
    except ValueError as exc:
        attempts.append(f"quadratic df=2: {exc}")

    try:
        model = smf.glm(f"{died_col} ~ {age_col}", data=data, family=sm.families.Binomial())
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            fit = model.fit()
        _check_glm_converged(fit)
        aicc = _aicc(float(fit.aic), len(fit.params), n_obs)
        candidates.append((fit, 1, aicc))
    except ValueError as exc:
        attempts.append(f"linear: {exc}")

    if not candidates:
        raise ValueError(
            f"No age term converged for '{died_col}' at any complexity. Attempts:\n  "
            + "\n  ".join(attempts)
        )

    best_fit, best_df, best_aic = min(candidates, key=lambda c: c[2])
    return best_fit, best_df


def _predict_cfr_curve(
    fit,
    age_grid: np.ndarray,
    *,
    age_col: str = "age",
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Predicted CFR(age) curve with a Wald CI back-transformed from the linear predictor."""
    new_data = pd.DataFrame({age_col: np.asarray(age_grid, dtype=float)})
    summary = fit.get_prediction(new_data).summary_frame(alpha=alpha)
    return pd.DataFrame(
        {
            "age": new_data[age_col].to_numpy(),
            "estimate": summary["mean"].to_numpy(),
            "lower_ci": summary["mean_ci_lower"].to_numpy(),
            "upper_ci": summary["mean_ci_upper"].to_numpy(),
        }
    )


def _get_design_info(fit):
    """The fitted formula's patsy DesignInfo, robust across statsmodels versions.

    statsmodels renamed this attribute on model.data from `design_info` to
    `model_spec` at some point around 0.15 (same underlying
    patsy.design_info.DesignInfo object either way -- confirmed by direct
    inspection, not just documentation). Hard-coding `.design_info`
    everywhere silently breaks under the newer name (AttributeError deep
    inside relative_curves_by_age/age_cfr_odds_ratio/CIF combination, with
    no hint that it's a version issue) -- this is why cross-environment
    testing matters even when "it works here."
    """
    data = fit.model.data
    if hasattr(data, "design_info"):
        return data.design_info
    if hasattr(data, "model_spec"):
        return data.model_spec
    raise AttributeError(
        "Could not find a patsy DesignInfo on this fit's model.data (looked for "
        "'design_info' and 'model_spec') -- statsmodels may have renamed it again; "
        f"available attributes: {[a for a in dir(data) if not a.startswith('_')]}"
    )


def age_cfr_odds_ratio(
    fit,
    age1: float,
    age2: float,
    *,
    age_col: str = "age",
    alpha: float = 0.05,
) -> Dict[str, float]:
    """Odds ratio of death at age2 relative to age1, from a fitted age_cfr_by_age model.

    Computed from the difference in the two ages' spline design rows rather
    than from their marginal CIs, so the covariance between the two
    predictions (they share the same spline basis/parameters) is accounted
    for correctly.
    """
    design_info = _get_design_info(fit)
    rows = np.asarray(patsy.build_design_matrices([design_info], {age_col: [age1, age2]})[0])
    diff = rows[1] - rows[0]
    params = np.asarray(fit.params)
    cov = np.asarray(fit.cov_params())

    log_or = float(diff @ params)
    se = float(np.sqrt(diff @ cov @ diff.T))
    z = norm.ppf(1 - alpha / 2)

    return {
        "age1": float(age1),
        "age2": float(age2),
        "odds_ratio": float(np.exp(log_or)),
        "lower_ci": float(np.exp(log_or - z * se)),
        "upper_ci": float(np.exp(log_or + z * se)),
    }


def cfr_naive_by_age(
    df: pd.DataFrame,
    *,
    age_col: str = "age",
    start_col: str = "start_date",
    outcome_date_col: str = "outcome_date",
    event_col: str = "event",
    analysis_date: Optional[pd.Timestamp] = None,
    death_label: str = "death",
    recovery_label: str = "recovery",
    spline_df: int = 4,
    degree: int = 3,
    age_grid: Optional[np.ndarray] = None,
    n_grid: int = 100,
    age_bounds: Optional[Tuple[float, float]] = None,
    alpha: float = 0.05,
) -> Dict[str, Any]:
    """Naive CFR (deaths / all reported cases) as a smooth function of continuous age.

    Regression analogue of cfr_naive: every row (deaths, recoveries, and
    still-open/censored cases alike) counts towards the denominator, matching
    the deaths/cases definition, but death probability is modelled as
    died ~ bs(age) rather than pooled into a single ratio.

    An individual only counts as a death/recovery if they have a valid,
    in-window outcome_date on record -- a labeled outcome with no date is
    treated as censored, via _prepare_individual_time_data, the same
    convention enforced everywhere else in this file (cfr_competing_risks,
    cfr_ghani_2005_km, cfr_parametric_mixture, cfr_competing_risks_by_age,
    group_death_ratio, ...). This used to just trust the raw event label
    regardless of a missing date, which let this function's results
    silently disagree with the rest of the file whenever outcome-date
    completeness wasn't 100% -- caught via Kikwit, which has 20 individuals
    with a death/recovery label but no recorded date; including vs.
    excluding them changed which spline complexity AIC selected.

    `df` must have a numeric age column and start_date/outcome_date/event
    columns (as produced by standardize_line_list). Pass `age_bounds` to fix
    the spline's domain when comparing several fits on different subsets --
    see age_time_relative_risk_curves.
    """
    time_event = _prepare_individual_time_data(
        df,
        start_col=start_col,
        outcome_date_col=outcome_date_col,
        event_col=event_col,
        analysis_date=analysis_date,
        death_label=death_label,
        recovery_label=recovery_label,
    )
    work = time_event.join(df[[age_col]]).dropna(subset=[age_col])
    work["died"] = (work["event"] == death_label).astype(int)

    if work["died"].nunique() < 2:
        raise ValueError("Need both deaths and non-deaths to fit an age-CFR curve.")

    fit, df_used = _fit_glm_with_model_selection(
        work, age_col, "died", spline_df=spline_df, degree=degree, age_bounds=age_bounds
    )

    if age_grid is None:
        lo, hi = age_bounds if age_bounds is not None else (work[age_col].min(), work[age_col].max())
        age_grid = np.linspace(lo, hi, n_grid)

    curve = _predict_cfr_curve(fit, age_grid, age_col=age_col, alpha=alpha)
    return {
        "fit": fit,
        "curve": curve,
        "n": int(len(work)),
        "n_deaths": int(work["died"].sum()),
        "df_used": df_used,
    }


def cfr_resolved_by_age(
    df: pd.DataFrame,
    *,
    age_col: str = "age",
    start_col: str = "start_date",
    outcome_date_col: str = "outcome_date",
    event_col: str = "event",
    analysis_date: Optional[pd.Timestamp] = None,
    death_label: str = "death",
    recovery_label: str = "recovery",
    spline_df: int = 4,
    degree: int = 3,
    age_grid: Optional[np.ndarray] = None,
    n_grid: int = 100,
    age_bounds: Optional[Tuple[float, float]] = None,
    alpha: float = 0.05,
) -> Dict[str, Any]:
    """Resolved-cohort CFR (deaths / (deaths + recoveries)) as a smooth function of age.

    Regression analogue of cfr_resolved_cohort: restricted to individuals
    with a known outcome (death or recovery), matching the resolved-cohort
    estimator's conditioning on resolution, with death probability modelled
    as died ~ bs(age) rather than pooled into a single ratio.

    Same date-validated event convention as cfr_naive_by_age (see its
    docstring) -- a labeled outcome with no recorded date doesn't count as
    resolved here, via _prepare_individual_time_data.

    `df` must have a numeric age column and start_date/outcome_date/event
    columns (as produced by standardize_line_list). Pass `age_bounds` to fix
    the spline's domain when comparing several fits on different subsets --
    see age_time_relative_risk_curves.
    """
    time_event = _prepare_individual_time_data(
        df,
        start_col=start_col,
        outcome_date_col=outcome_date_col,
        event_col=event_col,
        analysis_date=analysis_date,
        death_label=death_label,
        recovery_label=recovery_label,
    )
    work = time_event.join(df[[age_col]]).dropna(subset=[age_col])
    work = work.loc[work["event"].isin([death_label, recovery_label])].copy()
    work["died"] = (work["event"] == death_label).astype(int)

    if work["died"].nunique() < 2:
        raise ValueError("Need both deaths and recoveries to fit an age-CFR curve.")

    fit, df_used = _fit_glm_with_model_selection(
        work, age_col, "died", spline_df=spline_df, degree=degree, age_bounds=age_bounds
    )

    if age_grid is None:
        lo, hi = age_bounds if age_bounds is not None else (work[age_col].min(), work[age_col].max())
        age_grid = np.linspace(lo, hi, n_grid)

    curve = _predict_cfr_curve(fit, age_grid, age_col=age_col, alpha=alpha)
    return {
        "fit": fit,
        "curve": curve,
        "n": int(len(work)),
        "n_deaths": int(work["died"].sum()),
        "df_used": df_used,
    }


def relative_curves_by_age(
    fit,
    age_grid: np.ndarray,
    ref_age: float,
    *,
    age_col: str = "age",
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Odds ratio and risk ratio of death across age_grid, relative to ref_age.

    Both are computed on a log scale (log-odds difference for OR,
    log-probability difference for RR) and back-transformed, using the
    fitted spline's parameter covariance -- not the two points' marginal
    CIs -- so the shared-parameter covariance between each grid age and the
    reference age is accounted for correctly.
    """
    design_info = _get_design_info(fit)
    age_grid = np.asarray(age_grid, dtype=float)

    ref_row = np.asarray(patsy.build_design_matrices([design_info], {age_col: [float(ref_age)]})[0])[0]
    grid_rows = np.asarray(patsy.build_design_matrices([design_info], {age_col: age_grid})[0])

    params = np.asarray(fit.params)
    cov = np.asarray(fit.cov_params())
    z = norm.ppf(1 - alpha / 2)

    linpred_ref = float(ref_row @ params)
    var_ref = float(ref_row @ cov @ ref_row.T)
    p_ref = expit(linpred_ref)

    linpred_grid = grid_rows @ params
    var_grid = np.einsum("ij,jk,ik->i", grid_rows, cov, grid_rows)
    cov_grid_ref = grid_rows @ cov @ ref_row.T
    p_grid = expit(linpred_grid)

    # Odds ratio: OR(age) = exp(linpred(age) - linpred(ref_age))
    log_or = linpred_grid - linpred_ref
    var_log_or = var_grid + var_ref - 2 * cov_grid_ref
    se_log_or = np.sqrt(np.clip(var_log_or, 0, None))

    # Risk ratio: on the log scale, d log(expit(x))/dx = 1 - expit(x)
    log_rr = np.log(p_grid) - np.log(p_ref)
    var_log_rr = (
        (1 - p_grid) ** 2 * var_grid
        + (1 - p_ref) ** 2 * var_ref
        - 2 * (1 - p_grid) * (1 - p_ref) * cov_grid_ref
    )
    se_log_rr = np.sqrt(np.clip(var_log_rr, 0, None))

    return pd.DataFrame(
        {
            "age": age_grid,
            "ref_age": float(ref_age),
            "cfr": p_grid,
            "odds_ratio": np.exp(log_or),
            "or_lower_ci": np.exp(log_or - z * se_log_or),
            "or_upper_ci": np.exp(log_or + z * se_log_or),
            "risk_ratio": np.exp(log_rr),
            "rr_lower_ci": np.exp(log_rr - z * se_log_rr),
            "rr_upper_ci": np.exp(log_rr + z * se_log_rr),
        }
    )


def _linelist_at_cutoff(
    linelist: pd.DataFrame,
    cutoff: pd.Timestamp,
    *,
    death_label: str = "death",
    recovery_label: str = "recovery",
) -> pd.DataFrame:
    """Individual-level snapshot of a linelist as it would have looked at `cutoff`.

    Cases with start_date <= cutoff enter the cohort; outcomes that occur
    after cutoff (or haven't happened yet) are treated as censored at that
    date. Same expanding-window logic as running_cfr_from_line_list, applied
    per individual instead of aggregated into counts, so it can feed
    cfr_naive_by_age / cfr_resolved_by_age at each snapshot.
    """
    observed = linelist.loc[linelist["start_date"] <= cutoff].copy()
    resolved_death = (
        (observed["event"] == death_label)
        & observed["outcome_date"].notna()
        & (observed["outcome_date"] <= cutoff)
    )
    resolved_recovery = (
        (observed["event"] == recovery_label)
        & observed["outcome_date"].notna()
        & (observed["outcome_date"] <= cutoff)
    )
    observed["event"] = np.where(
        resolved_death, death_label, np.where(resolved_recovery, recovery_label, "censored")
    )
    return observed


def age_time_relative_risk_curves(
    df: pd.DataFrame,
    *,
    method: str = "naive",
    age_col: str = "age",
    snapshot_fractions: Sequence[float] = (0.25, 0.4, 0.55, 0.7, 0.85, 1.0),
    age_grid: Optional[np.ndarray] = None,
    n_grid: int = 60,
    ref_age: Optional[float] = None,
    spline_df: int = 3,
    degree: int = 3,
    alpha: float = 0.05,
    death_label: str = "death",
    recovery_label: str = "recovery",
    min_events: int = 15,
) -> pd.DataFrame:
    """OR(age) and RR(age) at successive snapshots of outbreak progression.

    `df` should be a standardized, single-outbreak linelist (start_date,
    outcome_date, event, age) -- e.g. from standardize_line_list applied to
    one outbreak's data. At each snapshot fraction of the outbreak's date
    range, the cohort is truncated as of that date (_linelist_at_cutoff),
    the age-CFR model is refit (cfr_naive_by_age or cfr_resolved_by_age),
    and OR/RR are computed relative to a fixed reference age (kept the same
    across snapshots for comparability).

    Snapshots where either outcome class has fewer than `min_events` cases
    are skipped (with a warning) rather than fit: with only a handful of
    events spread across a df=3 spline basis, the logistic fit can quasi-
    separate and produce enormous/infinite extrapolated ORs at the tails
    that reflect sample-size noise, not signal -- this happens in practice
    for the earliest snapshots of a small outbreak.
    """
    if method not in {"naive", "resolved"}:
        raise ValueError("method must be 'naive' or 'resolved'")
    fit_fn = cfr_naive_by_age if method == "naive" else cfr_resolved_by_age

    work = df.dropna(subset=[age_col]).copy()
    work["start_date"] = _to_datetime(work["start_date"])
    work["outcome_date"] = _to_datetime(work["outcome_date"])

    age_bounds = (float(work[age_col].min()), float(work[age_col].max()))
    if age_grid is None:
        age_grid = np.linspace(age_bounds[0], age_bounds[1], n_grid)
    if ref_age is None:
        ref_age = float(work[age_col].median())

    start = work["start_date"].min()
    end_candidates = [work["start_date"].max(), work["outcome_date"].max()]
    end_candidates = [d for d in end_candidates if pd.notna(d)]
    end = max(end_candidates) if end_candidates else start
    span = end - start

    frames = []
    for frac in snapshot_fractions:
        cutoff = start + frac * span
        snapshot = _linelist_at_cutoff(
            work, cutoff, death_label=death_label, recovery_label=recovery_label
        )

        if method == "naive":
            n_died = int((snapshot["event"] == death_label).sum())
            n_other = int(len(snapshot) - n_died)
        else:
            resolved = snapshot.loc[snapshot["event"].isin([death_label, recovery_label])]
            n_died = int((resolved["event"] == death_label).sum())
            n_other = int(len(resolved) - n_died)

        if n_died < min_events or n_other < min_events:
            warnings.warn(
                f"Skipping snapshot at {cutoff.date()} (frac={frac}): "
                f"{n_died} deaths / {n_other} other outcomes (< min_events={min_events})."
            )
            continue

        try:
            fit_result = fit_fn(
                snapshot,
                age_col=age_col,
                spline_df=spline_df,
                degree=degree,
                age_grid=age_grid,
                age_bounds=age_bounds,
                alpha=alpha,
            )
        except ValueError as exc:
            warnings.warn(f"Skipping snapshot at {cutoff.date()} (frac={frac}): {exc}")
            continue

        curve = relative_curves_by_age(fit_result["fit"], age_grid, ref_age, age_col=age_col, alpha=alpha)
        curve["cutoff"] = cutoff
        curve["frac"] = frac
        curve["days_since_start"] = (cutoff - start).days
        curve["n"] = fit_result["n"]
        curve["n_deaths"] = fit_result["n_deaths"]
        frames.append(curve)

    if not frames:
        raise ValueError("No snapshot produced a usable fit; check snapshot_fractions and data volume.")

    return pd.concat(frames, ignore_index=True)


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

    # Exclude impossible records: outcome before start (e.g. a missing onset
    # date falling back to a later proxy -- lab confirmation, hospitalisation
    # -- that turns out to postdate death; post-mortem confirmation is a
    # real, known occurrence, not just bad data entry). Without this, a
    # negative "time" silently reaches every downstream estimator that uses
    # this function, and PHReg-based ones (group_hazard_ratio,
    # cfr_competing_risks_by_age) reject it outright since Cox models
    # require non-negative event times. Same convention already used by
    # standardize_line_list.
    valid_outcome = work[outcome_date_col].isna() | (work[outcome_date_col] >= work[start_col])
    work = work.loc[valid_outcome].copy()

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


# ---------------------------------------------------------------------------
# Age-specific (continuous) competing risks: cause-specific hazards regression
#
# The direct regression extension of cfr_competing_risks (Aalen-Johansen):
# both use the identical risk-set definition (someone leaves the risk set
# for further death/recovery events the moment either occurs) and the
# identical product-integral recombination -- see the recursion in
# cfr_competing_risk1 above. The only change here is that each cause-specific
# hazard is itself a function of age (via a Cox model with a B-spline term,
# fit with statsmodels PHReg) instead of a single pooled rate.
# ---------------------------------------------------------------------------


def _check_phreg_converged(result, status_col: str, n_events: int, spline_df: int, degree: int) -> None:
    params = np.asarray(result.params)
    cov = np.asarray(result.cov_params())
    max_abs_coef = float(np.max(np.abs(params))) if params.size else 0.0
    if not np.all(np.isfinite(params)) or not np.all(np.isfinite(cov)) or max_abs_coef > _MAX_ABS_LOG_LINEAR_COEF:
        raise ValueError(
            f"Cox model for '{status_col}' failed to converge stably (non-finite "
            f"parameters, or |coef| up to {max_abs_coef:.3g} indicating quasi-separation) "
            f"with spline_df={spline_df}, degree={degree} on {n_events} events -- likely "
            f"too few events for this many parameters. Try a lower spline_df or more data."
        )


def _phreg_spline_fit(
    data: pd.DataFrame,
    age_col: str,
    time_col: str,
    status_col: str,
    *,
    spline_df: int = 4,
    degree: int = 3,
    age_bounds: Optional[Tuple[float, float]] = None,
):
    """Fit a cause-specific Cox model, time ~ bs(age), via statsmodels PHReg."""
    n_unique_ages = data[age_col].nunique()
    if n_unique_ages <= degree:
        raise ValueError(
            f"Only {n_unique_ages} distinct age value(s) available; need more than "
            f"degree={degree} to fit a spline. Reduce spline_df/degree or supply more data."
        )
    bounds_arg = f", lower_bound={age_bounds[0]}, upper_bound={age_bounds[1]}" if age_bounds is not None else ""
    formula = f"{time_col} ~ bs({age_col}, df={spline_df}, degree={degree}{bounds_arg})"
    # Efron ties: more accurate than the Breslow default when many individuals
    # share an event time (day-resolution outbreak data has plenty of this),
    # since Breslow's approximation biases toward the null as tie density grows.
    model = sm.PHReg.from_formula(formula, data=data, status=status_col, ties="efron")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        result = model.fit()

    _check_phreg_converged(result, status_col, int(data[status_col].sum()), spline_df, degree)
    return result


def _fit_phreg_with_fallback(
    data: pd.DataFrame,
    age_col: str,
    time_col: str,
    status_col: str,
    *,
    spline_df: int = 4,
    degree: int = 3,
    age_bounds: Optional[Tuple[float, float]] = None,
) -> Tuple[Any, int]:
    """Fit a cubic spline on age at several complexities (plus a plain
    linear age term), and keep whichever converges cleanly with the lowest
    AICc -- model selection among the candidates that fit properly, rather
    than a fallback ladder that just stops at the first one that happens to
    converge (that answers "what's the most flexible thing I could get away
    with," not "what does the data actually support"). AIC here is
    -2*llf + 2*n_params, the standard convention for comparing Cox models
    via the partial likelihood (PHRegResults doesn't expose .aic directly,
    unlike GLMResults, so it's computed by hand), then finite-sample
    corrected to AICc via _aicc (see its docstring: a dedicated study of
    spline complexity selection in Cox models found plain AIC has an
    elevated false-positive rate for spurious nonlinearity, especially in
    smaller samples -- exactly the regime several of these outbreaks are
    in). The correction uses the number of events for this cause as "n",
    not the total row count -- a partial likelihood's information content
    scales with events, not the size of the risk set, the same reasoning
    behind the events-per-parameter heuristics used elsewhere in this file.

    Quadratic (degree=2, df=2, 0 interior knots -- a single global
    parabola) is also included as its own candidate, not just a stepping
    stone between linear and cubic: it's the cheapest curve that can show
    a single dip-then-rise shape, and without it the search can only choose
    between "no curvature" and "a whole cubic's worth of flexibility" --
    confirmed concretely on this project's own data, where an outbreak's
    AIC-best naive-CFR model turned out to be quadratic once it was
    actually offered as an option.

    Returns (fit, df_used) where df_used is the number of parameters used
    (1=linear, 2=quadratic, 3=cubic/0 knots, 4=cubic/1 knot, ...) --
    unambiguous here since every candidate in this fixed ladder has a
    distinct parameter count. Raises ValueError, with every attempt's
    failure reason, only if nothing converges (including linear).
    """
    n_events = int(data[status_col].sum())
    candidates = []
    attempts = []
    for df_candidate in range(spline_df, degree - 1, -1):
        try:
            fit = _phreg_spline_fit(
                data, age_col, time_col, status_col,
                spline_df=df_candidate, degree=degree, age_bounds=age_bounds,
            )
            aic = -2.0 * float(fit.llf) + 2.0 * len(fit.params)
            candidates.append((fit, df_candidate, _aicc(aic, len(fit.params), n_events)))
        except ValueError as exc:
            attempts.append(f"spline df={df_candidate}: {exc}")

    try:
        fit = _phreg_spline_fit(data, age_col, time_col, status_col, spline_df=2, degree=2, age_bounds=age_bounds)
        aic = -2.0 * float(fit.llf) + 2.0 * len(fit.params)
        candidates.append((fit, 2, _aicc(aic, len(fit.params), n_events)))
    except ValueError as exc:
        attempts.append(f"quadratic df=2: {exc}")

    # Always include a plain linear age term as a candidate too, not just a
    # last resort -- it can legitimately win the AICc comparison even when a
    # spline also converges, if the extra flexibility isn't earning its keep.
    try:
        model = sm.PHReg.from_formula(f"{time_col} ~ {age_col}", data=data, status=status_col, ties="efron")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            fit = model.fit()
        _check_phreg_converged(fit, status_col, n_events, spline_df=1, degree=1)
        aic = -2.0 * float(fit.llf) + 2.0 * len(fit.params)
        candidates.append((fit, 1, _aicc(aic, len(fit.params), n_events)))
    except ValueError as exc:
        attempts.append(f"linear: {exc}")

    if not candidates:
        raise ValueError(
            f"No age term converged for '{status_col}' at any complexity. Attempts:\n  "
            + "\n  ".join(attempts)
        )

    best_fit, best_df, best_aic = min(candidates, key=lambda c: c[2])
    return best_fit, best_df


def hazard_ratio_curve_by_age(
    fit,
    age_grid: np.ndarray,
    ref_age: float,
    *,
    age_col: str = "age",
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Cause-specific hazard ratio across age_grid, relative to ref_age.

    Same delta-method construction as age_cfr_odds_ratio / the OR piece of
    relative_curves_by_age -- it's duck-typed against anything exposing a
    design-info-bearing model.data / params / cov_params(), so it works
    unchanged for a PHReg Cox fit here, not just the GLM logistic fits,
    since both are log-linear in the covariates (only the meaning of
    "linear predictor" differs: log-odds there, log-hazard here).
    """
    design_info = _get_design_info(fit)
    age_grid = np.asarray(age_grid, dtype=float)

    ref_row = np.asarray(patsy.build_design_matrices([design_info], {age_col: [float(ref_age)]})[0])[0]
    grid_rows = np.asarray(patsy.build_design_matrices([design_info], {age_col: age_grid})[0])

    params = np.asarray(fit.params)
    cov = np.asarray(fit.cov_params())
    z = norm.ppf(1 - alpha / 2)

    log_hr = grid_rows @ params - float(ref_row @ params)
    var_ref = float(ref_row @ cov @ ref_row.T)
    var_grid = np.einsum("ij,jk,ik->i", grid_rows, cov, grid_rows)
    cov_grid_ref = grid_rows @ cov @ ref_row.T
    se_log_hr = np.sqrt(np.clip(var_grid + var_ref - 2 * cov_grid_ref, 0, None))

    return pd.DataFrame(
        {
            "age": age_grid,
            "ref_age": float(ref_age),
            "hazard_ratio": np.exp(log_hr),
            "hr_lower_ci": np.exp(log_hr - z * se_log_hr),
            "hr_upper_ci": np.exp(log_hr + z * se_log_hr),
        }
    )


def _baseline_hazard_steps(fit) -> Tuple[np.ndarray, np.ndarray]:
    """Jump times and jump sizes of a PHReg fit's Breslow baseline cumulative hazard."""
    t, cumhaz, _ = fit.baseline_cumulative_hazard[0]
    t = np.asarray(t, dtype=float)
    cumhaz = np.asarray(cumhaz, dtype=float)
    jumps = np.diff(np.concatenate([[0.0], cumhaz]))
    return t, jumps


def cif_by_age_from_cause_specific_fits(
    fit_death,
    fit_recovery,
    age_grid: np.ndarray,
    *,
    age_col: str = "age",
    horizon: Optional[float] = None,
) -> pd.DataFrame:
    """CIF of death by age, recombined from two cause-specific Cox fits.

    Point estimates only -- a CI here would need the delta method propagated
    through the whole product integral (or a bootstrap), not done in this
    first pass.

    cfr_competing_risk1's raw nonparametric recursion works directly with
    count ratios (d_death/at_risk), which are always <= 1 by construction.
    Once hazard jumps are scaled per age by exp(linear predictor), that
    guarantee is gone -- a jump can come out above 1 for an age far from
    where the baseline was effectively centered (seen in practice: Yambuku's
    youngest ages pushed a scaled death-hazard jump past 1, and summing
    S(t-) * dH_death directly produced a "CIF" of 1.23). So each jump is
    converted to a proper bounded transition probability first,
    1 - exp(-total_hazard) split proportionally between the two causes by
    their hazard share, then combined via the standard product-limit
    survival -- this reduces to the same recursion as cfr_competing_risk1 in
    the small-hazard limit, but stays in [0, 1] regardless of scaling.
    """
    design_death = _get_design_info(fit_death)
    design_recovery = _get_design_info(fit_recovery)
    age_grid = np.asarray(age_grid, dtype=float)

    rows_death = np.asarray(patsy.build_design_matrices([design_death], {age_col: age_grid})[0])
    rows_recovery = np.asarray(patsy.build_design_matrices([design_recovery], {age_col: age_grid})[0])
    scale_death = np.exp(rows_death @ np.asarray(fit_death.params))
    scale_recovery = np.exp(rows_recovery @ np.asarray(fit_recovery.params))

    t_death, jump_death = _baseline_hazard_steps(fit_death)
    t_recovery, jump_recovery = _baseline_hazard_steps(fit_recovery)

    all_times = np.union1d(t_death, t_recovery)
    if horizon is not None:
        all_times = all_times[all_times <= horizon]

    jump_death_on_grid = pd.Series(jump_death, index=t_death).reindex(all_times, fill_value=0.0).to_numpy()
    jump_recovery_on_grid = pd.Series(jump_recovery, index=t_recovery).reindex(all_times, fill_value=0.0).to_numpy()

    cif = np.empty(age_grid.shape[0])
    for j in range(age_grid.shape[0]):
        dH_death = jump_death_on_grid * scale_death[j]
        dH_recovery = jump_recovery_on_grid * scale_recovery[j]
        total_hazard = dH_death + dH_recovery

        # bounded probability of leaving the "alive" state at each jump,
        # given alive just before it, split between the two causes by their
        # relative hazard share
        p_leave = -np.expm1(-total_hazard)  # = 1 - exp(-total_hazard), stable for small total_hazard
        with np.errstate(invalid="ignore", divide="ignore"):
            frac_death = np.where(total_hazard > 0, dH_death / total_hazard, 0.0)
        p_death = frac_death * p_leave

        S_prev = np.concatenate([[1.0], np.cumprod(1.0 - p_leave)[:-1]])
        cif[j] = float(np.sum(S_prev * p_death))

    return pd.DataFrame(
        {
            "age": age_grid,
            "cif_death": cif,
            "horizon": all_times[-1] if all_times.size else np.nan,
        }
    )


def cfr_competing_risks_by_age(
    df: pd.DataFrame,
    *,
    age_col: str = "age",
    start_col: str = "start_date",
    outcome_date_col: str = "outcome_date",
    event_col: str = "event",
    analysis_date: Optional[pd.Timestamp] = None,
    death_label: str = "death",
    recovery_label: str = "recovery",
    spline_df: int = 4,
    degree: int = 3,
    age_grid: Optional[np.ndarray] = None,
    n_grid: int = 100,
    ref_age: Optional[float] = None,
    age_bounds: Optional[Tuple[float, float]] = None,
    alpha: float = 0.05,
    horizon: Optional[float] = None,
) -> Dict[str, Any]:
    """Cause-specific hazards regression for death vs. recovery, with age
    entered via a B-spline -- the age-specific analogue of cfr_competing_risks.

    Pass `age_bounds` (and `age_grid`, `ref_age`) to fix the spline domain
    and reference age externally -- needed by the bootstrap CI below so
    every resample shares the same domain/reference as the point estimate
    (a with-replacement resample's own min/max age can only be a subset of
    the original's, so this is always a valid domain for it).

    Fits two Cox models (death-cause and recovery-cause), each
    time ~ bs(age); an individual is censored (removed from the risk set)
    for a given cause's model the moment their own event, or the competing
    event, or administrative censoring occurs -- the same risk set
    cfr_competing_risks/cfr_competing_risk1 use.

    Each cause's model starts at spline_df basis functions and backs off to
    fewer (then to a plain linear age term) if it fails to converge --
    see _fit_phreg_with_fallback. death_df_used / recovery_df_used in the
    returned dict report what actually got used (1 = linear fallback), so
    it's visible when a curve came from a lower-flexibility model than
    requested.

    Returns:
      - death_hazard_ratio / recovery_hazard_ratio: cause-specific HR(age)
        relative to a reference age, each with a Wald CI.
      - cif: the recombined CIF-of-death-by-age curve (point estimate only).
    """
    if df[age_col].dropna().empty:
        raise ValueError(f"No rows with '{age_col}' recorded; nothing to fit.")

    time_event = _prepare_individual_time_data(
        df,
        start_col=start_col,
        outcome_date_col=outcome_date_col,
        event_col=event_col,
        analysis_date=analysis_date,
        death_label=death_label,
        recovery_label=recovery_label,
    )
    work = time_event.join(df[[age_col]]).dropna(subset=[age_col])
    work["status_death"] = (work["event"] == death_label).astype(int)
    work["status_recovery"] = (work["event"] == recovery_label).astype(int)

    n_deaths = int(work["status_death"].sum())
    n_recoveries = int(work["status_recovery"].sum())
    if n_deaths < 2 or n_recoveries < 2:
        raise ValueError(
            f"Need at least 2 deaths and 2 recoveries with age recorded to fit "
            f"cause-specific hazard models (got {n_deaths} deaths, {n_recoveries} recoveries)."
        )

    if age_bounds is None:
        age_bounds = (float(work[age_col].min()), float(work[age_col].max()))
    if age_grid is None:
        age_grid = np.linspace(age_bounds[0], age_bounds[1], n_grid)
    if ref_age is None:
        ref_age = float(work[age_col].median())

    fit_death, death_df_used = _fit_phreg_with_fallback(
        work, age_col, "time", "status_death", spline_df=spline_df, degree=degree, age_bounds=age_bounds
    )
    fit_recovery, recovery_df_used = _fit_phreg_with_fallback(
        work, age_col, "time", "status_recovery", spline_df=spline_df, degree=degree, age_bounds=age_bounds
    )

    death_hr = hazard_ratio_curve_by_age(fit_death, age_grid, ref_age, age_col=age_col, alpha=alpha)
    recovery_hr = hazard_ratio_curve_by_age(fit_recovery, age_grid, ref_age, age_col=age_col, alpha=alpha)
    cif = cif_by_age_from_cause_specific_fits(fit_death, fit_recovery, age_grid, age_col=age_col, horizon=horizon)

    return {
        "fit_death": fit_death,
        "fit_recovery": fit_recovery,
        "death_hazard_ratio": death_hr,
        "recovery_hazard_ratio": recovery_hr,
        "cif": cif,
        "n": int(len(work)),
        "n_deaths": n_deaths,
        "n_recoveries": n_recoveries,
        "death_df_used": death_df_used,
        "recovery_df_used": recovery_df_used,
    }


def cfr_competing_risks_by_age_ci(
    df: pd.DataFrame,
    *,
    age_col: str = "age",
    start_col: str = "start_date",
    outcome_date_col: str = "outcome_date",
    event_col: str = "event",
    analysis_date: Optional[pd.Timestamp] = None,
    death_label: str = "death",
    recovery_label: str = "recovery",
    spline_df: int = 4,
    degree: int = 3,
    age_grid: Optional[np.ndarray] = None,
    n_grid: int = 100,
    ref_age: Optional[float] = None,
    alpha: float = 0.05,
    horizon: Optional[float] = None,
    n_boot: int = 200,
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    """cfr_competing_risks_by_age plus a bootstrap CI on the CIF curve.

    The CIF there is a point estimate only -- an analytic CI would need the
    delta method propagated through the whole product-integral recursion,
    and that's a fragile thing to hand-derive here specifically because the
    death and recovery models each independently fall back from a spline to
    a plain linear term if they fail to converge (_fit_phreg_with_fallback),
    so the parameter vector a gradient would be taken against doesn't even
    have a fixed dimension across cases.

    Case-resampling bootstrap sidesteps that entirely: each resample just
    refits both cause-specific models and recomputes the CIF with the exact
    same, already-validated cfr_competing_risks_by_age, whatever fallback
    level it happens to land on. age_grid/age_bounds/ref_age/horizon are
    all fixed to the point estimate's values so every resample's curve is
    directly comparable at the same x-axis positions (a with-replacement
    resample's own age range is always a subset of the original's, so this
    domain is always valid for it -- same reasoning as the age-time
    snapshots sharing one fixed domain).

    Percentile bootstrap (the 2.5th/97.5th percentile of the resampled CIF
    at each age, for the default alpha=0.05) -- the simplest, most standard
    variant. Resamples where either cause-specific model fails to converge
    even at the linear fallback are skipped and counted, not treated as an
    error -- that's an expected outcome for a modest-sized outbreak, not a
    bug, the same way individual snapshots get skipped in
    age_time_relative_risk_curves.
    """
    point = cfr_competing_risks_by_age(
        df,
        age_col=age_col,
        start_col=start_col,
        outcome_date_col=outcome_date_col,
        event_col=event_col,
        analysis_date=analysis_date,
        death_label=death_label,
        recovery_label=recovery_label,
        spline_df=spline_df,
        degree=degree,
        age_grid=age_grid,
        n_grid=n_grid,
        ref_age=ref_age,
        alpha=alpha,
        horizon=horizon,
    )
    age_grid_fixed = point["cif"]["age"].to_numpy()
    age_bounds_fixed = (float(age_grid_fixed.min()), float(age_grid_fixed.max()))
    ref_age_fixed = float(point["death_hazard_ratio"]["ref_age"].iloc[0])
    horizon_fixed = point["cif"]["horizon"].iloc[0]

    rng = np.random.default_rng(seed)
    n = len(df)
    boot_cifs = []
    n_failed = 0
    for _ in range(n_boot):
        resample = df.iloc[rng.integers(0, n, size=n)]
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                res_b = cfr_competing_risks_by_age(
                    resample,
                    age_col=age_col,
                    start_col=start_col,
                    outcome_date_col=outcome_date_col,
                    event_col=event_col,
                    analysis_date=analysis_date,
                    death_label=death_label,
                    recovery_label=recovery_label,
                    spline_df=spline_df,
                    degree=degree,
                    age_grid=age_grid_fixed,
                    ref_age=ref_age_fixed,
                    age_bounds=age_bounds_fixed,
                    alpha=alpha,
                    horizon=horizon_fixed,
                )
            boot_cifs.append(res_b["cif"]["cif_death"].to_numpy())
        except ValueError:
            n_failed += 1
            continue

    n_success = len(boot_cifs)
    if n_success < max(10, n_boot // 4):
        warnings.warn(
            f"Only {n_success}/{n_boot} bootstrap replicates converged -- CI may be unreliable "
            f"({n_failed} failed, typically from too few deaths/recoveries landing in a resample)."
        )

    cif_out = point["cif"].copy()
    if n_success > 0:
        boot_matrix = np.vstack(boot_cifs)
        cif_out["cif_lower_ci"] = np.percentile(boot_matrix, 100 * alpha / 2, axis=0)
        cif_out["cif_upper_ci"] = np.percentile(boot_matrix, 100 * (1 - alpha / 2), axis=0)
    else:
        cif_out["cif_lower_ci"] = np.nan
        cif_out["cif_upper_ci"] = np.nan

    point["cif"] = cif_out
    point["n_boot"] = n_boot
    point["n_boot_success"] = n_success
    return point


# ---------------------------------------------------------------------------
# Binary-covariate group comparison (e.g. healthcare worker vs. not)
#
# A binary covariate has no continuum to model a shape over, unlike age --
# splitting into groups and applying the existing pooled estimators to each
# is the natural, lossless analysis (a regression on a single 0/1 covariate
# reduces to exactly the group-specific estimates; it isn't an approximation
# the way binning a continuous covariate would be). What group splitting
# alone doesn't give you is a formal comparison between the groups with its
# own CI -- group_death_ratio and group_hazard_ratio below add that.
# ---------------------------------------------------------------------------


def cfr_group_comparison(
    df: pd.DataFrame,
    *,
    group_col: str = "is_hcw",
    start_col: str = "start_date",
    outcome_date_col: str = "outcome_date",
    event_col: str = "event",
    analysis_date: Optional[pd.Timestamp] = None,
    death_label: str = "death",
    recovery_label: str = "recovery",
    family: str = "gamma",
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Naive, resolved, competing-risks (Aalen-Johansen), adapted-KM (Ghani),
    and parametric-mixture CFR, computed separately for each level of a
    binary grouping column.

    Delay-adjusted (Nishiura) is deliberately not included here -- it
    operates on daily aggregate case/death arrays (a deconvolution over
    calendar time), which doesn't have a natural per-group analogue the way
    the other five individual-level estimators do.

    Rows with a missing group_col value are dropped (e.g. "possible HCW" /
    unknown occupation, which was left as NaN rather than guessed).
    """
    work = df.dropna(subset=[group_col]).copy()
    groups = sorted(work[group_col].unique(), key=str)

    rows = []
    for g in groups:
        sub = work.loc[work[group_col] == g]
        time_event = _prepare_individual_time_data(
            sub,
            start_col=start_col,
            outcome_date_col=outcome_date_col,
            event_col=event_col,
            analysis_date=analysis_date,
            death_label=death_label,
            recovery_label=recovery_label,
        ).assign(time=lambda d: d["time"])

        n = len(time_event)
        n_deaths = int((time_event["event"] == death_label).sum())
        n_recoveries = int((time_event["event"] == recovery_label).sum())

        naive = cfr_naive(n_deaths, n)
        resolved = (
            cfr_resolved_cohort(n_deaths, n_recoveries)
            if n_recoveries > 0
            else {"estimate": np.nan, "lower_ci": np.nan, "upper_ci": np.nan}
        )

        try:
            aj = cfr_competing_risks(time_event, time_col="time", event_col="event", alpha=alpha)
        except Exception as exc:
            aj = {"estimate": np.nan, "lower_ci": np.nan, "upper_ci": np.nan, "error": str(exc)}

        try:
            km = cfr_ghani_2005_km(time_event, time_col="time", event_col="event", alpha=alpha)
        except Exception as exc:
            km = {"estimate": np.nan, "lower_ci": np.nan, "upper_ci": np.nan, "error": str(exc)}

        try:
            mix = cfr_parametric_mixture(time_event, time_col="time", event_col="event", family=family, alpha=alpha)
        except Exception as exc:
            mix = {"estimate": np.nan, "lower_ci": np.nan, "upper_ci": np.nan, "error": str(exc)}

        for method_name, res in [
            ("naive", naive),
            ("resolved", resolved),
            ("competing_risks", aj),
            ("kaplan_meier_ghani", km),
            ("parametric_mixture", mix),
        ]:
            rows.append(
                {
                    "group": g,
                    "method": method_name,
                    "estimate": res.get("estimate", np.nan),
                    "lower_ci": res.get("lower_ci", np.nan),
                    "upper_ci": res.get("upper_ci", np.nan),
                    "n": n,
                    "n_deaths": n_deaths,
                    "n_recoveries": n_recoveries,
                }
            )

    return pd.DataFrame(rows)


def group_death_ratio(
    df: pd.DataFrame,
    *,
    group_col: str = "is_hcw",
    method: str = "naive",
    start_col: str = "start_date",
    outcome_date_col: str = "outcome_date",
    event_col: str = "event",
    analysis_date: Optional[pd.Timestamp] = None,
    death_label: str = "death",
    recovery_label: str = "recovery",
    alpha: float = 0.05,
) -> Dict[str, float]:
    """OR and RR of death for group_col=True vs. False, with Wald CIs, via a
    single-covariate logistic regression -- the formal-comparison analogue
    of cfr_naive_by_age/cfr_resolved_by_age, but for a binary covariate
    instead of a continuous spline (so no basis functions, no convergence
    fallback needed -- one parameter is about as stable as a fit gets).

    Reuses relative_curves_by_age's delta-method machinery directly (it's
    generic in the covariate, not spline-specific) rather than re-deriving
    the same OR/RR math a third time. For a single saturated binary
    covariate this isn't an approximation: the model's predicted
    probabilities equal the raw empirical group proportions exactly, so
    reference_cfr/exposed_cfr below match what you'd get from cfr_naive on
    each group directly.

    Use RR (not OR) if you want reference_cfr * risk_ratio to reconstruct
    the exposed group's CFR -- CFR is rarely a rare outcome, so OR does not
    approximate RR here (see the ties/tie-breaking-style discussion earlier:
    OR is always more extreme than RR away from the rare-outcome regime).

    method="naive": every row counts towards the denominator (matches
    cfr_naive's deaths/cases definition).
    method="resolved": restricted to rows with a death/recovery outcome
    (matches cfr_resolved_cohort's deaths/(deaths+recoveries) definition).
    """
    if method not in {"naive", "resolved"}:
        raise ValueError("method must be 'naive' or 'resolved'")

    time_event = _prepare_individual_time_data(
        df,
        start_col=start_col,
        outcome_date_col=outcome_date_col,
        event_col=event_col,
        analysis_date=analysis_date,
        death_label=death_label,
        recovery_label=recovery_label,
    )
    work = time_event.join(df[[group_col]]).dropna(subset=[group_col])
    if method == "resolved":
        work = work.loc[work["event"].isin([death_label, recovery_label])]
    work["died"] = (work["event"] == death_label).astype(int)
    work["group"] = work[group_col].astype(bool).astype(int)

    if work["died"].nunique() < 2 or work["group"].nunique() < 2:
        raise ValueError("Need both outcomes and both groups represented to fit a death ratio.")

    fit = smf.glm("died ~ group", data=work, family=sm.families.Binomial()).fit()
    ratios = relative_curves_by_age(fit, age_grid=np.array([1.0]), ref_age=0.0, age_col="group", alpha=alpha).iloc[0]
    reference_cfr = float(expit(fit.params["Intercept"]))

    return {
        "reference_cfr": reference_cfr,
        "exposed_cfr": float(ratios["cfr"]),
        "odds_ratio": float(ratios["odds_ratio"]),
        "or_lower_ci": float(ratios["or_lower_ci"]),
        "or_upper_ci": float(ratios["or_upper_ci"]),
        "risk_ratio": float(ratios["risk_ratio"]),
        "rr_lower_ci": float(ratios["rr_lower_ci"]),
        "rr_upper_ci": float(ratios["rr_upper_ci"]),
        "n": int(len(work)),
    }


def group_hazard_ratio(
    df: pd.DataFrame,
    *,
    group_col: str = "is_hcw",
    cause: str = "death",
    start_col: str = "start_date",
    outcome_date_col: str = "outcome_date",
    event_col: str = "event",
    analysis_date: Optional[pd.Timestamp] = None,
    death_label: str = "death",
    recovery_label: str = "recovery",
    alpha: float = 0.05,
) -> Dict[str, float]:
    """Cause-specific hazard ratio for group_col=True vs. False, with a Wald
    CI -- the group-comparison analogue of cfr_competing_risks_by_age, but
    for a binary covariate. Same risk-set convention as cfr_competing_risks
    (the other cause + administrative censoring remove someone from the
    risk set), same ties="efron" fit.
    """
    if cause not in {"death", "recovery"}:
        raise ValueError("cause must be 'death' or 'recovery'")

    time_event = _prepare_individual_time_data(
        df,
        start_col=start_col,
        outcome_date_col=outcome_date_col,
        event_col=event_col,
        analysis_date=analysis_date,
        death_label=death_label,
        recovery_label=recovery_label,
    )
    work = time_event.join(df[[group_col]]).dropna(subset=[group_col])
    target_label = death_label if cause == "death" else recovery_label
    work["status"] = (work["event"] == target_label).astype(int)
    work["group"] = work[group_col].astype(bool).astype(int)

    n_events = int(work["status"].sum())
    if n_events < 2 or work["group"].nunique() < 2:
        raise ValueError(f"Need at least 2 '{cause}' events and both groups represented to fit a hazard ratio.")

    model = sm.PHReg.from_formula("time ~ group", data=work, status="status", ties="efron")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        fit = model.fit()
    _check_phreg_converged(fit, f"status_{cause}", n_events, spline_df=1, degree=1)

    coef = float(fit.params[0])
    se = float(np.sqrt(np.asarray(fit.cov_params())[0, 0]))
    z = norm.ppf(1 - alpha / 2)

    return {
        "hazard_ratio": float(np.exp(coef)),
        "lower_ci": float(np.exp(coef - z * se)),
        "upper_ci": float(np.exp(coef + z * se)),
        "n": int(len(work)),
        "n_events": n_events,
    }


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

    n_dead = int(np.sum(d_death))
    n_recovered = int(np.sum(d_rec))
    if n_dead == 0 or n_recovered == 0:
        # theta0/(theta0+theta1) is mathematically well-defined here (not a
        # 0/0 case) but trivially collapses to exactly 1.0 (or 0.0) whenever
        # one outcome has never been observed -- the formula is built on
        # Ghani et al.'s assumption that unresolved cases will eventually
        # split death:recovery in the same ratio observed so far, and with
        # zero of one outcome that assumption is completely untested, not
        # just uncertain. Same convention as cfr_resolved_cohort's
        # zero-denominator guard.
        return {
            "estimate": np.nan,
            "lower_ci": np.nan,
            "upper_ci": np.nan,
            "se_cfr": np.nan,
            "theta0": float(theta0),
            "theta1": float(theta1),
            "n_cases": int(len(times)),
            "n_dead": n_dead,
            "n_recovered": n_recovered,
            "ci_method": "greenwood" if greenwood else "alt",
        }

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

    n_dead = int(np.sum(events == death_label))
    n_recovered = int(np.sum(events == recovery_label))
    if n_dead == 0 or n_recovered == 0:
        # Fitting two separate time-to-event sub-distributions (one per
        # outcome) needs observations of both outcomes -- with zero of
        # either, that sub-distribution is unidentifiable and the optimizer
        # will just drift to a boundary solution (p -> 0 or 1) rather than
        # failing loudly. Same convention as cfr_ghani_2005_km's guard.
        return {
            "estimate": np.nan,
            "lower_ci": np.nan,
            "upper_ci": np.nan,
            "family": family,
            "success": False,
            "message": f"Need both outcomes to fit the mixture model (got {n_dead} deaths, {n_recovered} recoveries).",
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

    def _apply_pseudo_realtime_downward_correction(series: np.ndarray, current_idx: int) -> None:
        """
        Apply a downward revision at the current row only to the historical values
        used for current-and-future estimates.

        This keeps already-emitted CFR rows unchanged, while ensuring the working
        cumulative history remains consistent from the revision date onward.
        """
        if current_idx <= 0:
            return

        current = series[current_idx]
        prev = series[current_idx - 1]
        if not np.isfinite(current) or not np.isfinite(prev) or current >= prev:
            return
        ratio = current/prev

        series[:current_idx] = np.where(
            np.isfinite(series[:current_idx]),
            np.rint(series[:current_idx]*ratio),
            series[:current_idx],
        )

    group_iter = [(None, work)] if not group_cols else list(work.groupby(group_cols, dropna=False, sort=False))

    for _, g in group_iter:
        g = g.sort_values(date_col).reset_index(drop=True)

        cases_raw = g[cases_col].ffill().fillna(0.0).to_numpy(dtype=float) if is_cumulative else g[cases_col].fillna(0.0).to_numpy(dtype=float)
        deaths_raw = g[deaths_col].ffill().fillna(0.0).to_numpy(dtype=float) if is_cumulative else g[deaths_col].fillna(0.0).to_numpy(dtype=float)

        recovered_raw = None
        if recovered_col and recovered_col in g.columns:
            recovered_raw = g[recovered_col].ffill().fillna(0.0).to_numpy(dtype=float) if is_cumulative else g[recovered_col].fillna(0.0).to_numpy(dtype=float)

        # Working copies for pseudo-real-time revision handling.
        # Earlier emitted rows stay unchanged; only the current row and later rows
        # use the revised historical series.
        cases = cases_raw.copy()
        deaths = deaths_raw.copy()
        recovered = recovered_raw.copy() if recovered_raw is not None else None

        for i, dt in enumerate(g[date_col]):
            if is_cumulative:
                _apply_pseudo_realtime_downward_correction(cases, i)
                _apply_pseudo_realtime_downward_correction(deaths, i)
                if recovered is not None:
                    _apply_pseudo_realtime_downward_correction(recovered, i)

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
                    cases_inc = np.diff(np.r_[0.0, cases[: i + 1]])
                    deaths_inc = np.diff(np.r_[0.0, deaths[: i + 1]])
                    recovered_inc = np.diff(np.r_[0.0, recovered[: i + 1]]) if recovered is not None else None
                    est, lo, hi = _unpack_est_ci(
                        cfr_delay_adjusted_nishiura(
                            deaths=deaths_inc,
                            cases=cases_inc,
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

    # Evaluate on every calendar day, not just days with new starts.
    start_date = pd.to_datetime(linelist["start_date"].min()).normalize()

    end_candidates = [linelist["start_date"].max(), linelist["outcome_date"].max()]
    end_candidates = [pd.to_datetime(d).normalize() for d in end_candidates if pd.notna(d)]
    end_date = max(end_candidates) if end_candidates else start_date

    dates = pd.date_range(start=start_date, end=end_date, freq="D")
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
    "parse_age_bands",
    "cfr_naive_by_age",
    "cfr_resolved_by_age",
    "age_cfr_odds_ratio",
    "relative_curves_by_age",
    "age_time_relative_risk_curves",
    "cfr_delay_adjusted_nishiura",
    "cfr_competing_risks",
    "cfr_competing_risks_by_age",
    "cfr_competing_risks_by_age_ci",
    "cfr_group_comparison",
    "group_death_ratio",
    "group_hazard_ratio",
    "hazard_ratio_curve_by_age",
    "cif_by_age_from_cause_specific_fits",
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
        cases_col="confirmed_cases",
        deaths_col="confirmed_deaths",
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
    case_categories: list[str] | None = None,
) -> pd.DataFrame:

    work = df.copy()

    # Restrict to requested case categories
    if case_categories is not None:
        if "Case_definition" not in work.columns:
            raise ValueError(
                "case_categories was supplied but 'Case_definition' "
                "is not present in the Rosello dataset."
            )

        allowed = {
            str(x).strip().lower()
            for x in case_categories
        }

        status = (
            work["Case_definition"]
            .astype(str)
            .str.strip()
            .str.lower()
        )

        work = work.loc[status.isin(allowed)].copy()

    work["analysis_origin_date"] = _to_datetime(
        work[start_date_col],
        dayfirst=True,
    )

    outcome = work["Outcome"].map(_safe_lower)

    event = pd.Series(
        np.where(
            outcome.isin({"dead", "death"}),
            "death",
            np.where(
                outcome.isin({"alive", "recovered", "discharged"}),
                "recovery",
                "censored",
            ),
        ),
        index=work.index,
    )

    outcome_date = pd.Series(
        pd.NaT,
        index=work.index,
        dtype="datetime64[ns]",
    )

    if "Date_of_Death" in work.columns:
        outcome_date = outcome_date.fillna(
            _to_datetime(work["Date_of_Death"], dayfirst=True)
        )

    if "Date_hospital_discharge" in work.columns:
        outcome_date = outcome_date.fillna(
            _to_datetime(work["Date_hospital_discharge"], dayfirst=True)
        )

    if "Date_disease_ended" in work.columns:
        outcome_date = outcome_date.fillna(
            _to_datetime(work["Date_disease_ended"], dayfirst=True)
        )

    work = pd.DataFrame(
        {
            "start_date": work["analysis_origin_date"],
            "outcome_date": outcome_date,
            "event": event,
        }
    )

    if "Age" in df.columns:
        work["age"] = _coerce_numeric(df["Age"])

    if "Occupation" in df.columns:
        occ = df["Occupation"].astype(str).str.strip()
        # "possible HCW" is left as unknown (NaN) rather than guessed either
        # way -- it's a small category (a handful of records per outbreak)
        # and guessing wrong would bias whichever group it's forced into.
        is_hcw = pd.Series(np.nan, index=df.index, dtype="object")
        is_hcw[occ == "HCW"] = True
        is_hcw[occ.isin(["no HCW", "Housewife", "Other", "Student", "Child"])] = False
        work["is_hcw"] = is_hcw

    return work.dropna(subset=["start_date"]).copy()
def adapt_rosello_to_linelist_by_outbreak(
    df: pd.DataFrame,
    outbreak_col: str = "Outbreak",
    case_categories_by_outbreak: dict[str, list[str]] | None = None,
    start_date_col_by_outbreak: dict[str, str] | None = None,
    default_start_date_col: str = "Date_of_onset_symp",
) -> dict[str, pd.DataFrame]:
    if outbreak_col not in df.columns:
        raise ValueError(f"Missing required column: {outbreak_col}")

    out = {}
    for outbreak_value, group in df.groupby(outbreak_col, dropna=False):
        key = "missing" if pd.isna(outbreak_value) else str(outbreak_value)

        case_categories = None
        if case_categories_by_outbreak is not None:
            case_categories = case_categories_by_outbreak.get(key)

        start_date_col = default_start_date_col
        if start_date_col_by_outbreak is not None:
            start_date_col = start_date_col_by_outbreak.get(key, default_start_date_col)

        ll = adapt_rosello_to_linelist(
            group,
            start_date_col=start_date_col,
            case_categories=case_categories,
        ).copy()

        ll[outbreak_col] = outbreak_value
        out[key] = ll.reset_index(drop=True)

    return out

def adapt_uganda_to_linelist(
    df: pd.DataFrame,
    confirmed_only: bool = True,
) -> pd.DataFrame:

    work = df.copy()

    # Restrict analysis to confirmed cases if requested
    if confirmed_only:
        if "Case_status" not in work.columns:
            raise ValueError(
                "confirmed_only=True but 'Case_status' is not present in the Uganda dataset."
            )

        work = work.loc[
            work["Case_status"]
            .astype(str)
            .str.strip()
            .str.lower()
            .eq("confirmed")
        ].copy()

    work["analysis_origin_date"] = _to_datetime(
        work["Date_onset"],
        dayfirst=False
    )

    fallback = work["analysis_origin_date"].copy()

    for c in ["Date_confirmation", "Date_hospitalisation", "Date_isolation"]:
        if c in work.columns:
            fallback = fallback.fillna(
                _to_datetime(work[c], dayfirst=False)
            )

    outcome = work["Outcome"].map(_safe_lower)

    event = pd.Series(
        np.where(
            outcome.isin({"death", "dead", "died", "deceased"}),
            "death",
            np.where(
                outcome.isin(
                    {"recovery", "recovered", "alive", "discharged"}
                ),
                "recovery",
                "censored",
            ),
        ),
        index=work.index,
    )

    outcome_date = pd.Series(
        pd.NaT,
        index=work.index,
        dtype="datetime64[ns]"
    )

    if "Date_Death" in work.columns:
        outcome_date = outcome_date.fillna(
            _to_datetime(work["Date_Death"], dayfirst=False)
        )

    if "Date_Recovered" in work.columns:
        outcome_date = outcome_date.fillna(
            _to_datetime(work["Date_Recovered"], dayfirst=False)
        )

    work = pd.DataFrame(
        {
            "start_date": fallback,
            "outcome_date": outcome_date,
            "event": event,
        }
    )

    if "Healthcare_worker" in df.columns:
        # Field is "Y" or blank, no explicit "N" -- blank is treated as "not
        # a healthcare worker" per user confirmation, not "unknown".
        work["is_hcw"] = df["Healthcare_worker"].astype(str).str.strip().eq("Y")

    work = work.dropna(subset=["start_date"]).copy()

    return work


def adapt_kenema_to_linelist(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    work["analysis_origin_date"] = _to_datetime(work["Date of admission"], dayfirst=False)
    fallback = work["analysis_origin_date"].copy()
    outcome = work["Outcome"].map(_safe_lower)
    event = pd.Series(
        np.where(
            outcome.isin({"died"}),
            "death",
            np.where(outcome.isin({"discharged"}), "recovery", "censored"),
        ),
        index=work.index,
    )
    outcome_date = pd.Series(pd.NaT, index=work.index, dtype="datetime64[ns]")
    if "Date of discharge" in work.columns:
        outcome_date = outcome_date.fillna(_to_datetime(work["Date of discharge"], dayfirst=False))
    work = pd.DataFrame({"start_date": fallback, "outcome_date": outcome_date, "event": event})
    if "Age" in df.columns:
        work["age"] = _coerce_numeric(df["Age"])
    work = work.dropna(subset=["start_date"]).copy()
    return work

def adapt_guinea_to_counts(df: pd.DataFrame) -> pd.DataFrame:
    return standardize_count_table(
        df,
        date_col="Date_case",
        cases_col="Cases",
        deaths_col="Deaths",
        recovered_col="Recoveries",
        is_cumulative=True,
    )
