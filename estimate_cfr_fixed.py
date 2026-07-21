# Not useful 
# Create functions to calculate CFR given correct input data
# load libraries
import os
import pandas as pd
import numpy as np
from scipy.optimize import minimize
from scipy.stats import beta
from scipy.stats import gamma as gamma_dist
import matplotlib.pyplot as plt

try:
    from rpy2.robjects.packages import importr
    from rpy2.robjects import pandas2ri
    from rpy2.robjects.conversion import localconverter
    import rpy2.robjects as ro

except ImportError:
    importr = None
    pandas2ri = None
    localconverter = None
    ro = None

if importr is not None:
    try:
        cfr = importr('cfr')
    except Exception as e:
        cfr = None
        print(f"Error importing cfr package: {e}")
        print("Please install cfr in R with: install.packages('cfr')")
else:
    cfr = None

# Naive
def naive_1(deaths, cases):
    """
    Calculate naive case fatality risk (CFR).
    
    Parameters:
    - deaths: total number of deaths
    - cases: total number of cases
    
    Returns:
    - Naive CFR as the ratio of deaths to cases
    """
    return deaths/cases


# Resolved cohort
def resolved_cohort_2(deaths, recoveries):
    """
    Calculate CFR using the resolved cohort method.
    
    Parameters:
    - deaths: total number of deaths
    - recoveries: total number of recoveries
    
    Returns:
    - Resolved cohort CFR as the ratio of deaths to total resolved outcomes (deaths + recoveries)
    """
    return deaths/(deaths+recoveries)

# Delay-adjusted Nishiura cCFR.
# Uses R package "cfr" and requires input of the paramters of the delay distribution.
# Requires cases and deaths per day, with column names in data as 'date', 'cases', and 'deaths'
def delay_adjusted_3(data, 
                     delay_shape=2.40,
                     delay_scale=3.33):
    """
    Calculate delay-adjusted CFR.

    If rpy2/cfr are available, the function will still try to use them, but
    it now falls back to a pure-Python day-by-day implementation so the method
    keeps working in environments without R.
    """
    try:
        df = data.copy()

        if "date" not in df.columns or "cases" not in df.columns or "deaths" not in df.columns:
            raise ValueError("Input data must have columns: date, cases, deaths")

        df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()
        df = df[df["date"].notna()].copy()
        if df.empty:
            raise ValueError("No usable rows after parsing dates")

        daily = summarise_daily_cfr_estimates(df[["date", "cases", "deaths"]], delay_shape=delay_shape, delay_scale=delay_scale)
        result = {
            "cfr": float(daily["method3_delay_adjusted"].iloc[-1]) if daily["method3_delay_adjusted"].notna().any() else np.nan,
            "daily_estimates": daily,
            "n_days": int(len(daily)),
            "delay_shape": float(delay_shape),
            "delay_scale": float(delay_scale),
        }

        if pandas2ri is not None and ro is not None and cfr is not None and localconverter is not None:
            try:
                with localconverter(ro.default_converter + pandas2ri.converter):
                    r_data = ro.conversion.py2rpy(df)
                    ro.globalenv['data'] = r_data
                    r_code = f"""
                    library(cfr)
                    data$date <- as.Date(data$date, format = "%Y-%m-%d")
                    cfr_result <- cfr_static(
                        data = data,
                        delay_density = function(x) dgamma(x, shape = {delay_shape}, scale = {delay_scale})
                    )
                    cfr_result
                    """
                    r_result = ro.r(r_code)
                    result["r_result"] = r_result
                    result["r_available"] = True
            except Exception:
                result["r_available"] = False
        return result
    except Exception as e:
        print(f"Error calculating delay-adjusted CFR: {e}")
        return None

# Kaplan Meier -- I did get copilot to write this based off a stata package so errors are possible
def kaplan_meier_5(data,
                   dead_col="dead",
                   rec_col="rec",
                   time_col="t",
                   origin_col=None,
                   cens=None,
                   greenwood=False,
                   untrans=False):
    """
    Calculate CFR using the adapted Kaplan-Meier estimator from Ghani et al. (2005).

    Parameters:
    - data: pandas DataFrame with individual-level outcome data
    - dead_col: indicator column for death (1 = death, 0 = otherwise)
    - rec_col: indicator column for recovery (1 = recovery, 0 = otherwise)
    - time_col: time-to-event column
    - origin_col: optional origin-time column; if omitted, origin is assumed to be 0
    - cens: optional censoring time; observations after this time are censored
    - greenwood: whether to use Greenwood-style variance for theta intervals
    - untrans: whether to use untransformed CFR confidence intervals

    Returns:
    - Dictionary matching the Stata casefat output as closely as practical
    """

    def _jeffreys_interval(successes, trials, alpha=0.05):
        if trials == 0:
            return np.nan, np.nan, np.nan, np.nan
        estimate = successes / trials
        lower = beta.ppf(alpha / 2, successes + 0.5, trials - successes + 0.5)
        upper = beta.ppf(1 - alpha / 2, successes + 0.5, trials - successes + 0.5)
        standard_error = np.sqrt(estimate * (1 - estimate) / trials)
        return estimate, standard_error, lower, upper

    try:
        df = data.copy()

        required_columns = {dead_col, rec_col, time_col}
        missing_columns = required_columns.difference(df.columns)
        if missing_columns:
            raise ValueError(f"Missing required columns: {', '.join(sorted(missing_columns))}")

        if origin_col is None:
            df["_origin"] = 0
            origin_col = "_origin"

        df[dead_col] = df[dead_col].astype(int)
        df[rec_col] = df[rec_col].astype(int)

        if ((df[dead_col] == 1) & (df[rec_col] == 1)).any():
            raise ValueError(f"Cannot have both {dead_col}=1 and {rec_col}=1 for the same observation")

        if cens is not None:
            after_censor = df[time_col] > cens
            df.loc[after_censor, [dead_col, rec_col]] = 0
            df.loc[after_censor, time_col] = cens

        df = df.sort_values(by=[time_col]).reset_index(drop=True)

        n_total = len(df)
        n_dead = int(df[dead_col].sum())
        n_rec = int(df[rec_col].sum())
        n_event = n_dead + n_rec
        n_cens = n_total - n_event

        event_times = sorted(df.loc[(df[dead_col] == 1) | (df[rec_col] == 1), time_col].unique())

        theta = 1.0
        theta_dead = 0.0
        theta_rec = 0.0
        theta_history = []
        h_dead_history = []
        h_rec_history = []
        n_dead_history = []
        n_rec_history = []

        for t in event_times:
            risk_set = df[time_col] >= t
            n_t = int(risk_set.sum())
            d_dead_t = int(((df[time_col] == t) & (df[dead_col] == 1)).sum())
            d_rec_t = int(((df[time_col] == t) & (df[rec_col] == 1)).sum())
            d_event_t = d_dead_t + d_rec_t

            h_dead = d_dead_t / n_t if n_t else 0.0
            h_rec = d_rec_t / n_t if n_t else 0.0
            h_event = d_event_t / n_t if n_t else 0.0

            theta_history.append(theta)
            h_dead_history.append(h_dead)
            h_rec_history.append(h_rec)
            n_dead_history.append(n_t)
            n_rec_history.append(n_t)

            theta_dead += h_dead * theta
            theta_rec += h_rec * theta
            theta *= (1 - h_event)

        cfr_point = theta_dead / (theta_dead + theta_rec) if (theta_dead + theta_rec) > 0 else np.nan

        e1, se_e1, le1, ue1 = _jeffreys_interval(n_dead, n_total)
        e2, se_e2, le2, ue2 = _jeffreys_interval(n_dead, n_event)

        if len(event_times) == 0:
            theta0 = np.nan
            theta1 = np.nan
            se_cfr = np.nan
            lcfr = np.nan
            ucfr = np.nan
        else:
            theta0 = theta_dead
            theta1 = theta_rec

            theta_arr = np.asarray(theta_history, dtype=float)
            h_dead_arr = np.asarray(h_dead_history, dtype=float)
            h_rec_arr = np.asarray(h_rec_history, dtype=float)
            n_dead_arr = np.asarray(n_dead_history, dtype=float)
            n_rec_arr = np.asarray(n_rec_history, dtype=float)

            if greenwood:
                cumulative_var = 0.0
                var_theta = []
                running_theta = 1.0
                for t in event_times:
                    n_t = int((df[time_col] >= t).sum())
                    d_event_t = int((((df[time_col] == t) & (df[dead_col] == 1)) | ((df[time_col] == t) & (df[rec_col] == 1))).sum())
                    if n_t > d_event_t and n_t > 0:
                        cumulative_var += d_event_t / (n_t * (n_t - d_event_t))
                    running_theta *= (1 - (d_event_t / n_t if n_t else 0.0))
                    var_theta.append((running_theta ** 2) * cumulative_var)

                var_theta = np.asarray(var_theta, dtype=float)
                om = np.diag(var_theta)
                for j in range(len(theta_arr)):
                    for k in range(j):
                        if theta_arr[k] != 0:
                            om[j, k] = var_theta[k] * theta_arr[j] / theta_arr[k]
                            om[k, j] = om[j, k]
            else:
                nstar = (n_total + n_event) / 2 if n_total else np.nan
                om = np.outer(theta_arr, 1 - theta_arr) / nstar if len(theta_arr) > 0 else np.zeros((0, 0))
                if len(theta_arr) > 0:
                    om = np.triu(om) + np.triu(om, 1).T

            hv_dead = h_dead_arr.reshape(-1, 1)
            hv_rec = h_rec_arr.reshape(-1, 1)
            a_dead = np.sum((theta_arr ** 2) * (h_dead_arr / n_dead_arr)) if len(theta_arr) > 0 else 0.0
            a_rec = np.sum((theta_arr ** 2) * (h_rec_arr / n_rec_arr)) if len(theta_arr) > 0 else 0.0
            b_dead = float(hv_dead.T @ om @ hv_dead) if len(theta_arr) > 0 else 0.0
            b_rec = float(hv_rec.T @ om @ hv_rec) if len(theta_arr) > 0 else 0.0
            cov01 = float(hv_dead.T @ om @ hv_rec) if len(theta_arr) > 0 else 0.0
            var_dead = a_dead + b_dead
            var_rec = a_rec + b_rec
            var_cfr = ((theta_rec ** 2) * var_dead + (theta_dead ** 2) * var_rec - 2 * theta_dead * theta_rec * cov01) / ((theta_dead + theta_rec) ** 4)
            se_cfr = np.sqrt(var_cfr) if var_cfr >= 0 else np.nan

            if untrans:
                lcfr = cfr_point - 1.96 * se_cfr
                ucfr = cfr_point + 1.96 * se_cfr
            else:
                if 0 < cfr_point < 1 and theta_dead > 0 and theta_rec > 0:
                    logit_cfr = np.log(cfr_point / (1 - cfr_point))
                    var_logit = var_dead / (theta_dead ** 2) + var_rec / (theta_rec ** 2) - 2 * cov01 / (theta_dead * theta_rec)
                    se_logit = np.sqrt(var_logit) if var_logit >= 0 else np.nan
                    lcfr = 1 / (1 + np.exp(-(logit_cfr - 1.96 * se_logit))) if np.isfinite(se_logit) else np.nan
                    ucfr = 1 / (1 + np.exp(-(logit_cfr + 1.96 * se_logit))) if np.isfinite(se_logit) else np.nan
                else:
                    lcfr = np.nan
                    ucfr = np.nan

        return {
            "N_rec": n_rec,
            "N_dead": n_dead,
            "N_event": n_event,
            "N_tot": n_total,
            "N_cens": n_cens,
            "e1": e1,
            "se_e1": se_e1,
            "lb_e1": le1,
            "ub_e1": ue1,
            "e2": e2,
            "se_e2": se_e2,
            "lb_e2": le2,
            "ub_e2": ue2,
            "theta0": theta0,
            "theta1": theta1,
            "cfr": cfr_point,
            "se_cfr": se_cfr,
            "lb_cfr": lcfr,
            "ub_cfr": ucfr,
            "theta_history": theta_history,
            "h_dead_history": h_dead_history,
            "h_rec_history": h_rec_history,
        }

    except Exception as e:
        print(f"Error calculating Kaplan-Meier CFR: {e}")
        return None

# Parametric mixture model
def parametric_mixture_6(data,
                         death_shape_init=3.0,
                         death_rate_init=0.5,
                         recovery_shape_init=4.0,
                         recovery_rate_init=0.5,
                         initial_pi=None,
                         maxiter=10000):
    """
    Fit a direct parametric mixture model for CFR using separate gamma delays
    for death and recovery, analogous to the HFR likelihood in the provided R code.

    The model treats each case as one of three possibilities:
    - death with probability pi_D and a gamma delay to death
    - recovery with probability 1 - pi_D and a gamma delay to recovery
    - still unresolved at the end of follow-up, contributing a survival term

    If the input data already has columns `t`, `dead`, and `rec`, they are used
    directly. Otherwise, Rosello-style raw data are converted via
    `prepare_rosello2015_km_data()`.

    Returns a dictionary containing the CFR estimate (`pi_D`), fitted delay
    parameters, and optimizer output.
    """

    def _prepare_dataframe(source):
        if {"t", "dead", "rec"}.issubset(source.columns):
            prepared = source.copy()
        else:
            prepared = prepare_rosello2015_km_data(source)

        prepared = prepared.copy()
        prepared["t"] = pd.to_numeric(prepared["t"], errors="coerce")
        prepared["dead"] = pd.to_numeric(prepared["dead"], errors="coerce").fillna(0).astype(int)
        prepared["rec"] = pd.to_numeric(prepared["rec"], errors="coerce").fillna(0).astype(int)
        prepared = prepared[prepared["t"].notna() & (prepared["t"] >= 0)].copy()

        if ((prepared["dead"] == 1) & (prepared["rec"] == 1)).any():
            raise ValueError("Rows cannot be both dead and recovered")

        return prepared

    def _interval_mass(time_value, shape, rate):
        scale = 1.0 / rate
        upper = gamma_dist.cdf(time_value + 1, a=shape, scale=scale)
        lower = gamma_dist.cdf(time_value, a=shape, scale=scale)
        return max(upper - lower, 0.0)

    def _survival_after_interval(time_value, shape, rate):
        scale = 1.0 / rate
        return max(1.0 - gamma_dist.cdf(time_value + 1, a=shape, scale=scale), 0.0)

    try:
        df = _prepare_dataframe(data)

        if df.empty:
            raise ValueError("No usable rows remain after preprocessing")

        dead_times = df.loc[df["dead"] == 1, "t"].to_numpy(dtype=float)
        rec_times = df.loc[df["rec"] == 1, "t"].to_numpy(dtype=float)
        cens_times = df.loc[(df["dead"] == 0) & (df["rec"] == 0), "t"].to_numpy(dtype=float)

        if initial_pi is None:
            resolved = int(df["dead"].sum() + df["rec"].sum())
            initial_pi = float(df["dead"].sum() / resolved) if resolved > 0 else 0.5
        initial_pi = float(np.clip(initial_pi, 1e-6, 1 - 1e-6))

        initial_params = np.array([
            np.log(initial_pi / (1 - initial_pi)),
            np.log(max(death_shape_init, 1e-6)),
            np.log(max(death_rate_init, 1e-6)),
            np.log(max(recovery_shape_init, 1e-6)),
            np.log(max(recovery_rate_init, 1e-6)),
        ], dtype=float)

        def _neg_log_likelihood(transformed_params):
            eta_pi, log_alpha_d, log_rate_d, log_alpha_r, log_rate_r = transformed_params
            pi_d = 1.0 / (1.0 + np.exp(-eta_pi))
            alpha_d = np.exp(log_alpha_d)
            rate_d = np.exp(log_rate_d)
            alpha_r = np.exp(log_alpha_r)
            rate_r = np.exp(log_rate_r)

            eps = 1e-12
            log_lik = 0.0

            for time_value in dead_times:
                mass = _interval_mass(time_value, alpha_d, rate_d)
                if mass <= 0 or not np.isfinite(mass):
                    return np.inf
                log_lik += np.log(max(pi_d, eps)) + np.log(mass)

            for time_value in rec_times:
                mass = _interval_mass(time_value, alpha_r, rate_r)
                if mass <= 0 or not np.isfinite(mass):
                    return np.inf
                log_lik += np.log(max(1.0 - pi_d, eps)) + np.log(mass)

            for time_value in cens_times:
                surv_d = _survival_after_interval(time_value, alpha_d, rate_d)
                surv_r = _survival_after_interval(time_value, alpha_r, rate_r)
                contribution = pi_d * surv_d + (1.0 - pi_d) * surv_r
                if contribution <= 0 or not np.isfinite(contribution):
                    return np.inf
                log_lik += np.log(contribution)

            return -log_lik

        fit = minimize(
            _neg_log_likelihood,
            initial_params,
            method="L-BFGS-B",
            options={"maxiter": maxiter},
        )

        eta_pi, log_alpha_d, log_rate_d, log_alpha_r, log_rate_r = fit.x
        pi_d = 1.0 / (1.0 + np.exp(-eta_pi))
        alpha_d = float(np.exp(log_alpha_d))
        rate_d = float(np.exp(log_rate_d))
        alpha_r = float(np.exp(log_alpha_r))
        rate_r = float(np.exp(log_rate_r))

        return {
            "cfr": pi_d,
            "pi_D": pi_d,
            "alpha_D": alpha_d,
            "rate_D": rate_d,
            "alpha_R": alpha_r,
            "rate_R": rate_r,
            "n_total": int(len(df)),
            "n_dead": int(df["dead"].sum()),
            "n_recovered": int(df["rec"].sum()),
            "n_censored": int(((df["dead"] == 0) & (df["rec"] == 0)).sum()),
            "log_likelihood": float(-fit.fun) if np.isfinite(fit.fun) else np.nan,
            "converged": bool(fit.success),
            "message": fit.message,
            "optimizer_result": fit,
        }

    except Exception as e:
        print(f"Error fitting parametric mixture CFR model: {e}")
        return None

# this is also made with copilot so should be checked. There is also a question of what to do with 
# cases without a date of onset or an event date
def prepare_rosello2015_km_data(data):
    """
    Convert Rosello 2015 supplementary data into Kaplan-Meier inputs.

    The article uses patient-level time-to-event data with a death indicator and a
    recovery indicator. This helper maps the Rosello CSV into the same layout by:

    - using Date_of_onset_symp as the origin date;
    - using Date_of_Death for deaths;
    - using Date_hospital_discharge, then Date_disease_ended, for recoveries;
    - treating rows with unknown outcome or missing event date as censored.

    Returns a dataframe with columns:
    - t: time from onset to outcome/censoring in days
    - dead: 1 if death, 0 otherwise
    - rec: 1 if recovery/discharge, 0 otherwise
    - origin_date: parsed onset date
    - event_date: parsed outcome or censoring date
    - event_type: 'dead', 'rec', or 'censored'
    """

    df = data.copy()

    def _parse_date(series):
        return pd.to_datetime(series, dayfirst=True, errors="coerce")

    onset = _parse_date(df["Date_of_onset_symp"])
    death = _parse_date(df["Date_of_Death"])
    discharge = _parse_date(df["Date_hospital_discharge"])
    ended = _parse_date(df["Date_disease_ended"])
    notification = _parse_date(df["Date_of_notification"])

    outcome = df["Outcome"].fillna("").astype(str).str.strip().str.lower()

    rec_date = discharge.combine_first(ended)
    event_date = pd.Series(pd.NaT, index=df.index)
    event_type = pd.Series("censored", index=df.index, dtype="object")

    dead_mask = outcome.eq("dead") & death.notna()
    rec_mask = outcome.eq("alive") & rec_date.notna()

    event_date.loc[dead_mask] = death.loc[dead_mask]
    event_type.loc[dead_mask] = "dead"

    event_date.loc[rec_mask] = rec_date.loc[rec_mask]
    event_type.loc[rec_mask] = "rec"

    censored_mask = event_type.eq("censored")
    event_date.loc[censored_mask] = notification.loc[censored_mask]

    t = (event_date - onset).dt.days

    km = pd.DataFrame(
        {
            "origin_date": onset,
            "event_date": event_date,
            "event_type": event_type,
            "t": t,
            "dead": (event_type == "dead").astype(int),
            "rec": (event_type == "rec").astype(int),
            "Outcome": df["Outcome"],
            "Person_ID": df.get("Person_ID"),
            "Outbreak": df.get("Outbreak"),
            "Year_outbreak": df.get("Year_outbreak"),
        }
    )

    km = km[km["t"].notna() & (km["t"] >= 0)].copy()
    km["t"] = km["t"].astype(int)
    return km


def _parse_datetime(series, dayfirst=None):
    if dayfirst is None:
        return pd.to_datetime(series, errors="coerce")
    return pd.to_datetime(series, errors="coerce", dayfirst=dayfirst)


def _daily_counts_from_dates(case_dates, death_dates, date_name="date"):
    case_dates = pd.to_datetime(case_dates, errors="coerce")
    death_dates = pd.to_datetime(death_dates, errors="coerce")

    case_counts = case_dates.dropna().dt.normalize().value_counts().sort_index()
    death_counts = death_dates.dropna().dt.normalize().value_counts().sort_index()
    all_dates = case_counts.index.union(death_counts.index).sort_values()

    daily = pd.DataFrame({date_name: all_dates})
    daily["cases"] = daily[date_name].map(case_counts).fillna(0).astype(int)
    daily["deaths"] = daily[date_name].map(death_counts).fillna(0).astype(int)
    return daily


def _daily_counts_from_km(km, date_col="event_date", date_name="date"):
    """Aggregate daily deaths and recoveries from a KM-style line list."""
    if km is None or len(km) == 0:
        return pd.DataFrame(columns=[date_name, "deaths", "recoveries"])

    df = km.copy()
    if date_col not in df.columns:
        raise ValueError(f"KM dataframe is missing required column: {date_col}")

    df[date_col] = pd.to_datetime(df[date_col], errors="coerce").dt.normalize()
    df = df[df[date_col].notna()].copy()

    deaths = (
        df.loc[df["dead"] == 1, date_col]
        .value_counts()
        .sort_index()
        .rename("deaths")
    )
    recov = (
        df.loc[df["rec"] == 1, date_col]
        .value_counts()
        .sort_index()
        .rename("recoveries")
    )

    all_dates = deaths.index.union(recov.index).sort_values()
    daily = pd.DataFrame({date_name: all_dates})
    daily["deaths"] = daily[date_name].map(deaths).fillna(0).astype(int)
    daily["recoveries"] = daily[date_name].map(recov).fillna(0).astype(int)
    return daily


def _fill_daily_dates(daily, date_col="date"):
    """Ensure a continuous daily date index and integer count columns."""
    df = daily.copy()
    if date_col not in df.columns:
        raise ValueError(f"Daily dataframe must contain a {date_col} column")

    df[date_col] = pd.to_datetime(df[date_col], errors="coerce").dt.normalize()
    df = df[df[date_col].notna()].copy()
    if df.empty:
        return df

    numeric_cols = [c for c in df.columns if c != date_col]
    for c in numeric_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

    df = df.groupby(date_col, as_index=False)[numeric_cols].sum().sort_values(date_col)
    full_idx = pd.date_range(df[date_col].min(), df[date_col].max(), freq="D")
    df = (
        df.set_index(date_col)
        .reindex(full_idx, fill_value=0)
        .rename_axis(date_col)
        .reset_index()
    )
    for c in numeric_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
    return df


def daily_naive_cfr_series(daily, date_col="date"):
    """Day-by-day naive CFR = cumulative deaths / cumulative cases."""
    df = _fill_daily_dates(daily, date_col=date_col).copy()
    if "cases" not in df.columns or "deaths" not in df.columns:
        raise ValueError("Daily data need cases and deaths columns")
    df["cum_cases"] = df["cases"].cumsum()
    df["cum_deaths"] = df["deaths"].cumsum()
    df["method1_naive"] = np.where(df["cum_cases"] > 0, df["cum_deaths"] / df["cum_cases"], np.nan)
    return df


def daily_resolved_cohort_cfr_series(daily, date_col="date"):
    """Day-by-day resolved cohort CFR = cumulative deaths / (deaths + recoveries)."""
    df = _fill_daily_dates(daily, date_col=date_col).copy()
    if "deaths" not in df.columns or "recoveries" not in df.columns:
        raise ValueError("Daily data need deaths and recoveries columns")
    df["cum_deaths"] = df["deaths"].cumsum()
    df["cum_recoveries"] = df["recoveries"].cumsum()
    resolved = df["cum_deaths"] + df["cum_recoveries"]
    df["method2_resolved"] = np.where(resolved > 0, df["cum_deaths"] / resolved, np.nan)
    return df


def daily_delay_adjusted_cfr_series(daily, delay_shape=2.40, delay_scale=3.33, date_col="date"):
    """Day-by-day delay-adjusted CFR using a gamma delay distribution.

    This is a pure-Python fallback that does not require R.
    It estimates the CFR at each date as:
        cumulative deaths / expected deaths observed by that date
    where the expected observed fraction is based on the delay CDF.
    """
    df = _fill_daily_dates(daily, date_col=date_col).copy()
    if "cases" not in df.columns or "deaths" not in df.columns:
        raise ValueError("Daily data need cases and deaths columns")

    df["cum_cases"] = df["cases"].cumsum()
    df["cum_deaths"] = df["deaths"].cumsum()

    cases = df["cases"].to_numpy(dtype=float)
    cfr_series = []
    expected_obs = []

    for t in range(len(df)):
        lags = np.arange(t + 1, dtype=float)
        weights = gamma_dist.cdf(lags, a=delay_shape, scale=delay_scale)
        # cases earlier in time have a higher chance of having had their outcome by day t
        expected = float(np.sum(cases[: t + 1][::-1] * weights))
        expected_obs.append(expected)
        cfr_series.append(df.loc[t, "cum_deaths"] / expected if expected > 0 else np.nan)

    df["expected_outcomes_by_day"] = expected_obs
    df["method3_delay_adjusted"] = cfr_series
    return df


def summarise_daily_cfr_estimates(daily, delay_shape=2.40, delay_scale=3.33, date_col="date"):
    """Build a single dataframe containing all supported day-by-day CFR estimates."""
    df = _fill_daily_dates(daily, date_col=date_col).copy()

    if "cases" not in df.columns or "deaths" not in df.columns:
        raise ValueError("Daily data need cases and deaths columns")

    out = daily_naive_cfr_series(df, date_col=date_col)[[date_col, "cum_cases", "cum_deaths", "method1_naive"]]

    if "recoveries" in df.columns:
        res = daily_resolved_cohort_cfr_series(df, date_col=date_col)
        out = out.merge(
            res[[date_col, "cum_recoveries", "method2_resolved"]],
            on=date_col,
            how="left",
        )
    else:
        out["cum_recoveries"] = np.nan
        out["method2_resolved"] = np.nan

    adj = daily_delay_adjusted_cfr_series(df, delay_shape=delay_shape, delay_scale=delay_scale, date_col=date_col)
    out = out.merge(
        adj[[date_col, "expected_outcomes_by_day", "method3_delay_adjusted"]],
        on=date_col,
        how="left",
    )
    return out


def plot_daily_cfr_estimates(daily_estimates, title, output_dir="cfr_plots", date_col="date"):
    """Save a plot of the day-by-day CFR estimates."""
    os.makedirs(output_dir, exist_ok=True)
    df = daily_estimates.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df[df[date_col].notna()].copy()

    fig, ax = plt.subplots(figsize=(11, 5))
    plotted = False

    for col, label in [
        ("method1_naive", "Method 1: naive"),
        ("method2_resolved", "Method 2: resolved cohort"),
        ("method3_delay_adjusted", "Method 3: delay-adjusted"),
    ]:
        if col in df.columns and df[col].notna().any():
            ax.plot(df[date_col], df[col], label=label, linewidth=2)
            plotted = True

    if not plotted:
        plt.close(fig)
        return None

    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("CFR")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.autofmt_xdate()

    safe_title = "".join(ch if ch.isalnum() or ch in (" ", "_", "-") else "_" for ch in title).strip().replace(" ", "_")
    out_path = os.path.join(output_dir, f"{safe_title}_daily_cfr.png")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path


def _daily_counts_from_cumulative(data, date_col, cases_col, deaths_col, case_daily_col=None, death_daily_col=None, date_name="date"):
    df = data.copy()
    df[date_col] = _parse_datetime(df[date_col])
    df = df[df[date_col].notna()].copy()

    numeric_columns = [col for col in [cases_col, deaths_col, case_daily_col, death_daily_col] if col and col in df.columns]
    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    grouped = df.groupby(date_col, as_index=False)[numeric_columns].sum(min_count=1) if numeric_columns else df[[date_col]].drop_duplicates()
    grouped = grouped.sort_values(date_col).reset_index(drop=True)

    if cases_col in grouped.columns:
        grouped[cases_col] = pd.to_numeric(grouped[cases_col], errors="coerce").fillna(0)
    if deaths_col in grouped.columns:
        grouped[deaths_col] = pd.to_numeric(grouped[deaths_col], errors="coerce").fillna(0)

    daily = pd.DataFrame({date_name: grouped[date_col]})

    if case_daily_col and case_daily_col in grouped.columns and grouped[case_daily_col].notna().any():
        daily["cases"] = grouped[case_daily_col].fillna(0).round().clip(lower=0).astype(int)
        daily["cumulative_cases"] = grouped[cases_col] if cases_col in grouped.columns else np.nan
    elif cases_col in grouped.columns:
        daily["cumulative_cases"] = grouped[cases_col]
        daily["cases"] = grouped[cases_col].diff().fillna(grouped[cases_col]).clip(lower=0).round().astype(int)
    else:
        daily["cases"] = 0
        daily["cumulative_cases"] = np.nan

    if death_daily_col and death_daily_col in grouped.columns and grouped[death_daily_col].notna().any():
        daily["deaths"] = grouped[death_daily_col].fillna(0).round().clip(lower=0).astype(int)
        daily["cumulative_deaths"] = grouped[deaths_col] if deaths_col in grouped.columns else np.nan
    elif deaths_col in grouped.columns:
        daily["cumulative_deaths"] = grouped[deaths_col]
        daily["deaths"] = grouped[deaths_col].diff().fillna(grouped[deaths_col]).clip(lower=0).round().astype(int)
    else:
        daily["deaths"] = 0
        daily["cumulative_deaths"] = np.nan

    return daily


def prepare_drc2018_total_cfr_inputs(data):
    """
    Prepare DRC 2018 Ministry of Health total time-series data.

    Compatible with:
    - 1: naive CFR via the latest counts
    - 2: resolved cohort CFR only if total_cured is available
    - 3: delay-adjusted CFR via the returned daily series

    Returns a dictionary with `counts` and `daily`.
    """

    df = data.copy()
    if "report_date" not in df.columns:
        raise ValueError("DRC total data needs a report_date column")

    daily = _daily_counts_from_cumulative(
        df,
        date_col="report_date",
        cases_col="total_cases",
        deaths_col="total_deaths",
        case_daily_col="total_cases_change" if "total_cases_change" in df.columns else None,
        death_daily_col="new_deaths" if "new_deaths" in df.columns else None,
    )

    latest = df.copy()
    latest["report_date"] = _parse_datetime(latest["report_date"])
    latest = latest[latest["report_date"].notna()].sort_values("report_date")
    latest_row = latest.iloc[-1] if not latest.empty else pd.Series(dtype="object")

    counts = {
        "cases": int(pd.to_numeric(latest_row.get("total_cases", np.nan), errors="coerce")) if not latest.empty and pd.notna(latest_row.get("total_cases", np.nan)) else int(daily["cumulative_cases"].dropna().iloc[-1]) if "cumulative_cases" in daily and daily["cumulative_cases"].notna().any() else np.nan,
        "deaths": int(pd.to_numeric(latest_row.get("total_deaths", np.nan), errors="coerce")) if not latest.empty and pd.notna(latest_row.get("total_deaths", np.nan)) else int(daily["cumulative_deaths"].dropna().iloc[-1]) if "cumulative_deaths" in daily and daily["cumulative_deaths"].notna().any() else np.nan,
        "recoveries": int(pd.to_numeric(latest_row.get("total_cured", np.nan), errors="coerce")) if not latest.empty and pd.notna(latest_row.get("total_cured", np.nan)) else np.nan,
    }

    return {"counts": counts, "daily": daily, "km": None, "source": "DRC2018_total"}


def prepare_drc2018_by_health_zone_cfr_inputs(data, health_zone=None):
    """
    Prepare DRC 2018 health-zone data for CFR estimation.

    Compatible with functions 1, 2, and 3 when the cumulative count columns are present.
    For 5 and 6 the file does not contain case-level outcomes, so it is not suitable.
    """

    df = data.copy()
    if health_zone is not None and "health_zone" in df.columns:
        df = df[df["health_zone"].astype(str) == str(health_zone)].copy()

    daily = _daily_counts_from_cumulative(
        df,
        date_col="report_date",
        cases_col="total_cases",
        deaths_col="total_deaths",
        case_daily_col="total_cases_change" if "total_cases_change" in df.columns else None,
        death_daily_col="new_deaths" if "new_deaths" in df.columns else None,
    )

    latest = df.copy()
    latest["report_date"] = _parse_datetime(latest["report_date"])
    latest = latest[latest["report_date"].notna()].sort_values("report_date")
    latest_row = latest.iloc[-1] if not latest.empty else pd.Series(dtype="object")

    counts = {
        "cases": int(pd.to_numeric(latest_row.get("total_cases", np.nan), errors="coerce")) if not latest.empty and pd.notna(latest_row.get("total_cases", np.nan)) else np.nan,
        "deaths": int(pd.to_numeric(latest_row.get("total_deaths", np.nan), errors="coerce")) if not latest.empty and pd.notna(latest_row.get("total_deaths", np.nan)) else np.nan,
        "recoveries": int(pd.to_numeric(latest_row.get("total_cured", np.nan), errors="coerce")) if not latest.empty and pd.notna(latest_row.get("total_cured", np.nan)) else np.nan,
    }

    return {"counts": counts, "daily": daily, "km": None, "source": "DRC2018_by_health_zone"}


def prepare_drc_consolidated_cfr_inputs(data, location_name=None, case_classification="confirmed"):
    """
    Prepare the consolidated DRC case/death line list for CFR estimation.

    Compatible with:
    - 1: naive CFR via counts
    - 2: resolved cohort CFR if recovery data are added externally
    - 3: delay-adjusted CFR via daily cases/deaths time series

    The file is a long-format cumulative series, so daily incidence is recovered
    by summing by date and differencing the cumulative series.
    """

    df = data.copy()
    if "reference_date" not in df.columns or "measure" not in df.columns or "value" not in df.columns:
        raise ValueError("Consolidated DRC data must contain reference_date, measure, and value")

    if location_name is not None and "location_name" in df.columns:
        df = df[df["location_name"].astype(str) == str(location_name)].copy()

    if case_classification is not None and "case_classification" in df.columns:
        df = df[df["case_classification"].astype(str) == str(case_classification)].copy()

    df = df[df["measure"].astype(str).isin(["cases", "deaths"])].copy()
    df["reference_date"] = _parse_datetime(df["reference_date"])
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df[df["reference_date"].notna() & df["value"].notna()].copy()

    pivot = df.pivot_table(index="reference_date", columns="measure", values="value", aggfunc="sum").sort_index().reset_index()
    pivot = pivot.rename(columns={"reference_date": "date"})
    if "cases" not in pivot.columns:
        pivot["cases"] = 0
    if "deaths" not in pivot.columns:
        pivot["deaths"] = 0

    daily = _daily_counts_from_cumulative(
        pivot,
        date_col="date",
        cases_col="cases",
        deaths_col="deaths",
    )

    counts = {
        "cases": int(daily["cumulative_cases"].dropna().iloc[-1]) if "cumulative_cases" in daily and daily["cumulative_cases"].notna().any() else int(daily["cases"].sum()),
        "deaths": int(daily["cumulative_deaths"].dropna().iloc[-1]) if "cumulative_deaths" in daily and daily["cumulative_deaths"].notna().any() else int(daily["deaths"].sum()),
        "recoveries": np.nan,
    }

    return {"counts": counts, "daily": daily, "km": None, "source": "drc_consolidated"}


def prepare_uganda2022_cfr_inputs(data):
    """
    Prepare Uganda 2022 global.health line-list data for CFR estimation.

    Compatible with:
    - 1: naive CFR via counts
    - 2: resolved cohort CFR via deaths and recoveries
    - 3: delay-adjusted CFR via daily onset/confirmation and death dates
    - 5/6: case-level Kaplan-Meier / mixture inputs via t, dead, rec
    """

    df = data.copy()

    onset = _parse_datetime(df["Date_onset"]) if "Date_onset" in df.columns else pd.Series(pd.NaT, index=df.index)
    confirmation = _parse_datetime(df["Date_confirmation"]) if "Date_confirmation" in df.columns else pd.Series(pd.NaT, index=df.index)
    first_consult = _parse_datetime(df["Date_of_first_consult"]) if "Date_of_first_consult" in df.columns else pd.Series(pd.NaT, index=df.index)
    hospitalisation = _parse_datetime(df["Date_hospitalisation"]) if "Date_hospitalisation" in df.columns else pd.Series(pd.NaT, index=df.index)
    death_date = _parse_datetime(df["Date_Death"]) if "Date_Death" in df.columns else pd.Series(pd.NaT, index=df.index)
    recovery_date = _parse_datetime(df["Date_Recovered"]) if "Date_Recovered" in df.columns else pd.Series(pd.NaT, index=df.index)
    censor_date = _parse_datetime(df["Date_last_modified"]) if "Date_last_modified" in df.columns else pd.Series(pd.NaT, index=df.index)

    origin = onset.combine_first(hospitalisation).combine_first(first_consult).combine_first(confirmation).combine_first(censor_date)
    event_date = pd.Series(pd.NaT, index=df.index)
    event_type = pd.Series("censored", index=df.index, dtype="object")

    outcome = df["Outcome"].fillna("").astype(str).str.strip().str.lower() if "Outcome" in df.columns else pd.Series("", index=df.index)
    dead_mask = outcome.eq("death") & death_date.notna()
    rec_mask = outcome.isin(["recovery", "recovered"]) & recovery_date.notna()

    event_date.loc[dead_mask] = death_date.loc[dead_mask]
    event_type.loc[dead_mask] = "dead"
    event_date.loc[rec_mask] = recovery_date.loc[rec_mask]
    event_type.loc[rec_mask] = "rec"

    censored_mask = event_type.eq("censored")
    event_date.loc[censored_mask] = censor_date.loc[censored_mask].combine_first(confirmation.loc[censored_mask])

    km = pd.DataFrame(
        {
            "origin_date": origin,
            "event_date": event_date,
            "event_type": event_type,
            "t": (event_date - origin).dt.days,
            "dead": (event_type == "dead").astype(int),
            "rec": (event_type == "rec").astype(int),
            "Outcome": df.get("Outcome"),
            "Case_status": df.get("Case_status"),
            "ID": df.get("ID"),
        }
    )
    km = km[km["t"].notna() & (km["t"] >= 0)].copy()
    km["t"] = km["t"].astype(int)

    daily = _daily_counts_from_dates(
        case_dates=origin,
        death_dates=death_date,
    )
    daily_rec = _daily_counts_from_km(km)
    if not daily_rec.empty:
        daily = daily.merge(daily_rec[["date", "recoveries"]], on="date", how="outer")
        daily = daily.sort_values("date").reset_index(drop=True)
        daily["cases"] = daily["cases"].fillna(0).astype(int)
        daily["deaths"] = daily["deaths"].fillna(0).astype(int)
        daily["recoveries"] = daily["recoveries"].fillna(0).astype(int)

    counts = {
        "cases": int(len(df)),
        "deaths": int((event_type == "dead").sum()),
        "recoveries": int((event_type == "rec").sum()),
    }

    return {"counts": counts, "daily": daily, "km": km, "source": "uganda_2022"}


def prepare_rosello2015_cfr_inputs(data):
    """
    Prepare Rosello 2015 supplementary data for CFR estimation.

    Compatible with 1, 2, 3, 5, and 6, provided you are comfortable using onset
    to death / onset to discharge as the time axis for the survival-style methods.
    """

    km = prepare_rosello2015_km_data(data)
    df = data.copy()
    onset = _parse_datetime(df["Date_of_onset_symp"], dayfirst=True)
    death_date = _parse_datetime(df["Date_of_Death"], dayfirst=True)
    recovery_date = _parse_datetime(df["Date_hospital_discharge"], dayfirst=True).combine_first(_parse_datetime(df["Date_disease_ended"], dayfirst=True))
    daily = _daily_counts_from_dates(case_dates=onset, death_dates=death_date)
    daily_rec = _daily_counts_from_km(km)
    if not daily_rec.empty:
        daily = daily.merge(daily_rec[["date", "recoveries"]], on="date", how="outer")
        daily = daily.sort_values("date").reset_index(drop=True)
        daily["cases"] = daily["cases"].fillna(0).astype(int)
        daily["deaths"] = daily["deaths"].fillna(0).astype(int)
        daily["recoveries"] = daily["recoveries"].fillna(0).astype(int)

    outcome = df["Outcome"].fillna("").astype(str).str.strip().str.lower()
    counts = {
        "cases": int(len(df)),
        "deaths": int(outcome.eq("dead").sum()),
        "recoveries": int(outcome.eq("alive").sum()),
    }

    return {"counts": counts, "daily": daily, "km": km, "source": "rosello_2015"}


def prepare_sierra_leone_case_counts(data):
    """
    Prepare the Sierra Leone 2014 case lists.

    These files only contain symptom onset and sample dates, so they can support
    function 1 (naive CFR) as case counts, but they do not contain death/recovery
    outcomes required for functions 2, 3, 5, or 6.
    """

    df = data.copy()
    onset_col = "Date of symptom onset " if "Date of symptom onset " in df.columns else "Date of symptom onset"
    onset = _parse_datetime(df[onset_col], dayfirst=True) if onset_col in df.columns else pd.Series(pd.NaT, index=df.index)
    sample = _parse_datetime(df["Date of sample tested"], dayfirst=True) if "Date of sample tested" in df.columns else pd.Series(pd.NaT, index=df.index)

    daily = _daily_counts_from_dates(case_dates=onset.combine_first(sample), death_dates=pd.Series(pd.NaT, index=df.index))
    counts = {"cases": int(len(df)), "deaths": np.nan, "recoveries": np.nan}

    return {"counts": counts, "daily": daily, "km": None, "source": "sierra_leone_2014"}


def run_all_cfr_examples(data_dir="Data", plot_output_dir="cfr_plots"):
    """
    Load each dataset in the Data folder, prepare it, and run every compatible CFR method.

    Methods:
    - 1: naive_1
    - 2: resolved_cohort_2
    - 3: delay_adjusted_3
    - 5: kaplan_meier_5
    - 6: parametric_mixture_6

    Also produces day-by-day CFR estimates and saves plots.
    """

    base_path = data_dir if isinstance(data_dir, str) else str(data_dir)
    dataset_specs = [
        {
            "name": "Rosello 2015",
            "file": "rosello2015_supplementary1.csv",
            "prep": prepare_rosello2015_cfr_inputs,
        },
        {
            "name": "Uganda 2022",
            "file": "Uganda2022globaldothealth.csv",
            "prep": prepare_uganda2022_cfr_inputs,
        },
        {
            "name": "DRC 2018 Total",
            "file": "DRC2018_humdata_MOH-Total.csv",
            "prep": prepare_drc2018_total_cfr_inputs,
        },
        {
            "name": "DRC Consolidated",
            "file": "drc_ebola_cases_consolidated.csv",
            "prep": prepare_drc_consolidated_cfr_inputs,
        },
        {
            "name": "DRC 2018 By Health Zone",
            "file": "DRC2018_humdata_MOH-By-Health-Zone.csv",
            "prep": prepare_drc2018_by_health_zone_cfr_inputs,
        },
        {
            "name": "Sierra Leone Confirmed",
            "file": "ebola_sierraleone_2014_confirmed.csv",
            "prep": prepare_sierra_leone_case_counts,
        },
        {
            "name": "Sierra Leone Suspected",
            "file": "ebola_sierraleone_2014_suspected.csv",
            "prep": prepare_sierra_leone_case_counts,
        },
    ]

    results = {}
    daily_rows = []

    for spec in dataset_specs:
        file_path = f"{base_path}/{spec['file']}"
        print(f"\n=== {spec['name']} ===")
        try:
            raw = pd.read_csv(file_path)
            prepared = spec["prep"](raw)
        except Exception as exc:
            print(f"Preparation failed: {exc}")
            results[spec["name"]] = {"error": str(exc)}
            continue

        counts = prepared.get("counts", {})
        daily = prepared.get("daily")
        km = prepared.get("km")
        print(f"Source: {prepared.get('source')}")
        print(f"Counts: {counts}")

        dataset_result = {}

        if pd.notna(counts.get("cases", np.nan)) and pd.notna(counts.get("deaths", np.nan)):
            try:
                dataset_result[1] = naive_1(int(counts["deaths"]), int(counts["cases"]))
                print(f"Method 1 naive CFR: {dataset_result[1]}")
            except Exception as exc:
                print(f"Method 1 failed: {exc}")

        if pd.notna(counts.get("deaths", np.nan)) and pd.notna(counts.get("recoveries", np.nan)):
            try:
                dataset_result[2] = resolved_cohort_2(int(counts["deaths"]), int(counts["recoveries"]))
                print(f"Method 2 resolved cohort CFR: {dataset_result[2]}")
            except Exception as exc:
                print(f"Method 2 failed: {exc}")

        if daily is not None and {"date", "cases", "deaths"}.issubset(daily.columns):
            try:
                dataset_result[3] = delay_adjusted_3(daily)
                print(f"Method 3 delay-adjusted CFR: {dataset_result[3]['cfr'] if isinstance(dataset_result[3], dict) else dataset_result[3]}")
            except Exception as exc:
                print(f"Method 3 failed: {exc}")

            try:
                daily_estimates = summarise_daily_cfr_estimates(daily)
                daily_estimates.insert(0, "dataset", spec["name"])
                daily_rows.append(daily_estimates)
                plot_path = plot_daily_cfr_estimates(daily_estimates, spec["name"], output_dir=plot_output_dir)
                if plot_path:
                    print(f"Saved daily CFR plot: {plot_path}")
                dataset_result["daily_estimates"] = daily_estimates
                dataset_result["daily_plot"] = plot_path
            except Exception as exc:
                print(f"Daily estimate plotting failed: {exc}")

        if km is not None and len(km) > 0:
            try:
                dataset_result[5] = kaplan_meier_5(km)
                print(f"Method 5 Kaplan-Meier CFR: {dataset_result[5]}")
            except Exception as exc:
                print(f"Method 5 failed: {exc}")

            try:
                dataset_result[6] = parametric_mixture_6(km)
                print(f"Method 6 parametric mixture CFR: {dataset_result[6]}")
            except Exception as exc:
                print(f"Method 6 failed: {exc}")

        results[spec["name"]] = dataset_result

    output_path = "cfr_results.csv"
    try:
        rows = []
        for dataset_name, dataset_result in results.items():
            if isinstance(dataset_result, dict) and "error" in dataset_result:
                rows.append(
                    {
                        "dataset": dataset_name,
                        "method": "error",
                        "status": "error",
                        "value": np.nan,
                        "error": dataset_result["error"],
                    }
                )
                continue
            for method_number in sorted(k for k in dataset_result.keys() if isinstance(k, int)):
                method_result = dataset_result[method_number]
                row = {
                    "dataset": dataset_name,
                    "method": method_number,
                    "status": "ok",
                    "value": method_result if not isinstance(method_result, dict) else method_result.get("cfr", method_result.get("value", np.nan)),
                    "error": "",
                }
                if isinstance(method_result, dict):
                    for key in [
                        "cfr",
                        "pi_D",
                        "theta0",
                        "theta1",
                        "cfr_low",
                        "cfr_high",
                        "lb_cfr",
                        "ub_cfr",
                        "N_tot",
                        "N_dead",
                        "N_rec",
                        "N_event",
                        "N_cens",
                        "n_total",
                        "n_dead",
                        "n_recovered",
                        "n_censored",
                        "converged",
                        "log_likelihood",
                    ]:
                        if key in method_result:
                            row[key] = method_result[key]
                    row["result"] = str(method_result)
                else:
                    row["result"] = str(method_result)
                rows.append(row)

        pd.DataFrame(rows).to_csv(output_path, index=False)
        print(f"\nSaved results to {output_path}")
    except Exception as exc:
        print(f"Could not save results to {output_path}: {exc}")

    if daily_rows:
        try:
            pd.concat(daily_rows, ignore_index=True).to_csv("cfr_daily_estimates.csv", index=False)
            print("Saved daily estimates to cfr_daily_estimates.csv")
        except Exception as exc:
            print(f"Could not save daily estimates: {exc}")

    return results


if __name__ == "__main__":

    run_all_cfr_examples()