# Create functions to calculate CFR given correct input data
# load libraries
import pandas as pd
import numpy as np
from scipy.stats import beta
from rpy2.robjects.packages import importr
from rpy2.robjects import pandas2ri
import rpy2.robjects as ro

# Activate pandas2ri for automatic conversion between pandas and R
pandas2ri.activate()

# Import R packages
try:
    cfr = importr('cfr')
except Exception as e:
    print(f"Error importing cfr package: {e}")
    print("Please install cfr in R with: install.packages('cfr')") 

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
    Calculate delay-adjusted Nishiura cCFR using R's cfr package.
    
    Parameters:
    - data: pandas DataFrame with columns 'date', 'cases', and 'deaths'
    - delay_shape: shape parameter for gamma distribution (default 2.40 for Ebola)
    - delay_scale: scale parameter for gamma distribution (default 3.33 for Ebola)
    
    Returns:
    - CFR estimate with confidence intervals from R cfr package
    """
    # Convert pandas DataFrame to R DataFrame
    r_data = pandas2ri.py2rpy(data)
    
    # Call R cfr package's cfr_static function
    try:
        # Create R function call to cfr::cfr_static with delay_density
        r_code = f"""
        library(cfr)
        cfr_result <- cfr_static(
            data = data,
            delay_density = function(x) dgamma(x, shape = {delay_shape}, scale = {delay_scale})
        )
        cfr_result
        """
        
        # Execute R code
        ro.r(f'data <- data')
        result = ro.r(r_code)
        return result
    except Exception as e:
        print(f"Error calculating Nishiura cCFR: {e}")
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
