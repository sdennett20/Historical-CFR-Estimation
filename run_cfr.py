from cfr_exact import running_cfr, adapt_drc_total_to_counts, adapt_rosello_to_linelist,adapt_guinea_to_counts
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import matplotlib.colors as mcolors
from typing import Optional
import warnings

# helper for plotting 
def plot_method_with_ci(ax, data, method, *, alpha=0.18, linewidth=2, label=None):
    color = METHOD_COLORS[method]
    y = data[method]
    lo_col = f"{method}_lower"
    hi_col = f"{method}_upper"

    # draw the line first
    line, = ax.plot(
        data["date"],
        y,
        linewidth=linewidth,
        label=label or method,
    )

    # draw the shaded CI if available
    if lo_col in data.columns and hi_col in data.columns:
        mask = y.notna() & data[lo_col].notna() & data[hi_col].notna()
        if mask.any():
            ax.fill_between(
                data.loc[mask, "date"],
                data.loc[mask, lo_col],
                data.loc[mask, hi_col],
                color=color,
                alpha=alpha,
                linewidth=0,
            )

    return line

METHOD_COLORS = {
    "naive": "#1f77b4",
    "resolved": "#ff7f0e",
    "delay_adjusted": "#2ca02c",
    "competing_risks": "#d62728",
    "kaplan_meier_ghani": "#9467bd",
    "parametric_mixture": "#8c564b",
}

EPICURVE_COLORS = {
    "cases": "#444444",
    "deaths": "#d62728",
    "recoveries": "#2ca02c",
}
methods = [
    "naive",
    "resolved",
    "delay_adjusted",
    "competing_risks",
    "kaplan_meier_ghani",
    "parametric_mixture",
]

def _prepare_count_table_epidemic_curve(
    raw_df: pd.DataFrame,
    *,
    date_col: str,
    cases_col: str,
    deaths_col: str,
    recovered_col: Optional[str] = None,
) -> pd.DataFrame:
    work = raw_df.copy()
    work[date_col] = pd.to_datetime(work[date_col], errors="coerce")
    work = work.dropna(subset=[date_col]).copy()
    if work.empty:
        return pd.DataFrame(columns=["date", "cases", "deaths", "recoveries"])

    idx = pd.date_range(
        start=work[date_col].min().normalize(),
        end=work[date_col].max().normalize(),
        freq="D",
    )

    def _series_for(col: Optional[str]) -> pd.Series:
        if col is None or col not in work.columns:
            return pd.Series(0.0, index=idx)
        s = (
            pd.to_numeric(work[col], errors="coerce")
            .groupby(work[date_col].dt.normalize())
            .last()
            .reindex(idx)
            .ffill()
            .fillna(0.0)
        )
        return s.astype(float)

    cases = _series_for(cases_col)
    deaths = _series_for(deaths_col)
    recoveries = _series_for(recovered_col)

    return pd.DataFrame(
        {
            "date": idx,
            "cases": cases.to_numpy(dtype=float),
            "deaths": deaths.to_numpy(dtype=float),
            "recoveries": recoveries.to_numpy(dtype=float),
        }
    )


def _prepare_linelist_epidemic_curve(ll: pd.DataFrame) -> pd.DataFrame:
    work = ll.copy()
    work["start_date"] = pd.to_datetime(work["start_date"], errors="coerce")
    work["outcome_date"] = pd.to_datetime(work["outcome_date"], errors="coerce")
    if "event" in work.columns:
        event = work["event"].astype(str).str.lower()
    else:
        event = pd.Series("", index=work.index, dtype="object")

    start_candidates = [s.min() for s in [work["start_date"], work["outcome_date"]] if s.notna().any()]
    end_candidates = [s.max() for s in [work["start_date"], work["outcome_date"]] if s.notna().any()]
    if not start_candidates or not end_candidates:
        return pd.DataFrame(columns=["date", "cases", "deaths", "recoveries"])

    idx = pd.date_range(
        start=pd.to_datetime(min(start_candidates)).normalize(),
        end=pd.to_datetime(max(end_candidates)).normalize(),
        freq="D",
    )

    cases_daily = (
        work.loc[work["start_date"].notna(), "start_date"].dt.normalize().value_counts().sort_index().reindex(idx, fill_value=0).astype(float)
    )
    deaths_daily = (
        work.loc[event.isin({"death", "dead", "died", "deceased"}) & work["outcome_date"].notna(), "outcome_date"]
        .dt.normalize()
        .value_counts()
        .sort_index()
        .reindex(idx, fill_value=0)
        .astype(float)
    )
    recoveries_daily = (
        work.loc[event.isin({"recovery", "recovered", "alive", "discharged", "discharge"}) & work["outcome_date"].notna(), "outcome_date"]
        .dt.normalize()
        .value_counts()
        .sort_index()
        .reindex(idx, fill_value=0)
        .astype(float)
    )

    return pd.DataFrame(
        {
            "date": idx,
            "cases": cases_daily.cumsum().to_numpy(dtype=float),
            "deaths": deaths_daily.cumsum().to_numpy(dtype=float),
            "recoveries": recoveries_daily.cumsum().to_numpy(dtype=float),
        }
    )


def _plot_cfr_and_curve_figure(
    *,
    plot_results: pd.DataFrame,
    epidemic_curve: pd.DataFrame,
    methods: list[str],
    title: str,
    output_path: str,
    include_recoveries: bool = True,
) -> None:
    fig, (ax_cfr, ax_curve) = plt.subplots(
        2,
        1,
        figsize=(10, 8),
        sharex=True,
        gridspec_kw={"height_ratios": [3, 1]},
    )

    for method in methods:
        if method in plot_results.columns:
            color = METHOD_COLORS[method]
            ax_cfr.plot(
                plot_results["date"],
                plot_results[method],
                linewidth=2,
                label=method,
                color=color,
            )
            lower = f"{method}_lower"
            upper = f"{method}_upper"
            if lower in plot_results.columns and upper in plot_results.columns:
                ax_cfr.fill_between(
                    plot_results["date"],
                    plot_results[lower],
                    plot_results[upper],
                    color=color,
                    alpha=0.2,
                )

    if not epidemic_curve.empty:
        ax_curve.plot(
            epidemic_curve["date"],
            epidemic_curve["cases"],
            linewidth=2,
            label="cases",
            color=EPICURVE_COLORS["cases"],
        )
        ax_curve.plot(
            epidemic_curve["date"],
            epidemic_curve["deaths"],
            linewidth=2,
            label="deaths",
            color=EPICURVE_COLORS["deaths"],
        )
        if include_recoveries and epidemic_curve["recoveries"].notna().any() and float(epidemic_curve["recoveries"].sum()) > 0:
            ax_curve.plot(
                epidemic_curve["date"],
                epidemic_curve["recoveries"],
                linewidth=2,
                label="recoveries",
                color=EPICURVE_COLORS["recoveries"],
            )

    ax_cfr.set_ylabel("Case Fatality Ratio")
    ax_cfr.set_title(title)
    ax_cfr.legend()
    ax_cfr.grid(True, alpha=0.3)

    ax_curve.set_xlabel("Date")
    ax_curve.set_ylabel("Cumulative count")
    ax_curve.set_title("Cumulative epidemic curve")
    ax_curve.legend(ncol=3)
    ax_curve.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _prepare_linelist_epidemic_curve(ll: pd.DataFrame) -> pd.DataFrame:
    work = ll.copy()
    work["start_date"] = pd.to_datetime(work["start_date"], errors="coerce")
    work["outcome_date"] = pd.to_datetime(work["outcome_date"], errors="coerce")

    start_candidates = [s.min() for s in [work["start_date"], work["outcome_date"]] if s.notna().any()]
    end_candidates = [s.max() for s in [work["start_date"], work["outcome_date"]] if s.notna().any()]
    if not start_candidates or not end_candidates:
        return pd.DataFrame(columns=["date", "cases", "deaths", "recoveries"])

    idx = pd.date_range(start=pd.to_datetime(min(start_candidates)).normalize(), end=pd.to_datetime(max(end_candidates)).normalize(), freq="D")

    cases_daily = (
        work.loc[work["start_date"].notna(), "start_date"].dt.normalize().value_counts().sort_index().reindex(idx, fill_value=0).astype(float)
    )
    deaths_daily = (
        work.loc[work["event"].astype(str).str.lower().isin({"death", "dead", "died", "deceased"}) & work["outcome_date"].notna(), "outcome_date"]
        .dt.normalize()
        .value_counts()
        .sort_index()
        .reindex(idx, fill_value=0)
        .astype(float)
    )
    recoveries_daily = (
        work.loc[work["event"].astype(str).str.lower().isin({"recovery", "recovered", "alive", "discharged", "discharge"}) & work["outcome_date"].notna(), "outcome_date"]
        .dt.normalize()
        .value_counts()
        .sort_index()
        .reindex(idx, fill_value=0)
        .astype(float)
    )

    cases = cases_daily.cumsum()
    deaths = deaths_daily.cumsum()
    recoveries = recoveries_daily.cumsum()

    return pd.DataFrame({
        "date": idx,
        "cases": cases.to_numpy(dtype=float),
        "deaths": deaths.to_numpy(dtype=float),
        "recoveries": recoveries.to_numpy(dtype=float),
    })


# DRC 2018 
df = pd.read_csv("Data/DRC2018_humdata_MOH-Total.csv", skiprows=range(1,11))
counts = adapt_drc_total_to_counts(df)
print(df.columns.values)



# this would estimate delay distributions from a different dataset
# ll = pd.read_csv() need to find which dataset to use to caluclate delay distributions from 

# delays = estimate_delay_distributions_from_individual_data(
#     ll,
#     onset_col="start_date",
#     outcome_date_col="outcome_date",
#     outcome_col="event",
#     family="gamma",
# )

# death_delay = delays["death"]["cdf"]

#### This uses published delay parameters
# Published gamma parameters from Lancet paper (2018 ebola outbreak team)

shape = 2.4
scale = 1/0.3

def death_delay_cdf(ages):
    ages = np.asarray(ages, dtype=float)
    return stats.gamma.cdf(ages, a=shape, loc=0, scale=scale)


results = running_cfr(counts,
    dataset_kind="count_table",
    methods=["naive", "resolved", "delay_adjusted"],
    delay_distribution_death=death_delay_cdf,
)

print(results.head())


# Save results to CSV
results.to_csv("Results/cfr_results_2018.csv", index=False)

# Convert date column for plotting
results["date"] = pd.to_datetime(results["date"])
results = results.sort_values("date")

# only plot resolved when there are resolved cases that day 
# Build the flag from the raw table
flag = df[["report_date", "total_cured"]].copy()
flag["report_date"] = pd.to_datetime(flag["report_date"])
flag["has_recovery_data"] = flag["total_cured"].notna()

# Merge the flag onto the CFR results
results = results.merge(
    flag[["report_date", "has_recovery_data"]],
    left_on="date",
    right_on="report_date",
    how="left",
)

results["has_recovery_data"] = results["has_recovery_data"].fillna(False)

mask = ~results["has_recovery_data"]
results.loc[mask, ["resolved", "resolved_lower", "resolved_upper"]] = np.nan

# Can change start date here
start_date = results["date"].min() + pd.Timedelta(days=0)
plot_results = results[results["date"] >= start_date].copy()

epidemic_curve = _prepare_count_table_epidemic_curve(
    counts,
    date_col="report_date",
    cases_col="confirmed_cases",
    deaths_col="confirmed_deaths",
    recovered_col="total_cured",
)

_plot_cfr_and_curve_figure(
    plot_results=plot_results,
    epidemic_curve=epidemic_curve,
    methods=["naive", "resolved", "delay_adjusted"],
    title="Running CFR Estimates - DRC 2018",
    output_path="Results/cfr_results_2018.png",
)

print("Results written to: cfr_results_2018.csv")
print("Figure written to: cfr_results_2018.png")

from cfr_exact import estimate_delay_distributions_from_individual_data, standardize_line_list

# Uganda 2022
from cfr_exact import (
    load_uganda_2022,
    adapt_uganda_to_linelist,
    estimate_delay_distributions_from_individual_data,
    running_cfr,
)

df = load_uganda_2022("Data/Uganda2022globaldothealth.csv")
ll = adapt_uganda_to_linelist(df)

print(ll["event"].value_counts(dropna=False))
print(ll[["start_date", "outcome_date", "event"]].head())

delays = estimate_delay_distributions_from_individual_data(
    ll,
    onset_col="start_date",
    outcome_date_col="outcome_date",
    outcome_col="event",
    dayfirst=False,
)

death_delay = delays["death"]["cdf"]
recovery_delay = delays.get("recovery",{}).get("cdf")

results = running_cfr(
    ll,
    dataset_kind="line_list",
    methods=["naive", "resolved", "delay_adjusted", "competing_risks", "kaplan_meier_ghani", "parametric_mixture"],
    delay_distribution_death=death_delay,
    delay_distribution_recovery=recovery_delay,
    dayfirst=False,
)

print(results.head())

# plot
# Convert date column for plotting
results["date"] = pd.to_datetime(results["date"])
results = results.sort_values("date").reset_index(drop=True)

# Only plot resolved after the first recovery has been observed
recovery_dates = (
    ll.loc[
        (ll["event"] == "recovery") & ll["outcome_date"].notna(),
        "outcome_date",
    ]
    .sort_values()
    .to_numpy(dtype="datetime64[ns]")
)

if len(recovery_dates) > 0:
    results["has_recovery_data"] = (
        np.searchsorted(
            recovery_dates,
            results["date"].to_numpy(dtype="datetime64[ns]"),
            side="right",
        ) > 0
    )
    results.loc[~results["has_recovery_data"], "resolved"] = np.nan

# Can change start date here
start_date = results["date"].min() + pd.Timedelta(days=0)
plot_results = results[results["date"] >= start_date].copy()
if "parametric_mixture_success" in plot_results.columns:
    plot_results.loc[
        ~plot_results["parametric_mixture_success"],
        "parametric_mixture",
    ] = np.nan
# Save results
results.to_csv("Results/cfr_results_uganda.csv", index=False)

epidemic_curve = _prepare_linelist_epidemic_curve(ll)

_plot_cfr_and_curve_figure(
    plot_results=plot_results,
    epidemic_curve=epidemic_curve,
    methods=methods,
    title="Running CFR Estimates - Uganda 2022",
    output_path="Results/cfr_results_uganda.png",
)

print("Results written to: cfr_results_uganda.csv")
print("Figure written to: cfr_results_uganda.png")

# Kenema 2014

from cfr_exact import (
    load_kenema_2014,
    adapt_kenema_to_linelist,
    estimate_delay_distributions_from_individual_data,
    running_cfr,
)

df = load_kenema_2014("Data/kenema_2014_ebola-data.csv")
ll = adapt_kenema_to_linelist(df)

print(ll["event"].value_counts(dropna=False))
print(ll[["start_date", "outcome_date", "event"]].head())

delays = estimate_delay_distributions_from_individual_data(
    ll,
    onset_col="start_date",
    outcome_date_col="outcome_date",
    outcome_col="event",
    dayfirst=False,
)

death_delay = delays["death"]["cdf"]
recovery_delay = delays.get("recovery",{}).get("cdf")

results = running_cfr(
    ll,
    dataset_kind="line_list",
    methods=["naive", "resolved", "delay_adjusted", "competing_risks", "kaplan_meier_ghani", "parametric_mixture"],
    delay_distribution_death=death_delay,
    delay_distribution_recovery=recovery_delay,
    dayfirst=False,
)

print(results.head())

# plot
# Convert date column for plotting
results["date"] = pd.to_datetime(results["date"])
results = results.sort_values("date").reset_index(drop=True)

# Only plot resolved after the first recovery has been observed
recovery_dates = (
    ll.loc[
        (ll["event"] == "recovery") & ll["outcome_date"].notna(),
        "outcome_date",
    ]
    .sort_values()
    .to_numpy(dtype="datetime64[ns]")
)

if len(recovery_dates) > 0:
    results["has_recovery_data"] = (
        np.searchsorted(
            recovery_dates,
            results["date"].to_numpy(dtype="datetime64[ns]"),
            side="right",
        ) > 0
    )
    results.loc[~results["has_recovery_data"], "resolved"] = np.nan

# Can change start date here
start_date = results["date"].min() + pd.Timedelta(days=0)
plot_results = results[results["date"] >= start_date].copy()
if "parametric_mixture_success" in plot_results.columns:
    plot_results.loc[
        ~plot_results["parametric_mixture_success"],
        "parametric_mixture",
    ] = np.nan
# Save results
results.to_csv("Results/cfr_results_kenema.csv", index=False)

epidemic_curve = _prepare_linelist_epidemic_curve(ll)

_plot_cfr_and_curve_figure(
    plot_results=plot_results,
    epidemic_curve=epidemic_curve,
    methods=methods,
    title="Running CFR Estimates - Kenema 2014",
    output_path="Results/cfr_results_kenema.png",
)

print("Results written to: cfr_results_kenema.csv")
print("Figure written to: cfr_results_kenema.png")



# # Rosello
from cfr_exact import adapt_rosello_to_linelist_by_outbreak

df = pd.read_csv("Data/rosello2015_supplementary1.csv")

linelists = adapt_rosello_to_linelist_by_outbreak(
    df,
    case_categories_by_outbreak={
        "Mweka2007": ["Confirmed"],
    },
    start_date_col_by_outbreak={
        "Mweka2007": "Date_of_notification",
        "Kikwit": "Date_of_onset_symp",
    },
)
print(linelists.keys())          # outbreak names
# print(linelists["Isiro"].head())
# print(linelists["Kikwit"].head())
# print(linelists["Boende"].head())
# print(linelists["Mweka2007"].head())
# print(linelists["Mweka2008"].head())
# print(linelists["Yambuku"].head())

# the below shows that there are only four cases with symptom onset dates in Mweka 2007
print(linelists["Mweka2007"].head(20))
print(linelists["Mweka2008"].head(20))
ll = linelists["Mweka2008"]

print(ll["event"].value_counts())

print(
    ll.groupby("event")["outcome_date"].apply(lambda x: x.notna().sum())
)
linelists["Mweka2007"].to_csv("Mweka2007_linelist.csv", index=False)
linelists["Boende"].to_csv("Boende_linelist.csv", index=False)



skip = {"Tandala"}
skip = {"Tandala","Boende","Mweka2008","Yambuku","Isiro","Mweka2007","Kikwit"}

for name, ll in linelists.items():
    if name in skip:
        print(f"Skipping {name}")
        continue
    print(f"Analyzing {name}...")

    # Fit delay distributions from the line list
    delays = estimate_delay_distributions_from_individual_data(
        ll,
        onset_col="start_date",
        outcome_date_col="outcome_date",
        outcome_col="event",
        dayfirst=True,   # Rosello-style dates; keep True unless you know otherwise
    )

    death_delay = delays["death"]["cdf"]
    recovery_delay = delays.get("recovery", {}).get("cdf")
    shape_death = delays["death"]["shape"]
    scale_death = delays["death"]["scale"]
    mean_death = shape_death*scale_death
    shape_recovery = delays.get("recovery", {}).get("shape")
    scale_recovery = delays.get("recovery", {}).get("scale")
    if shape_recovery is None:
        mean_recovery = None
    else:
        mean_recovery = shape_recovery*scale_recovery
    print(f"delay distribution death is Gamma {shape_death}, {scale_death},{mean_death}")
    print(f"delay distribution recovery is Gamma {shape_recovery}, {scale_recovery},{mean_recovery}")
    
    if name in ["Mweka2007","Boende","Mweka2008"]:
        methods_here = ["naive","delay_adjusted","competing_risks"]
    else:
        methods_here = methods

    # Run CFR estimators    
    results = running_cfr(
        ll,
        dataset_kind="line_list",
        methods=methods_here,
        delay_distribution_death=death_delay,
        delay_distribution_recovery=recovery_delay,
        dayfirst=True,
    )

    # Save raw results
    results.to_csv(f"Results/cfr_results_{name}.csv", index=False)

    # Prepare plotting
    results["date"] = pd.to_datetime(results["date"])
    results = results.sort_values("date").reset_index(drop=True)

    # Hide resolved until the first recovery has been observed
    recovery_dates = (
        ll.loc[
            (ll["event"] == "recovery") & ll["outcome_date"].notna(),
            "outcome_date",
        ]
        .sort_values()
        .to_numpy(dtype="datetime64[ns]")
    )

    if len(recovery_dates) > 0:
        results["has_recovery_data"] = (
            np.searchsorted(
                recovery_dates,
                results["date"].to_numpy(dtype="datetime64[ns]"),
                side="right",
            ) > 0
        )
        results.loc[~results["has_recovery_data"], "resolved"] = np.nan

    # Start plotting from one week after first observation
    start_date = results["date"].min() + pd.Timedelta(days=0)
    plot_results = results[results["date"] >= start_date].copy()

    # Mask failed mixture fits
    if "parametric_mixture_success" in plot_results.columns:
        plot_results.loc[
            ~plot_results["parametric_mixture_success"],
            "parametric_mixture",
        ] = np.nan

    epidemic_curve = _prepare_linelist_epidemic_curve(ll)

    _plot_cfr_and_curve_figure(
        plot_results=plot_results,
        epidemic_curve=epidemic_curve,
        methods=methods,
        title=f"Running CFR Estimates - {name}",
        output_path=f"Results/cfr_results_{name}.png",
    )

    print(f"Results written to: Results/cfr_results_{name}.csv")
    print(f"Figure written to: Results/cfr_results_{name}.png")

# Age-specific CFR: OR/RR of death by age, at successive outbreak snapshots.
# Only Rosello's linelists carry real per-individual continuous age, so this
# runs over `linelists` from the loop above. Naive and resolved-cohort are
# both fit; a given outbreak/method is skipped (with a printed reason)
# whenever it doesn't have enough deaths and enough non-deaths (or
# recoveries, for resolved) with age recorded -- see min_events in
# age_time_relative_risk_curves.
from cfr_exact import age_time_relative_risk_curves
from plot_age_time_cfr import plot_age_time_relative_risk

MIN_AGE_N = 20

for name, ll_outbreak in linelists.items():
    if "age" not in ll_outbreak.columns:
        continue
    ll_age = ll_outbreak.dropna(subset=["age"])
    if len(ll_age) < MIN_AGE_N:
        print(f"{name}: skipping age-specific CFR (only {len(ll_age)} individuals with age, need >= {MIN_AGE_N})")
        continue

    for method in ["naive", "resolved"]:
        try:
            curves = age_time_relative_risk_curves(ll_age, method=method, spline_df=3)
        except ValueError as exc:
            print(f"{name} ({method}): skipping age-specific CFR -- {exc}")
            continue

        curves.to_csv(f"Results/age_time_relative_risk_{name}_{method}.csv", index=False)

        fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
        plot_age_time_relative_risk(curves, value="odds_ratio", ax=axes[0], title=f"{name} ({method}) - OR by age")
        plot_age_time_relative_risk(curves, value="risk_ratio", ax=axes[1], title=f"{name} ({method}) - RR by age")
        fig.tight_layout()
        fig.savefig(f"Results/age_time_relative_risk_{name}_{method}.png", dpi=150)
        plt.close(fig)

        print(f"{name} ({method}): saved Results/age_time_relative_risk_{name}_{method}.png")

# Age-specific competing risks: cause-specific hazards regression (death vs.
# recovery), the direct regression extension of cfr_competing_risks
# (Aalen-Johansen) -- same risk-set definition and product-integral
# recombination, but each cause-specific hazard is a smooth function of age.
# Same `linelists` dict as above. No pre-filter on event counts: each cause's
# Cox model itself tries a cubic spline, backs off to fewer basis functions,
# then to a plain linear age term, checking actual convergence at each step
# (see _fit_phreg_with_fallback in cfr_exact.py) -- an outbreak is only
# skipped if nothing converges even at linear.
#
# Uses the bootstrap-CI wrapper (cfr_competing_risks_by_age_ci) so the CIF
# panel gets a proper CI band instead of a point estimate only -- each
# bootstrap resample refits both cause-specific models from scratch, so this
# is slower (~200 refits per outbreak) but still only seconds per outbreak.
from cfr_exact import cfr_competing_risks_by_age_ci, _prepare_individual_time_data, cfr_competing_risks
from plot_competing_risks_by_age import plot_hazard_ratio, plot_cif

for name, ll_outbreak in linelists.items():
    if "age" not in ll_outbreak.columns:
        continue
    ll_age = ll_outbreak.dropna(subset=["age"])

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = cfr_competing_risks_by_age_ci(ll_age, spline_df=4, n_boot=200, seed=0)
    except ValueError as exc:
        print(f"{name}: skipping age-specific competing risks -- {exc}")
        continue

    res["death_hazard_ratio"].to_csv(f"Results/competing_risks_by_age_{name}_death_hr.csv", index=False)
    res["recovery_hazard_ratio"].to_csv(f"Results/competing_risks_by_age_{name}_recovery_hr.csv", index=False)
    res["cif"].to_csv(f"Results/competing_risks_by_age_{name}_cif.csv", index=False)

    time_event = _prepare_individual_time_data(ll_age)
    pooled = cfr_competing_risks(time_event.assign(time=time_event["time"]), time_col="time", event_col="event")

    fig, axes = plt.subplots(1, 3, figsize=(19, 5.5))
    plot_hazard_ratio(axes[0], res["death_hazard_ratio"], title=f"{name} - cause-specific HR: death")
    plot_hazard_ratio(axes[1], res["recovery_hazard_ratio"], title=f"{name} - cause-specific HR: recovery")
    plot_cif(axes[2], res["cif"], pooled_cfr=pooled["estimate"], title=f"{name} - recombined CIF of death")
    fig.tight_layout()
    fig.savefig(f"Results/competing_risks_by_age_{name}.png", dpi=150)
    plt.close(fig)

    print(
        f"{name}: saved Results/competing_risks_by_age_{name}.png "
        f"(n={res['n']}, deaths={res['n_deaths']}, recoveries={res['n_recoveries']}, "
        f"death model df={res['death_df_used']}, recovery model df={res['recovery_df_used']}, "
        f"bootstrap {res['n_boot_success']}/{res['n_boot']})"
    )

# Healthcare worker vs. not: a binary covariate, so a direct group split is
# the natural (lossless) analysis rather than an approximation the way
# binning continuous age would be. cfr_group_comparison runs all six pooled
# estimators separately per group; group_death_ratio/group_hazard_ratio add
# a formal comparison (OR, RR, HR) with its own CI, via a single-covariate
# GLM/Cox fit -- no spline machinery needed, so no convergence fallback.
from cfr_exact import (
    load_uganda_2022,
    adapt_uganda_to_linelist,
    cfr_group_comparison,
    group_death_ratio,
    group_hazard_ratio,
)

hcw_sources = {
    "Kikwit": linelists["Kikwit"],
    "Isiro": linelists["Isiro"],
    "Boende": linelists["Boende"],
    "Uganda": adapt_uganda_to_linelist(load_uganda_2022("Data/Uganda2022globaldothealth.csv")),
}

for name, ll_hcw in hcw_sources.items():
    if "is_hcw" not in ll_hcw.columns:
        print(f"{name}: skipping HCW comparison -- no is_hcw column")
        continue
    ll_hcw = ll_hcw.dropna(subset=["is_hcw"])
    print(f"{name}: is_hcw counts {ll_hcw['is_hcw'].value_counts().to_dict()}")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        comparison = cfr_group_comparison(ll_hcw)
    comparison.to_csv(f"Results/hcw_comparison_{name}.csv", index=False)

    ratio_rows = []
    for method in ["naive", "resolved"]:
        try:
            r = group_death_ratio(ll_hcw, method=method)
            r["method"] = method
            ratio_rows.append(r)
        except ValueError as exc:
            print(f"{name} ({method} OR/RR): skipped -- {exc}")

    try:
        hr = group_hazard_ratio(ll_hcw, cause="death")
        hr["method"] = "competing_risks_hr"
        ratio_rows.append(hr)
    except ValueError as exc:
        print(f"{name} (death HR): skipped -- {exc}")

    if ratio_rows:
        pd.DataFrame(ratio_rows).to_csv(f"Results/hcw_ratios_{name}.csv", index=False)

    print(f"{name}: saved Results/hcw_comparison_{name}.csv and Results/hcw_ratios_{name}.csv")

# Current DRC outbreak
from cfr_exact import adapt_drc_consolidated_to_counts
# DRC 2026
df = pd.read_csv("Data/drc_ebola_cases_consolidated.csv")
counts = adapt_drc_consolidated_to_counts(df)
print(df.columns.values)
print(df.head())

#### This uses published delay parameters
# Published gamma parameters from Lancet paper (2018 ebola outbreak team)

shape = 2.4
scale = 1/0.3

def death_delay_cdf(ages):
    ages = np.asarray(ages, dtype=float)
    return stats.gamma.cdf(ages, a=shape, loc=0, scale=scale)


results = running_cfr(counts,
    dataset_kind="count_table",
    methods=["naive", "resolved", "delay_adjusted"],
    delay_distribution_death=death_delay_cdf,
)

print(results.head())


# Save results to CSV
results.to_csv("Results/cfr_results_2026.csv", index=False)

# Convert date column for plotting
results["date"] = pd.to_datetime(results["date"])
results = results.sort_values("date")

# only plot resolved when there are resolved cases that day 
counts["has_recovery_data"] = counts["total_cured"].notna()


# Hide resolved estimates where total_cured is missing
flag = counts[["report_date", "has_recovery_data"]].copy()
flag["report_date"] = pd.to_datetime(flag["report_date"])

results = results.merge(
    flag,
    left_on="date",
    right_on="report_date",
    how="left",
)
results["has_recovery_data"] = (
    results["has_recovery_data"]
    .eq(True)
)

mask = ~results["has_recovery_data"]

results.loc[
    mask,
    ["resolved", "resolved_lower", "resolved_upper"]
] = np.nan

# Can change start date here
start_date = results["date"].min() + pd.Timedelta(days=0)
plot_results = results[results["date"] >= start_date].copy()

epidemic_curve = _prepare_count_table_epidemic_curve(
    counts,
    date_col="report_date",
    cases_col="total_cases",
    deaths_col="total_deaths",
    recovered_col="total_cured",
)

_plot_cfr_and_curve_figure(
    plot_results=plot_results,
    epidemic_curve=epidemic_curve,
    methods=["naive", "resolved", "delay_adjusted"],
    title="Running CFR Estimates - DRC 2026",
    output_path="Results/cfr_results_2026.png",
)

print("Results written to: cfr_results_2026.csv")
print("Figure written to: cfr_results_2026.png")

# Guinea 2021
df = pd.read_csv("Data/guinea_2021_combined.csv")
counts = adapt_guinea_to_counts(df)
print(counts.columns.values)



shape = 2.4
scale = 1/0.3

def death_delay_cdf(ages):
    ages = np.asarray(ages, dtype=float)
    return stats.gamma.cdf(ages, a=shape, loc=0, scale=scale)


results = running_cfr(
    counts,
    dataset_kind="count_table",
    date_col="Date_case",
    cases_col="Cases",
    deaths_col="Deaths",
    recovered_col="Recoveries",
    methods=["naive", "resolved", "delay_adjusted"],
    delay_distribution_death=death_delay_cdf,
)

print(results.head())


# Save results to CSV
results.to_csv("Results/cfr_results_guinea_2021.csv", index=False)

# Convert date column for plotting
results["date"] = pd.to_datetime(results["date"])
results = results.sort_values("date")

# only plot resolved when there are resolved cases that day 
counts["has_recovery_data"] =counts["Recoveries"].notna()


# Hide resolved estimates where total_cured is missing
flag = counts[["Date_case", "has_recovery_data"]].copy()
flag["Date_case"] = pd.to_datetime(flag["Date_case"])

results = results.merge(
    flag,
    left_on="date",
    right_on="Date_case",
    how="left",
)
results["has_recovery_data"] = (
    results["has_recovery_data"]
    .eq(True)
)

mask = ~results["has_recovery_data"]
results.loc[mask, ["resolved", "resolved_lower", "resolved_upper"]] = np.nan

# Only keep data from one week after the first observation
start_date = results["date"].min() + pd.Timedelta(days=0)
plot_results = results[results["date"] >= start_date]

# Create the figure
epidemic_curve = _prepare_count_table_epidemic_curve(
    counts,
    date_col="Date_case",
    cases_col="Cases",
    deaths_col="Deaths",
    recovered_col="Recoveries",
)

_plot_cfr_and_curve_figure(
    plot_results=plot_results,
    epidemic_curve=epidemic_curve,
    methods=["naive", "resolved", "delay_adjusted"],
    title="Running CFR Estimates - Guinea 2021",
    output_path="Results/cfr_results_guinea_2021.png",
)

print("Results written to: cfr_results_guinea_2021.csv")
print("Figure written to: cfr_results_guinea_2021.png")