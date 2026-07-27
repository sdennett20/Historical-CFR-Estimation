from cfr_exact import running_cfr, adapt_drc_total_to_counts
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

# DRC 2018 
df = pd.read_csv("Data/DRC2018_humdata_MOH-Total.csv", skiprows=range(1,11))
# counts = adapt_drc_total_to_counts(df)
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


results = running_cfr(df,
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
df["has_recovery_data"] = df["total_cured"].notna()


# Hide resolved estimates where total_cured is missing
flag = df[["report_date", "has_recovery_data"]].copy()
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

results.loc[~results["has_recovery_data"], "resolved"] = np.nan

# Only keep data from one week after the first observation
start_date = results["date"].min() + pd.Timedelta(days=0)
plot_results = results[results["date"] >= start_date]

# Create the figure
plt.figure(figsize=(10, 6))

for method in ["naive", "resolved", "delay_adjusted"]:
    if method in plot_results.columns:
        plt.plot(plot_results["date"], plot_results[method],
                 linewidth=2, label=method)

plt.xlabel("Date")
plt.ylabel("Case Fatality Ratio")
plt.title("Running CFR Estimates")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.savefig("Results/cfr_results_2018.png", dpi=300, bbox_inches="tight")


print("Results written to: cfr_results_2018.csv")
print("Figure written to: cfr_results_2018.png")

# from cfr_exact import estimate_delay_distributions_from_individual_data, standardize_line_list

# try on Uganda 2022
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
    methods=["naive", "resolved", "delay_adjusted", "competing_risks", "kaplan_meier", "parametric_mixture"],
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

# Only keep data from one week after the first observation
start_date = results["date"].min() + pd.Timedelta(days=7)
plot_results = results[results["date"] >= start_date]
if "parametric_mixture_success" in plot_results.columns:
    plot_results.loc[
        ~plot_results["parametric_mixture_success"],
        "parametric_mixture",
    ] = np.nan
# Save results
results.to_csv("Results/cfr_results_uganda.csv", index=False)

# Plot all methods that are present
plt.figure(figsize=(10, 6))

methods = [
    "naive",
    "resolved",
    "delay_adjusted",
    "competing_risks",
    "kaplan_meier",
    "parametric_mixture",
]

for method in methods:
    if method in plot_results.columns:
        plt.plot(
            plot_results["date"],
            plot_results[method],
            linewidth=2,
            label=method,
        )

plt.xlabel("Date")
plt.ylabel("Case Fatality Ratio")
plt.title("Running CFR Estimates - Uganda 2022")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.savefig("Results/cfr_results_uganda.png", dpi=300, bbox_inches="tight")
plt.close()

print("Results written to: cfr_results_uganda.csv")
print("Figure written to: cfr_results_uganda.png")

# Rosello
from cfr_exact import adapt_rosello_to_linelist_by_outbreak
df = pd.read_csv("Data/rosello2015_supplementary1.csv")

linelists = adapt_rosello_to_linelist_by_outbreak(df)

print(linelists.keys())          # outbreak names
# print(linelists["Isiro"].head())
# print(linelists["Kikwit"].head())
# print(linelists["Boende"].head())
# print(linelists["Mweka2007"].head())
# print(linelists["Mweka2008"].head())
# print(linelists["Yambuku"].head())

# the below shows that there are only four cases with symptom onset dates in Mweka 2007
print(linelists["Mweka2007"])
linelists["Mweka2007"].to_csv("Mweka2007_linelist.csv", index=False)
linelists["Boende"].to_csv("Boende_linelist.csv", index=False)

methods = [
    "naive",
    "resolved",
    "delay_adjusted",
    "competing_risks",
    "kaplan_meier",
    "parametric_mixture",
]

skip = {"Mweka2007", "Tandala"}

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

    # Run CFR estimators
    results = running_cfr(
        ll,
        dataset_kind="line_list",
        methods=methods,
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
    start_date = results["date"].min() + pd.Timedelta(days=7)
    plot_results = results[results["date"] >= start_date].copy()

    # Mask failed mixture fits
    if "parametric_mixture_success" in plot_results.columns:
        plot_results.loc[
            ~plot_results["parametric_mixture_success"],
            "parametric_mixture",
        ] = np.nan

    # Plot
    plt.figure(figsize=(10, 6))

    for method in methods:
        if method in plot_results.columns:
            plt.plot(
                plot_results["date"],
                plot_results[method],
                linewidth=2,
                label=method,
            )

    plt.xlabel("Date")
    plt.ylabel("Case Fatality Ratio")
    plt.title(f"Running CFR Estimates - {name}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.savefig(f"Results/cfr_results_{name}.png", dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Results written to: cfr_results_{name}.csv")
    print(f"Figure written to: cfr_results_{name}.png")


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

results.loc[~results["has_recovery_data"], "resolved"] = np.nan

# Only keep data from one week after the first observation
start_date = results["date"].min() + pd.Timedelta(days=0)
plot_results = results[results["date"] >= start_date]

# Create the figure
plt.figure(figsize=(10, 6))

for method in ["naive", "resolved", "delay_adjusted"]:
    if method in plot_results.columns:
        plt.plot(plot_results["date"], plot_results[method],
                 linewidth=2, label=method)

plt.xlabel("Date")
plt.ylabel("Case Fatality Ratio")
plt.title("Running CFR Estimates")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.savefig("Results/cfr_results_2026.png", dpi=300, bbox_inches="tight")


print("Results written to: cfr_results_2026.csv")
print("Figure written to: cfr_results_2026.png")