from cfr_exact import running_cfr, adapt_drc_total_to_counts
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
df = pd.read_csv("Data/DRC2018_humdata_MOH-Total.csv", skiprows=range(1,11))
# counts = adapt_drc_total_to_counts(df)
print(df.columns.values)
# print(df["report_date"].dtype)

# print(df["report_date"].head(20))

# print(df["report_date"].map(type).value_counts())

# print(df["report_date"].unique()[:20])
# print(df.head())
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
results.to_csv("cfr_results.csv", index=False)

# Convert date column for plotting
results["date"] = pd.to_datetime(results["date"])
results = results.sort_values("date")

# Only keep data from one week after the first observation
start_date = results["date"].min() + pd.Timedelta(days=7)
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

plt.savefig("cfr_results.png", dpi=300, bbox_inches="tight")


print("Results written to: cfr_results.csv")
print("Figure written to: cfr_results.png")

from cfr_exact import estimate_delay_distributions_from_individual_data, standardize_line_list

# ll = standardize_line_list(df)
# delays = estimate_delay_distributions_from_individual_data(
#     ll,
#     onset_col="start_date",
#     outcome_date_col="outcome_date",
#     outcome_col="event",
# )

# death_delay = delays["death"]
# recovery_delay = delays.get("recovery")